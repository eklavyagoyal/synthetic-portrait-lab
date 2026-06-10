"""ARCHIVE — the run browser.

Scans ``<output_base>/*/manifest.json`` in a thread worker (the directory may
be large), renders a sortable/filterable DataTable, and opens runs as contact
sheets. Malformed manifests appear as dimmed ``unreadable`` rows — never a crash.
"""

from __future__ import annotations

from typing import Optional

from rich.markup import escape as esc
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, Static

from .. import glyphs, runscan
from ..messages import ArchiveScanned
from ..widgets import Hero

SORTS = ("when", "cost", "frames", "fail")

_STATUS_GLYPH = {
    "completed": (glyphs.DOT, "green"),
    "cancelled": (glyphs.DOT_HALF, "yellow"),
    "failed": (glyphs.CROSS, "red"),
    "running": (glyphs.DOT_HALF, "cyan"),
}


class ArchiveScreen(Screen):
    AUTO_FOCUS = "#archive-table"

    BINDINGS = [
        Binding("enter", "open_run", "open", show=False),
        Binding("o", "reveal", "reveal"),
        Binding("r", "rescan", "rescan"),
        Binding("slash", "focus_filter", "filter", key_display="/"),
        Binding("s", "cycle_sort", "sort"),
        Binding("escape,q", "to_studio", "studio"),
    ]

    DEFAULT_CSS = """
    ArchiveScreen {
        background: $background;

        #archive-headline { height: 1; padding: 0 2; color: $text-muted; }
        #archive-filter { margin: 0 2; width: 40; display: none; }
        &.-filtering #archive-filter { display: block; }
        #archive-table-card {
            height: 1fr;
            margin: 0 1 1 1;
            border: round $border-soft;
            border-title-color: $secondary;
            background: $surface;
        }
        #archive-table { height: 1fr; }
        #archive-empty {
            height: 1fr;
            content-align: center middle;
            color: $text-muted;
            display: none;
        }
        &.-empty #archive-empty { display: block; }
        &.-empty #archive-table { display: none; }
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._records: list[runscan.RunRecord] = []
        self._row_dirs: list[Optional[runscan.RunRecord]] = []
        self._sort = "when"
        self._needle = ""

    def compose(self) -> ComposeResult:
        yield Hero(active="archive")
        yield Static("", id="archive-headline")
        yield Input(placeholder="filter by run id / model / provider…", id="archive-filter")
        with Vertical(id="archive-table-card") as card:
            card.border_title = "archive"
            yield DataTable(id="archive-table", cursor_type="row", zebra_stripes=True)
            yield Static(
                f"{glyphs.DIAMOND_HOLLOW} no runs developed yet\n\n"
                f"[$text-muted]press [b]f2[/b] for the studio — the free mock model "
                "works offline[/]",
                id="archive-empty",
            )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#archive-table", DataTable)
        table.add_columns("WHEN", "RUN ID", "MODEL", "FRAMES", f"{glyphs.CHECK} OK",
                          f"{glyphs.CROSS} FAIL", "COST", "STATUS")
        self.action_rescan()

    def on_screen_resume(self) -> None:
        self.action_rescan()

    # ------------------------------------------------------------------ #
    # Scan (thread)
    # ------------------------------------------------------------------ #
    def action_rescan(self) -> None:
        self.query_one("#archive-headline", Static).update(
            f"{glyphs.BUSY} scanning archive…"
        )
        self._scan(str(self.app.config.settings.output_base_dir))

    @work(thread=True, exclusive=True, group="archive-scan")
    def _scan(self, base_dir: str) -> None:
        self.post_message(ArchiveScanned(runscan.scan_runs(base_dir)))

    @on(ArchiveScanned)
    def _scanned(self, message: ArchiveScanned) -> None:
        message.stop()
        self._records = message.records
        self._render_table()

    # ------------------------------------------------------------------ #
    # Table rendering
    # ------------------------------------------------------------------ #
    def _sorted_filtered(self) -> list[runscan.RunRecord]:
        records = self._records
        needle = self._needle.strip().lower()
        if needle:
            records = [
                r
                for r in records
                if needle in r.run_id.lower()
                or needle in r.model.lower()
                or needle in r.provider.lower()
            ]
        keys = {
            "when": lambda r: (r.when, r.run_id),
            "cost": lambda r: (r.cost if r.cost is not None else -1.0),
            "frames": lambda r: r.planned,
            "fail": lambda r: r.fail,
        }
        return sorted(records, key=keys[self._sort], reverse=True)

    def _render_table(self) -> None:
        table = self.query_one("#archive-table", DataTable)
        table.clear()
        self._row_dirs = []
        records = self._sorted_filtered()

        readable = [r for r in self._records if not r.unreadable]
        total_images = sum(r.ok for r in readable)
        known_costs = [r.cost for r in readable if r.cost is not None]
        total_cost = sum(known_costs)
        unpriced = len(readable) - len(known_costs)
        any_estimated = any(r.cost_is_estimated for r in readable)
        approx = " (~)" if any_estimated else ""
        unpriced_note = f" · {unpriced} unpriced" if unpriced else ""
        self.query_one("#archive-headline", Static).update(
            f"{len(self._records)} runs · {total_images} images · "
            f"${total_cost:.2f} lifetime{approx}{unpriced_note}"
            + (f"   ·   filter: [b]{esc(self._needle)}[/b]" if self._needle else "")
            + f"   ·   sort: {self._sort} ▼"
        )
        self.set_class(not self._records, "-empty")

        for record in records:
            self._row_dirs.append(record)
            if record.unreadable:
                table.add_row(
                    Text("—", style="dim"),
                    Text(record.run_id, style="dim"),
                    Text("(unreadable manifest)", style="dim"),
                    "", "", "", Text("—", style="dim"),
                    Text(f"{glyphs.WARN} unreadable", style="yellow"),
                )
                continue
            when = record.when.replace("T", " ").removesuffix("Z")[:16] or "—"
            ok_text = Text(str(record.ok), style="green" if record.ok else "dim")
            fail_text = Text(str(record.fail), style="red" if record.fail else "dim")
            if record.cost is None:
                cost_text = Text("$ —", style="yellow")
            elif record.cost == 0:
                cost_text = Text("FREE", style="dim")
            else:
                approx_mark = "~" if record.cost_is_estimated else ""
                cost_text = Text(f"${record.cost:.4f} {approx_mark}", justify="right")
            glyph, colour = _STATUS_GLYPH.get(record.status, (glyphs.DOT_HOLLOW, "dim"))
            table.add_row(
                when,
                record.run_id,
                record.model,
                str(record.planned),
                ok_text,
                fail_text,
                cost_text,
                Text(f"{glyph} {record.status}", style=colour),
            )

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #
    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_row(event.cursor_row)

    def _open_row(self, row: int) -> None:
        if 0 <= row < len(self._row_dirs):
            record = self._row_dirs[row]
            if record is None or record.unreadable:
                self.notify(
                    record.error or "Manifest unreadable.", severity="warning", title=record.run_id
                )
                return
            self.app.open_contact_sheet(record.run_dir)

    def action_open_run(self) -> None:
        table = self.query_one("#archive-table", DataTable)
        self._open_row(table.cursor_row)

    def action_reveal(self) -> None:
        table = self.query_one("#archive-table", DataTable)
        if 0 <= table.cursor_row < len(self._row_dirs):
            record = self._row_dirs[table.cursor_row]
            reveal = getattr(self.app, "reveal_path", None)
            if record is not None and reveal is not None:
                reveal(record.run_dir)

    def action_focus_filter(self) -> None:
        self.add_class("-filtering")
        self.query_one("#archive-filter", Input).focus()

    @on(Input.Changed, "#archive-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._needle = event.value
        self._render_table()

    @on(Input.Submitted, "#archive-filter")
    def _filter_submitted(self) -> None:
        self.query_one("#archive-table", DataTable).focus()

    def on_key(self, event) -> None:
        if event.key == "escape" and self.query_one("#archive-filter", Input).has_focus:
            event.stop()
            if not self._needle:
                self.remove_class("-filtering")
            self.query_one("#archive-table", DataTable).focus()

    def action_cycle_sort(self) -> None:
        idx = SORTS.index(self._sort)
        self._sort = SORTS[(idx + 1) % len(SORTS)]
        self._render_table()

    def action_to_studio(self) -> None:
        self.app.switch_mode("studio")
