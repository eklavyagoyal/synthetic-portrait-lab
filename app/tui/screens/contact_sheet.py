"""CONTACT SHEET — the per-run gallery.

Reads ``metadata.jsonl`` (ground truth, failures included) plus the manifest
header, lays the frames out on a grid, and decodes thumbnails lazily in a
thread worker so the sheet visibly "develops". Failed frames render as
hard-cornered error tiles — failure is the one place corners go square.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.markup import escape as esc
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

from .. import glyphs, imaging, labels, runscan
from ..messages import SheetLoaded, ThumbReady
from ..widgets import Hero
from .modals import LightboxModal, PromptPeekModal

THUMB_SIZES = {"S": (12, 6), "M": (22, 11), "L": (30, 15)}
MAX_TILES = 500
FILTERS = ("all", "success", "failed")


class ThumbTile(Static):
    """One frame on the sheet."""

    DEFAULT_CSS = """
    ThumbTile {
        width: auto;
        height: auto;
        padding: 0 1;
        border: round $border-soft;
        &.-selected { border: heavy $primary; }
        &.-failed { border: solid $error 50%; }
        &.-dimmed { opacity: 0.35; }
    }
    """

    def __init__(self, index: int, item: dict) -> None:
        super().__init__()
        self.index = index
        self.item = item
        self.border_subtitle = str(item.get("id") or "")[-6:]

    def on_click(self) -> None:
        screen = self.screen
        if isinstance(screen, ContactSheetScreen):
            screen.select(self.index, open_lightbox=True)


class ContactSheetScreen(Screen):
    BINDINGS = [
        Binding("escape,q", "back", "back"),
        Binding("left", "move(-1)", "prev", show=False),
        Binding("right", "move(1)", "next", show=False),
        Binding("up", "move_row(-1)", "up", show=False),
        Binding("down", "move_row(1)", "down", show=False),
        Binding("enter", "lightbox", "lightbox"),
        Binding("v", "prompt", "prompt"),
        Binding("f", "filter", "filter"),
        Binding("m", "metadata", "metadata"),
        Binding("plus,equals_sign", "thumb_size(1)", "bigger", key_display="+"),
        Binding("minus", "thumb_size(-1)", "smaller", key_display="-"),
        Binding("o", "reveal", "reveal"),
        Binding("c", "copy_path", "copy path"),
    ]

    DEFAULT_CSS = """
    ContactSheetScreen {
        background: $background;

        #sheet-header { height: auto; padding: 0 2; color: $text-muted; }
        #sheet-body { height: 1fr; padding: 0 1; }
        #frames-card {
            width: 1fr;
            height: 100%;
            border: round $border-soft;
            border-title-color: $secondary;
            background: $surface;
            padding: 0 1;
        }
        #frames-scroll { height: 1fr; }
        #frames-grid { grid-gutter: 0 1; height: auto; }
        #meta-card {
            width: 44;
            height: 100%;
            border: round $border-soft;
            border-title-color: $secondary;
            background: $surface;
            padding: 1 2;
            margin: 0 0 0 1;
        }
        #meta-card Static { height: auto; }
        &.-no-meta #meta-card { display: none; }
        &.-compact #meta-card { width: 30; }
        #sheet-empty { height: 1fr; content-align: center middle; color: $text-muted; }
    }
    """

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self._run_dir = Path(run_dir)
        self._items: list[dict] = []
        self._header: dict = {}
        self._selected = 0
        self._filter = "all"
        self._tiles: list[ThumbTile] = []

    @property
    def _thumb_size(self) -> tuple[int, int]:
        return THUMB_SIZES.get(getattr(self.app, "prefs", None) and self.app.prefs.thumb_size or "M", THUMB_SIZES["M"])

    def compose(self) -> ComposeResult:
        yield Hero(active="contact")
        yield Static("", id="sheet-header")
        with Horizontal(id="sheet-body"):
            with Vertical(id="frames-card") as card:
                card.border_title = "frames"
                # not focusable: a focused ScrollableContainer would consume the
                # up/down selection keys; select() scrolls tiles into view itself
                scroll = VerticalScroll(id="frames-scroll")
                scroll.can_focus = False
                with scroll:
                    yield Grid(id="frames-grid")
                yield Static("", id="sheet-empty")
            with Vertical(id="meta-card") as card:
                card.border_title = "frame"
                yield Static("", id="meta-body")
        yield Footer()

    def on_mount(self) -> None:
        self._apply_breakpoint(self.app.size.width)
        self.query_one("#sheet-empty").display = False
        self.query_one("#sheet-header", Static).update(
            f"{glyphs.BUSY} reading {self._run_dir.name}…"
        )
        self._load_sheet(self._run_dir)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_breakpoint(event.size.width)
        self._relayout_grid()

    def _apply_breakpoint(self, width: int) -> None:
        self.set_class(width < 110, "-compact")

    # ------------------------------------------------------------------ #
    # Data loading (thread)
    # ------------------------------------------------------------------ #
    @work(thread=True, exclusive=True, group="sheet-load")
    def _load_sheet(self, run_dir: Path) -> None:
        try:
            header, items = runscan.load_sheet(run_dir)
            self.post_message(SheetLoaded(header, items))
        except Exception as exc:  # noqa: BLE001
            self.post_message(SheetLoaded({}, [], error=f"{type(exc).__name__}: {exc}"))

    @on(SheetLoaded)
    async def _sheet_loaded(self, message: SheetLoaded) -> None:
        message.stop()
        self._header = message.header
        self._items = message.items[:MAX_TILES]
        truncated = len(message.items) - len(self._items)
        if message.error:
            self.query_one("#sheet-header", Static).update(
                f"[$error]{glyphs.WARN} {esc(message.error)}[/]"
            )
            return
        summary = self._header.get("summary") or {}
        billed = summary.get("provider_reported_cost_usd")
        burn = summary.get("estimated_cost_from_attempts_usd")
        legacy = summary.get("actual_total_usd")
        if billed is not None:
            cost_str = f"${billed:.4f}"
        elif burn is not None:
            cost_str = labels.badge("FREE", "$tele-ok") if burn == 0 else f"~${burn:.4f}"
        elif isinstance(legacy, (int, float)):
            cost_str = (
                labels.badge("FREE", "$tele-ok")
                if legacy == 0
                else f"${legacy:.4f}" + ("~" if summary.get("actual_cost_includes_estimates") else "")
            )
        else:
            cost_str = "$ —"
        seed = (self._header.get("request") or {}).get("seed")
        trunc_note = f" · showing first {MAX_TILES}" if truncated > 0 else ""
        self.query_one("#sheet-header", Static).update(
            f"archive {glyphs.ARROW} [b]{esc(self._run_dir.name)}[/b]"
            f" · {len(message.items)} frames"
            f" · {esc(str(self._header.get('model', '?')))} · {cost_str}"
            f" · seed {seed if seed is not None else 'random'}{trunc_note}"
            f"      [$text-muted]enter lightbox · v prompt · f filter · m metadata[/]"
        )
        await self._build_tiles()
        if not self._items:
            empty = self.query_one("#sheet-empty", Static)
            empty.display = True
            empty.update(
                f"{glyphs.DIAMOND_HOLLOW} no frames in this run\n"
                f"[$text-muted]see metadata.jsonl for details[/]"
            )
        else:
            self.select(0)
            self._decode_visible()

    async def _build_tiles(self) -> None:
        grid = self.query_one("#frames-grid", Grid)
        await grid.remove_children()
        self._tiles = []
        cols, rows = self._thumb_size
        for index, item in enumerate(self._items):
            tile = ThumbTile(index, item)
            if item.get("status") != "success" or not item.get("filename"):
                tile.add_class("-failed")
                tile.update(
                    imaging.skeleton(cols, rows, label=f"{glyphs.CROSS} FAILED")
                )
            else:
                tile.update(imaging.skeleton(cols, rows))
            self._tiles.append(tile)
        if self._tiles:
            await grid.mount_all(self._tiles)
        self._relayout_grid()

    def _relayout_grid(self) -> None:
        if not self._tiles:
            return
        grid = self.query_one("#frames-grid", Grid)
        cols, _rows = self._thumb_size
        tile_width = cols + 4  # padding + border
        available = max(tile_width, self.query_one("#frames-scroll").size.width or 80)
        per_row = max(1, available // (tile_width + 1))
        grid.styles.grid_size_columns = per_row
        grid.styles.grid_size_rows = 0

    @work(thread=True, exclusive=True, group="sheet-decode")
    def _decode_all(self, jobs: list[tuple[int, Path]], cols: int, rows: int) -> None:
        from textual.worker import get_current_worker

        worker = get_current_worker()
        for index, path in jobs:
            if worker.is_cancelled:
                return
            try:
                text = imaging.render_halfblock(path, cols, rows)
                self.post_message(ThumbReady(f"tile:{index}:{cols}", path, text))
            except imaging.ImageUnreadable:
                self.post_message(
                    ThumbReady(f"tile:{index}:{cols}", path, None, error="unreadable")
                )

    def _decode_visible(self) -> None:
        cols, rows = self._thumb_size
        jobs = []
        for index, item in enumerate(self._items):
            if item.get("status") == "success" and item.get("filename"):
                jobs.append((index, self._run_dir / "images" / str(item["filename"])))
        if jobs:
            self._decode_all(jobs, cols, rows)

    @on(ThumbReady)
    def _thumb_ready(self, message: ThumbReady) -> None:
        message.stop()
        cols, rows = self._thumb_size
        prefix, _, rest = message.key.partition(":")
        if prefix != "tile":
            return
        index_str, _, key_cols = rest.partition(":")
        if key_cols != str(cols):
            return  # stale decode from a previous thumb size
        index = int(index_str)
        if index >= len(self._tiles):
            return
        tile = self._tiles[index]
        if message.text is None:
            tile.add_class("-failed")
            tile.update(imaging.skeleton(cols, rows, label=f"{glyphs.CROSS} unreadable"))
        else:
            tile.update(message.text)

    # ------------------------------------------------------------------ #
    # Selection + filter
    # ------------------------------------------------------------------ #
    def select(self, index: int, *, open_lightbox: bool = False) -> None:
        if not self._items:
            return
        index = max(0, min(index, len(self._items) - 1))
        if self._tiles and self._selected < len(self._tiles):
            self._tiles[self._selected].remove_class("-selected")
        self._selected = index
        tile = self._tiles[index]
        tile.add_class("-selected")
        tile.scroll_visible()
        self._render_meta()
        if open_lightbox:
            self.action_lightbox()

    def _render_meta(self) -> None:
        item = self._items[self._selected] if self._items else None
        body = self.query_one("#meta-body", Static)
        if item is None:
            body.update("")
            return
        ok = item.get("status") == "success"
        status = (
            f"[$tele-ok]{glyphs.CHECK} success[/]" if ok else f"[$tele-fail]{glyphs.CROSS} failed[/]"
        )
        seed = item.get("seed")
        actual = item.get("actual_cost_usd")
        est = item.get("estimated_cost_usd")
        if isinstance(actual, (int, float)):
            cost_str = f"${actual:.4f}"
        elif isinstance(est, (int, float)):
            cost_str = f"~${est:.4f} est"
        else:
            cost_str = "—"
        prompt = str(item.get("prompt") or "")
        prompt_head = esc("\n".join(prompt.splitlines()[:6]))
        error = item.get("error")
        self.query_one("#meta-card").border_title = str(item.get("id") or "frame")
        body.update(
            labels.triple_chips(
                str(item.get("age_bucket", "")),
                str(item.get("gender_bucket", "")),
                str(item.get("ethnicity_bucket", "")),
            )
            + f"\n\nstatus  {status}      attempts {item.get('attempts', 1)} · retries {item.get('retries', 0)}"
            + f"\nseed {seed if seed is not None else 'random'} · {item.get('size', '?')} · {cost_str}"
            + f"\n[$text-muted]file  images/{esc(str(item.get('filename') or '—'))}[/]"
            + (f"\n[$tele-fail]{esc(str(error))}[/]" if error else "")
            + (
                f"\n[$text-muted]{'┄' * 30}[/]\n[$text-muted]{prompt_head}[/]"
                f"\n[$text-muted]v full prompt[/]"
                if prompt
                else ""
            )
        )

    def _apply_filter(self) -> None:
        for tile in self._tiles:
            status = tile.item.get("status")
            visible = (
                self._filter == "all"
                or (self._filter == "success" and status == "success")
                or (self._filter == "failed" and status != "success")
            )
            tile.set_class(not visible, "-dimmed")
        self.query_one("#frames-card").border_subtitle = (
            f"filter · {self._filter}" if self._filter != "all" else ""
        )

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _per_row(self) -> int:
        grid = self.query_one("#frames-grid", Grid)
        return max(1, int(grid.styles.grid_size_columns or 1))

    def action_move(self, delta: int) -> None:
        self.select(self._selected + delta)

    def action_move_row(self, delta: int) -> None:
        self.select(self._selected + delta * self._per_row())

    def action_lightbox(self) -> None:
        if self._items:
            self.app.push_screen(LightboxModal(self._items, self._selected, self._run_dir))

    def action_prompt(self) -> None:
        item = self._items[self._selected] if self._items else None
        if not item:
            return
        prompt = str(item.get("prompt") or "")
        if not prompt:
            self.notify("This run did not save prompts.", severity="warning", timeout=3)
            return
        self.app.push_screen(
            LightboxModal(self._items, self._selected, self._run_dir, show_prompt=True)
        )

    def action_filter(self) -> None:
        idx = FILTERS.index(self._filter)
        self._filter = FILTERS[(idx + 1) % len(FILTERS)]
        self._apply_filter()

    def action_metadata(self) -> None:
        self.toggle_class("-no-meta")

    def action_thumb_size(self, delta: int) -> None:
        order = ("S", "M", "L")
        prefs = self.app.prefs
        idx = order.index(prefs.thumb_size if prefs.thumb_size in order else "M")
        new = order[max(0, min(len(order) - 1, idx + delta))]
        if new == prefs.thumb_size:
            return
        prefs.thumb_size = new
        self.call_later(self._rebuild_for_size)

    async def _rebuild_for_size(self) -> None:
        await self._build_tiles()
        self.select(self._selected)
        self._decode_visible()

    def action_reveal(self) -> None:
        reveal = getattr(self.app, "reveal_path", None)
        if reveal:
            reveal(self._run_dir)

    def action_copy_path(self) -> None:
        item = self._items[self._selected] if self._items else None
        if item and item.get("filename"):
            path = self._run_dir / "images" / str(item["filename"])
        else:
            path = self._run_dir
        self.app.copy_to_clipboard(str(path))
        self.notify("Path copied.", timeout=2)

    def action_back(self) -> None:
        self.app.pop_screen()
