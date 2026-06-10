"""BatchMatrix — every planned item as one live glyph.

A custom widget (a DataTable cannot pack 1–2-char cells this tightly). Reads
cell states from the app-owned :class:`RunTelemetry` on each repaint tick;
holds a keyboard cursor; supports four colour modes (``state`` plus colouring
by each demographic dimension — a live bias check).

Glyphs: ``·`` pending  ``○`` queued  ``◐◓◑◒`` running  ``↻`` retrying
``■`` succeeded  ``✕`` failed  (dim struck ``·`` = skipped by cancel).
"""

from __future__ import annotations

from typing import Optional

from rich.style import Style
from rich.text import Text
from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from .. import glyphs, palette
from ..telemetry import CellState, ItemMeta, RunTelemetry

COLOR_MODES = ("state", "age", "gender", "ethnicity")

_STATE_GLYPH = {
    CellState.PENDING: glyphs.M_PENDING,
    CellState.QUEUED: glyphs.M_QUEUED,
    CellState.RETRY: glyphs.M_RETRY,
    CellState.OK: glyphs.M_OK,
    CellState.FAIL: glyphs.M_FAIL,
    CellState.SKIP: glyphs.M_SKIP,
}

_STATE_VAR = {
    CellState.PENDING: "tele-pending",
    CellState.QUEUED: "tele-queued",
    CellState.RUNNING: "tele-running",
    CellState.RETRY: "tele-retry",
    CellState.OK: "tele-ok",
    CellState.FAIL: "tele-fail",
    CellState.SKIP: "tele-pending",
}

_DIM_VAR = {"age": "age-accent", "gender": "gender-accent", "ethnicity": "eth-accent"}

MAX_RENDERED_ITEMS = 5000


def _blend(hex_a: str, hex_b: str, f: float) -> str:
    ra, ga, ba = palette._hex_rgb(hex_a)
    rb, gb, bb = palette._hex_rgb(hex_b)
    return "#{:02x}{:02x}{:02x}".format(
        round(ra + (rb - ra) * f), round(ga + (gb - ga) * f), round(ba + (bb - ba) * f)
    )


class BatchMatrix(Widget, can_focus=True):
    """The cell grid. Attach telemetry, then call :meth:`repaint_cells` on a ticker."""

    DEFAULT_CSS = """
    BatchMatrix {
        width: 1fr;
        height: auto;
        &:focus { background: $boost; }
    }
    """

    class CellHighlighted(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    class CellActivated(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    cursor: reactive[int] = reactive(0)
    color_mode: reactive[str] = reactive("state")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._telemetry: Optional[RunTelemetry] = None
        self._tick = 0
        self._bucket_colors: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ #
    # Data attachment
    # ------------------------------------------------------------------ #
    def attach(self, telemetry: RunTelemetry) -> None:
        self._telemetry = telemetry
        self._bucket_colors = {}
        self.cursor = 0
        self.refresh(layout=True)

    @property
    def total(self) -> int:
        return self._telemetry.total if self._telemetry else 0

    def item_meta(self, index: int) -> Optional[ItemMeta]:
        if self._telemetry and 0 <= index < len(self._telemetry.items):
            return self._telemetry.items[index]
        return None

    def repaint_cells(self, tick: int) -> None:
        """Called by the screen's fast ticker; advances spinner frames."""
        self._tick = tick
        self.refresh()

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _pitch(self) -> int:
        return 2 if self.total <= 100 else 1

    def _per_row(self, width: int) -> int:
        return max(1, width // self._pitch())

    def get_content_width(self, container, viewport) -> int:
        return container.width

    def get_content_height(self, container, viewport, width: int) -> int:
        total = min(self.total, MAX_RENDERED_ITEMS)
        if total <= 0:
            return 1
        rows = (total + self._per_row(width) - 1) // self._per_row(width)
        if self.total > MAX_RENDERED_ITEMS:
            rows += 1
        return rows

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _bucket_color(self, dim: str, value: str) -> str:
        if getattr(self, "_colors_theme", None) != self.app.theme:
            self._bucket_colors = {}  # theme changed — recompute accent shades
            self._colors_theme = self.app.theme
        per_dim = self._bucket_colors.get(dim)
        if per_dim is None or value not in per_dim:
            accent = palette.literal(self.app, _DIM_VAR[dim])
            counter = (
                self._telemetry.planned_by_bucket.get(dim, {}) if self._telemetry else {}
            )
            values = sorted(counter.keys())
            n = max(1, len(values))
            toward = "#FFFFFF" if self.app.current_theme.dark else "#000000"
            per_dim = {
                v: _blend(accent, toward, 0.55 * (i / max(1, n - 1)))
                for i, v in enumerate(values)
            }
            self._bucket_colors[dim] = per_dim
        return per_dim.get(value, palette.literal(self.app, _DIM_VAR[dim]))

    def _cell(self, index: int, state: CellState) -> tuple[str, Style]:
        app = self.app
        if state == CellState.RUNNING:
            glyph = glyphs.SPINNER_CELLS[(self._tick + index) % len(glyphs.SPINNER_CELLS)]
        else:
            glyph = _STATE_GLYPH[state]

        if self.color_mode == "state" or state == CellState.FAIL:
            colour = palette.literal(app, _STATE_VAR[state])
        else:
            meta = self.item_meta(index)
            value = getattr(meta, self.color_mode, "") if meta else ""
            colour = self._bucket_color(self.color_mode, value)

        style = Style(color=colour)
        if state in (CellState.PENDING, CellState.QUEUED):
            style += Style(dim=True)
        elif state == CellState.SKIP:
            style += Style(dim=True, strike=True)
        elif state in (CellState.RUNNING, CellState.RETRY):
            style += Style(bold=True)
        if index == self.cursor and self.has_focus:
            style += Style(reverse=True)
        return glyph, style

    def render(self) -> Text:
        tele = self._telemetry
        text = Text(no_wrap=True)
        if tele is None or tele.total == 0:
            text.append("no items planned", style=Style(dim=True))
            return text

        total = min(tele.total, MAX_RENDERED_ITEMS)
        width = max(1, self.content_size.width or 1)
        per_row = self._per_row(width)
        pitch = self._pitch()
        pad = " " * (pitch - 1)

        for start in range(0, total, per_row):
            if start:
                text.append("\n")
            for index in range(start, min(start + per_row, total)):
                glyph, style = self._cell(index, tele.states[index])
                text.append(glyph, style=style)
                if pad and index < min(start + per_row, total) - 1:
                    text.append(pad)
        if tele.total > MAX_RENDERED_ITEMS:
            text.append(
                f"\n{glyphs.ELLIPSIS} {tele.total - MAX_RENDERED_ITEMS} more items "
                f"(aggregate view) · ok {tele.success} · fail {tele.failed}",
                style=Style(dim=True),
            )
        return text

    # ------------------------------------------------------------------ #
    # Cursor + keys
    # ------------------------------------------------------------------ #
    def watch_cursor(self, _old: int, new: int) -> None:
        self.post_message(self.CellHighlighted(new))
        self._scroll_cursor_into_view(new)
        self.refresh()

    def _scroll_cursor_into_view(self, index: int) -> None:
        width = max(1, self.content_size.width or 1)
        row = index // self._per_row(width)
        parent = self.parent
        scroll_to = getattr(parent, "scroll_to", None)
        if scroll_to is None or parent is None:
            return
        try:
            visible = parent.container_size.height or 1
            current_y = int(parent.scroll_offset.y)
            if row < current_y or row >= current_y + visible:
                scroll_to(y=max(0, row - visible // 2), animate=False)
        except Exception:  # noqa: BLE001 - scrolling is best-effort
            pass

    def _move_cursor(self, delta: int) -> None:
        if self.total:
            self.cursor = max(0, min(self.total - 1, self.cursor + delta))

    def _jump_failed(self, direction: int) -> None:
        tele = self._telemetry
        if not tele:
            return
        n = tele.total
        for step in range(1, n + 1):
            idx = (self.cursor + direction * step) % n
            if tele.states[idx] == CellState.FAIL:
                self.cursor = idx
                return

    def set_cursor(self, index: int) -> None:
        if self.total:
            self.cursor = max(0, min(self.total - 1, index))

    def cycle_color_mode(self) -> str:
        idx = COLOR_MODES.index(self.color_mode)
        self.color_mode = COLOR_MODES[(idx + 1) % len(COLOR_MODES)]
        self.refresh()
        return self.color_mode

    def on_key(self, event: events.Key) -> None:
        width = max(1, self.content_size.width or 1)
        per_row = self._per_row(width)
        handlers = {
            "left": lambda: self._move_cursor(-1),
            "right": lambda: self._move_cursor(1),
            "up": lambda: self._move_cursor(-per_row),
            "down": lambda: self._move_cursor(per_row),
            "home": lambda: self.set_cursor(0),
            "end": lambda: self.set_cursor(self.total - 1),
            "n": lambda: self._jump_failed(1),
            "p": lambda: self._jump_failed(-1),
        }
        handler = handlers.get(event.key)
        if handler is not None:
            event.stop()
            handler()
        elif event.key == "enter":
            event.stop()
            self.post_message(self.CellActivated(self.cursor))

    def on_click(self, event: events.Click) -> None:
        width = max(1, self.content_size.width or 1)
        per_row = self._per_row(width)
        index = event.y * per_row + (event.x // self._pitch())
        if 0 <= index < self.total:
            self.focus()
            self.set_cursor(index)
