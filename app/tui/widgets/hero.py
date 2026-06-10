"""The hero strip: gradient wordmark, screen rail, global status segment.

One ``Hero`` per screen (cheap — three Statics and a 1 s ticker). The status
segment is the app-wide run indicator: a live run is never invisible, no
matter which screen you are on. Clicking a rail item navigates.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from .. import glyphs, palette

RAIL_TARGETS = ("studio", "darkroom", "contact", "archive")


class RailItem(Static):
    """One clickable destination in the screen rail."""

    DEFAULT_CSS = """
    RailItem {
        width: auto;
        height: 2;
        margin: 0 2 0 0;
        color: $text-muted;
        &.-active { color: $text; text-style: bold; }
        &:hover { color: $secondary; }
    }
    """

    def __init__(self, target: str, *, active: bool = False) -> None:
        underline = glyphs.RULE_HEAVY * len(target) if active else ""
        super().__init__(
            f"{target}\n[$secondary]{underline}[/]",
            classes="-active" if active else None,
        )
        self.target = target

    def on_click(self) -> None:
        navigate = getattr(self.app, "navigate_rail", None)
        if navigate is not None:
            navigate(self.target)


class Hero(Horizontal):
    """Wordmark + rail + status. ``active`` names the highlighted rail item."""

    DEFAULT_CSS = """
    Hero {
        dock: top;
        height: 3;
        padding: 0 2;
        background: $background;
        border-bottom: solid $border-soft;

        #hero-brand { width: auto; height: 100%; margin: 0 4 0 0; }
        #hero-wordmark { width: auto; }
        #hero-tagline { width: auto; color: $text-muted; }
        #hero-rail { width: 1fr; height: 100%; padding: 0 0 0 2; }
        #hero-status {
            width: auto;
            min-width: 24;
            height: 100%;
            content-align: right middle;
            color: $text-muted;
        }
    }
    """

    def __init__(self, active: str = "studio") -> None:
        super().__init__()
        self.active = active

    def compose(self) -> ComposeResult:
        with Vertical(id="hero-brand"):
            yield Static(
                palette.wordmark(self.app, f"{glyphs.DIAMOND}  {self.app.title.upper()}"),
                id="hero-wordmark",
            )
            yield Static("synthetic portrait dataset darkroom", id="hero-tagline")
        with Horizontal(id="hero-rail"):
            for target in RAIL_TARGETS:
                yield RailItem(target, active=(target == self.active))
        yield Static(id="hero-status")

    def on_mount(self) -> None:
        self._last_theme = ""
        self._refresh()
        self.set_interval(1.0, self._refresh)

    def _refresh(self) -> None:
        if self.app.theme != self._last_theme:
            self._last_theme = self.app.theme
            self.query_one("#hero-wordmark", Static).update(
                palette.wordmark(self.app, f"{glyphs.DIAMOND}  {self.app.title.upper()}")
            )
        status = getattr(self.app, "hero_status", None)
        if callable(status):
            try:
                self.query_one("#hero-status", Static).update(status())
            except Exception:  # noqa: BLE001 - hero must never crash a screen
                pass
