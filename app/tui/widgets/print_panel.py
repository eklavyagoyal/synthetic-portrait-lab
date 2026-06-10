"""PrintPanel — "the print": the latest developed portrait, in half-blocks.

Decodes run in a thread worker (Pillow is CPU-bound); the worker posts a
:class:`ThumbReady` back to this widget and the image fades in ("develops").
Pixels are sacred — images are never tinted by the theme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .. import glyphs, imaging
from ..messages import ThumbReady


class PrintPanel(Vertical):
    DEFAULT_CSS = """
    PrintPanel {
        width: auto;
        height: auto;

        #print-image { width: auto; height: auto; }
        #print-caption { color: $text-muted; height: auto; }
    }
    """

    def __init__(self, cols: int = 36, rows: int = 18, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self.rows = rows
        self._current_path: Optional[Path] = None
        self._pending_caption: str = ""

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder(), id="print-image")
        yield Static("", id="print-caption")

    def _placeholder(self):
        return imaging.skeleton(
            self.cols, self.rows, label=f"{glyphs.BUSY} developing first frame…"
        )

    def show(self, path: Path, caption: str = "") -> None:
        """Queue a decode for ``path``; applied when the thread worker reports back."""
        if path == self._current_path:
            self.query_one("#print-caption", Static).update(caption)
            return
        self._current_path = path
        self._pending_caption = caption
        self._decode(path, self.cols, self.rows)

    @work(thread=True, exclusive=True, group="print-decode")
    def _decode(self, path: Path, cols: int, rows: int) -> None:
        try:
            text = imaging.render_halfblock(path, cols, rows)
            self.post_message(ThumbReady(str(path), path, text))
        except imaging.ImageUnreadable as exc:
            self.post_message(ThumbReady(str(path), path, None, error=str(exc)))

    @on(ThumbReady)
    def _apply(self, message: ThumbReady) -> None:
        message.stop()
        if self._current_path is None or str(self._current_path) != message.key:
            return  # a newer frame superseded this decode
        image = self.query_one("#print-image", Static)
        if message.text is None:
            image.update(
                imaging.skeleton(self.cols, self.rows, label=f"{glyphs.CROSS} unreadable")
            )
            self.query_one("#print-caption", Static).update("")
            return
        image.update(message.text)
        self.query_one("#print-caption", Static).update(self._pending_caption)
        # the bordered card around this panel carries the filename
        parent = self.parent
        if parent is not None and getattr(parent, "border_title", None) is not None:
            parent.border_title = f"the print · {self._current_path.name}"
        # develop-in: fade the new print up from 15 %
        image.styles.opacity = 0.15
        image.styles.animate("opacity", 1.0, duration=0.35, easing="out_cubic")
