"""Half-block image rendering: a PNG on disk → a Rich ``Text`` of ``▀`` cells.

Each terminal cell carries two pixels (foreground = top, background = bottom),
so a ``cols × rows`` render decodes the image at ``cols × rows*2``. Adjacent
cells with identical colours are merged into single styled spans.

Pure module: no widget imports. Decoding is CPU-bound (~20 ms for a 1024²
PNG), so callers must run :func:`render_halfblock` inside a thread worker and
post the result back as a message. Results are LRU-cached on
``(path, mtime, cols, rows)`` so re-renders (theme switches, scrolling) are free.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rich.color import Color
from rich.style import Style
from rich.text import Text

HALF_BLOCK = "▀"


class ImageUnreadable(Exception):
    """The file is missing, not an image, or otherwise undecodable."""


@lru_cache(maxsize=256)
def _render_cached(path: str, mtime_ns: int, cols: int, rows: int) -> Text:
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            fitted = ImageOps.fit(rgb, (cols, rows * 2), Image.Resampling.LANCZOS)
    except Exception as exc:  # noqa: BLE001 - any decode failure is "unreadable"
        raise ImageUnreadable(str(exc)) from exc

    pixels = fitted.load()
    text = Text(no_wrap=True)
    for y in range(rows):
        run_start = 0
        run_top = pixels[0, y * 2]
        run_bottom = pixels[0, y * 2 + 1]
        for x in range(1, cols + 1):
            if x < cols:
                top = pixels[x, y * 2]
                bottom = pixels[x, y * 2 + 1]
                if top == run_top and bottom == run_bottom:
                    continue
            text.append(
                HALF_BLOCK * (x - run_start),
                style=Style(
                    color=Color.from_rgb(*run_top),
                    bgcolor=Color.from_rgb(*run_bottom),
                ),
            )
            if x < cols:
                run_start = x
                run_top = top
                run_bottom = bottom
        if y < rows - 1:
            text.append("\n")
    return text


def render_halfblock(path: str | Path, cols: int, rows: int) -> Text:
    """Render ``path`` as half-block art at ``cols × rows`` cells.

    Raises :class:`ImageUnreadable` for missing/corrupt files. The returned
    ``Text`` is shared via the cache — treat it as immutable.
    """
    cols = max(2, int(cols))
    rows = max(1, int(rows))
    p = Path(path)
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError as exc:
        raise ImageUnreadable(str(exc)) from exc
    return _render_cached(str(p), mtime_ns, cols, rows)


def skeleton(cols: int, rows: int, *, label: str = "", color: str = "#566073") -> Text:
    """A placeholder block shown while decoding (or for failed frames)."""
    cols = max(2, int(cols))
    rows = max(1, int(rows))
    text = Text(no_wrap=True)
    for y in range(rows):
        text.append("▒" * cols, style=Style(color=color))
        if y < rows - 1:
            text.append("\n")
    if label and rows >= 1:
        mid = rows // 2
        line_start = sum(cols + 1 for _ in range(mid))
        start = line_start + max(0, (cols - len(label)) // 2)
        end = min(start + len(label), line_start + cols)
        plain = text.plain
        new = plain[:start] + label[: end - start] + plain[end:]
        result = Text(new, no_wrap=True)
        result.stylize(Style(color=color))
        result.stylize(Style(color="#E8ECF4", bgcolor=None, bold=True), start, end)
        return result
    return text
