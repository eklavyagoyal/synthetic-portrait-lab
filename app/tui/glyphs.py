"""Glyph kit for the TUI — every status glyph in one place, with an ASCII fallback.

The whole app draws from this safe monospace set (no emoji, no Nerd Font).
Setting ``PORTRAIT_TUI_ASCII=1`` swaps in a plain-ASCII table for terminals
that cannot render the Unicode set.
"""

from __future__ import annotations

import os

ASCII_MODE = os.environ.get("PORTRAIT_TUI_ASCII", "").strip().lower() in ("1", "true", "yes")

if not ASCII_MODE:
    DIAMOND = "◆"
    DIAMOND_HOLLOW = "◇"
    DOT = "●"
    DOT_HALF = "◐"
    DOT_HOLLOW = "○"
    TRIANGLE = "▲"
    ARROW = "▸"
    CHECK = "✓"
    CROSS = "✕"
    RETRY = "↻"
    WARN = "⚠"
    RULE_HEAVY = "━"
    ELLIPSIS = "…"

    # batch-matrix cell glyphs
    M_PENDING = "·"
    M_QUEUED = "○"
    M_RETRY = "↻"
    M_OK = "■"
    M_FAIL = "✕"
    M_SKIP = "·"
    SPINNER_CELLS = "◐◓◑◒"

    # bars / meters
    BAR = "█"
    BAR_FAIL = "▒"
    BAR_EMPTY = "░"
    SPARK_LEVELS = " ▁▂▃▄▅▆▇█"
    DRAIN_ON = "▰"
    DRAIN_OFF = "▱"
    SPINNER_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    BUSY = "⠴"
else:
    DIAMOND = "*"
    DIAMOND_HOLLOW = "+"
    DOT = "o"
    DOT_HALF = "o"
    DOT_HOLLOW = "."
    TRIANGLE = "^"
    ARROW = ">"
    CHECK = "v"
    CROSS = "x"
    RETRY = "r"
    WARN = "!"
    RULE_HEAVY = "-"
    ELLIPSIS = "..."

    M_PENDING = "."
    M_QUEUED = "o"
    M_RETRY = "r"
    M_OK = "@"
    M_FAIL = "x"
    M_SKIP = "."
    SPINNER_CELLS = "|/-\\"

    BAR = "#"
    BAR_FAIL = "%"
    BAR_EMPTY = "."
    SPARK_LEVELS = " .:-=+*#%"
    DRAIN_ON = "="
    DRAIN_OFF = "-"
    SPINNER_BRAILLE = "|/-\\"
    BUSY = "~"


def spark_bar(values: list[float], width: int | None = None) -> str:
    """Render a list of values as a one-line sparkline string."""
    if not values:
        return ""
    if width is not None and len(values) > width > 0:
        values = values[-width:]
    top = max(values) or 1.0
    levels = SPARK_LEVELS
    out = []
    for v in values:
        idx = round((max(0.0, v) / top) * (len(levels) - 1))
        out.append(levels[idx])
    return "".join(out)


def meter(fraction: float, width: int, ok_chars: int | None = None) -> str:
    """A simple block meter, e.g. ``████░░░░``."""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return BAR * filled + BAR_EMPTY * (width - filled)
