"""The four hand-built themes and the literal-colour resolver.

Themes follow the "Darkroom" design language. Every theme defines the **full**
custom-variable set (CSS referencing an undefined ``$var`` fails to parse), and
:meth:`app.PortraitApp.get_theme_variable_defaults` returns :data:`VARIABLE_DEFAULTS`
so even Textual's built-in themes resolve our variables.

Rich renderables (RichLog lines, the batch matrix, half-block art captions)
cannot use ``$theme-variables`` — they need literal hex strings. :func:`literal`
resolves a variable name to a hex for the app's *current* theme, cached per theme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.theme import Theme

if TYPE_CHECKING:
    from textual.app import App

# Custom variable keys every theme must define (and the app-level defaults).
_DARKROOM_VARIABLES = {
    "border-soft": "#283146",
    "border-strong": "#3D4A66",
    "ink": "#0A0D14",
    "age-accent": "#F59E0B",
    "gender-accent": "#A78BFA",
    "eth-accent": "#2DD4BF",
    "grad-a": "#8B5CF6",
    "grad-b": "#22D3EE",
    "tele-pending": "#3B4254",
    "tele-queued": "#566073",
    "tele-running": "#22D3EE",
    "tele-retry": "#F5B82E",
    "tele-ok": "#34D399",
    "tele-fail": "#F4587A",
    "tele-est": "#8B5CF6",
    "tele-act": "#34D399",
    "tele-act-soft": "#A7F3D0",
    "footer-key-foreground": "#22D3EE",
    "block-cursor-background": "#8B5CF6",
    "block-cursor-foreground": "#0A0D14",
    "block-cursor-text-style": "bold",
    "input-selection-background": "#8B5CF6 35%",
    "link-color": "#22D3EE",
    "button-focus-text-style": "bold reverse",
}

# Defaults used when the active theme does not define our custom variables
# (e.g. the user picks a Textual built-in from the palette's theme picker).
VARIABLE_DEFAULTS: dict[str, str] = dict(_DARKROOM_VARIABLES)

DARKROOM = Theme(
    name="darkroom",
    primary="#8B5CF6",
    secondary="#22D3EE",
    accent="#F471B5",
    foreground="#E8ECF4",
    background="#0A0D14",
    surface="#11151F",
    panel="#1A2030",
    boost="#222A3D",
    success="#34D399",
    warning="#F5B82E",
    error="#F4587A",
    dark=True,
    variables=dict(_DARKROOM_VARIABLES),
)

GALLERY = Theme(
    name="gallery",
    primary="#6D28D9",
    secondary="#0E7490",
    accent="#BE185D",
    foreground="#20242E",
    background="#F4F2ED",
    surface="#FFFFFF",
    panel="#EAE7DF",
    boost="#E1DDD2",
    success="#047857",
    warning="#B45309",
    error="#BE123C",
    dark=False,
    variables={
        "border-soft": "#D8D3C8",
        "border-strong": "#B9B2A3",
        "ink": "#FFFFFF",
        "age-accent": "#B45309",
        "gender-accent": "#6D28D9",
        "eth-accent": "#0F766E",
        "grad-a": "#6D28D9",
        "grad-b": "#0E7490",
        "tele-pending": "#C9C3B6",
        "tele-queued": "#A8A092",
        "tele-running": "#0E7490",
        "tele-retry": "#B45309",
        "tele-ok": "#047857",
        "tele-fail": "#BE123C",
        "tele-est": "#6D28D9",
        "tele-act": "#047857",
        "tele-act-soft": "#065F46",
        "footer-key-foreground": "#0E7490",
        "block-cursor-background": "#6D28D9",
        "block-cursor-foreground": "#FFFFFF",
        "block-cursor-text-style": "bold",
        "input-selection-background": "#6D28D9 25%",
        "link-color": "#0E7490",
        "button-focus-text-style": "bold reverse",
    },
)

SYNTHWAVE = Theme(
    name="synthwave",
    primary="#FF7EDB",
    secondary="#36F9F6",
    accent="#FEDE5D",
    foreground="#F2E7FE",
    background="#1A1430",
    surface="#241B3E",
    panel="#2E2250",
    boost="#3B2C66",
    success="#72F1B8",
    warning="#FEDE5D",
    error="#FE4450",
    dark=True,
    variables={
        "border-soft": "#463569",
        "border-strong": "#5C478A",
        "ink": "#1A1430",
        "age-accent": "#FF9E64",
        "gender-accent": "#D67EFF",
        "eth-accent": "#36F9F6",
        "grad-a": "#FF7EDB",
        "grad-b": "#36F9F6",
        "tele-pending": "#463569",
        "tele-queued": "#5C478A",
        "tele-running": "#36F9F6",
        "tele-retry": "#FEDE5D",
        "tele-ok": "#72F1B8",
        "tele-fail": "#FE4450",
        "tele-est": "#FF7EDB",
        "tele-act": "#72F1B8",
        "tele-act-soft": "#B8FBD9",
        "footer-key-foreground": "#36F9F6",
        "block-cursor-background": "#FF7EDB",
        "block-cursor-foreground": "#1A1430",
        "block-cursor-text-style": "bold",
        "input-selection-background": "#FF7EDB 35%",
        "link-color": "#36F9F6",
        "button-focus-text-style": "bold reverse",
    },
)

SAFELIGHT = Theme(
    name="safelight",
    primary="#FF4D3D",
    secondary="#FF8A5C",
    accent="#FFC061",
    foreground="#FFD9C2",
    background="#0C0606",
    surface="#150A08",
    panel="#1E0F0C",
    boost="#291511",
    success="#FFE5C2",
    warning="#FFB13D",
    error="#FF2E2E",
    dark=True,
    variables={
        "border-soft": "#3A1B14",
        "border-strong": "#56281D",
        "ink": "#0C0606",
        "age-accent": "#FF9E3D",
        "gender-accent": "#FF6F61",
        "eth-accent": "#FFC999",
        "grad-a": "#FF2E2E",
        "grad-b": "#FFC061",
        "tele-pending": "#3A1B14",
        "tele-queued": "#56281D",
        "tele-running": "#FF8A5C",
        "tele-retry": "#FFB13D",
        "tele-ok": "#FFE5C2",
        "tele-fail": "#FF2E2E",
        "tele-est": "#FF4D3D",
        "tele-act": "#FFE5C2",
        "tele-act-soft": "#FFD9C2",
        "footer-key-foreground": "#FF8A5C",
        "block-cursor-background": "#FF4D3D",
        "block-cursor-foreground": "#0C0606",
        "block-cursor-text-style": "bold",
        "input-selection-background": "#FF4D3D 35%",
        "link-color": "#FF8A5C",
        "button-focus-text-style": "bold reverse",
    },
)

THEMES: tuple[Theme, ...] = (DARKROOM, GALLERY, SYNTHWAVE, SAFELIGHT)
THEME_CYCLE: tuple[str, ...] = tuple(t.name for t in THEMES)

# Keys (beyond the custom variables) resolvable through literal(): generated
# theme colours that Rich-markup contexts commonly need.
_THEME_FIELD_KEYS = (
    "primary", "secondary", "accent", "foreground", "background",
    "surface", "panel", "boost", "success", "warning", "error",
)

_literal_cache: dict[str, dict[str, str]] = {}


def literal(app: "App", name: str) -> str:
    """Resolve a variable/colour name (no ``$``) to a literal hex for the
    app's current theme. Falls back to the darkroom value."""
    name = name.lstrip("$")
    theme_name = app.theme
    cache = _literal_cache.get(theme_name)
    if cache is None:
        cache = {}
        theme = app.available_themes.get(theme_name)
        if theme is not None:
            for key in _THEME_FIELD_KEYS:
                value = getattr(theme, key, None)
                if value:
                    cache[key] = value
            cache.update(theme.variables)
        _literal_cache[theme_name] = cache
    if name in cache:
        return cache[name]
    if name in VARIABLE_DEFAULTS:
        return VARIABLE_DEFAULTS[name]
    return getattr(DARKROOM, name, None) or "#E8ECF4"


def text_muted(app: "App") -> str:
    """A muted text colour literal for the current theme."""
    theme = app.available_themes.get(app.theme)
    if theme is not None and not theme.dark:
        return "#6B7280"
    return "#8A93A6"


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def gradient_text(s: str, c1: str, c2: str, *, bold: bool = True) -> "Text":
    """Per-character ``c1 → c2`` gradient as a Rich ``Text`` (the wordmark)."""
    from rich.text import Text

    r1, g1, b1 = _hex_rgb(c1)
    r2, g2, b2 = _hex_rgb(c2)
    n = max(1, len(s) - 1)
    text = Text(no_wrap=True)
    for i, ch in enumerate(s):
        f = i / n
        colour = "#{:02x}{:02x}{:02x}".format(
            round(r1 + (r2 - r1) * f),
            round(g1 + (g2 - g1) * f),
            round(b1 + (b2 - b1) * f),
        )
        text.append(ch, style=f"bold {colour}" if bold else colour)
    return text


def wordmark(app: "App", s: str) -> "Text":
    """Gradient text using the current theme's ``grad-a``/``grad-b`` stops."""
    return gradient_text(s, literal(app, "grad-a"), literal(app, "grad-b"))
