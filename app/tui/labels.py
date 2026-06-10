"""Short labels and chip builders for the demographic dimensions.

Canonical bucket strings (see :mod:`app.core.buckets`) are long; panels use a
short-label dictionary, keeping the full string for tooltips and metadata
panes. Chips are Rich/Textual markup strings — they use ``$theme-variables``
so they re-colour with the active theme (only valid inside Textual widgets;
RichLog lines must resolve literals via :func:`app.tui.palette.literal`).
"""

from __future__ import annotations

from rich.markup import escape as esc

from . import glyphs

SHORT_LABELS: dict[str, str] = {
    # age
    "young adult, 18 to 25": "18–25",
    "adult, 26 to 40": "26–40",
    "middle-aged adult, 41 to 60": "41–60",
    "older adult, 61 to 80": "61–80",
    # gender
    "male-presenting": "masc",
    "female-presenting": "fem",
    "androgynous or non-binary-presenting": "nonbinary",
    # ethnicity
    "East Asian": "E. Asian",
    "South Asian": "S. Asian",
    "Southeast Asian": "SE Asian",
    "Black African descent": "Black African",
    "Middle Eastern or North African": "MENA",
    "Latino or Hispanic": "Latino",
    "White European": "White Eur.",
    "mixed heritage": "Mixed",
}

DIMENSIONS = ("age", "gender", "ethnicity")

DIM_GLYPH = {
    "age": glyphs.DIAMOND,
    "gender": glyphs.DOT,
    "ethnicity": glyphs.TRIANGLE,
}

DIM_VAR = {
    "age": "$age-accent",
    "gender": "$gender-accent",
    "ethnicity": "$eth-accent",
}


def short(bucket: str) -> str:
    """Short display label for a bucket (custom buckets are truncated).

    The result is markup-escaped — custom bucket strings may contain ``[``.
    """
    label = SHORT_LABELS.get(bucket)
    if label is not None:
        return label
    bucket = bucket.strip()
    if len(bucket) > 14:
        bucket = bucket[:13] + glyphs.ELLIPSIS
    return esc(bucket)


def chip(dim: str, bucket: str, *, solid: bool = True) -> str:
    """A coloured demographic chip as Textual markup."""
    glyph = DIM_GLYPH.get(dim, glyphs.DIAMOND)
    var = DIM_VAR.get(dim, "$accent")
    label = short(bucket)
    if solid:
        return f"[$ink on {var}] {glyph} {label} [/]"
    return f"[{var}]{glyph} {label}[/]"


def triple_chips(age: str, gender: str, ethnicity: str, *, solid: bool = False) -> str:
    """The three chips for one planned item, space-separated."""
    return "  ".join(
        chip(dim, bucket, solid=solid)
        for dim, bucket in (("age", age), ("gender", gender), ("ethnicity", ethnicity))
    )


def badge(text: str, var: str = "$primary") -> str:
    """A small status badge, e.g. ``FREE`` / ``NO PRICE`` / ``EST``."""
    return f"[$ink on {var}] {text} [/]"
