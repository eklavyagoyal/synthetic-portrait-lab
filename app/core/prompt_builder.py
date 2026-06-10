"""Builds the final portrait prompt from :class:`PromptOptions`.

Two invariants the test-suite enforces:

* every hard requirement is present in every prompt, at every variation level;
* every negative constraint is present in every prompt, at every variation level.

The variation system adds *permitted* natural variation on top of those
invariants — it never relaxes or contradicts them.
"""

from __future__ import annotations

from typing import Optional

from .models import PromptOptions
from .sizes import parse_size

# Hard requirements are always present, verbatim, in the order below.
HARD_REQUIREMENTS: list[str] = [
    "Front-facing face, looking directly at the camera.",
    "Full head visible, including the top of the hair/head, ears, jawline, and chin.",
    "Exactly one person in the image.",
    "Centered composition.",
    "Neutral, natural facial expression.",
    "Plain light gray or off-white background.",
    "Realistic skin texture and natural lighting.",
    "High-quality, sharp, print-ready image.",
    "Passport-style / studio portrait.",
    "1:1 square image.",
]

# Negative constraints are always present, verbatim.
NEGATIVE_CONSTRAINTS: list[str] = [
    "Cropped head",
    "Side angle",
    "Three-quarter view",
    "Hats",
    "Sunglasses",
    "Masks",
    "Face paint",
    "Heavy accessories",
    "Text",
    "Watermarks",
    "Logos",
    "Frames",
    "Multiple people",
    "Cartoon, illustration, CGI, glamour, fashion, or stylized appearance",
]

# Variation guidance, keyed by level. Higher levels are supersets in spirit:
# each describes how much natural variation is *permitted* without breaking the
# hard requirements above.
_VARIATION_GUIDANCE: dict[int, list[str]] = {
    0: [
        "Strict repeatability: keep the studio setup, framing and lighting as "
        "consistent and standardized as possible across images. Minimize variation.",
    ],
    1: [
        "Low variation: keep an essentially identical studio setup.",
        "Permit only very subtle differences in face shape and hairstyle.",
    ],
    2: [
        "Moderate natural variation, all within a standardized passport/studio setup:",
        "- subtle face-shape variation",
        "- natural hair length and style variation",
        "- slight, soft lighting variation",
        "- slight skin-texture variation",
        "- small differences within a neutral, natural expression",
    ],
    3: [
        "High natural variation while strictly preserving every passport/studio "
        "constraint above:",
        "- noticeable (but realistic) face-shape variation",
        "- varied natural hair length and style",
        "- varied soft studio lighting",
        "- varied realistic skin texture",
        "- slight camera-distance variation — the full head, hair top, ears, "
        "jawline and chin must remain fully visible and centered",
        "- small natural differences within a neutral expression",
    ],
}


def variation_guidance(level: int) -> list[str]:
    """Return the permitted-variation lines for a level (clamped to 0–3)."""
    level = max(0, min(3, int(level)))
    return list(_VARIATION_GUIDANCE[level])


# --------------------------------------------------------------------------- #
# Framing — how large the head appears in the frame
#
# IMPORTANT: ``head_height_pct`` is a *generation instruction*, not a verified
# measurement. The model approximates it; it is not mathematically reliable.
# For exact control, post-process the output:
#   TODO: detect the face/head bounding box and crop/pad each image so the head
#   occupies exactly head_height_pct of the height (center-face crop strategy).
# --------------------------------------------------------------------------- #
# Preset label -> head-height percentage (top of hair to chin).
FRAMING_PRESETS: list[tuple[str, int]] = [
    ("close headshot", 75),
    ("standard headshot", 60),
    ("loose headshot", 45),
    ("upper body", 30),
]
DEFAULT_HEAD_HEIGHT_PCT = 60


def framing_label(head_height_pct: int) -> str:
    """Name the framing for a head-height %, or ``"custom"`` if it's off-preset."""
    for label, pct in FRAMING_PRESETS:
        if pct == head_height_pct:
            return label
    return "custom"


def _shoulder_instruction(pct: int) -> str:
    """How much of the neck/shoulders/torso to include, by head-height %."""
    if pct >= 70:
        return "Tight headshot: only the neck and the top of the collar are visible."
    if pct >= 55:
        return "Standard ID-style headshot: the neck and the top of the shoulders are visible."
    if pct >= 40:
        return "Loose portrait: the full shoulders are visible."
    return "Upper-body portrait: the chest and upper torso are visible."


def framing_composition_lines(head_height_pct: int) -> list[str]:
    """Composition guidance describing how large the head should appear."""
    pct = max(10, min(95, int(head_height_pct)))
    return [
        "The subject is centered and front-facing.",
        "Include the full head with clean margin above the hair; "
        "do not crop the hair, ears, chin, or neck.",
        f"The head, measured from the top of the hair to the chin, occupies "
        f"approximately {pct}% of the image height.",
        _shoulder_instruction(pct),
    ]


def _orientation_requirement(size: Optional[str]) -> str:
    """Render the orientation hard-requirement from a ``"WxH"`` size.

    Defaults to the verbatim square requirement when the size is missing or
    square, so existing (square) prompts — and the prompt invariants the tests
    pin — are unchanged. Non-square sizes get an honest orientation line instead
    of the contradictory "1:1 square image."."""
    parsed = parse_size(size) if size else None
    if parsed is None or parsed[0] == parsed[1]:
        return "1:1 square image."
    width, height = parsed
    if height > width:
        return "Vertical (portrait) orientation, taller than wide — keep the full head centered."
    return "Horizontal (landscape) orientation, wider than tall — keep the full head centered."


def _as_requirement(text: str) -> str:
    """Normalize an overridable requirement to 'Capitalized sentence.' form so
    that the default values render byte-for-byte identical to HARD_REQUIREMENTS."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def build_prompt(options: PromptOptions) -> str:
    """Render the full prompt text for one portrait."""
    lines: list[str] = []
    lines.append(
        "Generate a photorealistic portrait of one synthetic human person who does not exist."
    )
    lines.append("")

    # Hard requirements — overridable text values are substituted in-place so the
    # request-level defaults (background/expression/lighting/style) take effect,
    # while the structural requirements stay fixed. Defaults render verbatim.
    hard = list(HARD_REQUIREMENTS)
    hard[4] = _as_requirement(options.expression)
    hard[5] = _as_requirement(options.background)
    hard[9] = _orientation_requirement(options.size)
    lines.append("Hard requirements:")
    for req in hard:
        lines.append(f"- {req}")
    for extra in options.extra_positive_constraints:
        if extra.strip():
            lines.append(f"- {extra.strip()}")
    lines.append("")

    # Composition / framing — how the head sits in the frame
    lines.append("Composition:")
    for line in framing_composition_lines(options.head_height_pct):
        lines.append(f"- {line}")
    lines.append("")

    # Demographic target
    lines.append("Demographic target:")
    lines.append(f"- Age: {options.age_bucket}")
    lines.append(f"- Gender presentation: {options.gender_bucket}")
    lines.append(
        f"- Apparent ancestry / skin-tone diversity target: {options.ethnicity_bucket}"
    )
    lines.append("")

    # Style + lighting (descriptive context, not a hard structural requirement)
    lines.append("Style:")
    lines.append(f"- {options.image_style}.")
    lines.append(f"- {options.lighting}.")
    lines.append("")

    # Permitted variation
    lines.append(f"Permitted natural variation (variation level {options.variation_level}):")
    for g in variation_guidance(options.variation_level):
        lines.append(g if g.startswith("-") else f"- {g}")
    if options.seed is not None:
        lines.append(f"- Deterministic seed in effect: {options.seed}.")
    lines.append("")

    # Negative constraints
    negatives = list(NEGATIVE_CONSTRAINTS)
    for extra in options.extra_negative_constraints:
        if extra.strip():
            negatives.append(extra.strip())
    lines.append("Do not include:")
    for neg in negatives:
        lines.append(f"- {neg}")

    return "\n".join(lines).strip()
