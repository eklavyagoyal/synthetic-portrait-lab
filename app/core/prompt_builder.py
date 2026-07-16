"""Builds the final portrait prompt from :class:`PromptOptions`.

Two invariants the test-suite enforces:

* every hard requirement is present in every prompt, at every variation level;
* every negative constraint is present in every prompt, at every variation level.

The variation system adds *permitted* natural variation on top of those
invariants — it never relaxes or contradicts them.
"""

from __future__ import annotations

from typing import Optional

from .models import CaptureModality, PromptOptions
from .sizes import A4_PORTRAIT_ASPECT, parse_size

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
    "Glasses, spectacles, or eyewear of any kind",
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

# Extra negatives injected only for A4 face-portrait mode (prompt-only, no post-processing).
FACE_PORTRAIT_NEGATIVES: list[str] = [
    "Cut-off, cropped, or clipped chin",
    "Cut-off, cropped, or clipped hair",
    "Cut-off, cropped, or clipped ears",
    "Any part of the face touching or crossing an image edge",
    "Shoulders, chest, torso, or upper body",
    "Neck below the jawline",
    "Clothing, collar, or shirt visible",
    "Tight zoom that leaves no empty margin around the head",
    "Landscape orientation",
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

_FACE_PORTRAIT_VARIATION: dict[int, list[str]] = {
    0: [
        "Strict repeatability: identical studio setup, identical head scale, "
        "identical margins, identical lighting. Minimize variation.",
    ],
    1: [
        "Low variation: keep the same framing, margins, and head scale.",
        "Permit only very subtle face-shape and hairstyle differences.",
    ],
    2: [
        "Moderate natural variation within the fixed face-portrait framing above:",
        "- subtle face-shape variation",
        "- natural hair length and style variation",
        "- slight, soft lighting variation",
        "- slight skin-texture variation",
        "- small differences within a neutral expression",
        "- do NOT change camera distance, head scale, or margins",
    ],
    3: [
        "High natural variation while strictly preserving every face-portrait rule above:",
        "- noticeable (but realistic) face-shape variation",
        "- varied natural hair length and style",
        "- varied soft studio lighting",
        "- varied realistic skin texture",
        "- small natural differences within a neutral expression",
        "- do NOT change camera distance, head scale, or margins",
        "- the full head, hair top, ears, jawline, and chin must remain fully visible "
        "with all required empty margins on every side",
    ],
}


def variation_guidance(level: int, *, face_crop: bool = False) -> list[str]:
    """Return the permitted-variation lines for a level (clamped to 0–3)."""
    level = max(0, min(3, int(level)))
    table = _FACE_PORTRAIT_VARIATION if face_crop else _VARIATION_GUIDANCE
    return list(table[level])


# --------------------------------------------------------------------------- #
# Framing — how large the head appears in the frame
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


def _face_portrait_framing_lines(head_height_pct: int, size: Optional[str]) -> list[str]:
    """Unambiguous prompt-only instructions for A4 face portraits."""
    pct = max(50, min(65, int(head_height_pct)))
    parsed = parse_size(size) if size else None
    if parsed:
        width, height = parsed
        aspect = width / height
        canvas_line = (
            f"Canvas: portrait orientation, exactly {width}×{height} pixels "
            f"(width-to-height ratio ≈ {aspect:.3f}, matching ISO A4 paper at {A4_PORTRAIT_ASPECT:.3f})."
        )
    else:
        canvas_line = (
            "Canvas: portrait orientation with ISO A4 paper aspect ratio "
            f"(210 mm wide × 297 mm tall; width-to-height ≈ {A4_PORTRAIT_ASPECT:.3f})."
        )

    return [
        "MANDATORY FACE PORTRAIT — follow every rule below exactly; they override all other guidance.",
        canvas_line,
        "Subject: exactly one person, front-facing, eyes looking directly at the camera.",
        (
            "Framing: show ONLY the head and face — from the highest point of the hair "
            "down to the lowest point of the chin. Nothing below the chin."
        ),
        (
            "The entire head must be 100% visible and uncropped: top of hair, both ears, "
            "full jawline, and the complete chin."
        ),
        (
            "Mandatory empty background margins (do not let the head touch any edge): "
            "at least 12% of image height above the hair, at least 10% below the chin, "
            "and at least 8% of image width on each side outside the ears."
        ),
        (
            f"Head scale: the head (hair top to chin) should occupy roughly {pct}% of the "
            "image height — large enough for print detail, small enough to keep every "
            "required margin."
        ),
        "Do NOT show shoulders, neck below the jaw, chest, collar, clothing, or torso.",
        (
            "Before finishing, verify: hair top visible with margin above; both ears fully "
            "visible; entire chin fully visible with margin below; no facial feature cut off "
            "or touching any edge."
        ),
    ]


def framing_composition_lines(
    head_height_pct: int,
    face_crop: bool = False,
    *,
    size: Optional[str] = None,
) -> list[str]:
    """Composition guidance describing how large the head should appear."""
    if face_crop:
        return _face_portrait_framing_lines(head_height_pct, size)

    pct = max(10, min(95, int(head_height_pct)))
    return [
        "The subject is centered and front-facing.",
        "Include the full head with clean margin above the hair; do not crop the hair, ears, chin, or neck.",
        f"The head, measured from the top of the hair to the chin, occupies approximately {pct}% of the image height.",
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
    """Render the full prompt text for one image.

    Dispatches on the capture modality. RGB_FACE (the default) renders the
    passport/studio face portrait below, unchanged; other modalities delegate to
    their own builder. Add a branch here when a new modality is introduced.
    """
    if options.modality == CaptureModality.IR_IRIS:
        return _build_iris_prompt(options)

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

    section_title = "Face portrait framing:" if options.face_crop else "Composition:"
    lines.append(section_title)
    for line in framing_composition_lines(
        options.head_height_pct,
        options.face_crop,
        size=options.size,
    ):
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

    # Distinct individual — the per-image appearance layer. This is what makes
    # every portrait a unique person rather than a re-roll of an identical prompt.
    # It never relaxes the hard requirements above; expression stays neutral, the
    # face stays front-facing, no eyewear is added, accessories stay minimal.
    if options.appearance is not None:
        lines.append(
            "Distinct individual — render ONE specific, unique person with exactly "
            "these features (do not blend toward a generic average face):"
        )
        for line in options.appearance.to_prompt_lines():
            lines.append(f"- {line}")
        lines.append("")

    # Style + lighting (descriptive context, not a hard structural requirement)
    lines.append("Style:")
    lines.append(f"- {options.image_style}.")
    lines.append(f"- {options.lighting}.")
    lines.append("")

    # Permitted variation
    lines.append(f"Permitted natural variation (variation level {options.variation_level}):")
    for g in variation_guidance(options.variation_level, face_crop=options.face_crop):
        lines.append(g if g.startswith("-") else f"- {g}")
    if options.seed is not None:
        lines.append(f"- Deterministic seed in effect: {options.seed}.")
    lines.append("")

    # Negative constraints
    negatives = list(NEGATIVE_CONSTRAINTS)
    if options.face_crop:
        negatives.extend(FACE_PORTRAIT_NEGATIVES)
    for extra in options.extra_negative_constraints:
        if extra.strip():
            negatives.append(extra.strip())
    lines.append("Do not include:")
    for neg in negatives:
        lines.append(f"- {neg}")

    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# IR_IRIS modality — monochrome near-infrared iris capture
# --------------------------------------------------------------------------- #
# The face invariants above do not apply to an iris capture (there is no head,
# framing, background or expression to constrain), so this modality carries its
# own hard requirements and negatives rather than reusing the portrait ones.
# Structural invariants — always true, regardless of any realism knob.
IRIS_INVARIANTS: list[str] = [
    "The image is a single monochrome (8-bit grayscale) frame — one luminance "
    "channel, absolutely no colour, no colour tint, no sepia or blue cast.",
    "It is a plain, sharp photograph framed tightly on exactly ONE human eye and the "
    "skin immediately around it, lit by near-infrared (~850 nm) light — the kind of "
    "frame a near-infrared eye camera records.",
    "Landscape framing with the iris centred and sitting well inside the frame (clear "
    "margin on every side, more to the left and right than above and below); the iris "
    "is never clipped by the image edge.",
    "The iris diameter spans roughly 40-55% of the image height: a moderate close-up, "
    "not an extreme macro.",
    "Correct near-infrared tones: a deep-black round pupil at the centre; a bright, "
    "smooth light-grey sclera; and, between them, a mid-grey richly textured iris ring.",
    "Even a heavily pigmented (dark-brown) iris must appear MID-GREY and full of visible "
    "texture — never a dark, flat or black featureless disc — because melanin is "
    "transparent to near-infrared light, so the stroma shows through for every eye colour.",
    "Anatomically correct, sharply resolved iris structure: an inner pupillary zone and "
    "an outer ciliary zone divided by the collarette; radial furrows; concentric "
    "contraction furrows; crypts near the collarette and periphery; fine trabecular "
    "striations — the natural stroma is a dense, irregular texture.",
    "Sharp focus on the iris; crisp fine detail; no motion blur, no depth-of-field blur, "
    "no artistic bokeh.",
]

# Ideal-capture requirements — appended only for an image that does NOT carry the
# matching non-ideal condition, so the prompt never contradicts itself.
_IDEAL_GAZE = (
    "The eye looks straight at the camera with a neutral, forward gaze; the iris is a "
    "full, front-on circle (not foreshortened)."
)
_IDEAL_EXPOSURE = (
    "The eye is open wide with the iris fully exposed; eyelids and lashes frame the eye "
    "but do not cover any part of the iris."
)
_IDEAL_SPECULAR = (
    "Exactly one small, controlled specular reflection from the illuminator, on the pupil "
    "or its edge — not lying over the iris and not washing out any texture."
)
# realism key -> the ideal requirement it suppresses.
_IDEAL_BY_REALISM: dict[str, str] = {
    "gaze": _IDEAL_GAZE,
    "occlusion": _IDEAL_EXPOSURE,
    "glasses": _IDEAL_SPECULAR,
}

# Negatives that always apply.
IRIS_NEGATIVES_ALWAYS: list[str] = [
    "Any text, letters, words, numbers, captions, labels, watermarks, logos, "
    "timestamps or writing of any kind, anywhere in the image",
    "Measurement scales, rulers, grids, reticles, crosshairs, corner brackets, "
    "bounding boxes, arrows or annotation marks",
    "Any user interface, HUD, on-screen display, device screen, dashboard, or "
    "scanner overlay",
    "Colour of any kind",
    "Colour photograph, RGB image, colour tint, sepia, or blue cast",
    "False-colour or thermal colour-mapping",
    "More than one eye, both eyes, or any part of the face beyond the immediate eye "
    "(no nose, mouth, or brow)",
    "The iris rendered as a flat, dark, or black featureless disc with no visible texture",
    "A glowing, neon, CGI, or galaxy/sci-fi stylized eye",
    "Cartoon, illustration, painting, 3D render, or any non-photographic look",
    "Blur, motion blur, an out-of-focus iris, or artistic background bokeh",
]

# Negatives dropped for an image that DELIBERATELY carries that condition (a realism
# knob put it in); added back for every image that does not.
_NEG_BY_REALISM: dict[str, str] = {
    "lens": "Contact lenses of any kind (clear, coloured, cosmetic or costume) or contact-lens edge glare",
    "glasses": "Glasses or spectacles, or any lens glare, reflection or refraction over the eye",
    "makeup": "Eye makeup, eyeliner, mascara or eyeshadow",
    "condition": "Visible eye disease, cloudiness, growths, redness, or pupil irregularity",
}

# Display order for the per-image "capture conditions" section.
_REALISM_ORDER = ("gaze", "occlusion", "lens", "makeup", "glasses", "condition")

_IRIS_VARIATION_GUIDANCE: dict[int, list[str]] = {
    0: [
        "Strict repeatability: standardized iris-camera setup — identical framing, "
        "illuminator position and pupil size. Minimize variation.",
    ],
    1: [
        "Low variation: keep the same framing and illumination.",
        "Permit only very subtle differences in pupil size and iris texture.",
    ],
    2: [
        "Moderate natural variation within a standardized iris-capture setup:",
        "- subtle pupil-size variation",
        "- natural variation in crypt and furrow detail",
        "- slight variation in the illuminator highlight position",
        "- small differences in eyelid openness and lash coverage",
    ],
    3: [
        "High natural variation while strictly preserving every iris-capture rule above:",
        "- noticeable pupil-size variation (still round and centred)",
        "- varied, realistic crypt/furrow/collarette detail",
        "- varied illuminator highlight position",
        "- varied eyelid openness and lash coverage — the full iris stays unclipped",
        "- keep the eye front-facing and the iris sharply in focus",
    ],
}


def iris_variation_guidance(level: int) -> list[str]:
    """Return the permitted-variation lines for an IR iris capture (level 0–3)."""
    level = max(0, min(3, int(level)))
    return list(_IRIS_VARIATION_GUIDANCE[level])


def _build_iris_prompt(options: PromptOptions) -> str:
    """Render the prompt for one monochrome near-infrared iris capture."""
    lines: list[str] = []
    lines.append(
        "Generate a single monochrome (grayscale) near-infrared photograph of one human "
        "eye belonging to a synthetic person who does not exist."
    )
    lines.append("")
    lines.append(
        "It must look exactly like a real frame from a near-infrared iris camera under "
        "~850 nm illumination: a plain, sharp, completely colourless close-up of just one "
        "eye. It is a photograph — NOT a colour image, NOT a stylized or CGI render, NOT a "
        "screen or device readout — and it contains NO text or graphics of any kind."
    )
    lines.append("")

    # Non-ideal conditions this specific image carries (empty = pristine capture).
    realism = options.iris.active_realism() if options.iris is not None else {}

    # Hard requirements: invariants + the ideal-capture rules NOT overridden by a
    # condition on this image.
    lines.append("Hard requirements:")
    for req in IRIS_INVARIANTS:
        lines.append(f"- {req}")
    for key, ideal in _IDEAL_BY_REALISM.items():
        if key not in realism:
            lines.append(f"- {ideal}")
    for extra in options.extra_positive_constraints:
        if extra.strip():
            lines.append(f"- {extra.strip()}")
    lines.append("")

    # Demographic target — the SAME three axes as the face modality. Iris structure
    # tracks ancestry; pupil size and corneal arcus track age. In near-infrared the
    # tone is mid-grey regardless of visible eye colour (melanin is NIR-transparent).
    lines.append(
        "Demographic target (match the iris structure and proportions typical of this "
        "person; the near-infrared tone stays mid-grey regardless of eye colour):"
    )
    lines.append(f"- Age: {options.age_bucket}")
    lines.append(f"- Gender presentation: {options.gender_bucket}")
    lines.append(f"- Apparent ancestry: {options.ethnicity_bucket}")
    lines.append("")

    # Distinct iris — the per-image diversity layer (mirrors the face path).
    if options.iris is not None:
        lines.append(
            "Distinct iris — render ONE specific, unique iris with exactly these "
            "features (do not blend toward a generic average iris):"
        )
        for line in options.iris.to_prompt_lines():
            lines.append(f"- {line}")
        lines.append("")

    # Non-ideal capture conditions for THIS image — only present when a realism knob
    # was enabled AND sampled in. These deliberately override the ideal rules above.
    if realism:
        lines.append(
            "Capture conditions for THIS image (render these exactly; where they "
            "conflict with the ideal-capture rules above, THESE take priority):"
        )
        for key in _REALISM_ORDER:
            if key in realism:
                lines.append(f"- {realism[key]}")
        lines.append("")

    lines.append(f"Permitted natural variation (variation level {options.variation_level}):")
    for g in iris_variation_guidance(options.variation_level):
        lines.append(g if g.startswith("-") else f"- {g}")
    if options.seed is not None:
        lines.append(f"- Deterministic seed in effect: {options.seed}.")
    lines.append("")

    # Negatives: the always-on set, plus a ban on anything this image does NOT carry.
    negatives = list(IRIS_NEGATIVES_ALWAYS)
    for key, neg in _NEG_BY_REALISM.items():
        if key not in realism:
            negatives.append(neg)
    for extra in options.extra_negative_constraints:
        if extra.strip():
            negatives.append(extra.strip())
    lines.append("Do not include:")
    for neg in negatives:
        lines.append(f"- {neg}")

    return "\n".join(lines).strip()
