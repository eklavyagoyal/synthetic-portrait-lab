"""Shared, auditable portrait scene and camera choices.

Presets resolve into ordinary prompt text. A fixed background reuses its scene
description; it does not promise identical pixels across generated images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

BACKGROUND_CHOICES = [
    ("Neutral grey / off-white", "neutral"), ("Plain white", "white"),
    ("Natural · indoor + outdoor", "natural"), ("Indoor · everyday rooms", "indoor"),
    ("Outdoor · everyday places", "outdoor"), ("Office / workspace", "office"),
    ("Home / living room", "home"), ("Park / garden", "park"),
    ("Street / courtyard", "street"), ("Indoor plants · reference", "plants"),
    ("Cafe", "cafe"), ("Library", "library"), ("Custom description", "custom"),
]
STYLE_CHOICES = [
    ("Auto · match the background", "auto"), ("Studio / ID portrait", "studio"),
    ("Natural environmental portrait", "environmental"),
    ("Casual phone / selfie", "phone"), ("Custom style", "custom"),
]
LIGHTING_CHOICES = [
    ("Auto · match the setting", "auto"), ("Soft studio", "studio"),
    ("Natural daylight", "daylight"), ("Overcast / open shade", "overcast"),
    ("Window light", "window"), ("Indoor ambient", "indoor"),
    ("Mixed window + indoor", "mixed"), ("Custom lighting", "custom"),
]
FOCUS_CHOICES = [
    ("Auto / no extra instruction", "auto"), ("Clear, recognizable background", "clear"),
    ("Gently softened background", "soft"), ("Strong background blur", "blurred"),
]
EXPRESSION_CHOICES = [
    ("Neutral", "neutral"), ("Relaxed / attentive", "relaxed"),
    ("Slight natural smile", "smile"), ("Custom expression", "custom"),
]

# Each scene includes its lighting context so mixed batches never ask for
# indoor ambient light in a park, or for outdoor sun inside a library.
_SCENES: dict[str, tuple[tuple[str, str], ...]] = {
    "neutral": (("plain light gray or off-white background", "studio"),),
    "white": (("plain, evenly lit white background", "studio"),),
    "office": (
        ("an everyday office workspace with desks, shelves and windows", "indoor"),
        ("a shared office lounge with chairs, plants and ordinary room details", "indoor"),
    ),
    "home": (
        ("an ordinary home living room with a sofa, houseplants and a window", "indoor"),
        ("an everyday home kitchen with cabinets and a countertop", "indoor"),
    ),
    "park": (
        ("a real leafy park with trees, grass and a footpath", "outdoor"),
        ("an everyday garden with shrubs, flowers and a fence", "outdoor"),
    ),
    "street": (
        ("a quiet residential street with houses, trees and pavement", "outdoor"),
        ("an ordinary courtyard with brick walls, planters and paving", "outdoor"),
    ),
    "plants": (
        ("a real indoor atrium with a tall dracaena plant close behind the person, "
         "long green leaves filling the background and glimpses of concrete architecture and windows", "indoor"),
        ("an everyday indoor room with a large leafy houseplant immediately behind the person, "
         "dense green leaves and ordinary room details visible between them", "indoor"),
    ),
    "cafe": (("an ordinary quiet cafe with tables, chairs and windows", "indoor"),),
    "library": (("an everyday library reading area with bookshelves and tables", "indoor"),),
}
_SCENES["indoor"] = _SCENES["home"] + _SCENES["office"] + _SCENES["cafe"] + _SCENES["library"]
_SCENES["outdoor"] = _SCENES["park"] + _SCENES["street"]
_SCENES["natural"] = tuple(
    scene for pair in zip(_SCENES["indoor"], _SCENES["outdoor"]) for scene in pair
)


class CaptureOptions(BaseModel):
    """Optional scene presets; custom defaults preserve legacy raw prompt fields."""

    background_preset: Literal[
        "neutral", "white", "natural", "indoor", "outdoor", "office", "home",
        "park", "street", "plants", "cafe", "library", "custom",
    ] = "custom"
    background_variation: Literal["fixed", "varied"] = "fixed"
    background_detail: str = ""
    style_preset: Literal["auto", "studio", "environmental", "phone", "custom"] = "custom"
    lighting_preset: Literal[
        "auto", "studio", "daylight", "overcast", "window", "indoor", "mixed", "custom",
    ] = "custom"
    focus_preset: Literal["auto", "clear", "soft", "blurred"] = "auto"
    expression_preset: Literal["neutral", "relaxed", "smile", "custom"] = "custom"
    headroom_pct: Optional[float] = Field(None, ge=0, le=40)


@dataclass(frozen=True)
class ResolvedCapture:
    background: str
    image_style: str
    lighting: str
    expression: str
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


def resolve_capture(
    capture: CaptureOptions, *, background: str, image_style: str, lighting: str,
    expression: str, index: int = 0, seed: Optional[int] = None,
    transparent_background: bool = False,
) -> ResolvedCapture:
    if capture.background_preset == "custom":
        environment = "studio" if background == "plain light gray or off-white background" else "custom"
    else:
        scenes = _SCENES[capture.background_preset]
        variant = (index + (seed or 0)) if capture.background_variation == "varied" else 0
        background, environment = scenes[variant % len(scenes)]
    if capture.background_detail.strip():
        background += "; " + capture.background_detail.strip()

    if transparent_background:
        background = (
            "fully transparent background (PNG alpha channel), with only the person visible; "
            "preserve fine hair and clean natural edges, with no backdrop, scenery, "
            "checkerboard pattern, floor or cast shadow outside the person"
        )
        environment = "studio"

    style = capture.style_preset
    if transparent_background and style == "environmental":
        style = "studio"
    if style == "auto":
        style = "studio" if environment == "studio" else "environmental"
    image_style = {
        "studio": (
            "photorealistic passport-style studio portrait" if environment == "studio"
            else "photorealistic formally posed ID-style portrait in the requested environment"
        ),
        "environmental": "photorealistic environmental portrait in a real everyday setting",
        "phone": "an ordinary unedited front-camera smartphone selfie, a casual personal snapshot",
    }.get(style, image_style)
    light = capture.lighting_preset
    if light == "auto":
        light = {"studio": "studio", "indoor": "mixed", "outdoor": "daylight"}.get(environment, "scene")
    lighting = {
        "studio": (
            "natural studio lighting" if environment == "studio"
            else "soft, even controlled light on the face, matched to the ambient light in the requested setting"
        ),
        "daylight": "available natural daylight consistent with the outdoor setting",
        "overcast": "soft diffuse daylight from an overcast sky or open shade",
        "window": "natural window light with realistic, gently uneven indoor shadows",
        "indoor": "ordinary available indoor ambient lighting with realistic exposure",
        "mixed": "available indoor ambient light mixed with window daylight, with natural exposure and shadows",
        "scene": "available natural ambient light consistent with the described environment",
    }.get(light, lighting)
    expression = {
        "neutral": "neutral, natural facial expression",
        "relaxed": "a relaxed, attentive natural facial expression",
        "smile": "a slight, natural closed-mouth smile",
    }.get(capture.expression_preset, expression)

    positive: list[str] = []
    negative: list[str] = []
    if environment != "studio":
        positive.append(
            "The person is physically present in this environment, with consistent perspective, "
            "lighting and shadows; no other people are present."
        )
    if style == "phone":
        positive.extend([
            "Use an eye-level front camera at comfortable arm length, with subtle natural wide-angle "
            "phone perspective; the phone itself is outside the frame.",
            "Preserve normal skin texture, ordinary exposure and subtle phone-camera noise; "
            "include a plain everyday top wherever it falls within the requested framing.",
        ])
        negative.extend(["Beauty retouching", "Polished corporate headshot", "Visible phone or camera"])
    if transparent_background:
        negative.extend(["Opaque background", "Painted checkerboard", "Background scenery"])
    elif capture.focus_preset == "clear":
        positive.append("Keep the background clear and recognizable with normal depth of field and visible scene details.")
        negative.extend(["Artificial bokeh", "Portrait-mode background blur"])
    elif capture.focus_preset == "soft":
        positive.append("Use gentle background softening while keeping the surrounding environment recognizable.")
    elif capture.focus_preset == "blurred":
        positive.append("Use a shallow depth of field with strong background blur while keeping the face sharply focused.")
    return ResolvedCapture(background, image_style, lighting, expression, tuple(positive), tuple(negative))


# Starting points only: choosing one never changes the canvas, model or buckets.
CAPTURE_PROFILES = {
    "studio": dict(background_preset="neutral", background_variation="fixed",
                   style_preset="studio", lighting_preset="auto", focus_preset="clear",
                   expression_preset="neutral", headroom_pct=None, head_area_pct=None,
                   head_height_pct=60),
    "natural": dict(background_preset="natural", background_variation="varied",
                    style_preset="environmental", lighting_preset="auto", focus_preset="clear",
                    expression_preset="neutral", headroom_pct=10, head_area_pct=25,
                    head_height_pct=60),
    "phone": dict(background_preset="plants", background_variation="varied",
                  style_preset="phone", lighting_preset="auto", focus_preset="clear",
                  expression_preset="relaxed", headroom_pct=25, head_area_pct=20,
                  head_height_pct=60),
}
