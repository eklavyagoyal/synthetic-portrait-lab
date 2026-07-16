"""Per-image appearance variation — the engine that makes every portrait unique.

The demographic buckets (4 ages × 3 genders × 8 ethnicities = 96 triples) are far
too coarse to differentiate hundreds of images: within one triple every prompt is
otherwise identical, and seedless providers (e.g. gpt-image) then return
look-alike faces. This module samples a rich, orthogonal *appearance* per image —
an exact age, a sub-ancestry, face shape, hair, facial hair, eyes, complexion,
distinguishing marks, and micro-variations of expression and lighting — so the
prompt describes a single, specific, unrepeatable individual. (No eyewear: glasses
are explicitly forbidden, both here and in the prompt's negative constraints.)

The combined space is in the trillions, so collisions are astronomically
unlikely; :func:`Appearance.signature` additionally gives the planner a stable
key to *guarantee* uniqueness via rejection sampling across the whole history.

Nothing here relaxes a hard requirement: expression stays within "neutral", the
head stays front-facing, accessories stay minimal. It only adds detail.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Optional

from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Finer ethnicity — sub-ancestries per configured ethnicity bucket
# --------------------------------------------------------------------------- #
ETHNICITY_SUBGROUPS: dict[str, list[str]] = {
    "East Asian": ["Han Chinese", "Japanese", "Korean", "Mongolian", "Taiwanese"],
    "South Asian": [
        "North Indian", "South Indian", "Bengali", "Punjabi",
        "Pakistani", "Sri Lankan", "Nepali",
    ],
    "Southeast Asian": [
        "Thai", "Vietnamese", "Filipino", "Indonesian",
        "Malay", "Burmese", "Khmer",
    ],
    "Black African descent": [
        "West African", "East African", "Central African", "Southern African",
        "African American", "Afro-Caribbean",
    ],
    "Middle Eastern or North African": [
        "Levantine", "Gulf Arab", "Persian", "Egyptian",
        "Maghrebi", "Turkish", "Kurdish",
    ],
    "Latino or Hispanic": [
        "Mexican", "Central American", "Caribbean Hispanic",
        "Andean", "Southern Cone", "Brazilian",
    ],
    "White European": [
        "Northern European", "Eastern European", "Southern European",
        "Western European", "Mediterranean", "Scandinavian",
    ],
    "mixed heritage": [],  # handled specially: a blend of two base ancestries
}

# Readable ancestry adjectives for composing a "mixed" blend.
_MIXED_BASES: list[str] = [
    "East Asian", "South Asian", "Southeast Asian", "Black African",
    "Middle Eastern", "North African", "Latino", "White European",
]

# Complexion candidates per ethnicity bucket (kept plausible within ancestry).
_COMPLEXION: dict[str, list[str]] = {
    "East Asian": ["fair", "light", "light-medium", "medium"],
    "South Asian": ["light-medium", "medium", "tan", "brown", "deep brown"],
    "Southeast Asian": ["light-medium", "medium", "tan", "brown"],
    "Black African descent": [
        "light brown", "medium brown", "brown", "deep brown", "dark brown", "ebony",
    ],
    "Middle Eastern or North African": ["fair", "light olive", "olive", "tan", "light brown"],
    "Latino or Hispanic": ["fair", "light olive", "olive", "tan", "brown", "deep brown"],
    "White European": ["very fair", "fair", "light", "light olive", "olive"],
    "mixed heritage": ["fair", "light", "olive", "tan", "brown", "deep brown"],
}

# Base (non-grey) hair colours per ethnicity. Age gating adds greys later.
_HAIR_COLOR_BASE: dict[str, list[str]] = {
    "East Asian": ["black", "soft black", "dark brown", "dark brown with subtle highlights"],
    "South Asian": ["black", "soft black", "dark brown", "dark brown with warm undertones"],
    "Southeast Asian": ["black", "soft black", "dark brown", "dark brown with subtle highlights"],
    "Black African descent": ["black", "soft black", "dark brown"],
    "Middle Eastern or North African": [
        "black", "dark brown", "brown", "dark brown with warm undertones",
    ],
    "Latino or Hispanic": ["black", "dark brown", "brown", "chestnut brown", "dark blonde"],
    "White European": [
        "dark brown", "brown", "light brown", "chestnut", "dark blonde",
        "blonde", "light blonde", "auburn", "red", "black",
    ],
    "mixed heritage": [
        "black", "dark brown", "brown", "chestnut", "auburn", "dark blonde",
    ],
}
_GREY_TONES = ["grey", "mostly grey", "salt-and-pepper", "silver-grey", "white"]
_GREYING_LIGHT = ["salt-and-pepper", "dark with some grey", "greying at the temples"]

# Hair texture leanings per ethnicity (lists may repeat to weight common cases).
_HAIR_TEXTURE: dict[str, list[str]] = {
    "East Asian": ["straight", "straight", "slightly wavy"],
    "South Asian": ["straight", "wavy", "wavy", "curly"],
    "Southeast Asian": ["straight", "straight", "wavy"],
    "Black African descent": ["curly", "tightly coiled", "coily", "kinky-curly"],
    "Middle Eastern or North African": ["wavy", "curly", "straight"],
    "Latino or Hispanic": ["straight", "wavy", "curly"],
    "White European": ["straight", "wavy", "curly"],
    "mixed heritage": ["wavy", "curly", "straight", "coily"],
}

_HAIR_LENGTH: dict[str, list[str]] = {
    "female-presenting": ["short", "chin-length", "shoulder-length", "mid-back length", "long"],
    "male-presenting": ["closely cropped", "short", "medium-short", "medium-length", "collar-length"],
    "androgynous or non-binary-presenting": [
        "short", "medium-short", "medium-length", "shoulder-length", "chin-length",
    ],
}

_EYEBROWS = ["softly arched", "straight", "gently rounded", "thick", "fine", "strong and defined"]
_FACE_SHAPE = ["oval", "round", "square", "heart-shaped", "oblong", "diamond-shaped"]
_JAWLINE = [
    "a soft jawline", "a defined jawline", "a strong jawline",
    "a rounded jawline", "a narrow jawline",
]
_NOSE = [
    "a straight nose", "a slightly rounded nose", "an aquiline nose",
    "a broad nose", "a narrow nose", "a small upturned nose",
]
_BUILD = ["a slim face", "an average build", "a fuller face", "a broad face", "a lean face"]

# Eye colour: dark tones are near-universal; lighter tones gated to some ancestries.
_EYE_BASE = ["dark brown", "brown", "brown", "deep brown", "hazel"]
_EYE_LIGHT = ["green", "blue", "grey", "light brown", "amber"]
_EYE_LIGHT_ETHNICITIES = {"White European", "Middle Eastern or North African", "mixed heritage"}

# "none" is repeated to keep most faces unmarked; the rest add quiet realism.
_MARKS = (
    ["none"] * 6
    + [
        "light freckles across the nose and cheeks",
        "a small beauty mark near the lip",
        "a faint mole on one cheek",
        "subtle smile lines",
        "light forehead lines",
        "dimples when relaxed",
        "a small faint scar above one eyebrow",
        "a cleft chin",
        "high, prominent cheekbones",
        "slightly fuller lips",
        "a light scattering of freckles",
    ]
)
_EXPRESSION = [
    "a relaxed, neutral expression",
    "a calm, composed neutral expression",
    "a neutral expression with a faint, barely-there closed-mouth smile",
    "a neutral, attentive expression",
    "a soft, neutral expression",
]
_LIGHTING = [
    "soft, even frontal studio lighting",
    "gentle diffused softbox lighting",
    "balanced high-key studio lighting",
    "soft lighting with a subtle key from one side",
    "clean, shadow-light studio lighting",
]


# --------------------------------------------------------------------------- #
# Age
# --------------------------------------------------------------------------- #
_AGE_RANGE_RE = re.compile(r"(\d+)\s*to\s*(\d+)")


def parse_age_range(age_bucket: str) -> tuple[int, int]:
    """Extract the (low, high) integer age span from a bucket label.

    ``"young adult, 18 to 25"`` -> ``(18, 25)``. Falls back to a sane adult span
    when the label carries no explicit range, so sampling never crashes on a
    custom bucket.
    """
    m = _AGE_RANGE_RE.search(age_bucket)
    if not m:
        return (25, 60)
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo, hi) if lo <= hi else (hi, lo)


def _is_young(age_bucket: str) -> bool:
    lo, hi = parse_age_range(age_bucket)
    return hi <= 40


def _is_older(age_bucket: str) -> bool:
    lo, hi = parse_age_range(age_bucket)
    return lo >= 60


def _is_middle(age_bucket: str) -> bool:
    lo, hi = parse_age_range(age_bucket)
    return 40 <= lo < 60 or (40 < hi <= 60)


# --------------------------------------------------------------------------- #
# The sampled appearance
# --------------------------------------------------------------------------- #
class Appearance(BaseModel):
    """One sampled, specific individual layered on top of a demographic triple."""

    exact_age: int
    sub_ancestry: str
    complexion: str
    face_shape: str
    jawline: str
    nose: str
    build: str
    hair_length: str
    hair_texture: str
    hair_color: str
    hairstyle: str
    facial_hair: Optional[str] = None
    eyebrows: str
    eye_color: str
    distinguishing_mark: str
    expression: str
    lighting: str

    def signature(self) -> str:
        """Stable hash over every rendered field — the dedup key.

        Two appearances that would render an identical prompt share a signature;
        any difference (including a one-year age change) yields a new one. Stored
        verbatim in metadata so future runs can dedup against this one.
        """
        parts = [
            self.exact_age, self.sub_ancestry, self.complexion, self.face_shape,
            self.jawline, self.nose, self.build, self.hair_length, self.hair_texture,
            self.hair_color, self.hairstyle, self.facial_hair or "", self.eyebrows,
            self.eye_color, self.distinguishing_mark,
            self.expression, self.lighting,
        ]
        canonical = "|".join(str(p) for p in parts)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]

    def to_prompt_lines(self) -> list[str]:
        """Human-readable bullet text (no leading dash) for the prompt body."""
        lines = [
            f"Apparent age: {self.exact_age} years old.",
            f"Specific ancestry within the target: {self.sub_ancestry}.",
            f"Skin tone / complexion: {self.complexion}.",
            f"Face: {self.face_shape}, with {self.jawline} and {self.nose}; {self.build}.",
            f"Hair: {self.hair_length}, {self.hair_texture}, {self.hair_color}, {self.hairstyle}.",
        ]
        if self.facial_hair and self.facial_hair != "clean-shaven":
            lines.append(f"Facial hair: {self.facial_hair}.")
        elif self.facial_hair == "clean-shaven":
            lines.append("Facial hair: clean-shaven.")
        lines.append(f"Eyebrows: {self.eyebrows}; eye colour: {self.eye_color}.")
        if self.distinguishing_mark != "none":
            lines.append(f"Distinguishing feature: {self.distinguishing_mark}.")
        lines.append("No glasses or eyewear of any kind.")
        lines.append(f"Expression: {self.expression}.")
        lines.append(f"Lighting on this subject: {self.lighting}.")
        return lines


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _sub_ancestry(rng: random.Random, ethnicity: str) -> str:
    if ethnicity == "mixed heritage":
        a, b = rng.sample(_MIXED_BASES, 2)
        return f"mixed {a} and {b} heritage"
    options = ETHNICITY_SUBGROUPS.get(ethnicity)
    if not options:
        return ethnicity
    return rng.choice(options)


def _hair_color(rng: random.Random, ethnicity: str, age_bucket: str) -> str:
    base = list(_HAIR_COLOR_BASE.get(ethnicity, _HAIR_COLOR_BASE["White European"]))
    if _is_older(age_bucket):
        candidates = _GREY_TONES + base[:2]
    elif _is_middle(age_bucket):
        candidates = base + _GREYING_LIGHT
    else:
        candidates = base
    return rng.choice(candidates)


def _hairstyle(
    rng: random.Random, gender: str, age_bucket: str, length: str, texture: str
) -> str:
    styles = [
        "neatly combed back", "side-parted", "centre-parted", "tousled and natural",
        "loosely swept to one side", "smoothed down", "gently layered",
    ]
    if texture in ("curly", "coily", "tightly coiled", "kinky-curly"):
        styles += [
            "worn as a neat natural afro", "in short natural coils",
            "in shoulder-length locs", "close-cropped natural curls",
        ]
    if gender == "female-presenting" and length in ("shoulder-length", "mid-back length", "long"):
        styles += [
            "worn loose over the shoulders", "tied back in a low ponytail",
            "gathered in a simple bun", "loosely braided",
        ]
    if gender == "male-presenting":
        styles += ["a short textured crop", "a classic short back and sides", "slicked back"]
        if not _is_young(age_bucket):
            styles += ["with a receding hairline", "thinning slightly on top", "close-cropped and balding"]
    return rng.choice(styles)


def _facial_hair(rng: random.Random, gender: str, age_bucket: str) -> Optional[str]:
    if gender == "female-presenting":
        return None
    if gender.startswith("androgynous"):
        return rng.choice(["clean-shaven", "clean-shaven", "light stubble"])
    # male-presenting
    options = [
        "clean-shaven", "clean-shaven", "light stubble", "short stubble",
        "a short trimmed beard", "a neatly trimmed beard", "a goatee",
        "a moustache", "a close-cropped beard",
    ]
    if not _is_young(age_bucket):
        options += ["a full beard", "a greying beard"]
    return rng.choice(options)


def _eye_color(rng: random.Random, ethnicity: str) -> str:
    candidates = list(_EYE_BASE)
    if ethnicity in _EYE_LIGHT_ETHNICITIES:
        candidates = candidates + _EYE_LIGHT
    return rng.choice(candidates)


def sample_appearance(
    rng: random.Random,
    *,
    age_bucket: str,
    gender_bucket: str,
    ethnicity_bucket: str,
) -> Appearance:
    """Draw one fully-specified individual for the given demographic triple.

    Every draw consumes ``rng`` deterministically, so re-sampling with the same
    RNG advances to a *different* individual — exactly what the planner relies on
    for collision-free rejection sampling.
    """
    lo, hi = parse_age_range(age_bucket)
    exact_age = rng.randint(lo, hi)
    length = rng.choice(_HAIR_LENGTH.get(gender_bucket, _HAIR_LENGTH["androgynous or non-binary-presenting"]))
    texture = rng.choice(_HAIR_TEXTURE.get(ethnicity_bucket, _HAIR_TEXTURE["White European"]))

    return Appearance(
        exact_age=exact_age,
        sub_ancestry=_sub_ancestry(rng, ethnicity_bucket),
        complexion=rng.choice(_COMPLEXION.get(ethnicity_bucket, _COMPLEXION["mixed heritage"])),
        face_shape=rng.choice(_FACE_SHAPE),
        jawline=rng.choice(_JAWLINE),
        nose=rng.choice(_NOSE),
        build=rng.choice(_BUILD),
        hair_length=length,
        hair_texture=texture,
        hair_color=_hair_color(rng, ethnicity_bucket, age_bucket),
        hairstyle=_hairstyle(rng, gender_bucket, age_bucket, length, texture),
        facial_hair=_facial_hair(rng, gender_bucket, age_bucket),
        eyebrows=rng.choice(_EYEBROWS),
        eye_color=_eye_color(rng, ethnicity_bucket),
        distinguishing_mark=rng.choice(_MARKS),
        expression=rng.choice(_EXPRESSION),
        lighting=rng.choice(_LIGHTING),
    )
