"""Per-image iris variation for the near-infrared (IR) capture modality.

This is the IR counterpart of :mod:`app.core.appearance`. Where that module
samples a whole face, this one samples a single **iris** as it appears in a
near-infrared (NIR, ~850 nm) iris-recognition capture: a monochrome, high-detail
close-up in which melanin is largely transparent, so even heavily pigmented eyes
reveal rich stromal texture (crypts, furrows, the collarette).

The design deliberately mirrors :class:`~app.core.appearance.Appearance`:

* the same three demographic axes drive it (age, gender, ancestry) — iris
  pigmentation and texture track ancestry, and pupil size / arcus track age;
* :meth:`IrisAppearance.signature` is a stable dedup key, so the planner's
  rejection-sampling and cross-run uniqueness machinery work unchanged;
* :meth:`IrisAppearance.to_prompt_lines` renders the per-image bullet text.

Nothing here decides colour: the prompt asks for a monochrome NIR image and
:func:`to_grayscale` enforces true single-channel output as a backstop.
"""

from __future__ import annotations

import hashlib
import io
import random
from typing import Optional

from PIL import Image
from pydantic import BaseModel

# Shared demographic helpers — reused verbatim so the two modalities stay in
# lock-step on age parsing and sub-ancestry (DRY: one source of truth).
from .appearance import (
    ETHNICITY_SUBGROUPS,  # noqa: F401  (kept for parity / external reference)
    _is_older,
    _is_young,
    _sub_ancestry,
    parse_age_range,
)

# --------------------------------------------------------------------------- #
# Iris pigmentation by ancestry (visible-light colour). This drives the NIR
# rendering: in near-infrared the stroma is always resolved in grayscale, but
# the underlying pigmentation still sets the overall tone and texture contrast.
# --------------------------------------------------------------------------- #
_PIGMENTATION: dict[str, list[str]] = {
    "East Asian": ["heavily pigmented, very dark brown", "heavily pigmented, dark brown"],
    "South Asian": [
        "heavily pigmented, very dark brown",
        "heavily pigmented, dark brown",
        "moderately pigmented, medium brown",
    ],
    "Southeast Asian": ["heavily pigmented, very dark brown", "heavily pigmented, dark brown"],
    "Black African descent": ["heavily pigmented, very dark brown", "heavily pigmented, dark brown"],
    "Middle Eastern or North African": [
        "heavily pigmented, dark brown",
        "moderately pigmented, medium brown",
        "moderately pigmented, light brown or hazel",
        "lightly pigmented, green or grey",
    ],
    "Latino or Hispanic": [
        "heavily pigmented, dark brown",
        "moderately pigmented, medium brown",
        "moderately pigmented, light brown or hazel",
    ],
    "White European": [
        "moderately pigmented, medium brown",
        "moderately pigmented, light brown or hazel",
        "lightly pigmented, blue",
        "lightly pigmented, grey",
        "lightly pigmented, green",
    ],
    "mixed heritage": [
        "heavily pigmented, dark brown",
        "moderately pigmented, medium brown",
        "moderately pigmented, hazel",
        "lightly pigmented, blue or green",
    ],
}

# The texture fields — these carry most of the per-iris uniqueness.
_CRYPT_DENSITY = [
    "sparse, shallow crypts",
    "a moderate scattering of crypts",
    "numerous well-defined Fuchs' crypts",
    "dense, deep crypts across the stroma",
]
_FURROWS = [
    "faint contraction furrows",
    "a few concentric contraction furrows",
    "prominent concentric contraction furrows",
    "deep radial and concentric furrows",
]
_COLLARETTE = [
    "a smooth, subtle collarette",
    "a clearly defined circular collarette",
    "an irregular, slightly zig-zag collarette",
    "a raised, prominent collarette",
]

# Age-gated: the limbal ring fades and the pupil tends to constrict with age.
_LIMBAL_YOUNG = [
    "a dark, sharply defined limbal ring",
    "a distinct limbal ring",
    "a moderately defined limbal ring",
]
_LIMBAL_OLD = [
    "a soft, faded limbal ring",
    "a faint limbal ring",
    "a moderately defined limbal ring",
]
_PUPIL_YOUNG = [
    "a mid-sized round pupil",
    "a moderately dilated round pupil",
    "a slightly constricted round pupil",
]
_PUPIL_OLD = [
    "a small, constricted round pupil",
    "a small round pupil (senile miosis)",
    "a mid-sized round pupil",
]

# "none" weighted so most irises are unmarked; the rest add quiet realism.
_PIGMENT_FEATURE = ["none"] * 5 + [
    "a small iris nevus (dark freckle) in one sector",
    "a few scattered iris freckles",
    "a subtle pigment spot near the collarette",
    "a small dark crypt cluster in one quadrant",
]
# Wolfflin–Krukmann nodules occur almost exclusively in lightly pigmented irises.
_WOLFFLIN = "faint Wolfflin nodules near the outer rim"

# Corneal arcus is an age marker — only sampled for older subjects.
_ARCUS_OLD = [
    "none",
    "none",
    "a faint corneal arcus (pale ring at the outer iris edge)",
    "a subtle arcus senilis at the periphery",
]

_SPECULAR = [
    "a single small specular highlight at the 11 o'clock position on the cornea",
    "a single small specular highlight at the 1 o'clock position on the cornea",
    "a single small specular highlight just above the pupil",
    "a single small specular highlight at the upper-left of the pupil",
]
_EYELASH = [
    "long, dense eyelashes framing the eye",
    "moderate eyelashes",
    "short, sparse eyelashes",
]
_EYE_SIDE = ["left eye", "right eye"]


# --------------------------------------------------------------------------- #
# Optional realism / non-ideal capture conditions (opt-in per run)
# --------------------------------------------------------------------------- #
# Each of these is OFF by default. When enabled on a run, the condition is
# sampled into a realistic FRACTION of that run's images (with per-image variety),
# so a batch is a mix of clean and affected captures rather than all-or-nothing.
# The prompt builder relaxes the matching ideal-capture rule and drops the
# matching negative for any image that carries the condition (see prompt_builder).
class IrisRealismOptions(BaseModel):
    """Per-run toggles for non-ideal iris-capture conditions (all default off)."""

    eyelid_occlusion: bool = False   # drooping lids / lashes over the iris
    off_gaze: bool = False           # slight off-axis gaze (iris a mild ellipse)
    contact_lenses: bool = False     # soft / hard / cosmetic / painted lenses
    ocular_conditions: bool = False  # minor eye/iris/sclera pathology
    glasses: bool = False            # spectacles with heavy glare / distortion
    eye_makeup: bool = False         # moderate → strong eye makeup

    def any_enabled(self) -> bool:
        return any(self.model_dump().values())


# Per-image incidence when the corresponding knob is ON (a realistic mix, not 100%).
_REALISM_INCIDENCE: dict[str, float] = {
    "occlusion": 0.60,
    "gaze": 0.55,
    "lens": 0.50,
    "condition": 0.40,
    "glasses": 0.35,
    "makeup": 0.50,
}
_REALISM_FIELDS = ("occlusion", "gaze", "lens", "condition", "glasses", "makeup")

# Each list is graduated from mild to heavy and weighted toward the mild end via
# repetition (as _MARKS/_PIGMENT_FEATURE do), so a batch spans a range of
# intensities instead of clustering at the extreme — generative models tend to
# exaggerate these conditions, so the light tiers are also worded to stay restrained.
_OCCLUSION = (
    [
        "a few eyelashes cross the upper iris and the upper eyelid rests just slightly low; "
        "most of the iris stays clear.",
        "the upper eyelid covers only the top ~10-15% of the iris, with a few lashes over it.",
    ] * 2
    + [
        "partially open lids leave the iris about 80% visible; eyelashes cast fine dark "
        "shadows over the upper rim.",
        "the upper lid covers about the top third of the iris and several eyelashes hang over it.",
    ]
    + [
        "a squinted eye — upper and lower lids narrow the aperture and occlude the top and "
        "bottom edges of the iris.",
        "dense eyelashes sweep across the upper iris, noticeably occluding part of it.",
    ]
)
_GAZE = [
    "the eye gazes slightly to the left, so the iris reads as a mild ellipse and the "
    "pupil sits a little left of centre — the whole iris stays clearly visible.",
    "the eye gazes slightly to the right; the iris is a gentle ellipse with the pupil a "
    "little right of centre, still fully readable.",
    "a slight upward gaze foreshortens the iris a touch; its lower rim shows a little more.",
    "a slight downward gaze; the upper iris shows a little more and the pupil sits marginally low.",
    "a small off-axis gaze — the iris is mildly foreshortened but remains clearly visible and readable.",
]
# Weighted toward the common clear/soft lens; cosmetic and painted (the PAD cases) rarer.
_LENS = (
    [
        "a clear soft contact lens over the eye — a faint circular lens edge is just visible "
        "outside the iris with a small extra glint; the natural iris texture still shows through.",
    ] * 3
    + [
        "a rigid gas-permeable (hard) contact lens, clearly smaller than the iris, on the "
        "cornea with a crisp bright lens-edge ring and a sharp edge reflection; the iris "
        "shows around and through it.",
    ] * 2
    + [
        "a cosmetic coloured contact lens with a printed limbal ring and a machine-regular "
        "dotted pattern sitting on top of and partly hiding the natural iris texture, its "
        "too-perfect circular pattern boundary giving it away.",
    ] * 2
    + [
        "a painted/patterned costume contact lens whose obviously printed, regular pattern "
        "overlays the real iris, contrasting with the irregular natural stroma beneath.",
    ]
)
_CONDITION = [
    "a faint corneal arcus — a pale grey ring just inside the limbus.",
    "a small pterygium — a fleshy wedge encroaching from the inner corner onto the edge of the cornea.",
    "a pinguecula — a small raised light-grey bump on the sclera beside the iris.",
    "an early cataract — the pupil is not fully black but shows a soft cloudy grey haze.",
    "a faint corneal scar/opacity — a small hazy grey patch overlying part of the iris.",
    "mild conjunctival irritation — the sclera looks a little mottled with faint surface "
    "vessels (fine dark lines in near-infrared).",
    "a small subconjunctival haemorrhage — a darker patch over part of the sclera.",
    "a small dark iris naevus — a pigmented spot in one sector of the stroma.",
    "mild pupil irregularity — the pupil is slightly oval rather than perfectly round.",
]
# Graduated from a mild reflection to a fully "cooked" frame, so not every glasses
# image is destroyed.
_GLASSES = [
    "eyeglasses in front of the eye with a moderate lens reflection over one part of the "
    "iris and a visible frame edge; the eye is only mildly distorted through the lens.",
    "eyeglasses with a strong bright lens reflection washing across part of the iris, a "
    "visible frame edge, and slight refraction distorting the eye.",
    "spectacle-lens glare — a large blown-out highlight covers a portion of the iris and "
    "sclera, with a visible frame rim and refraction shifting the eye's apparent position.",
    "thick-rimmed glasses with heavy overlapping reflections and a dark frame partly "
    "obscuring the eye, the lens noticeably distorting the iris outline.",
]
# Weighted heavily toward subtle (~60% subtle / ~27% moderate / ~13% strong). The
# light tiers explicitly tell the model to keep it understated, since it otherwise
# renders any eye makeup as dramatic in a macro eye shot.
_MAKEUP = (
    [
        "very subtle eye makeup — only a thin, faint eyeliner line at the lash base, no "
        "mascara build-up and no eyeshadow; keep it barely noticeable.",
        "light, natural eye makeup — a fine eyeliner line and lightly defined lashes only, "
        "no eyeshadow; subtle, not dramatic.",
        "just a little mascara on otherwise natural lashes and no other eye makeup; very understated.",
    ] * 3
    + [
        "moderate eye makeup — a defined eyeliner line and light mascara reading as clean "
        "dark lines in near-infrared; still restrained, not heavy.",
        "moderate eye makeup — mascara-darkened lashes and a soft hint of shadow around the "
        "socket, mid tones in near-infrared.",
    ] * 2
    + [
        "strong eye makeup — noticeable winged eyeliner and denser mascara with some dark "
        "eyeshadow, bold dark tones in near-infrared.",
        "heavy dramatic eye makeup — thick winged liner, dense clumped mascara and dark "
        "kohl-rimmed lids (near-infrared-opaque, strong black lines).",
    ]
)


# --------------------------------------------------------------------------- #
# The sampled iris
# --------------------------------------------------------------------------- #
class IrisAppearance(BaseModel):
    """One sampled, specific iris layered on top of a demographic triple.

    Mirrors :class:`~app.core.appearance.Appearance`'s public shape
    (``exact_age`` + :meth:`signature` + :meth:`to_prompt_lines` + ``model_dump``)
    so the planner, generator and metadata treat both modalities uniformly.
    """

    exact_age: int
    sub_ancestry: str
    eye_side: str
    pigmentation: str
    crypt_density: str
    furrows: str
    collarette: str
    limbal_ring: str
    pupil: str
    pigment_feature: str
    arcus: str
    specular: str
    eyelash: str
    # Optional non-ideal capture conditions (None unless the matching run knob is on
    # AND this image was sampled to carry it). See IrisRealismOptions.
    occlusion: Optional[str] = None
    gaze: Optional[str] = None
    lens: Optional[str] = None
    condition: Optional[str] = None
    glasses: Optional[str] = None
    makeup: Optional[str] = None

    def active_realism(self) -> dict[str, str]:
        """The non-ideal conditions carried by THIS image (field -> description)."""
        return {f: v for f in _REALISM_FIELDS if (v := getattr(self, f)) is not None}

    def signature(self) -> str:
        """Stable hash over every rendered field — the dedup key (see Appearance).

        Realism fields are appended only when present, so a clean iris hashes
        identically to one produced before realism existed (backward compatible),
        while each non-ideal condition adds entropy.
        """
        parts = [
            self.exact_age, self.sub_ancestry, self.eye_side, self.pigmentation,
            self.crypt_density, self.furrows, self.collarette, self.limbal_ring,
            self.pupil, self.pigment_feature, self.arcus, self.specular, self.eyelash,
        ]
        canonical = "iris|" + "|".join(str(p) for p in parts)
        for key, value in self.active_realism().items():
            canonical += f"|{key}={value}"
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]

    def to_prompt_lines(self) -> list[str]:
        """Human-readable bullet text (no leading dash) for the prompt body."""
        lines = [
            f"Which eye: {self.eye_side}.",
            f"Apparent age of the person: {self.exact_age} years old.",
            f"Ancestry of the person: {self.sub_ancestry}.",
            f"Near-infrared tone: {self._nir_tone()} (the eye is {self.pigmentation} in "
            "visible light, but melanin is transparent in near-infrared, so it images in "
            "grey with its structure fully revealed).",
            "Iris structure: an inner pupillary zone and an outer ciliary zone divided by "
            f"{self.collarette}; {self.crypt_density} near the collarette; and "
            f"{self.furrows} across the stroma; fine trabecular striations throughout.",
            f"Pupil: {self.pupil}, deep black, sharply bounded and centred.",
            f"Outer (limbal) boundary with the sclera: {self.limbal_ring}.",
        ]
        if self.pigment_feature != "none":
            lines.append(
                "Distinguishing structural feature (a small local grayscale tone "
                f"variation): {self.pigment_feature}."
            )
        if self.arcus != "none":
            lines.append(f"Peripheral detail: {self.arcus}.")
        lines.append(f"Illuminator reflection: {self.specular}.")
        lines.append(f"Eyelashes: {self.eyelash}.")
        return lines

    def _nir_tone(self) -> str:
        """Map the visible-light pigmentation to its near-infrared grayscale tone.

        In NIR the iris is never a dark blob — pigment is transparent — so every
        eye images as a grey, textured stroma; pigmentation only nudges the tone
        and contrast, which is what this maps.
        """
        if self.pigmentation.startswith("heavily"):
            return "mid-grey stroma with dense, high-contrast texture"
        if self.pigmentation.startswith("lightly"):
            return "lighter-grey stroma with finer, slightly lower-contrast texture"
        return "medium-grey stroma with well-defined texture"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _maybe(rng: random.Random, enabled: bool, key: str, choices: list[str]) -> Optional[str]:
    """Sample one realism condition when its knob is on and its incidence hits.

    Short-circuits so ``rng`` is untouched when the knob is off — keeping the draw
    sequence (and therefore signatures) stable across differing knob combinations.
    """
    if enabled and rng.random() < _REALISM_INCIDENCE[key]:
        return rng.choice(choices)
    return None


def sample_iris_appearance(
    rng: random.Random,
    *,
    age_bucket: str,
    gender_bucket: str,
    ethnicity_bucket: str,
    realism: Optional[IrisRealismOptions] = None,
) -> IrisAppearance:
    """Draw one fully-specified iris for the given demographic triple.

    Like :func:`~app.core.appearance.sample_appearance`, every draw consumes
    ``rng`` deterministically, so re-sampling with the same RNG advances to a
    *different* iris — exactly what the planner relies on for collision-free
    rejection sampling. ``gender_bucket`` is accepted for interface parity (the
    iris itself is not gendered) so the planner can call both samplers alike.

    ``realism`` (if given) enables optional non-ideal capture conditions, each
    sampled AFTER the base iris so enabling a knob never perturbs the base fields.
    """
    lo, hi = parse_age_range(age_bucket)
    exact_age = rng.randint(lo, hi)
    pigmentation = rng.choice(_PIGMENTATION.get(ethnicity_bucket, _PIGMENTATION["mixed heritage"]))
    is_light = pigmentation.startswith("lightly")

    if _is_young(age_bucket):
        limbal = rng.choice(_LIMBAL_YOUNG)
        pupil = rng.choice(_PUPIL_YOUNG)
        arcus = "none"
    elif _is_older(age_bucket):
        limbal = rng.choice(_LIMBAL_OLD)
        pupil = rng.choice(_PUPIL_OLD)
        arcus = rng.choice(_ARCUS_OLD)
    else:
        limbal = rng.choice(_LIMBAL_YOUNG + _LIMBAL_OLD)
        pupil = rng.choice(_PUPIL_YOUNG + _PUPIL_OLD)
        arcus = "none"

    feature_pool = list(_PIGMENT_FEATURE)
    if is_light:
        feature_pool = feature_pool + [_WOLFFLIN]
    pigment_feature = rng.choice(feature_pool)

    ap = IrisAppearance(
        exact_age=exact_age,
        sub_ancestry=_sub_ancestry(rng, ethnicity_bucket),
        eye_side=rng.choice(_EYE_SIDE),
        pigmentation=pigmentation,
        crypt_density=rng.choice(_CRYPT_DENSITY),
        furrows=rng.choice(_FURROWS),
        collarette=rng.choice(_COLLARETTE),
        limbal_ring=limbal,
        pupil=pupil,
        pigment_feature=pigment_feature,
        arcus=arcus,
        specular=rng.choice(_SPECULAR),
        eyelash=rng.choice(_EYELASH),
    )

    # Optional non-ideal conditions, sampled last so they never shift the base iris.
    if realism is not None:
        ap.occlusion = _maybe(rng, realism.eyelid_occlusion, "occlusion", _OCCLUSION)
        ap.gaze = _maybe(rng, realism.off_gaze, "gaze", _GAZE)
        ap.lens = _maybe(rng, realism.contact_lenses, "lens", _LENS)
        ap.condition = _maybe(rng, realism.ocular_conditions, "condition", _CONDITION)
        ap.glasses = _maybe(rng, realism.glasses, "glasses", _GLASSES)
        ap.makeup = _maybe(rng, realism.eye_makeup, "makeup", _MAKEUP)

    return ap


# --------------------------------------------------------------------------- #
# Output guarantee — true single-channel grayscale
# --------------------------------------------------------------------------- #
def to_grayscale(image_bytes: bytes) -> bytes:
    """Return ``image_bytes`` re-encoded as a true single-channel (mode "L") image.

    NIR captures are monochrome. The prompt asks the model for a monochrome
    image, and this backstop *guarantees* it: whatever the provider returns is
    flattened to one luminance channel, so an IR dataset can never silently
    contain colour. The original encoding (PNG) is preserved.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        fmt = img.format or "PNG"
        gray = img.convert("L")
    buf = io.BytesIO()
    gray.save(buf, format=fmt)
    return buf.getvalue()
