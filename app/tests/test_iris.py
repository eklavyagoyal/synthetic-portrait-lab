"""Tests for the IR_IRIS capture modality.

These pin the IR counterpart of the appearance layer: iris sampling is
deterministic and demographically gated, signatures are unique (so the planner's
uniqueness guarantee carries over), the prompt is a monochrome single-eye NIR
capture, the output is forced to true grayscale, and — critically — the default
RGB face path is entirely unaffected.
"""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image

from app.core.batch_planner import plan_batch
from app.core.buckets import AGE_BUCKETS, ETHNICITY_BUCKETS, GENDER_BUCKETS
from app.core.generator import Generator
from app.core.iris import (
    IrisAppearance,
    IrisRealismOptions,
    sample_iris_appearance,
    to_grayscale,
)
from app.core.models import (
    BatchGenerationRequest,
    CaptureModality,
    DistributionMode,
    GenerationResult,
    ModelInfo,
    PromptOptions,
)
from app.core.prompt_builder import build_prompt
from app.core.sizes import resolve_iris_capture_size

YOUNG = "young adult, 18 to 25"
OLDER = "older adult, 61 to 80"
ADULT = "adult, 26 to 40"
MALE = "male-presenting"
FEMALE = "female-presenting"


def _iris(seed: int, *, age=ADULT, ethnicity="White European", gender=FEMALE) -> IrisAppearance:
    return sample_iris_appearance(
        random.Random(seed),
        age_bucket=age,
        gender_bucket=gender,
        ethnicity_bucket=ethnicity,
    )


def _color_png(color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Determinism + uniqueness
# --------------------------------------------------------------------------- #
def test_same_rng_seed_reproduces_identical_iris() -> None:
    a = _iris(42)
    b = _iris(42)
    assert a == b
    assert a.signature() == b.signature()


def test_redraw_from_same_rng_yields_distinct_irises() -> None:
    rng = random.Random(1)
    sigs = {
        sample_iris_appearance(
            rng, age_bucket=ADULT, gender_bucket=FEMALE, ethnicity_bucket="White European"
        ).signature()
        for _ in range(50)
    }
    assert len(sigs) >= 49


def test_exact_age_within_bucket_bounds() -> None:
    for seed in range(200):
        assert 18 <= _iris(seed, age=YOUNG).exact_age <= 25


# --------------------------------------------------------------------------- #
# Demographic plausibility gating
# --------------------------------------------------------------------------- #
def test_young_iris_never_has_corneal_arcus() -> None:
    for seed in range(200):
        assert _iris(seed, age=YOUNG).arcus == "none"


def test_older_iris_can_have_corneal_arcus() -> None:
    assert any(_iris(seed, age=OLDER).arcus != "none" for seed in range(300))


def test_heavily_pigmented_ancestry_never_light_iris() -> None:
    # East Asian irises are heavily pigmented — never lightly pigmented.
    for seed in range(200):
        assert not _iris(seed, age=ADULT, ethnicity="East Asian").pigmentation.startswith("lightly")


def test_light_iris_possible_for_white_european() -> None:
    assert any(
        _iris(seed, age=ADULT, ethnicity="White European").pigmentation.startswith("lightly")
        for seed in range(200)
    )


# --------------------------------------------------------------------------- #
# Prompt rendering — monochrome, single eye, demographic axes preserved
# --------------------------------------------------------------------------- #
def _iris_options(**overrides) -> PromptOptions:
    base = dict(
        age_bucket=ADULT,
        gender_bucket=FEMALE,
        ethnicity_bucket="South Asian",
        modality=CaptureModality.IR_IRIS,
    )
    base.update(overrides)
    return PromptOptions(**base)


def test_iris_prompt_is_monochrome_single_eye_nir() -> None:
    iris = _iris(5, age=ADULT, ethnicity="South Asian")
    prompt = build_prompt(_iris_options(iris=iris))
    low = prompt.lower()
    assert "near-infrared" in low
    assert "monochrome" in low
    assert "one human eye" in low
    assert "Colour of any kind" in prompt  # negative constraint
    # Same three demographic axes as RGB.
    assert ADULT in prompt and FEMALE in prompt and "South Asian" in prompt
    # The distinct-iris layer is rendered.
    assert f"{iris.exact_age} years old" in prompt
    assert iris.eye_side in prompt
    # It must NOT be the face scaffold.
    assert "passport" not in low
    assert "1:1 square image." not in prompt
    assert "Full head visible" not in prompt


def test_iris_prompt_bans_rendered_text_and_ui() -> None:
    # The user's core worry: no gibberish text / scanner UI baked into the image.
    prompt = build_prompt(_iris_options(iris=_iris(5)))
    low = prompt.lower()
    assert "no text or graphics" in low          # stated up front
    for token in ("letters", "words", "watermarks", "reticles", "hud"):
        assert token in low, token


def test_iris_prompt_enforces_nir_tonality_and_anatomy() -> None:
    # NIR realism: a dark iris must NOT render as a black featureless disc, and the
    # anatomy (zones + collarette) must be spelled out.
    prompt = build_prompt(_iris_options(iris=_iris(5, ethnicity="East Asian")))
    low = prompt.lower()
    assert "melanin is transparent" in low
    assert "featureless disc" in low            # named as a thing to avoid
    assert "pupillary zone" in low and "ciliary zone" in low
    assert "collarette" in low


def test_iris_prompt_omits_distinct_section_without_iris() -> None:
    prompt = build_prompt(_iris_options())  # no iris descriptor
    assert "Distinct iris" not in prompt
    assert "monochrome" in prompt.lower()  # scaffold still present


# --------------------------------------------------------------------------- #
# Grayscale output guarantee
# --------------------------------------------------------------------------- #
def test_to_grayscale_flattens_to_single_channel() -> None:
    out = to_grayscale(_color_png())
    with Image.open(io.BytesIO(out)) as g:
        assert g.mode == "L"          # single luminance channel
        assert g.format == "PNG"      # encoding preserved


def test_generator_postprocess_grayscales_ir_and_passes_rgb_through() -> None:
    color = _color_png()
    ir_out = Generator._postprocess(CaptureModality.IR_IRIS, color)
    with Image.open(io.BytesIO(ir_out)) as g:
        assert g.mode == "L"
    # RGB is a pure passthrough — the bytes are untouched.
    assert Generator._postprocess(CaptureModality.RGB_FACE, color) == color


# --------------------------------------------------------------------------- #
# Planner-level: IR uses the iris field + keeps the uniqueness guarantee
# --------------------------------------------------------------------------- #
def _iris_request(total: int, *, seed: int | None = None) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        modality=CaptureModality.IR_IRIS,
        age_buckets=list(AGE_BUCKETS),
        gender_buckets=list(GENDER_BUCKETS),
        ethnicity_buckets=list(ETHNICITY_BUCKETS),
        distribution_mode=DistributionMode.RANDOM,
        total_count=total,
        seed=seed,
    )


def test_planner_ir_sets_iris_not_appearance() -> None:
    items = plan_batch(_iris_request(40, seed=7))
    assert all(it.prompt_options.modality == CaptureModality.IR_IRIS for it in items)
    assert all(it.prompt_options.iris is not None for it in items)
    assert all(it.prompt_options.appearance is None for it in items)


def test_planner_ir_produces_zero_repeated_signatures() -> None:
    items = plan_batch(_iris_request(400, seed=7))
    sigs = [it.prompt_options.iris.signature() for it in items]
    assert len(sigs) == 400
    assert len(set(sigs)) == 400


def test_planner_ir_dedupes_against_history() -> None:
    first = plan_batch(_iris_request(200, seed=7))
    history = {it.prompt_options.iris.signature() for it in first}
    second = plan_batch(_iris_request(200, seed=7), seen_signatures=history)
    second_sigs = {it.prompt_options.iris.signature() for it in second}
    assert history.isdisjoint(second_sigs)


# --------------------------------------------------------------------------- #
# Modality is recorded + defaults to RGB (dataset stays self-describing)
# --------------------------------------------------------------------------- #
def test_modality_defaults_to_rgb() -> None:
    assert PromptOptions(age_bucket=ADULT, gender_bucket=FEMALE, ethnicity_bucket="South Asian").modality \
        == CaptureModality.RGB_FACE
    assert BatchGenerationRequest(
        provider="mock", model_id="mock-image",
        age_buckets=[ADULT], gender_buckets=[FEMALE], ethnicity_buckets=["South Asian"],
        total_count=1,
    ).modality == CaptureModality.RGB_FACE


def test_generation_result_record_includes_modality() -> None:
    rec = GenerationResult(
        id="x", provider="p", model="m", modality="ir",
        age_bucket="a", gender_bucket="g", ethnicity_bucket="e",
        variation_level=0, size="1024x1024",
    ).to_record()
    assert rec["modality"] == "ir"


# --------------------------------------------------------------------------- #
# Optional realism / non-ideal capture conditions (opt-in knobs)
# --------------------------------------------------------------------------- #
def _sample(seed: int, *, realism=None) -> IrisAppearance:
    return sample_iris_appearance(
        random.Random(seed), age_bucket=ADULT, gender_bucket=FEMALE,
        ethnicity_bucket="South Asian", realism=realism,
    )


def test_realism_off_by_default_is_pristine() -> None:
    # No realism arg, and an all-off options object, both leave the iris clean.
    for seed in range(40):
        assert _sample(seed).active_realism() == {}
        assert _sample(seed, realism=IrisRealismOptions()).active_realism() == {}


def test_enabled_knob_produces_a_realistic_mix_not_all_or_nothing() -> None:
    realism = IrisRealismOptions(contact_lenses=True)
    lensed = sum(_sample(s, realism=realism).lens is not None for s in range(200))
    assert 0 < lensed < 200  # some clean, some lensed


def test_only_enabled_knobs_are_ever_sampled() -> None:
    realism = IrisRealismOptions(contact_lenses=True)  # lenses only
    saw_lens = False
    for s in range(200):
        ap = _sample(s, realism=realism)
        saw_lens = saw_lens or ap.lens is not None
        assert ap.occlusion is None and ap.gaze is None and ap.condition is None
        assert ap.glasses is None and ap.makeup is None
    assert saw_lens


def test_realism_adds_signature_entropy_and_stays_deterministic() -> None:
    base = _sample(1)
    lensed = base.model_copy(update={"lens": "a cosmetic coloured contact lens."})
    assert base.signature() != lensed.signature()          # entropy added
    assert lensed.signature() == lensed.model_copy().signature()  # still deterministic


def test_prompt_drops_ban_and_describes_an_active_condition() -> None:
    clean = _sample(2)
    lensed = clean.model_copy(
        update={"lens": "a cosmetic coloured contact lens with a printed limbal ring."}
    )
    clean_p = build_prompt(_iris_options(iris=clean))
    lensed_p = build_prompt(_iris_options(iris=lensed))
    # A clean image still bans lenses; the lensed image lifts the ban and describes it.
    assert "Contact lenses of any kind" in clean_p
    assert "Contact lenses of any kind" not in lensed_p
    assert "Capture conditions for THIS image" in lensed_p
    assert "cosmetic coloured contact lens with a printed limbal ring" in lensed_p


def test_prompt_relaxes_ideal_gaze_when_off_gaze_present() -> None:
    clean = _sample(3)
    gazed = clean.model_copy(update={"gaze": "the eye gazes slightly to the left."})
    assert "looks straight at the camera" in build_prompt(_iris_options(iris=clean))
    assert "looks straight at the camera" not in build_prompt(_iris_options(iris=gazed))


def test_makeup_intensity_is_weighted_subtle_not_dominated_by_strong() -> None:
    # Regression guard: makeup used to be a flat moderate/strong split and came out
    # ~80% heavy. It must now skew subtle, with strong a clear minority.
    realism = IrisRealismOptions(eye_makeup=True)
    makeups = [m for s in range(600) if (m := _sample(s, realism=realism).makeup) is not None]
    assert len(makeups) > 100
    strong = sum(m.startswith(("strong eye makeup", "heavy dramatic")) for m in makeups)
    subtle = sum(m.startswith(("very subtle", "light, natural", "just a little")) for m in makeups)
    assert strong / len(makeups) < 0.30      # heavy makeup is a minority
    assert subtle / len(makeups) > 0.45      # most are understated
    assert len(set(makeups)) >= 5            # genuine variety, not one string


@pytest.mark.parametrize(
    "field,opts",
    [
        ("occlusion", IrisRealismOptions(eyelid_occlusion=True)),
        ("gaze", IrisRealismOptions(off_gaze=True)),
        ("lens", IrisRealismOptions(contact_lenses=True)),
        ("condition", IrisRealismOptions(ocular_conditions=True)),
        ("glasses", IrisRealismOptions(glasses=True)),
        ("makeup", IrisRealismOptions(eye_makeup=True)),
    ],
)
def test_each_knob_produces_varied_values(field, opts) -> None:
    values = {getattr(_sample(s, realism=opts), field) for s in range(400)}
    values.discard(None)
    assert len(values) >= 3, (field, values)


# --------------------------------------------------------------------------- #
# Iris capture geometry — 4:3 landscape (ISO/IEC 19794-6), not square
# --------------------------------------------------------------------------- #
def _gpt_image_2() -> ModelInfo:
    return ModelInfo(
        provider="openai", model_id="gpt-image-2", display_name="GPT Image 2",
        supports_size=["1024x1024", "1024x1536", "1536x1024"],
        default_size="1024x1024", price_per_image_usd=0.07,
    )


def test_iris_size_prefers_high_res_4x3_for_gpt_image_2() -> None:
    size = resolve_iris_capture_size("openai", "gpt-image-2", _gpt_image_2())
    assert size == "2048x1536"
    w, h = (int(x) for x in size.split("x"))
    assert w > h and abs((w / h) - 4 / 3) < 1e-6   # genuine 4:3 landscape


def test_iris_size_falls_back_to_closest_landscape_preset() -> None:
    info = ModelInfo(
        provider="openai", model_id="gpt-image-1", display_name="GPT Image 1",
        supports_size=["1024x1024", "1024x1536", "1536x1024"],
        default_size="1024x1024", price_per_image_usd=0.07,
    )
    # Only 1536x1024 is landscape among the presets.
    assert resolve_iris_capture_size("openai", "gpt-image-1", info) == "1536x1024"
