"""Scene choices must produce coherent prompts and survive saved preferences."""

from __future__ import annotations

import json
import random

import pytest

from app.core.appearance import sample_appearance
from app.core.capture import BACKGROUND_CHOICES, CaptureOptions, resolve_capture
from app.core.models import PromptOptions
from app.core.prompt_builder import build_prompt
from app.tui.prefs import Prefs, load, save


def _options(**capture) -> PromptOptions:
    appearance = sample_appearance(
        random.Random(1), age_bucket="adult, 26 to 40", gender_bucket="female-presenting",
        ethnicity_bucket="South Asian",
    ).model_copy(update={"lighting": "frontal studio lighting", "expression": "a neutral expression"})
    return PromptOptions(
        age_bucket="adult, 26 to 40", gender_bucket="female-presenting", ethnicity_bucket="South Asian",
        appearance=appearance, size="2368x3344", head_area_pct=25,
        capture=CaptureOptions(style_preset="auto", lighting_preset="auto", **capture),
    )


@pytest.mark.parametrize("key", [key for _, key in BACKGROUND_CHOICES if key != "custom"])
def test_background_presets_match_lighting_and_style_without_studio_leakage(key):
    prompt = build_prompt(_options(background_preset=key))
    if key in ("white", "neutral"):
        assert "Passport-style / studio portrait." in prompt
    else:
        assert "studio" not in prompt.lower()
        assert "passport" not in prompt.lower()
        assert "environmental portrait" in prompt
        assert "physically present in this environment" in prompt
    if key == "white":
        assert "Plain, evenly lit white background." in prompt


def _resolve(capture, index=0, seed=None):
    return resolve_capture(
        capture, background="custom room", image_style="custom photo style",
        lighting="custom light", expression="neutral", index=index, seed=seed,
    )


def test_varied_natural_backgrounds_are_balanced_and_reproducible():
    capture = CaptureOptions(
        background_preset="natural", background_variation="varied", lighting_preset="auto",
    )
    scenes = [_resolve(capture, i, 100) for i in range(8)]
    assert len({s.background for s in scenes}) == 8
    assert sum("outdoor" in s.lighting for s in scenes) == 4
    assert scenes == [_resolve(capture, i, 100) for i in range(8)]


def test_fixed_background_reuses_description_across_items_and_seeds():
    capture = CaptureOptions(background_preset="indoor", background_variation="fixed")
    assert len({_resolve(capture, i, i * 31).background for i in range(10)}) == 1


def test_custom_scene_and_details_do_not_replace_custom_style_or_lighting():
    capture = CaptureOptions(background_detail="red brick walls, no posters")
    result = _resolve(capture)
    assert result.background == "custom room; red brick walls, no posters"
    assert result.image_style == "custom photo style"
    assert result.lighting == "custom light"


def test_auto_lighting_does_not_assume_custom_background_is_indoors():
    result = _resolve(CaptureOptions(lighting_preset="auto"))
    assert "described environment" in result.lighting
    assert "indoor" not in result.lighting


def test_phone_capture_supports_headroom_and_clear_background_independently():
    options = _options(background_preset="plants", focus_preset="clear", headroom_pct=25)
    options.capture.style_preset = "phone"
    prompt = build_prompt(options)
    assert "front-camera smartphone selfie" in prompt
    assert "approximately 25% of image height below the top edge" in prompt
    assert "25% of the TOTAL IMAGE AREA" in prompt
    assert "Useful size guide on this 2368×3344 canvas" in prompt
    assert "head horizontally centered" in prompt
    assert "normal depth of field" in prompt
    assert "studio" not in prompt.lower()
    options.capture.focus_preset = "blurred"
    blurred = build_prompt(options)
    assert "strong background blur" in blurred
    assert "Portrait-mode background blur" not in blurred


def test_explicit_expression_does_not_keep_sampled_neutral_expression():
    prompt = build_prompt(_options(background_preset="home", expression_preset="smile"))
    assert "slight, natural closed-mouth smile" in prompt
    assert "Expression: a neutral expression" not in prompt
    assert "within a neutral expression" not in prompt


def test_varied_scenes_use_the_base_seed_once_when_planned():
    from app.core.batch_planner import plan_batch
    from app.core.models import BatchGenerationRequest

    req = BatchGenerationRequest(
        provider="mock", model_id="mock-image", total_count=8, seed=100,
        age_buckets=["adult, 26 to 40"], gender_buckets=["female-presenting"],
        ethnicity_buckets=["South Asian"],
        capture=CaptureOptions(background_preset="natural", background_variation="varied", lighting_preset="auto"),
    )
    prompts = [build_prompt(item.prompt_options) for item in plan_batch(req)]
    assert sum("available natural daylight consistent with the outdoor setting" in p for p in prompts) == 4


def test_capture_and_custom_text_preferences_round_trip(tmp_path):
    path = tmp_path / "prefs.json"
    original = Prefs(
        size="2368x3344", head_area_pct=20,
        capture=CaptureOptions(background_preset="plants", style_preset="phone", headroom_pct=25).model_dump(),
        background="my fixed room", lighting="my ambient lighting", image_style="my capture style",
        expression="my expression", extra_positive="first rule\nsecond rule", extra_negative="one exclusion",
    )
    assert save(original, path)
    assert load(path) == original


def test_old_and_invalid_capture_preferences_load_safely(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"head_height_pct": 75}))
    assert load(path).capture["background_preset"] == "neutral"
    path.write_text(json.dumps({"capture": {"background_preset": "missing", "headroom_pct": "bad"}}))
    assert load(path).capture == Prefs().capture
