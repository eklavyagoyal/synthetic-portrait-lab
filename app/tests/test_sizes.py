"""Tests for :mod:`app.core.sizes`.

Covers size parsing, the gpt-image-2 custom-resolution constraints, and the
``validate_request_size`` policy (listed presets always pass; gpt-image-2 also
accepts any constraint-satisfying custom size; everything else is rejected).
"""

from __future__ import annotations

import pytest

from app.core.models import ModelInfo
from app.core.sizes import (
    accepts_custom_sizes,
    gpt_image_2_size_error,
    is_valid_gpt_image_2_size,
    parse_size,
    validate_request_size,
)

# The presets bundled into the registry for gpt-image-2 — every one must be valid.
GPT_IMAGE_2_PRESETS = [
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "1920x1088",
    "1088x1920",
    "3840x2160",
    "2160x3840",
]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1536x1024", (1536, 1024)),
        ("1024X1024", (1024, 1024)),
        (" 1024 x 1024 ", (1024, 1024)),
        ("1024", None),
        ("axb", None),
        ("1024x0", None),
        ("-16x16", None),
        ("", None),
    ],
)
def test_parse_size(text, expected) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("size", GPT_IMAGE_2_PRESETS)
def test_all_bundled_presets_are_valid(size) -> None:
    w, h = parse_size(size)
    assert is_valid_gpt_image_2_size(w, h), gpt_image_2_size_error(w, h)


def test_4k_landscape_sits_exactly_on_the_pixel_ceiling() -> None:
    # 3840x2160 = 8,294,400 px — the documented maximum, inclusive.
    assert is_valid_gpt_image_2_size(3840, 2160)


@pytest.mark.parametrize(
    "w,h,reason_contains",
    [
        (1920, 1080, "multiples of 16"),   # 1080 / 16 = 67.5
        (4096, 1024, "longest edge"),      # 4096 > 3840
        (2048, 512, "3:1"),                # ratio 4:1
        (512, 512, ">="),                  # 262,144 px < 655,360 minimum
        (3840, 3840, "<="),                # 14,745,600 px > maximum
    ],
)
def test_known_bad_sizes_rejected(w, h, reason_contains) -> None:
    reason = gpt_image_2_size_error(w, h)
    assert reason is not None
    assert reason_contains in reason
    assert not is_valid_gpt_image_2_size(w, h)


def test_accepts_custom_sizes_only_for_gpt_image_2() -> None:
    assert accepts_custom_sizes("openai", "gpt-image-2")
    assert not accepts_custom_sizes("openai", "gpt-image-1")
    assert not accepts_custom_sizes("mock", "mock-image")


def _model(provider: str, model_id: str, supports: list[str]) -> ModelInfo:
    return ModelInfo(
        provider=provider,
        model_id=model_id,
        display_name=f"{provider}/{model_id}",
        supports_size=supports,
        default_size=supports[0],
    )


def test_validate_listed_preset_always_passes() -> None:
    info = _model("mock", "mock-image", ["256x256", "1024x1024"])
    validate_request_size("1024x1024", info)  # no raise


def test_validate_custom_valid_size_passes_for_gpt_image_2() -> None:
    info = _model("openai", "gpt-image-2", ["1024x1024"])
    # 2048x1152 is not in supports_size but satisfies the constraints.
    validate_request_size("2048x1152", info)  # no raise


def test_validate_custom_invalid_size_raises_with_reason() -> None:
    info = _model("openai", "gpt-image-2", ["1024x1024"])
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_request_size("1920x1080", info)


def test_validate_malformed_size_raises() -> None:
    info = _model("openai", "gpt-image-2", ["1024x1024"])
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        validate_request_size("definitely-not-a-size", info)


def test_validate_unlisted_size_rejected_for_non_custom_model() -> None:
    info = _model("openai", "gpt-image-1", ["1024x1024", "1024x1536", "1536x1024"])
    # A perfectly valid gpt-image-2 size, but gpt-image-1 only takes its presets.
    with pytest.raises(ValueError, match="not supported"):
        validate_request_size("2048x2048", info)
