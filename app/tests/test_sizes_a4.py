"""Tests for A4 portrait size resolution."""

from __future__ import annotations

from app.core.models import ModelInfo
from app.core.sizes import resolve_a4_portrait_size


def _gpt_image_2() -> ModelInfo:
    return ModelInfo(
        provider="openai",
        model_id="gpt-image-2",
        display_name="GPT Image 2",
        supports_size=["1024x1024", "1024x1536", "1536x1024"],
        default_size="1024x1024",
        price_per_image_usd=0.07,
    )


def test_resolve_a4_portrait_size_prefers_high_res_custom_for_gpt_image_2() -> None:
    size = resolve_a4_portrait_size("openai", "gpt-image-2", _gpt_image_2())
    assert size == "2368x3344"


def test_resolve_a4_portrait_size_falls_back_to_closest_preset() -> None:
    info = ModelInfo(
        provider="openai",
        model_id="gpt-image-1",
        display_name="GPT Image 1",
        supports_size=["1024x1024", "1024x1536", "1536x1024"],
        default_size="1024x1024",
        price_per_image_usd=0.07,
    )
    assert resolve_a4_portrait_size("openai", "gpt-image-1", info) == "1024x1536"
