"""New image models must work with existing portrait and output settings."""

import pytest

from app.cli.generate import _build_parser, _build_request
from app.core.config import AppConfig
from app.core.generator import Generator
from app.core.models import BatchGenerationRequest
from app.core.sizes import (
    accepts_custom_sizes, resolve_a4_portrait_size, resolve_iris_capture_size,
    validate_request_size,
)
from app.tui.prefs import Prefs, load, save

MODELS = ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]


def _request(model_id, **kwargs):
    return BatchGenerationRequest(
        provider="openai", model_id=model_id, total_count=1,
        age_buckets=["adult, 26 to 40"], gender_buckets=["female-presenting"],
        ethnicity_buckets=["South Asian"], **kwargs,
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_new_models_keep_custom_sizes_and_report_unknown_cost(model_id):
    config = AppConfig.load()
    info = config.pricing.get_model_info("openai", model_id)
    assert not config.pricing.has_model("openai", "gpt-image-1")
    assert accepts_custom_sizes("openai", model_id)
    for size in info.supports_size + ["2368x3344"]:
        validate_request_size(size, info)
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_request_size("1920x1080", info)
    assert resolve_a4_portrait_size("openai", model_id, info) == "2368x3344"
    assert resolve_iris_capture_size("openai", model_id, info) == "2048x1536"

    run = Generator(config).create_run(_request(
        model_id, size="2368x3344", transparent_background=True,
    ))
    assert run.request.size == "2368x3344"
    assert run.plan[0].prompt_options.transparent_background
    assert not run.estimate.pricing_available
    assert run.estimate.estimated_total_usd is None
    assert run.estimate.price_per_image_usd is None
    assert not info.reports_actual_cost


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_cli_and_planner_accept_extended_quality(model_id, quality):
    config = AppConfig.load()
    args = _build_parser(config).parse_args([
        "--provider", "openai", "--model", model_id,
        "--count", "1", "--quality", quality, "--transparent-background",
    ])
    request = _build_request(args, config)
    assert request.quality == quality
    run = Generator(config).create_run(request)
    assert run.request.quality == quality
    assert run.request.transparent_background


@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_extended_quality_is_rejected_for_older_models_before_spend(quality):
    with pytest.raises(ValueError, match="not supported"):
        Generator(AppConfig.load()).create_run(_request("gpt-image-2", quality=quality))


@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_extended_quality_and_transparency_survive_saved_preferences(tmp_path, quality):
    path = tmp_path / "prefs.json"
    before = Prefs(
        model_key="openai::gpt-image-2.5-sunburst", quality=quality,
        transparent_background=True,
    )
    assert save(before, path)
    assert load(path) == before
    assert save(Prefs(model_key="openai::gpt-image-1", transparent_background=True), path)
    migrated = load(path)
    assert migrated.model_key == "openai::gpt-image-2.5-flare"
    assert migrated.transparent_background
