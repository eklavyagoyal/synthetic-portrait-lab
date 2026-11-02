"""Transparent output travels from settings through API requests and saved PNGs."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.cli.generate import _build_parser, _build_request
from app.core.batch_planner import plan_batch
from app.core.capture import CaptureOptions
from app.core.config import AppConfig
from app.core.generator import Generator
from app.core.models import BatchGenerationRequest
from app.core.prompt_builder import build_prompt
from app.core.providers.mock_provider import MockProvider
from app.core.providers.openai_provider import OpenAIProvider
from app.tui.prefs import Prefs, load, save


def _request(**overrides):
    return BatchGenerationRequest(**({
        "provider": "mock", "model_id": "mock-image", "total_count": 1,
        "age_buckets": ["adult, 26 to 40"], "gender_buckets": ["female-presenting"],
        "ethnicity_buckets": ["South Asian"], "size": "512x512",
        "transparent_background": True,
    } | overrides))


@pytest.mark.parametrize("style", ["auto", "environmental", "phone"])
def test_transparent_prompt_overrides_scene_and_keeps_portrait_settings(style):
    req = _request(
        size="1024x1536", quality="high", head_area_pct=20,
        capture=CaptureOptions(
            background_preset="plants", background_variation="varied",
            background_detail="red wallpaper", focus_preset="blurred",
            style_preset=style, lighting_preset="window", expression_preset="smile",
            headroom_pct=25,
        ),
    )
    opts = plan_batch(req)[0].prompt_options
    prompt = build_prompt(opts)
    assert opts.transparent_background
    assert "PNG alpha channel" in prompt
    assert "dracaena" not in prompt
    assert "red wallpaper" not in prompt
    assert "strong background blur" not in prompt
    assert "physically present in this environment" not in prompt
    assert "real everyday setting" not in prompt
    assert "natural window light" in prompt
    assert "slight, natural closed-mouth smile" in prompt
    assert "20% of the TOTAL IMAGE AREA" in prompt
    assert "approximately 25% of image height" in prompt
    assert opts.size == "1024x1536"
    if style == "phone":
        assert "front-camera smartphone selfie" in prompt


@pytest.mark.parametrize("overrides, message", [
    ({"provider": "google", "model_id": "nano-banana"}, "requires an OpenAI"),
    ({"provider": "openai", "model_id": "dall-e-3"}, "requires an OpenAI"),
    ({"modality": "ir"}, "RGB portraits only"),
])
def test_unsupported_transparency_is_rejected_before_generation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


@pytest.mark.parametrize("transparent", [False, True])
async def test_openai_payload_requests_native_transparency(transparent):
    expected = await MockProvider().generate(
        prompt="portrait", size="512x512", model_id="mock-image", transparent_background=True,
    )
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(expected.image_bytes).decode()}],
        })

    provider = OpenAIProvider(api_key="test-key")
    provider._client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1", transport=httpx.MockTransport(respond),
    )
    try:
        result = await provider.generate(
            prompt="portrait", size="1024x1536", model_id="gpt-image-1",
            quality="high", transparent_background=transparent,
        )
    finally:
        await provider.aclose()
    payload = payloads[0]
    assert payload["quality"] == "high"
    assert payload["size"] == "1024x1536"
    if transparent:
        assert payload["background"] == "transparent"
        assert payload["output_format"] == "png"
    else:
        assert "background" not in payload
        assert "output_format" not in payload
    assert result.image_bytes == expected.image_bytes


async def test_saved_png_preserves_alpha_and_records_settings(tmp_path):
    gen = Generator(AppConfig.load())
    req = _request(output_dir=str(tmp_path / "run"))
    run = gen.create_run(req)
    gen.confirm(run)
    run = await gen.execute(run, provider=MockProvider())
    assert run.success_count == 1
    result = run.results[0]
    with Image.open(run.output_dir / "images" / result.filename) as img:
        assert img.format == "PNG"
        assert img.mode == "RGBA"
        assert img.getchannel("A").getextrema() == (0, 255)
    assert result.to_record()["transparent_background"] is True
    manifest = json.loads((run.output_dir / "manifest.json").read_text())
    assert manifest["request"]["transparent_background"] is True
    record = json.loads((run.output_dir / "metadata.jsonl").read_text())
    assert record["transparent_background"] is True


def test_transparency_preferences_and_cli(tmp_path):
    path = tmp_path / "prefs.json"
    assert save(Prefs(transparent_background=True), path)
    assert load(path).transparent_background is True
    cfg = AppConfig.load()
    args = _build_parser(cfg).parse_args(["--count", "1", "--transparent-background"])
    assert _build_request(args, cfg).transparent_background is True
