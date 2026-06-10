"""Tests for the providers layer: the MockProvider contract, the registry, and
the engine's retry / auth-failure handling driven through in-test providers.

No real network call is ever made — only MockProvider and tiny in-test
:class:`ImageProvider` subclasses are used, always passed via
``Generator.execute(run, provider=...)``.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import AppConfig
from app.core.generator import Generator
from app.core.models import (
    BatchGenerationRequest,
    DistributionMode,
    ItemStatus,
    ProviderResult,
    RunStatus,
)
from app.core.providers import registry
from app.core.providers.base import (
    ImageProvider,
    ProviderAuthError,
    ProviderError,
)
from app.core.providers.mock_provider import MockProvider

PNG_SIGNATURE = b"\x89PNG"


# --------------------------------------------------------------------------- #
# In-test providers
# --------------------------------------------------------------------------- #
class FlakyProvider(ImageProvider):
    """Raises ProviderError for the first ``fail_times`` attempts, then succeeds."""

    provider_name = "flaky"
    requires_api_key = False

    def __init__(self, *, fail_times: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.attempts = 0

    async def generate(
        self, *, prompt: str, size: str, model_id: str, seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ProviderError(f"transient failure #{self.attempts}")
        return ProviderResult(image_bytes=PNG_SIGNATURE + b"-ok", actual_cost_usd=0.0)


class AuthFailProvider(ImageProvider):
    """ensure_ready passes, but generate raises ProviderAuthError every time."""

    provider_name = "authfail"
    requires_api_key = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attempts = 0

    async def generate(
        self, *, prompt: str, size: str, model_id: str, seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        self.attempts += 1
        raise ProviderAuthError("invalid credentials")


def _single_item_request(tmp_path, *, max_retries: int = 3) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=["adult, 26 to 40"],
        gender_buckets=["female-presenting"],
        ethnicity_buckets=["East Asian"],
        distribution_mode=DistributionMode.EVEN,
        total_count=1,
        size="1024x1024",
        output_dir=str(tmp_path / "run"),
        retry_failed=True,
        max_retries=max_retries,
    )


# --------------------------------------------------------------------------- #
# MockProvider contract
# --------------------------------------------------------------------------- #
async def test_mock_provider_returns_valid_png_no_network():
    result = await MockProvider().generate(
        prompt="hello", size="256x256", model_id="mock-image", seed=42
    )
    assert isinstance(result, ProviderResult)
    assert result.image_bytes  # non-empty
    assert result.image_bytes.startswith(PNG_SIGNATURE)


async def test_mock_provider_is_deterministic():
    kwargs = dict(prompt="same prompt", size="512x512", model_id="mock-image", seed=99)
    first = await MockProvider().generate(**kwargs)
    second = await MockProvider().generate(**kwargs)
    assert first.image_bytes == second.image_bytes

    # A different seed yields different bytes (so determinism isn't trivially "always equal").
    other = await MockProvider().generate(
        prompt="same prompt", size="512x512", model_id="mock-image", seed=100
    )
    assert other.image_bytes != first.image_bytes


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_builds_and_lists_mock():
    assert "mock" in registry.available_providers()
    provider = registry.build_provider("mock")
    assert isinstance(provider, MockProvider)
    assert provider.provider_name == "mock"


# --------------------------------------------------------------------------- #
# Retry / auth behaviour through the engine
# --------------------------------------------------------------------------- #
async def test_flaky_provider_retries_until_success(tmp_path, monkeypatch):
    # Avoid real backoff sleeps slowing the test.
    async def _no_sleep(_seconds):  # noqa: ANN001
        return None

    monkeypatch.setattr("app.core.generator.asyncio.sleep", _no_sleep)

    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_single_item_request(tmp_path, max_retries=3))
    gen.confirm(run)

    provider = FlakyProvider(fail_times=2)
    run = await gen.execute(run, provider=provider)

    assert run.status == RunStatus.COMPLETED
    assert run.success_count == 1
    assert run.failure_count == 0
    result = run.results[0]
    assert result.status == ItemStatus.SUCCESS
    # Two failures then success on the third attempt -> retries == 2.
    assert result.retries == 2
    assert provider.attempts == 3


async def test_auth_error_is_not_retried(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_single_item_request(tmp_path, max_retries=3))
    gen.confirm(run)

    provider = AuthFailProvider()
    run = await gen.execute(run, provider=provider)

    assert run.status == RunStatus.COMPLETED  # the run finishes; the item fails fast
    assert run.success_count == 0
    assert run.failure_count == 1

    result = run.results[0]
    assert result.status == ItemStatus.FAILED
    assert "auth error" in (result.error or "")
    # Auth errors are not worth retrying: generate is called exactly once.
    assert provider.attempts == 1
