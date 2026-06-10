"""End-to-end tests for the orchestration engine (:class:`Generator` / run_batch).

Every test passes an in-test provider (or the 'mock' provider) so there is no
coupling to real credentials or paid APIs.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from app.core.config import AppConfig
from app.core.generator import Generator, RunNotConfirmedError, run_batch
from app.core.models import (
    BatchGenerationRequest,
    DistributionMode,
    ItemStatus,
    ProviderResult,
    RunStatus,
)
from app.core.providers.base import ImageProvider, ProviderError
from app.core.providers.mock_provider import MockProvider

PNG_SIGNATURE = b"\x89PNG"


class PartialFailProvider(ImageProvider):
    """Fails on every Nth call (1-based), succeeds otherwise. Deterministic."""

    provider_name = "partialfail"
    requires_api_key = False

    def __init__(self, *, fail_every: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_every = fail_every
        self.calls = 0

    async def generate(
        self, *, prompt: str, size: str, model_id: str, seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        self.calls += 1
        if self.calls % self.fail_every == 0:
            raise ProviderError("deterministic partial failure")
        return ProviderResult(image_bytes=PNG_SIGNATURE + b"-img", actual_cost_usd=0.0)


def _request(
    tmp_path,
    *,
    total_count: int = 4,
    provider: str = "mock",
    model_id: str = "mock-image",
    retry_failed: bool = False,
) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider=provider,
        model_id=model_id,
        age_buckets=["adult, 26 to 40", "young adult, 18 to 25"],
        gender_buckets=["female-presenting", "male-presenting"],
        ethnicity_buckets=["East Asian"],
        distribution_mode=DistributionMode.EVEN,
        total_count=total_count,
        size="1024x1024",
        output_dir=str(tmp_path / "run_out"),
        retry_failed=retry_failed,
        max_retries=0,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_full_happy_path_with_mock(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    request = _request(tmp_path, total_count=4)

    run = gen.create_run(request)
    assert run.status == RunStatus.PLANNED
    assert run.total == 4

    gen.confirm(run)
    run = await gen.execute(run, provider=MockProvider())

    assert run.status == RunStatus.COMPLETED
    assert run.success_count == run.total == 4
    assert run.failure_count == 0

    # Every planned image exists on disk.
    for item in run.plan:
        img = run.images_dir / item.filename
        assert img.exists()
        assert img.read_bytes().startswith(PNG_SIGNATURE)

    # Metadata artifacts exist in the run output dir.
    assert (run.output_dir / "metadata.jsonl").exists()
    assert (run.output_dir / "metadata.csv").exists()
    manifest_path = run.output_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["succeeded"] == 4

    # JSONL has one line per item.
    jsonl_lines = [
        ln
        for ln in (run.output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    assert len(jsonl_lines) == 4


# --------------------------------------------------------------------------- #
# Confirmation gate
# --------------------------------------------------------------------------- #
async def test_execute_without_confirmation_raises(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_request(tmp_path, total_count=2))
    # Not confirmed -> execution must be refused.
    with pytest.raises(RunNotConfirmedError):
        await gen.execute(run, provider=MockProvider())


async def test_run_batch_requires_auto_confirm(tmp_path):
    config = AppConfig.load()
    with pytest.raises(RunNotConfirmedError):
        await run_batch(
            config,
            _request(tmp_path, total_count=2),
            auto_confirm=False,
            provider=MockProvider(),
        )


async def test_run_batch_with_auto_confirm_completes(tmp_path):
    config = AppConfig.load()
    run = await run_batch(
        config,
        _request(tmp_path, total_count=3),
        auto_confirm=True,
        provider=MockProvider(),
    )
    assert run.status == RunStatus.COMPLETED
    assert run.success_count == 3


# --------------------------------------------------------------------------- #
# Failures don't crash the batch
# --------------------------------------------------------------------------- #
async def test_partial_failures_do_not_crash_batch(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    # retry_failed False + max_retries 0 -> one attempt per item; some fail.
    request = _request(tmp_path, total_count=4, retry_failed=False)
    run = gen.create_run(request)
    gen.confirm(run)

    # concurrency defaults to 1, so calls are sequential and deterministic:
    # items 2 and 4 fail (every 2nd call).
    provider = PartialFailProvider(fail_every=2)
    run = await gen.execute(run, provider=provider)

    assert run.status == RunStatus.COMPLETED
    assert run.success_count + run.failure_count == run.total == 4
    assert run.failure_count == 2
    assert run.success_count == 2

    failed = [r for r in run.results if r.status == ItemStatus.FAILED]
    assert len(failed) == 2
    for r in failed:
        assert r.error
        assert r.filename is None  # failed items have no saved image


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
async def test_actual_cost_is_numeric_with_mock(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_request(tmp_path, total_count=3))
    gen.confirm(run)
    run = await gen.execute(run, provider=MockProvider())

    assert isinstance(run.actual_cost_usd, (int, float))
    assert run.actual_cost_usd >= 0.0


def test_estimated_cost_uses_registry_price(tmp_path):
    """With a registry-priced model, the estimate is count * per-image price."""
    config = AppConfig.load()
    gen = Generator(config)
    count = 5
    # openai/gpt-image-1 is priced per quality in the registry (medium by default).
    request = BatchGenerationRequest(
        provider="openai",
        model_id="gpt-image-1",
        age_buckets=["adult, 26 to 40"],
        gender_buckets=["female-presenting"],
        ethnicity_buckets=["East Asian"],
        distribution_mode=DistributionMode.EVEN,
        total_count=count,
        size="1024x1024",
        output_dir=str(tmp_path / "run_cost"),
    )
    run = gen.create_run(request)  # planning only — no spend, no network

    price = config.pricing.per_image_price("openai", "gpt-image-1")
    assert price is not None
    assert run.estimate.pricing_available is True
    assert run.estimate.price_per_image_usd == price
    assert run.estimate.estimated_total_usd == pytest.approx(count * price)
