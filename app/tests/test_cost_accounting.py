"""Cost-accounting regression tests.

These pin the distinction the engine must keep (and previously collapsed):

* successful outputs vs **billable API attempts** (retries are extra attempts);
* a pre-run estimate vs an attempt-based **burn** estimate;
* a locally-computed estimate vs a **provider-reported** spend (only real when
  the provider actually returns one);
* provider token **usage** is captured, not discarded.

Everything runs against injected in-test providers — no network, no spend.
"""

from __future__ import annotations

from typing import Optional

import pytest

from app.core.config import AppConfig
from app.core.generator import Generator
from app.core.models import (
    BatchGenerationRequest,
    DistributionMode,
    ProviderResult,
)
from app.core.providers.base import ImageProvider, ProviderError

PNG = b"\x89PNG-img"


class CostProvider(ImageProvider):
    """Always succeeds; optionally reports a per-image cost and a usage object."""

    provider_name = "costfake"
    requires_api_key = False

    def __init__(self, *, cost: Optional[float] = None, usage: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.cost = cost
        self.usage = usage
        self.calls = 0

    async def generate(self, *, prompt, size, model_id, seed=None, quality=None) -> ProviderResult:
        self.calls += 1
        return ProviderResult(image_bytes=PNG, actual_cost_usd=self.cost, usage=self.usage)


class FailOnceProvider(ImageProvider):
    """Raises on the very first call, then always succeeds — exactly one retry
    across a sequential (concurrency=1) run."""

    provider_name = "failonce"
    requires_api_key = False

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    async def generate(self, *, prompt, size, model_id, seed=None, quality=None) -> ProviderResult:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("transient first-call failure")
        return ProviderResult(image_bytes=PNG, actual_cost_usd=None)


def _request(tmp_path, *, total=8, provider="openai", model_id="gpt-image-1",
             retry_failed=False, max_retries=0) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider=provider,
        model_id=model_id,
        age_buckets=["adult, 26 to 40", "young adult, 18 to 25"],
        gender_buckets=["female-presenting", "male-presenting"],
        ethnicity_buckets=["East Asian"],
        distribution_mode=DistributionMode.EVEN,
        total_count=total,
        size="1024x1024",
        output_dir=str(tmp_path / "run_out"),
        retry_failed=retry_failed,
        max_retries=max_retries,
        concurrency=1,
    )


def _price(config, quality="medium"):
    return config.pricing.per_image_price("openai", "gpt-image-1", quality)


# Case A: 8 planned, 8 successful, no retries -> cost = 8 × per-attempt estimate.
async def test_case_a_no_retries(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_request(tmp_path, total=8))
    gen.confirm(run)
    run = await gen.execute(run, provider=CostProvider())

    price = _price(config)
    assert run.successful_outputs == 8
    assert run.failed_outputs == 0
    assert run.total_retries == 0
    assert run.api_attempts == 8
    assert run.estimated_cost_before_run_usd == pytest.approx(8 * price)
    assert run.estimated_cost_from_attempts_usd == pytest.approx(8 * price)


# Case B: 8 planned, one image retries once -> 9 attempts, 8 successes.
# Burn must include all 9 attempts, not only the 8 successful outputs.
async def test_case_b_retry_is_a_billable_attempt(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_request(tmp_path, total=8, retry_failed=True, max_retries=2))
    gen.confirm(run)
    provider = FailOnceProvider()
    run = await gen.execute(run, provider=provider)

    price = _price(config)
    assert provider.calls == 9                      # 8 + 1 retry actually hit the API
    assert run.successful_outputs == 8
    assert run.failed_outputs == 0
    assert run.total_retries == 1
    assert run.api_attempts == 9
    assert run.estimated_cost_before_run_usd == pytest.approx(8 * price)   # planned
    assert run.estimated_cost_from_attempts_usd == pytest.approx(9 * price)  # burn incl. retry


# Case C: provider returns usage/cost -> actual cost comes from the provider.
async def test_case_c_uses_provider_reported_cost_and_usage(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    # mock-image is the one registry model with reports_actual_cost=true.
    run = gen.create_run(_request(tmp_path, total=2, provider="mock", model_id="mock-image"))
    gen.confirm(run)
    usage = {"input_tokens": 10, "output_tokens": 100, "total_tokens": 110}
    run = await gen.execute(run, provider=CostProvider(cost=0.05, usage=usage))

    assert run.provider_reported_cost_usd == pytest.approx(0.10)   # 2 × 0.05, real
    assert run.has_estimated_costs is False                        # not an estimate
    assert run.provider_usage["total_tokens"] == 220              # 2 × 110, aggregated


# Case D: provider reports no billing/usage -> number must be labelled an estimate,
# and provider-reported spend is None (not a fake "actual").
async def test_case_d_no_billing_is_estimate_only(tmp_path):
    config = AppConfig.load()
    gen = Generator(config)
    run = gen.create_run(_request(tmp_path, total=2, provider="openai", model_id="gpt-image-1"))
    gen.confirm(run)
    run = await gen.execute(run, provider=CostProvider(cost=None))

    assert run.provider_reported_cost_usd is None
    assert run.has_estimated_costs is True
    assert run.estimated_cost_from_attempts_usd == pytest.approx(2 * _price(config))

    summary = run.manifest()["summary"]
    assert summary["api_attempts"] == 2
    assert summary["successful_outputs"] == 2
    assert summary["provider_reported_cost_usd"] is None
    assert summary["estimated_cost_from_attempts_usd"] == pytest.approx(2 * _price(config))
