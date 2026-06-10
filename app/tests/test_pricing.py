"""Tests for :mod:`app.core.pricing`.

Covers cost estimation against a known model, the two ways pricing can be
missing (model absent from the registry, or present but priced ``null``), and
the :meth:`PricingService.per_image_price` lookup helper. A small inline
registry keeps the assertions independent of the bundled defaults.
"""

from __future__ import annotations

import pytest

from app.core.config import load_model_registry
from app.core.models import BatchGenerationRequest, DistributionMode
from app.core.pricing import PricingService, UnknownModelError


def _registry() -> dict:
    """A minimal registry: one priced model and one explicitly unpriced one."""
    return {
        "providers": {
            "acme": {
                "priced-model": {
                    "display_name": "Acme Priced",
                    "supports_size": ["1024x1024"],
                    "default_size": "1024x1024",
                    "price_per_image_usd": 0.02,
                    "pricing_source": "acme test fixture",
                    "reports_actual_cost": False,
                },
                "unpriced-model": {
                    "display_name": "Acme Unpriced",
                    "supports_size": ["1024x1024"],
                    "default_size": "1024x1024",
                    "price_per_image_usd": None,
                    "pricing_source": "acme test fixture",
                    "reports_actual_cost": False,
                },
            }
        }
    }


def _request(provider: str, model_id: str, *, count: int = 5) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider=provider,
        model_id=model_id,
        age_buckets=["adult, 26 to 40"],
        gender_buckets=["female-presenting"],
        ethnicity_buckets=["White European"],
        distribution_mode=DistributionMode.EVEN,
        total_count=count,
    )


def test_estimate_known_model_multiplies_price_by_count() -> None:
    svc = PricingService(_registry())
    count = 5
    est = svc.estimate(_request("acme", "priced-model", count=count))

    assert est.pricing_available is True
    assert est.warning is None
    assert est.price_per_image_usd == 0.02
    assert est.total_count == count
    assert est.estimated_total_usd == pytest.approx(0.02 * count)


def test_estimate_unknown_model_flags_unavailable_with_warning() -> None:
    svc = PricingService(_registry())
    est = svc.estimate(_request("acme", "does-not-exist"))

    assert est.pricing_available is False
    assert est.estimated_total_usd is None
    assert est.warning and est.warning.strip()
    assert "does-not-exist" in est.warning


def test_estimate_null_price_flags_unavailable_with_warning() -> None:
    svc = PricingService(_registry())
    est = svc.estimate(_request("acme", "unpriced-model"))

    assert est.pricing_available is False
    assert est.price_per_image_usd is None
    assert est.estimated_total_usd is None
    assert est.warning and est.warning.strip()


def test_per_image_price_known_and_unknown() -> None:
    svc = PricingService(_registry())

    assert svc.per_image_price("acme", "priced-model") == 0.02
    # Unknown model (and unknown provider) both resolve to None, no exception.
    assert svc.per_image_price("acme", "does-not-exist") is None
    assert svc.per_image_price("nobody", "whatever") is None
    # A model that exists but is unpriced also yields None.
    assert svc.per_image_price("acme", "unpriced-model") is None


def test_get_model_info_unknown_raises() -> None:
    svc = PricingService(_registry())
    with pytest.raises(UnknownModelError):
        svc.get_model_info("acme", "does-not-exist")


def test_bundled_registry_loads_and_estimates() -> None:
    """Sanity-check against the real bundled registry via load_model_registry."""
    svc = PricingService(load_model_registry())
    # The mock provider is always present in the shipped registry.
    assert svc.has_model("mock", "mock-image")
    est = svc.estimate(_request("mock", "mock-image", count=3))
    assert est.pricing_available is True
    assert est.estimated_total_usd == pytest.approx(0.0)
