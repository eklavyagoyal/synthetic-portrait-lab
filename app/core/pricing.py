"""Cost estimation and (where possible) actual-cost tracking.

Prices live in the model registry, never in code as unquestioned truth. When a
model has no price, the estimate is explicitly flagged as unavailable so the
front-ends can require deliberate confirmation before spending.
"""

from __future__ import annotations

from typing import Optional

from .models import BatchGenerationRequest, CostEstimate, ModelInfo


class UnknownModelError(KeyError):
    """Raised when a provider/model pair is absent from the registry."""


class PricingService:
    """Resolves models and computes cost estimates from a model registry dict.

    The registry shape is::

        {"providers": {"<provider>": {"<model_id>": {<ModelInfo fields>}}}}
    """

    def __init__(self, registry: dict):
        self._providers: dict = (registry or {}).get("providers", {})

    # -- model resolution ------------------------------------------------- #
    def has_model(self, provider: str, model_id: str) -> bool:
        return model_id in self._providers.get(provider, {})

    def get_model_info(self, provider: str, model_id: str) -> ModelInfo:
        try:
            entry = self._providers[provider][model_id]
        except KeyError as exc:
            raise UnknownModelError(
                f"Model {provider}/{model_id} is not in the registry."
            ) from exc
        return ModelInfo(provider=provider, model_id=model_id, **entry)

    def list_models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for provider, models in self._providers.items():
            for model_id, entry in models.items():
                out.append(ModelInfo(provider=provider, model_id=model_id, **entry))
        return out

    def per_image_price(
        self, provider: str, model_id: str, quality: Optional[str] = None
    ) -> Optional[float]:
        if not self.has_model(provider, model_id):
            return None
        return self.get_model_info(provider, model_id).effective_price(quality)

    # -- estimation ------------------------------------------------------- #
    def estimate(self, request: BatchGenerationRequest) -> CostEstimate:
        """Pre-flight estimate for a request. Always returns a CostEstimate; the
        ``pricing_available`` flag tells the caller whether the figure is real."""
        count = request.total_count
        if not self.has_model(request.provider, request.model_id):
            return CostEstimate(
                provider=request.provider,
                model_id=request.model_id,
                total_count=count,
                pricing_available=False,
                warning=(
                    f"Model {request.provider}/{request.model_id} is not in the pricing "
                    "registry. Cost cannot be estimated — explicit confirmation required."
                ),
            )

        info = self.get_model_info(request.provider, request.model_id)
        price = info.effective_price(request.quality)
        if price is None:
            return CostEstimate(
                provider=request.provider,
                model_id=request.model_id,
                total_count=count,
                quality=request.quality,
                price_per_image_usd=None,
                pricing_available=False,
                pricing_source=info.pricing_source,
                warning=(
                    f"No price configured for {request.provider}/{request.model_id}. "
                    "Cost cannot be estimated — explicit confirmation required."
                ),
            )

        total = round(price * count, 6)
        return CostEstimate(
            provider=request.provider,
            model_id=request.model_id,
            total_count=count,
            quality=request.quality,
            price_per_image_usd=price,
            estimated_total_usd=total,
            pricing_available=True,
            pricing_source=info.pricing_source,
            warning=None,
        )

    def estimated_item_cost(
        self, provider: str, model_id: str, quality: Optional[str] = None
    ) -> Optional[float]:
        """Per-image estimate used to seed an item's ``estimated_cost_usd``."""
        return self.per_image_price(provider, model_id, quality)
