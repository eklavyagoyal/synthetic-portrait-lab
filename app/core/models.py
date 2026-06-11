"""Typed data models — the shared contract for the whole engine.

The most important object here is :class:`Run`: a generation run is the central
runtime entity. It owns its request, cost estimate, plan, results, failures and
on-disk locations. Front-ends observe a ``Run``; they never re-implement its logic.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DistributionMode(str, Enum):
    """How a total image count is spread across the selected demographic buckets."""

    EVEN = "even"
    RANDOM = "random"
    WEIGHTED = "weighted"
    EXACT = "exact"


class ItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class RunStatus(str, Enum):
    CREATED = "created"          # request captured, not yet planned
    PLANNED = "planned"          # plan + estimate computed, awaiting confirmation
    CONFIRMED = "confirmed"      # user approved the spend
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"            # the run itself errored out (not individual items)
    CANCELLED = "cancelled"


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    ITEM_STARTED = "item_started"
    ITEM_RETRYING = "item_retrying"
    ITEM_SUCCEEDED = "item_succeeded"
    ITEM_FAILED = "item_failed"
    RUN_COMPLETED = "run_completed"
    RUN_CANCELLED = "run_cancelled"


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Prompt + model description
# --------------------------------------------------------------------------- #
class PromptOptions(BaseModel):
    """Everything the prompt builder needs to render one portrait prompt."""

    age_bucket: str
    gender_bucket: str
    ethnicity_bucket: str
    variation_level: int = Field(0, ge=0, le=3)
    # Framing: head height (top of hair to chin) as a share of the image height.
    head_height_pct: int = Field(60, ge=20, le=90)
    # Canvas size "WxH" — informs the prompt's orientation line. None => square.
    size: Optional[str] = None
    background: str = "plain light gray or off-white background"
    expression: str = "neutral, natural facial expression"
    lighting: str = "natural studio lighting"
    image_style: str = "photorealistic passport-style studio portrait"
    extra_positive_constraints: list[str] = Field(default_factory=list)
    extra_negative_constraints: list[str] = Field(default_factory=list)
    face_crop: bool = False
    seed: Optional[int] = None


class ModelInfo(BaseModel):
    """A single provider/model entry resolved from the registry."""

    provider: str
    model_id: str
    display_name: str
    supports_size: list[str] = Field(default_factory=lambda: ["1024x1024"])
    default_size: str = "1024x1024"
    price_per_image_usd: Optional[float] = None  # flat price / fallback. None => unknown
    # Optional per-quality pricing for token-billed models (e.g. gpt-image:
    # image cost scales with size/quality). Falls back to the flat price above.
    quality_prices: Optional[dict[str, float]] = None
    default_quality: str = "medium"
    supports_quality: list[str] = Field(default_factory=list)
    pricing_source: Optional[str] = None
    reports_actual_cost: bool = False

    def effective_price(self, quality: Optional[str] = None) -> Optional[float]:
        """Per-image price for a quality. Uses the per-quality table when present
        (falling back to the model's default quality), else the flat price."""
        q = quality or self.default_quality
        if self.quality_prices:
            if q in self.quality_prices:
                return self.quality_prices[q]
            if self.default_quality in self.quality_prices:
                return self.quality_prices[self.default_quality]
        return self.price_per_image_usd

    @property
    def pricing_available(self) -> bool:
        return self.effective_price() is not None


# --------------------------------------------------------------------------- #
# Batch request
# --------------------------------------------------------------------------- #
class ExactCount(BaseModel):
    """One row of an EXACT distribution: a concrete bucket triple + count."""

    age_bucket: str
    gender_bucket: str
    ethnicity_bucket: str
    count: int = Field(ge=0)


class BatchGenerationRequest(BaseModel):
    """The complete, validated description of what the user asked to generate.

    This is the single object every front-end produces and hands to the engine.
    """

    provider: str
    model_id: str

    # Demographic selection
    age_buckets: list[str] = Field(default_factory=list)
    gender_buckets: list[str] = Field(default_factory=list)
    ethnicity_buckets: list[str] = Field(default_factory=list)

    # Distribution
    distribution_mode: DistributionMode = DistributionMode.EVEN
    total_count: int = Field(0, ge=0)
    weights: Optional[dict[str, float]] = None       # bucket-token -> weight (WEIGHTED)
    exact_counts: Optional[list[ExactCount]] = None  # explicit per-triple counts (EXACT)

    # Generation knobs
    variation_level: int = Field(0, ge=0, le=3)
    size: str = "1024x1024"
    # Render quality. On token-billed models (gpt-image) this drives the price.
    quality: str = "medium"   # low | medium | high | auto
    # Framing: how much of the image height the head (top of hair to chin) fills.
    # This is a generation instruction, not a verified measurement — see prompt_builder.
    head_height_pct: int = Field(60, ge=20, le=90)
    seed: Optional[int] = None

    # Output
    output_dir: Optional[str] = None      # explicit dir; if None storage derives a timestamped one
    filename_prefix: str = "portrait"

    # Reliability / throughput
    retry_failed: bool = True
    max_retries: int = Field(3, ge=0)
    concurrency: int = Field(1, ge=1, le=32)

    # Prompt defaults (overridable per request; applied to every item)
    background: str = "plain light gray or off-white background"
    expression: str = "neutral, natural facial expression"
    lighting: str = "natural studio lighting"
    image_style: str = "photorealistic passport-style studio portrait"
    extra_positive_constraints: list[str] = Field(default_factory=list)
    extra_negative_constraints: list[str] = Field(default_factory=list)
    face_crop: bool = False

    # Whether to persist the full rendered prompt into per-image metadata
    save_prompt: bool = True

    @field_validator("quality")
    @classmethod
    def _norm_quality(cls, v: str) -> str:
        v = (v or "medium").strip().lower()
        if v not in {"low", "medium", "high", "auto"}:
            raise ValueError("quality must be one of: low, medium, high, auto.")
        return v

    @field_validator("filename_prefix")
    @classmethod
    def _safe_prefix(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "portrait"
        # keep filenames filesystem-safe
        cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in v)
        return cleaned or "portrait"

    @model_validator(mode="after")
    def _check_distribution(self) -> "BatchGenerationRequest":
        if self.distribution_mode == DistributionMode.EXACT:
            if not self.exact_counts:
                raise ValueError("EXACT distribution requires non-empty `exact_counts`.")
            computed = sum(ec.count for ec in self.exact_counts)
            # total_count is derived from exact_counts in EXACT mode
            object.__setattr__(self, "total_count", computed)
            if computed <= 0:
                raise ValueError("EXACT distribution `exact_counts` must sum to a positive total.")
        else:
            if self.total_count <= 0:
                raise ValueError("`total_count` must be a positive integer.")
            if not (self.age_buckets and self.gender_buckets and self.ethnicity_buckets):
                raise ValueError(
                    "At least one age, one gender and one ethnicity bucket must be selected."
                )
        return self


# --------------------------------------------------------------------------- #
# Planning + results
# --------------------------------------------------------------------------- #
class PlannedItem(BaseModel):
    """One unit of work produced by the batch planner."""

    index: int                # 0-based position in the run
    id: str                   # e.g. "portrait_000001"
    filename: str             # e.g. "portrait_000001.png"
    prompt_options: PromptOptions


class CostEstimate(BaseModel):
    """Pre-flight cost estimate. ``pricing_available`` gates the spend confirmation."""

    provider: str
    model_id: str
    total_count: int
    quality: Optional[str] = None
    price_per_image_usd: Optional[float] = None
    estimated_total_usd: Optional[float] = None
    pricing_available: bool = False
    pricing_source: Optional[str] = None
    warning: Optional[str] = None

    def human_summary(self) -> str:
        if self.pricing_available and self.estimated_total_usd is not None:
            return (
                f"~${self.estimated_total_usd:.2f} "
                f"({self.total_count} x ${self.price_per_image_usd:.4f})"
            )
        return "unknown (pricing data missing — explicit confirmation required)"


class GenerationResult(BaseModel):
    """The outcome for a single planned item. Recorded for successes AND failures."""

    id: str
    filename: Optional[str] = None
    provider: str
    model: str
    prompt: str = ""
    age_bucket: str
    gender_bucket: str
    ethnicity_bucket: str
    variation_level: int
    size: str
    quality: str = "medium"
    seed: Optional[int] = None
    estimated_cost_usd: Optional[float] = None   # local per-image estimate
    actual_cost_usd: Optional[float] = None      # provider-reported $ ONLY (None if unreported)
    cost_is_estimated: bool = True               # True => no provider-reported cost available
    provider_usage: Optional[dict] = None        # provider usage object (tokens), if returned
    created_at: str = Field(default_factory=utcnow_iso)
    status: ItemStatus = ItemStatus.PENDING
    error: Optional[str] = None
    retries: int = 0
    attempts: int = 1                            # billable provider calls (>=1; includes retries)

    def to_record(self, include_prompt: bool = True) -> "OrderedDict[str, Any]":
        """Flat, CSV-friendly record. Identical key-set for success and failure rows."""
        rec: "OrderedDict[str, Any]" = OrderedDict()
        rec["id"] = self.id
        rec["filename"] = self.filename
        rec["provider"] = self.provider
        rec["model"] = self.model
        rec["age_bucket"] = self.age_bucket
        rec["gender_bucket"] = self.gender_bucket
        rec["ethnicity_bucket"] = self.ethnicity_bucket
        rec["variation_level"] = self.variation_level
        rec["size"] = self.size
        rec["quality"] = self.quality
        rec["seed"] = self.seed
        rec["estimated_cost_usd"] = self.estimated_cost_usd
        rec["actual_cost_usd"] = self.actual_cost_usd
        rec["cost_is_estimated"] = self.cost_is_estimated
        rec["created_at"] = self.created_at
        rec["status"] = self.status.value
        rec["error"] = self.error
        rec["retries"] = self.retries
        rec["attempts"] = self.attempts
        rec["prompt"] = self.prompt if include_prompt else ""
        return rec


@dataclass
class ProviderResult:
    """What a provider hands back for one successful image. ``image_bytes`` is the
    raw encoded image (PNG/JPEG); it is never serialized into metadata."""

    image_bytes: bytes
    content_type: str = "image/png"
    actual_cost_usd: Optional[float] = None  # None => provider did not report a USD cost
    usage: Optional[dict[str, Any]] = None   # provider usage object (e.g. token counts), if any
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationEvent:
    """Progress event emitted by the generator. Front-ends render these; the
    generator never imports any UI code."""

    type: EventType
    run_id: str
    index: Optional[int] = None
    total: Optional[int] = None
    item_id: Optional[str] = None
    message: str = ""
    result: Optional[GenerationResult] = None
    success_count: int = 0
    failure_count: int = 0


# --------------------------------------------------------------------------- #
# The central object: a Run
# --------------------------------------------------------------------------- #
class Run(BaseModel):
    """A generation run — settings, estimate, plan, assets, failures, metadata.

    Mutable: results are appended as items complete. Both the TUI and GUI keep a
    reference to one ``Run`` and read its derived properties for display.
    """

    model_config = {"arbitrary_types_allowed": True}

    run_id: str
    request: BatchGenerationRequest
    model_info: ModelInfo
    estimate: CostEstimate
    output_dir: Path
    api_endpoint: Optional[str] = None   # provider endpoint actually used (audit trail)
    plan: list[PlannedItem] = Field(default_factory=list)
    results: list[GenerationResult] = Field(default_factory=list)
    status: RunStatus = RunStatus.CREATED
    created_at: str = Field(default_factory=utcnow_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # ---- derived views -------------------------------------------------- #
    @property
    def total(self) -> int:
        return len(self.plan)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.results if r.status in (ItemStatus.SUCCESS, ItemStatus.FAILED))

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == ItemStatus.SUCCESS)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.status == ItemStatus.FAILED)

    @property
    def progress(self) -> float:
        """0.0–1.0 completion fraction."""
        return (self.completed_count / self.total) if self.total else 0.0

    @property
    def actual_cost_usd(self) -> float:
        """Provider-reported total (0.0 when the provider reports nothing). Prefer
        :meth:`spend_summary` / the explicit accounting properties for display."""
        return round(sum((r.actual_cost_usd or 0.0) for r in self.results), 6)

    @property
    def has_estimated_costs(self) -> bool:
        """True if any recorded cost is an estimate rather than a provider-reported value."""
        return any(r.status == ItemStatus.SUCCESS and r.cost_is_estimated for r in self.results)

    # ---- explicit cost accounting (kept deliberately separate) ---------- #
    @property
    def planned_outputs(self) -> int:
        return self.total

    @property
    def successful_outputs(self) -> int:
        return self.success_count

    @property
    def failed_outputs(self) -> int:
        return self.failure_count

    @property
    def api_attempts(self) -> int:
        """Total billable provider calls — every attempt, including retries."""
        return sum(max(1, r.attempts) for r in self.results)

    @property
    def total_retries(self) -> int:
        return sum(r.retries for r in self.results)

    @property
    def estimated_cost_before_run_usd(self) -> Optional[float]:
        """The pre-run estimate: per-image price × planned outputs."""
        return self.estimate.estimated_total_usd

    @property
    def estimated_cost_from_attempts_usd(self) -> Optional[float]:
        """Burn: per-image price × billable attempts (so retries are included)."""
        price = self.estimate.price_per_image_usd
        if price is None:
            return None
        return round(price * self.api_attempts, 6)

    @property
    def provider_reported_cost_usd(self) -> Optional[float]:
        """Real provider-billed spend — only when the provider actually reports a
        per-image USD amount. ``None`` means "the API gave us no billing figure"."""
        if not self.model_info.reports_actual_cost:
            return None
        costs = [
            r.actual_cost_usd for r in self.results
            if r.status == ItemStatus.SUCCESS and r.actual_cost_usd is not None
        ]
        return round(sum(costs), 6) if costs else None

    @property
    def provider_usage(self) -> dict:
        """Aggregated provider usage (e.g. token totals) summed across results."""
        agg: dict = {}
        for r in self.results:
            if r.provider_usage:
                for key, value in r.provider_usage.items():
                    if isinstance(value, (int, float)):
                        agg[key] = agg.get(key, 0) + value
        return agg

    def spend_summary(self) -> str:
        """One honest line for headlines/notifications."""
        billed = self.provider_reported_cost_usd
        if billed is not None:
            return f"${billed:.4f} billed"
        burn = self.estimated_cost_from_attempts_usd
        if burn is not None:
            return f"~${burn:.4f} est ({self.api_attempts} attempts)"
        return "cost unavailable"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    def record(self, result: GenerationResult) -> None:
        """Add or replace a result by id (idempotent across retries)."""
        for i, existing in enumerate(self.results):
            if existing.id == result.id:
                self.results[i] = result
                return
        self.results.append(result)

    def manifest(self) -> dict[str, Any]:
        """Auditable, JSON-serializable summary of the entire run."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "output_dir": str(self.output_dir),
            "provider": self.model_info.provider,
            "model": self.model_info.model_id,
            "model_display_name": self.model_info.display_name,
            "endpoint": self.api_endpoint,
            "size": self.request.size,
            "quality": self.request.quality,
            "variation_level": self.request.variation_level,
            "distribution_mode": self.request.distribution_mode.value,
            "seed": self.request.seed,
            "request": self.request.model_dump(mode="json"),
            "estimate": self.estimate.model_dump(mode="json"),
            "summary": {
                # output counts vs billable attempts — kept distinct on purpose
                "planned": self.total,
                "planned_outputs": self.total,
                "succeeded": self.success_count,
                "successful_outputs": self.success_count,
                "failed": self.failure_count,
                "failed_outputs": self.failure_count,
                "api_attempts": self.api_attempts,
                "retries": self.total_retries,
                # cost: pre-run estimate, attempt-based burn, real provider spend
                "estimated_total_usd": self.estimate.estimated_total_usd,
                "estimated_cost_before_run_usd": self.estimate.estimated_total_usd,
                "estimated_cost_from_attempts_usd": self.estimated_cost_from_attempts_usd,
                "provider_reported_cost_usd": self.provider_reported_cost_usd,
                "provider_usage": self.provider_usage,
                # back-compat aliases (older readers): actual == provider-reported
                "actual_total_usd": self.actual_cost_usd,
                "actual_cost_includes_estimates": self.has_estimated_costs,
            },
        }
