"""Batch generation orchestration — the engine.

Flow::

    request -> create_run() -> Run(PLANNED, with estimate + plan)
            -> confirm(run)  -> Run(CONFIRMED)
            -> execute(run)  -> Run(COMPLETED), images + metadata on disk

``create_run`` never spends money or writes files: it only plans and estimates.
``execute`` refuses to run until the run is CONFIRMED, enforcing the
"no generation before confirmation" rule at the engine layer rather than in any UI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .batch_planner import plan_batch
from .buckets import validate_selection
from .config import AppConfig
from .metadata import MetadataWriter
from .models import (
    BatchGenerationRequest,
    DistributionMode,
    EventType,
    GenerationEvent,
    GenerationResult,
    ItemStatus,
    PlannedItem,
    Run,
    RunStatus,
    utcnow_iso,
)
from .prompt_builder import build_prompt
from .providers.base import ImageProvider, ProviderAuthError, ProviderError
from .providers.registry import build_provider
from .sizes import resolve_a4_portrait_size, validate_request_size
from .storage import Storage, new_run_id

EventCallback = Callable[[GenerationEvent], None]
CancelCheck = Callable[[], bool]


class RunNotConfirmedError(RuntimeError):
    """Raised if execution is attempted on a run the user has not confirmed."""


class Generator:
    """Stateless-ish orchestrator. One instance can create and run many runs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.pricing = config.pricing
        self.storage = Storage(config.settings.output_base_dir)

    # ------------------------------------------------------------------ #
    # Planning + estimation (no spend, no files)
    # ------------------------------------------------------------------ #
    def create_run(self, request: BatchGenerationRequest) -> Run:
        self._validate_buckets(request)
        model_info = self.pricing.get_model_info(request.provider, request.model_id)

        if request.face_crop:
            request = request.model_copy(
                update={
                    "size": resolve_a4_portrait_size(
                        request.provider, request.model_id, model_info
                    )
                }
            )

        # Size must be a listed preset, or — for models that accept custom sizes
        # (gpt-image-2) — satisfy that model's resolution constraints.
        validate_request_size(request.size, model_info)

        plan = plan_batch(request)
        estimate = self.pricing.estimate(request)

        run_id = new_run_id()
        output_dir = self.storage.resolve_run_dir(run_id, request.output_dir)

        return Run(
            run_id=run_id,
            request=request,
            model_info=model_info,
            estimate=estimate,
            output_dir=output_dir,
            plan=plan,
            status=RunStatus.PLANNED,
        )

    def confirm(self, run: Run) -> Run:
        """Record explicit user approval to spend. Required before :meth:`execute`."""
        if run.status not in (RunStatus.PLANNED, RunStatus.CONFIRMED):
            raise RunNotConfirmedError(
                f"Run {run.run_id} cannot be confirmed from status {run.status}."
            )
        run.status = RunStatus.CONFIRMED
        return run

    def _validate_buckets(self, request: BatchGenerationRequest) -> None:
        cfg = self.config.buckets
        if request.distribution_mode == DistributionMode.EXACT and request.exact_counts:
            ages = sorted({ec.age_bucket for ec in request.exact_counts})
            genders = sorted({ec.gender_bucket for ec in request.exact_counts})
            eths = sorted({ec.ethnicity_bucket for ec in request.exact_counts})
        else:
            ages, genders, eths = (
                request.age_buckets,
                request.gender_buckets,
                request.ethnicity_buckets,
            )
        validate_selection(
            age_buckets=ages,
            gender_buckets=genders,
            ethnicity_buckets=eths,
            config=cfg,
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def execute(
        self,
        run: Run,
        *,
        provider: Optional[ImageProvider] = None,
        on_event: Optional[EventCallback] = None,
        should_cancel: Optional[CancelCheck] = None,
    ) -> Run:
        if run.status != RunStatus.CONFIRMED:
            raise RunNotConfirmedError(
                f"Run {run.run_id} is {run.status.value}; it must be CONFIRMED before execution. "
                "Show the cost estimate and call confirm() first."
            )

        owns_provider = provider is None
        if provider is None:
            provider = build_provider(
                run.request.provider, **self.config.credentials_for(run.request.provider)
            )
        # Record the endpoint actually used, for the cost audit trail.
        run.api_endpoint = getattr(provider, "api_endpoint", "") or None

        run.output_dir = self.storage.prepare(run.output_dir)
        writer = MetadataWriter(
            self.storage, run.output_dir, save_prompt=run.request.save_prompt
        )

        run.status = RunStatus.RUNNING
        run.started_at = utcnow_iso()
        writer.write_manifest(run)  # initial manifest (settings captured up-front)
        self._emit(
            on_event,
            GenerationEvent(
                type=EventType.RUN_STARTED,
                run_id=run.run_id,
                total=run.total,
                message=f"Starting {run.total} image(s) on "
                f"{run.request.provider}/{run.request.model_id}",
            ),
        )

        # validate credentials once, up-front, with a clear error
        try:
            provider.ensure_ready()
        except ProviderAuthError as exc:
            run.status = RunStatus.FAILED
            run.finished_at = utcnow_iso()
            writer.write_manifest(run)
            if owns_provider:
                await provider.aclose()
            raise

        semaphore = asyncio.Semaphore(max(1, run.request.concurrency))
        lock = asyncio.Lock()  # serialize result recording + metadata appends
        cancelled = False

        async def worker(item: PlannedItem) -> None:
            nonlocal cancelled
            if should_cancel and should_cancel():
                cancelled = True
                return
            async with semaphore:
                if should_cancel and should_cancel():
                    cancelled = True
                    return
                result = await self._generate_item(run, item, provider, on_event)
                async with lock:
                    run.record(result)
                    writer.append_result(result)
                    self._emit(
                        on_event,
                        GenerationEvent(
                            type=(
                                EventType.ITEM_SUCCEEDED
                                if result.status == ItemStatus.SUCCESS
                                else EventType.ITEM_FAILED
                            ),
                            run_id=run.run_id,
                            index=item.index,
                            total=run.total,
                            item_id=item.id,
                            result=result,
                            success_count=run.success_count,
                            failure_count=run.failure_count,
                            message=(
                                f"{item.id} ✓"
                                if result.status == ItemStatus.SUCCESS
                                else f"{item.id} ✗ {result.error}"
                            ),
                        ),
                    )

        try:
            await asyncio.gather(*(worker(item) for item in run.plan))
        finally:
            if owns_provider:
                await provider.aclose()

        run.finished_at = utcnow_iso()
        run.status = RunStatus.CANCELLED if cancelled else RunStatus.COMPLETED
        # deterministic on-disk ordering
        run.results.sort(key=lambda r: r.id)
        writer.finalize(run)
        self._emit(
            on_event,
            GenerationEvent(
                type=EventType.RUN_CANCELLED if cancelled else EventType.RUN_COMPLETED,
                run_id=run.run_id,
                total=run.total,
                success_count=run.success_count,
                failure_count=run.failure_count,
                message=(
                    f"Done: {run.success_count} succeeded, {run.failure_count} failed. "
                    f"{run.spend_summary()}"
                ),
            ),
        )
        return run

    async def _generate_item(
        self,
        run: Run,
        item: PlannedItem,
        provider: ImageProvider,
        on_event: Optional[EventCallback],
    ) -> GenerationResult:
        opts = item.prompt_options
        prompt = build_prompt(opts)
        est_price = self.pricing.per_image_price(
            run.request.provider, run.request.model_id, run.request.quality
        )

        base_result = GenerationResult(
            id=item.id,
            filename=item.filename,
            provider=run.request.provider,
            model=run.request.model_id,
            prompt=prompt,
            age_bucket=opts.age_bucket,
            gender_bucket=opts.gender_bucket,
            ethnicity_bucket=opts.ethnicity_bucket,
            variation_level=opts.variation_level,
            size=run.request.size,
            quality=run.request.quality,
            seed=opts.seed,
            estimated_cost_usd=est_price,
            status=ItemStatus.RUNNING,
        )

        self._emit(
            on_event,
            GenerationEvent(
                type=EventType.ITEM_STARTED,
                run_id=run.run_id,
                index=item.index,
                total=run.total,
                item_id=item.id,
                message=f"{item.id}: {opts.age_bucket} / {opts.gender_bucket} / {opts.ethnicity_bucket}",
            ),
        )

        max_attempts = (run.request.max_retries + 1) if run.request.retry_failed else 1
        last_error = "unknown error"
        attempts = 0  # billable provider calls actually made (every try, incl. retries)
        for attempt in range(max_attempts):
            attempts += 1
            try:
                pr = await provider.generate(
                    prompt=prompt,
                    size=run.request.size,
                    model_id=run.request.model_id,
                    seed=opts.seed,
                    quality=run.request.quality,
                )
                self.storage.save_image(run.output_dir, item.filename, pr.image_bytes)
                # "actual" is ONLY a provider-reported USD amount. If the provider
                # doesn't report one (OpenAI/Gemini/etc.), we keep it None and the
                # number stays an estimate — we never echo the estimate as "actual".
                provider_cost = (
                    pr.actual_cost_usd
                    if (pr.actual_cost_usd is not None and run.model_info.reports_actual_cost)
                    else None
                )
                return base_result.model_copy(
                    update={
                        "status": ItemStatus.SUCCESS,
                        "actual_cost_usd": provider_cost,
                        "cost_is_estimated": provider_cost is None,
                        "provider_usage": pr.usage,
                        "attempts": attempts,
                        "retries": attempts - 1,
                        "error": None,
                        "created_at": utcnow_iso(),
                    }
                )
            except ProviderAuthError as exc:
                last_error = f"auth error: {exc}"
                break  # retrying won't fix credentials
            except Exception as exc:  # noqa: BLE001 - a bad item must never kill the batch
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts - 1:
                    self._emit(
                        on_event,
                        GenerationEvent(
                            type=EventType.ITEM_RETRYING,
                            run_id=run.run_id,
                            index=item.index,
                            total=run.total,
                            item_id=item.id,
                            message=f"{item.id}: retry {attempt + 1}/{max_attempts - 1} ({last_error})",
                        ),
                    )
                    await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))

        return base_result.model_copy(
            update={
                "status": ItemStatus.FAILED,
                "filename": None,
                "actual_cost_usd": None,
                "attempts": attempts,
                "error": last_error,
                "retries": max(0, attempts - 1),
                "created_at": utcnow_iso(),
            }
        )

    @staticmethod
    def _emit(on_event: Optional[EventCallback], event: GenerationEvent) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            # A misbehaving UI callback must never interfere with generation.
            pass


async def run_batch(
    config: AppConfig,
    request: BatchGenerationRequest,
    *,
    auto_confirm: bool = False,
    on_event: Optional[EventCallback] = None,
    provider: Optional[ImageProvider] = None,
) -> Run:
    """Convenience one-shot: plan, (optionally) confirm, execute.

    ``auto_confirm`` exists for headless callers that have already obtained
    confirmation (e.g. a CLI ``--yes`` flag). It must be set deliberately.
    """
    gen = Generator(config)
    run = gen.create_run(request)
    if not auto_confirm:
        raise RunNotConfirmedError(
            "run_batch requires auto_confirm=True (confirmation must be handled by the caller)."
        )
    gen.confirm(run)
    return await gen.execute(run, provider=provider, on_event=on_event)
