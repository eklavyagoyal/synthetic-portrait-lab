"""Unit tests for the TUI's RunTelemetry reducer (pure, no Textual needed)."""

from __future__ import annotations

import pytest

from app.core.buckets import BucketConfig
from app.core.config import AppConfig, Settings, load_model_registry
from app.core.generator import Generator
from app.core.models import (
    BatchGenerationRequest,
    EventType,
    GenerationEvent,
    GenerationResult,
    ItemStatus,
)
from app.tui.telemetry import CellState, RunTelemetry, fmt_mmss


@pytest.fixture()
def run(tmp_path):
    config = AppConfig(Settings(), BucketConfig(), load_model_registry(None))
    request = BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=["adult, 26 to 40", "young adult, 18 to 25"],
        gender_buckets=["female-presenting"],
        ethnicity_buckets=["East Asian"],
        total_count=4,
        seed=7,
        output_dir=str(tmp_path / "run"),
        concurrency=2,
    )
    return Generator(config).create_run(request)


def _result(run, index: int, *, ok: bool, cost: float = 0.01, estimated: bool = True,
            attempts: int = 1, usage=None):
    item = run.plan[index]
    # Honest model: actual_cost_usd holds a provider-REPORTED amount only; an
    # estimated item (or a failure) carries no provider cost.
    provider_cost = None if (estimated or not ok) else cost
    return GenerationResult(
        id=item.id,
        filename=item.filename if ok else None,
        provider="mock",
        model="mock-image",
        age_bucket=item.prompt_options.age_bucket,
        gender_bucket=item.prompt_options.gender_bucket,
        ethnicity_bucket=item.prompt_options.ethnicity_bucket,
        variation_level=0,
        size="1024x1024",
        status=ItemStatus.SUCCESS if ok else ItemStatus.FAILED,
        actual_cost_usd=provider_cost,
        cost_is_estimated=estimated,
        provider_usage=usage,
        attempts=attempts,
        error=None if ok else "ReadTimeout: too slow",
    )


def _event(run, etype, index=None, *, ok_count=0, fail_count=0, result=None):
    return GenerationEvent(
        type=etype,
        run_id=run.run_id,
        index=index,
        total=run.total,
        item_id=run.plan[index].id if index is not None else None,
        result=result,
        success_count=ok_count,
        failure_count=fail_count,
        message="msg",
    )


def test_initial_state_from_plan(run):
    tele = RunTelemetry.from_run(run)
    assert tele.total == 4
    assert all(s == CellState.PENDING for s in tele.states)
    assert tele.planned_by_bucket["gender"]["female-presenting"] == 4
    assert tele.planned_by_bucket["age"]["adult, 26 to 40"] == 2
    assert len(tele.lanes) == 2


def test_lifecycle_reduction(run):
    tele = RunTelemetry.from_run(run)
    t0 = 100.0
    tele.reduce(_event(run, EventType.RUN_STARTED), now=t0)
    assert all(s == CellState.QUEUED for s in tele.states)
    assert tele.started_mono == t0

    tele.reduce(_event(run, EventType.ITEM_STARTED, 0), now=t0 + 1)
    assert tele.states[0] == CellState.RUNNING
    assert tele.lanes[0] == 0

    tele.reduce(_event(run, EventType.ITEM_RETRYING, 0), now=t0 + 2)
    assert tele.states[0] == CellState.RETRY
    assert tele.retries[0] == 1
    assert tele.retries_total == 1

    ok = _result(run, 0, ok=True, cost=0.04, estimated=True)
    tele.reduce(
        _event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok), now=t0 + 3
    )
    assert tele.states[0] == CellState.OK
    assert tele.lanes[0] is None
    assert tele.success == 1
    # Estimated item: no provider-reported $, and the retry counts as an attempt.
    assert tele.provider_cost == pytest.approx(0.0)
    assert tele.provider_reported_any is False
    assert tele.cost_has_estimates is True
    assert tele.api_attempts == 2  # 1 success + 1 retry
    assert tele.latencies == [pytest.approx(2.0)]
    assert tele.latest_print_path is not None
    assert tele.landed_by_bucket["ethnicity"]["East Asian"] == 1

    bad = _result(run, 1, ok=False)
    tele.reduce(_event(run, EventType.ITEM_STARTED, 1), now=t0 + 3)
    tele.reduce(
        _event(run, EventType.ITEM_FAILED, 1, ok_count=1, fail_count=1, result=bad),
        now=t0 + 4,
    )
    assert tele.states[1] == CellState.FAIL
    assert tele.errors[1] == "ReadTimeout: too slow"
    assert tele.error_signatures["ReadTimeout"] == 1
    assert tele.failed_by_bucket["gender"]["female-presenting"] == 1


def test_burn_estimate_includes_retries(run):
    tele = RunTelemetry.from_run(run)
    tele.price_per_image = 0.07  # pretend a token-billed model
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    tele.reduce(_event(run, EventType.ITEM_RETRYING, 0), now=1.0)
    ok = _result(run, 0, ok=True, estimated=True, attempts=2)
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok), now=2.0)
    assert tele.retries_total == 1
    assert tele.api_attempts == 2                       # 1 output + 1 retry
    assert tele.burn_estimate == pytest.approx(0.14)    # 2 attempts × $0.07


def test_provider_reported_cost_tracked(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    ok = _result(run, 0, ok=True, cost=0.05, estimated=False)  # provider reported it
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok), now=1.0)
    assert tele.provider_reported_any is True
    assert tele.provider_cost == pytest.approx(0.05)
    assert tele.cost_has_estimates is False


def test_cancel_marks_queued_as_skipped(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    ok = _result(run, 0, ok=True)
    tele.reduce(_event(run, EventType.ITEM_STARTED, 0), now=1.0)
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok), now=2.0)
    tele.reduce(
        _event(run, EventType.RUN_CANCELLED, ok_count=1, fail_count=0), now=3.0
    )
    assert tele.finished_status == "cancelled"
    assert tele.states[0] == CellState.OK
    assert all(s == CellState.SKIP for s in tele.states[1:])
    assert tele.is_finished


def test_eta_warms_up_then_estimates(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    assert tele.eta_seconds(now=1.0) is None  # nothing completed
    ok0 = _result(run, 0, ok=True)
    ok1 = _result(run, 1, ok=True)
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok0), now=2.0)
    assert tele.eta_seconds(now=2.5) is None  # done < 2
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 1, ok_count=2, result=ok1), now=4.0)
    tele.tick(now=4.0)
    eta = tele.eta_seconds(now=4.0)
    assert eta is not None and eta > 0


def test_stall_detection_suppresses_eta(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    ok0 = _result(run, 0, ok=True)
    ok1 = _result(run, 1, ok=True)
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 0, ok_count=1, result=ok0), now=1.0)
    tele.reduce(_event(run, EventType.ITEM_SUCCEEDED, 1, ok_count=2, result=ok1), now=2.0)
    tele.tick(now=2.0)
    assert not tele.stalled(now=10.0)
    assert tele.stalled(now=30.0)
    assert tele.eta_seconds(now=30.0) is None


def test_bins_and_rate(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    for i, t in enumerate((1.0, 2.0)):
        ok = _result(run, i, ok=True)
        tele.reduce(
            _event(run, EventType.ITEM_SUCCEEDED, i, ok_count=i + 1, result=ok), now=t
        )
    bins = tele.bins_per_second(seconds=10, now=5.0)
    assert sum(bins) == 2
    assert tele.instantaneous_rate(now=5.0, window=5.0) == pytest.approx(2 / 5)


def test_run_level_failure(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    tele.mark_failed("Authentication failed: no key", now=1.0)
    assert tele.finished_status == "failed"
    assert tele.is_finished
    assert tele.log[-1].kind == "fail"


def test_failure_bias_detection(run):
    tele = RunTelemetry.from_run(run)
    tele.reduce(_event(run, EventType.RUN_STARTED), now=0.0)
    # 3 failures, all in the same age bucket which is half the plan -> no flag
    fails = 0
    for index in range(4):
        item = run.plan[index]
        ok = item.prompt_options.age_bucket != "adult, 26 to 40"
        result = _result(run, index, ok=ok)
        fails += 0 if ok else 1
        tele.reduce(
            _event(
                run,
                EventType.ITEM_SUCCEEDED if ok else EventType.ITEM_FAILED,
                index,
                ok_count=index + 1 - fails,
                fail_count=fails,
                result=result,
            ),
            now=float(index),
        )
    # bucket share is 50% of plan (not < 40%) so it should NOT flag
    assert tele.failure_bias() is None


def test_fmt_mmss():
    assert fmt_mmss(None) == "--:--"
    assert fmt_mmss(0) == "00:00"
    assert fmt_mmss(75) == "01:15"
    assert fmt_mmss(-1) == "--:--"
