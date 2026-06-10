"""RunTelemetry — a pure, unit-testable reducer over the engine's event stream.

The app owns exactly one ``RunTelemetry`` per run and mutates it **only** in
the main-thread ``EngineEvent`` handler. Widgets never reduce events
themselves; they read this object on their repaint tickers. ``version`` bumps
on every reduction so painters can skip no-op refreshes.

Honest-numbers rules implemented here:

* the ETA warms up (≥2 terminal events and ≥3 s) before showing anything;
* a rate of ~0 for >20 s flags ``stalled`` and the ETA reads unknown;
* ``actual_cost`` is marked estimated whenever any item's cost was estimated.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from app.core.models import EventType, GenerationEvent, Run


class CellState(str, Enum):
    PENDING = "pending"      # plan preview only (run not started)
    QUEUED = "queued"        # run started, item not yet picked up
    RUNNING = "running"
    RETRY = "retry"
    OK = "ok"
    FAIL = "fail"
    SKIP = "skip"            # still queued when the run was cancelled


@dataclass
class ItemMeta:
    """Immutable per-item facts captured from the plan."""

    index: int
    item_id: str
    filename: str
    age: str
    gender: str
    ethnicity: str
    seed: Optional[int]


@dataclass
class LogEntry:
    """One structured activity-log line (rendered + filtered by the UI)."""

    t: float                  # seconds since run start
    kind: str                 # run | ok | fail | retry | info
    text: str
    item_id: Optional[str] = None
    error: Optional[str] = None


def _error_signature(error: Optional[str]) -> str:
    if not error:
        return "error"
    head = error.split(":", 1)[0].strip()
    return (head or error)[:48]


@dataclass
class RunTelemetry:
    """Everything the Darkroom renders, reduced from ``GenerationEvent``s."""

    total: int = 0
    concurrency: int = 1
    items: list[ItemMeta] = field(default_factory=list)
    states: list[CellState] = field(default_factory=list)
    retries: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    success: int = 0
    failed: int = 0
    retries_total: int = 0

    price_per_image: Optional[float] = None   # per-image estimate (drives the burn figure)
    provider_cost: float = 0.0                # sum of provider-REPORTED costs (real $ only)
    provider_reported_any: bool = False       # did any item carry a real provider-reported cost?
    cost_has_estimates: bool = False

    planned_by_bucket: dict[str, Counter] = field(default_factory=dict)
    landed_by_bucket: dict[str, Counter] = field(default_factory=dict)
    failed_by_bucket: dict[str, Counter] = field(default_factory=dict)

    lanes: list[Optional[int]] = field(default_factory=list)   # item index per lane
    lane_started: dict[int, float] = field(default_factory=dict)  # item index -> mono
    item_started: dict[int, float] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    completion_times: deque = field(default_factory=lambda: deque(maxlen=4096))
    ewma_rate: float = 0.0    # items/sec
    started_mono: Optional[float] = None
    finished_mono: Optional[float] = None

    error_signatures: Counter = field(default_factory=Counter)
    log: list[LogEntry] = field(default_factory=list)

    latest_print_index: Optional[int] = None
    latest_print_path: Optional[Path] = None
    latest_print_latency: Optional[float] = None

    cancel_requested: bool = False
    finished_status: Optional[str] = None   # completed | cancelled | failed
    version: int = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_run(cls, run: Run) -> "RunTelemetry":
        tele = cls(total=run.total, concurrency=run.request.concurrency)
        tele.items = [
            ItemMeta(
                index=item.index,
                item_id=item.id,
                filename=item.filename,
                age=item.prompt_options.age_bucket,
                gender=item.prompt_options.gender_bucket,
                ethnicity=item.prompt_options.ethnicity_bucket,
                seed=item.prompt_options.seed,
            )
            for item in run.plan
        ]
        tele.states = [CellState.PENDING] * run.total
        tele.retries = [0] * run.total
        tele.lanes = [None] * max(1, run.request.concurrency)
        tele.planned_by_bucket = {
            "age": Counter(m.age for m in tele.items),
            "gender": Counter(m.gender for m in tele.items),
            "ethnicity": Counter(m.ethnicity for m in tele.items),
        }
        tele.landed_by_bucket = {dim: Counter() for dim in tele.planned_by_bucket}
        tele.failed_by_bucket = {dim: Counter() for dim in tele.planned_by_bucket}
        tele._images_dir = run.images_dir
        tele.price_per_image = run.estimate.price_per_image_usd
        return tele

    _images_dir: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    @property
    def done(self) -> int:
        return self.success + self.failed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    @property
    def api_attempts(self) -> int:
        """Billable provider calls so far — terminal items plus every retry."""
        return self.done + self.retries_total

    @property
    def burn_estimate(self) -> Optional[float]:
        """Estimated spend from attempts: per-image price × api_attempts."""
        if self.price_per_image is None:
            return None
        return round(self.price_per_image * self.api_attempts, 6)

    @property
    def running_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.states) if s in (CellState.RUNNING, CellState.RETRY)]

    @property
    def is_finished(self) -> bool:
        return self.finished_status is not None

    def elapsed(self, now: Optional[float] = None) -> float:
        if self.started_mono is None:
            return 0.0
        if now is None:
            now = time.monotonic()
        end = self.finished_mono if self.finished_mono is not None else now
        return max(0.0, end - self.started_mono)

    def instantaneous_rate(self, now: Optional[float] = None, window: float = 30.0) -> float:
        """Terminal events per second over the trailing window."""
        if self.started_mono is None:
            return 0.0
        now = time.monotonic() if now is None else now
        window = min(window, max(1.0, now - self.started_mono))
        cutoff = now - window
        recent = sum(1 for t in self.completion_times if t >= cutoff)
        return recent / window

    def tick(self, now: Optional[float] = None) -> None:
        """1 Hz: fold the instantaneous rate into the EWMA."""
        if self.started_mono is None or self.is_finished:
            return
        inst = self.instantaneous_rate(now)
        self.ewma_rate = inst if self.ewma_rate == 0.0 else (0.3 * inst + 0.7 * self.ewma_rate)
        self.version += 1

    def stalled(self, now: Optional[float] = None) -> bool:
        """True when nothing has completed for >20 s mid-run."""
        if self.started_mono is None or self.is_finished or self.done >= self.total:
            return False
        now = time.monotonic() if now is None else now
        last = self.completion_times[-1] if self.completion_times else self.started_mono
        return (now - last) > 20.0

    def eta_seconds(self, now: Optional[float] = None) -> Optional[float]:
        """Honest ETA: ``None`` while warming up or stalled."""
        now = time.monotonic() if now is None else now
        if self.started_mono is None or self.is_finished:
            return None
        if self.done < 2 or (now - self.started_mono) < 3.0:
            return None
        if self.stalled(now) or self.ewma_rate <= 1e-6:
            return None
        return self.remaining / self.ewma_rate

    def bins_per_second(self, seconds: int = 60, now: Optional[float] = None) -> list[float]:
        """Completions per second for the trailing window (sparkline data)."""
        now = time.monotonic() if now is None else now
        bins = [0.0] * seconds
        for t in self.completion_times:
            age = now - t
            if 0 <= age < seconds:
                bins[seconds - 1 - int(age)] += 1
        return bins

    def latency_stats(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not self.latencies:
            return None, None, None
        return (
            sum(self.latencies) / len(self.latencies),
            min(self.latencies),
            max(self.latencies),
        )

    def failure_bias(self) -> Optional[str]:
        """A bucket token over-represented in failures (>60 % of failures,
        <40 % of the plan) — surfaced as a bias warning."""
        if self.failed < 3:
            return None
        for dim, failed_counts in self.failed_by_bucket.items():
            planned_counts = self.planned_by_bucket.get(dim) or Counter()
            for token, n_failed in failed_counts.items():
                if n_failed / self.failed > 0.6:
                    planned_share = (planned_counts.get(token, 0) / self.total) if self.total else 0
                    if planned_share < 0.4:
                        return token
        return None

    # ------------------------------------------------------------------ #
    # Reduction
    # ------------------------------------------------------------------ #
    def reduce(self, event: GenerationEvent, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self.version += 1
        etype = event.type
        index = event.index

        if etype == EventType.RUN_STARTED:
            self.started_mono = now
            self.states = [CellState.QUEUED] * self.total
            self._log(now, "run", event.message)
            return

        if etype == EventType.ITEM_STARTED and index is not None:
            self._set_state(index, CellState.RUNNING)
            self.item_started[index] = now
            self._assign_lane(index, now)
            return

        if etype == EventType.ITEM_RETRYING and index is not None:
            self._set_state(index, CellState.RETRY)
            self.retries[index] += 1
            self.retries_total += 1
            self._log(now, "retry", event.message, item_id=event.item_id)
            return

        if etype in (EventType.ITEM_SUCCEEDED, EventType.ITEM_FAILED) and index is not None:
            ok = etype == EventType.ITEM_SUCCEEDED
            self._set_state(index, CellState.OK if ok else CellState.FAIL)
            self._free_lane(index)
            started = self.item_started.get(index)
            latency = (now - started) if started is not None else None
            if latency is not None:
                self.latencies.append(latency)
            self.completion_times.append(now)
            self.success = event.success_count
            self.failed = event.failure_count

            result = event.result
            meta = self.items[index] if index < len(self.items) else None
            if result is not None:
                if ok:
                    if result.actual_cost_usd is not None:
                        self.provider_cost += result.actual_cost_usd
                        self.provider_reported_any = True
                    if result.cost_is_estimated:
                        self.cost_has_estimates = True
                    if result.filename and self._images_dir is not None:
                        self.latest_print_index = index
                        self.latest_print_path = self._images_dir / result.filename
                        self.latest_print_latency = latency
                else:
                    self.errors[index] = result.error or "error"
                    self.error_signatures[_error_signature(result.error)] += 1
            if meta is not None:
                target = self.landed_by_bucket if ok else self.failed_by_bucket
                target["age"][meta.age] += 1
                target["gender"][meta.gender] += 1
                target["ethnicity"][meta.ethnicity] += 1
            if ok:
                self._log(now, "ok", event.item_id or "", item_id=event.item_id)
            else:
                err = self.errors.get(index, "error") if index is not None else "error"
                self._log(now, "fail", event.item_id or "", item_id=event.item_id, error=err)
            return

        if etype in (EventType.RUN_COMPLETED, EventType.RUN_CANCELLED):
            self.finished_mono = now
            self.success = event.success_count
            self.failed = event.failure_count
            cancelled = etype == EventType.RUN_CANCELLED
            self.finished_status = "cancelled" if cancelled else "completed"
            if cancelled:
                for i, state in enumerate(self.states):
                    if state in (CellState.QUEUED, CellState.PENDING):
                        self.states[i] = CellState.SKIP
            self.lanes = [None] * len(self.lanes)
            self._log(now, "run", event.message)
            return

    def mark_failed(self, error: str, now: Optional[float] = None) -> None:
        """The run itself errored out (auth failure, crash) — terminal state."""
        now = time.monotonic() if now is None else now
        self.version += 1
        self.finished_mono = now
        self.finished_status = "failed"
        self.lanes = [None] * len(self.lanes)
        # nothing else will ever complete — don't leave cells frozen mid-state
        for i, state in enumerate(self.states):
            if state not in (CellState.OK, CellState.FAIL):
                self.states[i] = CellState.SKIP
        self._log(now, "fail", error, error=error)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _set_state(self, index: int, state: CellState) -> None:
        if 0 <= index < len(self.states):
            self.states[index] = state

    def _assign_lane(self, index: int, now: float) -> None:
        self.lane_started[index] = now
        for lane, occupant in enumerate(self.lanes):
            if occupant is None:
                self.lanes[lane] = index
                return
        self.lanes.append(index)

    def _free_lane(self, index: int) -> None:
        self.lane_started.pop(index, None)
        for lane, occupant in enumerate(self.lanes):
            if occupant == index:
                self.lanes[lane] = None
                return

    def _log(
        self,
        now: float,
        kind: str,
        text: str,
        *,
        item_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        t = (now - self.started_mono) if self.started_mono is not None else 0.0
        self.log.append(LogEntry(t=t, kind=kind, text=text, item_id=item_id, error=error))


def fmt_mmss(seconds: Optional[float]) -> str:
    """``m:ss`` (or ``--:--`` when unknown)."""
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
