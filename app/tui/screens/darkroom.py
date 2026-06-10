"""DARKROOM — the live run screen.

A pure *view* over the app-owned :class:`RunTelemetry`: it never reduces
engine events itself, so it can be popped (the run keeps going) and re-pushed
(it rebuilds from telemetry) at any time. Two tickers drive repaints — 250 ms
for the matrix/lanes/log, 1 s for clock/ETA/cost/coverage — and both stop at
the terminal state so an idle app is perfectly still.

Cancel is the inline double-press pattern: ``ctrl+x`` arms a 3-second window,
``ctrl+x`` again flips the engine's ``should_cancel`` flag; in-flight items
drain visibly and still-queued cells restyle as skipped.
"""

from __future__ import annotations

import time
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, ProgressBar, RichLog, Static

from app.core.models import Run

from rich.markup import escape as esc

from .. import glyphs, labels, palette
from ..telemetry import CellState, RunTelemetry, fmt_mmss
from ..widgets import (
    BatchMatrix,
    CostLedger,
    CoveragePanel,
    Hero,
    LaneBoard,
    PrintPanel,
    ThroughputPanel,
)
from .modals import FailureTriageModal, PromptPeekModal

LOG_FILTERS = ("all", "state-changes", "failures")


class DarkroomScreen(Screen):
    AUTO_FOCUS = "BatchMatrix"

    BINDINGS = [
        Binding("ctrl+x", "cancel", "cancel", tooltip="Press twice within 3 s to cancel the run"),
        Binding("c", "color_mode", "color", tooltip="Cycle matrix colouring: state/age/gender/ethnicity"),
        Binding("l", "log_filter", "log", tooltip="Cycle log filter"),
        Binding("f", "triage", "failures", tooltip="Failure triage table"),
        Binding("o", "reveal", "reveal", tooltip="Open the run directory"),
        Binding("enter", "contact_sheet", "contact sheet", show=False),
        Binding("escape", "back", "studio", tooltip="Back to the studio (the run keeps going)"),
    ]

    DEFAULT_CSS = """
    DarkroomScreen {
        background: $background;

        #darkroom-body { height: 1fr; padding: 0 1; }
        #dk-left { width: 1fr; min-width: 44; padding: 0 1 0 0; }
        #dk-right { width: 44; }

        .card {
            height: auto;
            border: round $border-soft;
            border-title-color: $secondary;
            background: $surface;
            padding: 0 1;
            margin: 0 0 1 0;
        }
        #matrix-card { max-height: 12; }
        #matrix-scroll { height: auto; max-height: 8; }
        #matrix-inspect { height: auto; color: $text-muted; }
        #cancel-strip {
            height: auto;
            background: $warning 20%;
            color: $warning;
            padding: 0 1;
            display: none;
            &.-visible { display: block; }
        }

        #dk-progress { width: 1fr; margin: 0 0 1 0; }
        #dk-progress Bar { width: 1fr; }
        #dk-progress Bar > .bar--complete { color: $success; }

        #print-log-row { height: auto; }
        #print-card { width: auto; margin: 0 1 1 0; }
        #log-card { width: 1fr; height: auto; }
        #dk-log { height: 14; background: $surface; padding: 0 1; }

        #summary-card { display: none; }
        &.-finished #summary-card { display: block; }
        &.-finished #lanes-card { display: none; }

        &.-compact #print-card, &.-compact #lanes-card, &.-compact #tp-card {
            display: none;
        }
        &.-compact #dk-right { width: 30; }
        &.-compact #dk-log { height: 8; }
        #right-scroll { height: 1fr; }
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._tick = 0
        self._fast_timer: Optional[Timer] = None
        self._slow_timer: Optional[Timer] = None
        self._log_cursor = 0
        self._log_filter = "all"
        self._cancel_armed_until = 0.0
        self._finished_rendered = False
        self._last_version = -1

    # ------------------------------------------------------------------ #
    @property
    def _telemetry(self) -> Optional[RunTelemetry]:
        return getattr(self.app, "telemetry", None)

    @property
    def _run(self) -> Optional[Run]:
        return getattr(self.app, "current_run", None)

    def compose(self) -> ComposeResult:
        yield Hero(active="darkroom")
        with Horizontal(id="darkroom-body"):
            with Vertical(id="dk-left"):
                with Vertical(classes="card", id="matrix-card") as card:
                    card.border_title = "batch"
                    with VerticalScroll(id="matrix-scroll"):
                        yield BatchMatrix(id="matrix")
                    yield Static("", id="matrix-inspect")
                yield Static("", id="cancel-strip")
                yield ProgressBar(total=100, show_eta=False, id="dk-progress")
                with Vertical(classes="card", id="lanes-card") as card:
                    card.border_title = "lanes"
                    yield LaneBoard(id="lanes")
                with Vertical(classes="card", id="summary-card") as card:
                    card.border_title = "debrief"
                    yield Static("", id="summary")
                with Horizontal(id="print-log-row"):
                    with Vertical(classes="card", id="print-card") as card:
                        card.border_title = "the print"
                        yield PrintPanel(id="print")
                    with Vertical(classes="card", id="log-card") as card:
                        card.border_title = "event log"
                        card.border_subtitle = "l filter"
                        yield RichLog(id="dk-log", markup=True, max_lines=2000, wrap=True)
            with Vertical(id="dk-right"):
                with VerticalScroll(id="right-scroll"):
                    with Vertical(classes="card", id="cost-card") as card:
                        card.border_title = "cost"
                        yield CostLedger(id="cost")
                    with Vertical(classes="card", id="tp-card") as card:
                        card.border_title = "throughput"
                        yield ThroughputPanel(id="throughput")
                    with Vertical(classes="card", id="coverage-card") as card:
                        card.border_title = "coverage"
                        yield CoveragePanel(id="coverage")
                    with Vertical(classes="card", id="failures-card") as card:
                        card.border_title = "failures"
                        yield Static("", id="failure-strip")
        yield Footer()

    def on_mount(self) -> None:
        tele, run = self._telemetry, self._run
        if tele is None or run is None:
            self.query_one("#matrix-inspect", Static).update(
                "[$text-muted]no active run[/]"
            )
            return
        matrix = self.query_one("#matrix", BatchMatrix)
        matrix.attach(tele)
        self.query_one("#lanes", LaneBoard).attach(tele)
        self.query_one("#coverage", CoveragePanel).attach(tele)
        self.query_one("#cost", CostLedger).attach(tele, run.estimate)
        self.query_one("#throughput", ThroughputPanel).attach(tele)
        self.query_one("#matrix-card").border_title = f"batch · {tele.total} items"
        self.query_one("#lanes-card").border_title = f"lanes · concurrency {tele.concurrency}"
        if tele.concurrency <= 1:
            self.query_one("#lanes-card").display = False

        progress = self.query_one("#dk-progress", ProgressBar)
        progress.update(total=max(1, tele.total), progress=tele.done)
        try:
            from textual.color import Gradient

            progress.gradient = Gradient.from_colors(
                palette.literal(self.app, "grad-a"), palette.literal(self.app, "grad-b")
            )
        except Exception:  # noqa: BLE001 - CSS fallback colours still apply
            pass

        self._apply_breakpoint(self.app.size.width)
        self._render_log(reset=True)
        self._sync_cancel_strip()  # re-pushed mid-cancel-drain: show CANCELLING
        self._repaint_fast()
        self._repaint_slow()
        if tele.is_finished:
            self._render_finished()
        else:
            self._fast_timer = self.set_interval(0.25, self._repaint_fast)
            self._slow_timer = self.set_interval(1.0, self._repaint_slow)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_breakpoint(event.size.width)

    def _apply_breakpoint(self, width: int) -> None:
        self.set_class(width < 110, "-compact")
        self.set_class(width >= 150, "-wide")

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "cancel":
            tele = self._telemetry
            if tele is None or tele.is_finished:
                return False  # hide 'cancel' from the footer once terminal
        return True

    # ------------------------------------------------------------------ #
    # Tickers
    # ------------------------------------------------------------------ #
    def _repaint_fast(self) -> None:
        tele = self._telemetry
        if tele is None:
            return
        self._tick += 1
        now = time.monotonic()
        if self._cancel_armed_until and now > self._cancel_armed_until:
            self._cancel_armed_until = 0.0
            self._sync_cancel_strip()
        if tele.version != self._last_version or any(
            s in (CellState.RUNNING, CellState.RETRY) for s in tele.states
        ):
            self.query_one("#matrix", BatchMatrix).repaint_cells(self._tick)
            self.query_one("#lanes", LaneBoard).repaint(now)
        self._render_inspect()
        self._render_log()
        self._render_print()
        if tele.is_finished and not self._finished_rendered:
            self._render_finished()

    def _repaint_slow(self) -> None:
        tele = self._telemetry
        if tele is None:
            return
        tele.tick()
        now = time.monotonic()
        self.query_one("#dk-progress", ProgressBar).update(
            total=max(1, tele.total), progress=tele.done
        )
        self.query_one("#cost", CostLedger).repaint()
        self.query_one("#throughput", ThroughputPanel).repaint()
        self.query_one("#coverage", CoveragePanel).repaint()
        self._render_failures()
        self._render_header(now)
        self._last_version = tele.version

    # ------------------------------------------------------------------ #
    # Panel renderers
    # ------------------------------------------------------------------ #
    def _render_header(self, now: float) -> None:
        tele, run = self._telemetry, self._run
        if tele is None or run is None:
            return
        elapsed = fmt_mmss(tele.elapsed(now))
        if tele.is_finished:
            state = {
                "completed": f"[$tele-ok]{glyphs.DOT} COMPLETE[/]",
                "cancelled": f"[$warning]{glyphs.DOT} CANCELLED[/]",
                "failed": f"[$tele-fail]{glyphs.CROSS} FAILED[/]",
            }.get(tele.finished_status or "", "")
            sub = f"{state} · {elapsed} · {tele.success}{glyphs.CHECK} {tele.failed}{glyphs.CROSS}"
        elif getattr(self.app, "cancel_requested", False):
            sub = f"[$warning]{glyphs.CROSS} CANCELLING — finishing in-flight items[/] · {elapsed}"
        elif tele.stalled(now):
            sub = f"[$warning]{glyphs.WARN} STALLED[/] · {elapsed} · ETA --:--"
        else:
            eta = tele.eta_seconds(now)
            rate = tele.ewma_rate * 60
            eta_str = fmt_mmss(eta) if eta is not None else (
                "warming up…" if tele.done < 2 else "--:--"
            )
            sub = (
                f"[$tele-running]{glyphs.DOT} RUNNING[/] · ELAPSED {elapsed}"
                f" · ETA {eta_str} · {rate:.1f} img/min"
            )
        self.sub_title = ""
        self.query_one("#matrix-card").border_subtitle = sub

    def _render_inspect(self) -> None:
        tele = self._telemetry
        if tele is None:
            return
        matrix = self.query_one("#matrix", BatchMatrix)
        meta = matrix.item_meta(matrix.cursor)
        if meta is None:
            self.query_one("#matrix-inspect", Static).update("")
            return
        state = tele.states[meta.index]
        retries = tele.retries[meta.index]
        seed = meta.seed if meta.seed is not None else "random"
        line = (
            f"[$text-muted]#{meta.index + 1:03d}[/] [b]{meta.item_id}[/b]"
            f" [{self._state_var(state)}]{state.value.upper()}[/]"
            + (f" [$tele-retry]{glyphs.RETRY}{retries}[/]" if retries else "")
            + f" · {labels.triple_chips(meta.age, meta.gender, meta.ethnicity)}"
            + f" · [$text-muted]seed {seed}[/]"
        )
        error = tele.errors.get(meta.index)
        if error:
            line += f" [$tele-fail]— {esc(error[:40])}[/]"
        self.query_one("#matrix-inspect", Static).update(line)

    @staticmethod
    def _state_var(state: CellState) -> str:
        return {
            CellState.PENDING: "$tele-pending",
            CellState.QUEUED: "$tele-queued",
            CellState.RUNNING: "$tele-running",
            CellState.RETRY: "$tele-retry",
            CellState.OK: "$tele-ok",
            CellState.FAIL: "$tele-fail",
            CellState.SKIP: "$tele-pending",
        }[state]

    def _render_log(self, reset: bool = False) -> None:
        tele = self._telemetry
        if tele is None:
            return
        log = self.query_one("#dk-log", RichLog)
        if reset:
            log.clear()
            self._log_cursor = 0
        muted = palette.text_muted(self.app)
        ok = palette.literal(self.app, "tele-ok")
        fail = palette.literal(self.app, "tele-fail")
        retry = palette.literal(self.app, "tele-retry")
        suppress_ok = self._log_filter == "failures"
        suppress_info = self._log_filter in ("state-changes", "failures")
        while self._log_cursor < len(tele.log):
            entry = tele.log[self._log_cursor]
            self._log_cursor += 1
            stamp = f"[{muted}]{fmt_mmss(entry.t)}[/]"
            if entry.kind == "ok":
                if suppress_ok:
                    continue
                log.write(f"{stamp} [{ok}]{glyphs.CHECK}[/] {esc(entry.text)}")
            elif entry.kind == "fail":
                line = f"{stamp} [{fail}]{glyphs.CROSS}[/] {esc(entry.text)}"
                if entry.error:
                    line += f" [{muted}]— {esc(entry.error)}[/]"
                log.write(line)
            elif entry.kind == "retry":
                if suppress_info:
                    continue
                log.write(f"{stamp} [{retry}]{glyphs.RETRY}[/] {esc(entry.text)}")
            elif entry.kind == "run":
                log.write(f"{stamp} [b]{esc(entry.text)}[/b]")
            else:
                if suppress_info:
                    continue
                log.write(f"{stamp} [{muted}]{esc(entry.text)}[/]")

    def _render_print(self) -> None:
        tele = self._telemetry
        if tele is None or tele.latest_print_path is None:
            return
        panel = self.query_one("#print", PrintPanel)
        index = tele.latest_print_index
        meta = tele.items[index] if index is not None and index < len(tele.items) else None
        caption = ""
        if meta is not None:
            duration = (
                f" · {tele.latest_print_latency:.1f}s"
                if tele.latest_print_latency is not None
                else ""
            )
            seed = meta.seed if meta.seed is not None else "random"
            caption = (
                labels.triple_chips(meta.age, meta.gender, meta.ethnicity)
                + f"\n[$text-muted]seed {seed}{duration}[/]"
            )
        panel.show(tele.latest_print_path, caption)

    def _render_failures(self) -> None:
        tele = self._telemetry
        if tele is None:
            return
        strip = self.query_one("#failure-strip", Static)
        if tele.failed == 0:
            strip.update(f"[$text-muted]{glyphs.DIAMOND_HOLLOW} no failures[/]")
            return
        signatures = " · ".join(
            f"{esc(sig)} ×{n}" for sig, n in tele.error_signatures.most_common(2)
        )
        bias = tele.failure_bias()
        bias_line = (
            f"\n[$warning]{glyphs.WARN} failures cluster in: {labels.short(bias)}[/]" if bias else ""
        )
        strip.update(
            f"[$tele-fail]{glyphs.CROSS} {tele.failed} failed[/] · {signatures}"
            f" — [$text-muted]press f to triage[/]{bias_line}"
        )

    def _render_finished(self) -> None:
        tele, run = self._telemetry, self._run
        if tele is None or run is None or self._finished_rendered:
            return
        self._finished_rendered = True
        if self._fast_timer is not None:
            self._fast_timer.stop()
        if self._slow_timer is not None:
            self._slow_timer.stop()
        self.add_class("-finished")
        self._cancel_armed_until = 0.0
        self._sync_cancel_strip()
        # final authoritative numbers come from the Run itself
        avg, _lo, _hi = tele.latency_stats()
        rate = tele.done / max(tele.elapsed(), 1.0) * 60
        est = run.estimate
        est_str = (
            f"${est.estimated_total_usd:.2f}" if est.estimated_total_usd is not None else "$ ?.??"
        )
        # Honest spend: real provider bill if the API reports one, else the
        # attempt-based burn estimate — never a fake "actual".
        billed = run.provider_reported_cost_usd
        burn = run.estimated_cost_from_attempts_usd
        if billed is not None:
            spend_str = f"BILL ${billed:.4f}"
        elif burn is not None:
            spend_str = f"BURN ~${burn:.4f}"
        else:
            spend_str = "BURN $ ?.??"
        status_word = {
            "completed": f"[$tele-ok b]{glyphs.DIAMOND} developed[/]",
            "cancelled": f"[$warning b]{glyphs.RETRY} cancelled — {run.success_count} frames kept[/]",
            "failed": f"[$tele-fail b]{glyphs.CROSS} run failed[/]",
        }.get(tele.finished_status or "completed", "")
        self.query_one("#summary", Static).update(
            f"{status_word}\n"
            f"{run.success_count} {glyphs.CHECK} · {run.failure_count} {glyphs.CROSS}"
            f" · {spend_str} vs EST {est_str}"
            f" · avg {rate:.1f} img/min"
            + (f" · {avg:.1f}s/frame" if avg is not None else "")
            + f"\n[$text-muted]{esc(str(run.output_dir))}[/]\n"
            f"[b]enter[/b] contact sheet · [b]o[/b] reveal · [b]esc[/b] studio"
        )
        self._repaint_slow_final()
        self._render_log()
        self.refresh_bindings()

    def _repaint_slow_final(self) -> None:
        self.query_one("#dk-progress", ProgressBar).update(
            total=max(1, self._telemetry.total), progress=self._telemetry.done
        )
        self.query_one("#cost", CostLedger).repaint()
        self.query_one("#throughput", ThroughputPanel).repaint()
        self.query_one("#coverage", CoveragePanel).repaint()
        self.query_one("#matrix", BatchMatrix).repaint_cells(self._tick)
        self._render_failures()
        self._render_header(time.monotonic())

    # ------------------------------------------------------------------ #
    # Matrix interactions
    # ------------------------------------------------------------------ #
    @on(BatchMatrix.CellHighlighted)
    def _cell_highlighted(self, _event: BatchMatrix.CellHighlighted) -> None:
        self._render_inspect()

    @on(BatchMatrix.CellActivated)
    def _cell_activated(self, event: BatchMatrix.CellActivated) -> None:
        run = self._run
        tele = self._telemetry
        if run is None:
            return
        if tele is not None and tele.is_finished:
            self.action_contact_sheet()
            return
        if event.index < len(run.plan):
            self.app.push_screen(PromptPeekModal(run.plan, event.index))

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def action_cancel(self) -> None:
        tele = self._telemetry
        if tele is None or tele.is_finished or getattr(self.app, "cancel_requested", False):
            return
        now = time.monotonic()
        if now <= self._cancel_armed_until:
            self._cancel_armed_until = 0.0
            self.app.request_cancel()
        else:
            self._cancel_armed_until = now + 3.0
        self._sync_cancel_strip()

    def _sync_cancel_strip(self) -> None:
        strip = self.query_one("#cancel-strip", Static)
        if getattr(self.app, "cancel_requested", False) and not (
            self._telemetry and self._telemetry.is_finished
        ):
            strip.update(
                f"{glyphs.CROSS} CANCELLING — in-flight items drain; queued frames are skipped."
            )
            strip.add_class("-visible")
        elif time.monotonic() <= self._cancel_armed_until:
            strip.update(
                f"{glyphs.WARN} press ctrl+x again within 3 s to cancel the run"
            )
            strip.add_class("-visible")
        else:
            strip.remove_class("-visible")

    def action_color_mode(self) -> None:
        mode = self.query_one("#matrix", BatchMatrix).cycle_color_mode()
        self.notify(f"matrix colour: {mode}", timeout=1.5)

    def action_log_filter(self) -> None:
        idx = LOG_FILTERS.index(self._log_filter)
        self._log_filter = LOG_FILTERS[(idx + 1) % len(LOG_FILTERS)]
        self.query_one("#log-card").border_subtitle = f"l filter · {self._log_filter}"
        self._render_log(reset=True)

    def action_triage(self) -> None:
        tele = self._telemetry
        if tele is None or tele.failed == 0:
            self.notify("No failures to triage.", timeout=2)
            return

        def _jump(index: Optional[int]) -> None:
            if index is not None:
                matrix = self.query_one("#matrix", BatchMatrix)
                matrix.set_cursor(index)
                matrix.focus()

        self.app.push_screen(FailureTriageModal(tele), _jump)

    def action_reveal(self) -> None:
        run = self._run
        reveal = getattr(self.app, "reveal_path", None)
        if run is not None and reveal is not None:
            reveal(run.output_dir)

    def action_contact_sheet(self) -> None:
        tele, run = self._telemetry, self._run
        if run is None or tele is None or not tele.is_finished:
            return
        self.app.open_contact_sheet(run.output_dir)

    def action_back(self) -> None:
        self.app.pop_screen()
