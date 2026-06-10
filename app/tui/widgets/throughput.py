"""ThroughputPanel — rate line, 60-second Sparkline, latency stats.

Sparkline data must be *assigned* to the reactive (in-place mutation does not
refresh — audited), and the widget has no default width, so CSS gives it one.
"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Sparkline, Static

from ..telemetry import RunTelemetry


class ThroughputPanel(Vertical):
    DEFAULT_CSS = """
    ThroughputPanel {
        width: 1fr;
        height: auto;

        #tp-rate { color: $secondary; text-style: bold; }
        #tp-spark { width: 1fr; height: 1; margin: 0 0 0 0; }
        #tp-spark > .sparkline--max-color { color: $accent; }
        #tp-spark > .sparkline--min-color { color: $accent 30%; }
        #tp-spark-caption { color: $text-muted; }
        #tp-latency, #tp-counters { color: $text-muted; }
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._telemetry: Optional[RunTelemetry] = None

    def compose(self) -> ComposeResult:
        yield Static("", id="tp-rate")
        yield Sparkline([], id="tp-spark")
        yield Static("last 60 s", id="tp-spark-caption")
        yield Static("", id="tp-latency")
        yield Static("", id="tp-counters")

    def attach(self, telemetry: RunTelemetry) -> None:
        self._telemetry = telemetry

    def repaint(self) -> None:
        tele = self._telemetry
        if tele is None:
            return
        per_min = tele.ewma_rate * 60.0
        rate = self.query_one("#tp-rate", Static)
        if tele.done < 2:
            rate.update("[$text-muted]warming up…[/]")
        else:
            rate.update(f"{per_min:.1f} img/min")

        self.query_one("#tp-spark", Sparkline).data = tele.bins_per_second(60)

        avg, lo, hi = tele.latency_stats()
        latency = self.query_one("#tp-latency", Static)
        if avg is None:
            latency.update("")
        else:
            latency.update(f"latency {avg:.1f}s avg · {lo:.1f} min · {hi:.1f} max")

        inflight = len(tele.running_indices)
        self.query_one("#tp-counters", Static).update(
            f"retries {tele.retries_total} · inflight {inflight}/{tele.concurrency}"
            f" · done {tele.done}/{tele.total}"
        )
