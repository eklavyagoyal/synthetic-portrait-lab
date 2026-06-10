"""LaneBoard — one row per concurrency lane, showing the item in flight.

A greedy visualization (events do not carry real task identity): ITEM_STARTED
claims the first free slot, terminal events free it. Hidden at concurrency 1.
"""

from __future__ import annotations

import time
from typing import Optional

from textual.widgets import Static

from .. import glyphs, labels
from ..telemetry import CellState, RunTelemetry, fmt_mmss

MAX_ROWS = 10


class LaneBoard(Static):
    DEFAULT_CSS = """
    LaneBoard { width: 1fr; height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._telemetry: Optional[RunTelemetry] = None
        self.tooltip = (
            "Concurrency lanes are a greedy visualization of in-flight items, "
            "not real task identity."
        )

    def attach(self, telemetry: RunTelemetry) -> None:
        self._telemetry = telemetry

    def repaint(self, now: Optional[float] = None) -> None:
        tele = self._telemetry
        if tele is None:
            self.update("")
            return
        now = time.monotonic() if now is None else now
        mean_latency = (
            sum(tele.latencies) / len(tele.latencies) if tele.latencies else None
        )
        lines: list[str] = []
        for lane, occupant in enumerate(tele.lanes[:MAX_ROWS]):
            label = f"[$text-muted]L{lane + 1}[/] "
            if occupant is None:
                lines.append(f"{label}[$text-disabled]─ idle[/]")
                continue
            meta = tele.items[occupant] if occupant < len(tele.items) else None
            state = tele.states[occupant] if occupant < len(tele.states) else None
            glyph = (
                f"[$tele-retry]{glyphs.RETRY}[/]"
                if state == CellState.RETRY
                else f"[$tele-running]{glyphs.DOT_HALF}[/]"
            )
            started = tele.lane_started.get(occupant)
            elapsed = fmt_mmss(now - started) if started is not None else "--:--"
            parts = [label + glyph]
            if meta is not None:
                parts.append(f"[b]{meta.item_id}[/]")
                parts.append(labels.triple_chips(meta.age, meta.gender, meta.ethnicity))
            parts.append(f"[$text-muted]{elapsed}[/]")
            if state == CellState.RETRY and occupant < len(tele.retries):
                parts.append(f"[$tele-retry]retry {tele.retries[occupant]}[/]")
            if (
                mean_latency is not None
                and started is not None
                and (now - started) > 3 * mean_latency
                and mean_latency > 0.5
            ):
                parts.append(f"[$warning]slow[/]")
            lines.append("  ".join(parts))
        extra = len(tele.lanes) - MAX_ROWS
        if extra > 0:
            lines.append(f"[$text-muted]+{extra} more lanes[/]")
        self.update("\n".join(lines))
