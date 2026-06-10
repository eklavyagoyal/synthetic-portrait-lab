"""CoveragePanel — planned-vs-landed bars per demographic bucket.

Denominators come from the *plan* (fixed at run start); ``█`` = succeeded,
``▒`` = failed, ``░`` = still pending. ``✓`` marks a bucket at parity.
"""

from __future__ import annotations

from typing import Optional

from textual.widgets import Static

from .. import glyphs, labels
from ..telemetry import RunTelemetry

BAR_WIDTH = 18


class CoveragePanel(Static):
    DEFAULT_CSS = """
    CoveragePanel { width: 1fr; height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._telemetry: Optional[RunTelemetry] = None

    def attach(self, telemetry: RunTelemetry) -> None:
        self._telemetry = telemetry

    def repaint(self) -> None:
        tele = self._telemetry
        if tele is None:
            self.update("")
            return
        lines: list[str] = []
        label_width = max(
            [len(labels.short(b)) for c in tele.planned_by_bucket.values() for b in c]
            or [6]
        )
        for dim in labels.DIMENSIONS:
            planned = tele.planned_by_bucket.get(dim)
            if not planned:
                continue
            var = labels.DIM_VAR[dim]
            lines.append(f"[{var} b]{dim.upper()}[/]")
            for bucket in sorted(planned, key=lambda b: labels.short(b)):
                total = planned[bucket]
                ok = tele.landed_by_bucket[dim].get(bucket, 0)
                bad = tele.failed_by_bucket[dim].get(bucket, 0)
                ok_w = round(BAR_WIDTH * ok / total) if total else 0
                bad_w = round(BAR_WIDTH * bad / total) if total else 0
                bad_w = min(bad_w, BAR_WIDTH - ok_w)
                rest = BAR_WIDTH - ok_w - bad_w
                bar = (
                    f"[{var}]{glyphs.BAR * ok_w}[/]"
                    f"[$tele-fail]{glyphs.BAR_FAIL * bad_w}[/]"
                    f"[$tele-pending]{glyphs.BAR_EMPTY * rest}[/]"
                )
                count = f"{ok}" + (f"+{bad}{glyphs.CROSS}" if bad else "")
                parity = f" [$tele-ok]{glyphs.CHECK}[/]" if ok == total else ""
                name = labels.short(bucket).ljust(label_width)
                lines.append(f"[$text-muted]{name}[/] {bar} {count}/{total}{parity}")
        self.update("\n".join(lines))
