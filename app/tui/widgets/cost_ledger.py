"""CostLedger — EST / BURN / BILL rows plus a Δ-vs-estimate line.

Three deliberately distinct numbers:

* **EST** — the pre-run plan: per-image price × planned outputs.
* **BURN** — estimated spend from *billable attempts* (retries included), so it
  can exceed EST. Always an estimate; rendered with a leading ``~``.
* **BILL** — real provider-reported spend, shown ONLY when the API actually
  returns a per-request USD amount; otherwise ``unavailable`` (never a fake
  number echoed from the estimate).

An unknown estimate renders ``$ ?.??`` (never ``0.00``).
"""

from __future__ import annotations

from typing import Optional

from textual.widgets import Static

from app.core.models import CostEstimate

from ..telemetry import RunTelemetry


class CostLedger(Static):
    DEFAULT_CSS = """
    CostLedger { width: 1fr; height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._telemetry: Optional[RunTelemetry] = None
        self._estimate: Optional[CostEstimate] = None

    def attach(self, telemetry: RunTelemetry, estimate: CostEstimate) -> None:
        self._telemetry = telemetry
        self._estimate = estimate

    def repaint(self) -> None:
        tele, est = self._telemetry, self._estimate
        if tele is None or est is None:
            self.update("")
            return
        lines: list[str] = []

        # EST — the pre-run plan.
        if est.pricing_available and est.estimated_total_usd is not None:
            per_image = est.price_per_image_usd or 0.0
            quality = f" {est.quality}" if est.quality else ""
            lines.append(
                f"[$tele-est b]EST [/]  [$tele-est]$ {est.estimated_total_usd:.2f}[/]"
                f"   [$text-muted]{est.total_count} imgs × ${per_image:.4f}{quality}[/]"
            )
        else:
            per_image = None
            lines.append(
                "[$warning b]EST [/]  [$warning]$ ?.??[/]   [$text-muted]pricing unavailable[/]"
            )

        # BURN — estimated spend from billable attempts (retries included).
        burn = tele.burn_estimate
        attempts = tele.api_attempts
        if burn is not None:
            elapsed_min = tele.elapsed() / 60.0
            rate = (burn / elapsed_min) if elapsed_min > 0.05 else None
            rate_str = f" · ${rate:.2f}/min" if rate is not None else ""
            retry_str = f" (+{tele.retries_total} retry)" if tele.retries_total else ""
            lines.append(
                f"[$tele-act b]BURN[/]  [$tele-act]~$ {burn:.4f}[/]"
                f"   [$text-muted]{attempts} attempts{retry_str}{rate_str}[/]"
            )
        else:
            lines.append(
                f"[$tele-act b]BURN[/]  [$tele-act]~$ ?.??[/]   "
                f"[$text-muted]{attempts} attempts · price unknown[/]"
            )

        # BILL — real provider-reported spend, only if the API returns one.
        if tele.provider_reported_any:
            lines.append(
                f"[$tele-ok b]BILL[/]  [$tele-ok]$ {tele.provider_cost:.4f}[/]"
                "   [$text-muted]provider-reported[/]"
            )
        else:
            lines.append(
                "[$text-muted b]BILL[/]  [$text-muted]unavailable — provider reports no per-request $[/]"
            )

        # Δ — burn vs the pre-run estimate (where retries/extra attempts surface).
        if per_image is not None and est.estimated_total_usd is not None and burn is not None:
            if tele.is_finished:
                delta = burn - est.estimated_total_usd
                label, detail = "Δ final", "burn vs estimate"
            else:
                projected = (attempts + tele.remaining) * per_image
                delta = projected - est.estimated_total_usd
                label, detail = "Δ proj", f"({tele.remaining} left)"
            if abs(delta) < 0.005:
                lines.append(f"[$text-muted]{label}  on estimate[/]")
            else:
                var = "$tele-ok" if delta < 0 else "$warning"
                sign = "−" if delta < 0 else "+"
                lines.append(
                    f"[{var}]{label} {sign}${abs(delta):.4f}[/] "
                    f"[$text-muted]{'under' if delta < 0 else 'over'} estimate {detail}[/]"
                )
        self.update("\n".join(lines))
