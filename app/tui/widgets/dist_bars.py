"""DistBars — per-bucket histograms of a *planned* distribution (compose-time
preview, EXPOSE modal summary, contact-sheet distribution block)."""

from __future__ import annotations

from collections import Counter

from textual.widgets import Static

from .. import glyphs, labels

BAR_MAX = 10


class DistBars(Static):
    DEFAULT_CSS = """
    DistBars { width: 1fr; height: auto; }
    """

    def update_from_counts(self, by_dim: dict[str, Counter]) -> None:
        """``by_dim`` maps dimension name → Counter(bucket → planned count)."""
        lines: list[str] = []
        top = max(
            [c for counter in by_dim.values() for c in counter.values()] or [1]
        )
        for dim in labels.DIMENSIONS:
            counter = by_dim.get(dim)
            if not counter:
                continue
            var = labels.DIM_VAR[dim]
            parts: list[str] = [f"[$text-muted]{dim[:6].ljust(6)}[/]"]
            for bucket in sorted(counter, key=lambda b: labels.short(b)):
                count = counter[bucket]
                width = max(1, round(BAR_MAX * count / top))
                parts.append(f"[{var}]{glyphs.BAR * width}[/] {count}")
            lines.append("  ".join(parts))
        self.update("\n".join(lines))
