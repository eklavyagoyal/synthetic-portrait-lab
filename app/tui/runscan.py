"""Defensive scanner for past runs: ``<output_base>/*/manifest.json`` → records.

Every manifest is parsed inside its own try/except — a malformed file becomes
an ``unreadable`` record (shown dimmed in the Archive), never a crash. The
scan is filesystem-only and safe to run in a thread worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunRecord:
    """One row of the Archive screen."""

    run_id: str
    run_dir: Path
    when: str = ""               # ISO timestamp (created_at) or "" if unknown
    provider: str = ""
    model: str = ""
    model_display: str = ""
    planned: int = 0
    ok: int = 0
    fail: int = 0
    cost: Optional[float] = None
    cost_is_estimated: bool = False
    status: str = "unknown"
    error: Optional[str] = None  # set for unreadable manifests
    manifest: dict = field(default_factory=dict)

    @property
    def unreadable(self) -> bool:
        return self.error is not None


def _record_from_manifest(run_dir: Path, data: dict) -> RunRecord:
    summary = data.get("summary") or {}
    # Prefer the honest cost fields; fall back to the legacy field for old runs.
    billed = summary.get("provider_reported_cost_usd")
    burn = summary.get("estimated_cost_from_attempts_usd")
    if billed is not None:
        cost, cost_is_estimated = billed, False
    elif burn is not None:
        cost, cost_is_estimated = burn, True
    else:  # legacy manifests
        cost = summary.get("actual_total_usd")
        cost_is_estimated = bool(summary.get("actual_cost_includes_estimates", False))
    return RunRecord(
        run_id=str(data.get("run_id") or run_dir.name),
        run_dir=run_dir,
        when=str(data.get("created_at") or ""),
        provider=str(data.get("provider") or ""),
        model=str(data.get("model") or ""),
        model_display=str(data.get("model_display_name") or data.get("model") or ""),
        planned=int(summary.get("planned") or 0),
        ok=int(summary.get("succeeded") or 0),
        fail=int(summary.get("failed") or 0),
        cost=cost,
        cost_is_estimated=cost_is_estimated,
        status=str(data.get("status") or "unknown"),
        manifest=data,
    )


def scan_runs(base_dir: str | Path) -> list[RunRecord]:
    """Scan ``base_dir`` for run directories, newest first."""
    base = Path(base_dir).expanduser()
    records: list[RunRecord] = []
    if not base.is_dir():
        return records
    try:
        children = sorted(base.iterdir())
    except OSError:
        return records
    for child in children:
        manifest_path = child / "manifest.json"
        if not child.is_dir() or not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("manifest is not a JSON object")
            records.append(_record_from_manifest(child, data))
        except Exception as exc:  # noqa: BLE001 - a bad manifest must never crash the scan
            records.append(
                RunRecord(run_id=child.name, run_dir=child, error=f"{type(exc).__name__}: {exc}")
            )
    records.sort(key=lambda r: (r.when, r.run_id), reverse=True)
    return records


def load_sheet(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """Load a run's manifest header + ``metadata.jsonl`` items (ground truth,
    including failures). Raises on a missing/unreadable directory; individual
    bad JSONL lines are skipped."""
    run_dir = Path(run_dir).expanduser()
    header: dict = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                header = data
        except Exception:  # noqa: BLE001
            header = {}
    items: list[dict] = []
    jsonl_path = run_dir / "metadata.jsonl"
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    items.append(record)
            except Exception:  # noqa: BLE001
                continue
    # latest write wins per id (retries / re-runs append)
    by_id: dict[str, dict] = {}
    for record in items:
        by_id[str(record.get("id"))] = record
    ordered = sorted(by_id.values(), key=lambda r: str(r.get("id")))
    return header, ordered
