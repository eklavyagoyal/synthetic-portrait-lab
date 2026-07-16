"""Cross-run uniqueness — the registry of appearance signatures already produced.

To guarantee "zero repetitions overall" (not just within one batch), the planner
needs to know which individuals every *prior* run already generated. This module
scans the sibling run directories of an output location for the
``appearance_signature`` recorded in each ``metadata.jsonl`` and returns them as
a set the planner seeds its rejection sampling with.

Reading is fully defensive: missing files, malformed lines and legacy runs
without signatures are simply skipped, never fatal.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_seen_signatures(scan_root: str | Path) -> set[str]:
    """Collect every ``appearance_signature`` from ``<scan_root>/*/metadata.jsonl``.

    ``scan_root`` is the parent of the run directory (so the run about to start —
    which does not exist yet — naturally contributes nothing, while all sibling
    runs do). Returns an empty set if the directory is missing or holds no
    signatures (e.g. legacy runs generated before the appearance layer).
    """
    root = Path(scan_root).expanduser()
    seen: set[str] = set()
    if not root.is_dir():
        return seen
    for jsonl in root.glob("*/metadata.jsonl"):
        try:
            with jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sig = rec.get("appearance_signature")
                    if sig:
                        seen.add(sig)
        except OSError:
            continue
    return seen
