"""Writes per-image metadata (JSONL + CSV) and the run manifest.

Design goals:

* **Crash-safe stream** — every result is appended to ``metadata.jsonl`` the
  moment it lands, so an interrupted run still has a complete record of what
  finished (successes *and* failures).
* **Uniform schema** — success and failure rows share the same key-set so the
  CSV is well-formed.
* **Auditable manifest** — ``manifest.json`` is valid JSON containing the full
  request, the estimate and the final summary. API keys never appear anywhere.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import GenerationResult, Run
from .storage import Storage


class MetadataWriter:
    """Persists metadata for a single run directory."""

    def __init__(self, storage: Storage, run_dir: str | Path, save_prompt: bool = True):
        self.storage = storage
        self.run_dir = Path(run_dir)
        self.save_prompt = save_prompt
        self._jsonl = storage.metadata_jsonl_path(run_dir)
        self._csv = storage.metadata_csv_path(run_dir)

    # -- streaming per-item ----------------------------------------------- #
    def append_result(self, result: GenerationResult) -> None:
        """Append one result to the JSONL stream (newline-delimited JSON)."""
        record = result.to_record(include_prompt=self.save_prompt)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- batch CSV (rewritten from the full result set) ------------------- #
    def write_csv(self, results: Iterable[GenerationResult]) -> Path:
        results = list(results)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Field order comes from the canonical record; prompt last (or omitted).
        sample = GenerationResult(
            id="_", provider="_", model="_", age_bucket="_",
            gender_bucket="_", ethnicity_bucket="_", variation_level=0, size="_",
        )
        fieldnames = list(sample.to_record(include_prompt=self.save_prompt).keys())
        with self._csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_record(include_prompt=self.save_prompt))
        return self._csv

    # -- manifest --------------------------------------------------------- #
    def write_manifest(self, run: Run) -> Path:
        path = self.storage.manifest_path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(run.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def rewrite_jsonl(self, results: Iterable[GenerationResult]) -> Path:
        """Rebuild the JSONL file from scratch (used by headless re-runs/tests)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._jsonl.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(
                    json.dumps(
                        r.to_record(include_prompt=self.save_prompt), ensure_ascii=False
                    )
                    + "\n"
                )
        return self._jsonl

    def finalize(self, run: Run) -> None:
        """Write CSV + manifest at the end of a run."""
        self.write_csv(run.results)
        self.write_manifest(run)
