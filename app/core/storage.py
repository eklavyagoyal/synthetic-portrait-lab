"""Output paths, run directories, filenames and image writing.

Layout produced for one run::

    <base>/run_YYYY_MM_DD_HHMM/
        images/
            portrait_000001.png
            ...
        metadata.jsonl
        metadata.csv
        manifest.json
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_RUN_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def new_run_id(prefix: str = "run", now: datetime | None = None) -> str:
    """Timestamped, filesystem-safe run id, e.g. ``run_2026_06_09_1430``."""
    ts = (now or datetime.now()).strftime("%Y_%m_%d_%H%M")
    return f"{prefix}_{ts}"


class Storage:
    """Filesystem layout for runs. One ``Storage`` per base output directory."""

    def __init__(self, base_output_dir: str | Path = "./outputs"):
        self.base = Path(base_output_dir).expanduser()

    # -- directory layout ------------------------------------------------- #
    def resolve_run_dir(self, run_id: str, explicit_dir: str | Path | None = None) -> Path:
        """Decide where a run's files live.

        If ``explicit_dir`` is given it is used verbatim; otherwise the run dir is
        ``<base>/<run_id>``. The path is *not* created here — call :meth:`prepare`.
        """
        if explicit_dir:
            return Path(explicit_dir).expanduser()
        safe = _RUN_ID_RE.sub("_", run_id)
        return self.base / safe

    def prepare(self, run_dir: str | Path) -> Path:
        """Create the run directory and its ``images/`` subdirectory."""
        run_dir = Path(run_dir).expanduser()
        (run_dir / "images").mkdir(parents=True, exist_ok=True)
        return run_dir

    # -- well-known file locations ---------------------------------------- #
    def images_dir(self, run_dir: str | Path) -> Path:
        return Path(run_dir) / "images"

    def image_path(self, run_dir: str | Path, filename: str) -> Path:
        return self.images_dir(run_dir) / filename

    def metadata_jsonl_path(self, run_dir: str | Path) -> Path:
        return Path(run_dir) / "metadata.jsonl"

    def metadata_csv_path(self, run_dir: str | Path) -> Path:
        return Path(run_dir) / "metadata.csv"

    def manifest_path(self, run_dir: str | Path) -> Path:
        return Path(run_dir) / "manifest.json"

    # -- image writing ---------------------------------------------------- #
    def save_image(self, run_dir: str | Path, filename: str, data: bytes) -> Path:
        path = self.image_path(run_dir, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
