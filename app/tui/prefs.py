"""Session preferences — ``~/.portrait_studio_tui.json``.

Stores only cosmetic/compose state (theme, last model, batch settings, bucket
selection, thumbnail size). Never secrets. Loading is fully defensive: a
missing or corrupt file silently yields factory defaults; saved values are
validated against the live config (unknown buckets dropped, unknown model
falls back) before use. Writes are atomic (tmp + rename).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PREFS_PATH = Path(
    os.environ.get("PORTRAIT_TUI_PREFS", "~/.portrait_studio_tui.json")
).expanduser()


@dataclass
class Prefs:
    theme: str = "darkroom"
    model_key: str = ""           # "provider::model_id"
    batch_size: int = 8
    variation: int = 0
    quality: str = "medium"       # low | medium | high | auto
    head_height_pct: int = 60     # framing: head share of image height
    distribution: str = "even"
    concurrency: int = 2
    prefix: str = "portrait"
    ages: Optional[list[str]] = None
    genders: Optional[list[str]] = None
    ethnicities: Optional[list[str]] = None
    retry_failed: bool = True
    max_retries: int = 3
    save_prompt: bool = True
    thumb_size: str = "M"         # S | M | L
    seen_welcome: bool = False
    face_crop: bool = False
    mask_print: bool = False
    mask_width_mm: float = 187.0
    mask_height_mm: float = 245.0
    mask_eye_inner_gap_mm: float = 40.0
    mask_eye_width_mm: float = 38.0
    mask_eye_height_mm: float = 18.0
    mask_eye_center_top_mm: float = 103.0
    mask_nose_width_mm: float = 40.0
    mask_nose_length_mm: float = 30.0
    mask_overlap_mm: float = 1.5
    mask_dpi: int = 300
    diversify: bool = True        # per-image appearance variation + uniqueness
    modality: str = "rgb"         # "rgb" (face) | "ir" (near-infrared iris)
    # IR iris realism knobs (opt-in; only used when modality == "ir")
    ir_occlusion: bool = False
    ir_off_gaze: bool = False
    ir_lenses: bool = False
    ir_conditions: bool = False
    ir_glasses: bool = False
    ir_makeup: bool = False
    extra: dict = field(default_factory=dict)


def load(path: Optional[Path] = None) -> Prefs:
    """Load preferences; any problem at all returns factory defaults."""
    path = path or PREFS_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return Prefs()
    except Exception:  # noqa: BLE001 - missing/corrupt prefs are normal
        return Prefs()
    prefs = Prefs()
    for key, value in raw.items():
        if not hasattr(prefs, key):
            continue
        current = getattr(prefs, key)
        numeric_compatible = isinstance(current, float) and isinstance(value, (int, float))
        if current is None or isinstance(value, type(current)) or numeric_compatible or key in (
            "ages", "genders", "ethnicities",
        ):
            setattr(prefs, key, value)
    if prefs.thumb_size not in ("S", "M", "L"):
        prefs.thumb_size = "M"
    prefs.batch_size = _clamp_int(prefs.batch_size, 1, 100_000, 8)
    prefs.variation = _clamp_int(prefs.variation, 0, 3, 0)
    prefs.head_height_pct = _clamp_int(prefs.head_height_pct, 20, 90, 60)
    prefs.concurrency = _clamp_int(prefs.concurrency, 1, 32, 2)
    prefs.max_retries = _clamp_int(prefs.max_retries, 0, 10, 3)
    prefs.mask_width_mm = _clamp_float(prefs.mask_width_mm, 100.0, 300.0, 187.0)
    prefs.mask_height_mm = _clamp_float(prefs.mask_height_mm, 150.0, 400.0, 245.0)
    prefs.mask_eye_inner_gap_mm = _clamp_float(prefs.mask_eye_inner_gap_mm, 15.0, 100.0, 40.0)
    prefs.mask_eye_width_mm = _clamp_float(prefs.mask_eye_width_mm, 15.0, 70.0, 38.0)
    prefs.mask_eye_height_mm = _clamp_float(prefs.mask_eye_height_mm, 8.0, 40.0, 18.0)
    prefs.mask_eye_center_top_mm = _clamp_float(prefs.mask_eye_center_top_mm, 40.0, 180.0, 103.0)
    prefs.mask_nose_width_mm = _clamp_float(prefs.mask_nose_width_mm, 20.0, 80.0, 40.0)
    prefs.mask_nose_length_mm = _clamp_float(prefs.mask_nose_length_mm, 15.0, 80.0, 30.0)
    prefs.mask_overlap_mm = _clamp_float(prefs.mask_overlap_mm, 0.0, 5.0, 1.5)
    prefs.mask_dpi = _clamp_int(prefs.mask_dpi, 150, 600, 300)
    if prefs.distribution not in ("even", "random", "weighted"):
        prefs.distribution = "even"
    if prefs.quality not in ("low", "medium", "high", "auto"):
        prefs.quality = "medium"
    if prefs.modality not in ("rgb", "ir"):
        prefs.modality = "rgb"
    return prefs


def save(prefs: Prefs, path: Optional[Path] = None) -> bool:
    """Atomically persist preferences. Returns False on any failure."""
    path = path or PREFS_PATH
    try:
        payload = json.dumps(asdict(prefs), ensure_ascii=False, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        return True
    except Exception:  # noqa: BLE001 - prefs are best-effort
        return False


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def validated_buckets(saved: Optional[list], allowed: list[str]) -> Optional[list[str]]:
    """Filter a saved bucket selection against the live config; ``None`` (or an
    empty intersection) means "use the app default"."""
    if not isinstance(saved, list):
        return None
    valid = [b for b in saved if isinstance(b, str) and b in allowed]
    return valid or None
