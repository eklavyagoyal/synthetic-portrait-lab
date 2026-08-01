"""Deterministic portrait-to-mask print packs.

This module intentionally does *not* call an image model.  It preserves the paid
portrait byte-for-byte, normalizes the already-standardized head framing onto a
measured mask surface, divides that surface into overlapping physical panels and
exports printable A4 assets at an exact DPI.

The first template is parametric rather than a claim of scan-level accuracy.  A
separate calibration page makes the remaining physical mismatch measurable, so
the template can be corrected after one test print without regenerating faces.
"""

from __future__ import annotations

import io
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Iterable

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageFont,
)

from .face_crop import _sample_background_color
from .models import MaskPrintOptions

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
_YUNET_MODEL_NAME = "face_detection_yunet_2026may.onnx"
_YUNET_MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


class MaskPrintError(RuntimeError):
    """A deterministic mask export could not be produced."""


@dataclass(frozen=True)
class MaskPrintPack:
    """Files created for one source portrait."""

    output_dir: Path
    preview_path: Path
    print_pdf_path: Path
    calibration_pdf_path: Path
    cutlines_svg_path: Path
    page_paths: tuple[Path, ...]
    metadata_path: Path

    def relative_record(self, run_dir: str | Path) -> dict[str, object]:
        """Metadata fields relative to a run directory."""
        root = Path(run_dir)

        def rel(path: Path) -> str:
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        return {
            "mask_preview_filename": rel(self.preview_path),
            "mask_print_pdf": rel(self.print_pdf_path),
            "mask_calibration_pdf": rel(self.calibration_pdf_path),
            "mask_cutlines_svg": rel(self.cutlines_svg_path),
            "mask_print_pages": [rel(path) for path in self.page_paths],
        }


@dataclass(frozen=True)
class _Panel:
    name: str
    label: str
    mask: Image.Image
    core_mask: Image.Image


@dataclass(frozen=True)
class FaceLandmarks:
    """Scored five-point face geometry returned by the bundled YuNet model.

    ``left_eye`` / ``right_eye`` refer to the left/right side of the image,
    which keeps the alignment math independent of anatomical naming.
    """

    face_bbox: tuple[float, float, float, float]
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]
    confidence: float
    detected_faces: int

    def metadata(self) -> dict[str, object]:
        return {
            "face_bbox": [round(value, 2) for value in self.face_bbox],
            "left_eye": [round(value, 2) for value in self.left_eye],
            "right_eye": [round(value, 2) for value in self.right_eye],
            "nose_tip": [round(value, 2) for value in self.nose_tip],
            "left_mouth": [round(value, 2) for value in self.left_mouth],
            "right_mouth": [round(value, 2) for value in self.right_mouth],
            "confidence": round(self.confidence, 5),
            "detected_faces": self.detected_faces,
        }


def _px(mm_value: float, dpi: int) -> int:
    return max(1, int(round(mm_value / 25.4 * dpi)))


def _mm(px_value: int, dpi: int) -> float:
    return px_value / dpi * 25.4


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 fallback
        return ImageFont.load_default()


def _sample_cubic(
    start: tuple[float, float],
    control_a: tuple[float, float],
    control_b: tuple[float, float],
    end: tuple[float, float],
    *,
    steps: int = 18,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        omt = 1.0 - t
        x = (
            omt**3 * start[0]
            + 3 * omt**2 * t * control_a[0]
            + 3 * omt * t**2 * control_b[0]
            + t**3 * end[0]
        )
        y = (
            omt**3 * start[1]
            + 3 * omt**2 * t * control_a[1]
            + 3 * omt * t**2 * control_b[1]
            + t**3 * end[1]
        )
        points.append((x, y))
    return points


def _outer_outline_mm(options: MaskPrintOptions) -> list[tuple[float, float]]:
    width = options.width_mm
    height = options.height_mm
    center = width / 2.0
    points = [(center, 0.0)]
    segments = (
        ((center, 0.0), (width * 0.22, -1.0), (width * 0.055, height * 0.10), (width * 0.032, height * 0.335)),
        ((width * 0.032, height * 0.335), (width * 0.005, height * 0.56), (width * 0.105, height * 0.80), (width * 0.29, height * 0.93)),
        ((width * 0.29, height * 0.93), (width * 0.36, height * 0.98), (width * 0.43, height), (center, height)),
        ((center, height), (width * 0.57, height), (width * 0.64, height * 0.98), (width * 0.71, height * 0.93)),
        ((width * 0.71, height * 0.93), (width * 0.895, height * 0.80), (width * 0.995, height * 0.56), (width * 0.968, height * 0.335)),
        ((width * 0.968, height * 0.335), (width * 0.945, height * 0.10), (width * 0.78, -1.0), (center, 0.0)),
    )
    for segment in segments:
        points.extend(_sample_cubic(*segment))
    return points


def _eye_outline_mm(
    options: MaskPrintOptions,
    *,
    side: str,
) -> list[tuple[float, float]]:
    center_x = options.width_mm / 2.0
    half_gap = options.eye_inner_gap_mm / 2.0
    width = options.eye_opening_width_mm
    half_height = options.eye_opening_height_mm / 2.0
    y = options.eye_center_from_top_mm

    if side == "left":
        inner_x = center_x - half_gap
        outer_x = inner_x - width
    else:
        inner_x = center_x + half_gap
        outer_x = inner_x + width

    start = (outer_x, y)
    end = (inner_x, y)
    direction = 1.0 if inner_x > outer_x else -1.0
    upper = _sample_cubic(
        start,
        (outer_x + direction * width * 0.28, y - half_height),
        (inner_x - direction * width * 0.25, y - half_height),
        end,
        steps=16,
    )
    lower = _sample_cubic(
        end,
        (inner_x - direction * width * 0.26, y + half_height),
        (outer_x + direction * width * 0.25, y + half_height),
        start,
        steps=16,
    )
    return [start, *upper, *lower]


def _to_pixels(
    points_mm: Iterable[tuple[float, float]],
    *,
    dpi: int,
) -> list[tuple[int, int]]:
    return [(_px(x, dpi), _px(y, dpi)) for x, y in points_mm]


def _face_surface_mask(options: MaskPrintOptions) -> Image.Image:
    size = (_px(options.width_mm, options.dpi), _px(options.height_mm, options.dpi))
    outer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(outer)
    draw.polygon(_to_pixels(_outer_outline_mm(options), dpi=options.dpi), fill=255)

    for side in ("left", "right"):
        draw.polygon(
            _to_pixels(_eye_outline_mm(options, side=side), dpi=options.dpi),
            fill=0,
        )
    return outer


@lru_cache(maxsize=1)
def _yunet_model_path() -> Path:
    """Return the verified bundled detector model.

    A corrupt or missing model is a hard error.  Silently falling back to the
    former hair/eyebrow heuristic would recreate exactly the failure this gate
    is intended to prevent.
    """
    path = Path(__file__).resolve().parent / "assets" / _YUNET_MODEL_NAME
    if not path.is_file():
        raise MaskPrintError(
            f"bundled face-landmark model is missing: {path}. Reinstall the project."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _YUNET_MODEL_SHA256:
        raise MaskPrintError(
            "bundled face-landmark model failed its integrity check; reinstall the project."
        )
    return path


def _detect_face_landmarks(source: Image.Image) -> FaceLandmarks:
    """Detect one frontal face and five landmarks with local YuNet inference."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise MaskPrintError(
            "reliable mask alignment needs opencv-python-headless and numpy; "
            "reinstall the project dependencies."
        ) from exc

    width, height = source.size
    if min(width, height) < 256:
        raise MaskPrintError(
            f"portrait is too small for reliable mask alignment ({width}x{height}); "
            "use at least 256 pixels on the shorter side."
        )

    # Bound inference cost while retaining ample face detail.  Coordinates are
    # converted back to source pixels before validation and alignment.
    inference_scale = min(1.0, 1024.0 / max(width, height))
    rgb = np.asarray(source.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inference_size = (
        max(1, int(round(width * inference_scale))),
        max(1, int(round(height * inference_scale))),
    )
    if inference_size != (width, height):
        bgr = cv2.resize(bgr, inference_size, interpolation=cv2.INTER_AREA)

    try:
        detector = cv2.FaceDetectorYN.create(
            str(_yunet_model_path()),
            "",
            inference_size,
            0.70,
            0.30,
            5000,
        )
        _, raw_faces = detector.detect(bgr)
    except Exception as exc:  # noqa: BLE001 - normalize backend-specific errors
        raise MaskPrintError(f"face-landmark detector failed: {exc}") from exc

    if raw_faces is None or len(raw_faces) == 0:
        raise MaskPrintError(
            "no face was detected; mask print export requires one clear, frontal face."
        )

    rows = sorted(raw_faces, key=lambda row: float(row[14]), reverse=True)
    primary = max(rows, key=lambda row: float(row[2] * row[3]) * float(row[14]))
    primary_area = float(primary[2] * primary[3])
    significant = [
        row
        for row in rows
        if float(row[14]) >= 0.80
        and float(row[2] * row[3]) >= max(1.0, primary_area * 0.12)
    ]
    if len(significant) != 1:
        raise MaskPrintError(
            f"detected {len(significant)} significant faces; mask print export "
            "requires exactly one person."
        )

    values = [float(value) / inference_scale for value in primary[:14]]
    confidence = float(primary[14])
    x, y, face_width, face_height = values[:4]
    eye_a = (values[4], values[5])
    eye_b = (values[6], values[7])
    mouth_a = (values[10], values[11])
    mouth_b = (values[12], values[13])
    left_eye, right_eye = sorted((eye_a, eye_b), key=lambda point: point[0])
    left_mouth, right_mouth = sorted((mouth_a, mouth_b), key=lambda point: point[0])

    landmarks = FaceLandmarks(
        face_bbox=(x, y, face_width, face_height),
        left_eye=left_eye,
        right_eye=right_eye,
        nose_tip=(values[8], values[9]),
        left_mouth=left_mouth,
        right_mouth=right_mouth,
        confidence=confidence,
        detected_faces=len(significant),
    )
    _validate_face_landmarks(landmarks, source_size=source.size)
    return landmarks


def _validate_face_landmarks(
    landmarks: FaceLandmarks,
    *,
    source_size: tuple[int, int],
) -> dict[str, float | str]:
    """Reject ambiguous, off-axis or anatomically implausible detections."""
    image_width, image_height = source_size
    x, y, face_width, face_height = landmarks.face_bbox
    left_eye = landmarks.left_eye
    right_eye = landmarks.right_eye
    nose = landmarks.nose_tip
    left_mouth = landmarks.left_mouth
    right_mouth = landmarks.right_mouth

    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_distance = math.hypot(eye_dx, eye_dy)
    mean_eye_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    mouth_mid_x = (left_mouth[0] + right_mouth[0]) / 2.0
    mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
    roll_degrees = math.degrees(math.atan2(eye_dy, eye_dx)) if eye_dx else 90.0
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(landmarks.confidence >= 0.88, f"low detector confidence {landmarks.confidence:.3f}")
    require(face_width > 0 and face_height > 0, "invalid face bounding box")
    require(0.20 <= face_width / image_width <= 0.75, "implausible face width")
    require(0.25 <= face_height / image_height <= 0.75, "implausible face height")
    require(abs((x + face_width / 2.0) - image_width / 2.0) <= image_width * 0.16, "face is not centered")
    require(eye_distance >= image_width * 0.10, "eyes are too close for reliable alignment")
    require(0.30 <= eye_distance / max(1.0, face_width) <= 0.62, "implausible eye spacing")
    require(abs(roll_degrees) <= 8.0, f"head roll is too large ({roll_degrees:.1f} degrees)")
    require(y + face_height * 0.20 <= mean_eye_y <= y + face_height * 0.55, "eyes are outside the expected face region")
    require(0.25 <= (nose[1] - mean_eye_y) / max(1.0, eye_distance) <= 0.95, "nose vertical position is implausible")
    require(0.20 <= (mouth_mid_y - nose[1]) / max(1.0, eye_distance) <= 0.85, "mouth vertical position is implausible")
    require(abs(nose[0] - eye_mid_x) <= eye_distance * 0.18, "nose is too far off the facial centerline")
    require(abs(mouth_mid_x - eye_mid_x) <= eye_distance * 0.22, "mouth is too far off the facial centerline")
    require(0.45 <= (right_mouth[0] - left_mouth[0]) / max(1.0, eye_distance) <= 1.20, "mouth width is implausible")
    require(abs(right_mouth[1] - left_mouth[1]) <= eye_distance * 0.14, "mouth tilt is too large")
    left_nose_span = nose[0] - left_eye[0]
    right_nose_span = right_eye[0] - nose[0]
    require(left_nose_span > 0 and right_nose_span > 0, "nose is not between the eyes")
    if left_nose_span > 0 and right_nose_span > 0:
        yaw_ratio = left_nose_span / right_nose_span
        require(0.62 <= yaw_ratio <= 1.62, f"face yaw is too large (balance {yaw_ratio:.2f})")
    else:
        yaw_ratio = 0.0

    if failures:
        raise MaskPrintError("landmark quality check failed: " + "; ".join(failures) + ".")

    return {
        "status": "passed",
        "detector": "YuNet face_detection_yunet_2026may",
        "confidence": round(landmarks.confidence, 5),
        "interocular_distance_px": round(eye_distance, 2),
        "interocular_to_face_width": round(eye_distance / face_width, 5),
        "roll_degrees": round(roll_degrees, 3),
        "yaw_balance": round(yaw_ratio, 5),
    }


def _transform_point(
    point: tuple[float, float],
    matrix: list[list[float]],
) -> tuple[float, float]:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2],
    )


def _normalize_portrait(
    image_bytes: bytes,
    options: MaskPrintOptions,
) -> tuple[Image.Image, dict[str, object]]:
    try:
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - normalize the public error
        raise MaskPrintError(f"portrait image could not be decoded: {exc}") from exc

    landmarks = _detect_face_landmarks(source)
    quality = _validate_face_landmarks(landmarks, source_size=source.size)
    left_eye = landmarks.left_eye
    right_eye = landmarks.right_eye
    target_width_px = _px(options.width_mm, options.dpi)
    target_height_px = _px(options.height_mm, options.dpi)
    target_left_eye_x = _px(
        options.width_mm / 2
        - options.eye_inner_gap_mm / 2
        - options.eye_opening_width_mm / 2,
        options.dpi,
    )
    target_right_eye_x = _px(
        options.width_mm / 2
        + options.eye_inner_gap_mm / 2
        + options.eye_opening_width_mm / 2,
        options.dpi,
    )
    target_eye_y = _px(options.eye_center_from_top_mm, options.dpi)

    source_eye_dx = right_eye[0] - left_eye[0]
    source_eye_dy = right_eye[1] - left_eye[1]
    source_eye_separation = math.hypot(source_eye_dx, source_eye_dy)
    target_eye_separation = target_right_eye_x - target_left_eye_x
    if source_eye_separation <= 1:
        raise MaskPrintError("portrait eye spacing could not be detected.")
    scale = target_eye_separation / source_eye_separation
    rotation = -math.atan2(source_eye_dy, source_eye_dx)
    cos_value = math.cos(rotation) * scale
    sin_value = math.sin(rotation) * scale
    source_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    source_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    target_mid_x = (target_left_eye_x + target_right_eye_x) / 2.0
    target_mid_y = float(target_eye_y)
    transform = [
        [
            cos_value,
            -sin_value,
            target_mid_x - cos_value * source_mid_x + sin_value * source_mid_y,
        ],
        [
            sin_value,
            cos_value,
            target_mid_y - sin_value * source_mid_x - cos_value * source_mid_y,
        ],
    ]

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise MaskPrintError(
            "reliable mask alignment needs opencv-python-headless and numpy."
        ) from exc

    background = _sample_background_color(source)
    aligned_rgb = cv2.warpAffine(
        np.asarray(source),
        np.asarray(transform, dtype=np.float64),
        (target_width_px, target_height_px),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    fitted = Image.fromarray(aligned_rgb, mode="RGB").convert("RGBA")
    fitted.putalpha(_face_surface_mask(options))

    aligned_landmarks = {
        "left_eye": _transform_point(landmarks.left_eye, transform),
        "right_eye": _transform_point(landmarks.right_eye, transform),
        "nose_tip": _transform_point(landmarks.nose_tip, transform),
        "left_mouth": _transform_point(landmarks.left_mouth, transform),
        "right_mouth": _transform_point(landmarks.right_mouth, transform),
    }
    max_eye_error = max(
        math.dist(aligned_landmarks["left_eye"], (target_left_eye_x, target_eye_y)),
        math.dist(aligned_landmarks["right_eye"], (target_right_eye_x, target_eye_y)),
    )
    if max_eye_error > 1.25:
        raise MaskPrintError(
            f"internal alignment verification failed ({max_eye_error:.2f}px eye error)."
        )

    nose_mm = (
        _mm(int(round(aligned_landmarks["nose_tip"][0])), options.dpi),
        _mm(int(round(aligned_landmarks["nose_tip"][1])), options.dpi),
    )
    mouth_y_mm = _mm(
        int(round((aligned_landmarks["left_mouth"][1] + aligned_landmarks["right_mouth"][1]) / 2.0)),
        options.dpi,
    )
    center_mm = options.width_mm / 2.0
    if not (
        abs(nose_mm[0] - center_mm) <= 14.0
        and options.eye_center_from_top_mm + 20.0 <= nose_mm[1] <= options.eye_center_from_top_mm + 70.0
        and nose_mm[1] + 12.0 <= mouth_y_mm <= min(options.height_mm - 18.0, nose_mm[1] + 65.0)
    ):
        raise MaskPrintError(
            "aligned facial landmarks do not fit the measured mask geometry; "
            "review the portrait and mask measurements."
        )

    # Invert the verified affine map to record which source region feeds the
    # physical mask.  This is audit metadata, not a second heuristic crop.
    inverse = cv2.invertAffineTransform(np.asarray(transform, dtype=np.float64))
    source_corners = [
        _transform_point((0.0, 0.0), inverse.tolist()),
        _transform_point((float(target_width_px), 0.0), inverse.tolist()),
        _transform_point((0.0, float(target_height_px)), inverse.tolist()),
        _transform_point((float(target_width_px), float(target_height_px)), inverse.tolist()),
    ]
    xs = [point[0] for point in source_corners]
    ys = [point[1] for point in source_corners]
    source_region = {
        "left": int(math.floor(min(xs))),
        "top": int(math.floor(min(ys))),
        "right": int(math.ceil(max(xs))),
        "bottom": int(math.ceil(max(ys))),
    }
    return fitted, {
        "source_region_px": source_region,
        "source_landmarks": landmarks.metadata(),
        "aligned_landmarks_px": {
            name: [round(value, 2) for value in point]
            for name, point in aligned_landmarks.items()
        },
        "alignment_transform": {
            "type": "similarity",
            "scale": round(scale, 8),
            "rotation_degrees": round(math.degrees(rotation), 5),
            "matrix": [[round(value, 10) for value in row] for row in transform],
            "max_eye_registration_error_px": round(max_eye_error, 6),
        },
        "quality_control": quality,
    }


def _panel_polygons_mm(
    options: MaskPrintOptions,
) -> dict[str, list[tuple[float, float]]]:
    width = options.width_mm
    height = options.height_mm
    center = width / 2.0
    eye_y = options.eye_center_from_top_mm
    overlap = options.overlap_mm

    # The 30 mm nose measurement describes the raised lower plane.  The print
    # island also contains the bridge above it, hence the extra 24 mm.
    nose_bottom = min(height * 0.70, eye_y + options.nose_length_mm + 24.0)
    mouth_bottom = min(height * 0.82, nose_bottom + 42.0)
    chin_seam = min(height * 0.86, mouth_bottom + 12.0)
    nose_half = options.nose_base_width_mm / 2.0

    return {
        "forehead": [
            (-overlap, -overlap),
            (width + overlap, -overlap),
            (width + overlap, eye_y - 9.0 + overlap),
            (center + 32.0, eye_y - 4.0 + overlap),
            (center, eye_y - 16.0 + overlap),
            (center - 32.0, eye_y - 4.0 + overlap),
            (-overlap, eye_y - 9.0 + overlap),
        ],
        "left_cheek": [
            (-overlap, eye_y - 18.0 - overlap),
            (center - 12.0 + overlap, eye_y - 18.0 - overlap),
            (center - nose_half + overlap, nose_bottom + 10.0),
            (center - 43.0 + overlap, mouth_bottom),
            (center - 24.0 + overlap, chin_seam + overlap),
            (-overlap, chin_seam + overlap),
        ],
        "right_cheek": [
            (width + overlap, eye_y - 18.0 - overlap),
            (center + 12.0 - overlap, eye_y - 18.0 - overlap),
            (center + nose_half - overlap, nose_bottom + 10.0),
            (center + 43.0 - overlap, mouth_bottom),
            (center + 24.0 - overlap, chin_seam + overlap),
            (width + overlap, chin_seam + overlap),
        ],
        "nose": [
            (center - 15.0 - overlap, eye_y - 20.0 - overlap),
            (center + 15.0 + overlap, eye_y - 20.0 - overlap),
            (center + nose_half + overlap, nose_bottom),
            (center + 18.0 + overlap, nose_bottom + 14.0 + overlap),
            (center - 18.0 - overlap, nose_bottom + 14.0 + overlap),
            (center - nose_half - overlap, nose_bottom),
        ],
        "mouth": [
            (center - 24.0 - overlap, nose_bottom - 4.0 - overlap),
            (center + 24.0 + overlap, nose_bottom - 4.0 - overlap),
            (center + 48.0 + overlap, mouth_bottom),
            (center + 35.0 + overlap, chin_seam + overlap),
            (center - 35.0 - overlap, chin_seam + overlap),
            (center - 48.0 - overlap, mouth_bottom),
        ],
        "chin": [
            (-overlap, chin_seam - overlap),
            (width + overlap, chin_seam - overlap),
            (width + overlap, height + overlap),
            (-overlap, height + overlap),
        ],
    }


def _build_panels(
    surface_mask: Image.Image,
    options: MaskPrintOptions,
) -> list[_Panel]:
    labels = {
        "forehead": "01 STIRN",
        "left_cheek": "02 LINKE WANGE",
        "right_cheek": "03 RECHTE WANGE",
        "nose": "04 NASE",
        "mouth": "05 MUNDZONE",
        "chin": "06 KINN",
    }
    panels: list[_Panel] = []
    core_options = options.model_copy(update={"overlap_mm": 0.0})
    core_polygons = _panel_polygons_mm(core_options)
    for name, polygon_mm in _panel_polygons_mm(options).items():
        region = Image.new("L", surface_mask.size, 0)
        ImageDraw.Draw(region).polygon(
            _to_pixels(polygon_mm, dpi=options.dpi),
            fill=255,
        )
        region = ImageChops.multiply(region, surface_mask)
        core = Image.new("L", surface_mask.size, 0)
        ImageDraw.Draw(core).polygon(
            _to_pixels(core_polygons[name], dpi=options.dpi),
            fill=255,
        )
        core = ImageChops.multiply(core, surface_mask)
        panels.append(
            _Panel(name=name, label=labels[name], mask=region, core_mask=core)
        )
    return panels


def _panel_asset(
    skin: Image.Image,
    panel: _Panel,
    *,
    dpi: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    bbox = panel.mask.getbbox()
    if bbox is None:
        raise MaskPrintError(f"panel {panel.name!r} has no printable area.")
    crop = skin.crop(bbox)
    alpha = panel.mask.crop(bbox)
    core = panel.core_mask.crop(bbox)
    crop.putalpha(alpha)

    # A one-pixel boundary is a cut guide at 300 dpi (~0.085 mm).  It lies on
    # the outermost overlap/bleed pixel, so careful cutting removes it.
    inner = alpha.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.subtract(alpha, inner)
    line = Image.new("RGBA", crop.size, (35, 35, 35, 255))
    crop.alpha_composite(Image.composite(line, Image.new("RGBA", crop.size), edge))

    # The core boundary lies inside the 1.5 mm bleed and is therefore covered
    # by the adjoining panel after assembly.  A dashed blue registration line
    # makes the intended overlap explicit without remaining visible on the
    # finished mask.
    core_inner = core.filter(ImageFilter.MinFilter(3))
    core_edge = ImageChops.subtract(core, core_inner)
    cut_neighbourhood = edge.filter(ImageFilter.MaxFilter(7))
    seam_edge = ImageChops.subtract(core_edge, cut_neighbourhood)
    dash_mask = Image.new("L", crop.size, 0)
    dash_draw = ImageDraw.Draw(dash_mask)
    dash = max(2, _px(1.2, dpi))
    gap = max(2, _px(0.9, dpi))
    period = dash + gap
    for start_x in range(-crop.height, crop.width, period):
        dash_draw.polygon(
            [
                (start_x, 0),
                (start_x + dash, 0),
                (start_x + dash + crop.height, crop.height),
                (start_x + crop.height, crop.height),
            ],
            fill=255,
        )
    seam_edge = ImageChops.multiply(seam_edge, dash_mask)
    guide = Image.new("RGBA", crop.size, (25, 105, 170, 220))
    crop.alpha_composite(
        Image.composite(guide, Image.new("RGBA", crop.size), seam_edge)
    )
    return crop, bbox


def _new_a4_page(options: MaskPrintOptions) -> Image.Image:
    return Image.new(
        "RGB",
        (_px(A4_WIDTH_MM, options.dpi), _px(A4_HEIGHT_MM, options.dpi)),
        "white",
    )


def _layout_print_pages(
    skin: Image.Image,
    panels: list[_Panel],
    options: MaskPrintOptions,
    *,
    asset_id: str,
    title: str = "MASKEN-DRUCKTEILE - AUSSENSEITE",
    footer_note: str | None = None,
    page_number_offset: int = 0,
) -> tuple[list[Image.Image], list[dict[str, object]]]:
    margin = _px(9.0, options.dpi)
    gap = _px(5.0, options.dpi)
    label_height = _px(7.0, options.dpi)
    footer_height = _px(8.0, options.dpi)
    page_width = _px(A4_WIDTH_MM, options.dpi)
    page_height = _px(A4_HEIGHT_MM, options.dpi)
    usable_right = page_width - margin
    usable_bottom = page_height - margin - footer_height

    pages: list[Image.Image] = []
    placements: list[dict[str, object]] = []
    page = _new_a4_page(options)
    draw = ImageDraw.Draw(page)
    title_font = _font(max(18, _px(3.5, options.dpi)), bold=True)
    label_font = _font(max(16, _px(3.0, options.dpi)), bold=True)
    small_font = _font(max(14, _px(2.4, options.dpi)))
    x = margin
    y = margin + label_height
    row_height = 0
    page_number = 1

    def finish_page() -> None:
        nonlocal page, draw, x, y, row_height, page_number
        draw.text(
            (margin, page_height - margin - footer_height + _px(1.0, options.dpi)),
            f"{asset_id} | Seite {page_number + page_number_offset} | "
            f"100% / Tatsachliche Groesse | "
            f"{options.dpi} dpi | {options.template_version}"
            + (f" | {footer_note}" if footer_note else ""),
            fill=(40, 40, 40),
            font=small_font,
        )
        pages.append(page)
        page_number += 1
        page = _new_a4_page(options)
        draw = ImageDraw.Draw(page)
        x = margin
        y = margin + label_height
        row_height = 0

    draw.text(
        (margin, margin - _px(1.5, options.dpi)),
        title,
        fill=(20, 20, 20),
        font=title_font,
    )

    for panel in panels:
        asset, bbox = _panel_asset(skin, panel, dpi=options.dpi)
        item_width = asset.width
        item_height = label_height + asset.height
        if item_width > page_width - 2 * margin:
            raise MaskPrintError(
                f"panel {panel.name!r} is wider than the printable A4 area."
            )

        if x + item_width > usable_right and x > margin:
            x = margin
            y += row_height + gap
            row_height = 0
        if y + item_height > usable_bottom:
            finish_page()
            draw.text(
                (margin, margin - _px(1.5, options.dpi)),
                title,
                fill=(20, 20, 20),
                font=title_font,
            )

        draw.text(
            (x, y),
            f"{panel.label}  |  OBEN ^",
            fill=(20, 20, 20),
            font=label_font,
        )
        paste_y = y + label_height
        page.paste(asset, (x, paste_y), asset)
        placements.append(
            {
                "panel": panel.name,
                "page": page_number + page_number_offset,
                "x_mm": round(_mm(x, options.dpi), 2),
                "y_mm": round(_mm(paste_y, options.dpi), 2),
                "source_bbox_px": list(bbox),
                "printed_width_mm": round(_mm(asset.width, options.dpi), 2),
                "printed_height_mm": round(_mm(asset.height, options.dpi), 2),
            }
        )
        x += item_width + gap
        row_height = max(row_height, item_height)

    finish_page()
    return pages, placements


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int],
    width: int,
    closed: bool = False,
) -> None:
    if len(points) < 2:
        return
    path = points + [points[0]] if closed else points
    draw.line(path, fill=fill, width=width, joint="curve")


def _calibration_surface(options: MaskPrintOptions) -> Image.Image:
    """A segmented 10 mm grid used to test the physical shell before colour."""
    size = (_px(options.width_mm, options.dpi), _px(options.height_mm, options.dpi))
    surface = Image.new("RGBA", size, (252, 252, 252, 255))
    draw = ImageDraw.Draw(surface)
    thin = max(1, _px(0.15, options.dpi))
    strong = max(1, _px(0.25, options.dpi))
    for x_mm in range(0, int(math.ceil(options.width_mm)) + 1, 10):
        x = _px(float(x_mm), options.dpi)
        draw.line(
            [(x, 0), (x, size[1])],
            fill=(135, 135, 135, 255) if x_mm % 50 == 0 else (205, 205, 205, 255),
            width=strong if x_mm % 50 == 0 else thin,
        )
    for y_mm in range(0, int(math.ceil(options.height_mm)) + 1, 10):
        y = _px(float(y_mm), options.dpi)
        draw.line(
            [(0, y), (size[0], y)],
            fill=(135, 135, 135, 255) if y_mm % 50 == 0 else (205, 205, 205, 255),
            width=strong if y_mm % 50 == 0 else thin,
        )
    center_x = _px(options.width_mm / 2.0, options.dpi)
    draw.line(
        [(center_x, 0), (center_x, size[1])],
        fill=(120, 30, 30, 255),
        width=max(1, _px(0.25, options.dpi)),
    )
    surface.putalpha(_face_surface_mask(options))
    return surface


def _calibration_page(options: MaskPrintOptions) -> Image.Image:
    page = _new_a4_page(options)
    draw = ImageDraw.Draw(page)
    page_w, page_h = page.size
    mask_w = _px(options.width_mm, options.dpi)
    mask_h = _px(options.height_mm, options.dpi)
    offset_x = (page_w - mask_w) // 2
    offset_y = _px(20.0, options.dpi)

    title_font = _font(max(18, _px(3.5, options.dpi)), bold=True)
    text_font = _font(max(15, _px(2.6, options.dpi)))
    draw.text(
        (_px(8.0, options.dpi), _px(5.0, options.dpi)),
        "MASKEN-KALIBRIERUNG - SEITE 1 BEI 100% DRUCKEN",
        fill=(15, 15, 15),
        font=title_font,
    )
    draw.text(
        (_px(8.0, options.dpi), _px(11.0, options.dpi)),
        f"Sollmass: {options.width_mm:.1f} x {options.height_mm:.1f} mm | "
        f"Augen innen: {options.eye_inner_gap_mm:.1f} mm | Danach Rasterteile an Probe-Maske testen",
        fill=(55, 55, 55),
        font=text_font,
    )

    # 10 mm physical grid, clipped visually by the face outline through a mask.
    grid_layer = Image.new("RGB", page.size, "white")
    grid_draw = ImageDraw.Draw(grid_layer)
    thin = max(1, _px(0.15, options.dpi))
    for x_mm in range(0, int(math.ceil(options.width_mm)) + 1, 10):
        x = offset_x + _px(float(x_mm), options.dpi)
        grid_draw.line(
            [(x, offset_y), (x, offset_y + mask_h)],
            fill=(210, 210, 210),
            width=thin,
        )
    for y_mm in range(0, int(math.ceil(options.height_mm)) + 1, 10):
        y = offset_y + _px(float(y_mm), options.dpi)
        grid_draw.line(
            [(offset_x, y), (offset_x + mask_w, y)],
            fill=(210, 210, 210),
            width=thin,
        )
    face_mask = _face_surface_mask(options)
    page.paste(
        grid_layer.crop((offset_x, offset_y, offset_x + mask_w, offset_y + mask_h)),
        (offset_x, offset_y),
        face_mask,
    )

    outline = [
        (offset_x + x, offset_y + y)
        for x, y in _to_pixels(_outer_outline_mm(options), dpi=options.dpi)
    ]
    _draw_polyline(
        draw,
        outline,
        fill=(20, 20, 20),
        width=max(2, _px(0.35, options.dpi)),
        closed=True,
    )
    for side in ("left", "right"):
        eye = [
            (offset_x + x, offset_y + y)
            for x, y in _to_pixels(
                _eye_outline_mm(options, side=side), dpi=options.dpi
            )
        ]
        _draw_polyline(
            draw,
            eye,
            fill=(170, 25, 25),
            width=max(2, _px(0.3, options.dpi)),
            closed=True,
        )

    center_x = offset_x + mask_w // 2
    draw.line(
        [(center_x, offset_y), (center_x, offset_y + mask_h)],
        fill=(80, 80, 80),
        width=max(1, _px(0.2, options.dpi)),
    )

    # A physical 100 mm control bar catches all print-driver scaling mistakes.
    bar_y = page_h - _px(13.0, options.dpi)
    bar_x = _px(15.0, options.dpi)
    bar_len = _px(100.0, options.dpi)
    stroke = max(2, _px(0.35, options.dpi))
    draw.line([(bar_x, bar_y), (bar_x + bar_len, bar_y)], fill=(0, 0, 0), width=stroke)
    tick = _px(3.0, options.dpi)
    draw.line([(bar_x, bar_y - tick), (bar_x, bar_y + tick)], fill=(0, 0, 0), width=stroke)
    draw.line(
        [(bar_x + bar_len, bar_y - tick), (bar_x + bar_len, bar_y + tick)],
        fill=(0, 0, 0),
        width=stroke,
    )
    draw.text(
        (bar_x, bar_y - _px(7.0, options.dpi)),
        "Diese Linie muss nach dem Druck exakt 100 mm lang sein.",
        fill=(20, 20, 20),
        font=text_font,
    )
    return page


def _write_pdf(
    path: Path,
    pages: Iterable[Image.Image],
    *,
    title: str,
) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as exc:
        raise MaskPrintError(
            "PDF export needs reportlab. Install the project again so its "
            "declared reportlab dependency is available."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=A4, pageCompression=1)
    canvas.setTitle(title)
    canvas.setAuthor("Synthetic Portrait Lab")
    for page in pages:
        buffer = io.BytesIO()
        page.convert("RGB").save(buffer, format="PNG", dpi=(300, 300))
        buffer.seek(0)
        canvas.drawImage(
            ImageReader(buffer),
            0,
            0,
            width=A4[0],
            height=A4[1],
            preserveAspectRatio=False,
        )
        canvas.showPage()
    canvas.save()


def _svg_path(points: list[tuple[float, float]], *, close: bool = True) -> str:
    if not points:
        return ""
    chunks = [f"M {points[0][0]:.3f} {points[0][1]:.3f}"]
    chunks.extend(f"L {x:.3f} {y:.3f}" for x, y in points[1:])
    if close:
        chunks.append("Z")
    return " ".join(chunks)


def _write_cutlines_svg(
    path: Path,
    options: MaskPrintOptions,
    *,
    asset_id: str,
) -> None:
    width = options.width_mm
    height = options.height_mm
    outer = _svg_path(_outer_outline_mm(options))
    eye_left = _svg_path(_eye_outline_mm(options, side="left"))
    eye_right = _svg_path(_eye_outline_mm(options, side="right"))
    panel_paths = "\n".join(
        f'    <path id="{escape(name)}" d="{_svg_path(points)}"/>'
        for name, points in _panel_polygons_mm(options).items()
    )
    content = f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{width:.3f}mm" height="{height:.3f}mm"
  viewBox="0 0 {width:.3f} {height:.3f}">
  <title>{escape(asset_id)} mask cut lines - {escape(options.template_version)}</title>
  <defs>
    <clipPath id="mask-surface">
      <path d="{outer}"/>
    </clipPath>
  </defs>
  <g fill="none" stroke="#111" stroke-width="0.25"
     vector-effect="non-scaling-stroke" clip-path="url(#mask-surface)">
{panel_paths}
  </g>
  <path d="{outer}" fill="none" stroke="#111" stroke-width="0.35"
        vector-effect="non-scaling-stroke"/>
  <path d="{eye_left}" fill="none" stroke="#d02020" stroke-width="0.3"
        vector-effect="non-scaling-stroke"/>
  <path d="{eye_right}" fill="none" stroke="#d02020" stroke-width="0.3"
        vector-effect="non-scaling-stroke"/>
</svg>
"""
    path.write_text(content, encoding="utf-8")


def _preview(
    skin: Image.Image,
    panels: list[_Panel],
) -> Image.Image:
    preview = skin.copy()
    for panel in panels:
        inner = panel.core_mask.filter(ImageFilter.MinFilter(5))
        edge = ImageChops.subtract(panel.core_mask, inner)
        seam = Image.new("RGBA", preview.size, (35, 35, 35, 220))
        preview.alpha_composite(Image.composite(seam, Image.new("RGBA", preview.size), edge))
    # Store a normal white-background preview so image viewers that ignore alpha
    # do not reveal the discarded source pixels inside eye holes or outside the mask.
    flattened = Image.new("RGBA", preview.size, (255, 255, 255, 255))
    flattened.alpha_composite(preview)
    return flattened.convert("RGB")


def create_mask_print_pack(
    image_bytes: bytes,
    output_dir: str | Path,
    *,
    asset_id: str,
    options: MaskPrintOptions | None = None,
    source_filename: str | None = None,
) -> MaskPrintPack:
    """Create a complete, exact-scale A4 print pack for one portrait."""
    options = options or MaskPrintOptions()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    safe_id = "".join(
        char if (char.isalnum() or char in "-_") else "_" for char in asset_id
    ).strip("_") or "portrait"

    # Landmark inference and all geometry gates run before any printable asset is
    # emitted.  A bad portrait therefore produces an explicit error, never a
    # plausible-looking but misregistered PDF.
    skin, normalization = _normalize_portrait(image_bytes, options)
    surface_mask = _face_surface_mask(options)
    panels = _build_panels(surface_mask, options)
    pages, placements = _layout_print_pages(
        skin,
        panels,
        options,
        asset_id=safe_id,
        footer_note="Schwarz schneiden; Blau ist Ueberlappungsmarke",
    )
    calibration_overview = _calibration_page(options)
    calibration_panels, _ = _layout_print_pages(
        _calibration_surface(options),
        panels,
        options,
        asset_id=f"{safe_id}-KALIBRIERUNG",
        title="PASSFORM-KALIBRIERUNG - RASTERTEILE",
        footer_note="Vor Farbdruck ausschneiden und an Probe-Maske testen",
        page_number_offset=1,
    )

    preview_path = output / f"{safe_id}_mask_preview.png"
    page_paths = tuple(
        output / f"{safe_id}_print_page_{index}.png"
        for index in range(1, len(pages) + 1)
    )
    print_pdf_path = output / f"{safe_id}_print.pdf"
    calibration_pdf_path = output / f"{safe_id}_calibration.pdf"
    cutlines_svg_path = output / f"{safe_id}_cutlines.svg"
    metadata_path = output / f"{safe_id}_mask.json"

    _preview(skin, panels).save(
        preview_path,
        format="PNG",
        dpi=(options.dpi, options.dpi),
    )
    for page, page_path in zip(pages, page_paths):
        page.save(page_path, format="PNG", dpi=(options.dpi, options.dpi))
    _write_pdf(
        print_pdf_path,
        pages,
        title=f"{safe_id} segmented mask print",
    )
    _write_pdf(
        calibration_pdf_path,
        [calibration_overview, *calibration_panels],
        title=f"{safe_id} mask calibration",
    )
    _write_cutlines_svg(
        cutlines_svg_path,
        options,
        asset_id=safe_id,
    )

    metadata = {
        "asset_id": safe_id,
        "source_filename": source_filename,
        "template": options.model_dump(mode="json"),
        # Kept for readers of prototype-v1 metadata; v2 records the complete
        # detector/transform audit trail in ``normalization`` below.
        "source_crop_px": normalization["source_region_px"],
        "normalization": normalization,
        "normalized_surface_px": list(skin.size),
        "calibration_required": True,
        "measurement_assumption": (
            "width and height are treated as surface dimensions; physical "
            "curvature still requires the calibration print"
        ),
        "panels": placements,
        "assembly": {
            "cut_line": "black outer line",
            "overlap_registration": "blue dashed line; cover it with the adjoining panel",
            "recommended_order": [
                "forehead",
                "left_cheek",
                "right_cheek",
                "chin",
                "mouth",
                "nose",
            ],
            "calibration_pdf_pages": 1 + len(calibration_panels),
        },
        "outputs": {
            "preview": preview_path.name,
            "print_pdf": print_pdf_path.name,
            "calibration_pdf": calibration_pdf_path.name,
            "cutlines_svg": cutlines_svg_path.name,
            "pages": [path.name for path in page_paths],
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return MaskPrintPack(
        output_dir=output,
        preview_path=preview_path,
        print_pdf_path=print_pdf_path,
        calibration_pdf_path=calibration_pdf_path,
        cutlines_svg_path=cutlines_svg_path,
        page_paths=page_paths,
        metadata_path=metadata_path,
    )
