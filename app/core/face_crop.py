"""Post-processing face crop for passport-style portraits.

Goals:
- Frame the head only (hair top through chin), not shoulders.
- Never clip any part of the face; pad with background when needed.
- Output a consistent A4 portrait aspect ratio at print resolution.
"""

from __future__ import annotations

import io
import statistics

from PIL import Image

# A4 portrait at 300 DPI (210 mm × 297 mm).
OUTPUT_WIDTH = 2480
OUTPUT_HEIGHT = 3508
ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT  # ≈ 0.707 (1 : √2)

_BG_THRESHOLD = 200
_CONTENT_RATIO = 0.05


def _sample_background_color(img: Image.Image) -> tuple[int, int, int]:
    """Estimate the flat background colour from image corners."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    corners = [
        rgb.getpixel((2, 2)),
        rgb.getpixel((w - 3, 2)),
        rgb.getpixel((2, h - 3)),
        rgb.getpixel((w - 3, h - 3)),
    ]
    return (
        int(statistics.median(c[0] for c in corners)),
        int(statistics.median(c[1] for c in corners)),
        int(statistics.median(c[2] for c in corners)),
    )


def _row_content_span(
    pixels,
    y: int,
    width: int,
    *,
    x_start: int = 0,
    x_end: int | None = None,
) -> tuple[int, int, int]:
    """Return (left, right, span_width) of non-background pixels on one row."""
    x_end = width - 1 if x_end is None else x_end
    left = width
    right = -1
    for x in range(x_start, x_end + 1):
        if pixels[x, y] < _BG_THRESHOLD:
            left = min(left, x)
            right = max(right, x)
    if right < left:
        return 0, 0, 0
    return left, right, right - left


def _find_vertical_content_bounds(img: Image.Image) -> tuple[int, int]:
    """Top and bottom rows that contain subject pixels."""
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    top = 0
    for y in range(h):
        _, _, span = _row_content_span(pixels, y, w)
        if span / w > _CONTENT_RATIO:
            top = y
            break

    bottom = h - 1
    for y in range(h - 1, -1, -1):
        _, _, span = _row_content_span(pixels, y, w)
        if span / w > _CONTENT_RATIO:
            bottom = y
            break

    return top, bottom


def _find_chin_row(
    pixels,
    *,
    width: int,
    top: int,
    bottom: int,
) -> int:
    """Locate the bottom of the jaw using the neck width dip below the face."""
    spans: list[tuple[int, int]] = []
    for y in range(top, bottom + 1):
        _, _, span = _row_content_span(pixels, y, width)
        spans.append((y, span))

    if not spans or all(s == 0 for _, s in spans):
        return bottom

    content_h = bottom - top
    if content_h <= 0:
        return bottom

    # Head width reference: peak span in the upper ~70% (excludes shoulders).
    head_zone_end = top + int(content_h * 0.70)
    head_peak = max(span for y, span in spans if y <= head_zone_end)
    if head_peak <= 0:
        head_peak = max(span for _, span in spans)

    # Neck: narrowest non-zero span in the lower half of the subject.
    neck_search_start = top + int(content_h * 0.50)
    neck_span = head_peak
    neck_y = bottom
    for y, span in spans:
        if y >= neck_search_start and 0 < span < neck_span:
            neck_span = span
            neck_y = y

    # If no clear neck (tight headshot), fall back to a head-proportion estimate.
    if neck_span >= head_peak * 0.85:
        head_w = head_peak  # span proxy; refined by horizontal bounds later
        estimated = top + int(head_w * 1.15)
        chin = min(bottom, max(top, estimated))
    else:
        # Chin is the last row above the neck that is still jaw-wide.
        jaw_threshold = max(neck_span * 1.35, head_peak * 0.55)
        chin = top
        for y, span in spans:
            if y < neck_y and span >= jaw_threshold:
                chin = y

    # Safety margin below the detected jaw line.
    head_h_guess = max(1, chin - top)
    chin = min(bottom, chin + max(2, int(head_h_guess * 0.04)))
    return chin


def _find_horizontal_head_bounds(
    pixels,
    *,
    width: int,
    height: int,
    top: int,
    chin: int,
) -> tuple[int, int]:
    """Left/right bounds using only head rows (excludes wide shoulders below the chin)."""
    left = width
    right = -1
    for y in range(top, chin + 1):
        row_left, row_right, span = _row_content_span(pixels, y, width)
        if span <= 0:
            continue
        left = min(left, row_left)
        right = max(right, row_right)

    if right < left:
        return 0, width - 1
    return left, right


def _find_head_bounds(img: Image.Image) -> tuple[int, int, int, int]:
    """Return (top, chin, left, right) for the head region only."""
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    top, body_bottom = _find_vertical_content_bounds(img)
    chin = _find_chin_row(pixels, width=w, top=top, bottom=body_bottom)
    left, right = _find_horizontal_head_bounds(
        pixels, width=w, height=h, top=top, chin=chin
    )
    return top, chin, left, right


def _crop_with_padding(
    img: Image.Image,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    bg_color: tuple[int, int, int],
) -> Image.Image:
    """Extract a crop box, padding with background instead of shifting/clipping the face."""
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    if crop_w <= 0 or crop_h <= 0:
        return img.copy()

    canvas = Image.new("RGB", (crop_w, crop_h), bg_color)
    src = img.convert("RGB")

    dest_x = max(0, -crop_left)
    dest_y = max(0, -crop_top)
    src_left = max(0, crop_left)
    src_top = max(0, crop_top)
    src_right = min(src.width, crop_right)
    src_bottom = min(src.height, crop_bottom)

    if src_right > src_left and src_bottom > src_top:
        region = src.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(region, (dest_x, dest_y))

    return canvas


def apply_face_crop(image_bytes: bytes) -> bytes:
    """Crop to a tight head framing on an A4 portrait canvas.

    The head (hair top through chin) is detected, padded generously, centred,
    and fitted to the A4 aspect ratio. If the crop box exceeds the source image,
    background padding is added so no facial feature is ever clipped.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    head_top, chin, head_left, head_right = _find_head_bounds(img)
    head_h = chin - head_top
    head_w = head_right - head_left

    if head_h <= 0 or head_w <= 0:
        return image_bytes

    head_cx = (head_left + head_right) / 2
    head_cy = (head_top + chin) / 2

    # Generous padding so hair, ears, and chin never touch the frame edge.
    pad_top = head_h * 0.12
    pad_bottom = head_h * 0.14
    pad_x = head_w * 0.10

    inner_h = head_h + pad_top + pad_bottom
    inner_w = head_w + 2 * pad_x

    # Expand to A4 aspect while fully containing the padded head box.
    if inner_w / inner_h > ASPECT:
        crop_h = inner_h
        crop_w = inner_h * ASPECT
    else:
        crop_w = inner_w
        crop_h = inner_w / ASPECT

    crop_left = int(round(head_cx - crop_w / 2))
    crop_top = int(round(head_cy - crop_h / 2))
    crop_right = crop_left + int(round(crop_w))
    crop_bottom = crop_top + int(round(crop_h))

    bg_color = _sample_background_color(img)
    cropped = _crop_with_padding(
        img, crop_left, crop_top, crop_right, crop_bottom, bg_color
    )
    result = cropped.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = img.format or "PNG"
    result.save(buf, format=fmt)
    return buf.getvalue()
