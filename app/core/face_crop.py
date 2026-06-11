"""Post-processing face-mask crop for passport-style portraits.

Since all generated images are centred, front-facing passport portraits with a
plain background, we use a simple luminance/edge-based approach to find where
the head actually is, then crop tightly around it:

* Full hair visible with a small margin above.
* Chin visible with a tiny margin below — minimal/no neck.
* Sides tightened so the face fills more of the frame.

The result matches the "zoomed-in screenshot" look the user wants for face
masks: complete head including all hair, but no neck/shoulders/torso.
"""

from __future__ import annotations

import io

from PIL import Image


def _find_head_bounds(img: Image.Image) -> tuple[int, int, int, int]:
    """Find approximate head bounding box using the background colour.

    Passport-style portraits have a plain light-gray / off-white background.
    We find the head by scanning for the first/last rows and columns that
    contain non-background pixels.

    Returns (top, bottom, left, right) in pixel coordinates.
    """
    # Convert to grayscale for analysis
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    # Background threshold: pixels brighter than this are "background".
    # Passport photos have light gray / off-white backgrounds (typically > 200).
    bg_threshold = 200
    # Minimum fraction of non-bg pixels in a row/col to consider it "content"
    content_ratio = 0.05

    # Scan top-down to find where the head starts
    top = 0
    for y in range(h):
        non_bg = sum(1 for x in range(w) if pixels[x, y] < bg_threshold)
        if non_bg / w > content_ratio:
            top = y
            break

    # Scan bottom-up to find where the head/body ends
    bottom = h - 1
    for y in range(h - 1, -1, -1):
        non_bg = sum(1 for x in range(w) if pixels[x, y] < bg_threshold)
        if non_bg / w > content_ratio:
            bottom = y
            break

    # Scan left-to-right
    left = 0
    for x in range(w):
        non_bg = sum(1 for y in range(h) if pixels[x, y] < bg_threshold)
        if non_bg / h > content_ratio:
            left = x
            break

    # Scan right-to-left
    right = w - 1
    for x in range(w - 1, -1, -1):
        non_bg = sum(1 for y in range(h) if pixels[x, y] < bg_threshold)
        if non_bg / h > content_ratio:
            right = x
            break

    return top, bottom, left, right


def apply_face_crop(image_bytes: bytes) -> bytes:
    """Crop *image_bytes* (PNG/JPEG) to a tight face-mask framing.

    Strategy:
    1. Detect the head bounds (top of hair to bottom of body, left ear to right ear).
    2. The "chin" is estimated at ~55% of the head-content height (from top of
       hair to bottom of visible body). This works because in passport photos
       the head-to-shoulder ratio is predictable.
    3. Crop with: small margin above hair, cut just below chin, moderate side padding.

    Returns the cropped image as PNG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    head_top, body_bottom, head_left, head_right = _find_head_bounds(img)

    # Total content height (hair to bottom of visible body)
    content_h = body_bottom - head_top
    # Total content width (left ear to right ear)
    content_w = head_right - head_left

    if content_h <= 0 or content_w <= 0:
        # Fallback: can't detect head, return original
        return image_bytes

    # Estimate chin position: in a standard passport photo, the chin is at
    # roughly 60-70% of the way from the top of the hair to the bottom of
    # the visible body (rest is neck/shoulders/collar).
    chin_y = head_top + int(content_h * 0.65)

    # --- Build crop box --------------------------------------------------- #
    # Top: small margin above hair (~5% of head height)
    margin_top = int(content_h * 0.05)
    crop_top = max(0, head_top - margin_top)

    # Bottom: generous margin below estimated chin (~15% of head height)
    margin_bottom = int(content_h * 0.15)
    crop_bottom = min(h, chin_y + margin_bottom)

    # Sides: some padding around the ears (~12% of content width on each side)
    margin_side = int(content_w * 0.12)
    crop_left = max(0, head_left - margin_side)
    crop_right = min(w, head_right + margin_side)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    buf = io.BytesIO()
    fmt = img.format or "PNG"
    cropped.save(buf, format=fmt)
    return buf.getvalue()
