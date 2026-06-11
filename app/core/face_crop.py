"""Post-processing face-mask crop for passport-style portraits.

Strategy (designed for centred, front-facing passport photos on a plain
light background):

1. Detect the head bounding box via luminance scan of the background.
2. Measure the head height (top-of-hair to chin).
3. Build a crop box that is a *fixed multiple* of the head height, centred
   on the face, with a fixed 3:4 portrait aspect ratio.
4. Resize every output to a consistent resolution (1200×1600 px).

Because the crop box is always sized relative to the detected head, and every
output is the same resolution, the result is **consistent across images**:
the face always occupies the same proportion of the frame.
"""

from __future__ import annotations

import io

from PIL import Image

# ---- output settings ---------------------------------------------------- #
OUTPUT_WIDTH = 2480
OUTPUT_HEIGHT = 3508
ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT  # A4 proportion (1 : 1.414)


def _find_head_bounds(img: Image.Image) -> tuple[int, int, int, int]:
    """Find the head bounding box via background-colour scanning.

    Returns (top, bottom, left, right) in pixel coordinates.
    """
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    bg_threshold = 200  # pixels brighter than this = background
    content_ratio = 0.05  # min fraction of dark pixels to count as "content"

    top = 0
    for y in range(h):
        non_bg = sum(1 for x in range(w) if pixels[x, y] < bg_threshold)
        if non_bg / w > content_ratio:
            top = y
            break

    bottom = h - 1
    for y in range(h - 1, -1, -1):
        non_bg = sum(1 for x in range(w) if pixels[x, y] < bg_threshold)
        if non_bg / w > content_ratio:
            bottom = y
            break

    left = 0
    for x in range(w):
        non_bg = sum(1 for y_i in range(h) if pixels[x, y_i] < bg_threshold)
        if non_bg / h > content_ratio:
            left = x
            break

    right = w - 1
    for x in range(w - 1, -1, -1):
        non_bg = sum(1 for y_i in range(h) if pixels[x, y_i] < bg_threshold)
        if non_bg / h > content_ratio:
            right = x
            break

    return top, bottom, left, right


def apply_face_crop(image_bytes: bytes) -> bytes:
    """Crop and resize to a tight, consistent face-mask framing.

    Every output is exactly OUTPUT_WIDTH × OUTPUT_HEIGHT with the face
    occupying the same relative area, regardless of the source dimensions.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    head_top, body_bottom, head_left, head_right = _find_head_bounds(img)

    content_h = body_bottom - head_top
    content_w = head_right - head_left

    if content_h <= 0 or content_w <= 0:
        return image_bytes

    # --- estimate face geometry ------------------------------------------ #
    # In passport photos the chin is at ~60-65% of total content height
    # (content = hair-top to bottom-of-body). The face centre (bridge of
    # nose) is at ~40% from the top of the hair.
    face_center_y = head_top + int(content_h * 0.40)
    face_center_x = (head_left + head_right) // 2

    # Head height: from top of hair to estimated chin (~65% of content)
    head_h = int(content_h * 0.65)

    # --- build crop box -------------------------------------------------- #
    # The crop height = head_height * scale_factor.  A factor of ~1.10 means
    # the head fills ~90% of the frame vertically (tight face-mask style).
    crop_h = int(head_h * 1.10)
    crop_w = int(crop_h * ASPECT)  # maintain output ratio

    # Centre the crop on the face, biased slightly upward so the forehead/hair
    # has a tiny margin and the chin sits comfortably above the bottom.
    # Shift the box so the top of the hair has ~4% padding.
    crop_top = head_top - int(crop_h * 0.04)
    crop_bottom = crop_top + crop_h
    crop_left = face_center_x - crop_w // 2
    crop_right = crop_left + crop_w

    # --- clamp to image bounds ------------------------------------------- #
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_bottom > h:
        crop_top -= (crop_bottom - h)
        crop_bottom = h
        crop_top = max(0, crop_top)
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_right > w:
        crop_left -= (crop_right - w)
        crop_right = w
        crop_left = max(0, crop_left)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Resize to the fixed output resolution for consistency.
    result = cropped.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = img.format or "PNG"
    result.save(buf, format=fmt)
    return buf.getvalue()
