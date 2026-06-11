"""Tests for head-only A4 face cropping."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from app.core.face_crop import (
    ASPECT,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    _find_head_bounds,
    apply_face_crop,
)


def _make_portrait(
    *,
    width: int = 1024,
    height: int = 1536,
    head_top: int = 120,
    head_bottom: int = 900,
    head_left: int = 280,
    head_right: int = 740,
    shoulder_bottom: int = 1200,
) -> bytes:
    """Synthetic light-background portrait with an oval head and wider shoulders."""
    img = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (head_left, head_top, head_right, head_bottom),
        fill=(120, 90, 70),
    )
    # Shoulders are wider than the head and must not define the crop box.
    draw.rectangle(
        (head_left - 120, head_bottom + 20, head_right + 120, shoulder_bottom),
        fill=(40, 40, 90),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_find_head_bounds_ignores_shoulders():
    data = _make_portrait()
    img = Image.open(io.BytesIO(data))
    top, chin, left, right = _find_head_bounds(img)

    assert 115 <= top <= 130
    assert 850 <= chin <= 930
    assert 275 <= left <= 285
    assert 735 <= right <= 745


def test_apply_face_crop_a4_dimensions_and_contains_head():
    data = _make_portrait()
    cropped_bytes = apply_face_crop(data)
    out = Image.open(io.BytesIO(cropped_bytes))

    assert out.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    assert abs((OUTPUT_WIDTH / OUTPUT_HEIGHT) - ASPECT) < 0.001

    top, chin, left, right = _find_head_bounds(out)
    margin_top = top
    margin_bottom = out.height - 1 - chin
    margin_left = left
    margin_right = out.width - 1 - right

    assert margin_top >= 40
    assert margin_bottom >= 40
    assert margin_left >= 20
    assert margin_right >= 20


def test_apply_face_crop_never_clips_tight_source():
    """A near full-bleed headshot still keeps the entire head inside the frame."""
    data = _make_portrait(
        head_top=40,
        head_bottom=1350,
        head_left=40,
        head_right=980,
        shoulder_bottom=1500,
    )
    cropped_bytes = apply_face_crop(data)
    out = Image.open(io.BytesIO(cropped_bytes))

    top, chin, left, right = _find_head_bounds(out)
    assert top > 0
    assert chin < out.height - 1
    assert left > 0
    assert right < out.width - 1
