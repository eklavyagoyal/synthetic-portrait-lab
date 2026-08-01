"""Tests for deterministic segmented physical mask print packs."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image, ImageDraw

from app.core.mask_print import FaceLandmarks, MaskPrintError, create_mask_print_pack
from app.core.models import MaskPrintOptions


def _portrait_bytes() -> bytes:
    image = Image.new("RGB", (900, 1280), (242, 242, 242))
    draw = ImageDraw.Draw(image)
    draw.ellipse((205, 120, 695, 1050), fill=(188, 132, 104))
    draw.ellipse((335, 500, 395, 535), fill=(40, 30, 25))
    draw.ellipse((505, 500, 565, 535), fill=(40, 30, 25))
    draw.line((415, 785, 485, 785), fill=(90, 35, 35), width=12)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture_landmarks(source: Image.Image) -> FaceLandmarks:
    width, height = source.size
    return FaceLandmarks(
        face_bbox=(width * 0.25, height * 0.15, width * 0.50, height * 0.70),
        left_eye=(width * 0.40, height * 0.42),
        right_eye=(width * 0.60, height * 0.42),
        nose_tip=(width * 0.50, height * 0.54),
        left_mouth=(width * 0.44, height * 0.62),
        right_mouth=(width * 0.56, height * 0.62),
        confidence=0.99,
        detected_faces=1,
    )


def test_mask_print_options_match_first_physical_prototype() -> None:
    options = MaskPrintOptions()
    assert options.width_mm == 187.0
    assert options.height_mm == 245.0
    assert options.eye_inner_gap_mm == 40.0
    assert options.nose_base_width_mm == 40.0
    assert options.nose_length_mm == 30.0
    assert options.dpi == 300
    assert options.template_version == "landmarks-v2"


def test_create_mask_print_pack_writes_exact_scale_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.mask_print._detect_face_landmarks", _fixture_landmarks)
    options = MaskPrintOptions(dpi=300)
    pack = create_mask_print_pack(
        _portrait_bytes(),
        tmp_path / "print" / "portrait_000001",
        asset_id="portrait_000001",
        options=options,
        source_filename="portrait_000001.png",
    )

    expected_preview = (
        round(options.width_mm / 25.4 * options.dpi),
        round(options.height_mm / 25.4 * options.dpi),
    )
    with Image.open(pack.preview_path) as preview:
        assert preview.size == expected_preview

    assert len(pack.page_paths) >= 2
    expected_a4 = (
        round(210 / 25.4 * options.dpi),
        round(297 / 25.4 * options.dpi),
    )
    for page_path in pack.page_paths:
        with Image.open(page_path) as page:
            assert page.size == expected_a4

    assert pack.print_pdf_path.read_bytes().startswith(b"%PDF")
    assert pack.calibration_pdf_path.read_bytes().startswith(b"%PDF")

    svg = pack.cutlines_svg_path.read_text(encoding="utf-8")
    assert 'width="187.000mm"' in svg
    assert 'height="245.000mm"' in svg
    assert 'id="nose"' in svg

    metadata = json.loads(pack.metadata_path.read_text(encoding="utf-8"))
    assert metadata["calibration_required"] is True
    assert metadata["template"]["template_version"] == "landmarks-v2"
    assert metadata["normalization"]["quality_control"]["status"] == "passed"
    assert metadata["normalization"]["alignment_transform"]["max_eye_registration_error_px"] <= 1.25
    assert {panel["panel"] for panel in metadata["panels"]} == {
        "forehead",
        "left_cheek",
        "right_cheek",
        "nose",
        "mouth",
        "chin",
    }


def test_mask_print_rejects_image_without_a_detectable_face(tmp_path) -> None:
    image = Image.new("RGB", (900, 1280), (242, 242, 242))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with pytest.raises(MaskPrintError, match="no face was detected"):
        create_mask_print_pack(
            buffer.getvalue(),
            tmp_path / "rejected",
            asset_id="no-face",
            options=MaskPrintOptions(dpi=150),
        )

    assert not list((tmp_path / "rejected").glob("*.pdf"))


def test_mask_print_rejects_implausible_landmarks(tmp_path, monkeypatch) -> None:
    def bad_landmarks(source: Image.Image) -> FaceLandmarks:
        landmarks = _fixture_landmarks(source)
        return FaceLandmarks(
            **{
                **landmarks.__dict__,
                "right_eye": (source.width * 0.90, source.height * 0.20),
            }
        )

    monkeypatch.setattr("app.core.mask_print._detect_face_landmarks", bad_landmarks)
    with pytest.raises(MaskPrintError, match="landmark quality check failed"):
        create_mask_print_pack(
            _portrait_bytes(),
            tmp_path / "bad-landmarks",
            asset_id="bad-landmarks",
            options=MaskPrintOptions(dpi=150),
        )
