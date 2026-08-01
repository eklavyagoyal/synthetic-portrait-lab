"""Create a segmented physical mask print pack from an existing portrait."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..core.mask_print import MaskPrintError, create_mask_print_pack
from ..core.models import MaskPrintOptions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.mask_print",
        description=(
            "Turn one standardized portrait into segmented, exact-scale A4 "
            "mask print pages without another image-model call."
        ),
    )
    parser.add_argument("input", help="Source portrait PNG or JPEG.")
    parser.add_argument(
        "--output",
        help="Output directory (default: <input-folder>/<input-stem>_mask_print).",
    )
    parser.add_argument("--width-mm", type=float, default=187.0)
    parser.add_argument("--height-mm", type=float, default=245.0)
    parser.add_argument("--eye-inner-gap-mm", type=float, default=40.0)
    parser.add_argument("--eye-width-mm", type=float, default=38.0)
    parser.add_argument("--eye-height-mm", type=float, default=18.0)
    parser.add_argument("--eye-center-from-top-mm", type=float, default=103.0)
    parser.add_argument("--nose-width-mm", type=float, default=40.0)
    parser.add_argument("--nose-length-mm", type=float, default=30.0)
    parser.add_argument("--overlap-mm", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source = Path(args.input).expanduser()
    if not source.is_file():
        print(f"Error: source portrait not found: {source}", file=sys.stderr)
        return 2

    output = (
        Path(args.output).expanduser()
        if args.output
        else source.parent / f"{source.stem}_mask_print"
    )
    try:
        options = MaskPrintOptions(
            width_mm=args.width_mm,
            height_mm=args.height_mm,
            eye_inner_gap_mm=args.eye_inner_gap_mm,
            eye_opening_width_mm=args.eye_width_mm,
            eye_opening_height_mm=args.eye_height_mm,
            eye_center_from_top_mm=args.eye_center_from_top_mm,
            nose_base_width_mm=args.nose_width_mm,
            nose_length_mm=args.nose_length_mm,
            overlap_mm=args.overlap_mm,
            dpi=args.dpi,
        )
        pack = create_mask_print_pack(
            source.read_bytes(),
            output,
            asset_id=source.stem,
            options=options,
            source_filename=source.name,
        )
    except (MaskPrintError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Mask print pack: {pack.output_dir}")
    print(f"  preview:    {pack.preview_path.name}")
    print(f"  print PDF:  {pack.print_pdf_path.name}")
    print(f"  calibration:{pack.calibration_pdf_path.name}")
    print(f"  cut lines:  {pack.cutlines_svg_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
