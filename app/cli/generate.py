"""Headless ``argparse`` CLI over the generation engine.

This is a thin front-end: it parses flags into a :class:`BatchGenerationRequest`,
asks :class:`Generator` to plan + estimate, shows the cost confirmation block,
and (after confirmation) drives :meth:`Generator.execute`. All policy — bucket
validation, the no-spend-before-confirm rule, retries, metadata — lives in the
engine; this module only renders and prompts.

Usage::

    python -m app.cli.generate --provider mock --model mock-image --count 8 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional, Sequence

from ..core.buckets import BucketValidationError
from ..core.config import AppConfig
from ..core.generator import Generator, RunNotConfirmedError
from ..core.iris import IrisRealismOptions
from ..core.models import (
    BatchGenerationRequest,
    CaptureModality,
    DistributionMode,
    EventType,
    GenerationEvent,
    Run,
)
from ..core.pricing import UnknownModelError
from ..core.prompt_builder import framing_label

_AFFIRMATIVE = {"yes", "y"}

# Framing presets -> head-height percentage (top of hair to chin).
_FRAMING_PCT = {"close": 75, "standard": 60, "loose": 45, "upper-body": 30}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser(cfg: AppConfig) -> argparse.ArgumentParser:
    """Build the argument parser, defaulting provider/model from config."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.generate",
        description="Generate a batch of demographic portrait images.",
    )
    parser.add_argument(
        "--provider",
        default=cfg.settings.default_provider,
        help="Image provider (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=cfg.settings.default_model,
        help="Model id within the provider (default: %(default)s).",
    )
    parser.add_argument(
        "--modality",
        choices=[m.value for m in CaptureModality],
        default=CaptureModality.RGB_FACE.value,
        help="Imaging modality: 'rgb' (colour face portrait) or 'ir' "
        "(monochrome near-infrared iris capture). Default: %(default)s.",
    )
    iris_group = parser.add_argument_group(
        "IR iris realism (opt-in; only affects --modality ir)",
        "Each flag mixes a non-ideal capture condition into a realistic fraction "
        "of the batch (with per-image variety). Off by default.",
    )
    iris_group.add_argument(
        "--ir-occlusion", action="store_true",
        help="Drooping eyelids / eyelashes occluding the iris.",
    )
    iris_group.add_argument(
        "--ir-off-gaze", action="store_true",
        help="Slight off-axis gaze (iris a mild ellipse); stays readable.",
    )
    iris_group.add_argument(
        "--ir-lenses", action="store_true",
        help="Contact lenses (soft / hard / cosmetic / painted).",
    )
    iris_group.add_argument(
        "--ir-conditions", action="store_true",
        help="Minor eye / iris / sclera conditions (arcus, pterygium, cataract, ...).",
    )
    iris_group.add_argument(
        "--ir-glasses", action="store_true",
        help="Spectacles with heavy glare / distortion over the eye.",
    )
    iris_group.add_argument(
        "--ir-makeup", action="store_true",
        help="Moderate to strong eye makeup around the eye.",
    )
    parser.add_argument("--count", type=int, help="Total number of images to generate.")
    parser.add_argument(
        "--variation",
        type=int,
        choices=range(0, 4),
        default=0,
        metavar="{0,1,2,3}",
        help="Variation level 0-3 (default: 0).",
    )
    parser.add_argument("--output", help="Output directory (default: a timestamped dir).")
    parser.add_argument(
        "--distribution",
        choices=[m.value for m in DistributionMode],
        default=DistributionMode.EVEN.value,
        help="How to spread the count across buckets (default: %(default)s).",
    )
    parser.add_argument(
        "--age",
        action="append",
        default=None,
        metavar="BUCKET",
        help='Age bucket; repeatable (e.g. --age "adult, 26 to 40"). Defaults to all configured.',
    )
    parser.add_argument(
        "--gender",
        action="append",
        default=None,
        metavar="BUCKET",
        help="Gender bucket; repeatable. Defaults to all configured.",
    )
    parser.add_argument(
        "--ethnicity",
        action="append",
        default=None,
        metavar="BUCKET",
        help="Ethnicity bucket; repeatable. Defaults to all configured.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="TOKEN=VALUE",
        help="Bucket weight for weighted distribution (repeatable), e.g. --weight East\\ Asian=2.",
    )
    parser.add_argument("--size", default="1024x1024", help="Image size (default: %(default)s).")
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "auto"],
        default="medium",
        help="Render quality; drives price on token-billed models (default: %(default)s).",
    )
    parser.add_argument(
        "--framing",
        choices=list(_FRAMING_PCT),
        help="Head-size framing preset: close 75, standard 60, loose 45, upper-body 30 "
        "(default: standard).",
    )
    parser.add_argument(
        "--head-height-pct",
        type=int,
        metavar="PCT",
        help="Head height (top of hair to chin) as a share of image height, 20-90. "
        "Overrides --framing.",
    )
    parser.add_argument(
        "--prefix", default="portrait", help="Filename prefix (default: %(default)s)."
    )
    parser.add_argument("--seed", type=int, help="Base seed for reproducibility.")
    parser.add_argument(
        "--no-diversify",
        action="store_true",
        help="Disable the per-image appearance layer (faces will look alike on "
        "seedless models). Diversification is ON by default.",
    )
    parser.add_argument(
        "--no-dedup-history",
        action="store_true",
        help="Do not dedup against prior runs in the output directory; still keeps "
        "the current batch internally unique.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Parallel requests (default: %(default)s)."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per failed item (default: %(default)s).",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable retrying failed items.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm the spend; skip the interactive prompt.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the models in the registry and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan + estimate and print the confirmation block without generating.",
    )
    return parser


def _collect_buckets(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Clean + de-duplicate repeatable bucket flags.

    Each ``--age/--gender/--ethnicity`` occurrence is ONE whole bucket. We do not
    split on commas: bucket names themselves contain commas (e.g. ``adult, 26 to
    40``), so splitting would corrupt them. Returns ``None`` when the flag was
    omitted entirely, so the caller can apply the "all configured buckets" default.
    """
    if values is None:
        return None
    out: list[str] = []
    for raw in values:
        token = raw.strip()
        if token and token not in out:
            out.append(token)
    return out


def _parse_weights(values: Optional[Sequence[str]]) -> Optional[dict[str, float]]:
    """Parse repeatable ``token=value`` weight flags into a mapping."""
    if not values:
        return None
    weights: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --weight {raw!r}; expected TOKEN=VALUE.")
        token, _, value = raw.partition("=")
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid --weight {raw!r}; token must be non-empty.")
        try:
            weights[token] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid --weight {raw!r}; {value!r} is not a number.") from exc
    return weights


def _build_request(args: argparse.Namespace, cfg: AppConfig) -> BatchGenerationRequest:
    """Translate parsed args into a validated :class:`BatchGenerationRequest`."""
    if args.count is None:
        raise ValueError("--count is required (number of images to generate).")

    age = _collect_buckets(args.age) or list(cfg.buckets.age)
    gender = _collect_buckets(args.gender) or list(cfg.buckets.gender)
    ethnicity = _collect_buckets(args.ethnicity) or list(cfg.buckets.ethnicity)

    distribution = DistributionMode(args.distribution)
    weights = _parse_weights(args.weight)
    if distribution == DistributionMode.WEIGHTED and not weights:
        raise ValueError("WEIGHTED distribution requires at least one --weight TOKEN=VALUE.")

    if args.head_height_pct is not None:
        head_height_pct = args.head_height_pct
    elif args.framing:
        head_height_pct = _FRAMING_PCT[args.framing]
    else:
        head_height_pct = 60
    if not (20 <= head_height_pct <= 90):
        raise ValueError("--head-height-pct must be between 20 and 90.")

    iris_realism = IrisRealismOptions(
        eyelid_occlusion=args.ir_occlusion,
        off_gaze=args.ir_off_gaze,
        contact_lenses=args.ir_lenses,
        ocular_conditions=args.ir_conditions,
        glasses=args.ir_glasses,
        eye_makeup=args.ir_makeup,
    )
    return BatchGenerationRequest(
        provider=args.provider,
        model_id=args.model,
        modality=CaptureModality(args.modality),
        iris_realism=iris_realism,
        age_buckets=age,
        gender_buckets=gender,
        ethnicity_buckets=ethnicity,
        distribution_mode=distribution,
        total_count=args.count,
        weights=weights,
        variation_level=args.variation,
        size=args.size,
        quality=args.quality,
        head_height_pct=head_height_pct,
        seed=args.seed,
        output_dir=args.output,
        filename_prefix=args.prefix,
        retry_failed=not args.no_retry,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        diversify=not args.no_diversify,
        dedup_history=not args.no_dedup_history,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _list_models(cfg: AppConfig) -> None:
    """Print every registered model: provider/model, price and supported sizes."""
    models = cfg.pricing.list_models()
    if not models:
        print("No models found in the registry.")
        return
    print("Available models:")
    for info in models:
        price = (
            f"${info.price_per_image_usd:.4f}/image"
            if info.price_per_image_usd is not None
            else "price unknown"
        )
        sizes = ", ".join(info.supports_size)
        print(f"  {info.provider}/{info.model_id}  [{info.display_name}]")
        print(f"      {price}  | sizes: {sizes}")


def _print_confirmation(run: Run) -> None:
    """Render the spend-confirmation block from the planned run."""
    est = run.estimate
    if est.pricing_available and est.price_per_image_usd is not None:
        per_image = f"${est.price_per_image_usd:.4f}"
    else:
        per_image = "unknown"

    pct = run.request.head_height_pct
    print(f"Provider: {run.request.provider}")
    print(f"Model: {run.model_info.display_name}")
    print(f"Modality: {run.request.modality.value}")
    print(f"Images: {run.total}")
    print(f"Size: {run.request.size}")
    print(f"Quality: {run.request.quality}")
    if run.request.modality == CaptureModality.IR_IRIS:
        print("Framing: near-infrared iris capture (4:3 landscape)")
    else:
        print(f"Framing: {framing_label(pct)} (head ~{pct}% of image height)")
    print(f"Estimated price per image: {per_image}")
    print(f"Estimated total cost: {est.human_summary()}")
    print("  (estimate is for planned outputs; retried/failed attempts are billed too)")
    print(f"Output directory: {run.output_dir}")
    if not est.pricing_available:
        print(
            "WARNING: pricing data is unavailable for this model. "
            "The cost shown is not reliable; confirm explicitly before spending."
            + (f"\n         {est.warning}" if est.warning else "")
        )


def _make_event_handler():
    """Return a concise ``on_event`` callback updating a single status line."""

    def on_event(event: GenerationEvent) -> None:
        if event.type in (EventType.ITEM_SUCCEEDED, EventType.ITEM_FAILED):
            done = (event.success_count or 0) + (event.failure_count or 0)
            total = event.total or 0
            line = (
                f"\r[{done}/{total}] "
                f"ok={event.success_count} fail={event.failure_count} "
                f"{event.item_id or ''}"
            )
            # Pad to clear any leftover characters from a longer previous line.
            sys.stdout.write(line.ljust(72)[:72])
            sys.stdout.flush()

    return on_event


def _print_summary(run: Run) -> None:
    """Print the final, honestly-labelled cost accounting and any failures."""
    print()  # finish the carriage-return status line
    print(
        f"Done: {run.successful_outputs} succeeded, {run.failed_outputs} failed "
        f"· {run.api_attempts} API attempts ({run.total_retries} retries)."
    )
    est = run.estimated_cost_before_run_usd
    burn = run.estimated_cost_from_attempts_usd
    billed = run.provider_reported_cost_usd
    if est is not None:
        print(f"  EST  (planned)        ~${est:.4f}  for {run.planned_outputs} outputs")
    if burn is not None:
        print(f"  BURN (from attempts)  ~${burn:.4f}  for {run.api_attempts} attempts (retries billed)")
    if billed is not None:
        print(f"  BILL (provider)        ${billed:.4f}")
    else:
        print("  BILL (provider)        unavailable — API returns no per-request $ (BURN is an estimate)")
    if run.provider_usage:
        print(f"  usage                  {run.provider_usage}")
    print(f"Output directory: {run.output_dir}")

    failures = [r for r in run.results if r.status.value == "failed"]
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r.id}: {r.error}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args, plan, confirm and (unless dry-run) generate. Returns an exit code."""
    cfg = AppConfig.load()
    parser = _build_parser(cfg)
    args = parser.parse_args(argv)

    if args.list_models:
        _list_models(cfg)
        return 0

    # Refuse early if a real provider was chosen without a configured key.
    if not cfg.settings.has_key_for(args.provider):
        print(
            f"Error: no API key configured for provider '{args.provider}'. "
            "Set the appropriate variable in your .env file (or use --provider mock).",
            file=sys.stderr,
        )
        return 2

    try:
        request = _build_request(args, cfg)
        gen = Generator(cfg)
        run = gen.create_run(request)
    except (BucketValidationError, UnknownModelError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_confirmation(run)

    if args.dry_run:
        return 0

    if not args.yes:
        try:
            answer = input("Proceed? [yes/no] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 0
        if answer not in _AFFIRMATIVE:
            print("Aborted.")
            return 0

    try:
        gen.confirm(run)
        run = asyncio.run(gen.execute(run, on_event=_make_event_handler()))
    except RunNotConfirmedError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1

    _print_summary(run)
    return 1 if run.failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
