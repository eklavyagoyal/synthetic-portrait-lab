"""NiceGUI web front-end — a thin, reactive UI over :mod:`app.core`.

It exposes the same capability as the CLI/TUI: pick a provider/model, choose
demographic buckets and a distribution, preview the cost, confirm the spend and
watch a batch generate live. All domain logic lives in the engine; this module
only collects a :class:`BatchGenerationRequest`, renders progress events and
displays the resulting images.

Run with ``python -m app.gui.main``. Importing the module never starts a server.

Visual design — "Studio Noir, Aurora accent": a premium, dark-first, minimal
look built on the shared design system. The styling lives entirely in CSS +
class/prop hints; none of the engine wiring is touched.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from nicegui import ui

from app.core.buckets import AGE_BUCKETS, ETHNICITY_BUCKETS, GENDER_BUCKETS
from app.core.config import AppConfig
from app.core.generator import Generator, RunNotConfirmedError
from app.core.batch_planner import plan_batch
from app.core.prompt_builder import build_prompt, framing_label
import json
from app.core.models import (
    BatchGenerationRequest,
    DistributionMode,
    EventType,
    GenerationEvent,
    ItemStatus,
    ModelInfo,
)

APP_TITLE = "Synthetic Portrait Lab"

# --------------------------------------------------------------------------- #
# Design system — "Studio Noir, Aurora accent"
# --------------------------------------------------------------------------- #
# Palette
COLOR_CANVAS = "#0B0E14"
COLOR_SURFACE = "#151A23"
COLOR_ELEVATED = "#1C2230"
COLOR_HAIRLINE = "#2A3240"
COLOR_TEXT = "#E6EAF2"
COLOR_MUTED = "#8A93A6"
COLOR_VIOLET = "#8B5CF6"
COLOR_CYAN = "#22D3EE"
COLOR_EMERALD = "#34D399"
COLOR_ROSE = "#FB7185"
COLOR_AMBER = "#FBBF24"

# Demographic dimension accents
ACCENT_AGE = "#F59E0B"        # amber
ACCENT_GENDER = "#A78BFA"     # violet
ACCENT_ETHNICITY = "#2DD4BF"  # teal

_VARIATION_CHOICES = [
    ("0 · strict repeatability", 0),
    ("1 · low variation", 1),
    ("2 · moderate variation", 2),
    ("3 · high variation", 3),
]

_FRAMING_CHOICES = {
    75: "close headshot · head 75%",
    60: "standard headshot · head 60%",
    45: "loose headshot · head 45%",
    30: "upper body · head 30%",
}

_QUALITY_CHOICES = ["low", "medium", "high", "auto"]

# Distribution modes that require demographic bucket selections (everything but
# EXACT, which is request-only and not exposed in this lightweight UI).
_SELECTABLE_MODES = [
    DistributionMode.EVEN,
    DistributionMode.RANDOM,
    DistributionMode.WEIGHTED,
]

# Custom CSS — fonts, canvas background, rounded-2xl cards with hairline border +
# soft shadow on the elevated surface, refined scrollbars, aurora text/button.
_HEAD_HTML = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --canvas: {COLOR_CANVAS};
    --surface: {COLOR_SURFACE};
    --elevated: {COLOR_ELEVATED};
    --hairline: {COLOR_HAIRLINE};
    --text: {COLOR_TEXT};
    --muted: {COLOR_MUTED};
    --violet: {COLOR_VIOLET};
    --cyan: {COLOR_CYAN};
  }}

  html, body, .q-page-container, .nicegui-content {{
    background: var(--canvas) !important;
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    letter-spacing: 0.1px;
  }}

  /* Soft aurora glow anchored top-left behind the canvas. */
  body::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
      radial-gradient(60rem 40rem at 12% -8%, rgba(139, 92, 246, 0.16), transparent 60%),
      radial-gradient(52rem 34rem at 100% 0%, rgba(34, 211, 238, 0.10), transparent 55%);
  }}
  .nicegui-content {{ position: relative; z-index: 1; }}

  /* Cards: rounded-2xl, hairline border, soft shadow, elevated surface. */
  .studio-card {{
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 1.25rem !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset,
                0 18px 40px -24px rgba(0,0,0,0.85) !important;
    transition: border-color 160ms ease, box-shadow 160ms ease;
  }}
  .studio-card.elevated {{ background: var(--elevated) !important; }}

  /* Small uppercase muted section labels. */
  .section-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
  }}

  /* Aurora gradient text (brand wordmark). */
  .aurora-text {{
    background: linear-gradient(95deg, {COLOR_VIOLET} 0%, {COLOR_CYAN} 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
  }}

  /* Aurora gradient primary button. */
  .aurora-btn {{
    background: linear-gradient(95deg, {COLOR_VIOLET} 0%, {COLOR_CYAN} 100%) !important;
    color: #0B0E14 !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    border-radius: 0.85rem !important;
    box-shadow: 0 10px 30px -10px rgba(139, 92, 246, 0.55) !important;
    transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
  }}
  .aurora-btn:hover {{
    filter: brightness(1.06);
    transform: translateY(-1px);
    box-shadow: 0 16px 40px -12px rgba(34, 211, 238, 0.55) !important;
  }}

  /* Big estimated-cost total. */
  .cost-total {{
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.01em;
  }}
  .cost-total.ok {{ color: var(--text); }}
  .cost-total.unavailable {{ color: {COLOR_AMBER}; }}
  .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace !important; }}

  /* Aurora progress fill. */
  .aurora-progress .q-linear-progress__track {{
    background: rgba(255,255,255,0.06) !important;
  }}
  .aurora-progress .q-linear-progress__model {{
    background: linear-gradient(95deg, {COLOR_VIOLET} 0%, {COLOR_CYAN} 100%) !important;
  }}

  /* Gallery tiles: rounded, hairline, subtle hover lift. */
  .gallery-tile {{
    background: var(--elevated);
    border: 1px solid var(--hairline);
    border-radius: 0.9rem;
    padding: 6px;
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
  }}
  .gallery-tile:hover {{
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.55);
    box-shadow: 0 18px 32px -18px rgba(0,0,0,0.9);
  }}
  .gallery-tile img {{ border-radius: 0.6rem; }}
  .gallery-cap {{
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.66rem;
    color: var(--muted);
  }}

  /* Refined scrollbars. */
  *::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  *::-webkit-scrollbar-track {{ background: transparent; }}
  *::-webkit-scrollbar-thumb {{
    background: #313a4b; border-radius: 8px; border: 2px solid var(--canvas);
  }}
  *::-webkit-scrollbar-thumb:hover {{ background: #3d4860; }}

  /* Quasar field surfaces blend with the dark theme. */
  .q-field--outlined .q-field__control {{ border-radius: 0.75rem; }}
  .q-log {{
    background: #0E121B !important;
    border: 1px solid var(--hairline);
    border-radius: 0.75rem;
  }}
</style>
"""


def _model_options(models: list[ModelInfo]) -> dict[str, str]:
    """Map ``"provider/model_id"`` keys to human labels for ``ui.select``."""
    options: dict[str, str] = {}
    for m in models:
        price = (
            f"${m.price_per_image_usd:.4f}/img"
            if m.price_per_image_usd is not None
            else "price unknown"
        )
        options[f"{m.provider}/{m.model_id}"] = f"{m.display_name}  —  {price}"
    return options


def _split_model_key(key: str) -> tuple[str, str]:
    """Split a ``"provider/model_id"`` select value back into its parts."""
    provider, _, model_id = key.partition("/")
    return provider, model_id


def _image_data_uri(path: Path) -> str:
    """Read a saved image and encode it as an inline base64 data URI.

    Inlining avoids fragile static-file routes and works for every client.
    """
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _tint_chip_select(select: ui.select, accent: str) -> ui.select:
    """Tint a multi-select's chips with a dimension accent color.

    Uses an inline CSS custom property consumed by a scoped style rule so the
    selection chips render in the dimension's accent without touching globals.
    """
    select.props(f'use-chips color="white"')
    select.style(
        f"--chip-accent: {accent};"
    )
    # Scoped style: chips inside this field pick up the accent tint.
    select.classes("chip-tinted")
    return select


@ui.page("/")
def index() -> None:  # noqa: PLR0915 - one cohesive page builder
    """Build the single-page UI. Each connected client gets isolated state."""
    cfg = AppConfig.load()
    gen = Generator(cfg)

    models = cfg.pricing.list_models()
    model_options = _model_options(models)
    default_key = (
        f"{cfg.settings.default_provider}/{cfg.settings.default_model}"
        if f"{cfg.settings.default_provider}/{cfg.settings.default_model}" in model_options
        else (next(iter(model_options), None))
    )

    # Per-client mutable state (kept off `self`; this is a closure per client).
    state: dict[str, object] = {"busy": False}

    # Design system: palette + custom dark theme. Default to dark mode ON.
    ui.colors(
        primary=COLOR_VIOLET,
        secondary=COLOR_CYAN,
        accent=COLOR_CYAN,
        positive=COLOR_EMERALD,
        negative=COLOR_ROSE,
        warning=COLOR_AMBER,
        dark=COLOR_CANVAS,
    )
    ui.add_head_html(_HEAD_HTML)
    # Per-dimension chip tint rules (scoped to fields tagged .chip-tinted).
    ui.add_head_html(
        "<style>"
        ".chip-tinted .q-chip {"
        " background: color-mix(in srgb, var(--chip-accent) 22%, transparent) !important;"
        " color: var(--text) !important;"
        " border: 1px solid color-mix(in srgb, var(--chip-accent) 55%, transparent) !important;"
        " border-radius: 0.6rem !important; font-weight: 500; }"
        ".chip-tinted .q-chip .q-icon { color: var(--chip-accent) !important; }"
        "</style>"
    )

    dark = ui.dark_mode()
    dark.enable()  # dark-first by default

    # ------------------------------------------------------------------ #
    # Header / hero — gradient wordmark, tagline, dark/light toggle
    # ------------------------------------------------------------------ #
    with ui.header().classes(
        "items-center justify-between px-6 py-3"
    ).style(
        f"background: {COLOR_SURFACE}; border-bottom: 1px solid {COLOR_HAIRLINE};"
        " box-shadow: 0 10px 30px -24px rgba(0,0,0,0.9);"
    ):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.html(
                '<span style="display:inline-flex;width:0.7rem;height:0.7rem;'
                "border-radius:9999px;background:linear-gradient(95deg,"
                f"{COLOR_VIOLET},{COLOR_CYAN});box-shadow:0 0 16px rgba(139,92,246,0.7);"
                '"></span>'
            )
            with ui.column().classes("gap-0"):
                ui.label("SYNTHETIC PORTRAIT LAB").classes(
                    "aurora-text text-xl font-bold"
                ).style("letter-spacing: 0.12em;")
                ui.label("Synthetic portrait batch generator").classes(
                    "text-xs"
                ).style(f"color: {COLOR_MUTED};")
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("dark_mode").classes("text-sm").style(f"color: {COLOR_MUTED};")
            ui.switch(value=True, on_change=lambda e: dark.set_value(e.value)).props(
                "color=secondary keep-color"
            )
            ui.icon("light_mode").classes("text-sm").style(f"color: {COLOR_MUTED};")

    # ------------------------------------------------------------------ #
    # Reactive cost recomputation
    # ------------------------------------------------------------------ #
    def build_request() -> BatchGenerationRequest:
        """Assemble a validated request from the current widget values.

        Raises ``ValueError`` (or pydantic validation errors) on invalid input.
        """
        provider, model_id = _split_model_key(str(model_select.value or ""))
        dist_mode = DistributionMode(dist_select.value)
        weights = None
        if dist_mode == DistributionMode.WEIGHTED:
            weights = {}
            for bucket, input_field in weight_inputs.items():
                try:
                    weights[bucket] = float(input_field.value)
                except (ValueError, TypeError):
                    weights[bucket] = 1.0

        return BatchGenerationRequest(
            provider=provider,
            model_id=model_id,
            age_buckets=list(age_select.value or []),
            gender_buckets=list(gender_select.value or []),
            ethnicity_buckets=list(ethnicity_select.value or []),
            distribution_mode=dist_mode,
            total_count=int(batch_number.value or 0),
            weights=weights,
            variation_level=int(variation_select.value or 0),
            size=str(size_select.value or "1024x1024"),
            quality=str(quality_select.value or "medium"),
            head_height_pct=int(framing_select.value or 60),
            seed=int(seed_number.value) if seed_number.value not in (None, "") else None,
            concurrency=max(1, int(concurrency_number.value or 1)),
            output_dir=(output_input.value or None) or None,
            filename_prefix=str(prefix_input.value or "portrait"),
            retry_failed=bool(retry_switch.value),
            max_retries=int(retries_number.value or 3),
            save_prompt=bool(save_prompt_switch.value),
            background=str(opt_bg.value or "plain light gray or off-white background"),
            expression=str(opt_exp.value or "neutral, natural facial expression"),
            lighting=str(opt_light.value or "natural studio lighting"),
            image_style=str(opt_style.value or "photorealistic passport-style studio portrait"),
            extra_positive_constraints=[line.strip() for line in (opt_pos.value or "").splitlines() if line.strip()],
            extra_negative_constraints=[line.strip() for line in (opt_neg.value or "").splitlines() if line.strip()],
        )

    def refresh_cost() -> None:
        """Recompute the model + cost summary card from current inputs."""
        cost_card.clear()
        with cost_card:
            key = str(model_select.value or "")
            if not key:
                ui.label("Select a model to see pricing.").classes("text-sm").style(
                    f"color: {COLOR_MUTED};"
                )
                return
            provider, model_id = _split_model_key(key)
            try:
                info = cfg.pricing.get_model_info(provider, model_id)
                sizes = info.supports_size or ["1024x1024"]
                size_select.options = sizes
                if size_select.value not in sizes:
                    size_select.value = info.default_size if info.default_size in sizes else sizes[0]
            except Exception:  # noqa: BLE001 - unknown model => no pricing
                ui.label("pricing unavailable — model not in registry").classes(
                    "text-sm text-warning"
                )
                return

            # Estimate first so the headline total can lead the card.
            estimate = None
            estimate_error: Optional[Exception] = None
            try:
                req = build_request()
                estimate = cfg.pricing.estimate(req)
            except Exception as exc:  # noqa: BLE001 - invalid selection so far
                estimate_error = exc

            ui.label("Estimated total").classes("section-label")
            if estimate_error is not None:
                ui.label("—").classes("cost-total ok")
                ui.label(f"enter valid settings ({estimate_error})").classes(
                    "text-xs"
                ).style(f"color: {COLOR_MUTED};")
            elif estimate is not None and estimate.pricing_available:
                amount = (
                    f"~${estimate.estimated_total_usd:.2f}"
                    if estimate.estimated_total_usd is not None
                    else "~$0.00"
                )
                ui.label(amount).classes("cost-total ok")
                ui.label(estimate.human_summary()).classes("text-xs mono").style(
                    f"color: {COLOR_MUTED};"
                )
            else:
                ui.label("unavailable").classes("cost-total unavailable")
                ui.label("pricing data missing — explicit confirmation required").classes(
                    "text-xs"
                ).style(f"color: {COLOR_MUTED};")
                if estimate is not None and estimate.warning:
                    ui.label(estimate.warning).classes("text-xs").style(
                        f"color: {COLOR_MUTED};"
                    )

            ui.separator().style(f"background: {COLOR_HAIRLINE};")

            # Detail grid: provider / model / per-image / planned count.
            per_image = (
                f"${info.price_per_image_usd:.4f}"
                if info.price_per_image_usd is not None
                else "unknown"
            )
            planned = str(estimate.total_count) if estimate is not None else "—"
            with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-3"):
                for lbl, val in (
                    ("Provider", provider),
                    ("Model", info.display_name),
                    ("Price / image", per_image),
                    ("Images planned", planned),
                ):
                    with ui.column().classes("gap-0"):
                        ui.label(lbl).classes("section-label")
                        ui.label(val).classes("text-sm").style(
                            f"color: {COLOR_TEXT};"
                        )

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    with ui.column().classes("w-full max-w-6xl mx-auto gap-5 p-6"):
        with ui.row().classes("w-full gap-5 items-stretch no-wrap max-[1024px]:flex-wrap"):
            # ---- Generation settings (primary column) ---------------- #
            with ui.card().classes("studio-card flex-1 p-6 gap-5").style(
                "min-width: 360px;"
            ):
                with ui.column().classes("gap-1"):
                    ui.label("Generation settings").classes(
                        "text-lg font-semibold"
                    ).style(f"color: {COLOR_TEXT};")
                    ui.label("Compose a batch and a distribution").classes(
                        "text-xs"
                    ).style(f"color: {COLOR_MUTED};")

                # Model + Size + Quality
                with ui.row().classes("w-full gap-4 items-start no-wrap max-[640px]:flex-wrap"):
                    with ui.column().classes("flex-[2] gap-1"):
                        ui.label("Model").classes("section-label")
                        model_select = ui.select(
                            model_options, value=default_key, on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Size").classes("section-label")
                        size_select = ui.select(
                            ["1024x1024"], value="1024x1024", on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Quality").classes("section-label")
                        quality_select = ui.select(
                            _QUALITY_CHOICES, value="medium", on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")

                # Batch + distribution + Framing + Variation
                with ui.row().classes("w-full gap-4 items-start no-wrap max-[640px]:flex-wrap"):
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Batch size").classes("section-label")
                        batch_number = ui.number(
                            value=4, min=1, step=1, precision=0, on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Distribution").classes("section-label")
                        dist_select = ui.select(
                            {m.value: m.value.capitalize() for m in _SELECTABLE_MODES}, value=DistributionMode.EVEN.value, on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")
                with ui.row().classes("w-full gap-4 items-start no-wrap max-[640px]:flex-wrap"):
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Framing").classes("section-label")
                        framing_select = ui.select(
                            _FRAMING_CHOICES, value=60, on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Variation").classes("section-label")
                        variation_select = ui.select(
                            {v: k for k, v in _VARIATION_CHOICES}, value=0, on_change=lambda _: refresh_cost(),
                        ).props("outlined dense").classes("w-full")

                ui.separator().style(f"background: {COLOR_HAIRLINE};")

                # Demographic buckets
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Demographics").classes("section-label")
                with ui.row().classes("w-full gap-4 items-start no-wrap max-[768px]:flex-wrap"):
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Age").classes("section-label").style(f"color: {ACCENT_AGE};")
                        age_select = ui.select(AGE_BUCKETS, value=[AGE_BUCKETS[1]], multiple=True, on_change=lambda _: refresh_cost()).props("outlined dense").classes("w-full")
                        _tint_chip_select(age_select, ACCENT_AGE)
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Gender").classes("section-label").style(f"color: {ACCENT_GENDER};")
                        gender_select = ui.select(GENDER_BUCKETS, value=[GENDER_BUCKETS[0], GENDER_BUCKETS[1]], multiple=True, on_change=lambda _: refresh_cost()).props("outlined dense").classes("w-full")
                        _tint_chip_select(gender_select, ACCENT_GENDER)
                    with ui.column().classes("flex-1 gap-1"):
                        ui.label("Ethnicity").classes("section-label").style(f"color: {ACCENT_ETHNICITY};")
                        ethnicity_select = ui.select(ETHNICITY_BUCKETS, value=[ETHNICITY_BUCKETS[0]], multiple=True, on_change=lambda _: refresh_cost()).props("outlined dense").classes("w-full")
                        _tint_chip_select(ethnicity_select, ACCENT_ETHNICITY)

                # Weights Container (Dynamic)
                weight_inputs: dict[str, ui.number] = {}
                weights_card = ui.card().classes("w-full gap-2 p-4 mt-2").style(f"background: {COLOR_ELEVATED}; border: 1px solid {COLOR_HAIRLINE}; border-radius: 0.75rem;")
                weights_card.set_visibility(False)
                
                with weights_card:
                    ui.label("Weights (Weighted Distribution)").classes("section-label text-xs")
                    weights_grid = ui.grid(columns=3).classes("w-full gap-3")
                
                def update_weights_ui(*args):
                    mode = dist_select.value
                    if mode != DistributionMode.WEIGHTED.value:
                        weights_card.set_visibility(False)
                        return
                    weights_card.set_visibility(True)
                    weights_grid.clear()
                    selected = (age_select.value or []) + (gender_select.value or []) + (ethnicity_select.value or [])
                    with weights_grid:
                        for b in selected:
                            if b not in weight_inputs:
                                weight_inputs[b] = ui.number(value=1.0, step=0.1, precision=1, on_change=lambda _: refresh_cost()).props('outlined dense')
                            else:
                                existing_val = weight_inputs[b].value
                                weight_inputs[b] = ui.number(value=existing_val, step=0.1, precision=1, on_change=lambda _: refresh_cost()).props('outlined dense')
                            with ui.row().classes("items-center gap-2"):
                                ui.label(b).classes("text-xs flex-1 truncate")
                                weight_inputs[b].classes("w-16")

                dist_select.on_value_change(update_weights_ui)
                age_select.on_value_change(update_weights_ui)
                gender_select.on_value_change(update_weights_ui)
                ethnicity_select.on_value_change(update_weights_ui)

                ui.separator().style(f"background: {COLOR_HAIRLINE};")

                with ui.expansion("Advanced Prompt Options").classes("w-full text-sm"):
                    with ui.column().classes("w-full gap-3 py-2"):
                        opt_bg = ui.input(value="plain light gray or off-white background").props('outlined dense label="Background"').classes("w-full")
                        opt_exp = ui.input(value="neutral, natural facial expression").props('outlined dense label="Expression"').classes("w-full")
                        opt_light = ui.input(value="natural studio lighting").props('outlined dense label="Lighting"').classes("w-full")
                        opt_style = ui.input(value="photorealistic passport-style studio portrait").props('outlined dense label="Image Style"').classes("w-full")
                        with ui.row().classes("w-full gap-4"):
                            opt_pos = ui.textarea().props('outlined dense label="Extra positive constraints"').classes("flex-1")
                            opt_neg = ui.textarea().props('outlined dense label="Extra negative constraints"').classes("flex-1")

                ui.separator().style(f"background: {COLOR_HAIRLINE};")

                # Output
                ui.label("Output").classes("section-label")
                with ui.row().classes("w-full gap-4 items-start no-wrap max-[640px]:flex-wrap"):
                    output_input = ui.input(
                        placeholder="leave blank for a timestamped folder",
                    ).props('outlined dense label="Output directory (optional)"').classes(
                        "flex-1"
                    )
                    prefix_input = ui.input(
                        value="portrait",
                    ).props('outlined dense label="Filename prefix"').classes("flex-1")

                with ui.row().classes("w-full gap-4 items-start no-wrap max-[640px]:flex-wrap"):
                    seed_number = ui.number(value=None, step=1, precision=0).props('outlined dense label="Seed (optional)"').classes("flex-1")
                    concurrency_number = ui.number(value=2, min=1, max=32, step=1, precision=0).props('outlined dense label="Concurrency"').classes("flex-1")
                    retries_number = ui.number(value=3, min=0, max=10, step=1, precision=0).props('outlined dense label="Max Retries"').classes("w-24")
                
                with ui.row().classes("w-full gap-6 items-center"):
                    retry_switch = ui.switch("Retry Failed", value=True).props("color=secondary")
                    save_prompt_switch = ui.switch("Save Prompt", value=True).props("color=secondary")

            # ---- Side column: cost + actions ------------------------- #
            with ui.column().classes("gap-5").style("width: 360px; min-width: 320px;"):
                # Model & cost summary — prominent
                with ui.card().classes("studio-card elevated w-full p-6 gap-3"):
                    ui.label("Model & cost").classes("section-label")
                    cost_card = ui.column().classes("w-full gap-2")

                # Actions
                with ui.card().classes("studio-card w-full p-5 gap-3"):
                    generate_button = ui.button(
                        "Generate", on_click=lambda: open_confirm()
                    ).classes("aurora-btn w-full").props("unelevated no-caps size=lg")
                    with ui.row().classes("w-full gap-2 no-wrap"):
                        preview_button = ui.button("Plan Preview", on_click=open_plan_preview, icon="visibility").props("flat no-caps color=secondary").classes("flex-1")
                        archive_button = ui.button("Archive", on_click=open_archive, icon="history").props("flat no-caps color=secondary").classes("flex-1")
                    with ui.row().classes("w-full gap-2 no-wrap"):
                        metadata_button = ui.button("Metadata", on_click=lambda: open_metadata(), icon="description").props("flat no-caps color=secondary").classes("flex-1")
                        metadata_button.disable()
                        output_button = ui.button("Output", on_click=lambda: show_output(), icon="folder_open").props("flat no-caps color=secondary").classes("flex-1")
                        output_button.disable()

        # ---- Progress ------------------------------------------------ #
        with ui.card().classes("studio-card w-full p-6 gap-3"):
            ui.label("Progress").classes("section-label")
            progress_bar = ui.linear_progress(
                value=0.0, show_value=False, size="10px"
            ).classes("w-full aurora-progress").style("border-radius: 9999px;")
            with ui.row().classes("w-full gap-6 items-center justify-between"):
                current_label = ui.label("Idle.").classes("text-sm flex-1").style(
                    f"color: {COLOR_TEXT};"
                )
                with ui.row().classes("items-center gap-3 no-wrap"):
                    counts_label = ui.label("0 ✓ / 0 ✗").classes(
                        "text-sm font-semibold mono"
                    ).style(f"color: {COLOR_MUTED};")
            log = ui.log(max_lines=200).classes("w-full h-40 text-xs mono")

        # ---- Preview gallery ----------------------------------------- #
        with ui.card().classes("studio-card w-full p-6 gap-3"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Preview").classes("section-label")
            gallery = ui.row().classes("w-full gap-3 flex-wrap")

    refresh_cost()

    # ------------------------------------------------------------------ #
    # Output / metadata affordances
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Plan Preview / Lightbox / Archive
    # ------------------------------------------------------------------ #
    def show_lightbox(result, img_path):
        with ui.dialog() as dialog, ui.card().classes("studio-card elevated w-full max-w-4xl p-0 gap-0 overflow-hidden"):
            with ui.row().classes("w-full no-wrap"):
                ui.image(_image_data_uri(img_path)).classes("w-1/2 h-[512px] object-cover pointer-events-none")
                with ui.column().classes("w-1/2 p-6 gap-4 overflow-y-auto h-[512px]"):
                    with ui.row().classes("w-full justify-between items-start"):
                        ui.label(result.id).classes("text-lg font-bold aurora-text")
                        ui.button(icon="close", on_click=dialog.close).props("flat round dense")
                    ui.label(f"Provider: {result.provider} / {result.model}").classes("text-xs mono text-muted")
                    with ui.row().classes("gap-2 text-xs"):
                        ui.label(result.age_bucket).classes("q-chip").style(f"background: color-mix(in srgb, {ACCENT_AGE} 22%, transparent); border: 1px solid {ACCENT_AGE}; border-radius: 0.6rem; padding: 2px 6px;")
                        ui.label(result.gender_bucket).classes("q-chip").style(f"background: color-mix(in srgb, {ACCENT_GENDER} 22%, transparent); border: 1px solid {ACCENT_GENDER}; border-radius: 0.6rem; padding: 2px 6px;")
                        ui.label(result.ethnicity_bucket).classes("q-chip").style(f"background: color-mix(in srgb, {ACCENT_ETHNICITY} 22%, transparent); border: 1px solid {ACCENT_ETHNICITY}; border-radius: 0.6rem; padding: 2px 6px;")
                    ui.label(f"Size: {result.size} | Quality: {result.quality} | Variation: {result.variation_level}").classes("text-xs text-muted")
                    ui.separator().style(f"background: {COLOR_HAIRLINE};")
                    ui.label("Prompt").classes("section-label")
                    ui.code(result.prompt).classes("w-full text-xs overflow-auto")
        dialog.open()

    def open_plan_preview():
        try:
            req = build_request()
            sampled = req.distribution_mode in (DistributionMode.RANDOM, DistributionMode.WEIGHTED) and req.seed is None
            plan_req = req.model_copy(update={"seed": 1729}) if sampled else req
            plan = plan_batch(plan_req)
        except Exception as exc:
            ui.notify(f"Cannot preview plan: {exc}", type="negative")
            return
            
        with ui.dialog() as dialog, ui.card().classes("studio-card elevated w-full max-w-2xl p-6 gap-4"):
            title = "Expected Plan (Sampled Preview)" if sampled else "Exact Plan"
            ui.label(f"{title} — {len(plan)} images").classes("text-lg font-semibold aurora-text")
            
            with ui.column().classes("w-full gap-2 max-h-96 overflow-y-auto"):
                for item in plan[:10]:
                    opts = item.prompt_options
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"#{item.index+1}").classes("text-xs mono text-muted w-8")
                        ui.label(opts.age_bucket).classes("text-xs").style(f"color: {ACCENT_AGE};")
                        ui.label("·").classes("text-muted")
                        ui.label(opts.gender_bucket).classes("text-xs").style(f"color: {ACCENT_GENDER};")
                        ui.label("·").classes("text-muted")
                        ui.label(opts.ethnicity_bucket).classes("text-xs").style(f"color: {ACCENT_ETHNICITY};")
                if len(plan) > 10:
                    ui.label(f"... and {len(plan)-10} more.").classes("text-xs text-muted")
            
            ui.separator().style(f"background: {COLOR_HAIRLINE};")
            ui.label("Sample Prompt (#1)").classes("section-label")
            ui.code(build_prompt(plan[0].prompt_options)).classes("text-xs w-full max-h-40 overflow-auto")
            
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props("flat no-caps color=secondary")
        dialog.open()

    def open_archive():
        runs = []
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for d in sorted(outputs_dir.iterdir(), reverse=True):
                if d.is_dir() and (d / "manifest.json").exists():
                    try:
                        mani = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                        runs.append(mani)
                    except Exception:
                        pass
                        
        with ui.dialog() as dialog, ui.card().classes("studio-card elevated w-full max-w-4xl p-6 gap-4"):
            ui.label("Archive").classes("text-lg font-semibold aurora-text")
            if not runs:
                ui.label("No previous runs found.").classes("text-muted")
            else:
                with ui.column().classes("w-full gap-2 max-h-[600px] overflow-y-auto"):
                    for r in runs:
                        dt = r.get('created_at', '')[:16].replace('T', ' ')
                        summary = r.get('summary', {})
                        cost = summary.get('provider_reported_cost_usd') or summary.get('estimated_cost_from_attempts_usd') or 0.0
                        with ui.card().classes("w-full p-3 gap-1 cursor-pointer gallery-tile").on('click', lambda run_dir=r.get('output_dir'): ui.notify(f"View {run_dir} in files")):
                            with ui.row().classes("w-full justify-between items-center"):
                                ui.label(dt).classes("text-sm font-bold text-white")
                                ui.label(f"${cost:.2f}").classes("text-sm font-bold ok")
                            with ui.row().classes("w-full justify-between text-xs text-muted"):
                                ui.label(f"{r.get('provider')}/{r.get('model')} · {r.get('size')} · {r.get('quality')}")
                                ui.label(f"{summary.get('successful_outputs', 0)} ✓ / {summary.get('failed_outputs', 0)} ✗")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props("flat no-caps color=secondary")
        dialog.open()

    def show_output() -> None:
        run = state.get("run")
        if run is None:
            ui.notify("No run yet.", type="warning")
            return
        ui.notify(f"Output: {run.output_dir}", type="info", multi_line=True)

    def open_metadata() -> None:
        run = state.get("run")
        if run is None:
            ui.notify("No run yet.", type="warning")
            return
        out_dir = Path(run.output_dir)
        jsonl = out_dir / "metadata.jsonl"
        csv_path = out_dir / "metadata.csv"
        if jsonl.exists():
            text = jsonl.read_text(encoding="utf-8")
            source = "metadata.jsonl"
        elif csv_path.exists():
            text = csv_path.read_text(encoding="utf-8")
            source = "metadata.csv"
        else:
            ui.notify("No metadata written yet.", type="warning")
            return

        with ui.dialog() as dialog, ui.card().classes(
            "studio-card w-full max-w-3xl p-6 gap-2"
        ):
            ui.label(f"Run metadata · {source}").classes("text-lg font-semibold").style(
                f"color: {COLOR_TEXT};"
            )
            ui.label(str(out_dir)).classes("text-xs mono").style(
                f"color: {COLOR_MUTED};"
            )
            ui.code(text).classes("w-full max-h-96 overflow-auto text-xs")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props(
                    "flat no-caps color=secondary"
                )
        dialog.open()

    # ------------------------------------------------------------------ #
    # Confirmation + generation
    # ------------------------------------------------------------------ #
    def open_confirm() -> None:
        if state.get("busy"):
            ui.notify("A run is already in progress.", type="warning")
            return
        try:
            req = build_request()
            run = gen.create_run(req)
        except RunNotConfirmedError as exc:  # pragma: no cover - defensive
            ui.notify(str(exc), type="negative")
            return
        except Exception as exc:  # noqa: BLE001 - surface validation clearly
            ui.notify(f"Cannot start: {exc}", type="negative", multi_line=True)
            return

        estimate = run.estimate
        with ui.dialog() as dialog, ui.card().classes(
            "studio-card elevated p-6 gap-4"
        ).style("min-width: 26rem;"):
            ui.label("Confirm generation").classes("text-lg font-semibold").style(
                f"color: {COLOR_TEXT};"
            )
            ui.label(
                f"{run.total} image(s) · {req.provider}/{req.model_id}"
            ).classes("text-sm mono").style(f"color: {COLOR_MUTED};")

            ui.separator().style(f"background: {COLOR_HAIRLINE};")

            ui.label("Estimated cost").classes("section-label")
            if estimate.pricing_available:
                amount = (
                    f"~${estimate.estimated_total_usd:.2f}"
                    if estimate.estimated_total_usd is not None
                    else "~$0.00"
                )
                ui.label(amount).classes("cost-total ok")
                ui.label(estimate.human_summary()).classes("text-xs mono").style(
                    f"color: {COLOR_MUTED};"
                )
            else:
                ui.label("unavailable").classes("cost-total unavailable")
                ui.label("pricing data missing").classes("text-xs").style(
                    f"color: {COLOR_MUTED};"
                )
                if estimate.warning:
                    ui.label(estimate.warning).classes("text-xs").style(
                        f"color: {COLOR_MUTED};"
                    )

            ui.separator().style(f"background: {COLOR_HAIRLINE};")
            with ui.row().classes("w-full justify-end gap-2 items-center"):
                ui.button("Cancel", on_click=dialog.close).props(
                    "flat no-caps color=secondary"
                )

                async def _confirm() -> None:
                    dialog.close()
                    await start_generation(run)

                ui.button("Confirm", on_click=_confirm).classes("aurora-btn").props(
                    "unelevated no-caps"
                )
        dialog.open()

    async def start_generation(run) -> None:
        """Confirm the run and execute it, streaming events into the UI."""
        state["busy"] = True
        state["run"] = run
        generate_button.disable()
        metadata_button.disable()
        output_button.disable()
        gallery.clear()
        log.clear()
        progress_bar.set_value(0.0)
        current_label.set_text("Starting…")
        counts_label.set_text("0 ✓ / 0 ✗")

        def on_event(event: GenerationEvent) -> None:
            # Same event loop as the page — safe to mutate elements directly.
            if event.message:
                log.push(event.message)
            if event.total:
                progress_bar.set_value(run.progress)
            counts_label.set_text(
                f"{event.success_count} ✓ / {event.failure_count} ✗"
            )

            if event.type == EventType.ITEM_STARTED:
                current_label.set_text(event.message)
            elif event.type == EventType.ITEM_SUCCEEDED and event.result:
                result = event.result
                if result.status == ItemStatus.SUCCESS and result.filename:
                    img_path = Path(run.output_dir) / "images" / result.filename
                    if img_path.exists():
                        with gallery:
                            with ui.column().classes("items-center gap-1 gallery-tile cursor-pointer").on('click', lambda r=result, p=img_path: show_lightbox(r, p)):
                                ui.image(_image_data_uri(img_path)).classes(
                                    "w-32 h-32 object-cover pointer-events-none"
                                )
                                ui.label(result.id).classes("gallery-cap")
            elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_CANCELLED):
                progress_bar.set_value(1.0 if run.total else 0.0)
                current_label.set_text(event.message or "Done.")

        try:
            gen.confirm(run)
            await gen.execute(run, on_event=on_event)
            ui.notify(
                f"Completed: {run.success_count} ✓ / {run.failure_count} ✗ "
                f"→ {run.output_dir}",
                type="positive",
                multi_line=True,
            )
            metadata_button.enable()
            output_button.enable()
        except RunNotConfirmedError as exc:  # pragma: no cover - defensive
            ui.notify(str(exc), type="negative")
            current_label.set_text("Not confirmed.")
        except Exception as exc:  # noqa: BLE001 - auth/validation/runtime
            ui.notify(f"Generation failed: {exc}", type="negative", multi_line=True)
            current_label.set_text(f"Failed: {exc}")
            log.push(f"ERROR: {exc}")
        finally:
            state["busy"] = False
            generate_button.enable()


def main(argv: Optional[list[str]] = None) -> None:
    """Launch the GUI (blocking).

    Defaults to a **native desktop window** rendered with the operating system's
    built-in webview (via ``pywebview``) — no browser tab, and the local server it
    uses is bound to loopback only (nothing is exposed to the network). Pass
    ``--web`` to run it as a local web app in your browser instead. If the native
    webview is unavailable (e.g. ``pywebview`` not installed or no display), it
    falls back to web mode with a clear message.
    """
    import argparse
    import importlib.util
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m app.gui.main", description=f"{APP_TITLE} — desktop GUI"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run as a local web app (opens a browser tab) instead of a native window.",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for web mode (default: %(default)s)."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Web mode: do not auto-open the browser."
    )
    args = parser.parse_args(argv)

    native_available = importlib.util.find_spec("webview") is not None
    if not args.web and native_available:
        # Native desktop window. NiceGUI manages the local server + webview; it
        # picks a free loopback port internally, so nothing is published.
        ui.run(reload=False, title=APP_TITLE, native=True, window_size=(1280, 900))
        return

    if not args.web and not native_available:
        print(
            "Native window unavailable: 'pywebview' is not installed.\n"
            "  Install it for a native desktop window:  pip install pywebview\n"
            f"  Falling back to local web mode at http://localhost:{args.port}",
            file=sys.stderr,
        )

    # Local web mode (loopback only — not reachable from the network).
    ui.run(reload=False, title=APP_TITLE, port=args.port, show=not args.no_show)


if __name__ in {"__main__", "__mp_main__"}:
    main()
