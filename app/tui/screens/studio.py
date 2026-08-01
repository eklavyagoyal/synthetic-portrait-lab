"""STUDIO — the compose screen.

Three columns: configure (model/output) · demographics (+ weights, prompt
knobs, live prompt sample) · readout (estimate, plan preview, EXPOSE). Every
settings change re-plans via the pure ``plan_batch``/``pricing.estimate`` on a
150 ms debounce, so the readout column is always *exactly* what would run.

The plan-preview honesty rule: EVEN (or any seeded mode) shows ``EXACT PLAN``;
RANDOM/WEIGHTED without a seed shows a fixed-seed *sample* labelled as such —
a sample is never presented as the plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError
from rich.markup import escape as esc
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.validation import Integer, Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TextArea,
)

from app.core.batch_planner import plan_batch
from app.core.iris import IrisRealismOptions
from app.core.models import (
    BatchGenerationRequest,
    CaptureModality,
    DistributionMode,
    MaskPrintOptions,
    PlannedItem,
)
from app.core.prompt_builder import framing_label
from app.core.sizes import resolve_iris_capture_size

from .. import glyphs, labels, prefs
from ..widgets import BucketList, DistBars, Hero, MoneyBlock
from .modals import ModelPickerModal, PromptPeekModal, _model_key

PREVIEW_SEED = 1729  # fixed seed for *sampled* previews of unseeded random modes

_VARIATION_CHOICES = [
    ("0 · strict repeatability", 0),
    ("1 · low variation", 1),
    ("2 · moderate variation", 2),
    ("3 · high variation", 3),
]

# Framing presets: label -> head height (top of hair to chin) as % of image height.
_FRAMING_CHOICES = [
    ("close headshot · head 75%", 75),
    ("standard headshot · head 60%", 60),
    ("loose headshot · head 45%", 45),
    ("upper body · head 30%", 30),
]
_FRAMING_VALUES = {pct for _, pct in _FRAMING_CHOICES}

# Render quality — drives price on token-billed models (gpt-image).
_QUALITY_CHOICES = [
    ("low · cheapest", "low"),
    ("medium · default", "medium"),
    ("high · priciest", "high"),
    ("auto", "auto"),
]
_QUALITY_VALUES = {value for _, value in _QUALITY_CHOICES}

_DISTRIBUTION_CHOICES = [
    (DistributionMode.EVEN.value, DistributionMode.EVEN.value),
    (DistributionMode.RANDOM.value, DistributionMode.RANDOM.value),
    (DistributionMode.WEIGHTED.value, DistributionMode.WEIGHTED.value),
]

# Imaging modality — an open list (thermal etc. can be appended later), not a toggle.
_MODALITY_CHOICES = [
    ("RGB · colour face portrait", CaptureModality.RGB_FACE.value),
    ("IR · near-infrared iris (grayscale)", CaptureModality.IR_IRIS.value),
]
_MODALITY_VALUES = {value for _, value in _MODALITY_CHOICES}


@dataclass
class Problem:
    text: str
    blocking: bool = True


def _split_model_key(key: str) -> tuple[str, str]:
    provider, _, model_id = key.partition("::")
    return provider, model_id


class ModelCard(Static, can_focus=True):
    """The current model summary; click/enter opens the picker."""

    DEFAULT_CSS = """
    ModelCard {
        height: auto;
        padding: 0 1;
        border: round $border-soft;
        &:focus { border: round $secondary; background: $boost; }
        &:hover { border: round $border-strong; }
    }
    """

    def _open_picker(self) -> None:
        # Widget.run_action is async in Textual 8.x; call the screen action directly.
        action = getattr(self.screen, "action_model_picker", None)
        if action is not None:
            action()

    def on_click(self) -> None:
        self._open_picker()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            self._open_picker()


class WeightInput(Input):
    """A per-bucket weight field (weighted distribution only)."""

    def __init__(self, bucket: str, value: float) -> None:
        super().__init__(
            value=f"{value:g}",
            type="number",
            validators=[Number(minimum=0)],
            valid_empty=False,
            compact=True,
            classes="weight-input",
        )
        self.bucket = bucket


class StudioScreen(Screen):
    """Compose a batch; everything is preview until EXPOSE is confirmed."""

    AUTO_FOCUS = "#batch-size"

    BINDINGS = [
        # priority=True: these must win over the focused Input's emacs-style keys
        Binding("ctrl+g", "generate", "expose", priority=True,
                tooltip="Review cost and start the batch"),
        Binding("ctrl+e", "prompt_peek", "prompt", priority=True,
                tooltip="Preview the exact prompts"),
        Binding("ctrl+n", "model_picker", "model", priority=True,
                tooltip="Choose provider/model"),
        Binding("ctrl+r", "randomize_seed", "seed", priority=True,
                tooltip="Randomize the seed"),
        Binding("ctrl+o", "toggle_advanced", "advanced", show=False, priority=True,
                tooltip="Toggle advanced prompt options"),
    ]

    DEFAULT_CSS = """
    StudioScreen {
        background: $background;

        #studio-body { height: 1fr; padding: 0 1; layout: horizontal; }
        #col-configure { width: 36; padding: 0 1 0 0; }
        #col-demo { width: 1fr; min-width: 34; padding: 0 1 0 0; }
        #col-readout { width: 46; }

        &.-compact #studio-body { layout: vertical; overflow-y: auto; }
        &.-compact #col-configure, &.-compact #col-demo, &.-compact #col-readout {
            width: 100%; height: auto;
        }

        .card {
            height: auto;
            border: round $border-soft;
            border-title-color: $secondary;
            background: $surface;
            padding: 1 2;
            margin: 0 0 1 0;
        }
        .field-label { color: $text-muted; margin: 1 0 0 0; }
        .field-label:first-of-type { margin: 0; }
        Input, Select { margin: 0; }
        Input:focus { border: tall $secondary; }

        #model-card-box { padding: 1 1; }
        #model-display { height: auto; }

        .switch-row { height: auto; align-vertical: middle; }
        .switch-row Label { padding: 1 1 0 0; color: $text-muted; }
        .switch-row Input { width: 8; margin: 0 0 0 1; }

        #weights-box { height: auto; max-height: 12; }
        .weight-row { height: 1; margin: 0 0 0 0; }
        .weight-row Label { width: 16; color: $text-muted; }
        .weight-row Input { width: 10; }
        .weight-row .weight-share { width: 8; color: $text-muted; padding: 0 0 0 1; }

        #advanced-box Input { margin: 0 0 1 0; }
        #advanced-box TextArea { height: 4; margin: 0 0 1 0; }
        #portrait-style-fields { height: auto; }
        #iris-realism-fields { height: auto; }
        #mask-print-controls {
            height: auto;
            margin: 0 0 1 0;
            border: tall $primary;
            background: $surface;
        }
        #mask-print-fields { height: auto; padding: 0 1 1 1; }
        #mask-seg-intro { height: auto; color: $text-muted; margin: 0 0 1 0; }
        #mask-geometry-preview {
            height: auto;
            border: round $border-soft;
            background: $panel;
            padding: 1 1;
            margin: 0 0 1 0;
        }
        #mask-pipeline {
            height: auto;
            color: $text-muted;
            border-left: thick $secondary;
            padding: 0 0 0 1;
            margin: 0 0 1 0;
        }
        .mask-section-title {
            height: 1;
            color: $primary;
            text-style: bold;
            margin: 1 0 0 0;
        }
        .mask-measure-row { height: auto; }
        .mask-field { width: 1fr; height: auto; margin: 0 1 0 0; }
        .mask-field:last-child { margin: 0; }
        .mask-field Label { height: auto; color: $text-muted; }
        .mask-field Input { width: 1fr; margin: 0; }

        #prompt-sample-card { max-height: 14; }
        #prompt-sample { color: $text-muted; height: auto; }

        #plan-headline { height: auto; margin: 0 0 1 0; }
        #plan-items { height: auto; margin: 1 0 0 0; }

        #expose-row { height: auto; margin: 0 0 0 0; }
        #expose-btn {
            width: 1fr;
            background: $primary;
            color: $ink;
            text-style: bold;
            border: none;
            &:hover { background: $secondary; }
            &.-disabled { background: $panel; color: $text-disabled; }
        }
        #why-disabled { height: auto; color: $error; margin: 0 0 1 0; }
        #why-disabled.-warn-only { color: $warning; }
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._refresh_timer: Optional[Timer] = None
        self._weights: dict[str, float] = {}
        self._weights_built_for: tuple = ()
        self._model_key: str = ""
        self._problems: list[Problem] = []
        self._preview_plan: list[PlannedItem] = []
        self._run_locked = False

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    @property
    def _config(self):
        return self.app.config

    def compose(self) -> ComposeResult:
        config = self._config
        p: prefs.Prefs = self.app.prefs
        models = config.pricing.list_models()
        keys = {_model_key(m.provider, m.model_id) for m in models}
        want = _model_key(config.settings.default_provider, config.settings.default_model)
        if p.model_key in keys:
            self._model_key = p.model_key
        elif want in keys:
            self._model_key = want
        else:
            self._model_key = next(iter(sorted(keys)), "")

        yield Hero(active="studio")

        if not models:
            registry_path = config.settings.model_registry_path or "app/core/model_registry.json"
            with Vertical(id="studio-body"):
                yield Static(
                    f"[$error b]{glyphs.WARN}  No models in the registry[/]\n\n"
                    f"[$text-muted]The model registry at[/] {registry_path} "
                    f"[$text-muted]contains no providers.\n"
                    "Set PORTRAIT_MODEL_REGISTRY_PATH to a valid registry and restart.[/]",
                    classes="card",
                )
            yield Footer()
            return

        ages = prefs.validated_buckets(p.ages, config.buckets.age) or list(config.buckets.age)
        genders = (
            prefs.validated_buckets(p.genders, config.buckets.gender)
            or list(config.buckets.gender)
        )
        eths = (
            prefs.validated_buckets(p.ethnicities, config.buckets.ethnicity)
            or config.buckets.ethnicity[:2]
        )

        with Container(id="studio-body"):
            with VerticalScroll(id="col-configure"):
                with Vertical(classes="card", id="model-card-box") as card:
                    card.border_title = "model"
                    yield ModelCard(id="model-display")
                    yield Label("modality", classes="field-label")
                    yield Select(
                        _MODALITY_CHOICES,
                        value=p.modality if p.modality in _MODALITY_VALUES else "rgb",
                        allow_blank=False,
                        id="modality",
                    )
                    yield Label("size", classes="field-label", id="size-label")
                    yield Select(
                        [("1024x1024", "1024x1024")],
                        value="1024x1024",
                        allow_blank=False,
                        id="size",
                    )
                    yield Label("quality (affects price)", classes="field-label")
                    yield Select(
                        _QUALITY_CHOICES,
                        value=p.quality if p.quality in _QUALITY_VALUES else "medium",
                        allow_blank=False,
                        id="quality",
                    )
                    yield Label(
                        "framing · head height (hair→chin)",
                        classes="field-label",
                        id="framing-label",
                    )
                    yield Select(
                        _FRAMING_CHOICES,
                        value=p.head_height_pct if p.head_height_pct in _FRAMING_VALUES else 60,
                        allow_blank=False,
                        id="framing",
                    )
                    yield Label("batch size (images)", classes="field-label")
                    yield Input(
                        value=str(p.batch_size),
                        type="integer",
                        validators=[Integer(minimum=1, maximum=100_000)],
                        id="batch-size",
                    )
                    yield Label("variation level", classes="field-label")
                    yield Select(
                        _VARIATION_CHOICES, value=p.variation, allow_blank=False, id="variation"
                    )
                    yield Label("distribution", classes="field-label")
                    yield Select(
                        _DISTRIBUTION_CHOICES,
                        value=p.distribution,
                        allow_blank=False,
                        id="distribution",
                    )
                    yield Label("seed (blank = random)", classes="field-label")
                    yield Input(
                        placeholder="optional integer",
                        type="integer",
                        valid_empty=True,
                        id="seed",
                    )
                    yield Label("concurrency (1–32)", classes="field-label")
                    yield Input(
                        value=str(p.concurrency),
                        type="integer",
                        validators=[Integer(minimum=1, maximum=32)],
                        id="concurrency",
                    )
                with Vertical(classes="card") as card:
                    card.border_title = "output"
                    yield Label("directory (blank = auto)", classes="field-label")
                    yield Input(placeholder="auto (timestamped)", valid_empty=True, id="output-dir")
                    yield Label("filename prefix", classes="field-label")
                    yield Input(value=p.prefix, id="prefix")
                    with Horizontal(classes="switch-row"):
                        yield Switch(value=p.retry_failed, id="retry-switch")
                        yield Label("retry failed · max")
                        yield Input(
                            value=str(p.max_retries),
                            type="integer",
                            validators=[Integer(minimum=0, maximum=10)],
                            id="max-retries",
                        )
                    with Horizontal(classes="switch-row"):
                        yield Switch(value=p.save_prompt, id="save-prompt-switch")
                        yield Label("save prompt into metadata")
                with Collapsible(
                    title="3D MASK SEGMENTATION · measured shell → six print panels",
                    id="mask-print-controls",
                    collapsed=False,
                ):
                    with Vertical(id="mask-print-fields"):
                        with Horizontal(classes="switch-row"):
                            yield Switch(value=p.mask_print, id="mask-print-switch")
                            yield Label("enable local 3D mask segmentation for every RGB portrait")
                        yield Static(
                            "Measured white-mask profile. The paid portrait is preserved; "
                            "all segmentation, quality control and PDF/SVG export run locally.",
                            id="mask-seg-intro",
                        )
                        yield Static("", id="mask-geometry-preview")
                        yield Static(
                            f"{glyphs.DIAMOND} portrait  {glyphs.ARROW}  five landmarks  "
                            f"{glyphs.ARROW}  fail-closed geometry gate\n"
                            f"{glyphs.DIAMOND_HOLLOW} six surface panels  {glyphs.ARROW}  "
                            "2-page colour PDF + 3-page calibration PDF + SVG",
                            id="mask-pipeline",
                        )

                        yield Static("SHELL SURFACE", classes="mask-section-title")
                        with Horizontal(classes="mask-measure-row"):
                            with Vertical(classes="mask-field"):
                                yield Label("edge-to-edge width · mm")
                                yield Input(
                                    value=str(p.mask_width_mm), type="number",
                                    validators=[Number(minimum=100, maximum=300)],
                                    id="mask-width-mm",
                                )
                            with Vertical(classes="mask-field"):
                                yield Label("top-to-bottom height · mm")
                                yield Input(
                                    value=str(p.mask_height_mm), type="number",
                                    validators=[Number(minimum=150, maximum=400)],
                                    id="mask-height-mm",
                                )

                        yield Static("EYE APERTURES", classes="mask-section-title")
                        with Horizontal(classes="mask-measure-row"):
                            with Vertical(classes="mask-field"):
                                yield Label("inner-corner gap · mm")
                                yield Input(
                                    value=str(p.mask_eye_inner_gap_mm), type="number",
                                    validators=[Number(minimum=15, maximum=100)],
                                    id="mask-eye-gap-mm",
                                )
                            with Vertical(classes="mask-field"):
                                yield Label("opening width · mm")
                                yield Input(
                                    value=str(p.mask_eye_width_mm), type="number",
                                    validators=[Number(minimum=15, maximum=70)],
                                    id="mask-eye-width-mm",
                                )
                        with Horizontal(classes="mask-measure-row"):
                            with Vertical(classes="mask-field"):
                                yield Label("opening height · mm")
                                yield Input(
                                    value=str(p.mask_eye_height_mm), type="number",
                                    validators=[Number(minimum=8, maximum=40)],
                                    id="mask-eye-height-mm",
                                )
                            with Vertical(classes="mask-field"):
                                yield Label("centre from top · mm")
                                yield Input(
                                    value=str(p.mask_eye_center_top_mm), type="number",
                                    validators=[Number(minimum=40, maximum=180)],
                                    id="mask-eye-top-mm",
                                )

                        yield Static("NOSE PLANE + ASSEMBLY", classes="mask-section-title")
                        with Horizontal(classes="mask-measure-row"):
                            with Vertical(classes="mask-field"):
                                yield Label("nose base width · mm")
                                yield Input(
                                    value=str(p.mask_nose_width_mm), type="number",
                                    validators=[Number(minimum=20, maximum=80)],
                                    id="mask-nose-width-mm",
                                )
                            with Vertical(classes="mask-field"):
                                yield Label("nose plane length · mm")
                                yield Input(
                                    value=str(p.mask_nose_length_mm), type="number",
                                    validators=[Number(minimum=15, maximum=80)],
                                    id="mask-nose-length-mm",
                                )
                        with Horizontal(classes="mask-measure-row"):
                            with Vertical(classes="mask-field"):
                                yield Label("panel overlap · mm")
                                yield Input(
                                    value=str(p.mask_overlap_mm), type="number",
                                    validators=[Number(minimum=0, maximum=5)],
                                    id="mask-overlap-mm",
                                )
                            with Vertical(classes="mask-field"):
                                yield Label("A4 raster resolution · dpi")
                                yield Input(
                                    value=str(p.mask_dpi), type="integer",
                                    validators=[Integer(minimum=150, maximum=600)],
                                    id="mask-dpi",
                                )

            with VerticalScroll(id="col-demo"):
                yield BucketList("age", config.buckets.age, ages, id="age")
                yield BucketList("gender", config.buckets.gender, genders, id="gender")
                yield BucketList("ethnicity", config.buckets.ethnicity, eths, id="ethnicity")
                with Collapsible(title="weights (weighted mode)", id="weights-collapsible"):
                    yield Vertical(id="weights-box")
                with Collapsible(title="advanced prompt options · ctrl+o", id="advanced"):
                    with Vertical(id="advanced-box"):
                        with Horizontal(classes="switch-row"):
                            yield Switch(value=p.diversify, id="diversify-switch")
                            yield Label("diversify (unique appearance per image — prevents look-alikes)")
                        with Horizontal(classes="switch-row", id="face-crop-row"):
                            yield Switch(value=p.face_crop, id="face-crop-switch")
                            yield Label("A4 face portrait (prompt-only — head only, no clipping)")
                        with Vertical(id="portrait-style-fields"):
                            yield Label("background", classes="field-label")
                            yield Input(
                                value="plain light gray or off-white background", id="opt-background"
                            )
                            yield Label("expression", classes="field-label")
                            yield Input(value="neutral, natural facial expression", id="opt-expression")
                            yield Label("lighting", classes="field-label")
                            yield Input(value="natural studio lighting", id="opt-lighting")
                            yield Label("image style", classes="field-label")
                            yield Input(
                                value="photorealistic passport-style studio portrait", id="opt-style"
                            )
                        yield Label("extra positive constraints (one per line)", classes="field-label")
                        yield TextArea(id="opt-extra-pos")
                        yield Label("extra negative constraints (one per line)", classes="field-label")
                        yield TextArea(id="opt-extra-neg")
                        with Vertical(id="iris-realism-fields"):
                            yield Label(
                                "IR iris realism (only for IR · mixes non-ideal captures into the batch)",
                                classes="field-label",
                            )
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_occlusion, id="ir-occlusion-switch")
                                yield Label("eyelid / eyelash occlusion")
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_off_gaze, id="ir-off-gaze-switch")
                                yield Label("slight off-gaze")
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_lenses, id="ir-lenses-switch")
                                yield Label("contact lenses (soft/hard/cosmetic/painted)")
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_conditions, id="ir-conditions-switch")
                                yield Label("minor eye conditions")
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_glasses, id="ir-glasses-switch")
                                yield Label("glasses (heavy glare)")
                            with Horizontal(classes="switch-row"):
                                yield Switch(value=p.ir_makeup, id="ir-makeup-switch")
                                yield Label("eye makeup (moderate/strong)")
                with Vertical(classes="card", id="prompt-sample-card") as card:
                    card.border_title = "prompt sample"
                    card.border_subtitle = "ctrl+e full preview"
                    with VerticalScroll():
                        yield Static("", id="prompt-sample")

            with Vertical(id="col-readout"):
                with Vertical(classes="card", id="estimate-card") as card:
                    card.border_title = "estimate"
                    yield MoneyBlock(id="money")
                with Vertical(classes="card", id="plan-card") as card:
                    card.border_title = "plan preview"
                    yield Static("", id="plan-headline")
                    yield DistBars(id="plan-bars")
                    yield Static("", id="plan-items")
                with Horizontal(id="expose-row"):
                    yield Button("", id="expose-btn")
                yield Static("", id="why-disabled")

        yield Footer()

    # Controls that only make sense for the RGB face portrait; hidden for any
    # other modality (an IR iris capture has no head, framing, size choice,
    # background, expression or image-style to set).
    _FACE_ONLY_SELECTORS = (
        "#framing-label", "#framing",
        "#size-label", "#size",
        "#face-crop-row",
        "#mask-print-controls",
        "#portrait-style-fields",
    )
    # Controls that only make sense for the IR iris modality.
    _IRIS_ONLY_SELECTORS = ("#iris-realism-fields",)

    def on_mount(self) -> None:
        self.app.studio_screen = self
        self._apply_breakpoint(self.app.size.width)
        if self.query("#model-display"):
            self._sync_model_card()
            self._apply_modality_fields()
            self._schedule_refresh()

    def _apply_modality_fields(self) -> None:
        """Show only the controls relevant to the selected imaging modality."""
        if not self.query("#modality"):
            return
        modality = str(self.query_one("#modality", Select).value)
        is_face = modality == CaptureModality.RGB_FACE.value
        is_iris = modality == CaptureModality.IR_IRIS.value
        for selector in self._FACE_ONLY_SELECTORS:
            for widget in self.query(selector):
                widget.display = is_face
        for selector in self._IRIS_ONLY_SELECTORS:
            for widget in self.query(selector):
                widget.display = is_iris

    def on_resize(self, event: events.Resize) -> None:
        self._apply_breakpoint(event.size.width)

    def _apply_breakpoint(self, width: int) -> None:
        self.set_class(width < 110, "-compact")
        self.set_class(110 <= width < 150, "-standard")
        self.set_class(width >= 150, "-wide")

    def on_screen_resume(self) -> None:
        if self.query("#expose-btn"):
            self.set_run_active(getattr(self.app, "run_active", False))
            self._schedule_refresh()

    # ------------------------------------------------------------------ #
    # Change plumbing (debounced)
    # ------------------------------------------------------------------ #
    @on(Input.Changed)
    @on(Select.Changed)
    @on(BucketList.SelectedChanged)
    @on(Switch.Changed)
    @on(TextArea.Changed)
    def _settings_changed(self, event) -> None:
        if isinstance(event, Input.Changed) and isinstance(event.input, WeightInput):
            try:
                self._weights[event.input.bucket] = max(0.0, float(event.value))
            except ValueError:
                pass
        if isinstance(event, Select.Changed) and event.select.id == "size":
            pass  # size has no downstream rebuild beyond the estimate
        if isinstance(event, Select.Changed) and event.select.id == "modality":
            self._apply_modality_fields()  # show/hide face-only controls
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_timer(0.15, self._refresh_all)

    # ------------------------------------------------------------------ #
    # Draft request + problems
    # ------------------------------------------------------------------ #
    def _selected_buckets(self) -> tuple[list[str], list[str], list[str]]:
        return (
            list(self.query_one("#age", BucketList).selected),
            list(self.query_one("#gender", BucketList).selected),
            list(self.query_one("#ethnicity", BucketList).selected),
        )

    @property
    def _composed(self) -> bool:
        """False in the zero-models error layout (no compose widgets exist)."""
        return bool(self.query("#expose-btn"))

    def _field_int(
        self, selector: str, default: Optional[int] = None, *, strict: bool = True
    ) -> Optional[int]:
        raw = self.query_one(selector, Input).value.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            if strict:
                raise
            return default

    def _field_float(
        self, selector: str, default: Optional[float] = None, *, strict: bool = True
    ) -> Optional[float]:
        raw = self.query_one(selector, Input).value.strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            if strict:
                raise
            return default

    def _extra_lines(self, selector: str) -> list[str]:
        return [
            line.strip()
            for line in self.query_one(selector, TextArea).text.splitlines()
            if line.strip()
        ]

    def _draft_request(self) -> BatchGenerationRequest:
        """Build the full request from widget state. Raises on invalid input."""
        provider, model_id = _split_model_key(self._model_key)
        ages, genders, eths = self._selected_buckets()
        distribution = DistributionMode(str(self.query_one("#distribution", Select).value))
        weights = None
        if distribution == DistributionMode.WEIGHTED:
            selected = set(ages) | set(genders) | set(eths)
            weights = {b: self._weights.get(b, 1.0) for b in selected}
        size_value = self.query_one("#size", Select).value
        quality_value = self.query_one("#quality", Select).value
        quality = str(quality_value) if quality_value != Select.NULL else "medium"
        framing_value = self.query_one("#framing", Select).value
        head_height_pct = int(framing_value) if framing_value != Select.NULL else 60
        modality = CaptureModality(str(self.query_one("#modality", Select).value))
        size = str(size_value) if size_value != Select.NULL else "1024x1024"
        face_crop = self.query_one("#face-crop-switch", Switch).value
        if modality == CaptureModality.IR_IRIS:
            # An IR iris capture has no head/framing and uses a standardized 4:3
            # canvas (resolve_iris_capture_size); face-crop is meaningless here.
            face_crop = False
            try:
                model_info = self._config.pricing.get_model_info(provider, model_id)
                size = resolve_iris_capture_size(provider, model_id, model_info)
            except Exception:  # noqa: BLE001 - fall back to a valid 4:3 default
                size = "1536x1152"
        mask_print = None
        if (
            modality == CaptureModality.RGB_FACE
            and self.query_one("#mask-print-switch", Switch).value
        ):
            # Preserve explicit zero/negative input so Pydantic rejects it. Using
            # ``value or default`` here would silently turn a dangerous physical
            # measurement such as 0 mm back into the preset.
            def mask_float(selector: str, default: float) -> float:
                value = self._field_float(selector, default)
                return default if value is None else value

            def mask_int(selector: str, default: int) -> int:
                value = self._field_int(selector, default)
                return default if value is None else value

            mask_print = MaskPrintOptions(
                width_mm=mask_float("#mask-width-mm", 187.0),
                height_mm=mask_float("#mask-height-mm", 245.0),
                eye_inner_gap_mm=mask_float("#mask-eye-gap-mm", 40.0),
                eye_opening_width_mm=mask_float("#mask-eye-width-mm", 38.0),
                eye_opening_height_mm=mask_float("#mask-eye-height-mm", 18.0),
                eye_center_from_top_mm=mask_float("#mask-eye-top-mm", 103.0),
                nose_base_width_mm=mask_float("#mask-nose-width-mm", 40.0),
                nose_length_mm=mask_float("#mask-nose-length-mm", 30.0),
                overlap_mm=mask_float("#mask-overlap-mm", 1.5),
                dpi=mask_int("#mask-dpi", 300),
            )
            # The face-only prompt gives the landmark gate the standardized
            # single-head input it requires.
            face_crop = True
        iris_realism = IrisRealismOptions(
            eyelid_occlusion=self.query_one("#ir-occlusion-switch", Switch).value,
            off_gaze=self.query_one("#ir-off-gaze-switch", Switch).value,
            contact_lenses=self.query_one("#ir-lenses-switch", Switch).value,
            ocular_conditions=self.query_one("#ir-conditions-switch", Switch).value,
            glasses=self.query_one("#ir-glasses-switch", Switch).value,
            eye_makeup=self.query_one("#ir-makeup-switch", Switch).value,
        )
        return BatchGenerationRequest(
            provider=provider,
            model_id=model_id,
            modality=modality,
            iris_realism=iris_realism,
            age_buckets=ages,
            gender_buckets=genders,
            ethnicity_buckets=eths,
            distribution_mode=distribution,
            total_count=self._field_int("#batch-size", 0) or 0,
            weights=weights,
            variation_level=int(self.query_one("#variation", Select).value),
            size=size,
            quality=quality,
            head_height_pct=head_height_pct,
            seed=self._field_int("#seed"),
            output_dir=self.query_one("#output-dir", Input).value.strip() or None,
            filename_prefix=self.query_one("#prefix", Input).value or "portrait",
            retry_failed=self.query_one("#retry-switch", Switch).value,
            max_retries=self._field_int("#max-retries", 3) or 0,
            concurrency=max(1, min(32, self._field_int("#concurrency", 1) or 1)),
            background=self.query_one("#opt-background", Input).value or "plain light gray or off-white background",
            expression=self.query_one("#opt-expression", Input).value or "neutral, natural facial expression",
            lighting=self.query_one("#opt-lighting", Input).value or "natural studio lighting",
            image_style=self.query_one("#opt-style", Input).value or "photorealistic passport-style studio portrait",
            extra_positive_constraints=self._extra_lines("#opt-extra-pos"),
            extra_negative_constraints=self._extra_lines("#opt-extra-neg"),
            save_prompt=self.query_one("#save-prompt-switch", Switch).value,
            face_crop=face_crop,
            mask_print=mask_print,
            diversify=self.query_one("#diversify-switch", Switch).value,
        )

    def _compose_problems(self, draft_error: Optional[str]) -> list[Problem]:
        problems: list[Problem] = []
        config = self._config
        if not config.pricing.list_models():
            return [Problem("the model registry is empty")]
        if draft_error is not None:
            problems.append(Problem(draft_error))
        ages, genders, eths = self._selected_buckets()
        for dim, selection in (("age", ages), ("gender", genders), ("ethnicity", eths)):
            if not selection:
                problems.append(Problem(f"select at least one {dim} bucket"))
        distribution = str(self.query_one("#distribution", Select).value)
        if distribution == DistributionMode.WEIGHTED.value:
            # every triple's weight is a product across dimensions, so one fully
            # zeroed dimension makes the whole plan unbuildable
            for dim, selection in (("age", ages), ("gender", genders), ("ethnicity", eths)):
                if selection and all(self._weights.get(b, 1.0) <= 0 for b in selection):
                    problems.append(Problem(f"every selected {dim} bucket has weight 0"))
                    break
        provider, _ = _split_model_key(self._model_key)
        if not config.settings.has_key_for(provider):
            problems.append(
                Problem(f"no API key for '{provider}' — set it in .env or use the mock model")
            )
        return problems

    # ------------------------------------------------------------------ #
    # The big refresh
    # ------------------------------------------------------------------ #
    async def _refresh_all(self) -> None:
        if not self.query("#expose-btn"):
            return  # zero-models layout
        config = self._config
        await self._maybe_rebuild_weights()

        draft: Optional[BatchGenerationRequest] = None
        draft_error: Optional[str] = None
        try:
            draft = self._draft_request()
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            draft_error = str(first.get("msg", "invalid settings")).removeprefix("Value error, ")
        except (ValueError, TypeError) as exc:
            draft_error = str(exc) or "invalid settings"

        self._refresh_mask_geometry(draft, draft_error)
        self._problems = self._compose_problems(draft_error)
        self._refresh_estimate(draft)
        self._refresh_plan_preview(draft)
        self._refresh_expose_button(draft)

    def _refresh_mask_geometry(
        self,
        draft: Optional[BatchGenerationRequest],
        draft_error: Optional[str],
    ) -> None:
        """Render the dedicated 3D-mask geometry/QC readout."""
        if not self.query("#mask-geometry-preview"):
            return
        preview = self.query_one("#mask-geometry-preview", Static)
        enabled = self.query_one("#mask-print-switch", Switch).value
        if not enabled:
            preview.border_title = "segmentation status"
            preview.update(
                f"{labels.badge('OFF', '$text-muted')}  portraits are saved without mask assets\n"
                f"[$text-muted]Enable this category to add six physical panels, "
                "calibration pages, cut lines and per-frame QC.[/]"
            )
            return

        mask = draft.mask_print if draft is not None else None
        if mask is None:
            preview.border_title = "geometry needs attention"
            preview.update(
                f"{labels.badge('BLOCKED', '$error')}  "
                f"{esc(draft_error or 'complete the mask measurements')}\n"
                "[$text-muted]No generation can start with an invalid physical template.[/]"
            )
            return

        preview.border_title = "measured white-mask profile"
        preview.update(
            f"{labels.badge('ACTIVE', '$tele-ok')}  "
            f"{labels.badge('LOCAL $0', '$secondary')}  "
            f"{labels.badge('FAIL-CLOSED QC', '$warning')}\n"
            f"[b]surface[/]  {mask.width_mm:.1f} W × {mask.height_mm:.1f} H mm\n"
            f"[b]eyes[/]     {mask.eye_inner_gap_mm:.1f} inner gap · "
            f"{mask.eye_opening_width_mm:.1f}×{mask.eye_opening_height_mm:.1f} opening · "
            f"y={mask.eye_center_from_top_mm:.1f}\n"
            f"[b]nose[/]     {mask.nose_base_width_mm:.1f} base × "
            f"{mask.nose_length_mm:.1f} plane mm\n"
            f"[b]panels[/]   6 · {mask.overlap_mm:.1f} mm overlap · A4 @ {mask.dpi} dpi\n"
            f"[$text-muted]QC: exactly one YuNet face · confidence ≥ 0.88 · "
            "roll ≤ 8° · eye/nose/mouth geometry verified[/]"
        )

    async def _maybe_rebuild_weights(self) -> None:
        collapsible = self.query_one("#weights-collapsible", Collapsible)
        distribution = str(self.query_one("#distribution", Select).value)
        weighted = distribution == DistributionMode.WEIGHTED.value
        collapsible.display = weighted
        if not weighted:
            return
        collapsible.collapsed = False
        ages, genders, eths = self._selected_buckets()
        signature = tuple(ages) + ("|",) + tuple(genders) + ("|",) + tuple(eths)
        if signature == self._weights_built_for:
            return
        self._weights_built_for = signature
        box = self.query_one("#weights-box", Vertical)
        await box.remove_children()
        rows = []
        for dim, buckets in (("age", ages), ("gender", genders), ("ethnicity", eths)):
            for bucket in buckets:
                weight = self._weights.setdefault(bucket, 1.0)
                row = Horizontal(classes="weight-row")
                rows.append((row, dim, bucket, weight))
        for row, dim, bucket, weight in rows:
            await box.mount(row)
            await row.mount(
                Label(f"[{labels.DIM_VAR[dim]}]{labels.DIM_GLYPH[dim]}[/] {labels.short(bucket)}"),
                WeightInput(bucket, weight),
            )

    def _refresh_estimate(self, draft: Optional[BatchGenerationRequest]) -> None:
        money = self.query_one("#money", MoneyBlock)
        if draft is None or draft.total_count <= 0:
            money.set_amount(
                "0.00",
                "incomplete",
                meta="[$text-muted]complete the settings to see an estimate[/]",
                flash=False,
            )
            return
        config = self._config
        estimate = config.pricing.estimate(draft)
        has_model = config.pricing.has_model(draft.provider, draft.model_id)
        info = config.pricing.get_model_info(draft.provider, draft.model_id) if has_model else None
        model_line = (
            f"[$text-muted]{esc(draft.provider)} · {esc(info.display_name)}[/]" if info else ""
        )
        act_badge = (
            labels.badge("BILL: provider-reported " + glyphs.CHECK, "$tele-ok")
            if info and info.reports_actual_cost
            else labels.badge("BILL: estimated only (no provider $)", "$tele-pending")
        )
        source = (
            f"[$text-muted]src: {esc(estimate.pricing_source)}[/]"
            if estimate.pricing_source
            else ""
        )
        if estimate.pricing_available and estimate.estimated_total_usd is not None:
            free = estimate.estimated_total_usd == 0
            meta = (
                f"{estimate.total_count} images × ${estimate.price_per_image_usd:.4f}"
                f" · {draft.distribution_mode.value}"
            )
            if free:
                meta = f"{labels.badge('FREE', '$tele-ok')} {meta}"
            money.set_amount(
                f"{estimate.estimated_total_usd:.2f}",
                "free" if free else "ok",
                meta=meta,
                model=f"{model_line}  {act_badge}",
                source=source,
            )
        else:
            money.set_amount(
                "-.--",
                "warn",
                meta=f"{labels.badge('NO PRICE', '$warning')} "
                "[$warning]pricing unavailable — typed confirmation required[/]",
                model=model_line,
                source=source,
            )

    def _refresh_plan_preview(self, draft: Optional[BatchGenerationRequest]) -> None:
        headline = self.query_one("#plan-headline", Static)
        bars = self.query_one("#plan-bars", DistBars)
        items_line = self.query_one("#plan-items", Static)
        sample = self.query_one("#prompt-sample", Static)
        if draft is None or any(p.blocking for p in self._problems):
            headline.update(
                f"[$text-muted]{glyphs.DIAMOND_HOLLOW} the plan appears here once "
                "the settings are valid[/]"
            )
            bars.update_from_counts({})
            items_line.update("")
            sample.update("")
            self._preview_plan = []
            return

        sampled = (
            draft.distribution_mode in (DistributionMode.RANDOM, DistributionMode.WEIGHTED)
            and draft.seed is None
        )
        plan_request = (
            draft.model_copy(update={"seed": PREVIEW_SEED}) if sampled else draft
        )
        try:
            plan = plan_batch(plan_request)
        except Exception as exc:  # noqa: BLE001 - planning errors render as problems
            headline.update(f"[$error]{glyphs.CROSS} {esc(str(exc))}[/]")
            bars.update_from_counts({})
            items_line.update("")
            self._preview_plan = []
            return
        self._preview_plan = plan

        age_counts = Counter(i.prompt_options.age_bucket for i in plan)
        gender_counts = Counter(i.prompt_options.gender_bucket for i in plan)
        eth_counts = Counter(i.prompt_options.ethnicity_bucket for i in plan)

        def _axis_spread(counts: Counter, selected: list[str]) -> int:
            # Include selected-but-unused buckets as 0 so a fully-skewed axis shows.
            vals = [counts.get(b, 0) for b in selected]
            return (max(vals) - min(vals)) if vals else 0

        # Per-axis spread, NOT a single joint "triple spread": the latter reads 0
        # when every (age,gender,ethnicity) triple is unique even though one axis
        # is completely skewed (e.g. all 8 images East Asian).
        asp = _axis_spread(age_counts, draft.age_buckets)
        gsp = _axis_spread(gender_counts, draft.gender_buckets)
        esp = _axis_spread(eth_counts, draft.ethnicity_buckets)
        badge = (
            labels.badge("EXPECTED (sampled preview)", "$warning")
            if sampled
            else labels.badge("EXACT PLAN", "$success")
        )
        if draft.modality == CaptureModality.IR_IRIS:
            geom = "iris capture · 4:3 landscape"
        else:
            geom = f"{framing_label(draft.head_height_pct)} · head {draft.head_height_pct}%"
            if draft.mask_print:
                geom += (
                    f" · 3D mask {draft.mask_print.width_mm:.0f}×"
                    f"{draft.mask_print.height_mm:.0f} mm + landmark QC"
                )
        headline.update(
            f"{badge}  [b]{len(plan)}[/b] imgs · {draft.distribution_mode.value}\n"
            f"[$text-muted]spread — age {asp} · gender {gsp} · ethnicity {esp}[/]\n"
            f"[$text-muted]size {esc(draft.size)} · {esc(draft.quality)} · {esc(geom)}[/]"
        )

        bars.update_from_counts(
            {"age": age_counts, "gender": gender_counts, "ethnicity": eth_counts}
        )
        rows = []
        for item in plan[:3]:
            opts = item.prompt_options
            rows.append(
                f"[$text-muted]#{item.index + 1}[/] "
                + labels.triple_chips(opts.age_bucket, opts.gender_bucket, opts.ethnicity_bucket)
            )
        if len(plan) > 3:
            rows.append(f"[$text-muted]{glyphs.ELLIPSIS} {len(plan) - 3} more[/]")
        items_line.update("\n".join(rows))

        from app.core.prompt_builder import build_prompt

        first = plan[0].prompt_options
        note = (
            f"[$text-muted]prompt for planned item #1 of {len(plan)} "
            f"({esc(first.ethnicity_bucket)}) — other items use other buckets; "
            f"ctrl+e pages all[/]\n\n"
        )
        sample.update(note + build_prompt(first))

    def _refresh_expose_button(self, draft: Optional[BatchGenerationRequest]) -> None:
        button = self.query_one("#expose-btn", Button)
        why = self.query_one("#why-disabled", Static)
        blockers = [p for p in self._problems if p.blocking]
        if self._run_locked:
            button.label = f"{glyphs.DOT_HALF}  developing — see darkroom (ctrl+g)"
            button.disabled = False
            why.update("")
            return
        count = draft.total_count if draft else 0
        free = False
        if draft is not None:
            estimate = self._config.pricing.estimate(draft)
            free = bool(estimate.pricing_available and (estimate.estimated_total_usd or 0) == 0)
        verb = "DEVELOP" if free else "EXPOSE"
        suffix = " (FREE)" if free else ""
        button.label = f"{glyphs.DIAMOND}  {verb} {count} FRAMES{suffix}   ·   ctrl+g"
        button.disabled = bool(blockers)
        button.tooltip = "\n".join(f"{glyphs.CROSS} {p.text}" for p in blockers) or None
        if blockers:
            why.remove_class("-warn-only")
            why.update(f"{glyphs.CROSS} {blockers[0].text}")
        elif any(not p.blocking for p in self._problems):
            why.add_class("-warn-only")
            why.update(f"{glyphs.WARN} {self._problems[0].text}")
        else:
            why.update("")

    # ------------------------------------------------------------------ #
    # Model picking
    # ------------------------------------------------------------------ #
    def _sync_model_card(self) -> None:
        config = self._config
        provider, model_id = _split_model_key(self._model_key)
        card = self.query_one("#model-display", ModelCard)
        if not config.pricing.has_model(provider, model_id):
            card.update(f"[$error]{glyphs.CROSS} unknown model {self._model_key}[/]")
            return
        info = config.pricing.get_model_info(provider, model_id)
        price = (
            f"${info.price_per_image_usd:.4f}"
            if info.price_per_image_usd is not None
            else "[$warning]$ —[/]"
        )
        key_badge = (
            f"[$tele-ok]{glyphs.DOT} key ok[/]"
            if config.settings.has_key_for(provider)
            else f"[$warning]{glyphs.DOT_HOLLOW} no key[/]"
        )
        card.update(
            f"[b]{esc(info.display_name)}[/b]\n"
            f"[$text-muted]{esc(provider)} · {price}/img ·[/] {key_badge}"
            f"   [$text-muted]ctrl+n[/]"
        )
        size_select = self.query_one("#size", Select)
        sizes = info.supports_size or ["1024x1024"]
        size_select.set_options([(s, s) for s in sizes])
        size_select.value = info.default_size if info.default_size in sizes else sizes[0]

    def set_model(self, key: str) -> None:
        if not self._composed or self._run_locked:
            return
        self._model_key = key
        self.app.prefs.model_key = key
        self._sync_model_card()
        self._schedule_refresh()

    def action_model_picker(self) -> None:
        if not self._composed or self._run_locked:
            return

        def _picked(key: Optional[str]) -> None:
            if key:
                self.set_model(key)

        self.app.push_screen(ModelPickerModal(self._config, self._model_key), _picked)

    @on(Button.Pressed, "#expose-btn")
    async def _expose_pressed(self) -> None:
        await self.app.action_generate()

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    async def action_generate(self) -> None:
        await self.app.action_generate()

    def action_prompt_peek(self) -> None:
        if not self._preview_plan:
            self.notify("Nothing to preview — complete the settings first.", severity="warning")
            return
        self.app.push_screen(PromptPeekModal(self._preview_plan, 0))

    def action_randomize_seed(self) -> None:
        import random

        if not self._composed or self._run_locked:
            return
        self.query_one("#seed", Input).value = str(random.randint(0, 999_999))

    def clear_seed(self) -> None:
        if not self._composed or self._run_locked:
            return
        self.query_one("#seed", Input).value = ""

    def action_toggle_advanced(self) -> None:
        if not self._composed:
            return
        advanced = self.query_one("#advanced", Collapsible)
        advanced.collapsed = not advanced.collapsed

    def set_distribution(self, mode: str) -> None:
        if not self._composed or self._run_locked:
            return
        self.query_one("#distribution", Select).value = mode

    def select_all_demographics(self) -> None:
        if not self._composed or self._run_locked:
            return
        for selector in ("#age", "#gender", "#ethnicity"):
            self.query_one(selector, BucketList).select_all()

    def reset_demographics(self) -> None:
        if not self._composed or self._run_locked:
            return
        config = self._config
        self.query_one("#age", BucketList).select_all()
        self.query_one("#gender", BucketList).select_all()
        eth = self.query_one("#ethnicity", BucketList)
        eth.deselect_all()
        for bucket in config.buckets.ethnicity[:2]:
            eth.select(bucket)

    def focus_field(self, field_id: str) -> None:
        try:
            self.query_one(f"#{field_id}").focus()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Run-state lock + prefs
    # ------------------------------------------------------------------ #
    def set_run_active(self, active: bool) -> None:
        self._run_locked = active
        if not self.query("#expose-btn"):
            return
        for widget in self.query("Input, Select, Switch, TextArea, BucketList, ModelCard"):
            widget.disabled = active
        body = self.query_one("#studio-body")
        body.styles.opacity = 0.55 if active else 1.0
        self._refresh_expose_button(None if active else self._safe_draft())

    def _safe_draft(self) -> Optional[BatchGenerationRequest]:
        try:
            return self._draft_request()
        except Exception:  # noqa: BLE001
            return None

    def build_request_or_notify(self) -> Optional[BatchGenerationRequest]:
        """Final validation pass for the generate flow (pydantic is authoritative)."""
        if not self._composed:
            self.notify("No models in the registry — nothing to generate.", severity="error")
            return None
        try:
            return self._draft_request()
        except ValidationError as exc:
            reasons = "; ".join(
                str(err.get("msg", "invalid value")).removeprefix("Value error, ")
                for err in exc.errors()
            )
            self.notify(reasons or str(exc), severity="error", title="Invalid settings")
        except (ValueError, TypeError) as exc:
            self.notify(str(exc), severity="error", title="Invalid settings")
        return None

    def snapshot_prefs(self) -> None:
        """Fold current compose state into ``app.prefs`` (called before save)."""
        p: prefs.Prefs = self.app.prefs
        if not self.query("#expose-btn"):
            return
        ages, genders, eths = self._selected_buckets()
        p.model_key = self._model_key
        p.batch_size = (
            self._field_int("#batch-size", p.batch_size, strict=False) or p.batch_size
        )
        p.variation = int(self.query_one("#variation", Select).value)
        quality_value = self.query_one("#quality", Select).value
        if quality_value != Select.NULL:
            p.quality = str(quality_value)
        framing_value = self.query_one("#framing", Select).value
        if framing_value != Select.NULL:
            p.head_height_pct = int(framing_value)
        p.distribution = str(self.query_one("#distribution", Select).value)
        p.concurrency = (
            self._field_int("#concurrency", p.concurrency, strict=False) or p.concurrency
        )
        p.prefix = self.query_one("#prefix", Input).value or "portrait"
        p.ages, p.genders, p.ethnicities = ages, genders, eths
        p.retry_failed = self.query_one("#retry-switch", Switch).value
        p.max_retries = self._field_int("#max-retries", p.max_retries, strict=False) or 0
        p.save_prompt = self.query_one("#save-prompt-switch", Switch).value
        p.face_crop = self.query_one("#face-crop-switch", Switch).value
        p.mask_print = self.query_one("#mask-print-switch", Switch).value
        p.mask_width_mm = self._field_float(
            "#mask-width-mm", p.mask_width_mm, strict=False
        ) or p.mask_width_mm
        p.mask_height_mm = self._field_float(
            "#mask-height-mm", p.mask_height_mm, strict=False
        ) or p.mask_height_mm
        p.mask_eye_inner_gap_mm = self._field_float(
            "#mask-eye-gap-mm", p.mask_eye_inner_gap_mm, strict=False
        ) or p.mask_eye_inner_gap_mm
        p.mask_eye_width_mm = self._field_float(
            "#mask-eye-width-mm", p.mask_eye_width_mm, strict=False
        ) or p.mask_eye_width_mm
        p.mask_eye_height_mm = self._field_float(
            "#mask-eye-height-mm", p.mask_eye_height_mm, strict=False
        ) or p.mask_eye_height_mm
        p.mask_eye_center_top_mm = self._field_float(
            "#mask-eye-top-mm", p.mask_eye_center_top_mm, strict=False
        ) or p.mask_eye_center_top_mm
        p.mask_nose_width_mm = self._field_float(
            "#mask-nose-width-mm", p.mask_nose_width_mm, strict=False
        ) or p.mask_nose_width_mm
        p.mask_nose_length_mm = self._field_float(
            "#mask-nose-length-mm", p.mask_nose_length_mm, strict=False
        ) or p.mask_nose_length_mm
        overlap_value = self._field_float(
            "#mask-overlap-mm", p.mask_overlap_mm, strict=False
        )
        if overlap_value is not None:
            p.mask_overlap_mm = overlap_value
        p.mask_dpi = self._field_int("#mask-dpi", p.mask_dpi, strict=False) or p.mask_dpi
        p.diversify = self.query_one("#diversify-switch", Switch).value
        modality_value = self.query_one("#modality", Select).value
        if modality_value != Select.NULL:
            p.modality = str(modality_value)
        p.ir_occlusion = self.query_one("#ir-occlusion-switch", Switch).value
        p.ir_off_gaze = self.query_one("#ir-off-gaze-switch", Switch).value
        p.ir_lenses = self.query_one("#ir-lenses-switch", Switch).value
        p.ir_conditions = self.query_one("#ir-conditions-switch", Switch).value
        p.ir_glasses = self.query_one("#ir-glasses-switch", Switch).value
        p.ir_makeup = self.query_one("#ir-makeup-switch", Switch).value
