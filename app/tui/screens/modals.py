"""All modal screens: EXPOSE (confirm spend), model picker, prompt peek,
lightbox, failure triage, and the quit guard.

EXPOSE is the one irreversible action in the app, so it gets choreography:
REVIEW → (enter) → ARMED with a 5-second disarm drain → (enter) → confirmed.
A free run skips arming; an un-priced run demands typing the word ``spend``.
``generator.confirm(run)`` is only ever called *after* this modal returns True.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from rich.markup import escape as esc
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    DataTable,
    Digits,
    Input,
    Label,
    OptionList,
    Rule,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from app.core.config import AppConfig
from app.core.models import PlannedItem, Run
from app.core.prompt_builder import build_prompt

from .. import glyphs, imaging, labels, palette
from ..messages import ThumbReady
from ..telemetry import RunTelemetry

DRAIN_SECONDS = 5


def _model_key(provider: str, model_id: str) -> str:
    return f"{provider}::{model_id}"


# --------------------------------------------------------------------------- #
# EXPOSE — confirm spend
# --------------------------------------------------------------------------- #
class ExposeModal(ModalScreen[bool]):
    """Dismisses ``True`` to spend, ``False`` to walk away unharmed."""

    DEFAULT_CSS = """
    ExposeModal {
        align: center middle;

        #expose-card {
            width: 76;
            max-width: 96%;
            height: auto;
            max-height: 90%;
            overflow-y: auto;
            padding: 1 3;
            background: $surface;
            border: thick $warning;
            border-title-color: $warning;
        }
        &.-armed #expose-card { border: thick $success; border-title-color: $success; }
        &.-free #expose-card { border: thick $primary; border-title-color: $primary; }

        #expose-title { content-align: center middle; height: 1; }
        #expose-rows { height: auto; margin: 1 0 0 0; }
        #expose-rows Static { height: auto; }
        .expose-eyebrow { color: $text-muted; text-style: bold; margin: 1 0 0 0; }
        #expose-money-row { height: auto; align-vertical: middle; }
        #expose-currency { width: 2; padding: 1 0 0 0; color: $success; text-style: bold; }
        #expose-digits { width: auto; color: $success; }
        &.-unpriced #expose-digits, &.-unpriced #expose-currency { color: $warning; }
        #expose-warning { color: $warning; height: auto; margin: 1 0 0 0; }
        #expose-spend-input { margin: 1 0 0 0; border: tall $warning; }
        #expose-actions { height: auto; margin: 1 0 0 0; align-horizontal: center; }
        #expose-hint { content-align: center middle; color: $text-muted; height: auto; }
        #expose-confirm { width: 1fr; }
        #expose-drain { content-align: center middle; color: $success; height: 1; }
    }
    """

    BINDINGS = [
        Binding("enter", "advance", "arm / confirm", priority=True),
        Binding("escape", "back", "step back"),
    ]

    def __init__(self, run: Run) -> None:
        super().__init__()
        self._run = run
        self._armed = False
        self._drain_left = DRAIN_SECONDS
        self._drain_timer: Optional[Timer] = None
        est = run.estimate
        self._free = bool(est.pricing_available and (est.estimated_total_usd or 0) == 0)
        self._priced = bool(est.pricing_available and est.estimated_total_usd is not None)

    # -- layout ----------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        run = self._run
        est, info, req = run.estimate, run.model_info, run.request

        with Vertical(id="expose-card") as card:
            card.border_title = "confirm spend" if not self._free else "free run"
            yield Static(id="expose-title")
            yield Rule(line_style="heavy")
            with Vertical(id="expose-rows"):
                per_frame = (
                    f"${est.price_per_image_usd:.4f}" if self._priced else "[$warning]unknown[/]"
                )
                seed = req.seed if req.seed is not None else "random"
                yield Static(
                    f"[$text-muted]model[/]    {esc(info.provider)} · {esc(info.display_name)}"
                    f"      [$text-muted]per frame[/] {per_frame}"
                )
                yield Static(
                    f"[$text-muted]frames[/]   [b]{run.total}[/b] · {req.distribution_mode.value}"
                    f" · variation {req.variation_level} · seed {seed} · {req.size}"
                )
                yield Static(self._buckets_line())
                yield Static(self._plan_line())
                yield Static(f"[$text-muted]output[/]   {esc(str(run.output_dir))}")
            yield Static(
                "THIS RUN IS FREE" if self._free else "THIS RUN WILL SPEND",
                classes="expose-eyebrow",
            )
            with Horizontal(id="expose-money-row"):
                yield Label("$", id="expose-currency")
                amount = f"{est.estimated_total_usd:.2f}" if self._priced else "-.--"
                yield Digits(amount, id="expose-digits")
            if not self._priced:
                self.add_class("-unpriced")
                yield Static(
                    f"{glyphs.WARN}  {esc(est.warning or 'No price is configured for this model.')}",
                    id="expose-warning",
                )
                yield Input(
                    placeholder='type "spend" to arm',
                    id="expose-spend-input",
                )
            with Vertical(id="expose-actions"):
                yield Static(self._hint_text(), id="expose-hint")
                yield Button("confirm", id="expose-confirm", variant="success")
                yield Static("", id="expose-drain")

    def on_mount(self) -> None:
        self.query_one("#expose-confirm", Button).display = False
        if self._free:
            self.add_class("-free")
        title = (
            f"{glyphs.DIAMOND}  DEVELOP {self._run.total} FRAMES (FREE)"
            if self._free
            else f"{glyphs.DIAMOND}  EXPOSE {self._run.total} FRAMES"
        )
        self.query_one("#expose-title", Static).update(palette.wordmark(self.app, title))
        card = self.query_one("#expose-card")
        card.styles.opacity = 0.0
        card.styles.animate("opacity", 1.0, duration=0.18, easing="out_cubic")

    def _buckets_line(self) -> str:
        req = self._run.request
        chips = []
        for dim, buckets in (
            ("age", req.age_buckets),
            ("gender", req.gender_buckets),
            ("ethnicity", req.ethnicity_buckets),
        ):
            chips.extend(labels.chip(dim, b, solid=False) for b in buckets[:4])
            if len(buckets) > 4:
                chips.append(f"[$text-muted]+{len(buckets) - 4}[/]")
        return "[$text-muted]buckets[/]  " + " ".join(chips)

    def _plan_line(self) -> str:
        counts = Counter(
            (
                item.prompt_options.age_bucket,
                item.prompt_options.gender_bucket,
                item.prompt_options.ethnicity_bucket,
            )
            for item in self._run.plan
        )
        spread = (max(counts.values()) - min(counts.values())) if counts else 0
        mode = self._run.request.distribution_mode.value
        seeded = self._run.request.seed is not None
        exact = mode == "even" or seeded
        badge = (
            labels.badge("EXACT PLAN", "$success")
            if exact
            else labels.badge("SAMPLED", "$warning")
        )
        return (
            f"[$text-muted]plan[/]     {badge} · {len(counts)} triples · "
            f"spread max−min: {spread}"
        )

    def _hint_text(self) -> str:
        if self._free:
            return "[b]ENTER[/b]  develop      [b]ESC[/b]  cancel"
        if self._armed:
            return ""
        if not self._priced:
            return '[b]type "spend"[/b] then [b]ENTER[/b] to arm      [b]ESC[/b]  cancel'
        return "[b]ENTER[/b]  arm shutter      [b]ESC[/b]  step back"

    # -- choreography ------------------------------------------------------ #
    def _spend_typed(self) -> bool:
        if self._priced:
            return True
        try:
            value = self.query_one("#expose-spend-input", Input).value
        except Exception:  # noqa: BLE001
            return False
        return value.strip().lower() == "spend"

    def action_advance(self) -> None:
        if self._free:
            self.dismiss(True)
            return
        if not self._armed:
            if not self._spend_typed():
                self.app.bell()
                return
            self._arm()
        else:
            self.dismiss(True)

    def _arm(self) -> None:
        self._armed = True
        self._drain_left = DRAIN_SECONDS
        self.add_class("-armed")
        amount = (
            f"${self._run.estimate.estimated_total_usd:.2f}"
            if self._priced
            else "an un-estimated amount"
        )
        button = self.query_one("#expose-confirm", Button)
        button.display = True
        button.label = f"{glyphs.ARROW} CONFIRM — SPEND {amount} — ENTER"
        button.focus()
        self.query_one("#expose-hint", Static).update("")
        self._render_drain()
        self._drain_timer = self.set_interval(1.0, self._drain_tick)

    def _disarm(self) -> None:
        self._armed = False
        self.remove_class("-armed")
        if self._drain_timer is not None:
            self._drain_timer.stop()
            self._drain_timer = None
        self.query_one("#expose-confirm", Button).display = False
        self.query_one("#expose-drain", Static).update("")
        self.query_one("#expose-hint", Static).update(self._hint_text())

    def _drain_tick(self) -> None:
        self._drain_left -= 1
        if self._drain_left <= 0:
            self._disarm()
        else:
            self._render_drain()

    def _render_drain(self) -> None:
        drain = glyphs.DRAIN_ON * self._drain_left + glyphs.DRAIN_OFF * (
            DRAIN_SECONDS - self._drain_left
        )
        self.query_one("#expose-drain", Static).update(
            f"{drain}  [$text-muted]auto-disarm[/]"
        )

    def action_back(self) -> None:
        if self._armed:
            self._disarm()
        else:
            self.dismiss(False)

    @on(Button.Pressed, "#expose-confirm")
    def _confirm_pressed(self) -> None:
        self.dismiss(True)

    @on(Input.Submitted, "#expose-spend-input")
    def _spend_submitted(self) -> None:
        self.action_advance()


# --------------------------------------------------------------------------- #
# Model picker
# --------------------------------------------------------------------------- #
class ModelPickerModal(ModalScreen[Optional[str]]):
    """Pick a provider/model; dismisses with ``provider::model_id`` or None."""

    DEFAULT_CSS = """
    ModelPickerModal {
        align: center middle;

        #picker-card {
            width: 72;
            max-width: 96%;
            height: auto;
            max-height: 80%;
            padding: 1 2;
            background: $surface;
            border: thick $primary;
            border-title-color: $primary;
        }
        #picker-filter { margin: 0 0 1 0; }
        #picker-list { height: auto; max-height: 16; }
    }
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, config: AppConfig, current_key: str = "") -> None:
        super().__init__()
        self._config = config
        self._current_key = current_key

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-card") as card:
            card.border_title = "choose model"
            yield Input(placeholder="filter models…", id="picker-filter")
            yield OptionList(id="picker-list")

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#picker-filter", Input).focus()

    def _rows(self) -> list[tuple[str, str]]:
        rows = []
        for m in self._config.pricing.list_models():
            key = _model_key(m.provider, m.model_id)
            has_key = self._config.settings.has_key_for(m.provider)
            key_badge = (
                f"[$tele-ok]{glyphs.DOT} key ok [/]"
                if has_key
                else f"[$warning]{glyphs.DOT_HOLLOW} no key [/]"
            )
            price = (
                f"${m.price_per_image_usd:.4f}"
                if m.price_per_image_usd is not None
                else "[$warning]$  —[/]"
            )
            marker = glyphs.ARROW if key == self._current_key else " "
            display = (
                f"{marker} {key_badge} [$text-muted]{m.provider:<10}[/] "
                f"{esc(m.display_name):<36} {price:>9}"
            )
            rows.append((display, key))
        return rows

    def _populate(self, needle: str) -> None:
        option_list = self.query_one("#picker-list", OptionList)
        option_list.clear_options()
        needle = needle.strip().lower()
        options = [
            Option(display, id=key)
            for display, key in self._rows()
            if not needle or needle in display.lower() or needle in key.lower()
        ]
        if options:
            option_list.add_options(options)
        else:
            option_list.add_options([Option("[$text-muted]no models match[/]", disabled=True)])

    @on(Input.Changed, "#picker-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._populate(event.value)

    @on(Input.Submitted, "#picker-filter")
    def _filter_submitted(self) -> None:
        self.query_one("#picker-list", OptionList).focus()

    @on(OptionList.OptionSelected, "#picker-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- #
# Prompt peek
# --------------------------------------------------------------------------- #
class PromptPeekModal(ModalScreen[None]):
    """Browse the exact prompt for each planned item. Preview only."""

    DEFAULT_CSS = """
    PromptPeekModal {
        align: center middle;

        #peek-card {
            width: 90;
            max-width: 96%;
            height: 80%;
            padding: 1 2;
            background: $surface;
            border: thick $secondary;
            border-title-color: $secondary;
        }
        #peek-header { height: auto; margin: 0 0 1 0; }
        #peek-body { height: 1fr; }
        #peek-footer { height: 1; color: $text-muted; margin: 1 0 0 0; }
    }
    """

    BINDINGS = [
        Binding("escape", "close", "close"),
        # priority: the focused read-only TextArea would otherwise consume arrows
        Binding("left", "page(-1)", "prev item", priority=True),
        Binding("right", "page(1)", "next item", priority=True),
        Binding("c", "copy", "copy prompt"),
    ]

    def __init__(self, plan: list[PlannedItem], index: int = 0, header_note: str = "") -> None:
        super().__init__()
        self._plan = plan
        self._index = max(0, min(index, len(plan) - 1)) if plan else 0
        self._header_note = header_note

    def compose(self) -> ComposeResult:
        with Vertical(id="peek-card") as card:
            card.border_title = "prompt preview"
            yield Static(id="peek-header")
            yield TextArea("", id="peek-body", read_only=True, soft_wrap=True)
            yield Static(
                f"{glyphs.ARROW} preview only — nothing is generated · ←/→ items · c copy · esc close",
                id="peek-footer",
            )

    def on_mount(self) -> None:
        self._show_current()
        self.query_one("#peek-body", TextArea).focus()

    def _show_current(self) -> None:
        if not self._plan:
            self.query_one("#peek-header", Static).update("[$warning]empty plan[/]")
            return
        item = self._plan[self._index]
        opts = item.prompt_options
        seed_note = (
            f"seed {opts.seed}" if opts.seed is not None else "seed: random at provider"
        )
        header = (
            f"item [b]{self._index + 1}/{len(self._plan)}[/b] · {item.id} · {seed_note}\n"
            + labels.triple_chips(opts.age_bucket, opts.gender_bucket, opts.ethnicity_bucket)
        )
        if self._header_note:
            header = f"{self._header_note}\n{header}"
        self.query_one("#peek-header", Static).update(header)
        body = self.query_one("#peek-body", TextArea)
        body.text = build_prompt(opts)

    def action_page(self, delta: int) -> None:
        if not self._plan:
            return
        self._index = (self._index + delta) % len(self._plan)
        self._show_current()

    def action_copy(self) -> None:
        if self._plan:
            self.app.copy_to_clipboard(build_prompt(self._plan[self._index].prompt_options))
            self.notify("Prompt copied to clipboard.", timeout=2)

    def action_close(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- #
# Lightbox
# --------------------------------------------------------------------------- #
class LightboxModal(ModalScreen[None]):
    """Near-fullscreen view of one frame; ←/→ pages through the run."""

    DEFAULT_CSS = """
    LightboxModal {
        align: center middle;

        #lightbox-card {
            width: auto;
            max-width: 98%;
            height: auto;
            max-height: 98%;
            padding: 1 2;
            background: $surface;
            border: heavy $primary;
            border-title-color: $primary;
        }
        #lightbox-image { width: auto; height: auto; }
        #lightbox-prompt { width: 76; height: 24; display: none; }
        &.-prompt #lightbox-image { display: none; }
        &.-prompt #lightbox-prompt { display: block; }
        #lightbox-caption { height: auto; color: $text-muted; margin: 1 0 0 0; }
    }
    """

    BINDINGS = [
        Binding("escape", "close", "close"),
        Binding("left", "page(-1)", "prev"),
        Binding("right", "page(1)", "next"),
        Binding("v", "toggle_prompt", "prompt"),
        Binding("o", "reveal", "reveal file"),
    ]

    def __init__(
        self, items: list[dict], index: int, run_dir: Path, *, show_prompt: bool = False
    ) -> None:
        super().__init__()
        self._items = items
        self._index = max(0, min(index, len(items) - 1)) if items else 0
        self._run_dir = run_dir
        self._show_prompt = show_prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="lightbox-card"):
            yield Static(id="lightbox-image")
            with VerticalScroll(id="lightbox-prompt"):
                yield Static(id="lightbox-prompt-text")
            yield Static(id="lightbox-caption")

    def on_mount(self) -> None:
        if self._show_prompt:
            self.add_class("-prompt")
        self._show_current()

    def _image_size(self) -> tuple[int, int]:
        width, height = self.app.size
        rows = max(8, min(height - 10, (width - 12) // 2))
        return rows * 2, rows

    def _current(self) -> Optional[dict]:
        return self._items[self._index] if self._items else None

    def _show_current(self) -> None:
        item = self._current()
        if item is None:
            return
        card = self.query_one("#lightbox-card")
        card.border_title = str(item.get("id") or "frame")
        cols, rows = self._image_size()
        image = self.query_one("#lightbox-image", Static)
        caption = self.query_one("#lightbox-caption", Static)
        chips = labels.triple_chips(
            str(item.get("age_bucket", "")),
            str(item.get("gender_bucket", "")),
            str(item.get("ethnicity_bucket", "")),
        )
        seed = item.get("seed")
        cost = item.get("actual_cost_usd")
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "—"
        position = f"[$text-muted]{self._index + 1}/{len(self._items)}[/]"
        caption.update(
            f"{chips}   seed {seed if seed is not None else 'random'} · {cost_str}   {position}"
        )
        # prompt view must always reflect the CURRENT item, failed frames included
        prompt_text = str(item.get("prompt") or "(prompt not saved for this run)")
        self.query_one("#lightbox-prompt-text", Static).update(esc(prompt_text))
        filename = item.get("filename")
        if item.get("status") != "success" or not filename:
            image.update(
                imaging.skeleton(
                    min(cols, 48), min(rows, 12),
                    label=f"{glyphs.CROSS} {item.get('error') or 'no image'}",
                )
            )
            return
        image.update(imaging.skeleton(cols, rows, label="developing…"))
        self._decode(self._run_dir / "images" / str(filename), cols, rows, self._index)

    @work(thread=True, exclusive=True, group="lightbox-decode")
    def _decode(self, path: Path, cols: int, rows: int, index: int) -> None:
        try:
            text = imaging.render_halfblock(path, cols, rows)
            self.post_message(ThumbReady(f"idx:{index}", path, text))
        except imaging.ImageUnreadable as exc:
            self.post_message(ThumbReady(f"idx:{index}", path, None, error=str(exc)))

    @on(ThumbReady)
    def _apply(self, message: ThumbReady) -> None:
        message.stop()
        if message.key != f"idx:{self._index}":
            return
        image = self.query_one("#lightbox-image", Static)
        if message.text is None:
            cols, rows = self._image_size()
            image.update(
                imaging.skeleton(
                    min(cols, 48), min(rows, 12), label=f"{glyphs.CROSS} unreadable"
                )
            )
        else:
            image.update(message.text)

    def action_page(self, delta: int) -> None:
        if not self._items:
            return
        self.remove_class("-prompt")
        self._index = (self._index + delta) % len(self._items)
        self._show_current()

    def action_toggle_prompt(self) -> None:
        self.toggle_class("-prompt")

    def action_reveal(self) -> None:
        item = self._current()
        reveal = getattr(self.app, "reveal_path", None)
        if item and reveal:
            filename = item.get("filename")
            target = self._run_dir / "images" / filename if filename else self._run_dir
            reveal(target)

    def action_close(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- #
# Failure triage
# --------------------------------------------------------------------------- #
class FailureTriageModal(ModalScreen[Optional[int]]):
    """All failures in one table; selecting a row jumps the matrix cursor."""

    DEFAULT_CSS = """
    FailureTriageModal {
        align: center middle;

        #triage-card {
            width: 100;
            max-width: 98%;
            height: auto;
            max-height: 80%;
            padding: 1 2;
            background: $surface;
            border: thick $error;
            border-title-color: $error;
        }
        #triage-summary { height: auto; margin: 0 0 1 0; }
        #triage-table { height: auto; max-height: 18; }
    }
    """

    BINDINGS = [Binding("escape", "close", "close")]

    def __init__(self, telemetry: RunTelemetry) -> None:
        super().__init__()
        self._telemetry = telemetry
        self._row_indices: list[int] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="triage-card") as card:
            card.border_title = "failure triage"
            yield Static(id="triage-summary")
            yield DataTable(id="triage-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        tele = self._telemetry
        signatures = " · ".join(
            f"{esc(sig)} ×{count}" for sig, count in tele.error_signatures.most_common(4)
        )
        bias = tele.failure_bias()
        bias_line = (
            f"\n[$warning]{glyphs.WARN} failures cluster in: {labels.short(bias)}[/]"
            if bias
            else ""
        )
        self.query_one("#triage-summary", Static).update(
            f"[$tele-fail b]{glyphs.CROSS} {tele.failed} failed[/] · {signatures or 'no failures'}"
            + bias_line
        )
        table = self.query_one("#triage-table", DataTable)
        table.add_columns("ITEM", "AGE", "GENDER", "ETHNICITY", "RETRIES", "ERROR")
        for index, error in sorted(tele.errors.items()):
            meta = tele.items[index] if index < len(tele.items) else None
            if meta is None:
                continue
            self._row_indices.append(index)
            table.add_row(
                meta.item_id,
                labels.short(meta.age),
                labels.short(meta.gender),
                labels.short(meta.ethnicity),
                str(tele.retries[index]),
                error[:60],
            )
        table.focus()

    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._row_indices):
            self.dismiss(self._row_indices[event.cursor_row])

    def action_close(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- #
# Quit guard
# --------------------------------------------------------------------------- #
class QuitGuardModal(ModalScreen[bool]):
    """Quit requested while a run is live. True = cancel the run and quit."""

    DEFAULT_CSS = """
    QuitGuardModal {
        align: center middle;

        #quit-card {
            width: 64;
            height: auto;
            padding: 1 3;
            background: $surface;
            border: thick $warning;
            border-title-color: $warning;
        }
        #quit-card Static { height: auto; }
        #quit-hint { color: $text-muted; margin: 1 0 0 0; }
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "cancel run + quit", priority=True),
        Binding("escape", "stay", "stay"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-card") as card:
            card.border_title = "run in progress"
            yield Static(
                f"{glyphs.WARN}  A run is still developing. Quitting now will cancel it;\n"
                "frames already developed stay on disk (metadata is crash-safe)."
            )
            yield Static(
                "[b]ENTER[/b]  cancel the run and quit      [b]ESC[/b]  keep developing",
                id="quit-hint",
            )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_stay(self) -> None:
        self.dismiss(False)
