"""PortraitApp — modes, themes, global bindings, and the run singleton.

The app owns the run lifecycle so screens stay disposable views:

* the generation worker runs **here** (a popped Darkroom never kills a run);
* engine events arrive as :class:`EngineEvent` messages and are reduced into
  the app-owned :class:`RunTelemetry` on the main thread;
* cancellation is a plain bool handed to the engine's ``should_cancel``;
* ``generator.confirm(run)`` is called only after the EXPOSE modal returns True.
"""

from __future__ import annotations

import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from rich.markup import escape as esc
from textual import on, work
from textual.app import App, ScreenStackError
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown

from app.core.config import AppConfig
from app.core.generator import Generator, RunNotConfirmedError
from app.core.models import Run
from app.core.providers.base import ProviderAuthError, ProviderError

from . import glyphs, palette, prefs, runscan
from .commands import StudioCommandsProvider
from .messages import EngineEvent, RunFinished
from .screens.archive import ArchiveScreen
from .screens.contact_sheet import ContactSheetScreen
from .screens.darkroom import DarkroomScreen
from .screens.modals import ExposeModal, QuitGuardModal
from .screens.studio import StudioScreen
from .telemetry import RunTelemetry

GUIDE_MD = f"""\
# {glyphs.DIAMOND} Synthetic Portrait Lab — workflow guide

**Studio** (`f2`) — compose the batch: model, size, count, demographics,
distribution, seed. The readout column always shows the *exact* plan and cost.

**EXPOSE** (`ctrl+g`) — the one irreversible step. Review the spend, press
`enter` to arm, `enter` again to confirm. Free runs (mock) confirm in one
press; un-priced models require typing `spend`.

**Darkroom** — the live run. Matrix of every frame, lanes, throughput, cost
ledger, coverage, and the latest print developing in half-blocks.
`ctrl+x` twice cancels (in-flight frames finish; everything on disk stays).
`esc` returns to the studio — the run keeps going.

**Contact sheet** — every frame of a finished run, decoded into the terminal.
`enter` opens the lightbox, `v` shows the exact prompt.

**Archive** (`f3`) — every past run found in the output directory, sortable
and filterable, reconstructed purely from `manifest.json` + `metadata.jsonl`.

The mock provider is free, offline, and instant — perfect for a first run.

`ctrl+p` opens the command palette; `f1` lists every key.
"""


class GuideModal(ModalScreen[None]):
    DEFAULT_CSS = """
    GuideModal {
        align: center middle;
        #guide-card {
            width: 80;
            max-width: 96%;
            height: 80%;
            background: $surface;
            border: thick $primary;
            border-title-color: $primary;
            padding: 0 1;
        }
    }
    """
    BINDINGS = [Binding("escape,q", "close", "close")]

    def compose(self):
        with VerticalScroll(id="guide-card") as card:
            card.border_title = "workflow guide"
            yield Markdown(GUIDE_MD)

    def action_close(self) -> None:
        self.dismiss(None)


class PortraitApp(App[None]):
    """The Synthetic Portrait Lab terminal app."""

    TITLE = "Synthetic Portrait Lab"
    SUB_TITLE = "synthetic portrait dataset darkroom"

    MODES = {"studio": StudioScreen, "archive": ArchiveScreen}
    DEFAULT_MODE = "studio"
    COMMANDS = App.COMMANDS | {StudioCommandsProvider}

    BINDINGS = [
        Binding("f1", "show_help_panel", "keys", tooltip="Show every key binding"),
        Binding("f2", "goto_studio", "studio", tooltip="Compose screen"),
        Binding("f3", "goto_archive", "archive", tooltip="Past runs"),
        Binding("ctrl+t", "cycle_theme", "theme",
                tooltip="Cycle darkroom → gallery → synthwave → safelight"),
    ]

    CSS = """
    Footer { background: $surface; }
    Toast { background: $panel; }
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.generator = Generator(config)
        self.prefs = prefs.load()
        self.studio_screen: Optional[StudioScreen] = None

        self.current_run: Optional[Run] = None
        self.telemetry: Optional[RunTelemetry] = None
        self.run_state: str = "idle"  # idle | running | done | cancelled | failed
        self._cancel_requested = False
        self._quit_after_run = False

        for theme in palette.THEMES:
            self.register_theme(theme)
        self.theme = (
            self.prefs.theme if self.prefs.theme in palette.THEME_CYCLE else "darkroom"
        )

    def get_theme_variable_defaults(self) -> dict[str, str]:
        # Built-in themes (reachable via the palette's theme picker) must still
        # resolve our custom $variables.
        return dict(palette.VARIABLE_DEFAULTS)

    def clear_selection(self) -> None:
        # Workaround for a Textual 8.2.7 bug: Input(value=...)'s selection
        # watcher calls app.clear_selection() while the very first mode screen is
        # still composing — before it is appended to the screen stack. The base
        # method only guards against NoScreen, but `self.screen` raises
        # ScreenStackError on an empty stack, so the exception escapes and the
        # app dies on startup under a real TTY (run_test composes later and
        # dodges it). No selection can exist before the first screen mounts, so
        # skipping the call here is safe.
        if not self.screen_stack:
            return
        try:
            super().clear_selection()
        except ScreenStackError:
            pass

    def on_mount(self) -> None:
        self.theme_changed_signal.subscribe(self, self._theme_changed)
        if not self.prefs.seen_welcome:
            self.prefs.seen_welcome = True
            self.notify(
                "Welcome. The mock model is free and offline — press ctrl+g for a "
                "five-second test batch. f1 lists every key.",
                title=f"{glyphs.DIAMOND} Synthetic Portrait Lab",
                timeout=8,
            )

    def _theme_changed(self, _theme) -> None:
        self.prefs.theme = self.theme

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #
    @property
    def run_active(self) -> bool:
        return self.run_state == "running"

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested and self.run_active

    def hero_status(self) -> str:
        tele = self.telemetry
        if self.run_state == "running" and tele is not None:
            if self.cancel_requested:
                return f"[$warning]{glyphs.CROSS} cancelling — draining[/]"
            burn = tele.burn_estimate
            cost = f" · ~${burn:.2f}" if burn else ""
            return (
                f"[$tele-running]{glyphs.DOT_HALF} developing "
                f"{tele.done}/{tele.total}{cost}[/]"
            )
        if self.run_state == "done":
            return f"[$tele-ok]{glyphs.DOT} complete[/]"
        if self.run_state == "cancelled":
            return f"[$warning]{glyphs.DOT_HALF} cancelled[/]"
        if self.run_state == "failed":
            return f"[$tele-fail]{glyphs.CROSS} failed[/]"
        provider = self.config.settings.default_provider
        if self.studio_screen is not None and self.studio_screen._model_key:
            provider = self.studio_screen._model_key.partition("::")[0]
        return f"[$text-muted]{glyphs.DOT} {esc(provider)} · ready[/]"

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    async def _goto_mode_base(self, mode: str) -> None:
        """Switch to a mode; if already there, pop back to its base screen
        (switch_mode is a no-op for the current mode, so f2 from a pushed
        Darkroom would otherwise do nothing)."""
        if self.current_mode != mode:
            await self.switch_mode(mode)
            return
        while len(self.screen_stack) > 1:
            await self.pop_screen()

    async def action_goto_studio(self) -> None:
        await self._goto_mode_base("studio")

    async def action_goto_archive(self) -> None:
        await self._goto_mode_base("archive")

    def navigate_rail(self, target: str) -> None:
        if target == "studio":
            self.call_later(self._goto_mode_base, "studio")
        elif target == "archive":
            self.call_later(self._goto_mode_base, "archive")
        elif target == "darkroom":
            self.open_darkroom()
        elif target == "contact":
            self.open_last_run()

    def open_darkroom(self) -> None:
        if self.telemetry is None:
            self.notify(
                "No run yet — compose one in the studio (ctrl+g).", severity="warning"
            )
            return

        async def _go() -> None:
            # the darkroom lives on the studio stack only — this keeps a single
            # instance app-wide (a second one would double-fold the EWMA ticker)
            if self.current_mode != "studio":
                await self.switch_mode("studio")
            if not any(isinstance(s, DarkroomScreen) for s in self.screen_stack):
                self.push_screen(DarkroomScreen())

        self.call_later(_go)

    def open_contact_sheet(self, run_dir: Path | str) -> None:
        run_dir = Path(run_dir)

        async def _go() -> None:
            if self.current_mode != "archive":
                await self.switch_mode("archive")
            top = self.screen
            if isinstance(top, ContactSheetScreen) and top._run_dir == run_dir:
                return  # rapid double-activation guard
            self.push_screen(ContactSheetScreen(run_dir))

        self.call_later(_go)

    def open_last_run(self) -> None:
        records = [
            r for r in runscan.scan_runs(self.config.settings.output_base_dir)
            if not r.unreadable
        ]
        if not records:
            self.notify(
                "No runs developed yet — the archive is empty.", severity="warning"
            )
            return
        self.open_contact_sheet(records[0].run_dir)

    def with_studio(self, fn: Callable[[StudioScreen], None]) -> None:
        """Run ``fn`` against the studio screen, switching modes first if needed."""

        async def _go() -> None:
            if self.current_mode != "studio":
                await self.switch_mode("studio")
            if self.studio_screen is not None:
                fn(self.studio_screen)

        self.call_later(_go)

    def show_guide(self) -> None:
        self.push_screen(GuideModal())

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def action_cycle_theme(self) -> None:
        cycle = palette.THEME_CYCLE
        try:
            idx = cycle.index(self.theme)
        except ValueError:
            idx = -1
        self.set_app_theme(cycle[(idx + 1) % len(cycle)])

    def set_app_theme(self, name: str) -> None:
        self.theme = name
        self.notify(f"theme: {name}", timeout=1.5)

    # ------------------------------------------------------------------ #
    # Generate flow
    # ------------------------------------------------------------------ #
    async def action_generate(self) -> None:
        if self.run_active:
            self.open_darkroom()
            return
        if any(
            isinstance(s, ExposeModal) for s in self.screen_stack
        ):
            return  # an EXPOSE flow is already open — don't stack a second one
        if self.current_mode != "studio":
            await self.switch_mode("studio")
        studio = self.studio_screen
        if studio is None:
            return
        request = studio.build_request_or_notify()
        if request is None:
            return
        if not self.config.settings.has_key_for(request.provider):
            self.notify(
                f"No API key configured for '{request.provider}'. Set it in your "
                ".env, or use the mock provider (free, offline).",
                severity="error",
                title="Missing API key",
            )
            return
        try:
            run = self.generator.create_run(request)
        except Exception as exc:  # noqa: BLE001 - planning/validation surface as toasts
            self.notify(str(exc), severity="error", title="Cannot create run")
            return
        self.push_screen(ExposeModal(run), partial(self._expose_decided, run))

    def _expose_decided(self, run: Run, confirmed: Optional[bool]) -> None:
        if not confirmed:
            self.notify("Generation cancelled — nothing was spent.", timeout=4)
            return
        if self.run_active:
            # belt-and-braces: never let a second confirmation kill a live run
            # (the exclusive worker group would cancel it without a terminal event)
            self.notify("A run is already in progress.", severity="warning")
            return
        try:
            self.generator.confirm(run)
        except RunNotConfirmedError as exc:
            self.notify(esc(str(exc)), severity="error", title="Cannot confirm")
            return
        self._begin_run(run)

    def _begin_run(self, run: Run) -> None:
        self.current_run = run
        self.telemetry = RunTelemetry.from_run(run)
        self._cancel_requested = False
        self._quit_after_run = False
        self.run_state = "running"
        if self.studio_screen is not None:
            self.studio_screen.snapshot_prefs()
            self.studio_screen.set_run_active(True)
        prefs.save(self.prefs)
        self._generation_worker(run)
        self.push_screen(DarkroomScreen())

    @work(exclusive=True, group="generation")
    async def _generation_worker(self, run: Run) -> None:
        def on_event(event) -> None:
            # post_message is thread-safe; the callback never touches widgets.
            self.post_message(EngineEvent(event))

        try:
            finished = await self.generator.execute(
                run,
                on_event=on_event,
                should_cancel=lambda: self._cancel_requested,
            )
            self.post_message(RunFinished(finished, None))
        except ProviderAuthError as exc:
            self.post_message(RunFinished(None, f"Authentication failed: {exc}"))
        except ProviderError as exc:
            self.post_message(RunFinished(None, f"Provider error: {exc}"))
        except Exception as exc:  # noqa: BLE001 - the UI must never crash with the run
            self.post_message(RunFinished(None, f"{type(exc).__name__}: {exc}"))

    @on(EngineEvent)
    def _engine_event(self, message: EngineEvent) -> None:
        if self.telemetry is not None:
            self.telemetry.reduce(message.event)

    @on(RunFinished)
    def _run_finished(self, message: RunFinished) -> None:
        tele = self.telemetry
        if message.error:
            self.run_state = "failed"
            if tele is not None and not tele.is_finished:
                tele.mark_failed(message.error)
            hint = ""
            if "auth" in message.error.lower() or "key" in message.error.lower():
                hint = " Check the provider key in your .env."
            self.notify(
                esc(message.error) + hint, severity="error", title="Run failed", timeout=8
            )
        else:
            run = message.run
            if run is not None and run.status.value == "cancelled":
                self.run_state = "cancelled"
                self.notify(
                    f"{glyphs.RETRY} cancelled · {run.success_count} frames kept · "
                    f"{run.spend_summary()} · {esc(run.output_dir.name)}",
                    severity="warning",
                    timeout=6,
                )
            elif run is not None:
                self.run_state = "done"
                severity = "warning" if run.failure_count else "information"
                self.notify(
                    f"{glyphs.DIAMOND} {run.success_count} developed, "
                    f"{run.failure_count} failed · {run.spend_summary()} · "
                    f"{esc(run.output_dir.name)} — enter opens the contact sheet",
                    title="Run complete",
                    severity=severity,
                    timeout=6,
                )
        self._cancel_requested = False
        if self.studio_screen is not None:
            self.studio_screen.set_run_active(False)
        if self._quit_after_run:
            self._quit_now()

    def request_cancel(self) -> None:
        if not self.run_active:
            return
        self._cancel_requested = True
        self.notify(
            "Cancelling — in-flight frames will finish; queued frames are skipped.",
            severity="warning",
            timeout=4,
        )

    # ------------------------------------------------------------------ #
    # Quit guard + prefs
    # ------------------------------------------------------------------ #
    async def action_quit(self) -> None:
        if self.run_active:
            def _decided(quit_now: Optional[bool]) -> None:
                if not quit_now:
                    return
                if self.run_active:
                    self._quit_after_run = True
                    self._cancel_requested = True
                else:
                    # the run finished while the guard was open — quit now
                    self._quit_now()

            self.push_screen(QuitGuardModal(), _decided)
            return
        self._quit_now()

    def _quit_now(self) -> None:
        if self.studio_screen is not None:
            self.studio_screen.snapshot_prefs()
        prefs.save(self.prefs)
        self.exit()

    def save_settings_as_defaults(self) -> None:
        if self.studio_screen is not None:
            self.studio_screen.snapshot_prefs()
        if prefs.save(self.prefs):
            self.notify("Settings saved as defaults.", timeout=3)
        else:
            self.notify("Could not write the prefs file.", severity="warning")

    def reset_settings_to_factory(self) -> None:
        self.prefs = prefs.Prefs(seen_welcome=True)
        prefs.save(self.prefs)
        self.notify("Preferences reset — restart to apply factory defaults.", timeout=4)

    # ------------------------------------------------------------------ #
    # OS integration
    # ------------------------------------------------------------------ #
    def reveal_path(self, path: Path | str) -> None:
        path = Path(path)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            else:
                subprocess.Popen(["explorer", str(path)])
            self.notify(f"Opening {path}", timeout=2)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not open {path}: {exc}", severity="warning")
