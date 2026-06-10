"""Command palette provider (ctrl+p) — every app action, searchable.

``discover()`` surfaces a curated six before the user types; ``search()``
fuzzy-matches the full set. Run-state-dependent commands (cancel) only appear
while relevant.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from textual.command import DiscoveryHit, Hit, Hits, Provider

from . import palette

if TYPE_CHECKING:
    from .app import PortraitApp


class StudioCommandsProvider(Provider):
    """All Synthetic Portrait Lab commands."""

    @property
    def _papp(self) -> "PortraitApp":
        return self.app  # type: ignore[return-value]

    def _commands(self) -> list[tuple[str, str, Callable]]:
        app = self._papp
        commands: list[tuple[str, str, Callable]] = [
            (
                "Generate batch…",
                "Validate, review the cost, and expose the batch",
                app.action_generate,
            ),
            ("Open last run", "Contact sheet for the most recent run", app.open_last_run),
            (
                "Open output folder",
                "Reveal the output base directory in your file manager",
                lambda: app.reveal_path(Path(app.config.settings.output_base_dir)),
            ),
            ("Go to Studio", "Compose screen", partial(app.switch_mode, "studio")),
            ("Go to Archive", "Browse past runs", partial(app.switch_mode, "archive")),
            ("Preview prompt", "See the exact prompt text for the planned items",
             partial(app.with_studio, lambda s: s.action_prompt_peek())),
            ("Choose model…", "Pick a provider/model",
             partial(app.with_studio, lambda s: s.action_model_picker())),
            ("Use mock provider (free, offline)", "No API key needed",
             partial(app.with_studio, lambda s: s.set_model("mock::mock-image"))),
            ("Randomize seed", "Set a random seed for a repeatable run",
             partial(app.with_studio, lambda s: s.action_randomize_seed())),
            ("Clear seed", "Let the provider choose randomness",
             partial(app.with_studio, lambda s: s.clear_seed())),
            ("Toggle advanced options", "Prompt knobs and extra constraints",
             partial(app.with_studio, lambda s: s.action_toggle_advanced())),
            ("Demographics: select all", "Every bucket in every dimension",
             partial(app.with_studio, lambda s: s.select_all_demographics())),
            ("Demographics: reset to defaults", "All ages/genders, first two ethnicities",
             partial(app.with_studio, lambda s: s.reset_demographics())),
            ("Focus: batch size", "Jump to the batch size field",
             partial(app.with_studio, lambda s: s.focus_field("batch-size"))),
            ("Focus: seed", "Jump to the seed field",
             partial(app.with_studio, lambda s: s.focus_field("seed"))),
            ("Focus: concurrency", "Jump to the concurrency field",
             partial(app.with_studio, lambda s: s.focus_field("concurrency"))),
            ("Focus: output directory", "Jump to the output directory field",
             partial(app.with_studio, lambda s: s.focus_field("output-dir"))),
            ("Focus: filename prefix", "Jump to the prefix field",
             partial(app.with_studio, lambda s: s.focus_field("prefix"))),
            ("Save current settings as defaults", "Persist compose settings to prefs",
             app.save_settings_as_defaults),
            ("Reset settings to factory", "Forget saved preferences",
             app.reset_settings_to_factory),
            ("Workflow guide", "How the studio → darkroom → archive flow works",
             app.show_guide),
        ]
        for mode in ("even", "random", "weighted"):
            commands.append(
                (
                    f"Distribution: {mode}",
                    "How the batch spreads across demographic triples",
                    partial(app.with_studio, lambda s, m=mode: s.set_distribution(m)),
                )
            )
        for theme in palette.THEME_CYCLE:
            commands.append(
                (
                    f"Theme: {theme}",
                    "Switch the colour theme",
                    partial(app.set_app_theme, theme),
                )
            )
        if app.run_active:
            commands.append(
                (
                    "Cancel current run",
                    "Request cancellation — in-flight items drain first",
                    app.request_cancel,
                )
            )
        return commands

    async def discover(self) -> Hits:
        discovery = (
            "Generate batch…",
            "Open last run",
            "Go to Archive",
            "Preview prompt",
            "Randomize seed",
            "Workflow guide",
        )
        by_title = {title: (help_text, callback) for title, help_text, callback in self._commands()}
        for title in discovery:
            if title in by_title:
                help_text, callback = by_title[title]
                yield DiscoveryHit(title, callback, text=title, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, callback in self._commands():
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title), callback, text=title, help=help_text)
