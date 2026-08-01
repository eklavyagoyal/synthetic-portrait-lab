"""End-to-end Pilot tests for the Textual TUI.

Everything runs headless against the free mock provider in a tmp output dir;
prefs are redirected to tmp so user state is never touched.
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import sys
import time
from pathlib import Path

import pytest

from app.core.config import AppConfig
from app.core.providers import registry
from app.core.providers.mock_provider import MockProvider
from app.tui import prefs as tui_prefs
from app.tui.app import PortraitApp
from app.tui.screens.archive import ArchiveScreen
from app.tui.screens.contact_sheet import ContactSheetScreen
from app.tui.screens.darkroom import DarkroomScreen
from app.tui.screens.modals import (
    ExposeModal,
    ModelPickerModal,
    PromptPeekModal,
    QuitGuardModal,
)
from app.tui.screens.studio import StudioScreen
from app.tui.telemetry import CellState
from app.tui.widgets import BatchMatrix, BucketList, MoneyBlock


@pytest.fixture()
def app(tmp_path, monkeypatch) -> PortraitApp:
    monkeypatch.setenv("PORTRAIT_OUTPUT_BASE_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(tui_prefs, "PREFS_PATH", tmp_path / "prefs.json")
    return PortraitApp(AppConfig.load())


class SlowMock(MockProvider):
    """Mock with latency so mid-run states are observable (pilot presses and
    pauses have real overhead, so the run must comfortably outlast them)."""

    delay = 0.6

    async def generate(self, **kwargs):
        await asyncio.sleep(self.delay)
        return await super().generate(**kwargs)


@pytest.fixture()
def slow_provider(monkeypatch):
    original = registry._load_class

    def patched(name: str):
        if name == "mock":
            return SlowMock
        return original(name)

    monkeypatch.setattr(registry, "_load_class", patched)


def _studio(app: PortraitApp) -> StudioScreen:
    assert app.studio_screen is not None
    return app.studio_screen


async def _wait_run_end(pilot, app: PortraitApp, ticks: int = 200) -> None:
    for _ in range(ticks):
        await pilot.pause(0.05)
        if app.run_state != "running":
            return
    raise AssertionError("run did not finish in time")


# --------------------------------------------------------------------------- #
# Studio
# --------------------------------------------------------------------------- #
async def test_studio_mounts_with_live_estimate(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        assert isinstance(app.screen, StudioScreen)
        money = app.screen.query_one("#money", MoneyBlock)
        assert money.has_class("-free")  # mock model is $0.00
        headline = app.screen.query_one("#plan-headline").render()
        assert "EXACT PLAN" in str(headline)
        assert not app.screen.query_one("#expose-btn").disabled


async def test_empty_dimension_blocks_expose(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        app.screen.query_one("#ethnicity", BucketList).deselect_all()
        await pilot.pause(0.4)
        button = app.screen.query_one("#expose-btn")
        assert button.disabled
        why = str(app.screen.query_one("#why-disabled").render())
        assert "ethnicity" in why


async def test_modality_ir_swaps_face_controls_for_iris_realism(app):
    from textual.widgets import Select, Switch

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)

        # RGB (default): face-portrait controls visible; iris realism hidden.
        for sel in (
            "#framing", "#face-crop-row", "#portrait-style-fields", "#size",
            "#mask-print-controls",
        ):
            assert studio.query_one(sel).display, sel
        assert not studio.query_one("#iris-realism-fields").display

        # Switch to IR — face-only controls disappear (a 75% head makes no sense),
        # and the iris realism knobs appear.
        studio.query_one("#modality", Select).value = "ir"
        await pilot.pause(0.3)
        for sel in ("#framing", "#framing-label", "#face-crop-row",
                    "#portrait-style-fields", "#size", "#size-label",
                    "#mask-print-controls"):
            assert not studio.query_one(sel).display, sel
        assert studio.query_one("#iris-realism-fields").display

        # Plan preview stops claiming a head height; the draft drops face-crop.
        headline = str(studio.query_one("#plan-headline").render())
        assert "iris capture" in headline
        draft = studio._draft_request()
        assert draft.modality.value == "ir" and draft.face_crop is False

        # A realism knob flows into the request (opt-in per run).
        studio.query_one("#ir-lenses-switch", Switch).value = True
        await pilot.pause(0.2)
        assert studio._draft_request().iris_realism.contact_lenses is True

        # Back to RGB: face controls return, iris realism hides again.
        studio.query_one("#modality", Select).value = "rgb"
        await pilot.pause(0.3)
        assert studio.query_one("#framing").display
        assert not studio.query_one("#iris-realism-fields").display


async def test_mask_print_controls_flow_into_request(app):
    from textual.widgets import Input, Switch

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)
        studio.query_one("#mask-print-switch", Switch).value = True
        studio.query_one("#mask-width-mm", Input).value = "187.5"
        studio.query_one("#mask-dpi", Input).value = "300"
        await pilot.pause(0.3)

        draft = studio._draft_request()
        assert draft.mask_print is not None
        assert draft.mask_print.width_mm == 187.5
        assert draft.mask_print.dpi == 300
        assert draft.face_crop is True
        assert "landmark QC" in str(studio.query_one("#plan-headline").render())

        # An explicit unsafe measurement must block the run, never fall back to
        # the 187 mm preset behind the user's back.
        studio.query_one("#mask-width-mm", Input).value = "0"
        await pilot.pause(0.3)
        assert studio.query_one("#expose-btn").disabled
        blocked = str(studio.query_one("#mask-geometry-preview").render())
        assert "BLOCKED" in blocked
        assert "greater than or equal to 100" in blocked


async def test_3d_mask_segmentation_runs_end_to_end_from_tui(app, monkeypatch):
    """The Studio category must drive the real local export, not only a draft."""
    from textual.widgets import Input, Switch

    from app.core.mask_print import FaceLandmarks

    def fixture_landmarks(source):
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

    monkeypatch.setattr("app.core.mask_print._detect_face_landmarks", fixture_landmarks)
    revealed: list[Path] = []
    monkeypatch.setattr(app, "reveal_path", revealed.append)

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)
        studio.query_one("#batch-size", Input).value = "1"
        studio.query_one("#mask-print-switch", Switch).value = True
        studio.query_one("#mask-dpi", Input).value = "150"
        await pilot.pause(0.4)

        geometry = str(studio.query_one("#mask-geometry-preview").render())
        assert "187.0 W × 245.0 H mm" in geometry
        assert "6 · 1.5 mm overlap · A4 @ 150 dpi" in geometry
        assert "FAIL-CLOSED QC" in geometry

        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        expose_rows = "\n".join(
            str(widget.render()) for widget in app.screen.query("#expose-rows Static")
        )
        assert "3D mask" in expose_rows
        await pilot.press("enter")
        await _wait_run_end(pilot, app)
        await pilot.pause(0.5)

        run = app.current_run
        assert run is not None and len(run.results) == 1
        result = run.results[0]
        assert result.mask_print_error is None
        assert result.mask_print_pdf and result.mask_calibration_pdf
        assert result.mask_preview_filename and result.mask_cutlines_svg
        assert result.mask_print_pages and len(result.mask_print_pages) == 2

        expected = (
            result.mask_preview_filename,
            result.mask_print_pdf,
            result.mask_calibration_pdf,
            result.mask_cutlines_svg,
            *result.mask_print_pages,
        )
        assert all((run.output_dir / relative).is_file() for relative in expected)

        pack_dir = (run.output_dir / result.mask_print_pdf).parent
        metadata = json.loads((pack_dir / f"{result.id}_mask.json").read_text())
        assert metadata["template"]["template_version"] == "landmarks-v2"
        assert metadata["template"]["width_mm"] == 187.0
        assert metadata["template"]["height_mm"] == 245.0
        assert metadata["normalization"]["quality_control"]["status"] == "passed"
        assert metadata["normalization"]["alignment_transform"][
            "max_eye_registration_error_px"
        ] <= 1.25

        await pilot.press("enter")
        await pilot.pause(0.8)
        assert isinstance(app.screen, ContactSheetScreen)
        meta = str(app.screen.query_one("#meta-body").render())
        assert "3D mask segmentation verified" in meta

        await pilot.press("p")
        await pilot.press("k")
        assert revealed == [
            run.output_dir / result.mask_print_pdf,
            run.output_dir / result.mask_calibration_pdf,
        ]


async def test_prompt_peek_and_model_picker_modals(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+e")
        await pilot.pause(0.2)
        assert isinstance(app.screen, PromptPeekModal)
        await pilot.press("escape")
        await pilot.pause(0.2)
        await pilot.press("ctrl+n")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ModelPickerModal)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, StudioScreen)


async def test_theme_cycle_binding(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.3)
        assert app.theme == "darkroom"
        await pilot.press("ctrl+t")
        assert app.theme == "gallery"


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #
async def test_full_run_to_contact_sheet_and_archive(app, tmp_path):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        await pilot.press("enter")  # free variant confirms in one press
        await pilot.pause(0.2)
        assert isinstance(app.screen, DarkroomScreen)
        await _wait_run_end(pilot, app)
        await pilot.pause(0.6)

        tele = app.telemetry
        assert app.run_state == "done"
        assert tele.success == tele.total
        assert all(s == CellState.OK for s in tele.states)

        run_dir = app.current_run.output_dir
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "metadata.csv").is_file()
        images = list((run_dir / "images").glob("*.png"))
        assert len(images) == tele.total

        # enter from the terminal darkroom opens the contact sheet
        await pilot.press("enter")
        await pilot.pause(0.8)
        assert isinstance(app.screen, ContactSheetScreen)
        assert app.current_mode == "archive"
        tiles = app.screen.query("ThumbTile")
        assert len(tiles) == tele.total

        # esc lands on the archive, which lists the run
        await pilot.press("escape")
        await pilot.pause(0.6)
        assert isinstance(app.screen, ArchiveScreen)
        table = app.screen.query_one("#archive-table")
        assert table.row_count == 1


async def test_studio_locked_during_run_and_unlocked_after(app, slow_provider):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)
        studio.query_one("#batch-size").value = "8"
        await pilot.pause(0.3)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.2)
        # pop back to the studio mid-run: inputs are locked, run continues
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, StudioScreen)
        assert app.run_state == "running"
        assert app.screen.query_one("#batch-size").disabled
        # ctrl+g now reopens the live darkroom instead of composing
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        assert isinstance(app.screen, DarkroomScreen)
        await _wait_run_end(pilot, app)
        await pilot.pause(0.4)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert not app.screen.query_one("#batch-size").disabled


async def test_cancel_double_press(app, slow_provider):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)
        studio.query_one("#batch-size").value = "24"
        studio.query_one("#concurrency").value = "2"
        await pilot.pause(0.3)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.4)  # let a couple of items start
        await pilot.press("ctrl+x")
        await pilot.pause(0.1)
        assert not app.cancel_requested  # armed, not yet cancelled
        await pilot.press("ctrl+x")
        await pilot.pause(0.1)
        assert app.cancel_requested
        await _wait_run_end(pilot, app)
        assert app.run_state == "cancelled"
        tele = app.telemetry
        assert tele.finished_status == "cancelled"
        assert any(s == CellState.SKIP for s in tele.states)
        # the partial run is still fully recorded on disk
        run_dir = app.current_run.output_dir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "cancelled"


async def test_quit_guard_during_run(app, slow_provider):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        studio = _studio(app)
        studio.query_one("#batch-size").value = "16"
        await pilot.pause(0.3)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.3)
        await pilot.press("ctrl+q")
        await pilot.pause(0.2)
        assert isinstance(app.screen, QuitGuardModal)
        await pilot.press("escape")  # stay
        await pilot.pause(0.2)
        assert app.run_state == "running"
        assert not isinstance(app.screen, QuitGuardModal)
        await _wait_run_end(pilot, app)


# --------------------------------------------------------------------------- #
# Pricing-unavailable flow
# --------------------------------------------------------------------------- #
async def test_unpriced_model_requires_typed_spend(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "providers": {
                    "mock": {
                        "mock-image": {
                            "display_name": "Mock (no price)",
                            "supports_size": ["1024x1024"],
                            "default_size": "1024x1024",
                            "price_per_image_usd": None,
                            "reports_actual_cost": True,
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("PORTRAIT_OUTPUT_BASE_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PORTRAIT_MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.setattr(tui_prefs, "PREFS_PATH", tmp_path / "prefs.json")
    app = PortraitApp(AppConfig.load())
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        money = app.screen.query_one("#money", MoneyBlock)
        assert money.has_class("-warn")
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        await pilot.press("enter")  # refused without the typed word
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        app.screen.query_one("#expose-spend-input").focus()
        await pilot.press(*"spend")
        await pilot.press("enter")  # arm
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        await pilot.press("enter")  # confirm
        await pilot.pause(0.3)
        assert isinstance(app.screen, DarkroomScreen)
        await _wait_run_end(pilot, app)
        assert app.run_state == "done"


# --------------------------------------------------------------------------- #
# Review-confirmed regressions
# --------------------------------------------------------------------------- #
async def test_model_card_enter_and_expose_click(app):
    """ModelCard enter must open the picker; clicking EXPOSE must work."""
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        card = app.screen.query_one("#model-display")
        card.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ModelPickerModal)
        await pilot.press("escape")
        await pilot.pause(0.3)
        await pilot.click("#expose-btn")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)


async def test_markup_unsafe_model_name_renders(app):
    """'FAL FLUX.1 [dev]' contains literal Rich markup — must never crash."""
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        app.studio_screen.set_model("fal::flux/dev")
        await pilot.pause(0.4)  # would raise MarkupError without escaping
        await pilot.press("ctrl+n")
        await pilot.pause(0.3)  # picker renders every display name
        assert isinstance(app.screen, ModelPickerModal)


async def test_generate_flow_not_reentrant(app):
    """A second generate while EXPOSE is open must not stack a second modal."""
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ExposeModal)
        await app.action_generate()
        await pilot.pause(0.2)
        modals = [s for s in app.screen_stack if isinstance(s, ExposeModal)]
        assert len(modals) == 1


async def test_f2_returns_from_darkroom(app):
    """f2 from a pushed Darkroom must pop back to the Studio (not no-op)."""
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await _wait_run_end(pilot, app)
        await pilot.pause(0.3)
        assert isinstance(app.screen, DarkroomScreen)
        await pilot.press("f2")
        await pilot.pause(0.3)
        assert isinstance(app.screen, StudioScreen)


async def test_peek_arrow_paging(app):
    """←/→ must page items even though the read-only TextArea is focused."""
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+e")
        await pilot.pause(0.2)
        assert isinstance(app.screen, PromptPeekModal)
        assert app.screen._index == 0
        await pilot.press("right")
        assert app.screen._index == 1
        await pilot.press("left")
        assert app.screen._index == 0


# --------------------------------------------------------------------------- #
# Matrix interactions
# --------------------------------------------------------------------------- #
async def test_matrix_cursor_and_color_modes(app):
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.4)
        await pilot.press("ctrl+g")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await _wait_run_end(pilot, app)
        await pilot.pause(0.5)
        matrix = app.screen.query_one("#matrix", BatchMatrix)
        matrix.focus()
        start = matrix.cursor
        await pilot.press("right")
        assert matrix.cursor == start + 1
        await pilot.press("end")
        assert matrix.cursor == matrix.total - 1
        assert matrix.color_mode == "state"
        await pilot.press("c")
        assert matrix.color_mode == "age"


# --------------------------------------------------------------------------- #
# Real-TTY startup (the headless run_test driver composes after the screen is
# stacked and so cannot catch crashes that only happen under a real terminal —
# e.g. Input(value=...) selection watchers firing during the first compose).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not hasattr(os, "fork"), reason="pty boot needs os.fork (POSIX)")
def test_boots_under_real_tty(tmp_path):
    """Boot the actual `python -m app.tui.main` under a pty and assert it does
    not crash on startup, then quit cleanly with `q`."""
    import pty

    env = dict(os.environ)
    env["PORTRAIT_OUTPUT_BASE_DIR"] = str(tmp_path / "outputs")
    env["PORTRAIT_TUI_PREFS"] = str(tmp_path / "prefs.json")
    env["TERM"] = "xterm-256color"

    pid, fd = pty.fork()
    if pid == 0:  # child: become the TUI
        os.execvpe(sys.executable, [sys.executable, "-m", "app.tui.main"], env)

    chunks: list[bytes] = []
    deadline = time.time() + 12
    quit_sent = False
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.3)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        # once it has had a moment to render, ask it to quit
        if not quit_sent and time.time() > deadline - 9:
            try:
                os.write(fd, b"q")
            except OSError:
                pass
            quit_sent = True
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError:
        pass

    output = b"".join(chunks).decode("utf-8", "replace")
    # the wordmark is gradient-styled per character (cursor moves between glyphs),
    # so it is not a contiguous substring — assert on robust signals instead.
    for marker in ("Traceback", "ScreenStackError", "NoScreen", "MarkupError", "Error"):
        assert marker not in output, f"startup crash detected ({marker}):\n{output[-2000:]}"
    assert "\x1b[?1049h" in output, "app never entered the alternate screen buffer"
    assert len(output) > 4000, "app produced almost no render — likely aborted early"
