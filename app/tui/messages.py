"""Message dataclasses crossing the worker → main-thread boundary.

The engine's event callback and all thread workers (image decodes, archive
scans) communicate with widgets exclusively by posting these messages —
no widget is ever touched off the main thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.message import Message

from app.core.models import GenerationEvent, Run

from .runscan import RunRecord


class EngineEvent(Message):
    """One :class:`GenerationEvent` from the generation worker."""

    def __init__(self, event: GenerationEvent) -> None:
        self.event = event
        super().__init__()


class RunFinished(Message):
    """The generation worker completed (successfully, cancelled, or errored)."""

    def __init__(self, run: Optional[Run], error: Optional[str]) -> None:
        self.run = run
        self.error = error
        super().__init__()


class ThumbReady(Message):
    """A thread worker finished decoding an image to half-block art."""

    def __init__(self, key: str, path: Path, text: Optional[Text], error: Optional[str] = None) -> None:
        self.key = key
        self.path = path
        self.text = text
        self.error = error
        super().__init__()


class ArchiveScanned(Message):
    """The archive scanner finished reading manifests off the main thread."""

    def __init__(self, records: list[RunRecord]) -> None:
        self.records = records
        super().__init__()


class SheetLoaded(Message):
    """Contact-sheet metadata (manifest + metadata.jsonl) finished loading."""

    def __init__(self, header: dict, items: list[dict], error: Optional[str] = None) -> None:
        self.header = header
        self.items = items
        self.error = error
        super().__init__()
