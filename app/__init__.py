"""Synthetic portrait data-collection app.

A shared core engine (``app.core``) drives three interchangeable front-ends:
a TUI (``app.tui``), a GUI (``app.gui``) and a headless CLI (``app.cli``).
None of the front-ends contain business logic; they all build a
:class:`app.core.models.BatchGenerationRequest` and hand it to the engine.
"""

__version__ = "0.1.0"
