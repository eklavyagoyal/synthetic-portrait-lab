"""Entry point for the Synthetic Portrait Lab TUI.

The application lives in :mod:`app.tui.app` (screens, widgets, telemetry and
themes are siblings in this package). Importing this module never starts the
app — call :func:`main` for that, or use the ``portrait-tui`` console script.
"""

from __future__ import annotations

from app.core.config import AppConfig

from .app import PortraitApp


def main() -> None:
    """Load config and run the TUI."""
    config = AppConfig.load()
    PortraitApp(config).run()


if __name__ == "__main__":
    main()
