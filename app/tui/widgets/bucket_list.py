"""Demographic bucket picker — a SelectionList with dimension-accent chrome
and bulk-select keys (``a`` all · ``n`` none · ``i`` invert)."""

from __future__ import annotations

from textual import events, on
from textual.widgets import SelectionList
from textual.widgets.selection_list import Selection

from .. import labels


class BucketList(SelectionList[str]):
    """One demographic dimension. Border shows the dimension name + count."""

    DEFAULT_CSS = """
    BucketList {
        height: auto;
        max-height: 7;
        margin: 0 0 1 0;
        padding: 0 1;
        border: round $border-soft;
        background: $surface;
        &:focus { background: $boost; border: round $border-strong; }
        &.-empty { border: round $error; }
        &.-dim-age { border-title-color: $age-accent; }
        &.-dim-gender { border-title-color: $gender-accent; }
        &.-dim-ethnicity { border-title-color: $eth-accent; }
    }
    """

    def __init__(self, dim: str, buckets: list[str], selected: list[str], **kwargs) -> None:
        selections = [
            Selection(labels.short(bucket), bucket, bucket in selected)
            for bucket in buckets
        ]
        super().__init__(*selections, **kwargs)
        self.dim = dim
        self.add_class(f"-dim-{dim}")
        self.border_title = f"{labels.DIM_GLYPH.get(dim, '')} {dim}"
        self.tooltip = (
            f"{dim} buckets — space toggles, a all, n none, i invert.\n"
            + "\n".join(f"• {b}" for b in buckets)
        )
        self._total = len(buckets)

    def on_mount(self) -> None:
        self._sync_chrome()

    @on(SelectionList.SelectedChanged)
    def _sync_chrome(self) -> None:
        count = len(self.selected)
        self.border_subtitle = f"{count}/{self._total}"
        self.set_class(count == 0, "-empty")

    def on_key(self, event: events.Key) -> None:
        if event.key == "a":
            event.stop()
            self.select_all()
        elif event.key == "n":
            event.stop()
            self.deselect_all()
        elif event.key == "i":
            event.stop()
            self.toggle_all()
