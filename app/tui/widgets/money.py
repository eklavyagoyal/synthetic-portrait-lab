"""MoneyBlock — the big Digits cost readout with honest states.

States: ``ok`` (priced, green) · ``warn`` (pricing unavailable — shows ``-.--``,
never a fake 0.00) · ``free`` (priced at $0) · ``incomplete`` (form not valid
yet). The Digits charset has no em-dash (audited), hence ``-.--``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Digits, Label, Static

UNKNOWN_AMOUNT = "-.--"


class MoneyBlock(Vertical):
    DEFAULT_CSS = """
    MoneyBlock {
        height: auto;

        .money-eyebrow { color: $text-muted; text-style: bold; }
        #money-row { height: auto; align-vertical: middle; }
        #money-currency { width: 2; padding: 1 0 0 0; color: $tele-act; text-style: bold; }
        #money-digits { color: $tele-act; width: auto; }
        #money-meta { color: $text; }
        #money-model { color: $text-muted; }
        #money-source { color: $text-muted; text-style: italic; }

        &.-warn #money-digits, &.-warn #money-currency { color: $warning; }
        &.-free #money-digits, &.-free #money-currency { color: $text-muted; }
        &.-incomplete #money-digits, &.-incomplete #money-currency {
            color: $text-disabled;
        }
    }
    """

    _STATES = ("ok", "warn", "free", "incomplete")

    def __init__(self, eyebrow: str = "ESTIMATED TOTAL", **kwargs) -> None:
        super().__init__(**kwargs)
        self._eyebrow = eyebrow
        self._last_amount = ""

    def compose(self) -> ComposeResult:
        yield Static(self._eyebrow, classes="money-eyebrow")
        with Horizontal(id="money-row"):
            yield Label("$", id="money-currency")
            yield Digits("0.00", id="money-digits")
        yield Static("", id="money-meta")
        yield Static("", id="money-model")
        yield Static("", id="money-source")

    def set_amount(
        self,
        amount: str,
        state: str = "ok",
        *,
        meta: str = "",
        model: str = "",
        source: str = "",
        flash: bool = True,
    ) -> None:
        """Update the readout. ``amount`` is the pre-formatted digit string."""
        digits = self.query_one("#money-digits", Digits)
        for s in self._STATES:
            self.set_class(s == state, f"-{s}")
        changed = amount != self._last_amount
        self._last_amount = amount
        digits.update(amount)
        self.query_one("#money-meta", Static).update(meta)
        self.query_one("#money-model", Static).update(model)
        self.query_one("#money-source", Static).update(source)
        if flash and changed:
            # camera-flash blink: 1 -> 0.35 -> 1
            digits.styles.animate(
                "opacity",
                0.35,
                duration=0.10,
                easing="out_cubic",
                on_complete=lambda: digits.styles.animate(
                    "opacity", 1.0, duration=0.14, easing="out_cubic"
                ),
            )
