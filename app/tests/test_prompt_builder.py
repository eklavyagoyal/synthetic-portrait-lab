"""Tests for :mod:`app.core.prompt_builder`.

These lock in the two engine-wide invariants — every hard requirement and every
negative constraint appears verbatim in every prompt at every variation level —
plus demographic substitution, the synthetic-person framing, extra constraints
and variation-level differentiation.
"""

from __future__ import annotations

import pytest

from app.core.models import PromptOptions
from app.core.prompt_builder import (
    HARD_REQUIREMENTS,
    NEGATIVE_CONSTRAINTS,
    build_prompt,
    framing_label,
)

AGE = "adult, 26 to 40"
GENDER = "female-presenting"
ETHNICITY = "South Asian"

SYNTHETIC_FRAMING = "one synthetic human person who does not exist"


def _options(level: int = 0, **overrides: object) -> PromptOptions:
    base: dict[str, object] = dict(
        age_bucket=AGE,
        gender_bucket=GENDER,
        ethnicity_bucket=ETHNICITY,
        variation_level=level,
    )
    base.update(overrides)
    return PromptOptions(**base)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_all_hard_requirements_present_at_every_level(level: int) -> None:
    prompt = build_prompt(_options(level=level))
    for req in HARD_REQUIREMENTS:
        assert req in prompt, f"missing hard requirement at level {level}: {req!r}"


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_all_negative_constraints_present_at_every_level(level: int) -> None:
    prompt = build_prompt(_options(level=level))
    for neg in NEGATIVE_CONSTRAINTS:
        assert neg in prompt, f"missing negative constraint at level {level}: {neg!r}"


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_synthetic_person_framing_present(level: int) -> None:
    prompt = build_prompt(_options(level=level))
    assert SYNTHETIC_FRAMING in prompt


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_demographic_values_inserted_verbatim(level: int) -> None:
    prompt = build_prompt(_options(level=level))
    assert AGE in prompt
    assert GENDER in prompt
    assert ETHNICITY in prompt


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_variation_level_number_present(level: int) -> None:
    prompt = build_prompt(_options(level=level))
    assert f"variation level {level}" in prompt


def test_variation_guidance_differs_between_low_and_high() -> None:
    low = build_prompt(_options(level=0))
    high = build_prompt(_options(level=3))
    assert low != high
    # The level-0 guidance emphasises strict repeatability; level 3 permits
    # noticeably more natural variation. Their guidance text must differ.
    assert "Minimize variation" in low
    assert "Minimize variation" not in high
    assert "High natural variation" in high
    assert "High natural variation" not in low


def test_extra_positive_constraints_appear() -> None:
    extra = "Subject wearing a plain crew-neck t-shirt"
    prompt = build_prompt(_options(extra_positive_constraints=[extra]))
    assert extra in prompt


def test_extra_negative_constraints_appear() -> None:
    extra = "Jewelry"
    prompt = build_prompt(_options(extra_negative_constraints=[extra]))
    assert extra in prompt


def test_extra_constraints_absent_when_not_provided() -> None:
    """Default options must not leak placeholder extra-constraint markers."""
    only_marker = "Subject wearing a plain crew-neck t-shirt"
    prompt = build_prompt(_options())
    assert only_marker not in prompt


def test_overridable_expression_and_background_substituted() -> None:
    opts = _options(
        expression="warm, friendly smile",
        background="solid blue studio backdrop",
    )
    prompt = build_prompt(opts)
    assert "Warm, friendly smile." in prompt
    assert "Solid blue studio backdrop." in prompt


def test_seed_line_present_only_when_seed_set() -> None:
    with_seed = build_prompt(_options(seed=12345))
    assert "12345" in with_seed
    without_seed = build_prompt(_options())
    assert "Deterministic seed in effect" not in without_seed


# --- framing -------------------------------------------------------------- #
def test_composition_section_states_head_height() -> None:
    prompt = build_prompt(_options(head_height_pct=60))
    assert "Composition:" in prompt
    assert "approximately 60% of the image height" in prompt


@pytest.mark.parametrize(
    "pct,marker",
    [
        (75, "Tight headshot"),
        (60, "Standard ID-style headshot"),
        (45, "Loose portrait"),
        (30, "Upper-body portrait"),
    ],
)
def test_shoulder_instruction_varies_by_framing(pct: int, marker: str) -> None:
    assert marker in build_prompt(_options(head_height_pct=pct))


def test_framing_label_maps_presets_and_custom() -> None:
    assert framing_label(75) == "close headshot"
    assert framing_label(60) == "standard headshot"
    assert framing_label(45) == "loose headshot"
    assert framing_label(30) == "upper body"
    assert framing_label(62) == "custom"


# --- orientation ---------------------------------------------------------- #
def test_square_default_keeps_verbatim_requirement() -> None:
    # The invariant tests pin "1:1 square image."; the square default (and a
    # missing size) must keep it byte-for-byte.
    assert "1:1 square image." in build_prompt(_options())
    assert "1:1 square image." in build_prompt(_options(size="1024x1024"))


def test_landscape_and_portrait_replace_the_square_line() -> None:
    landscape = build_prompt(_options(size="1536x1024"))
    assert "1:1 square image." not in landscape
    assert "Horizontal (landscape) orientation" in landscape

    portrait = build_prompt(_options(size="1024x1536"))
    assert "1:1 square image." not in portrait
    assert "Vertical (portrait) orientation" in portrait
