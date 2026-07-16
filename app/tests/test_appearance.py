"""Tests for the per-image appearance layer and the uniqueness guarantee.

These pin the behaviour that actually fixes the look-alike problem: every image
gets a distinct, demographically-plausible individual, the prompt invariants are
untouched, and the planner produces zero repeated appearance signatures — both
within a batch and against a supplied history.
"""

from __future__ import annotations

import random

from app.core.appearance import (
    Appearance,
    parse_age_range,
    sample_appearance,
)
from app.core.batch_planner import plan_batch
from app.core.buckets import AGE_BUCKETS, ETHNICITY_BUCKETS, GENDER_BUCKETS
from app.core.models import BatchGenerationRequest, DistributionMode
from app.core.prompt_builder import (
    HARD_REQUIREMENTS,
    NEGATIVE_CONSTRAINTS,
    build_prompt,
)

YOUNG = "young adult, 18 to 25"
OLDER = "older adult, 61 to 80"
ADULT = "adult, 26 to 40"
MALE = "male-presenting"
FEMALE = "female-presenting"


def _sample(seed: int, *, age=ADULT, gender=MALE, ethnicity="South Asian") -> Appearance:
    return sample_appearance(
        random.Random(seed),
        age_bucket=age,
        gender_bucket=gender,
        ethnicity_bucket=ethnicity,
    )


# --------------------------------------------------------------------------- #
# Age
# --------------------------------------------------------------------------- #
def test_parse_age_range_extracts_bounds() -> None:
    assert parse_age_range("young adult, 18 to 25") == (18, 25)
    assert parse_age_range("older adult, 61 to 80") == (61, 80)
    # No range -> safe adult fallback, never crashes.
    assert parse_age_range("some custom label") == (25, 60)


def test_exact_age_within_bucket_bounds() -> None:
    for seed in range(200):
        appr = _sample(seed, age=YOUNG)
        assert 18 <= appr.exact_age <= 25


# --------------------------------------------------------------------------- #
# Demographic plausibility gating
# --------------------------------------------------------------------------- #
def test_female_presenting_has_no_facial_hair() -> None:
    for seed in range(100):
        appr = _sample(seed, gender=FEMALE)
        assert appr.facial_hair is None


def test_male_presenting_has_a_facial_hair_value() -> None:
    for seed in range(100):
        appr = _sample(seed, gender=MALE)
        assert appr.facial_hair is not None


def test_young_subjects_never_have_grey_or_white_hair() -> None:
    grey_markers = ("grey", "white", "salt-and-pepper", "silver")
    for seed in range(300):
        appr = _sample(seed, age=YOUNG)
        assert not any(m in appr.hair_color for m in grey_markers), appr.hair_color


def test_sub_ancestry_is_specific_for_mixed_heritage() -> None:
    appr = _sample(3, ethnicity="mixed heritage")
    assert "mixed" in appr.sub_ancestry and " and " in appr.sub_ancestry


# --------------------------------------------------------------------------- #
# Signature + determinism
# --------------------------------------------------------------------------- #
def test_same_rng_seed_reproduces_identical_appearance() -> None:
    a = _sample(42)
    b = _sample(42)
    assert a == b
    assert a.signature() == b.signature()


def test_one_year_age_change_changes_signature() -> None:
    base = _sample(7)
    bumped = base.model_copy(update={"exact_age": base.exact_age + 1})
    assert base.signature() != bumped.signature()


def test_redraw_from_same_rng_yields_distinct_individuals() -> None:
    rng = random.Random(1)
    sigs = {
        sample_appearance(
            rng, age_bucket=ADULT, gender_bucket=MALE, ethnicity_bucket="White European"
        ).signature()
        for _ in range(50)
    }
    # 50 consecutive draws from one RNG should essentially never collide.
    assert len(sigs) >= 49


# --------------------------------------------------------------------------- #
# Prompt rendering keeps every invariant
# --------------------------------------------------------------------------- #
def test_prompt_with_appearance_keeps_all_invariants() -> None:
    from app.core.models import PromptOptions

    appr = _sample(5, age=ADULT, gender=FEMALE, ethnicity="East Asian")
    opts = PromptOptions(
        age_bucket=ADULT,
        gender_bucket=FEMALE,
        ethnicity_bucket="East Asian",
        appearance=appr,
    )
    prompt = build_prompt(opts)
    for req in HARD_REQUIREMENTS:
        assert req in prompt, f"missing hard requirement: {req!r}"
    for neg in NEGATIVE_CONSTRAINTS:
        assert neg in prompt, f"missing negative constraint: {neg!r}"
    assert "one synthetic human person who does not exist" in prompt
    # Bucket strings still present verbatim (other tooling greps for them).
    assert ADULT in prompt and FEMALE in prompt and "East Asian" in prompt
    # The distinct-individual section and exact age are rendered.
    assert "Distinct individual" in prompt
    assert f"{appr.exact_age} years old" in prompt


def test_prompt_without_appearance_omits_the_section() -> None:
    from app.core.models import PromptOptions

    opts = PromptOptions(age_bucket=ADULT, gender_bucket=FEMALE, ethnicity_bucket="East Asian")
    prompt = build_prompt(opts)
    assert "Distinct individual" not in prompt


# --------------------------------------------------------------------------- #
# Planner-level uniqueness — the actual "0 repetitions" guarantee
# --------------------------------------------------------------------------- #
def _full_request(total: int, *, seed: int | None = None) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=list(AGE_BUCKETS),
        gender_buckets=list(GENDER_BUCKETS),
        ethnicity_buckets=list(ETHNICITY_BUCKETS),
        distribution_mode=DistributionMode.RANDOM,
        total_count=total,
        seed=seed,
    )


def _signatures(items) -> list[str]:
    return [it.prompt_options.appearance.signature() for it in items]


def test_planner_produces_zero_repeated_signatures_in_a_batch() -> None:
    items = plan_batch(_full_request(800, seed=123))
    sigs = _signatures(items)
    assert len(sigs) == 800
    assert len(set(sigs)) == 800  # not one repeat across the whole batch


def test_planner_dedupes_against_supplied_history() -> None:
    first = plan_batch(_full_request(300, seed=123))
    history = set(_signatures(first))

    # A second batch (even with the same seed) must avoid every prior signature.
    second = plan_batch(_full_request(300, seed=123), seen_signatures=history)
    second_sigs = set(_signatures(second))
    assert history.isdisjoint(second_sigs)
    assert len(second_sigs) == 300


def test_diversify_off_leaves_appearance_unset() -> None:
    req = _full_request(5, seed=1).model_copy(update={"diversify": False})
    items = plan_batch(req)
    assert all(it.prompt_options.appearance is None for it in items)
