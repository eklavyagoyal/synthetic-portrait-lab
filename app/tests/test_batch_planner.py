"""Tests for :mod:`app.core.batch_planner`.

Verifies each distribution mode (EVEN, RANDOM, WEIGHTED, EXACT), that every
planned triple stays inside the selected buckets, the deterministic id/filename
naming, and seed behaviour: identical seeds reproduce identical plans while
per-item seeds still differ so generated images stay distinct.
"""

from __future__ import annotations

from collections import Counter

from app.core.batch_planner import plan_batch
from app.core.buckets import AGE_BUCKETS, ETHNICITY_BUCKETS, GENDER_BUCKETS
from app.core.models import (
    BatchGenerationRequest,
    DistributionMode,
    ExactCount,
)
from app.core.prompt_builder import build_prompt

AGES = ["young adult, 18 to 25", "adult, 26 to 40"]
GENDERS = ["male-presenting", "female-presenting"]
ETHNICITIES = ["East Asian", "White European"]


def _selection() -> set[tuple[str, str, str]]:
    return {(a, g, e) for a in AGES for g in GENDERS for e in ETHNICITIES}


def _request(
    mode: DistributionMode,
    *,
    total: int = 0,
    seed: int | None = None,
    weights: dict[str, float] | None = None,
    exact_counts: list[ExactCount] | None = None,
) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=list(AGES),
        gender_buckets=list(GENDERS),
        ethnicity_buckets=list(ETHNICITIES),
        distribution_mode=mode,
        total_count=total,
        seed=seed,
        weights=weights,
        exact_counts=exact_counts,
    )


def _triple(item) -> tuple[str, str, str]:
    o = item.prompt_options
    return (o.age_bucket, o.gender_bucket, o.ethnicity_bucket)


def test_framing_and_size_threaded_into_prompt_options() -> None:
    req = _request(DistributionMode.EVEN, total=4).model_copy(
        update={"head_height_pct": 75, "size": "1536x1024"}
    )
    items = plan_batch(req)
    assert items, "expected a non-empty plan"
    for it in items:
        assert it.prompt_options.head_height_pct == 75
        assert it.prompt_options.size == "1536x1024"


def test_even_balances_combos_within_selection() -> None:
    combos = _selection()
    total = len(combos) * 3 + 2  # not a clean multiple, to force a +1 imbalance
    items = plan_batch(_request(DistributionMode.EVEN, total=total))

    assert len(items) == total

    counts = Counter(_triple(it) for it in items)
    # Every selected combo is used at least once.
    assert set(counts) == combos
    # Round-robin: per-combo counts differ by at most one.
    assert max(counts.values()) - min(counts.values()) <= 1
    # Every triple is within the selected buckets.
    assert all(t in combos for t in counts)


def test_random_stays_within_selection_and_hits_total() -> None:
    combos = _selection()
    total = 50
    items = plan_batch(_request(DistributionMode.RANDOM, total=total, seed=7))

    assert len(items) == total
    assert all(_triple(it) in combos for it in items)


def test_weighted_stays_within_selection_and_hits_total() -> None:
    combos = _selection()
    total = 40
    weights = {
        "young adult, 18 to 25": 3.0,
        "female-presenting": 2.0,
        "East Asian": 1.5,
    }
    items = plan_batch(
        _request(DistributionMode.WEIGHTED, total=total, seed=11, weights=weights)
    )

    assert len(items) == total
    assert all(_triple(it) in combos for it in items)


def test_exact_counts_match_per_triple() -> None:
    exact_counts = [
        ExactCount(
            age_bucket="adult, 26 to 40",
            gender_bucket="female-presenting",
            ethnicity_bucket="White European",
            count=3,
        ),
        ExactCount(
            age_bucket="young adult, 18 to 25",
            gender_bucket="male-presenting",
            ethnicity_bucket="East Asian",
            count=2,
        ),
    ]
    items = plan_batch(_request(DistributionMode.EXACT, exact_counts=exact_counts))

    assert len(items) == sum(ec.count for ec in exact_counts)

    counts = Counter(_triple(it) for it in items)
    for ec in exact_counts:
        triple = (ec.age_bucket, ec.gender_bucket, ec.ethnicity_bucket)
        assert counts[triple] == ec.count


def test_seed_reproducible_random_and_weighted() -> None:
    rnd_a = plan_batch(_request(DistributionMode.RANDOM, total=30, seed=99))
    rnd_b = plan_batch(_request(DistributionMode.RANDOM, total=30, seed=99))
    assert [_triple(i) for i in rnd_a] == [_triple(i) for i in rnd_b]

    weights = {"male-presenting": 2.0, "White European": 3.0}
    w_a = plan_batch(
        _request(DistributionMode.WEIGHTED, total=30, seed=99, weights=weights)
    )
    w_b = plan_batch(
        _request(DistributionMode.WEIGHTED, total=30, seed=99, weights=weights)
    )
    assert [_triple(i) for i in w_a] == [_triple(i) for i in w_b]


def test_ids_are_zero_padded_sequential_and_filenames_match() -> None:
    items = plan_batch(_request(DistributionMode.EVEN, total=3))

    expected_ids = ["portrait_000001", "portrait_000002", "portrait_000003"]
    assert [it.id for it in items] == expected_ids
    assert [it.index for it in items] == [0, 1, 2]
    assert all(it.filename == f"{it.id}.png" for it in items)


def test_per_item_seed_differs_but_plan_is_reproducible() -> None:
    base_seed = 1000
    items_a = plan_batch(_request(DistributionMode.RANDOM, total=5, seed=base_seed))

    # Per-item seeds are derived (base_seed + index) so images stay distinct.
    seeds = [it.prompt_options.seed for it in items_a]
    assert seeds == [base_seed + i for i in range(5)]
    assert len(set(seeds)) == len(seeds)

    # The plan as a whole is still reproducible for the same base seed.
    items_b = plan_batch(_request(DistributionMode.RANDOM, total=5, seed=base_seed))
    assert [_triple(i) for i in items_a] == [_triple(i) for i in items_b]
    assert [it.prompt_options.seed for it in items_b] == seeds


def test_no_seed_leaves_per_item_seeds_none() -> None:
    items = plan_batch(_request(DistributionMode.EVEN, total=2, seed=None))
    assert all(it.prompt_options.seed is None for it in items)


# --------------------------------------------------------------------------- #
# EVEN marginal balance — regression for the "all East Asian" distribution bug
# (8 imgs / 8 ethnicities previously produced 8× the first ethnicity).
# --------------------------------------------------------------------------- #
def _even_req(*, ages, genders, eths, total, seed=None) -> BatchGenerationRequest:
    return BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=list(ages),
        gender_buckets=list(genders),
        ethnicity_buckets=list(eths),
        distribution_mode=DistributionMode.EVEN,
        total_count=total,
        seed=seed,
    )


def _axis_counts(items):
    return (
        Counter(i.prompt_options.age_bucket for i in items),
        Counter(i.prompt_options.gender_bucket for i in items),
        Counter(i.prompt_options.ethnicity_bucket for i in items),
    )


def _spread(counter, selected) -> int:
    vals = [counter.get(b, 0) for b in selected]
    return max(vals) - min(vals)


def test_even_case_a_eight_images_eight_ethnicities_each_once():
    eths = ETHNICITY_BUCKETS  # 8 buckets
    items = plan_batch(
        _even_req(ages=AGE_BUCKETS[:1], genders=GENDER_BUCKETS[:1], eths=eths, total=8)
    )
    _, _, ec = _axis_counts(items)
    assert len(items) == 8
    assert all(ec[e] == 1 for e in eths), ec  # one image per ethnicity, not 8 of one


def test_even_case_b_ten_images_four_ethnicities_differ_by_one():
    eths = ETHNICITY_BUCKETS[:4]
    items = plan_batch(
        _even_req(ages=AGE_BUCKETS[:1], genders=GENDER_BUCKETS[:1], eths=eths, total=10)
    )
    _, _, ec = _axis_counts(items)
    assert sorted(ec[e] for e in eths) == [2, 2, 3, 3]
    assert _spread(ec, eths) <= 1


def test_even_case_c_three_axes_marginally_balanced():
    ages, genders, eths = AGE_BUCKETS, GENDER_BUCKETS[:2], ETHNICITY_BUCKETS  # 4, 2, 8
    items = plan_batch(_even_req(ages=ages, genders=genders, eths=eths, total=8))
    ac, gc, ec = _axis_counts(items)
    assert all(ac[a] == 2 for a in ages), ac      # 8 / 4 ages  = 2 each
    assert all(gc[g] == 4 for g in genders), gc   # 8 / 2 genders = 4 each
    assert all(ec[e] == 1 for e in eths), ec      # 8 / 8 ethnicities = 1 each (the bug)
    assert _spread(ac, ages) == 0
    assert _spread(gc, genders) == 0
    assert _spread(ec, eths) == 0


def test_even_case_d_deterministic_and_seed_independent():
    kw = dict(ages=AGE_BUCKETS, genders=GENDER_BUCKETS[:2], eths=ETHNICITY_BUCKETS, total=20)
    a = [_triple(i) for i in plan_batch(_even_req(seed=42, **kw))]
    b = [_triple(i) for i in plan_batch(_even_req(seed=42, **kw))]
    c = [_triple(i) for i in plan_batch(_even_req(seed=None, **kw))]
    assert a == b == c  # EVEN is fully deterministic (seed-independent)


def test_even_case_e_prompt_uses_each_items_own_ethnicity():
    eths = ETHNICITY_BUCKETS  # 8
    items = plan_batch(
        _even_req(ages=AGE_BUCKETS[:1], genders=GENDER_BUCKETS[:1], eths=eths, total=8)
    )
    # Each planned item's prompt must carry that item's OWN ethnicity, not a
    # global/first value — and the batch must cover all eight ethnicities.
    for it in items:
        assert it.prompt_options.ethnicity_bucket in build_prompt(it.prompt_options)
    assert {it.prompt_options.ethnicity_bucket for it in items} == set(eths)
