"""Turns a :class:`BatchGenerationRequest` into a concrete, ordered work list.

Supports four distribution modes:

* ``EVEN``     — round-robin across every selected demographic triple (counts
  differ by at most one).
* ``RANDOM``   — each slot draws a triple uniformly at random from the selection.
* ``WEIGHTED`` — each slot is sampled with probability proportional to the
  product of its three buckets' weights (unspecified buckets default to 1.0).
* ``EXACT``    — explicit per-triple counts; total is their sum.

A seed makes every mode (including RANDOM/WEIGHTED) fully reproducible. Per-item
seeds are derived as ``base_seed + index`` so that images stay distinct while the
whole run remains repeatable.
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import product

from .models import (
    BatchGenerationRequest,
    DistributionMode,
    PlannedItem,
    PromptOptions,
)

Triple = tuple[str, str, str]  # (age, gender, ethnicity)


class PlanningError(ValueError):
    """Raised when a request cannot be turned into a valid plan."""


def _combos(req: BatchGenerationRequest) -> list[Triple]:
    combos = [
        (a, g, e)
        for a in req.age_buckets
        for g in req.gender_buckets
        for e in req.ethnicity_buckets
    ]
    if not combos:
        raise PlanningError("No demographic combinations from the selected buckets.")
    return combos


def _marginally_balanced_order(combos: list[Triple], length: int) -> list[Triple]:
    """Order ``combos`` so every prefix is as marginally balanced as possible
    across each axis (age, gender, ethnicity) *independently*.

    Greedy: repeatedly take the unused combo whose currently most-used bucket
    (across its three axes) is smallest, tie-broken toward the smallest total
    bucket usage and then the original combo index — fully deterministic. This
    keeps each axis's per-bucket counts within one of each other at every step,
    so 8 images across 8 ethnicities yield one image per ethnicity rather than
    eight of the first. Returns ``min(length, len(combos))`` combos.

    (The previous radical-inverse sort grouped the prefix by the highest-radix
    axis — e.g. all of ethnicity 0 first — which is exactly the bug this fixes.)
    """
    age_counts: Counter = Counter()
    gender_counts: Counter = Counter()
    eth_counts: Counter = Counter()
    used = [False] * len(combos)
    chosen: list[Triple] = []
    for _ in range(min(length, len(combos))):
        best_idx = -1
        best_key: tuple | None = None
        for idx, (age, gender, eth) in enumerate(combos):
            if used[idx]:
                continue
            ca, cg, ce = age_counts[age], gender_counts[gender], eth_counts[eth]
            key = (max(ca, cg, ce), ca + cg + ce, idx)
            if best_key is None or key < best_key:
                best_key, best_idx = key, idx
        age, gender, eth = combos[best_idx]
        age_counts[age] += 1
        gender_counts[gender] += 1
        eth_counts[eth] += 1
        used[best_idx] = True
        chosen.append(combos[best_idx])
    return chosen


def _even(req: BatchGenerationRequest) -> list[Triple]:
    """EVEN: balance the marginal counts of each selected axis as closely as
    possible. For N images and an axis with K selected buckets, each bucket
    appears ``floor(N/K)`` or ``ceil(N/K)`` times (exactly ``N/K`` when N % K == 0).

    When N >= the number of combos we order ALL combos into one balanced
    permutation and cycle it (covering every combo, marginals still balanced);
    when N < combos we take just the balanced prefix we need.
    """
    combos = _combos(req)
    span = len(combos) if req.total_count >= len(combos) else req.total_count
    ordered = _marginally_balanced_order(combos, span)
    return [ordered[i % len(ordered)] for i in range(req.total_count)]


def _random(req: BatchGenerationRequest, rng: random.Random) -> list[Triple]:
    combos = _combos(req)
    return rng.choices(combos, k=req.total_count)


def _weighted(req: BatchGenerationRequest, rng: random.Random) -> list[Triple]:
    combos = _combos(req)
    weights_map = req.weights or {}

    def combo_weight(triple: Triple) -> float:
        w = 1.0
        for token in triple:
            w *= float(weights_map.get(token, 1.0))
        return w

    weights = [combo_weight(c) for c in combos]
    if sum(weights) <= 0:
        raise PlanningError("WEIGHTED distribution produced a total weight of zero.")
    return rng.choices(combos, weights=weights, k=req.total_count)


def _exact(req: BatchGenerationRequest) -> list[Triple]:
    if not req.exact_counts:
        raise PlanningError("EXACT distribution requires exact_counts.")
    triples: list[Triple] = []
    for ec in req.exact_counts:
        triples.extend(
            [(ec.age_bucket, ec.gender_bucket, ec.ethnicity_bucket)] * ec.count
        )
    if not triples:
        raise PlanningError("EXACT distribution resolved to zero items.")
    return triples


def plan_batch(req: BatchGenerationRequest) -> list[PlannedItem]:
    """Produce the ordered list of :class:`PlannedItem` for a request."""
    rng = random.Random(req.seed)

    if req.distribution_mode == DistributionMode.EVEN:
        triples = _even(req)
    elif req.distribution_mode == DistributionMode.RANDOM:
        triples = _random(req, rng)
    elif req.distribution_mode == DistributionMode.WEIGHTED:
        triples = _weighted(req, rng)
    elif req.distribution_mode == DistributionMode.EXACT:
        triples = _exact(req)
    else:  # pragma: no cover - exhaustive
        raise PlanningError(f"Unknown distribution mode: {req.distribution_mode}")

    items: list[PlannedItem] = []
    for index, (age, gender, ethnicity) in enumerate(triples):
        item_seed = None if req.seed is None else req.seed + index
        item_id = f"{req.filename_prefix}_{index + 1:06d}"
        options = PromptOptions(
            age_bucket=age,
            gender_bucket=gender,
            ethnicity_bucket=ethnicity,
            variation_level=req.variation_level,
            head_height_pct=req.head_height_pct,
            size=req.size,
            background=req.background,
            expression=req.expression,
            lighting=req.lighting,
            image_style=req.image_style,
            extra_positive_constraints=list(req.extra_positive_constraints),
            extra_negative_constraints=list(req.extra_negative_constraints),
            seed=item_seed,
        )
        items.append(
            PlannedItem(
                index=index,
                id=item_id,
                filename=f"{item_id}.png",
                prompt_options=options,
            )
        )
    return items
