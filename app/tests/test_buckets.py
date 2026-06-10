"""Tests for :mod:`app.core.buckets` selection validation."""

from __future__ import annotations

import pytest

from app.core.buckets import (
    AGE_BUCKETS,
    ETHNICITY_BUCKETS,
    GENDER_BUCKETS,
    BucketConfig,
    BucketValidationError,
    validate_selection,
)

VALID_AGE = AGE_BUCKETS[0]
VALID_GENDER = GENDER_BUCKETS[0]
VALID_ETHNICITY = ETHNICITY_BUCKETS[0]


def test_valid_default_selection_passes() -> None:
    # Should not raise.
    validate_selection(
        age_buckets=list(AGE_BUCKETS),
        gender_buckets=list(GENDER_BUCKETS),
        ethnicity_buckets=list(ETHNICITY_BUCKETS),
    )


def test_single_valid_bucket_per_dimension_passes() -> None:
    validate_selection(
        age_buckets=[VALID_AGE],
        gender_buckets=[VALID_GENDER],
        ethnicity_buckets=[VALID_ETHNICITY],
    )


@pytest.mark.parametrize(
    ("age_buckets", "gender_buckets", "ethnicity_buckets"),
    [
        (["not-an-age"], [VALID_GENDER], [VALID_ETHNICITY]),
        ([VALID_AGE], ["not-a-gender"], [VALID_ETHNICITY]),
        ([VALID_AGE], [VALID_GENDER], ["not-an-ethnicity"]),
    ],
    ids=["unknown_age", "unknown_gender", "unknown_ethnicity"],
)
def test_unknown_bucket_in_any_dimension_raises(
    age_buckets: list[str],
    gender_buckets: list[str],
    ethnicity_buckets: list[str],
) -> None:
    with pytest.raises(BucketValidationError):
        validate_selection(
            age_buckets=age_buckets,
            gender_buckets=gender_buckets,
            ethnicity_buckets=ethnicity_buckets,
        )


def test_allow_custom_accepts_otherwise_rejected_bucket() -> None:
    cfg = BucketConfig(allow_custom=True)
    custom_age = "centenarian, 100+"
    # Without allow_custom this would raise; with it, the custom value passes.
    with pytest.raises(BucketValidationError):
        validate_selection(
            age_buckets=[custom_age],
            gender_buckets=[VALID_GENDER],
            ethnicity_buckets=[VALID_ETHNICITY],
        )
    validate_selection(
        age_buckets=[custom_age],
        gender_buckets=[VALID_GENDER],
        ethnicity_buckets=[VALID_ETHNICITY],
        config=cfg,
    )


def test_allow_custom_still_rejects_empty_string_bucket() -> None:
    cfg = BucketConfig(allow_custom=True)
    with pytest.raises(BucketValidationError):
        validate_selection(
            age_buckets=["   "],
            gender_buckets=[VALID_GENDER],
            ethnicity_buckets=[VALID_ETHNICITY],
            config=cfg,
        )


@pytest.mark.parametrize(
    ("age_buckets", "gender_buckets", "ethnicity_buckets"),
    [
        ([], [VALID_GENDER], [VALID_ETHNICITY]),
        ([VALID_AGE], [], [VALID_ETHNICITY]),
        ([VALID_AGE], [VALID_GENDER], []),
    ],
    ids=["empty_age", "empty_gender", "empty_ethnicity"],
)
def test_empty_selection_in_any_dimension_raises(
    age_buckets: list[str],
    gender_buckets: list[str],
    ethnicity_buckets: list[str],
) -> None:
    with pytest.raises(BucketValidationError):
        validate_selection(
            age_buckets=age_buckets,
            gender_buckets=gender_buckets,
            ethnicity_buckets=ethnicity_buckets,
        )
