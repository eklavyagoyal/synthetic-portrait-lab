"""Demographic buckets and their validation.

Defaults are editable via :class:`BucketConfig` (which the app config can load
from a JSON file). The :func:`validate_selection` function is the single
authority that decides whether a requested bucket is acceptable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Defaults (editable in app config — these are not hard-coded into logic)
# --------------------------------------------------------------------------- #
AGE_BUCKETS: list[str] = [
    "young adult, 18 to 25",
    "adult, 26 to 40",
    "middle-aged adult, 41 to 60",
    "older adult, 61 to 80",
]

GENDER_BUCKETS: list[str] = [
    "male-presenting",
    "female-presenting",
    "androgynous or non-binary-presenting",
]

ETHNICITY_BUCKETS: list[str] = [
    "East Asian",
    "South Asian",
    "Southeast Asian",
    "Black African descent",
    "Middle Eastern or North African",
    "Latino or Hispanic",
    "White European",
    "mixed heritage",
]


class BucketValidationError(ValueError):
    """Raised when a requested bucket is not part of the active configuration."""


class BucketConfig(BaseModel):
    """The active set of selectable buckets for a session.

    ``allow_custom`` relaxes validation so that buckets outside the configured
    lists are accepted (useful for ad-hoc experiments). When False (the default),
    unknown buckets are rejected.
    """

    age: list[str] = Field(default_factory=lambda: list(AGE_BUCKETS))
    gender: list[str] = Field(default_factory=lambda: list(GENDER_BUCKETS))
    ethnicity: list[str] = Field(default_factory=lambda: list(ETHNICITY_BUCKETS))
    allow_custom: bool = False

    @classmethod
    def default(cls) -> "BucketConfig":
        return cls()

    def _check(self, kind: str, value: str, allowed: list[str]) -> None:
        if self.allow_custom:
            if not value or not value.strip():
                raise BucketValidationError(f"{kind} bucket must be a non-empty string.")
            return
        if value not in allowed:
            raise BucketValidationError(
                f"Unknown {kind} bucket: {value!r}. "
                f"Allowed: {allowed}. (Enable allow_custom to use custom buckets.)"
            )

    def validate_age(self, value: str) -> None:
        self._check("age", value, self.age)

    def validate_gender(self, value: str) -> None:
        self._check("gender", value, self.gender)

    def validate_ethnicity(self, value: str) -> None:
        self._check("ethnicity", value, self.ethnicity)


def validate_selection(
    *,
    age_buckets: list[str],
    gender_buckets: list[str],
    ethnicity_buckets: list[str],
    config: BucketConfig | None = None,
) -> None:
    """Validate a whole selection. Raises :class:`BucketValidationError` on the
    first offending bucket. An empty selection in any dimension is rejected."""
    cfg = config or BucketConfig.default()
    if not age_buckets:
        raise BucketValidationError("No age bucket selected.")
    if not gender_buckets:
        raise BucketValidationError("No gender bucket selected.")
    if not ethnicity_buckets:
        raise BucketValidationError("No ethnicity bucket selected.")
    for a in age_buckets:
        cfg.validate_age(a)
    for g in gender_buckets:
        cfg.validate_gender(g)
    for e in ethnicity_buckets:
        cfg.validate_ethnicity(e)
