"""Image-size catalogue and validation.

Most models accept only the fixed sizes listed in their registry
``supports_size``. OpenAI's ``gpt-image-2`` additionally accepts arbitrary
custom resolutions that satisfy a set of constraints, so for that model we
validate against the math rather than a fixed preset list.

The constraints below are OpenAI's documented limits for ``gpt-image-2``. They
are bundled as a convenience; verify against the current OpenAI image docs.
"""

from __future__ import annotations

from typing import Optional

from .models import ModelInfo

# gpt-image-2 custom-size constraints.
GPT_IMAGE_2_EDGE_MULTIPLE = 16
GPT_IMAGE_2_MAX_EDGE = 3840
GPT_IMAGE_2_MAX_ASPECT = 3.0
GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400

# Provider/model pairs that accept custom sizes beyond their listed presets.
_CUSTOM_SIZE_MODELS = {("openai", "gpt-image-2")}


def parse_size(size: str) -> Optional[tuple[int, int]]:
    """Parse ``"WIDTHxHEIGHT"`` into ``(width, height)``; ``None`` if malformed."""
    if not isinstance(size, str):
        return None
    parts = size.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        return None
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def accepts_custom_sizes(provider: str, model_id: str) -> bool:
    """Whether the model accepts arbitrary valid sizes (not just its presets)."""
    return (provider, model_id) in _CUSTOM_SIZE_MODELS


def gpt_image_2_size_error(width: int, height: int) -> Optional[str]:
    """Return a human-readable reason ``width x height`` is invalid for
    gpt-image-2, or ``None`` if the size is valid."""
    if width % GPT_IMAGE_2_EDGE_MULTIPLE or height % GPT_IMAGE_2_EDGE_MULTIPLE:
        return f"both edges must be multiples of {GPT_IMAGE_2_EDGE_MULTIPLE} (got {width}x{height})"
    if max(width, height) > GPT_IMAGE_2_MAX_EDGE:
        return f"the longest edge must be <= {GPT_IMAGE_2_MAX_EDGE}px (got {max(width, height)})"
    ratio = max(width, height) / min(width, height)
    if ratio > GPT_IMAGE_2_MAX_ASPECT + 1e-9:
        return f"the aspect ratio must be <= 3:1 (got {ratio:.2f}:1)"
    pixels = width * height
    if pixels < GPT_IMAGE_2_MIN_PIXELS:
        return f"total pixels must be >= {GPT_IMAGE_2_MIN_PIXELS:,} (got {pixels:,})"
    if pixels > GPT_IMAGE_2_MAX_PIXELS:
        return f"total pixels must be <= {GPT_IMAGE_2_MAX_PIXELS:,} (got {pixels:,})"
    return None


def is_valid_gpt_image_2_size(width: int, height: int) -> bool:
    """True if ``width x height`` satisfies every gpt-image-2 constraint."""
    return gpt_image_2_size_error(width, height) is None


def validate_request_size(size: str, model_info: ModelInfo) -> None:
    """Validate a requested size for a model; raise ``ValueError`` if unusable.

    Fast path: any size listed in the model's ``supports_size`` is accepted. For
    models that accept custom sizes (gpt-image-2), any size satisfying the
    provider's constraints is accepted too. Everything else is rejected with a
    message that explains why.
    """
    if size in model_info.supports_size:
        return
    if accepts_custom_sizes(model_info.provider, model_info.model_id):
        parsed = parse_size(size)
        if parsed is None:
            raise ValueError(
                f"Invalid size {size!r}; expected a 'WIDTHxHEIGHT' value, e.g. 1536x1024."
            )
        reason = gpt_image_2_size_error(*parsed)
        if reason is None:
            return
        raise ValueError(
            f"Size {size!r} is not valid for {model_info.provider}/{model_info.model_id}: {reason}."
        )
    raise ValueError(
        f"Size {size!r} not supported by {model_info.provider}/{model_info.model_id}. "
        f"Supported: {model_info.supports_size}"
    )
