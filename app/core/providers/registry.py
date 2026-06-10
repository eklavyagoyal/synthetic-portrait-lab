"""Provider factory.

Maps a provider name to its :class:`ImageProvider` subclass. Optional providers
are imported lazily so a missing/under-construction provider module never breaks
the engine or the always-available ``mock`` provider.
"""

from __future__ import annotations

from typing import Type

from .base import ImageProvider, ProviderError

# Provider names that the registry knows how to construct.
KNOWN_PROVIDERS = ("mock", "openai", "fal", "replicate", "google")


def _load_class(provider_name: str) -> Type[ImageProvider]:
    if provider_name == "mock":
        from .mock_provider import MockProvider

        return MockProvider
    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider
    if provider_name == "fal":
        from .fal_provider import FalProvider

        return FalProvider
    if provider_name == "replicate":
        from .replicate_provider import ReplicateProvider

        return ReplicateProvider
    if provider_name == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider
    raise ProviderError(f"Unknown provider: {provider_name!r}")


def available_providers() -> list[str]:
    """Provider names whose module imports successfully right now."""
    available = []
    for name in KNOWN_PROVIDERS:
        try:
            _load_class(name)
            available.append(name)
        except Exception:
            continue
    return available


def build_provider(provider_name: str, **kwargs) -> ImageProvider:
    """Instantiate a provider by name. ``kwargs`` are passed to its constructor
    (typically ``api_key`` / ``base_url`` from :meth:`AppConfig.credentials_for`)."""
    try:
        cls = _load_class(provider_name)
    except ImportError as exc:  # provider module not present / not yet implemented
        raise ProviderError(
            f"Provider '{provider_name}' is not available: {exc}"
        ) from exc
    return cls(**kwargs)
