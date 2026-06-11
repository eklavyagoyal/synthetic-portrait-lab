"""Provider interface.

A provider's one job: take a fully-rendered prompt and return encoded image
bytes (plus an actual cost, if the API reports one). Providers know nothing
about runs, planning, metadata or UIs.

Contract for new providers (followed by ``openai``/``fal``/``replicate``):

* subclass :class:`ImageProvider`;
* set the class attribute ``provider_name``;
* accept ``api_key``/``base_url``/``timeout`` via ``__init__`` (kwargs tolerated);
* implement ``async def generate(self, *, prompt, size, model_id, seed, quality) -> ProviderResult``;
* raise :class:`ProviderAuthError` for missing/invalid credentials and
  :class:`ProviderError` for any other failure (the engine handles retries);
* return :class:`app.core.models.ProviderResult` with real image bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from ..models import ProviderResult


class ProviderError(RuntimeError):
    """Generic, retryable provider failure."""


class ProviderAuthError(ProviderError):
    """Missing or invalid credentials. Not worth retrying."""


class ImageProvider(ABC):
    provider_name: ClassVar[str] = "base"
    requires_api_key: ClassVar[bool] = True
    api_endpoint: ClassVar[str] = ""   # the API path this provider calls (audit trail)

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        **kwargs,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.options = kwargs

    def ensure_ready(self) -> None:
        """Validate that the provider can be used. Raises :class:`ProviderAuthError`."""
        if self.requires_api_key and not self.api_key:
            raise ProviderAuthError(
                f"Provider '{self.provider_name}' requires an API key but none was provided. "
                "Set the appropriate variable in your .env file."
            )

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        size: str,
        model_id: str,
        seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        """Generate one image. Must return real encoded bytes or raise."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any held resources (e.g. an httpx client). Optional."""
        return None
