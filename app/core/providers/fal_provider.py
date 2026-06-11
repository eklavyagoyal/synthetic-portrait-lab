"""FAL image provider.

Talks to FAL's synchronous run endpoint (``https://fal.run/{endpoint}``), which
takes a JSON body and returns an ``images`` array of hosted URLs. We download the
first image and hand its bytes back as a :class:`ProviderResult`.

The registry ``model_id`` (e.g. ``"imagen4"``, ``"flux/dev"``) is mapped to a FAL
endpoint path; unknown ids fall back to ``fal-ai/{model_id}``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..models import ProviderResult
from .base import ImageProvider, ProviderAuthError, ProviderError

_FAL_BASE_URL = "https://fal.run"

# Map registry model ids to FAL endpoint paths.
_ENDPOINTS: dict[str, str] = {
    "imagen4": "fal-ai/imagen4/preview",
    "flux/dev": "fal-ai/flux/dev",
}

# FAL preset names for common square sizes, keyed by "WxH".
_SIZE_PRESETS: dict[str, str] = {
    "1024x1024": "square_hd",
}


def _endpoint_for(model_id: str) -> str:
    """Resolve a registry ``model_id`` to a FAL endpoint path."""
    return _ENDPOINTS.get(model_id, f"fal-ai/{model_id}")


def _image_size(size: str) -> object:
    """Translate a ``"WxH"`` size string into FAL's ``image_size`` value.

    Returns a known preset string when available, otherwise an explicit
    ``{"width": W, "height": H}`` object. Falls back to ``"square_hd"`` if the
    string cannot be parsed.
    """
    preset = _SIZE_PRESETS.get(size.lower())
    if preset is not None:
        return preset
    try:
        w, h = size.lower().split("x")
        return {"width": int(w), "height": int(h)}
    except (ValueError, AttributeError):
        return "square_hd"


def _short_body(text: str, limit: int = 300) -> str:
    """Truncate a response body for inclusion in error messages."""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class FalProvider(ImageProvider):
    """Generate images via FAL's synchronous HTTP API."""

    provider_name = "fal"
    requires_api_key = True
    api_endpoint = "fal.run/{model}"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or _FAL_BASE_URL,
            timeout=timeout,
            **kwargs,
        )
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the shared async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def generate(
        self,
        *,
        prompt: str,
        size: str,
        model_id: str,
        seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        self.ensure_ready()

        endpoint = _endpoint_for(model_id)
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        headers = {"Authorization": f"Key {self.api_key}"}
        body: dict[str, object] = {
            "prompt": prompt,
            "image_size": _image_size(size),
        }
        if seed is not None:
            body["seed"] = seed

        client = self._get_client()
        try:
            response = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"FAL request to {endpoint} failed: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"FAL authentication failed ({response.status_code}). "
                "Check that FAL_KEY is set to a valid key."
            )
        if not response.is_success:
            raise ProviderError(
                f"FAL request to {endpoint} returned "
                f"{response.status_code}: {_short_body(response.text)}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"FAL returned a non-JSON response: {_short_body(response.text)}"
            ) from exc

        images = payload.get("images") if isinstance(payload, dict) else None
        if not images:
            raise ProviderError(
                f"FAL response contained no images: {_short_body(response.text)}"
            )

        first = images[0]
        image_url = first.get("url") if isinstance(first, dict) else None
        if not image_url:
            raise ProviderError("FAL image entry is missing a 'url'.")

        try:
            image_response = await client.get(image_url)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Failed to download FAL image: {exc}"
            ) from exc
        if not image_response.is_success:
            raise ProviderError(
                f"Failed to download FAL image ({image_response.status_code})."
            )

        content_type = (
            first.get("content_type")
            or image_response.headers.get("content-type")
            or "image/png"
        )
        return ProviderResult(
            image_bytes=image_response.content,
            content_type=content_type,
            actual_cost_usd=None,
            provider_metadata={
                "endpoint": endpoint,
                "model_id": model_id,
                "size": size,
            },
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
