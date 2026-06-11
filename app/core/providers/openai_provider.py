"""OpenAI Images provider.

Calls the OpenAI Images "generations" endpoint and returns decoded PNG bytes.
Targets ``gpt-image-*`` (which returns base64 image data by default) but also
handles the legacy/URL response shape.

Cost note: gpt-image is billed per *token* (image tokens scale with size and
quality), and this endpoint returns a ``usage`` token object but **no USD
amount**. So ``actual_cost_usd`` is always ``None`` — the run's cost stays an
estimate — and we surface the returned ``usage`` for the audit trail rather
than discarding it.
"""

from __future__ import annotations

import base64
from typing import Optional

import httpx

from ..models import ProviderResult
from .base import ImageProvider, ProviderAuthError, ProviderError

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_BODY_CHARS = 500  # truncate error bodies so we never log huge payloads


class OpenAIProvider(ImageProvider):
    """Generate images via the OpenAI Images API (``gpt-image-1``)."""

    provider_name = "openai"
    requires_api_key = True
    api_endpoint = "v1/images/generations"

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
            base_url=base_url or _DEFAULT_BASE_URL,
            timeout=timeout,
            **kwargs,
        )
        # Created lazily on first use so construction/import never touches the network.
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout,
            )
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
        """Generate one image. ``seed`` is ignored (gpt-image does not support it).

        ``quality`` (low/medium/high/auto) is forwarded to the API and is the
        primary driver of token cost. The endpoint returns a ``usage`` token
        object but no USD amount, so ``actual_cost_usd`` stays ``None``.
        """
        self.ensure_ready()
        client = self._get_client()
        payload = {"model": model_id, "prompt": prompt, "size": size, "n": 1}
        if quality:
            payload["quality"] = quality
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = await client.post(
                "/images/generations", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"OpenAI rejected the API key (HTTP {response.status_code})."
            )
        if not response.is_success:
            raise ProviderError(
                f"OpenAI returned HTTP {response.status_code}: "
                f"{self._truncate(response.text)}"
            )

        body = self._parse_body(response)
        image_bytes = await self._image_bytes_from_body(body, response)
        usage = body.get("usage")
        return ProviderResult(
            image_bytes=image_bytes,
            content_type="image/png",
            # Token-billed endpoint: usage is reported, a USD cost is not.
            actual_cost_usd=None,
            usage=usage if isinstance(usage, dict) else None,
            provider_metadata={"model_id": model_id, "size": size, "quality": quality},
        )

    @staticmethod
    def _parse_body(response: httpx.Response) -> dict:
        """Parse the JSON body, raising a clear ProviderError on a bad shape."""
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"OpenAI returned a non-JSON response: {OpenAIProvider._truncate(response.text)}"
            ) from exc
        if not isinstance(body, dict):
            raise ProviderError(
                f"OpenAI returned an unexpected response shape: "
                f"{OpenAIProvider._truncate(response.text)}"
            )
        return body

    async def _image_bytes_from_body(self, body: dict, response: httpx.Response) -> bytes:
        """Pull PNG bytes out of a parsed OpenAI images response (b64_json or url)."""
        try:
            data = body["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"OpenAI returned an unexpected response shape: "
                f"{self._truncate(response.text)}"
            ) from exc

        b64 = data.get("b64_json")
        if b64:
            try:
                return base64.b64decode(b64)
            except (ValueError, TypeError) as exc:
                raise ProviderError(
                    f"OpenAI returned invalid base64 image data: {exc}"
                ) from exc

        url = data.get("url")
        if url:
            return await self._download(url)

        raise ProviderError(
            "OpenAI response contained neither 'b64_json' nor 'url' image data."
        )

    async def _download(self, url: str) -> bytes:
        """Download image bytes from a returned URL."""
        try:
            resp = await self._get_client().get(url)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Failed to download OpenAI image: {exc}") from exc
        if not resp.is_success:
            raise ProviderError(
                f"Failed to download OpenAI image (HTTP {resp.status_code})."
            )
        return resp.content

    @staticmethod
    def _truncate(text: str) -> str:
        text = text.strip()
        if len(text) > _MAX_BODY_CHARS:
            return text[:_MAX_BODY_CHARS] + "..."
        return text

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
