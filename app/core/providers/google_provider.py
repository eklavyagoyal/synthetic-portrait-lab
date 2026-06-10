"""Google Gemini image provider ("Nano Banana").

Calls the Gemini ``generateContent`` REST endpoint and returns decoded image
bytes. The generated image is returned inline (base64) in the response under
``candidates[0].content.parts[].inlineData.data``; we find the first such part
and decode it.

Confirmed against the official Gemini API image-generation docs (June 2026):

* Endpoint: ``POST .../v1beta/models/{model}:generateContent``.
* Auth: the ``x-goog-api-key`` HTTP header (the key is never placed in the URL).
* Request: a ``contents`` array carrying the text prompt, plus
  ``generationConfig.responseModalities == ["IMAGE"]`` and a 1:1 square via
  ``generationConfig.imageConfig.aspectRatio``.
* Response: image bytes live in ``inlineData.data`` (base64) with the MIME type
  in ``inlineData.mimeType``. (The wire format may use snake_case
  ``inline_data``/``mime_type`` interchangeably, so we accept both.)

Model id mapping (registry id -> real Gemini model name):
``nano-banana`` -> ``gemini-2.5-flash-image`` and
``nano-banana-pro`` -> ``gemini-3-pro-image-preview``.

Gemini does not report a per-image USD cost on this endpoint, so
``actual_cost_usd`` is always left as ``None``.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import httpx

from ..models import ProviderResult
from .base import ImageProvider, ProviderAuthError, ProviderError

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MAX_BODY_CHARS = 500  # truncate error bodies so we never log huge payloads

# Registry model ids -> real Gemini model names.
_MODEL_MAP = {
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-pro": "gemini-3-pro-image-preview",
}

# We always request square portraits.
_ASPECT_RATIO = "1:1"


class GoogleProvider(ImageProvider):
    """Generate images via the Google Gemini ``generateContent`` API."""

    provider_name = "google"
    requires_api_key = True
    api_endpoint = "v1beta/models/{model}:generateContent"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
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

    @staticmethod
    def _resolve_model(model_id: str) -> str:
        """Map a registry model id to the real Gemini model name (verbatim fallback)."""
        return _MODEL_MAP.get(model_id, model_id)

    async def generate(
        self,
        *,
        prompt: str,
        size: str,
        model_id: str,
        seed: Optional[int] = None,
        quality: Optional[str] = None,
    ) -> ProviderResult:
        """Generate one image. ``size`` is honoured only as a 1:1 aspect ratio."""
        self.ensure_ready()
        client = self._get_client()

        gemini_model = self._resolve_model(model_id)

        generation_config: dict[str, Any] = {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": _ASPECT_RATIO},
        }
        # The Gemini API accepts ``seed`` in generationConfig; include it only when
        # supplied so we never send a meaningless field for the default case.
        if seed is not None:
            generation_config["seed"] = seed

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        # Key travels in the header only — never in the URL, body, or any log line.
        headers = {
            "x-goog-api-key": self.api_key or "",
            "Content-Type": "application/json",
        }

        try:
            response = await client.post(
                f"/models/{gemini_model}:generateContent",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Google Gemini request failed: {exc}") from exc

        self._raise_for_status(response)

        image_bytes, content_type = self._extract_image(response)
        return ProviderResult(
            image_bytes=image_bytes,
            content_type=content_type or "image/png",
            actual_cost_usd=None,  # Gemini does not report a per-image USD cost here.
            provider_metadata={
                "model_id": model_id,
                "gemini_model": gemini_model,
                "size": size,
                "aspect_ratio": _ASPECT_RATIO,
            },
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate non-2xx responses into the right provider exception."""
        if response.is_success:
            return

        status = response.status_code
        snippet = self._truncate(response.text)

        # Auth failures: HTTP 401/403, or a Google status of
        # PERMISSION_DENIED / UNAUTHENTICATED in the error body.
        if status in (401, 403) or self._is_auth_status(response):
            raise ProviderAuthError(
                f"Google Gemini rejected the API key (HTTP {status})."
            )

        raise ProviderError(f"Google Gemini returned HTTP {status}: {snippet}")

    @staticmethod
    def _is_auth_status(response: httpx.Response) -> bool:
        """Detect PERMISSION_DENIED / UNAUTHENTICATED in a Gemini error body."""
        try:
            error = response.json().get("error", {})
        except ValueError:
            return False
        if not isinstance(error, dict):
            return False
        status = error.get("status")
        return status in ("PERMISSION_DENIED", "UNAUTHENTICATED")

    def _extract_image(self, response: httpx.Response) -> tuple[bytes, Optional[str]]:
        """Return ``(image_bytes, mime_type)`` from the first inline-image part."""
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"Google Gemini returned a non-JSON response: "
                f"{self._truncate(response.text)}"
            ) from exc

        for candidate in body.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                # Accept both camelCase (REST response) and snake_case.
                inline = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline, dict):
                    continue
                data = inline.get("data")
                if not data:
                    continue
                mime_type = inline.get("mimeType") or inline.get("mime_type")
                try:
                    return base64.b64decode(data), mime_type
                except (ValueError, TypeError) as exc:
                    raise ProviderError(
                        f"Google Gemini returned invalid base64 image data: {exc}"
                    ) from exc

        raise ProviderError(
            "Google Gemini response contained no inline image data: "
            f"{self._truncate(response.text)}"
        )

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
