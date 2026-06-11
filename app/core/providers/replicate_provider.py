"""Replicate image provider.

Drives Replicate's official-model predictions API. A ``model_id`` is the
``owner/name`` slug (e.g. ``black-forest-labs/flux-schnell``). We create a
prediction synchronously via the ``Prefer: wait`` header and, should the request
return before the model finishes, fall back to bounded polling of the
prediction's ``urls.get`` endpoint.

Credentials come from ``REPLICATE_API_TOKEN`` (passed in as ``api_key``). The
token is only ever placed in the Authorization header — never logged, persisted
or echoed in error messages.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from ..models import ProviderResult
from .base import ImageProvider, ProviderAuthError, ProviderError

_API_BASE = "https://api.replicate.com/v1"

# Synchronous-mode budget (the server holds the connection up to ~60s).
_SYNC_WAIT_SECONDS = 60

# Bounded polling fallback for the non-blocking ("starting"/"processing") case.
_POLL_INTERVAL_SECONDS = 2.0
_MAX_POLLS = 90  # ~3 minutes worst case before giving up

_TERMINAL_STATES = {"succeeded", "failed", "canceled"}
_PENDING_STATES = {"starting", "processing"}

# Keep error bodies short so we never dump large payloads (and never headers).
_MAX_ERR_BODY = 300


class ReplicateProvider(ImageProvider):
    """Generate one image per call via Replicate's predictions API."""

    provider_name = "replicate"
    requires_api_key = True
    api_endpoint = "v1/predictions"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the shared async HTTP client."""
        if self._client is None:
            base = (self.base_url or _API_BASE).rstrip("/")
            self._client = httpx.AsyncClient(
                base_url=base,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
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

        owner, name = self._split_model_id(model_id)
        client = self._get_client()

        prediction = await self._create_prediction(
            client, owner, name, prompt=prompt, seed=seed
        )
        prediction = await self._await_terminal(client, prediction)

        status = prediction.get("status")
        if status != "succeeded":
            detail = prediction.get("error") or f"prediction ended with status {status!r}"
            raise ProviderError(f"Replicate prediction failed: {detail}")

        image_url = self._first_output_url(prediction.get("output"))
        if not image_url:
            raise ProviderError("Replicate prediction succeeded but returned no image URL.")

        image_bytes, content_type = await self._download(client, image_url)
        return ProviderResult(
            image_bytes=image_bytes,
            content_type=content_type,
            actual_cost_usd=None,  # Replicate bills by predict-time, not reported here
            provider_metadata={"model_id": model_id, "prediction_id": prediction.get("id")},
        )

    # ------------------------------------------------------------------ #
    # API calls
    # ------------------------------------------------------------------ #
    async def _create_prediction(
        self,
        client: httpx.AsyncClient,
        owner: str,
        name: str,
        *,
        prompt: str,
        seed: Optional[int],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"input": {"prompt": prompt}}
        if seed is not None:
            body["input"]["seed"] = seed

        try:
            resp = await client.post(
                f"/models/{owner}/{name}/predictions",
                json=body,
                headers={"Prefer": f"wait={_SYNC_WAIT_SECONDS}"},
            )
        except httpx.HTTPError as exc:  # network / timeout / DNS
            raise ProviderError(f"Replicate request failed: {exc!s}") from exc

        self._raise_for_status(resp)
        return self._parse_json(resp)

    async def _await_terminal(
        self, client: httpx.AsyncClient, prediction: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the prediction once it reaches a terminal state, polling if needed."""
        status = prediction.get("status")
        if status in _TERMINAL_STATES:
            return prediction

        get_url = (prediction.get("urls") or {}).get("get")
        if not get_url:
            raise ProviderError(
                f"Replicate prediction is {status!r} but provided no poll URL."
            )

        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            try:
                resp = await client.get(get_url)
            except httpx.HTTPError as exc:
                raise ProviderError(f"Replicate polling failed: {exc!s}") from exc
            self._raise_for_status(resp)
            prediction = self._parse_json(resp)
            status = prediction.get("status")
            # Check immediately after each fetch so the final poll's result is honored.
            if status in _TERMINAL_STATES:
                return prediction

        raise ProviderError(
            f"Replicate prediction did not finish within the poll budget (last status {status!r})."
        )

    async def _download(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[bytes, str]:
        try:
            # Output files live on a different host; don't send the auth header.
            resp = await client.get(url, headers={"Authorization": ""})
        except httpx.HTTPError as exc:
            raise ProviderError(f"Replicate image download failed: {exc!s}") from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"Replicate image download returned HTTP {resp.status_code}."
            )
        content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip()
        return resp.content, content_type or "image/png"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_model_id(model_id: str) -> tuple[str, str]:
        parts = model_id.strip().split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ProviderError(
                f"Replicate model_id must be 'owner/name', got {model_id!r}."
            )
        return parts[0], parts[1]

    @staticmethod
    def _first_output_url(output: Any) -> Optional[str]:
        """Extract the first image URL from Replicate's flexible ``output`` shape."""
        if isinstance(output, str):
            return output or None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str) and item:
                    return item
        if isinstance(output, dict):
            # Some models nest the URL(s) under a key.
            for value in output.values():
                url = ReplicateProvider._first_output_url(value)
                if url:
                    return url
        return None

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code in (401, 403):
            raise ProviderAuthError(
                f"Replicate rejected the API token (HTTP {resp.status_code})."
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Replicate API error (HTTP {resp.status_code}): {self._short_body(resp)}"
            )

    @staticmethod
    def _parse_json(resp: httpx.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError("Replicate returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise ProviderError("Replicate returned an unexpected response shape.")
        return data

    @staticmethod
    def _short_body(resp: httpx.Response) -> str:
        text = (resp.text or "").strip().replace("\n", " ")
        return text[:_MAX_ERR_BODY] if text else "(empty body)"
