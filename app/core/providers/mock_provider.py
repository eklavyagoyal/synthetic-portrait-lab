"""Free, offline provider used by tests and for dry-runs.

Produces a *real*, valid PNG (so the gallery/preview and image-saving paths are
exercised end-to-end) without any network call. Output is deterministic for a
given (prompt, seed): the same inputs yield the same image bytes.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Optional

from ..models import ProviderResult
from .base import ImageProvider

# A minimal valid 1x1 PNG, used only if Pillow is somehow unavailable.
_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return max(1, min(2048, int(w))), max(1, min(2048, int(h)))
    except Exception:
        return 512, 512


def _color_from(seed_material: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    # muted, portrait-ish tones (avoid pure/black-white extremes)
    return (60 + digest[0] % 150, 60 + digest[1] % 150, 60 + digest[2] % 150)


class MockProvider(ImageProvider):
    provider_name = "mock"
    requires_api_key = False
    api_endpoint = "mock://local (offline)"

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
        width, height = _parse_size(size)
        material = f"{prompt}|{seed}|{model_id}|{size}"
        image_bytes = self._render(width, height, material)
        return ProviderResult(
            image_bytes=image_bytes,
            content_type="image/png",
            actual_cost_usd=0.0,
            provider_metadata={"mock": True, "model_id": model_id, "size": size},
        )

    @staticmethod
    def _render(width: int, height: int, material: str) -> bytes:
        try:
            from PIL import Image, ImageDraw  # type: ignore

            base = _color_from(material)
            img = Image.new("RGB", (width, height), base)
            draw = ImageDraw.Draw(img)
            # faux centered "head" so the placeholder reads as a portrait
            cx, cy = width // 2, int(height * 0.45)
            r = int(min(width, height) * 0.28)
            head = tuple(min(255, c + 45) for c in base)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=head)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return _FALLBACK_PNG
