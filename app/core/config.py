"""Loads `.env`, typed settings, the bucket config and the model registry.

Secrets are held as :class:`pydantic.SecretStr` so they never appear in logs,
reprs or metadata. The only place a key is unwrapped is when a provider client
is constructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .buckets import BucketConfig
from .pricing import PricingService

_DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


class Settings(BaseSettings):
    """Runtime settings sourced from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Provider credentials (raw env var names, e.g. OPENAI_API_KEY)
    openai_api_key: Optional[SecretStr] = None
    fal_key: Optional[SecretStr] = None
    replicate_api_token: Optional[SecretStr] = None
    # Google Gemini (Nano Banana). Accept either GEMINI_API_KEY or GOOGLE_API_KEY.
    gemini_api_key: Optional[SecretStr] = Field(
        None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    openai_base_url: str = "https://api.openai.com/v1"

    # App-level defaults (PORTRAIT_-prefixed env vars)
    default_provider: str = Field("mock", validation_alias="PORTRAIT_DEFAULT_PROVIDER")
    default_model: str = Field("mock-image", validation_alias="PORTRAIT_DEFAULT_MODEL")
    output_base_dir: str = Field("./outputs", validation_alias="PORTRAIT_OUTPUT_BASE_DIR")
    allow_custom_buckets: bool = Field(
        False, validation_alias="PORTRAIT_ALLOW_CUSTOM_BUCKETS"
    )
    model_registry_path: Optional[str] = Field(
        None, validation_alias="PORTRAIT_MODEL_REGISTRY_PATH"
    )

    # -- credential access (the only place secrets are unwrapped) --------- #
    _KEY_FIELDS = {
        "openai": "openai_api_key",
        "fal": "fal_key",
        "replicate": "replicate_api_token",
        "google": "gemini_api_key",
    }

    def api_key_for(self, provider: str) -> Optional[str]:
        field = self._KEY_FIELDS.get(provider)
        if not field:
            return None
        secret: Optional[SecretStr] = getattr(self, field, None)
        return secret.get_secret_value() if secret else None

    def has_key_for(self, provider: str) -> bool:
        return bool(self.api_key_for(provider)) or provider == "mock"


def load_model_registry(path: str | Path | None = None) -> dict:
    """Load the model/pricing registry from ``path`` or the bundled default."""
    target = Path(path).expanduser() if path else _DEFAULT_REGISTRY_PATH
    if not target.exists():
        raise FileNotFoundError(f"Model registry not found: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if "providers" not in data:
        raise ValueError(f"Model registry {target} has no 'providers' key.")
    return data


class AppConfig:
    """Aggregates everything a front-end needs to build and run a request."""

    def __init__(self, settings: Settings, buckets: BucketConfig, registry: dict):
        self.settings = settings
        self.buckets = buckets
        self.registry = registry
        self.pricing = PricingService(registry)

    @classmethod
    def load(cls, *, env_file: str | None = None) -> "AppConfig":
        # Make sure variables from .env are present in the environment first.
        load_dotenv(env_file or ".env", override=False)
        settings = Settings()  # type: ignore[call-arg]
        buckets = BucketConfig(allow_custom=settings.allow_custom_buckets)
        registry = load_model_registry(settings.model_registry_path)
        return cls(settings=settings, buckets=buckets, registry=registry)

    def credentials_for(self, provider: str) -> dict:
        """Constructor kwargs for a provider client (key + base url)."""
        kwargs: dict = {"api_key": self.settings.api_key_for(provider)}
        if provider == "openai":
            kwargs["base_url"] = self.settings.openai_base_url
        return kwargs
