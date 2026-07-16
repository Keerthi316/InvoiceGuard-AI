"""
Application configuration loaded from environment variables / .env file.

Uses pydantic-settings so every setting is type-checked at startup.
Import `settings` (singleton) from this module everywhere in the codebase.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings derived from environment variables.

    Sensitive keys (API keys) are read from .env and never logged.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM Provider — OpenRouter (OpenAI-compatible endpoint)
    # ------------------------------------------------------------------
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider identifier (openrouter)",
    )

    # OpenRouter credentials — loaded from .env, never hardcoded
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )

    # Model is read from the MODEL env var; fall back to a sensible default
    model: str = Field(
        default="google/gemini-2.5-flash",
        description="Model name as recognised by OpenRouter (e.g. google/gemini-2.5-flash)",
    )

    # Alias kept so the rest of the codebase can still reference default_model
    @property
    def default_model(self) -> str:  # type: ignore[override]
        return self.model

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Project root directory",
    )
    log_dir: Path = Field(default=Path("logs"))
    audit_dir: Path = Field(default=Path("audit"))
    audit_file: str = Field(default="audit_log.jsonl")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ------------------------------------------------------------------
    # Business Rules / Thresholds
    # ------------------------------------------------------------------
    price_variance_threshold: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Max % price variance before exception",
    )
    quantity_variance_threshold: float = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description="Max % quantity variance before exception",
    )
    approval_threshold: float = Field(
        default=1000.0,
        ge=0.0,
        description=(
            "Global invoice amount above which approval is required. "
            "Used as a fallback when the PO does not specify approval_required_above "
            "and when no PO is uploaded. Set via APPROVAL_THRESHOLD env var."
        ),
    )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    tesseract_cmd: str = Field(
        default="",
        description="Full path to tesseract executable (blank = not available)",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_dir", "audit_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: object) -> Path:
        p = Path(str(v))
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Helpers — interface unchanged so the rest of the codebase is unaffected
    # ------------------------------------------------------------------
    def get_api_key(self) -> str:
        """Return the OpenRouter API key."""
        return self.openrouter_api_key

    def has_api_key(self) -> bool:
        """True if OPENROUTER_API_KEY is set and non-empty."""
        return bool(self.openrouter_api_key)

    @property
    def audit_log_path(self) -> Path:
        return self.audit_dir / self.audit_file

    @property
    def available_models(self) -> dict[str, list[str]]:
        """
        Models offered through OpenRouter that this app exposes in the UI.

        The list is informational — any valid OpenRouter model slug works.
        The default is driven by the MODEL env var; the first entry here is
        shown as the UI fallback when MODEL is not set.
        """
        return {
            "openrouter": [
                "google/gemini-2.5-flash",
                "google/gemini-2.5-pro",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-haiku",
                "meta-llama/llama-3.1-8b-instruct",
                "mistralai/mistral-7b-instruct",
            ],
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached so the .env file is read only once per process.
    Call `get_settings.cache_clear()` in tests to reload.
    """
    return Settings()


# Convenience singleton — import this in all modules
settings: Settings = get_settings()
