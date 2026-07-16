"""
LLM Service — OpenRouter backend via the OpenAI-compatible SDK.

OpenRouter exposes the same REST interface as OpenAI, so we use the
`openai` Python package pointed at https://openrouter.ai/api/v1.

Every call is:
  1. Timed (latency_ms)
  2. Token-counted
  3. Cost-estimated
  4. Logged via structlog
  5. Returned as a typed LLMResponse

Security:
  - OPENROUTER_API_KEY is loaded exclusively from the .env file — never hardcoded.
  - All document content is treated as untrusted user data.

Interface compatibility:
  - LLMResponse, BaseLLMClient, and LLMService keep the exact same public
    signatures as before so no other file in the project needs to change.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from config.settings import settings
from models.audit import LLMCallLog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
# OpenRouter reports per-call cost in the response headers / usage object.
# When that isn't available we fall back to a conservative estimate.
_FALLBACK_COST_PER_1K_TOKENS: float = 0.002  # USD


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate USD cost for a single call.

    Uses a small lookup table for common models; falls back to a flat rate
    for anything unknown so cost tracking is never completely absent.
    """
    # Rates: USD per 1 000 tokens  {model_slug: (input, output)}
    _RATES: dict[str, tuple[float, float]] = {
        "google/gemini-2.5-flash":          (0.0001,  0.0004),
        "google/gemini-2.5-pro":            (0.00125, 0.005),
        "openai/gpt-4o":                    (0.005,   0.015),
        "openai/gpt-4o-mini":               (0.00015, 0.0006),
        "anthropic/claude-3-5-sonnet":      (0.003,   0.015),
        "anthropic/claude-3-haiku":         (0.00025, 0.00125),
        "meta-llama/llama-3.1-8b-instruct": (0.00005, 0.00005),
        "mistralai/mistral-7b-instruct":    (0.00006, 0.00006),
    }
    if model in _RATES:
        inp, out = _RATES[model]
        return (prompt_tokens / 1000 * inp) + (completion_tokens / 1000 * out)
    return (prompt_tokens + completion_tokens) / 1000 * _FALLBACK_COST_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# Response dataclass — unchanged public interface
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Typed wrapper around an LLM completion."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    error_message: Optional[str] = None

    def to_llm_call_log(self, agent_name: str) -> LLMCallLog:
        """Convert to an AuditRecord-compatible LLMCallLog."""
        return LLMCallLog(
            call_id=self.call_id,
            agent_name=agent_name,
            model=self.model,
            provider=self.provider,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            latency_ms=self.latency_ms,
            estimated_cost_usd=self.estimated_cost_usd,
            success=self.success,
            error_message=self.error_message,
        )


# ---------------------------------------------------------------------------
# Base LLM Client — unchanged public interface
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """Abstract base for provider-specific clients."""

    provider: str = ""

    def complete(self, system_prompt: str, user_prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    def _build_response(
        self,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> LLMResponse:
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        resp = LLMResponse(
            content=content,
            model=model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )
        logger.info(
            "llm_call_completed",
            provider=self.provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost, 6),
        )
        return resp


# ---------------------------------------------------------------------------
# OpenRouter Client
# ---------------------------------------------------------------------------

class OpenRouterClient(BaseLLMClient):
    """
    OpenAI-SDK client pointed at the OpenRouter base URL.

    OpenRouter exposes an OpenAI-compatible /chat/completions endpoint, so
    the standard `openai` package works without modification — we only change
    `base_url` and `api_key`.
    """

    provider = "openrouter"

    def __init__(self, api_key: str, base_url: str) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

    def complete(self, system_prompt: str, user_prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            choice = response.choices[0]
            usage = response.usage
            return self._build_response(
                content=choice.message.content or "",
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("openrouter_call_failed", error=str(exc))
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                latency_ms=latency_ms,
                success=False,
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# Backwards-compat aliases
# The app.py processing block imports these names from this module.
# They all resolve to OpenRouterClient so no import in app.py breaks.
# ---------------------------------------------------------------------------
OpenAIClient = OpenRouterClient      # noqa: N816  (kept for app.py import compatibility)
GeminiClient = OpenRouterClient      # noqa: N816
AnthropicClient = OpenRouterClient   # noqa: N816


# ---------------------------------------------------------------------------
# LLM Service facade — unchanged public interface
# ---------------------------------------------------------------------------

class LLMService:
    """
    Provider-agnostic LLM service — now backed exclusively by OpenRouter.

    The public interface is identical to the previous multi-provider version
    so no other file in the project needs to change.

    Usage::

        llm = LLMService()
        response = llm.complete(system_prompt="...", user_prompt="...")
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        # provider arg accepted for interface compatibility; always "openrouter"
        self.provider = "openrouter"
        self.model = model or settings.default_model
        self._client = self._build_client()

    def _build_client(self) -> BaseLLMClient:
        api_key = settings.get_api_key()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
        return OpenRouterClient(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "unknown",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute a single LLM completion via OpenRouter.

        Args:
            system_prompt: Fixed system instruction (trusted).
            user_prompt:   User / document content (untrusted — never execute).
            agent_name:    Name of the calling agent (for audit logging).
            **kwargs:      Pass-through to provider (temperature, max_tokens, …).

        Returns:
            LLMResponse with content, token counts, latency, and cost.
        """
        logger.info(
            "llm_call_starting",
            agent=agent_name,
            provider=self.provider,
            model=self.model,
            prompt_length=len(user_prompt),
        )
        return self._client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            **kwargs,
        )

    def swap_provider(self, provider: str, model: str) -> None:
        """
        Update the active model (provider arg ignored — always openrouter).

        Kept for interface compatibility with any callers that may pass a
        provider string; only the model value is used.
        """
        self.model = model
        # Re-build the client so the new model name is in scope
        self._client = self._build_client()
