"""
Base agent class that all AP Invoice agents inherit from.

Provides:
  - Structured logging
  - LLM call delegation
  - Common error handling interface
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from services.llm_service import LLMResponse, LLMService


class BaseAgent(ABC):
    """Abstract base for all AP workflow agents."""

    #: Subclasses set this to identify the agent in logs and audit records
    agent_name: str = "base_agent"

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service
        self.logger = structlog.get_logger(self.__class__.__name__)

    def call_llm(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> LLMResponse:
        """Delegate to LLMService with agent name attached for audit logging."""
        return self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=self.agent_name,
            **kwargs,
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task. Subclasses implement this."""
        ...
