"""Shared data types for LLM backends."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    """Result from LLM generation with optional tracking data."""
    text: str
    prompt: str | None = None
    raw_output: str | None = None
    token_usage: dict[str, int] | None = None  # prompt_tokens / completion_tokens / total_tokens
    error_reason: str | None = None


class LLMCallError(Exception):
    """Raised when an LLM call fails; preserves context across process boundaries.

    LiteLLM exceptions often embed unpicklable objects (httpx.Response etc.),
    so they can't cross the ProcessPoolExecutor boundary. Workers catch those
    exceptions, stringify them into a dict, and the parent reraises as
    LLMCallError with the original info plus model/api_base context.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None,
        error_type: str,
        message: str,
        traceback_str: str = "",
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.error_type = error_type
        self.message = message
        self.traceback_str = traceback_str
        super().__init__(f"{error_type}: {message}")
