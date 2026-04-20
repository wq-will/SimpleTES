"""LLM backends for SimpleTES.

Two concrete clients share one async interface (``generate`` /
``generate_batch`` / ``close``) so they are interchangeable from the
generator's point of view:

- ``LLMClient`` — LiteLLM-based, any of LiteLLM's 100+ providers, goes
  through a process pool for true parallelism.
- ``VLLMTokenForcingClient`` — direct httpx to a vLLM server with
  two-phase GPT-OSS Harmony token forcing (reasoning budget control).

Use ``create_llm_client(config)`` to pick the backend based on
``EngineConfig.llm_backend``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from simpletes.llm.litellm_client import LLMClient
from simpletes.llm.types import LLMCallError, LLMResult
from simpletes.llm.vllm_forcing import VLLMTokenForcingClient

if TYPE_CHECKING:
    from simpletes.config import EngineConfig


@runtime_checkable
class LLMBackend(Protocol):
    """Structural type of an LLM backend used by the generator."""

    async def generate(
        self, prompt: str, instance_id: str = "", track_io: bool = False,
    ) -> LLMResult: ...

    async def generate_batch(
        self, prompt: str, n: int, instance_id: str = "", track_io: bool = False,
    ) -> list[LLMResult]: ...

    def close(self) -> None: ...


def create_llm_client(config: EngineConfig) -> LLMBackend:
    """Pick and instantiate the LLM backend declared by ``config.llm_backend``."""
    if config.llm_backend == "vllm_token_forcing":
        return VLLMTokenForcingClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_base=config.api_base,
            api_key=config.api_key,
            timeout=config.timeout,
            reasoning_effort=config.reasoning_effort,
            tokenizer_path=config.tokenizer_path,
            context_window=config.context_window,
            reasoning_budget=config.reasoning_budget,
            response_budget=config.response_budget,
            pool_size=config.gen_concurrency,
        )
    return LLMClient(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_total_tokens=config.max_total_tokens,
        api_key=config.api_key,
        api_base=config.api_base,
        timeout=config.timeout,
        retry=config.retry,
        pool_size=config.gen_concurrency,
        reasoning_effort=config.reasoning_effort,
    )


__all__ = [
    "LLMBackend",
    "LLMCallError",
    "LLMClient",
    "LLMResult",
    "VLLMTokenForcingClient",
    "create_llm_client",
]
