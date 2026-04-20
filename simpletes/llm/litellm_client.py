"""LiteLLM-based client with a process pool for true parallelism.

Workers return ``("ok", payload)`` or ``("error", info)`` tuples so that
unpicklable provider exceptions (which embed httpx.Response etc.) never
cross the pickle boundary as raised exceptions.
"""
from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from litellm import completion, get_supported_openai_params, token_counter

from simpletes.evaluator import rich_print
from simpletes.llm.types import LLMCallError, LLMResult


# Global process pool (initialized lazily)
_process_pool: ProcessPoolExecutor | None = None
_pool_size: int = 0


def _is_pool_broken(pool: ProcessPoolExecutor) -> bool:
    return bool(getattr(pool, "_broken", False))


def _is_pool_shutdown(pool: ProcessPoolExecutor) -> bool:
    return bool(getattr(pool, "_shutdown_thread", False))


def _init_process_pool(size: int, *, force: bool = False) -> ProcessPoolExecutor:
    """Initialize, resize, or recreate the global process pool."""
    global _process_pool, _pool_size
    need_new = (
        _process_pool is None
        or _pool_size != size
        or force
        or _is_pool_broken(_process_pool)
        or _is_pool_shutdown(_process_pool)
    )
    if need_new:
        if _process_pool is not None:
            _process_pool.shutdown(wait=False)
        _pool_size = size
        _process_pool = ProcessPoolExecutor(
            max_workers=size,
            mp_context=mp.get_context("spawn"),
        )
    return _process_pool


def _cleanup_pool() -> None:
    global _process_pool, _pool_size
    if _process_pool is not None:
        _process_pool.shutdown(wait=False)
        _process_pool = None
        _pool_size = 0


atexit.register(_cleanup_pool)


# ============================================================================
# Response parsing (runs in parent OR worker)
# ============================================================================

def _extract_token_usage(resp: Any) -> dict[str, int] | None:
    """Extract token usage dict from a litellm response, or None if unavailable."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
    result = {k: int(v) for k in keys if (v := getattr(usage, k, None)) is not None}
    return result or None


def _build_raw_output(msg: Any, text: str) -> str:
    """Build raw output including optional reasoning/thinking blocks."""
    raw_parts = []
    if msg is not None:
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            raw_parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
        thinking = getattr(msg, "thinking", None)
        if thinking:
            raw_parts.append(f"<thinking>\n{thinking}\n</thinking>")
    raw_parts.append(text)
    return "\n".join(raw_parts)


def _worker_error_info(exc: BaseException) -> dict[str, str]:
    """Stringify an exception into a picklable dict for parent-process consumption."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


# ============================================================================
# Worker functions (run in child processes)
# ============================================================================

def _worker_generate(
    model: str,
    messages: list,
    call_kwargs: dict[str, Any],
    track_io: bool,
) -> tuple[str, Any]:
    """Run a single completion inside a worker process.

    Returns ``("ok", (text, raw_output, token_usage))`` on success;
    ``("error", {type, message, traceback})`` on failure.
    """
    try:
        resp = completion(model=model, messages=messages, **call_kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ("ok", ("", "", None))
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        text = content or ""
        raw_output = _build_raw_output(msg, text)
        token_usage = _extract_token_usage(resp) if track_io else None
        return ("ok", (text, raw_output, token_usage))
    except Exception as exc:
        return ("error", _worker_error_info(exc))


def _worker_generate_batch(
    model: str,
    messages: list,
    call_kwargs: dict[str, Any],
    track_io: bool,
    n: int,
) -> tuple[str, Any]:
    """Run a batch completion (n choices) inside a worker process."""
    try:
        resp = completion(model=model, messages=messages, n=n, **call_kwargs)
    except Exception as exc:
        return ("error", _worker_error_info(exc))

    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ("ok", [("", "", None)])

    shared_token_usage = _extract_token_usage(resp) if track_io else None
    results: list[tuple[str, str, dict[str, int] | None]] = []
    for choice in choices:
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        text = content or ""
        results.append((text, _build_raw_output(msg, text), shared_token_usage))
    return ("ok", results)


# ============================================================================
# LLMClient
# ============================================================================

class LLMClient:
    """LLM client using LiteLLM SDK primitives, run through a process pool."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        retry: int = 0,
        drop_params: bool = True,
        pool_size: int = 1,
        max_total_tokens: int | None = None,
        reasoning_effort: str = "medium",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_total_tokens = max_total_tokens
        self.reasoning_effort = reasoning_effort

        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

        # LiteLLM uses `num_retries` for SDK retry behavior.
        self.num_retries = max(int(retry or 0), 0)

        # If True, LiteLLM will drop unsupported OpenAI params instead of raising.
        self.drop_params = drop_params

        # Process pool settings
        self.pool_size = pool_size
        self._pool: ProcessPoolExecutor | None = None
        if self.pool_size > 0:
            self._pool = _init_process_pool(self.pool_size)

        # Determine token parameter name
        self._token_param_name: str = "max_tokens"
        try:
            supported = set(get_supported_openai_params(model=self.model) or [])
            if "max_tokens" in supported:
                self._token_param_name = "max_tokens"
            elif "max_completion_tokens" in supported:
                self._token_param_name = "max_completion_tokens"
        except Exception as e:
            rich_print(f"[dim]Could not resolve supported params for model={self.model}: {e}[/dim]")

    # ==================== Call kwargs ====================

    def _common_call_kwargs(self) -> dict[str, Any]:
        """Shared kwargs for LiteLLM calls; None values are dropped."""
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            self._token_param_name: self.max_tokens,
            "timeout": self.timeout,
            "num_retries": self.num_retries,
            "drop_params": self.drop_params,
        }
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key

        # gpt-oss routes reasoning_effort through LiteLLM's allowed_openai_params
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["allowed_openai_params"] = ["reasoning_effort"]

        return {k: v for k, v in kwargs.items() if v is not None}

    # ==================== Pool management ====================

    def _recreate_pool(self) -> None:
        rich_print("[yellow]Recreating broken process pool...[/yellow]")
        self._pool = _init_process_pool(self.pool_size, force=True)

    def _sync_pool(self) -> ProcessPoolExecutor:
        self._pool = _init_process_pool(self.pool_size)
        return self._pool

    @staticmethod
    def _is_shutdown_runtime_error(exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        return "cannot schedule new futures after shutdown" in str(exc).lower()

    async def _run_in_pool(self, fn, *args):
        """Run a worker with robust pool-recreation on BrokenProcessPool."""
        if self.pool_size <= 0:
            raise RuntimeError("Process pool not initialized. Ensure pool_size > 0.")

        loop = asyncio.get_running_loop()
        pool = self._sync_pool()
        try:
            return await loop.run_in_executor(pool, fn, *args)
        except (BrokenProcessPool, RuntimeError) as e:
            if isinstance(e, BrokenProcessPool) or self._is_shutdown_runtime_error(e):
                self._recreate_pool()
                pool = self._sync_pool()
                return await loop.run_in_executor(pool, fn, *args)
            raise

    async def _cap_tokens_for_prompt(self, call_kwargs: dict[str, Any], messages: list) -> None:
        """Apply max_total_tokens capping to call_kwargs in place."""
        if self.max_total_tokens is None:
            return
        try:
            prompt_tokens = await asyncio.to_thread(token_counter, model=self.model, messages=messages)
            capped = min(self.max_tokens, max(1, self.max_total_tokens - prompt_tokens))
            call_kwargs[self._token_param_name] = capped
        except (ValueError, TypeError, KeyError):
            pass  # fall back to uncapped

    def _unwrap_worker_result(self, result: tuple[str, Any]) -> Any:
        """Convert worker ('ok'|'error', payload) tuple into payload or raise."""
        status, payload = result
        if status == "ok":
            return payload
        raise LLMCallError(
            model=self.model,
            api_base=self.api_base,
            error_type=payload.get("type", "UnknownError"),
            message=payload.get("message", ""),
            traceback_str=payload.get("traceback", ""),
        )

    # ==================== Public interface ====================

    def preflight(self, *, timeout: float = 10.0) -> None:
        """Validate credentials/endpoint with a 1-token ping in the main process."""
        call_kwargs = self._common_call_kwargs()
        call_kwargs["timeout"] = timeout
        call_kwargs["num_retries"] = 0
        call_kwargs[self._token_param_name] = 1
        messages = [{"role": "user", "content": "ping"}]
        try:
            completion(model=self.model, messages=messages, **call_kwargs)
        except Exception as exc:
            raise LLMCallError(
                model=self.model,
                api_base=self.api_base,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_str=traceback.format_exc(),
            ) from exc

    async def generate(
        self, prompt: str, instance_id: str = "", track_io: bool = False
    ) -> LLMResult:
        """Generate a single completion.

        Raises ``LLMCallError`` on backend failure with model/api_base context.
        """
        messages = [{"role": "user", "content": prompt}]
        call_kwargs = self._common_call_kwargs()
        await self._cap_tokens_for_prompt(call_kwargs, messages)

        raw = await self._run_in_pool(
            _worker_generate, self.model, messages, call_kwargs, track_io,
        )
        text, raw_output, token_usage = self._unwrap_worker_result(raw)
        return LLMResult(
            text=text,
            prompt=prompt if track_io else None,
            raw_output=raw_output if track_io else None,
            token_usage=token_usage,
        )

    async def generate_batch(
        self, prompt: str, n: int, instance_id: str = "", track_io: bool = False
    ) -> list[LLMResult]:
        """Generate n completions. Falls back to n sequential calls if the
        provider doesn't support batch n."""
        if n <= 1:
            return [await self.generate(prompt, instance_id, track_io)]

        messages = [{"role": "user", "content": prompt}]
        call_kwargs = self._common_call_kwargs()
        await self._cap_tokens_for_prompt(call_kwargs, messages)

        try:
            raw = await self._run_in_pool(
                _worker_generate_batch, self.model, messages, call_kwargs, track_io, n,
            )
            batch_results = self._unwrap_worker_result(raw)
            return [
                LLMResult(
                    text=text,
                    prompt=prompt if track_io else None,
                    raw_output=raw_output if track_io else None,
                    token_usage=token_usage,
                )
                for text, raw_output, token_usage in batch_results
            ]
        except Exception:
            # Provider doesn't support n>1; fall back to sequential calls.
            results = await asyncio.gather(
                *(self.generate(prompt, instance_id, track_io) for _ in range(n))
            )
            return list(results)

    def close(self) -> None:
        """No-op for interface compatibility."""
        return None
