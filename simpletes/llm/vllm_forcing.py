"""Two-phase token-forcing client for vLLM-served GPT-OSS / Harmony models.

When reasoning effort is high, the model may consume the entire token budget
on its analysis channel and never emit the final channel. We drive
vLLM's raw /v1/completions endpoint in two phases:

  1. Phase 1: let the model reason freely up to a budget.
  2. Phase 2: if Phase 1 didn't produce a final channel, force-inject
     Harmony transition tokens and continue generation within a
     reserved response budget.

Uses ``httpx.Client`` wrapped with ``asyncio.to_thread`` for the async
interface. Intentionally bypasses LiteLLM because LiteLLM does not
expose token-level prompt injection.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from simpletes.evaluator import rich_print
from simpletes.llm.types import LLMResult


class VLLMTokenForcingClient:
    """Raw vLLM completions client with GPT-OSS Harmony token forcing."""

    # Harmony token-forcing constants
    PHASE2_PREFILL = (
        "\n\n... okay, I am out of thinking tokens. "
        "I need to write the code in my final message now."
    )
    FINAL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
    END_TOKEN = "<|end|>"
    RETURN_TOKEN = "<|return|>"
    CONTEXT_BUFFER = 64

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str = "medium",
        tokenizer_path: str | None = None,
        context_window: int = 8192,
        reasoning_budget: int | None = None,
        response_budget: int = 3000,
        pool_size: int = 384,
        **kwargs,
    ):
        self.model = self._normalize_model_name(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        normalized_api_base = (api_base or "http://127.0.0.1:18000/v1").rstrip("/")
        if not normalized_api_base.endswith("/v1"):
            normalized_api_base = f"{normalized_api_base}/v1"
        self.api_base = normalized_api_base
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout or 3000.0
        self.reasoning_effort = reasoning_effort
        self.context_window = context_window
        self.reasoning_budget = reasoning_budget
        self.response_budget = response_budget

        # Lazy-load httpx and tokenizer
        import httpx as _httpx
        from transformers import AutoTokenizer as _AutoTok

        self._client = _httpx.Client(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
            limits=_httpx.Limits(
                max_connections=pool_size * 2,
                max_keepalive_connections=0,
            ),
        )
        tokenizer_id = tokenizer_path or model
        self._tokenizer = _AutoTok.from_pretrained(tokenizer_id)
        self._executor = ThreadPoolExecutor(max_workers=pool_size)
        rich_print(
            f"[dim]VLLMTokenForcingClient initialized: model={model}, "
            f"context_window={context_window}[/dim]"
        )

    # ---- low-level helpers ----

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        expanded_model = os.path.expanduser(model)
        if model.startswith(("/", "./", "../", "~")) or os.path.exists(expanded_model):
            return expanded_model
        # Strip LiteLLM provider prefix for names like "openai/gpt-oss-20b".
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
        )

    def _completion(
        self,
        *,
        prompt: str | None = None,
        prompt_token_ids: list[int] | None = None,
        max_tokens: int = 1,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stop": [self.RETURN_TOKEN] if stop is None else stop,
            "return_token_ids": True,
        }
        if prompt_token_ids is not None:
            body["prompt"] = prompt_token_ids
        elif prompt is not None:
            body["prompt"] = prompt
        else:
            raise ValueError("Either prompt or prompt_token_ids must be provided")

        resp = self._client.post(f"{self.api_base}/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        token_ids = choice.get("token_ids") or []
        text = choice.get("text", "")
        if token_ids:
            # vLLM strips special markers from choice["text"], but phase
            # handling depends on raw markers being preserved.
            text = self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "token_ids": token_ids,
        }

    def _force_tokens(self, forced_text: str) -> dict[str, Any]:
        forced_token_ids = self._encode(forced_text)
        forced_decoded = self._tokenizer.decode(
            forced_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return {"text": forced_decoded, "token_ids": forced_token_ids}

    @staticmethod
    def _contains_final_channel(text: str) -> bool:
        return "<|channel|>final<|message|>" in text

    @staticmethod
    def _extract_final_text(raw_text: str) -> str | None:
        marker = "<|channel|>final<|message|>"
        if marker not in raw_text:
            return None
        tail = raw_text.split(marker, 1)[1]
        tail = tail.split("<|return|>", 1)[0]
        tail = tail.split("<|end|>", 1)[0]
        return tail.strip()

    # ---- core two-phase logic ----

    def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the two-phase completion. Returns dict with raw_text and final_text."""
        prompt = self._render_prompt(messages)
        prompt_ids = self._encode(prompt)
        prompt_len = len(prompt_ids)

        phase1_total = self.reasoning_budget
        if phase1_total is None:
            phase1_total = self.context_window - self.response_budget

        phase1_max = min(
            phase1_total - prompt_len,
            self.context_window - prompt_len - self.response_budget - self.CONTEXT_BUFFER,
        )
        if phase1_max <= 0:
            raise ValueError(
                f"Prompt too long. prompt_len={prompt_len}, "
                f"phase1_total={phase1_total}, response_budget={self.response_budget}"
            )

        phase1 = self._completion(prompt=prompt, max_tokens=phase1_max)
        phase1_text = phase1["text"]
        phase1_token_ids = phase1["token_ids"]

        # If phase 1 already reached the final channel naturally, we're done.
        # If the final channel started but got truncated, continue generating.
        if self._contains_final_channel(phase1_text):
            if phase1["finish_reason"] != "length":
                raw_text = phase1_text
                return {
                    "raw_text": raw_text,
                    "final_text": self._extract_final_text(raw_text),
                    "usage": phase1["usage"],
                    "error_reason": None,
                }
            remaining_continue = (
                self.context_window
                - prompt_len
                - len(phase1_token_ids)
                - self.CONTEXT_BUFFER
            )
            continue_max = min(self.response_budget, max(0, remaining_continue))
            if continue_max <= 0:
                rich_print("[yellow]⚠ Final channel truncated but no budget to continue[/yellow]")
                return {
                    "raw_text": phase1_text,
                    "final_text": self._extract_final_text(phase1_text),
                    "usage": phase1["usage"],
                    "error_reason": "final_channel_truncated",
                }
            continue_ids = prompt_ids + phase1_token_ids
            cont = self._completion(prompt_token_ids=continue_ids, max_tokens=continue_max)
            raw_text = phase1_text + cont["text"]
            final_text = self._extract_final_text(raw_text)
            if not final_text:
                rich_print("[yellow]⚠ Continuation completed without a parsable final response[/yellow]")
                return {
                    "raw_text": raw_text,
                    "final_text": final_text,
                    "usage": phase1["usage"],
                    "error_reason": "final_channel_truncated",
                }
            return {
                "raw_text": raw_text,
                "final_text": final_text,
                "usage": phase1["usage"],
                "error_reason": None,
            }

        # Phase 1 ended without a usable final answer. Force-inject transition
        # tokens and continue in a "final channel" Phase 2.
        forced_text = self.PHASE2_PREFILL
        if not phase1_text.endswith(self.END_TOKEN):
            forced_text += self.END_TOKEN
        forced_text += self.FINAL_MARKER

        forced = self._force_tokens(forced_text)
        forced_token_ids = forced["token_ids"]

        remaining_after_force = (
            self.context_window
            - prompt_len
            - len(phase1_token_ids)
            - len(forced_token_ids)
            - self.CONTEXT_BUFFER
        )
        raw_text = phase1_text
        if remaining_after_force >= 0:
            raw_text += forced["text"]

        if remaining_after_force < 0:
            rich_print("[yellow]⚠ Phase2 skipped: no room left for forced final marker[/yellow]")
            return {
                "raw_text": raw_text,
                "final_text": None,
                "usage": phase1["usage"],
                "error_reason": "phase2_budget_exhausted",
            }

        phase2_prompt_ids = prompt_ids + phase1_token_ids + forced_token_ids
        phase2_max = min(self.response_budget, max(0, remaining_after_force))
        if phase2_max <= 0:
            rich_print("[yellow]⚠ Phase2 skipped: no output budget remained after forcing[/yellow]")
            return {
                "raw_text": raw_text,
                "final_text": None,
                "usage": phase1["usage"],
                "error_reason": "phase2_budget_exhausted",
            }

        phase2 = self._completion(prompt_token_ids=phase2_prompt_ids, max_tokens=phase2_max)
        raw_text += phase2["text"]
        final_text = self._extract_final_text(raw_text)
        if not final_text:
            rich_print("[yellow]⚠ Phase2 completed without a parsable final response[/yellow]")
            return {
                "raw_text": raw_text,
                "final_text": final_text,
                "usage": phase1["usage"],
                "error_reason": "phase2_missing_final",
            }

        return {
            "raw_text": raw_text,
            "final_text": final_text,
            "usage": phase1["usage"],
            "error_reason": None,
        }

    # ---- SimpleTES client interface ----

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def generate(
        self, prompt: str, instance_id: str = "", track_io: bool = False
    ) -> LLMResult:
        messages = [{"role": "user", "content": prompt}]
        result = await self._run_in_executor(self._complete, messages)

        error_reason = result.get("error_reason")
        text = "" if error_reason else (result["final_text"] or result["raw_text"])
        return LLMResult(
            text=text,
            prompt=prompt if track_io else None,
            raw_output=result["raw_text"] if track_io else None,
            token_usage=result["usage"] if track_io else None,
            error_reason=error_reason,
        )

    async def generate_batch(
        self, prompt: str, n: int, instance_id: str = "", track_io: bool = False
    ) -> list[LLMResult]:
        if n <= 1:
            return [await self.generate(prompt, instance_id, track_io)]
        results = await asyncio.gather(
            *(self.generate(prompt, instance_id, track_io) for _ in range(n))
        )
        return list(results)

    def close(self) -> None:
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()
