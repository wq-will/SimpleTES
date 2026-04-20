import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletes.generator import GenerationTask, Generator
from simpletes.llm import LLMResult, VLLMTokenForcingClient
from simpletes.node import EvolveBlockContext
import simpletes.llm.vllm_forcing as vllm_mod


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase1: dict,
    phase2: dict | None = None,
    prompt_ids: list[int] | None = None,
    forced_token_ids: list[int] | None = None,
    reasoning_budget: int = 10,
    response_budget: int = 5,
    context_window: int = 24,
) -> tuple[VLLMTokenForcingClient, list[dict], str]:
    async def _run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(vllm_mod, "rich_print", lambda *args, **kwargs: None)

    client = object.__new__(VLLMTokenForcingClient)
    client._run_in_executor = _run_inline
    client.model = "test-model"
    client.temperature = 0.0
    client.max_tokens = 32
    client.reasoning_effort = "medium"
    client.context_window = context_window
    client.reasoning_budget = reasoning_budget
    client.response_budget = response_budget
    client._phase2_count = 0
    client.CONTEXT_BUFFER = 0

    prompt_ids = list(prompt_ids or [11, 12])
    forced_token_ids = list(forced_token_ids or [21, 22, 23])
    forced_text = client.PHASE2_PREFILL + client.END_TOKEN + client.FINAL_MARKER
    completion_calls: list[dict] = []

    def _render_prompt(messages):
        assert messages == [{"role": "user", "content": "prompt"}]
        return "PROMPT"

    def _encode(text):
        assert text == "PROMPT"
        return list(prompt_ids)

    def _completion(*, prompt=None, prompt_token_ids=None, max_tokens=1, stop=None):
        del max_tokens, stop
        completion_calls.append({
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
        })
        if prompt is not None:
            return phase1
        assert phase2 is not None
        return phase2

    def _force_tokens(text):
        assert text == forced_text
        return {"text": text, "token_ids": list(forced_token_ids)}

    client._render_prompt = _render_prompt
    client._encode = _encode
    client._completion = _completion
    client._force_tokens = _force_tokens
    return client, completion_calls, forced_text


def test_vllm_token_forcing_preserves_local_model_paths():
    assert (
        VLLMTokenForcingClient._normalize_model_name("/local/models/my-model")
        == "/local/models/my-model"
    )
    assert VLLMTokenForcingClient._normalize_model_name("openai/gpt-4o") == "gpt-4o"


def test_vllm_completion_decodes_raw_special_tokens_and_uses_relative_path():
    raw_text = (
        "<|channel|>analysis<|message|>reasoning<|end|>"
        "<|start|>assistant<|channel|>final<|message|>done<|return|>"
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "text": "analysisreasoningassistantfinaldone",
                    "finish_reason": "stop",
                    "token_ids": [1, 2, 3],
                }],
                "usage": {"completion_tokens": 3},
            }

    class FakeHttpClient:
        def __init__(self):
            self.calls = []

        def post(self, path, json):
            self.calls.append((path, json))
            return FakeResponse()

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            assert token_ids == [1, 2, 3]
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return raw_text

    client = object.__new__(VLLMTokenForcingClient)
    client.model = "test-model"
    client.temperature = 0.0
    client.api_base = "http://unit.test/v1"
    client._client = FakeHttpClient()
    client._tokenizer = FakeTokenizer()

    result = client._completion(prompt="PROMPT", max_tokens=7)

    assert client._client.calls == [(
        "http://unit.test/v1/completions",
        {
            "model": "test-model",
            "max_tokens": 7,
            "temperature": 0.0,
            "stop": [client.RETURN_TOKEN],
            "return_token_ids": True,
            "prompt": "PROMPT",
        },
    )]
    assert result["text"] == raw_text
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"completion_tokens": 3}
    assert result["token_ids"] == [1, 2, 3]


def test_vllm_token_forcing_uses_phase1_final_without_phase2(monkeypatch):
    phase1 = {
        "text": "<|channel|>final<|message|>```python\nprint('ok')\n```<|return|>",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 4},
        "token_ids": [1, 2, 3, 4],
    }
    client, completion_calls, _ = _build_client(monkeypatch, phase1=phase1)

    result = client._complete([{"role": "user", "content": "prompt"}])

    assert result["final_text"] == "```python\nprint('ok')\n```"
    assert result["error_reason"] is None
    assert len(completion_calls) == 1


def test_vllm_token_forcing_uses_phase2_after_phase1_stop_without_final(monkeypatch):
    final_text = "```python\nprint('ok')\n```"
    phase1 = {
        "text": "still reasoning",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 4},
        "token_ids": [1, 2, 3, 4],
    }
    phase2 = {
        "text": f"{final_text}<|return|>",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 5},
        "token_ids": [5, 6, 7, 8, 9],
    }
    client, completion_calls, forced_text = _build_client(
        monkeypatch,
        phase1=phase1,
        phase2=phase2,
    )

    result = asyncio.run(client.generate("prompt", track_io=True))

    assert result.text == final_text
    assert result.error_reason is None
    assert len(completion_calls) == 2
    assert completion_calls[1]["prompt_token_ids"] == [11, 12, 1, 2, 3, 4, 21, 22, 23]
    assert result.raw_output == f"still reasoning{forced_text}{phase2['text']}"


def test_vllm_token_forcing_continues_truncated_final_channel(monkeypatch):
    phase1 = {
        "text": "<|channel|>final<|message|>```python\nprint('ok')",
        "finish_reason": "length",
        "usage": {"completion_tokens": 4},
        "token_ids": [1, 2, 3, 4],
    }
    phase2 = {
        "text": "\n```<|return|>",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 2},
        "token_ids": [5, 6],
    }
    client, completion_calls, _ = _build_client(
        monkeypatch,
        phase1=phase1,
        phase2=phase2,
        context_window=24,
        response_budget=8,
    )

    result = client._complete([{"role": "user", "content": "prompt"}])

    assert result["final_text"] == "```python\nprint('ok')\n```"
    assert result["error_reason"] is None
    assert len(completion_calls) == 2
    assert completion_calls[1]["prompt_token_ids"] == [11, 12, 1, 2, 3, 4]


def test_vllm_token_forcing_reports_budget_exhaustion(monkeypatch):
    phase1 = {
        "text": "still reasoning",
        "finish_reason": "length",
        "usage": {"completion_tokens": 8},
        "token_ids": list(range(8)),
    }
    client, _, _ = _build_client(
        monkeypatch,
        phase1=phase1,
        reasoning_budget=10,
        response_budget=2,
        context_window=12,
    )

    result = asyncio.run(client.generate("prompt", track_io=True))

    assert result.text == ""
    assert result.error_reason == "phase2_budget_exhausted"
    assert result.raw_output == "still reasoning"


def test_vllm_token_forcing_reports_missing_final_after_phase2(monkeypatch):
    phase1 = {
        "text": "still reasoning",
        "finish_reason": "length",
        "usage": {"completion_tokens": 4},
        "token_ids": [1, 2, 3, 4],
    }
    phase2 = {
        "text": "<|return|>",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 1},
        "token_ids": [9],
    }
    client, _, forced_text = _build_client(monkeypatch, phase1=phase1, phase2=phase2)

    result = asyncio.run(client.generate("prompt", track_io=True))

    assert result.text == ""
    assert result.error_reason == "phase2_missing_final"
    assert result.raw_output == f"still reasoning{forced_text}<|return|>"


def test_vllm_token_forcing_generate_returns_final_text(monkeypatch):
    final_text = "```python\n# EVOLVE-BLOCK-START\nprint('ok')\n# EVOLVE-BLOCK-END\n```"
    phase1 = {
        "text": "still reasoning",
        "finish_reason": "length",
        "usage": {"completion_tokens": 4},
        "token_ids": [1, 2, 3, 4],
    }
    phase2 = {
        "text": f"{final_text}<|return|>",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 6},
        "token_ids": [5, 6, 7, 8, 9, 10],
    }
    client, _, forced_text = _build_client(monkeypatch, phase1=phase1, phase2=phase2)

    result = asyncio.run(client.generate("prompt", track_io=True))

    assert result.text == final_text
    assert result.error_reason is None
    assert result.raw_output == f"still reasoning{forced_text}{phase2['text']}"


def test_generator_prefers_llm_error_reason(monkeypatch):
    class FakeLLM:
        async def generate_batch(self, prompt, n, instance_id="", track_io=False):
            del prompt, n, instance_id, track_io
            return [
                LLMResult(
                    text="",
                    raw_output="debug raw output",
                    error_reason="phase2_budget_exhausted",
                )
            ]

    generator = object.__new__(Generator)
    generator._llm = FakeLLM()
    generator._evolve_context = EvolveBlockContext.from_program(
        "def solve():\n"
        "    # EVOLVE-BLOCK-START\n"
        "    pass\n"
        "    # EVOLVE-BLOCK-END\n"
    )

    task = GenerationTask(
        prompt="prompt",
        inspiration_ids=[],
        k=1,
        chain_idx=0,
        gen_id=1,
    )

    results = asyncio.run(generator.generate(task, instance_id="test", track_io=False))

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].reason == "phase2_budget_exhausted"
    assert results[0].llm_output == "debug raw output"
