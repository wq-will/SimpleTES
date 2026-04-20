import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletes.engine.checkpoint import CheckpointManager
from simpletes.config import EngineConfig
from simpletes.node import Node, NodeDatabase, Status
from simpletes.policies import PendingFinalize, create_selector


REFLECTION_POLICIES = [
    "balance",
    "puct",
    "rpucg",
    "llm_puct",
    "llm_rpucg",
    "llm_elite",
]


def _build_policy(name: str):
    return create_selector(
        name,
        num_chains=1,
        max_generations=4,
        k=1,
        c=1.0,
        gamma=0.8,
        reflection_mode=True,
        llm_policy_model="reflection-model",
        llm_policy_api_base="https://reflection.example/v1",
        llm_policy_api_key="secret-key",
        llm_policy_pool_size=2,
    )


@pytest.mark.parametrize("policy_name", REFLECTION_POLICIES)
def test_create_selector_propagates_reflection_config(policy_name):
    policy = _build_policy(policy_name)

    assert policy.reflection_mode is True
    assert policy.llm_policy_model == "reflection-model"
    assert policy.llm_policy_api_base == "https://reflection.example/v1"

    state = policy.state_dict()
    assert state["reflection_mode"] is True
    assert state["llm_policy_model"] == "reflection-model"
    assert state["llm_policy_api_base"] == "https://reflection.example/v1"
    assert "llm_policy_api_key" not in state


@pytest.mark.parametrize("policy_name", REFLECTION_POLICIES)
def test_finalize_batch_sets_reflection_for_batch_best(policy_name, monkeypatch):
    policy = _build_policy(policy_name)
    db = NodeDatabase()

    root = Node(
        id="root",
        code="def solve():\n    return 0\n",
        metrics={"combined_score": 0.0},
        score=0.0,
        status=Status.DONE,
    )
    child = Node(
        id="child",
        code="def solve():\n    return 1\n",
        parent_ids=["root"],
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )
    db.add(root)
    db.add(child)

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return SimpleNamespace(
            content="Approach: try a better heuristic\nInsight: keep the stronger path"
        )

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    policy.register_batch(gen_id=7, chain_idx=0, inspiration_ids=["root"], k=1)
    pending = PendingFinalize(
        gen_id=7,
        chain_idx=0,
        children=[("child", None)],
        inspirations=["root"],
    )

    completion = asyncio.run(policy.finalize_batch(pending, db))

    assert completion.best_node_id == "child"
    assert child.reflection == "Approach: try a better heuristic\nInsight: keep the stronger path"
    assert "child" in policy.chains[0]


def test_reflect_on_winner_accepts_raw_string_llm_response(monkeypatch):
    policy = _build_policy("balance")
    node = Node(
        id="child",
        code="def solve():\n    return 1\n",
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return "Approach: direct string\nInsight: still parsed"

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    reflection = asyncio.run(policy.reflect_on_winner(node))

    assert reflection == "Approach: direct string\nInsight: still parsed"


def test_reflect_on_winner_returns_empty_for_missing_content(monkeypatch):
    policy = _build_policy("balance")
    node = Node(
        id="child",
        code="def solve():\n    return 1\n",
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return SimpleNamespace(content=None)

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    assert asyncio.run(policy.reflect_on_winner(node)) == ""


def test_checkpoint_config_serializes_effective_llm_policy_values(tmp_path):
    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
        model="generator-model",
        api_base="https://generator.example/v1",
        llm_backend="vllm_token_forcing",
        context_window=32768,
        reasoning_budget=26000,
        response_budget=6768,
        reflection_mode=True,
    )
    manager = CheckpointManager(config, "instance-id", str(tmp_path))

    serialized = manager._config_to_dict()

    assert serialized["reflection_mode"] is True
    assert serialized["llm_policy_model"] == "generator-model"
    assert serialized["llm_policy_api_base"] == "https://generator.example/v1"
    assert serialized["llm_backend"] == "vllm_token_forcing"
    assert serialized["context_window"] == 32768
    assert serialized["reasoning_budget"] == 26000
    assert serialized["response_budget"] == 6768
