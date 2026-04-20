"""
Base class for inspiration selection policies.

IMPORTANT: Policy methods are designed to be called from async contexts.
The internal lock is a threading.Lock for fast synchronization.
All critical sections should be kept minimal to avoid blocking the event loop.

"""
from __future__ import annotations

import bisect
import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

if TYPE_CHECKING:
    from simpletes.node import Node, NodeDatabase, NodeDatabaseSnapshot

from simpletes.node import score_key
from simpletes.templates import REFLECTION_PROMPT_TEMPLATE
from simpletes.utils.text import extract_approach_insight, metrics_to_text, summarize_error


_MAX_REFLECTION_OUTPUT_CHARS = 10000


# Policy registry for automatic registration
SELECTOR_REGISTRY: dict[str, type[Selector]] = {}
_FAILURE_PATTERN_ERROR_MAX_CHARS = 240


@dataclass(frozen=True)
class BatchCompletion:
    """Policy notification emitted when a generation batch is fully processed."""

    gen_id: int
    best_node_id: str | None


@dataclass
class PendingFinalize:
    """Batch complete, awaiting finalize (reflection + policy hooks + commit).

    Returned by on_child_done when the last child of a batch completes.
    Engine should call policy.finalize_batch() to complete the batch.
    """

    gen_id: int
    chain_idx: int
    children: list[tuple[str, str | None]]  # (node_id, error_msg)
    inspirations: list[str]


def register_selector(name: str):
    """Decorator to register an inspiration policy.
    
    Usage:
        @register_selector("my_policy")
        class MyPolicy(Selector):
            ...
    """
    def decorator(cls: type[Selector]) -> type[Selector]:
        SELECTOR_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


class Selector:
    """Base class for inspiration selection policies.

    Thread-safety:
        Non-reentrant threading.Lock; keep critical sections short. Use the
        `_unlocked` helpers instead of nesting acquisitions.
    """

    name: str = "base"

    def get_info(self) -> dict[str, Any]:
        """Return policy-specific configuration for display.

        Override in subclasses to provide policy-specific parameters.
        Returns dict of {param_name: value} for display in initialization panel.
        """
        return {}

    def get_policy_context(self, chain_idx: int, db: NodeDatabase | NodeDatabaseSnapshot) -> str:
        """Return policy-specific context to include in generation prompt.

        Override in subclasses to provide additional context for the LLM.
        For example, llm_elite returns an overview of the elite pool.

        Args:
            chain_idx: The chain index for which context is requested.
            db: The node database for looking up nodes.

        Returns:
            A string to include in the prompt, or "" if no additional context.
        """
        return ""

    def __init__(self):
        """Initialize policy with non-reentrant lock for fast concurrent access."""
        self._lock = threading.Lock()

    def select(
        self,
        db: NodeDatabase | NodeDatabaseSnapshot,
        n: int,
        stats: dict[str, int] | None,
        eval_concurrency: int,
        backpressure_multiplier: float = 1.0,
    ) -> tuple[list[Node], dict[str, float], int]:
        """Implements ``SelectInspirations`` of Algorithm~\\ref{alg:async_method}.

        Returns ``(inspirations, failure_patterns, chain_idx)``. ``inspirations``
        is the set $S^{(c)}$ chosen by the policy; ``failure_patterns`` is the
        current failure histogram $F^{(c)}$ (mapped onto the prompt by the
        generator); ``chain_idx`` identifies the chain $c$ for this batch.
        Returning an empty inspirations list signals backpressure.
        """
        raise NotImplementedError

    def register_batch(
        self,
        gen_id: int,
        chain_idx: int,
        inspiration_ids: list[str],
        k: int,
    ) -> None:
        """Register a batch for tracking. Called after GenerationTask is created.

        Args:
            gen_id: Batch identifier (from GenerationTask)
            chain_idx: Chain index (from select())
            inspiration_ids: List of inspiration node IDs
            k: Number of candidates to generate
        """
        pass  # Default no-op for non-chain policies

    def on_child_done(self, child: Node, parents: Sequence[Node]) -> PendingFinalize | None:
        """Optional hook for credit assignment after a child evaluation completes.

        Returns PendingFinalize when batch is complete (for chain policies).
        """
        return None

    def on_generation_failed(self, gen_id: int) -> PendingFinalize | None:
        """Optional hook called when a generation fails.

        Args:
            gen_id: The batch identifier for the failed generation.

        Returns:
            PendingFinalize if this failure completed a batch.
        """
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        pass

    def reconcile_with_db(self, db: NodeDatabase | NodeDatabaseSnapshot) -> None:
        """Optional hook to reconcile internal state after DB load."""
        pass


def compute_chain_budgets(total: int, num_chains: int) -> dict[int, int]:
    """Compute per-chain budget allocation."""
    total = max(0, int(total))
    chains = max(1, int(num_chains))
    base = total // chains
    remainder = total % chains
    return {i: base + (1 if i < remainder else 0) for i in range(chains)}


class TrajectoryPolicyBase(Selector):
    """Shared base for chain-based inspiration policies."""

    _BACKPRESSURE_MULTIPLIER = 1000.0

    def get_info(self) -> dict[str, Any]:
        """Return chain-policy common parameters."""
        info = {
            "num_chains": self.num_chains,
            "k_candidates": self.k,
            "restart_every_n": self.restart_every_n,
        }
        if self.reflection_mode and self.llm_policy_model:
            info["llm_policy_model"] = self.llm_policy_model
        return info

    def __init__(
        self,
        num_chains: int = 4,
        max_generations: int = 100,
        k: int = 1,
        restart_every_n: int = 25,
        min_inspirations_cnt: int | None = None,
        max_inspirations_cnt: int | None = None,
        # Reflection & LLM policy config (unified)
        reflection_mode: bool = False,
        llm_policy_model: str | None = None,
        llm_policy_api_base: str | None = None,
        llm_policy_api_key: str | None = None,
        # Task instruction for context
        task_instruction: str | None = None,
        **kwargs,  # Accept and ignore unknown params from subclasses
    ):
        super().__init__()
        self.num_chains = max(1, int(num_chains))
        self.max_generations = max_generations
        self.k = max(1, int(k))
        # Chain-local restart period R from Alg.~\ref{alg:async_method}:
        # after every R commits to this chain's history, _reset_chain_locked
        # truncates the chain to its local best and clears DAG + visit counts.
        self.restart_every_n = int(restart_every_n)
        self.min_inspirations_cnt = min_inspirations_cnt
        self.max_inspirations_cnt = max_inspirations_cnt

        # LLM config for reflection and policy operations
        self.reflection_mode = reflection_mode
        self.llm_policy_model = llm_policy_model
        self.llm_policy_api_base = llm_policy_api_base
        self.llm_policy_api_key = llm_policy_api_key
        self.task_instruction = task_instruction

        self.chain_gen_budget = compute_chain_budgets(self.max_generations, self.num_chains)
        self.prompt_budget = {
            i: (budget + self.k - 1) // self.k if budget > 0 else 0
            for i, budget in self.chain_gen_budget.items()
        }
        self._validate_restart_every_n()

        # Chain state
        self.chains: dict[int, list[str]] = {i: [] for i in range(self.num_chains)}
        self.chain_prompt_count: dict[int, int] = {i: 0 for i in range(self.num_chains)}
        # Full per-chain history for analysis/plotting.
        self.chain_history: dict[int, list[str]] = {i: [] for i in range(self.num_chains)}

        # Sorted caches per chain: list of (neg_score, node_id) tuples, ascending by neg_score
        # (i.e. descending by score). Maintained incrementally with bisect.
        self._chain_sorted: dict[int, list[tuple[float, str]]] = {i: [] for i in range(self.num_chains)}

        # Ready chain tracking
        self._ready_chains: list[int] = [i for i, budget in self.prompt_budget.items() if budget > 0]

        # Batch state tracking (gen_id -> batch info)
        self._batches: dict[int, dict[str, Any]] = {}

        # Root node tracking
        self._root_id: str | None = None
        self._initialized: bool = False

        # Error tracking per chain
        self._chain_error_counts: dict[int, dict[str, int]] = {i: {} for i in range(self.num_chains)}
        self._chain_total_counts: dict[int, int] = {i: 1 for i in range(self.num_chains)}

        # Restart counter per chain (consumed by restart_every_n logic)
        self.nodes_since_restart: dict[int, int] = {i: 0 for i in range(self.num_chains)}

    def _validate_restart_every_n(self) -> None:
        if self.restart_every_n <= 0:
            raise ValueError("restart_every_n must be > 0")

    def _insert_sorted(self, chain_idx: int, node_id: str, score: float | None) -> None:
        if score is None:
            raise ValueError("chain insert requires finite score")
        entry = (-float(score), node_id)
        chain = self._chain_sorted[chain_idx]
        bisect.insort(chain, entry)

    def _on_chain_reset_locked(self, chain_idx: int, kept_node_id: str | None) -> None:
        """Subclass hook for clearing chain-local state on restart."""
        pass

    def _reset_chain_locked(self, chain_idx: int) -> str | None:
        """Chain-local restart from Alg.~\\ref{alg:async_method}: truncate chain
        to its best entry, clear its DAG + visit counts. Triggered every
        ``restart_every_n`` commits; see ``__init__`` for the period semantics."""
        kept_entry = self._chain_sorted.get(chain_idx, [])
        best_entry = kept_entry[0] if kept_entry else None
        kept_node_id = best_entry[1] if best_entry is not None else None

        if kept_node_id is None:
            self.chains[chain_idx] = []
            self._chain_sorted[chain_idx] = []
        else:
            self.chains[chain_idx] = [kept_node_id]
            self._chain_sorted[chain_idx] = [best_entry]

        self.nodes_since_restart[chain_idx] = 0
        self._chain_error_counts[chain_idx] = {}
        self._chain_total_counts[chain_idx] = 1
        self._on_chain_reset_locked(chain_idx, kept_node_id)
        return kept_node_id

    def _initialize_root(self, db: NodeDatabase | NodeDatabaseSnapshot) -> bool:
        if self._initialized:
            return True
        nodes = db.all_nodes_sorted()
        if not nodes:
            return False
        root = next((nd for nd in nodes if not nd.parent_ids), nodes[0])
        self._root_id = root.id
        for i in range(self.num_chains):
            self.chains[i].append(root.id)
            self.chain_history[i].append(root.id)
            self._insert_sorted(i, root.id, root.score)
        self._initialized = True
        return True

    def _get_chain_nodes(
        self,
        db: NodeDatabase | NodeDatabaseSnapshot,
        chain_idx: int,
    ) -> list[Node]:
        chain = self._chain_sorted[chain_idx]
        if not chain:
            # Build cache from chain list if missing.
            nodes = [db.nodes[nid] for nid in self.chains[chain_idx]]
            nodes.sort(key=score_key, reverse=True)
            self._chain_sorted[chain_idx] = [(-float(n.score), n.id) for n in nodes]
            return nodes
        return [db.nodes[nid] for _, nid in chain]

    def _sample_inspiration_count(self, requested_n: int, chain_len: int) -> int:
        if requested_n <= 0 or chain_len <= 0:
            return 0
        max_allowed = min(requested_n, chain_len)

        if self.min_inspirations_cnt is None or self.max_inspirations_cnt is None:
            return max_allowed

        low = int(self.min_inspirations_cnt)
        high = int(self.max_inspirations_cnt)
        if low <= 0 or high <= 0:
            return max_allowed
        if low > high:
            low, high = high, low

        sampled = random.randint(low, high)
        return min(sampled, max_allowed)

    def _get_top_failures(self, chain_idx: int, top_k: int = 10) -> dict[str, float]:
        """Get top failure patterns for a chain as {error_msg: ratio}."""
        total = self._chain_total_counts[chain_idx]
        error_counts = self._chain_error_counts[chain_idx]
        folded_counts: dict[str, int] = {}
        for raw_error, count in error_counts.items():
            normalized_error = summarize_error(str(raw_error), _FAILURE_PATTERN_ERROR_MAX_CHARS)
            if not normalized_error:
                continue
            folded_counts[normalized_error] = folded_counts.get(normalized_error, 0) + count
        sorted_errors = sorted(folded_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {error: count / total for error, count in sorted_errors}

    def _select_from_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
        n: int,
    ) -> list[Node]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Finalize batch: reflection + policy hooks + commit
    # ------------------------------------------------------------------

    def _find_best_node(
        self,
        db: NodeDatabase,
        children: list[tuple[str, str | None]],
    ) -> Node | None:
        """Find highest-scoring node from children list."""
        best_node, best_score = None, -float("inf")
        for node_id, _ in children:
            node = db.nodes.get(node_id)
            if node and node.score > best_score:
                best_score = node.score
                best_node = node
        return best_node

    async def _llm_generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        """Unified async LLM interface for all policy operations.

        Args:
            messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Optional max tokens limit

        Returns:
            ``choices[0].message`` from LiteLLM (use ``.content``, etc. in the caller).

        Raises:
            Exception: If LLM call fails (caller should handle)
        """
        if not self.llm_policy_model:
            raise ValueError("llm_policy_model not configured")

        from litellm import acompletion

        kwargs: dict[str, Any] = {
            "model": self.llm_policy_model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.llm_policy_api_key:
            kwargs["api_key"] = self.llm_policy_api_key
        if self.llm_policy_api_base:
            kwargs["api_base"] = self.llm_policy_api_base
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res = await acompletion(**kwargs)
        return res.choices[0].message

    async def reflect_on_winner(self, node: Node) -> str:
        """Reflection step of ``OnBatchComplete`` (Alg.~\\ref{alg:async_method}):
        produce a short Approach/Insight summary of the batch winner via the
        policy's auxiliary LLM. Returns the empty string when reflection is
        not configured or the call fails."""
        if not node.llm_input or not node.code or not isinstance(node.metrics, dict):
            return ""

        if not self.llm_policy_model:
            return ""

        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            llm_input=node.llm_input,
            code=node.code,
            metrics=metrics_to_text(node.metrics),
        )
        try:
            msg = await self._llm_generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            content = msg if isinstance(msg, str) else getattr(msg, "content", "")
            text = (content or "").strip()
        except Exception:
            return ""
        text = extract_approach_insight(text)
        if len(text) > _MAX_REFLECTION_OUTPUT_CHARS:
            text = text[:_MAX_REFLECTION_OUTPUT_CHARS].rstrip()
        return text

    async def _finalize_hook_unlocked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
    ) -> Any:
        """Policy-specific unlocked operations before commit.

        Override in subclass for async/LLM operations.
        Return value is passed to _finalize_hook_locked.
        """
        return None

    def _finalize_hook_locked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
        unlocked_result: Any,
    ) -> None:
        """Policy-specific locked operations after commit.

        Override in subclass to update policy auxiliary structures.
        Called after best_node is committed to chain.
        """
        pass

    async def finalize_batch(
        self,
        pending: PendingFinalize,
        db: NodeDatabase,
    ) -> BatchCompletion:
        """Implements ``OnBatchComplete`` of Alg.~\\ref{alg:async_method}:
        archive insert, edge add, memory update (reflection + failure histogram),
        best-of-K commit to $\\mathcal{H}_c$, visit-count bump, and optional
        chain-local restart. Engine calls this after ``on_child_done`` returns a
        ``PendingFinalize``.

        Args:
            pending: Batch info from on_child_done
            db: Node database for looking up nodes by ID

        Returns:
            BatchCompletion with best_node_id
        """
        best_node = None
        unlocked_result = None
        try:
            # --- Unlocked phase: can do expensive operations ---
            best_node = self._find_best_node(db, pending.children)

            # Reflection (using llm_policy config)
            if best_node and self.reflection_mode:
                best_node.reflection = await self.reflect_on_winner(best_node)

            # Policy-specific unlocked operations
            unlocked_result = await self._finalize_hook_unlocked(pending, best_node)
        except Exception:
            # Continue to locked phase for cleanup even if unlocked phase fails
            pass

        # --- Locked phase: commit and state updates (ALWAYS runs) ---
        with self._lock:
            return self._finalize_locked(pending, best_node, unlocked_result)

    def _finalize_locked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
        unlocked_result: Any,
    ) -> BatchCompletion:
        """Locked phase: stats, commit, hooks, scheduling.

        Must be called while holding self._lock.
        """
        chain_idx = pending.chain_idx

        # Stats
        if pending.children:
            self._chain_total_counts[chain_idx] += len(pending.children)

            # Track errors
            error_counts = self._chain_error_counts[chain_idx]
            for _, error_msg in pending.children:
                if error_msg:
                    error_counts[error_msg] = error_counts.get(error_msg, 0) + 1

        # Commit to chain
        if best_node and best_node.score > -float("inf"):
            self.chains[chain_idx].append(best_node.id)
            self.chain_history[chain_idx].append(best_node.id)
            self._insert_sorted(chain_idx, best_node.id, best_node.score)
            self.nodes_since_restart[chain_idx] += 1

        # Policy-specific locked operations (wrapped in try to ensure cleanup)
        try:
            self._finalize_hook_locked(pending, best_node, unlocked_result)
        except Exception:
            pass

        if best_node and best_node.score > -float("inf"):
            if self.nodes_since_restart[chain_idx] >= self.restart_every_n:
                self._reset_chain_locked(chain_idx)

        # Scheduling state (ALWAYS runs for cleanup)
        self.chain_prompt_count[chain_idx] += 1
        del self._batches[pending.gen_id]

        if self.chain_prompt_count[chain_idx] >= self.prompt_budget.get(chain_idx, 0):
            if chain_idx in self._ready_chains:
                self._ready_chains.remove(chain_idx)

        return BatchCompletion(
            gen_id=pending.gen_id,
            best_node_id=best_node.id if best_node else None,
        )

    def register_batch(
        self,
        gen_id: int,
        chain_idx: int,
        inspiration_ids: list[str],
        k: int,
    ) -> None:
        """Register a batch for tracking. Called after GenerationTask is created."""
        with self._lock:
            assert gen_id not in self._batches, f"gen_id {gen_id} already registered"
            assert 0 <= chain_idx < self.num_chains, f"invalid chain_idx {chain_idx}"
            self._batches[gen_id] = {
                "chain_idx": chain_idx,
                "inspirations": list(inspiration_ids),
                "submitted": k,
                "done": 0,
                "children": [],
            }

    def select(
        self,
        db: NodeDatabase | NodeDatabaseSnapshot,
        n: int,
        stats: dict[str, int] | None,
        eval_concurrency: int,
        backpressure_multiplier: float = 1.0,
    ) -> tuple[list[Node], dict[str, float], int]:
        with self._lock:
            if not self._ready_chains:
                return [], {}, 0

            chain_pending_count = {k: 0 for k in self._ready_chains}
            for batch in self._batches.values():
                if batch["chain_idx"] not in self._ready_chains:
                    continue
                chain_pending_count[batch["chain_idx"]] += 1

            if min(chain_pending_count.values()) > backpressure_multiplier:
                return [], {}, 0

            if not self._initialize_root(db):
                return [], {}, 0

            available = [k for k in self._ready_chains if chain_pending_count[k] <= backpressure_multiplier]
            if not available:
                return [], {}, 0
            chain_idx = available[0]
            self._ready_chains.remove(chain_idx)
            self._ready_chains.append(chain_idx)

            chain_nodes = self._get_chain_nodes(db, chain_idx)
            # Filter out -inf nodes (errors)
            chain_nodes = [nd for nd in chain_nodes if nd.score > -float("inf")]
            if not chain_nodes:
                return [], {}, 0

            n_select = self._sample_inspiration_count(n, len(chain_nodes))
            if n_select <= 0:
                return [], {}, 0

            selected = self._select_from_chain(chain_idx, chain_nodes, n_select)
            if not selected:
                return [], {}, 0

            # Get failure patterns for this chain
            failure_patterns = self._get_top_failures(chain_idx, top_k=10)

            # Return selected nodes and chain_idx; batch registration happens in engine
            return list(selected), failure_patterns, chain_idx

    def on_child_done(self, child: Node, parents: Sequence[Node]) -> PendingFinalize | None:
        """Track child completion. Returns PendingFinalize when batch is complete.

        Args:
            child: Completed child node (has score, metrics, etc.)
            parents: Parent nodes (inspirations)

        Returns:
            PendingFinalize if this was the last child and batch needs finalization.
            None if batch is still in progress.
        """
        gen_id = child.gen_id
        assert gen_id is not None, "on_child_done called with node missing gen_id"

        with self._lock:
            batch = self._batches.get(gen_id)
            if batch is None:
                return None

            error_msg = child.metrics.get("error", None) if child.metrics else None
            error_summary = (
                summarize_error(str(error_msg), _FAILURE_PATTERN_ERROR_MAX_CHARS) if error_msg else None
            )
            batch["children"].append((child.id, error_summary))
            batch["done"] += 1

            if batch["done"] >= batch["submitted"]:
                # Batch complete - return PendingFinalize for engine to call finalize_batch
                return PendingFinalize(
                    gen_id=gen_id,
                    chain_idx=batch["chain_idx"],
                    children=list(batch["children"]),
                    inspirations=list(batch["inspirations"]),
                )
            return None

    def on_generation_failed(self, gen_id: int) -> PendingFinalize | None:
        """Track generation failure. Returns PendingFinalize when batch is complete.

        Args:
            gen_id: Generation ID of the failed attempt

        Returns:
            PendingFinalize if this failure completed the batch.
            None if batch is still in progress.
        """
        with self._lock:
            batch = self._batches.get(gen_id)
            if batch is None:
                return None

            batch["done"] += 1
            if batch["done"] >= batch["submitted"]:
                return PendingFinalize(
                    gen_id=gen_id,
                    chain_idx=batch["chain_idx"],
                    children=list(batch["children"]),
                    inspirations=list(batch["inspirations"]),
                )
            return None

    def _state_dict_extra(self) -> dict[str, Any]:
        return {}

    def _load_state_extra(self, state: dict[str, Any]) -> None:
        pass

    def reconcile_with_db(self, db: NodeDatabase | NodeDatabaseSnapshot) -> None:
        with self._lock:
            valid_ids = set(db.nodes.keys())
            self.chains = {
                int(k): [nid for nid in v if nid in valid_ids]
                for k, v in self.chains.items()
            }
            self.chain_history = {
                int(k): [nid for nid in v if nid in valid_ids]
                for k, v in self.chain_history.items()
            }
            if self._root_id not in valid_ids:
                self._root_id = None

            # Rebuild sorted caches from chains.
            self._chain_sorted = {}
            for chain_idx, ids in self.chains.items():
                nodes = [db.nodes[nid] for nid in ids]
                nodes.sort(key=score_key, reverse=True)
                self._chain_sorted[chain_idx] = [(-float(n.score), n.id) for n in nodes]

            for i in range(self.num_chains):
                self.nodes_since_restart.setdefault(i, 0)

            # Note: _batches is cleared on resume (load_state_dict), so no normalization needed

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "num_chains": self.num_chains,
                "max_generations": self.max_generations,
                "k": self.k,
                "restart_every_n": self.restart_every_n,
                "min_inspirations_cnt": self.min_inspirations_cnt,
                "max_inspirations_cnt": self.max_inspirations_cnt,
                "reflection_mode": self.reflection_mode,
                "llm_policy_model": self.llm_policy_model,
                "llm_policy_api_base": self.llm_policy_api_base,
                "chain_gen_budget": {str(k): v for k, v in self.chain_gen_budget.items()},
                "prompt_budget": {str(k): v for k, v in self.prompt_budget.items()},
                "chains": {str(k): list(v) for k, v in self.chains.items()},
                "chain_history": {str(k): list(v) for k, v in self.chain_history.items()},
                "chain_prompt_count": {str(k): v for k, v in self.chain_prompt_count.items()},
                "nodes_since_restart": {str(k): v for k, v in self.nodes_since_restart.items()},
                "ready_chains": self._ready_chains,
                # Note: batches are cleared on resume anyway
                "batches": {str(k): dict(v) for k, v in self._batches.items()},
                "root_id": self._root_id,
                "initialized": self._initialized,
                "chain_sorted": {
                    str(k): list(v) for k, v in self._chain_sorted.items()
                },
                # Legacy keys kept for backward-compat reading (written but not read back)
                "chain_sorted_ids": {
                    str(k): [nid for _, nid in v] for k, v in self._chain_sorted.items()
                },
                "chain_sorted_neg_scores": {
                    str(k): [neg for neg, _ in v] for k, v in self._chain_sorted.items()
                },
                "chain_error_counts": {
                    str(k): dict(v) for k, v in self._chain_error_counts.items()
                },
                "chain_total_counts": {str(k): v for k, v in self._chain_total_counts.items()},
                **self._state_dict_extra(),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            if not state:
                return

            self.num_chains = int(state.get("num_chains", self.num_chains))
            self.max_generations = int(state.get("max_generations", self.max_generations))
            self.k = int(state.get("k", self.k))
            self.restart_every_n = int(state.get("restart_every_n", self.restart_every_n))
            self.min_inspirations_cnt = state.get("min_inspirations_cnt", self.min_inspirations_cnt)
            self.max_inspirations_cnt = state.get("max_inspirations_cnt", self.max_inspirations_cnt)

            self.chain_gen_budget = {
                int(k): int(v) for k, v in state.get("chain_gen_budget", {}).items()
            }
            if not self.chain_gen_budget:
                self.chain_gen_budget = compute_chain_budgets(self.max_generations, self.num_chains)

            self.prompt_budget = {
                int(k): int(v) for k, v in state.get("prompt_budget", {}).items()
            }
            if not self.prompt_budget:
                self.prompt_budget = {
                    i: (budget + self.k - 1) // self.k if budget > 0 else 0
                    for i, budget in self.chain_gen_budget.items()
                }

            self.chains = {int(k): list(v) for k, v in state.get("chains", {}).items()}
            if "chain_history" not in state:
                raise ValueError("policy state missing required field: chain_history")
            self.chain_history = {
                int(k): list(v) for k, v in state.get("chain_history", {}).items()
            }
            self.chain_prompt_count = {
                int(k): int(v) for k, v in state.get("chain_prompt_count", {}).items()
            }
            self.nodes_since_restart = {
                int(k): int(v) for k, v in state.get("nodes_since_restart", {}).items()
            }

            # Clear pending batches on resume - they can never complete
            # since the workers that would call on_child_done are gone
            self._batches = {}
            self._root_id = state.get("root_id")
            self._initialized = bool(state.get("initialized", False))

            # Load sorted cache: prefer new unified format, fall back to legacy parallel lists
            if "chain_sorted" in state:
                self._chain_sorted = {
                    int(k): [tuple(entry) for entry in v]
                    for k, v in state["chain_sorted"].items()
                }
            elif "chain_sorted_ids" in state and "chain_sorted_neg_scores" in state:
                ids_map = {int(k): list(v) for k, v in state["chain_sorted_ids"].items()}
                neg_map = {int(k): list(v) for k, v in state["chain_sorted_neg_scores"].items()}
                self._chain_sorted = {
                    k: list(zip(neg_map.get(k, []), ids_map.get(k, [])))
                    for k in ids_map
                }
            else:
                self._chain_sorted = {}
            self._chain_error_counts = {}
            for k, v in state.get("chain_error_counts", {}).items():
                folded: dict[str, int] = {}
                for err, cnt in dict(v).items():
                    norm_err = summarize_error(str(err), _FAILURE_PATTERN_ERROR_MAX_CHARS)
                    if not norm_err:
                        continue
                    folded[norm_err] = folded.get(norm_err, 0) + int(cnt)
                self._chain_error_counts[int(k)] = folded
            self._chain_total_counts = {
                int(k): int(v) for k, v in state.get("chain_total_counts", {}).items()
            }

            for i in range(self.num_chains):
                self.chains.setdefault(i, [])
                self.chain_history.setdefault(i, [])
                self.chain_prompt_count.setdefault(i, 0)
                self.nodes_since_restart.setdefault(i, 0)
                self.chain_gen_budget.setdefault(i, 0)
                self.prompt_budget.setdefault(i, 0)
                self._chain_sorted.setdefault(i, [])
                self._chain_error_counts.setdefault(i, {})
                self._chain_total_counts.setdefault(i, 1)

            self._validate_restart_every_n()

            # Rebuild _ready_chains: all chains with remaining budget
            # (no pending batches since we cleared _batches on resume)
            self._ready_chains = [
                i for i in range(self.num_chains)
                if self.chain_prompt_count.get(i, 0) < self.prompt_budget.get(i, 0)
                and self.prompt_budget.get(i, 0) > 0
            ]

            self._load_state_extra(state)
