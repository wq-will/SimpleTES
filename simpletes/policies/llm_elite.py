from __future__ import annotations

import random
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from simpletes.evaluator import rich_print
from simpletes.templates import (
    ELITE_SELECTION_PROMPT_TEMPLATE,
    ELITE_CONTEXT_TEMPLATE,
    ELITE_ENTRY_TEMPLATE,
)

from .base import TrajectoryPolicyBase, PendingFinalize, register_selector

if TYPE_CHECKING:
    from simpletes.node import Node, NodeDatabase, NodeDatabaseSnapshot


@register_selector("llm_elite")
class LLMElitePolicy(TrajectoryPolicyBase):
    """
    LLM-managed Elite Pool Policy with unified decision logic.

    The LLM always decides whether to add/replace/reject new nodes,
    regardless of whether the pool is full. This ensures diversity
    from the start by rejecting similar solutions even when pool has space.

    Actions:
    - ADD: Add new node to pool (only when pool not full)
    - REPLACE {index}: Replace existing node at index with new node
    - REJECT: Do not add the new node (too similar or low value)

    Finalize flow:
    - _finalize_hook_unlocked: LLM decides action (add/replace/reject)
    - _finalize_hook_locked: Apply the decision to elite_sets
    """

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["elite_limit"] = self.elite_limit
        info["elite_selection_strategy"] = self.elite_selection_strategy
        if self.elite_selection_strategy == "balance":
            info["exploitation_ratio"] = self.exploitation_ratio
            info["exploration_ratio"] = self.exploration_ratio
            info["elite_ratio"] = self.elite_ratio
        if self.llm_policy_model:
            info["llm_policy_model"] = self.llm_policy_model
        return info

    def _log(self, icon: str, msg: str) -> str:
        """Format log message with timestamp."""
        now_str = datetime.now().strftime("%H:%M:%S")
        return f"({now_str}) {icon} \\[Policy: llm_elite] {msg}"

    def __init__(
        self,
        num_chains: int = 4,
        max_generations: int = 100,
        k: int = 1,
        min_inspirations_cnt: int | None = None,
        max_inspirations_cnt: int | None = None,
        llm_policy_model: str | None = None,
        llm_policy_api_base: str | None = None,
        llm_policy_api_key: str | None = None,
        llm_policy_pool_size: int | None = None,
        # Selection strategy: "linear_rank" or "balance"
        elite_selection_strategy: str = "linear_rank",
        # Balance strategy parameters (shared with balance policy via config)
        # When elite_selection_strategy="balance", these are used for selection within elite pool
        exploitation_ratio: float = 0.7,
        exploration_ratio: float = 0.2,
        elite_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            num_chains=num_chains,
            max_generations=max_generations,
            k=k,
            min_inspirations_cnt=min_inspirations_cnt,
            max_inspirations_cnt=max_inspirations_cnt,
            llm_policy_model=llm_policy_model,
            llm_policy_api_base=llm_policy_api_base,
            llm_policy_api_key=llm_policy_api_key,
            **kwargs,
        )
        self.elite_limit = llm_policy_pool_size if llm_policy_pool_size is not None else 15

        # Selection strategy
        self.elite_selection_strategy = elite_selection_strategy
        self.exploitation_ratio = exploitation_ratio
        self.exploration_ratio = exploration_ratio
        self.elite_ratio = elite_ratio

        # Structure: {chain_idx: [Node, ...]}
        self.elite_sets: dict[int, list[Node]] = {
            i: [] for i in range(self.num_chains)
        }

        # History of elite pool changes for analysis
        self.elite_history: list[dict[str, Any]] = []

    async def _finalize_hook_unlocked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
    ) -> dict[str, Any]:
        """LLM decides whether to add/replace/reject. Returns empty dict on failure."""
        chain_idx = pending.chain_idx

        if not best_node or best_node.score is None:
            return {}

        elite_list = self.elite_sets[chain_idx]

        # Check for duplicate
        if any(node.id == best_node.id for node in elite_list):
            return {"action": "reject", "reason": "duplicate"}

        pool_is_full = len(elite_list) >= self.elite_limit
        current_size = len(elite_list)

        # Build prompt
        pool_description = ""
        if elite_list:
            for i, node in enumerate(elite_list):
                pool_description += f"[{i}] Score: {node.score:.6f} | {node.reflection or '(no reflection)'}\n"
        else:
            pool_description = "(empty pool)\n"

        new_candidate_description = f"Score: {best_node.score:.6f} | {best_node.reflection or '(no reflection)'}\n"

        if pool_is_full:
            decision_instruction = "Pool is FULL. Choose: REPLACE {index} or REJECT"
        elif current_size == 0:
            decision_instruction = "Pool is EMPTY. Choose: ADD or REJECT"
        else:
            decision_instruction = (
                f"Pool has space ({current_size}/{self.elite_limit}). Choose:\n"
                "- ADD: if candidate brings a NEW distinct approach\n"
                "- REPLACE {{index}}: if candidate is BETTER version of existing approach (removes redundancy)\n"
                "- REJECT: if candidate is too similar or low value"
            )

        prompt = ELITE_SELECTION_PROMPT_TEMPLATE.format(
            task_instruction=self.task_instruction,
            current_size=current_size,
            elite_limit=self.elite_limit,
            pool_description=pool_description,
            new_candidate_description=new_candidate_description,
            decision_instruction=decision_instruction,
        )

        try:
            message = await self._llm_generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            resp_text = (message.content or "").strip()
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is None:
                psf = getattr(message, "provider_specific_fields", None)
                if isinstance(psf, dict):
                    reasoning = psf.get("reasoning_content")
            if reasoning:
                rs = str(reasoning).strip()
                llm_output = (
                    f"<|channel|>analysis<|message|>{rs}<|end|>"
                    f"<|start|>assistant<|channel|>final<|message|>{resp_text}"
                )
            else:
                llm_output = resp_text

            action_match = re.search(r"## ACTION\s+(\w+)(?:\s+(\d+))?", resp_text)
            reason_match = re.search(r"## REASON:\s*(.+?)(?:\n|$)", resp_text, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else ""

            base = {"llm_input": prompt, "llm_output": llm_output}

            if action_match:
                action_str = action_match.group(1).upper()
                idx_str = action_match.group(2)

                if action_str == "ADD" and not pool_is_full:
                    return {**base, "action": "add", "reason": reason}
                elif action_str == "REJECT":
                    return {**base, "action": "reject", "reason": reason}
                elif action_str == "REPLACE" and idx_str and 0 <= int(idx_str) < len(elite_list):
                    return {**base, "action": "replace", "remove_idx": int(idx_str), "reason": reason}

            return base  # Parse failed: still pass input/output for recording, fallback in locked
        except Exception as e:
            rich_print(f"[red]\\[llm_elite]\\[chain={chain_idx}] LLM Error: {e}[/red]")
            return {"llm_input": prompt, "llm_output": None}  # Record attempt, fallback in locked

    def _finalize_hook_locked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
        unlocked_result: Any,
    ) -> None:
        """Apply decision to elite_sets. Uses fallback if LLM failed or race detected."""
        chain_idx = pending.chain_idx
        elite_list = self.elite_sets[chain_idx]

        if not best_node or best_node.score is None:
            return

        result = unlocked_result if isinstance(unlocked_result, dict) else {}
        action = result.get("action")
        reason = result.get("reason", "")
        used_fallback = False

        # Fallback if LLM returned empty or invalid action
        if not action or action not in ("add", "replace", "reject"):
            action, reason, remove_idx = self._fallback_logic(elite_list, best_node)
            used_fallback = True
            if action == "replace":
                result = {**result, "action": "replace", "remove_idx": remove_idx}

        # Handle race condition for replace
        if action == "replace":
            remove_idx = result.get("remove_idx", -1)
            if not (0 <= remove_idx < len(elite_list)):
                action, reason, remove_idx = self._fallback_logic(elite_list, best_node)
                used_fallback = True

        # Override: never reject a new best
        if action == "reject" and elite_list:
            new_score = best_node.score  # guaranteed not None
            pool_max = max([n.score for n in elite_list])
            if new_score > pool_max:
                if len(elite_list) < self.elite_limit:
                    action = "add"
                    reason = f"[Override] New best {new_score:.4f} > pool max {pool_max:.4f}, add"
                else:
                    lowest_idx = min(range(len(elite_list)), key=lambda i: elite_list[i].score)
                    action = "replace"
                    remove_idx = lowest_idx
                    result = {**result, "action": "replace", "remove_idx": lowest_idx}
                    reason = f"[Override] New best {new_score:.4f} > pool max {pool_max:.4f}, replace idx={lowest_idx}"
                used_fallback = True

        # Execute action
        removed_node = None
        remove_idx_final = None

        if action == "add":
            elite_list.append(best_node)
            rich_print(self._log("📥", f"\\[chain={chain_idx}] [green]Add[/green]: {best_node.score:.6f}, pool={len(elite_list)}/{self.elite_limit}"))

        elif action == "replace":
            remove_idx_final = result.get("remove_idx") if not used_fallback else remove_idx
            removed_node = elite_list[remove_idx_final]
            elite_list[remove_idx_final] = best_node
            rich_print(self._log("🔄", f"\\[chain={chain_idx}] [yellow]Replace[/yellow] idx={remove_idx_final}: {removed_node.score:.6f} -> {best_node.score:.6f}"))

        elif action == "reject":
            rich_print(self._log("❌", f"\\[chain={chain_idx}] [red]Reject[/red]: {best_node.score:.6f}"))

        if reason:
            rich_print(f"[dim]  Reason: {reason}[/dim]")

        # Record history (include LLM input/output from unlocked phase)
        llm_input = result.get("llm_input") if isinstance(result, dict) else None
        llm_output = result.get("llm_output") if isinstance(result, dict) else None
        self._record_history(
            pending, action, best_node, removed_node, remove_idx_final, reason, used_fallback,
            llm_input, llm_output,
        )

    def _fallback_logic(
        self, elite_list: list[Node], best_node: Node
    ) -> tuple:
        """Simple fallback: only add/replace if new node is new best."""
        new_score = best_node.score  # guaranteed not None
        pool_max = max((n.score for n in elite_list), default=-1e9)

        if new_score > pool_max:
            if len(elite_list) < self.elite_limit:
                return "add", "[Fallback] new best, pool not full", None
            else:
                lowest_idx = min(range(len(elite_list)), key=lambda i: elite_list[i].score)
                return "replace", f"[Fallback] new best > {pool_max:.6f}", lowest_idx
        else:
            return "reject", "[Fallback] not new best", None

    def _record_history(
        self,
        pending: PendingFinalize,
        action: str,
        new_node: Node | None,
        removed_node: Node | None,
        removed_index: int | None,
        reason: str,
        used_fallback: bool,
        llm_input: str | None = None,
        llm_output: str | None = None,
    ) -> None:
        """Record an elite pool change event, including LLM input/output when available."""
        elite_list = self.elite_sets[pending.chain_idx]
        scores = [n.score for n in elite_list if n.score is not None]
        pool_avg = sum(scores) / len(scores) if scores else 0.0
        pool_max = max(scores) if scores else 0.0

        self.elite_history.append({
            "timestamp": datetime.now().isoformat(),
            "gen_id": pending.gen_id,
            "chain_idx": pending.chain_idx,
            "action": action,
            "new_node_id": new_node.id if new_node else "",
            "new_node_score": new_node.score if new_node else None,
            "removed_node_id": removed_node.id if removed_node else "",
            "removed_node_score": removed_node.score if removed_node else None,
            "removed_index": removed_index if removed_index is not None else -1,
            "llm_reason": reason,
            "used_fallback": used_fallback,
            "pool_size": len(elite_list),
            "pool_avg_score": pool_avg,
            "pool_max_score": pool_max,
            "llm_input": llm_input,
            "llm_output": llm_output,
        })

    def _on_chain_reset_locked(self, chain_idx: int, kept_node_id: str | None) -> None:
        self.elite_sets[chain_idx] = []

    def _select_from_chain(self, chain_idx: int, chain_nodes: list[Node], n: int) -> list[Node]:
        elite_list = self.elite_sets[chain_idx]

        # Initialize elite pool from chain nodes if empty
        if not elite_list:
            elite_list.extend(chain_nodes[:self.elite_limit])
            self.elite_sets[chain_idx] = elite_list

        if not elite_list:
            return []

        if self.elite_selection_strategy == "all":
            assert n <= self.elite_limit, "n must not be greater than elite_limit when elite_selection_strategy is all"
            return list(elite_list)  # Return all nodes
        elif self.elite_selection_strategy == "balance":
            return self._select_balance(elite_list, n)
        else:  # default: linear_rank
            return self._select_linear_rank(elite_list, n)

    def _select_linear_rank(self, nodes: list[Node], n: int) -> list[Node]:
        """Linear Rank Selection: higher scores get larger weights."""
        # Sort by score ascending: higher scores at the end get larger weights
        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score is not None else -1e9)
        m = len(sorted_nodes)

        # Rank weights: 1, 2, 3, ..., m
        weights = [i + 1 for i in range(m)]

        selected: list[Node] = []
        available_indices = list(range(m))

        while len(selected) < n and available_indices:
            current_weights = [weights[i] for i in available_indices]
            relative_idx = random.choices(range(len(available_indices)), weights=current_weights, k=1)[0]
            actual_idx = available_indices.pop(relative_idx)
            selected.append(sorted_nodes[actual_idx])

        return selected

    def _select_balance(self, nodes: list[Node], n: int) -> list[Node]:
        """Balance Selection: exploitation/exploration/random tiers."""
        if not nodes or n <= 0:
            return []
        if len(nodes) <= n:
            return list(nodes)

        # Sort by score descending (best first)
        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score is not None else -1e9, reverse=True)
        m = len(sorted_nodes)

        # Define tiers
        elite_end = max(1, int(m * self.elite_ratio))
        mid_start = int(m * 0.1)
        mid_end = int(m * 0.6)

        selected: list[Node] = []
        used: set = set()

        # Always include best node
        selected.append(sorted_nodes[0])
        used.add(sorted_nodes[0].id)

        for _ in range(n - 1):
            roll = random.random()
            if roll < self.exploitation_ratio:
                candidates = sorted_nodes[:elite_end]
            elif roll < self.exploitation_ratio + self.exploration_ratio:
                candidates = sorted_nodes[mid_start:mid_end] if mid_end > mid_start else sorted_nodes
            else:
                candidates = sorted_nodes

            # Pick random unused from candidates
            available = [p for p in candidates if p.id not in used]
            if not available:
                available = [p for p in sorted_nodes if p.id not in used]
                if not available:
                    break

            pick = random.choice(available)
            selected.append(pick)
            used.add(pick.id)

        return selected

    # ------------------------------------------------------------------
    # Policy Context for Generator
    # ------------------------------------------------------------------
    def get_policy_context(self, chain_idx: int, db: NodeDatabase | NodeDatabaseSnapshot) -> str:
        """Return elite pool overview for inclusion in generation prompt.

        Provides the LLM with an overview of all diverse solutions in the elite pool,
        showing their scores, metrics, and reflections (without code).
        """
        elite_list = self.elite_sets.get(chain_idx, [])
        if not elite_list:
            return ""

        # Sort by score descending for display
        sorted_elites = sorted(
            elite_list,
            key=lambda x: x.score if x.score is not None else -1e9,
            reverse=True
        )

        entries = []
        for i, node in enumerate(sorted_elites, 1):
            score_str = f"{node.score:.6f}" if node.score is not None else "N/A"

            # Format metrics
            metrics_line = ""
            if node.metrics:
                metrics_parts = []
                for k, v in node.metrics.items():
                    if k == "error":
                        continue
                    if isinstance(v, float):
                        metrics_parts.append(f"{k}={v:.4f}")
                    else:
                        metrics_parts.append(f"{k}={v}")
                if metrics_parts:
                    metrics_line = f"   Metrics: {', '.join(metrics_parts)}\n"

            # Format reflection
            reflection_line = ""
            if node.reflection:
                reflection = node.reflection.strip()
                # if len(reflection) > 200:
                #     reflection = reflection[:197] + "..."
                reflection_line = f"   Reflection: {reflection}\n"

            entries.append(ELITE_ENTRY_TEMPLATE.format(
                index=i,
                score=score_str,
                metrics_line=metrics_line,
                reflection_line=reflection_line,
            ))

        return ELITE_CONTEXT_TEMPLATE.format(
            num_elites=len(sorted_elites),
            chain_idx=chain_idx,
            entries="".join(entries),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _state_dict_extra(self) -> dict[str, Any]:
        return {
            "elite_limit": self.elite_limit,
            "elite_selection_strategy": self.elite_selection_strategy,
            "exploitation_ratio": self.exploitation_ratio,
            "exploration_ratio": self.exploration_ratio,
            "elite_ratio": self.elite_ratio,
            "elite_node_ids": {
                str(c_idx): [node.id for node in nodes]
                for c_idx, nodes in self.elite_sets.items()
            },
            "elite_history": self.elite_history,
        }

    def _load_state_extra(self, state: dict[str, Any]) -> None:
        self.elite_limit = state.get("elite_limit", self.elite_limit)
        self.elite_selection_strategy = state.get("elite_selection_strategy", self.elite_selection_strategy)
        self.exploitation_ratio = float(state.get("exploitation_ratio", self.exploitation_ratio))
        self.exploration_ratio = float(state.get("exploration_ratio", self.exploration_ratio))
        self.elite_ratio = float(state.get("elite_ratio", self.elite_ratio))
        # Node references will be restored by reconcile_with_db
        self._pending_elite_ids = state.get("elite_node_ids", {})
        self.elite_history = state.get("elite_history", [])

    def reconcile_with_db(self, db) -> None:
        """Restore elite_sets node references from DB after loading state."""
        super().reconcile_with_db(db)

        if hasattr(self, "_pending_elite_ids"):
            for chain_idx_str, node_ids in self._pending_elite_ids.items():
                chain_idx = int(chain_idx_str)
                nodes = []
                for node_id in node_ids:
                    node = db.nodes.get(node_id)
                    if node:
                        nodes.append(node)
                self.elite_sets[chain_idx] = nodes
            del self._pending_elite_ids
