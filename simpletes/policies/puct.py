"""PUCT (Predictor + Upper Confidence bounds applied to Trees) policy.

Score: Q(s) + c * scale * P(s) * sqrt(1+T) / (1+n(s))
where Q = max(R(s), max_child_reward), scale = r_max-r_min over all nodes,
P = rank prior, T = total_expansions, n = visit count.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .base import TrajectoryPolicyBase, PendingFinalize, register_selector

if TYPE_CHECKING:
    from simpletes.node import Node


@register_selector("puct")
class PuctPolicy(TrajectoryPolicyBase):
    """Independent chains, each selecting expansions by PUCT score."""

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["puct_c"] = self.c
        return info

    def __init__(
        self,
        num_chains: int = 4,
        max_generations: int = 100,
        c: float = 1.0,
        k: int = 1,
        min_inspirations_cnt: int | None = None,
        max_inspirations_cnt: int | None = None,
        **kwargs,
    ):
        super().__init__(
            num_chains=num_chains,
            max_generations=max_generations,
            k=k,
            min_inspirations_cnt=min_inspirations_cnt,
            max_inspirations_cnt=max_inspirations_cnt,
            **kwargs,
        )
        self.c = c

        # Per-chain PUCT state
        self.chain_visit_counts: dict[int, dict[str, int]] = {i: {} for i in range(self.num_chains)}
        self.chain_max_child_reward: dict[int, dict[str, float]] = {i: {} for i in range(self.num_chains)}
        self.chain_total_expansions: dict[int, int] = {i: 0 for i in range(self.num_chains)}

    def _compute_puct_scores_for_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
    ) -> list[tuple[str, float]]:
        """Compute PUCT scores for nodes in a specific chain.

        Args:
            chain_idx: Index of the chain
            chain_nodes: List of nodes in the chain (already sorted by score)

        Returns:
            List of (node_id, puct_score) tuples, sorted by score descending.
        """
        if not chain_nodes:
            return []

        # Compute reward range for scaling (chain_nodes sorted by score desc)
        r_max = chain_nodes[0].score
        r_min = chain_nodes[-1].score
        scale = max(r_max - r_min, 1e-6)  # Avoid division by zero

        # Compute rank-based priors
        # P(s) = (|nodes| - rank(s)) / sum(|nodes| - rank(s'))
        n_nodes = len(chain_nodes)
        rank_sum = n_nodes * (n_nodes + 1) / 2

        # Get chain-specific PUCT state
        visit_counts = self.chain_visit_counts[chain_idx]
        max_child_reward = self.chain_max_child_reward[chain_idx]
        total_expansions = self.chain_total_expansions[chain_idx]

        # Compute PUCT scores
        puct_scores = []
        for rank, node in enumerate(chain_nodes):
            # Q(s) = max(R(s), max_child_reward[s])
            q_value = max(node.score, max_child_reward.get(node.id, -float('inf')))

            # P(s) = (|nodes| - rank(s)) / sum(|nodes| - rank(s'))
            prior = (n_nodes - rank) / rank_sum if rank_sum > 0 else 1.0 / n_nodes

            # n(s) = visit_counts[s]
            visit_count = visit_counts.get(node.id, 0)

            # PUCT formula: Q(s) + c * scale * P(s) * sqrt(1+T) / (1+n(s))
            exploration_term = (
                self.c * scale * prior * math.sqrt(1 + total_expansions) / (1 + visit_count)
            )

            puct_score = q_value + exploration_term
            puct_scores.append((node.id, puct_score))

        # Sort by PUCT score descending
        puct_scores.sort(key=lambda x: x[1], reverse=True)
        return puct_scores

    def _select_from_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
        n: int,
    ) -> list[Node]:
        puct_scores = self._compute_puct_scores_for_chain(chain_idx, chain_nodes)
        if not puct_scores:
            return []
        selected_ids = [node_id for node_id, _ in puct_scores[:n]]
        node_map = {node.id: node for node in chain_nodes}
        return [node_map[nid] for nid in selected_ids if nid in node_map]

    def _finalize_hook_locked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
        unlocked_result: Any,
    ) -> None:
        """Update PUCT state after batch commit."""
        chain_idx = pending.chain_idx
        best_score = best_node.score if best_node else -float("inf")

        visit_counts = self.chain_visit_counts[chain_idx]
        max_child_reward = self.chain_max_child_reward[chain_idx]
        for parent_id in pending.inspirations:
            current_max = max_child_reward.get(parent_id, -float("inf"))
            max_child_reward[parent_id] = max(current_max, best_score)
            visit_counts[parent_id] = visit_counts.get(parent_id, 0) + 1

        self.chain_total_expansions[chain_idx] += 1

    def _on_chain_reset_locked(self, chain_idx: int, kept_node_id: str | None) -> None:
        self.chain_visit_counts[chain_idx] = {}
        self.chain_max_child_reward[chain_idx] = {}
        self.chain_total_expansions[chain_idx] = 0

    def _state_dict_extra(self) -> dict[str, Any]:
        return {
            "c": self.c,
            "chain_visit_counts": {
                str(k): dict(v) for k, v in self.chain_visit_counts.items()
            },
            "chain_max_child_reward": {
                str(k): dict(v) for k, v in self.chain_max_child_reward.items()
            },
            "chain_total_expansions": {
                str(k): v for k, v in self.chain_total_expansions.items()
            },
        }

    def _load_state_extra(self, state: dict[str, Any]) -> None:
        self.c = float(state.get("c", self.c))
        self.chain_visit_counts = {
            int(k): {str(nid): int(cnt) for nid, cnt in v.items()}
            for k, v in state.get("chain_visit_counts", {}).items()
        }
        self.chain_max_child_reward = {
            int(k): {str(nid): float(rew) for nid, rew in v.items()}
            for k, v in state.get("chain_max_child_reward", {}).items()
        }
        self.chain_total_expansions = {
            int(k): int(v) for k, v in state.get("chain_total_expansions", {}).items()
        }

        for i in range(self.num_chains):
            self.chain_visit_counts.setdefault(i, {})
            self.chain_max_child_reward.setdefault(i, {})
            self.chain_total_expansions.setdefault(i, 0)
