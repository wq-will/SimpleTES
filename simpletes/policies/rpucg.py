"""
RPUCG (Rank-based PUCT with Gamma decay) policy for SimpleTES.

Key differences from PUCT:
1. DAG-aware Q-value with exponential decay:
   V(s) = max(raw_reward(s), γ · max_child V(c))
   propagated bottom-up through the full parent→child graph.
   Q is the percentile rank of V(s) across the entire population.

2. Percentile-rank normalization: Both Q and P scores are percentile ranks
   over the global population (not per-chain), so no scale factor is needed.

3. Anti-inbreeding multi-parent selection: When selecting k parents, exclude
   the 1-hop neighborhood (parents + children) of already-selected nodes.
"""
from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Any

from .base import TrajectoryPolicyBase, PendingFinalize, register_selector

if TYPE_CHECKING:
    from simpletes.node import Node, NodeDatabaseSnapshot


@register_selector("rpucg")
class RpucgPolicy(TrajectoryPolicyBase):
    """Rank-based PUCT with Gamma decay.

    Uses DAG-aware value propagation and percentile-rank normalization
    for both Q and P terms. Anti-inbreeding selection excludes 1-hop
    neighbors of already-selected parents.
    """

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["puct_c"] = self.c
        info["gamma"] = self.gamma
        return info

    def __init__(
        self,
        num_chains: int = 4,
        max_generations: int = 100,
        c: float = 1.0,
        gamma: float = 0.8,
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
        self.gamma = gamma

        # Per-chain PUCT state
        self.chain_visit_counts: dict[int, dict[str, int]] = {i: {} for i in range(self.num_chains)}
        self.chain_total_expansions: dict[int, int] = {i: 0 for i in range(self.num_chains)}

        # Temporary: stashed DB snapshot for _select_from_chain to access global population
        self._current_db: NodeDatabaseSnapshot | None = None

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_children_map(db_snapshot: NodeDatabaseSnapshot) -> dict[str, set[str]]:
        """Build parent→children mapping from all nodes' parent_ids. O(n)."""
        children: dict[str, set[str]] = {}
        for node in db_snapshot.nodes.values():
            for pid in node.parent_ids:
                if pid not in children:
                    children[pid] = set()
                children[pid].add(node.id)
        return children

    def _compute_v_values(
        self,
        db_snapshot: NodeDatabaseSnapshot,
        children_map: dict[str, set[str]],
    ) -> dict[str, float]:
        """Bottom-up V-value propagation with gamma decay.

        V(s) = max(raw_reward(s), γ · max(V(c) for c in children(s)))

        Nodes are processed in reverse creation order (children before parents).
        """
        # Sort by created_at descending so children are processed before parents
        nodes_by_time = sorted(
            db_snapshot.nodes.values(),
            key=lambda n: n.created_at,
            reverse=True,
        )

        v: dict[str, float] = {}
        for node in nodes_by_time:
            raw = node.score if node.score is not None else -float("inf")
            child_ids = children_map.get(node.id)
            if child_ids:
                max_child_v = max((v.get(cid, -float("inf")) for cid in child_ids), default=-float("inf"))
                v[node.id] = max(raw, self.gamma * max_child_v)
            else:
                v[node.id] = raw
        return v

    @staticmethod
    def _compute_percentile_ranks(values: dict[str, float]) -> dict[str, float]:
        """Compute percentile rank in [0, 1) for each entry using bisect."""
        if not values:
            return {}
        sorted_vals = sorted(values.values())
        n = len(sorted_vals)
        return {
            nid: bisect.bisect_left(sorted_vals, val) / n
            for nid, val in values.items()
        }

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select(
        self,
        db,
        n: int,
        stats=None,
        eval_concurrency: int = 1,
        backpressure_multiplier: float = 1.0,
    ):
        """Override to stash db snapshot for _select_from_chain."""
        self._current_db = db
        try:
            return super().select(db, n, stats, eval_concurrency, backpressure_multiplier)
        finally:
            self._current_db = None

    def _select_from_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
        n: int,
    ) -> list[Node]:
        if not chain_nodes:
            return []

        db_snapshot = self._current_db
        if db_snapshot is None:
            # Fallback: just return top-n by score
            return chain_nodes[:n]

        # Build DAG structures over full population
        children_map = self._build_children_map(db_snapshot)
        v_values = self._compute_v_values(db_snapshot, children_map)

        # Global percentile ranks
        q_rank = self._compute_percentile_ranks(v_values)
        raw_scores = {nid: node.score for nid, node in db_snapshot.nodes.items()}
        p_rank = self._compute_percentile_ranks(raw_scores)

        # PUCT scoring for chain nodes
        visit_counts = self.chain_visit_counts[chain_idx]
        total_expansions = self.chain_total_expansions[chain_idx]

        scored: list[tuple[str, float]] = []
        for node in chain_nodes:
            q = q_rank.get(node.id, 0.0)
            p = p_rank.get(node.id, 0.0)
            nc = visit_counts.get(node.id, 0)
            score = q + self.c * p * math.sqrt(1 + total_expansions) / (1 + nc)
            scored.append((node.id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build parent set map for anti-inbreeding
        parent_map: dict[str, set[str]] = {
            node.id: set(node.parent_ids) for node in db_snapshot.nodes.values()
        }

        # Anti-inbreeding greedy selection
        node_map = {node.id: node for node in chain_nodes}
        selected: list[Node] = []
        excluded: set[str] = set()

        for nid, _ in scored:
            if nid in excluded:
                continue
            if nid not in node_map:
                continue
            selected.append(node_map[nid])
            if len(selected) >= n:
                break
            # Exclude 1-hop neighborhood: self + parents + children
            excluded.add(nid)
            excluded.update(parent_map.get(nid, set()))
            excluded.update(children_map.get(nid, set()))

        return selected

    # ------------------------------------------------------------------
    # Batch completion
    # ------------------------------------------------------------------

    def _finalize_hook_locked(
        self,
        pending: PendingFinalize,
        best_node: Node | None,
        unlocked_result: Any,
    ) -> None:
        """Update RPUCG state after batch commit."""
        chain_idx = pending.chain_idx
        visit_counts = self.chain_visit_counts[chain_idx]
        for parent_id in pending.inspirations:
            visit_counts[parent_id] = visit_counts.get(parent_id, 0) + 1
        self.chain_total_expansions[chain_idx] += 1

    def _on_chain_reset_locked(self, chain_idx: int, kept_node_id: str | None) -> None:
        self.chain_visit_counts[chain_idx] = {}
        self.chain_total_expansions[chain_idx] = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _state_dict_extra(self) -> dict[str, Any]:
        return {
            "c": self.c,
            "gamma": self.gamma,
            "chain_visit_counts": {
                str(k): dict(v) for k, v in self.chain_visit_counts.items()
            },
            "chain_total_expansions": {
                str(k): v for k, v in self.chain_total_expansions.items()
            },
        }

    def _load_state_extra(self, state: dict[str, Any]) -> None:
        self.c = float(state.get("c", self.c))
        self.gamma = float(state.get("gamma", self.gamma))
        self.chain_visit_counts = {
            int(k): {str(nid): int(cnt) for nid, cnt in v.items()}
            for k, v in state.get("chain_visit_counts", {}).items()
        }
        self.chain_total_expansions = {
            int(k): int(v) for k, v in state.get("chain_total_expansions", {}).items()
        }

        for i in range(self.num_chains):
            self.chain_visit_counts.setdefault(i, {})
            self.chain_total_expansions.setdefault(i, 0)
