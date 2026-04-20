"""Balanced chain policy: each chain samples exploitation / exploration / random
by configurable ratios (see balanced_sample)."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from .base import TrajectoryPolicyBase, register_selector

if TYPE_CHECKING:
    from simpletes.node import Node


def balanced_sample(
    nodes: list[Node],
    n: int,
    exploitation_ratio: float = 0.7,
    exploration_ratio: float = 0.2,
    elite_ratio: float = 0.2,
) -> list[Node]:
    """Sample n nodes with exploitation/exploration balance.
    
    Simple implementation optimized for typical use cases (n=5, nodes < 100).
    """
    if not nodes or n <= 0:
        return []
    if len(nodes) <= n:
        return list(nodes)

    result: list[Node] = [nodes[0]]  # Always include best
    if len(result) >= n:
        return result

    used = {nodes[0].id}
    pool_size = len(nodes)
    elite_end = max(1, int(pool_size * elite_ratio))
    mid_start = max(1, int(pool_size * 0.1))
    mid_end = max(2, int(pool_size * 0.6))

    for _ in range(n - 1):
        roll = random.random()
        if roll < exploitation_ratio:
            candidates = nodes[:elite_end]
        elif roll < exploitation_ratio + exploration_ratio:
            candidates = nodes[mid_start:mid_end] if mid_end > mid_start else nodes
        else:
            candidates = nodes

        # Pick random unused from candidates
        available = [p for p in candidates if p.id not in used]
        if not available:
            available = [p for p in nodes if p.id not in used]
            if not available:
                break
        
        pick = random.choice(available)
        result.append(pick)
        used.add(pick.id)

    return result


@register_selector("balance")
class BalancePolicy(TrajectoryPolicyBase):
    """Balanced chains with optional local-best selection."""

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update({
            "exploitation_ratio": self.exploitation_ratio,
            "exploration_ratio": self.exploration_ratio,
            "elite_ratio": self.elite_ratio,
        })
        return info

    def __init__(
        self,
        num_chains: int = 4,
        max_generations: int = 100,
        k: int = 4,
        exploitation_ratio: float = 0.7,
        exploration_ratio: float = 0.2,
        elite_ratio: float = 0.2,
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
        self.exploitation_ratio = exploitation_ratio
        self.exploration_ratio = exploration_ratio
        self.elite_ratio = elite_ratio

    def _select_from_chain(
        self,
        chain_idx: int,
        chain_nodes: list[Node],
        n: int,
    ) -> list[Node]:
        return balanced_sample(
            chain_nodes,
            n,
            self.exploitation_ratio,
            self.exploration_ratio,
            self.elite_ratio,
        )

    def _state_dict_extra(self) -> dict[str, Any]:
        return {
            "exploitation_ratio": self.exploitation_ratio,
            "exploration_ratio": self.exploration_ratio,
            "elite_ratio": self.elite_ratio,
        }

    def _load_state_extra(self, state: dict[str, Any]) -> None:
        self.exploitation_ratio = float(state.get("exploitation_ratio", self.exploitation_ratio))
        self.exploration_ratio = float(state.get("exploration_ratio", self.exploration_ratio))
        self.elite_ratio = float(state.get("elite_ratio", self.elite_ratio))
