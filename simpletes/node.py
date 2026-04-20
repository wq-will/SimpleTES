"""
Node data model and database for SimpleTES.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import math
from typing import Any


def score_key(node: Node) -> float:
    """Key function for sorting nodes by score. -inf is allowed; NaN is not."""
    if math.isnan(node.score):
        raise ValueError(f"score is NaN for node {node.id}")
    return node.score


def score_from_metrics(metrics: Any) -> float:
    """Coerce a metrics dict into a finite float score, returning -inf on error/missing/NaN."""
    if not isinstance(metrics, dict) or metrics.get("error"):
        return -float("inf")
    try:
        score = float(metrics.get("combined_score", -float("inf")))
    except (TypeError, ValueError):
        return -float("inf")
    return score if math.isfinite(score) else -float("inf")


class Status(Enum):
    """Node lifecycle status.
    
    Lifecycle: EVAL_PENDING -> DONE
    - EVAL_PENDING: Code generated, awaiting evaluation
    - DONE: Evaluated and stored in database
    """
    EVAL_PENDING = auto()  # Code generated, awaiting evaluation  
    DONE = auto()          # Evaluated and in database


@dataclass
class Node:
    """A DAG node representing a program candidate."""
    id: str
    code: str
    parent_ids: list[str] = field(default_factory=list)
    gen_id: int | None = None       # Batch identifier (from GenerationTask)
    chain_idx: int | None = None    # Chain index (from policy)
    shared_construction_id: str | None = None  # Shared construction snapshot seen by this node
    metrics: dict[str, Any] | None = None
    score: float | None = None
    status: Status = Status.EVAL_PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # LLM I/O tracking (only populated when save_llm_io is enabled)
    llm_input: str | None = None  # Full prompt sent to LLM
    llm_output: str | None = None  # Full raw output including reasoning
    token_usage: dict[str, int] | None = None  # Token counts if available
    reflection: str | None = None  # Short approach + success/failure analysis

    def to_dict(self, include_llm_io: bool = False) -> dict[str, Any]:
        d = {
            "id": self.id,
            "code": self.code,
            "metrics": self.metrics,
            "score": self.score,
            "parent_ids": self.parent_ids,
            "gen_id": self.gen_id,
            "chain_idx": self.chain_idx,
            "shared_construction_id": self.shared_construction_id,
            "status": self.status.name,
            "created_at": self.created_at,
        }
        if self.reflection is not None:
            d["reflection"] = self.reflection
        if include_llm_io:
            if self.llm_input is not None:
                d["llm_input"] = self.llm_input
            if self.llm_output is not None:
                d["llm_output"] = self.llm_output
            if self.token_usage is not None:
                d["token_usage"] = self.token_usage
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        status_str = d.get("status", "DONE").upper()
        try:
            status = Status[status_str]
        except KeyError:
            status = Status.DONE
            
        return cls(
            id=d["id"],
            code=d["code"],
            parent_ids=list(d.get("parent_ids", [])),
            gen_id=d.get("gen_id"),
            chain_idx=d.get("chain_idx"),
            shared_construction_id=d.get("shared_construction_id"),
            metrics=d.get("metrics"),
            score=d.get("score"),
            status=status,
            created_at=d.get("created_at", datetime.utcnow().isoformat()),
            llm_input=d.get("llm_input"),
            llm_output=d.get("llm_output"),
            token_usage=d.get("token_usage"),
            reflection=d.get("reflection"),
        )


def _require_finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, got bool")
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number (got {value!r})")
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite (got {num})")
    return num


def validate_node_for_db(node: Node) -> None:
    """Validate node invariants before inserting into the database."""
    if node.status is not Status.DONE:
        raise ValueError(f"node status must be DONE (got {node.status})")
    if not isinstance(node.code, str) or not node.code.strip():
        raise ValueError("node code must be a non-empty string")
    if not isinstance(node.metrics, dict):
        raise ValueError("metrics must be a dict")
    if "combined_score" not in node.metrics:
        raise ValueError("metrics must include combined_score")
    # -inf scores are accepted; NaN is not (validated by score_key)
    _require_finite_number("metrics.combined_score", node.metrics.get("combined_score"))


class NodeDatabase:
    """Minimal in-memory database for evaluated program nodes.
    
    All nodes in the database are evaluated (DONE status). Uses lazy sorting
    to avoid O(n) insert overhead - only sorts when all_nodes_sorted() is called.
    Nodes must satisfy validate_node_for_db().
    
    Thread-safety:
        This class is NOT thread-safe. External synchronization is required
        when accessing from multiple coroutines. Use snapshot() for read-only
        access in concurrent contexts.
    """
    
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self._sorted_dirty = True
        self._sorted_cache: list[Node] = []
        # Version counter for detecting changes (useful for caching)
        self._version: int = 0

    def add(self, node: Node) -> None:
        """Add an evaluated node to the database.
        
        All nodes added should be evaluated (DONE status with score).
        """
        validate_node_for_db(node)
        self.nodes[node.id] = node
        self._sorted_dirty = True
        self._version += 1

    def get(self, node_id: str) -> Node:
        """Get a node by ID."""
        return self.nodes[node_id]

    def all_nodes_sorted(self) -> list[Node]:
        """Return all nodes sorted by score (descending).
        
        All nodes in the database are evaluated, so this returns all nodes.
        Uses lazy sorting - only sorts when the database has changed.
        """
        if self._sorted_dirty:
            self._sorted_cache = sorted(
                self.nodes.values(),
                key=score_key,
                reverse=True
            )
            self._sorted_dirty = False
        return self._sorted_cache

    def best(self) -> Node | None:
        """Return the best scoring node, or None if empty.

        Uses the sorted cache when available for O(1) access.
        """
        if not self.nodes:
            return None
        sorted_nodes = self.all_nodes_sorted()
        return sorted_nodes[0] if sorted_nodes else None
    
    def __len__(self) -> int:
        """Return the number of nodes in the database."""
        return len(self.nodes)

    def snapshot(self) -> NodeDatabaseSnapshot:
        """Create a read-only snapshot of the current database state.
        
        Returns a lightweight snapshot that can be safely used for read-only
        operations (like inspiration selection) without holding locks.
        The snapshot shares Node references with the original (nodes are immutable
        after evaluation).
        """
        return NodeDatabaseSnapshot(
            nodes=dict(self.nodes),  # Shallow copy of dict
            sorted_cache=list(self._sorted_cache) if not self._sorted_dirty else None,
            version=self._version,
        )


class NodeDatabaseSnapshot:
    """Read-only snapshot of NodeDatabase for concurrent access.
    
    This is a lightweight snapshot that shares Node references with the original
    database. Safe for read-only operations like inspiration selection.
    """
    
    def __init__(self, nodes: dict[str, Node], sorted_cache: list[Node] | None, version: int):
        self.nodes = nodes
        self._sorted_cache = sorted_cache
        self._version = version

    def all_nodes_sorted(self) -> list[Node]:
        """Return all nodes sorted by score (descending)."""
        if self._sorted_cache is None:
            self._sorted_cache = sorted(
                self.nodes.values(),
                key=score_key,
                reverse=True
            )
        return self._sorted_cache

    def best(self) -> Node | None:
        """Return the best scoring node, or None if empty."""
        if not self.nodes:
            return None
        sorted_nodes = self.all_nodes_sorted()
        return sorted_nodes[0] if sorted_nodes else None
    
    def __len__(self) -> int:
        """Return the number of nodes in the snapshot."""
        return len(self.nodes)


from simpletes.utils.code_extract import (
    EvolveBlockContext as EvolveBlockContext,
    extract_code as extract_code,
    extract_code_detailed as extract_code_detailed,
)


def save_score_statistics(
    nodes: list[dict[str, Any]],
    output_dir: str,
    completed_evaluations: int,
    best_score: float,
) -> None:
    """Save score statistics to CSV and generate quantile plot.

    Args:
        nodes: List of node dicts (from checkpoint state).
        output_dir: Directory to save the CSV and plot.
        completed_evaluations: Current number of completed evaluations (used in filename).
        best_score: Current best score (shown in plot title).
    """
    import csv
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    node_without_scores = [n for n in nodes if n.get("score") is None]
    assert len(node_without_scores) == 0, f"node_without_scores: {node_without_scores}"

    # nodes_with_scores = [
    #     (n["id"], n["score"], n.get("created_at", ""), n.get("parent_ids", []),
    #      n.get("gen_id"), n.get("chain_idx"), n.get('metrics', {}).get('error', ""))
    #     for n in nodes if n.get("score") is not None
    # ]

    # Save CSV
    csv_path = os.path.join(output_dir, f"scores_{completed_evaluations:06d}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score", "created_at", "parent_ids", "gen_id", "chain_idx", "error"])
        for n in nodes:
            writer.writerow([
                n["id"], n["score"], n["created_at"], ";".join(n.get("parent_ids", [])), 
                n["gen_id"], n["chain_idx"], n.get('metrics', {}).get('error', "")
            ])

    # Generate quantile plot
    scores = np.array([n["score"] for n in nodes])

    # Compute quantiles at 1 - 2^{-k} for k = 1, 2, 3, ...
    # k=1 -> 0.5, k=2 -> 0.75, k=3 -> 0.875, ...
    max_k = min(15, max(1, int(np.log2(len(scores))) + 2))  # Reasonable range
    ks = np.arange(1, max_k + 1)
    quantiles = 1 - 2.0 ** (-ks)
    quantile_values = np.quantile(scores, quantiles)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, quantile_values, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(f"Score at quantile (1 - 2⁻ᵏ)", fontsize=12)
    ax.set_title(f"Score Distribution (n={len(scores)}, Best={best_score:.6f})", fontsize=14)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    # Annotate y values on each point
    for k, y in zip(ks, quantile_values):
        ax.annotate(f"{y:.6f}", (k, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"quantile_plot_{completed_evaluations:06d}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Generate score trend analysis plot
    try:
        from simpletes.utils.plot_scores import plot_score_trend
        plot_score_trend(csv_path)
    except Exception as e:
        import sys
        print(f"Warning: Failed to generate score trend plot: {e}", file=sys.stderr)


