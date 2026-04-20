#!/usr/bin/env python3
"""
Analyze and visualize elite_history.csv from SimpleTES runs.

Usage:
    python scripts/plot_elite_history.py --data-dir /path/to/db_state_xxx

Output: elite_history.png saved to the same directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_elite_history(data_dir: Path) -> pd.DataFrame:
    """Load elite_history.csv from the specified directory."""
    csv_path = data_dir / "elite_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"elite_history.csv not found in {data_dir}")
    return pd.read_csv(csv_path)


def plot_elite_summary(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Create 4-panel summary figure."""
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1 (top-left): Replace counts AND delta score per chain
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    replace_df = df[df["action"] == "replace"].copy()
    chains = sorted(df["chain_idx"].unique())

    # Replace counts (bars)
    chain_counts = replace_df.groupby("chain_idx").size().reindex(chains, fill_value=0)
    bars = ax1.bar(chains, chain_counts.values, color="steelblue", alpha=0.6, label="Replace Count")
    ax1.set_xlabel("Chain Index")
    ax1.set_ylabel("Replace Count", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Delta score (line with markers)
    if not replace_df.empty:
        replace_df["delta_score"] = replace_df["new_node_score"] - replace_df["removed_node_score"]
        delta_mean = replace_df.groupby("chain_idx")["delta_score"].mean().reindex(chains, fill_value=0)
        ax1_twin.plot(chains, delta_mean.values, "o-", color="darkorange", label="Avg Delta Score")
        ax1_twin.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1_twin.set_ylabel("Avg Delta Score", color="darkorange")
    ax1_twin.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title("Replacements & Delta Score per Chain")
    ax1.set_xticks(chains)

    # Panel 2 (top-right): Pool max score by chain over time
    # X-axis: node index per chain (count of add/replace events)
    ax2 = axes[0, 1]
    for chain_idx in sorted(df["chain_idx"].unique()):
        chain_df = df[df["chain_idx"] == chain_idx].sort_values("gen_id")
        # Filter to only add/replace events and create node index
        pool_change_df = chain_df[chain_df["action"].isin(["add", "replace"])].copy()
        pool_change_df["node_idx"] = range(1, len(pool_change_df) + 1)
        ax2.plot(
            pool_change_df["node_idx"],
            pool_change_df["pool_max_score"],
            label=f"Chain {chain_idx}",
            alpha=0.8,
        )
    ax2.set_xlabel("Node Index (per Chain)")
    ax2.set_ylabel("Pool Max Score")
    ax2.set_title("Pool Max Score by Chain")
    ax2.legend(loc="lower right", fontsize=8)

    # Panel 3 (bottom-left): Node-wise score changes (replace: vertical line, reject: gray X)
    ax3 = axes[1, 0]

    # Plot replacements as vertical lines (old -> new)
    replace_df = df[df["action"] == "replace"].copy()
    if not replace_df.empty:
        for _, row in replace_df.iterrows():
            gen_id = row["gen_id"]
            old_score = row["removed_node_score"]
            new_score = row["new_node_score"]
            color = f"C{int(row['chain_idx']) % 10}"
            ax3.plot([gen_id, gen_id], [old_score, new_score], color=color, alpha=0.6, linewidth=1.5)
            # Arrow head at new score
            ax3.scatter([gen_id], [new_score], color=color, s=15, zorder=5)

    # Plot rejects as gray X
    reject_df = df[df["action"] == "reject"].copy()
    if not reject_df.empty:
        ax3.scatter(
            reject_df["gen_id"],
            reject_df["new_node_score"],
            marker="x",
            color="gray",
            alpha=0.4,
            s=20,
            label="Rejected",
        )

    ax3.set_xlabel("Generation ID")
    ax3.set_ylabel("Score")
    ax3.set_title("Score Changes (lines: replace, X: reject)")
    if not reject_df.empty:
        ax3.legend(loc="lower right", fontsize=8)

    # Panel 4 (bottom-right): Pool mean and std trend by chain
    # Reconstruct elite pool scores from history to compute actual std
    # X-axis: node index within chain (count of add/replace events, not rejects)
    ax4 = axes[1, 1]

    for chain_idx in sorted(df["chain_idx"].unique()):
        chain_df = df[df["chain_idx"] == chain_idx].sort_values("gen_id")

        # Reconstruct pool scores over time
        pool_scores: list[float] = []
        node_indices = []
        stds = []
        node_idx = 0

        for _, row in chain_df.iterrows():
            action = row["action"]
            new_score = row["new_node_score"]
            removed_score = row["removed_node_score"]

            if action == "add":
                pool_scores.append(new_score)
                node_idx += 1
            elif action == "replace":
                # Remove the old score and add new score
                if removed_score in pool_scores:
                    pool_scores.remove(removed_score)
                elif pool_scores:
                    # Find closest score if exact match not found (float precision)
                    closest_idx = min(range(len(pool_scores)), key=lambda i: abs(pool_scores[i] - removed_score))
                    pool_scores.pop(closest_idx)
                pool_scores.append(new_score)
                node_idx += 1
            else:
                # reject: pool unchanged, don't increment node_idx
                continue

            if pool_scores and len(pool_scores) > 1:
                node_indices.append(node_idx)
                stds.append(np.std(pool_scores))

        if node_indices:
            node_indices = np.array(node_indices)
            stds = np.array(stds)

            ax4.plot(node_indices, stds, label=f"Chain {chain_idx}", alpha=0.8)

    ax4.set_xlabel("Node Index (per Chain)")
    ax4.set_ylabel("Pool STD")
    ax4.set_yscale("log")
    ax4.set_title("Pool STD of Scores by Chain")
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze elite_history.csv from SimpleTES runs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to db_state directory containing elite_history.csv",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 1

    try:
        df = load_elite_history(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded {len(df)} records from {data_dir / 'elite_history.csv'}")
    print(f"Actions: {df['action'].value_counts().to_dict()}")
    print(f"Chains: {sorted(df['chain_idx'].unique())}")

    output_path = data_dir / "elite_history.png"
    plot_elite_summary(df, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
