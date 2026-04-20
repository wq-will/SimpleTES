#!/usr/bin/env python3
"""Visualize all chains' elite pool as slot timelines.

Each subplot is one chain.
Each row in a subplot is a slot.
X-axis: event sequence (per-chain chronological order)
Color: log10(global_best - score), using global max across all chains.

Usage:
    python scripts/plot_elite_history.py --data-dir path/to/db_state_XXXXXX
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def process_chain(chain_idx: int, df_full: pd.DataFrame, max_slots: int):
    """Process a single chain and return slots, reject_slot, and all scores."""
    chain_df = df_full[df_full["chain_idx"] == chain_idx].copy()
    chain_df = chain_df.sort_values("timestamp").reset_index(drop=True)
    chain_df["local_seq"] = range(len(chain_df))

    slots = {i: [] for i in range(max_slots)}
    reject_slot = []
    all_scores = []

    # Find init_program info from first replace at removed_index=0
    init_replaces = chain_df[(chain_df["action"] == "replace") & (chain_df["removed_index"] == 0)]
    if len(init_replaces) > 0:
        init_info = init_replaces.iloc[0]
        init_node_id = init_info["removed_node_id"]
        init_score = init_info["removed_node_score"]
    else:
        any_replace = chain_df[chain_df["action"] == "replace"]
        if len(any_replace) > 0:
            init_node_id = "__init__"
            init_score = any_replace.iloc[0]["removed_node_score"] if pd.notna(any_replace.iloc[0]["removed_node_score"]) else 0.0
        else:
            init_node_id = "__init__"
            init_score = 0.0

    if pd.notna(init_score):
        pool_nodes = [init_node_id]
        slots[0].append({
            "node_id": str(init_node_id)[:8] if isinstance(init_node_id, str) else "init",
            "score": init_score,
            "start_seq": 0,
            "end_seq": None
        })
        all_scores.append(init_score)
    else:
        pool_nodes = []

    for _, row in chain_df.iterrows():
        action = row["action"]
        seq_id = row["local_seq"]

        if action == "add":
            new_node_id = row["new_node_id"]
            new_score = row["new_node_score"]

            if pd.isna(new_node_id) or pd.isna(new_score):
                continue

            slot_idx = len(pool_nodes)
            if slot_idx < max_slots:
                pool_nodes.append(new_node_id)
                slots[slot_idx].append({
                    "node_id": str(new_node_id)[:8],
                    "score": new_score,
                    "start_seq": seq_id,
                    "end_seq": None
                })
                all_scores.append(new_score)

        elif action == "replace":
            new_node_id = row["new_node_id"]
            new_score = row["new_node_score"]
            removed_index = int(row["removed_index"]) if pd.notna(row["removed_index"]) else -1

            if pd.isna(new_node_id) or pd.isna(new_score):
                continue

            if 0 <= removed_index < max_slots:
                if slots[removed_index] and slots[removed_index][-1]["end_seq"] is None:
                    slots[removed_index][-1]["end_seq"] = seq_id

                slots[removed_index].append({
                    "node_id": str(new_node_id)[:8],
                    "score": new_score,
                    "start_seq": seq_id,
                    "end_seq": None
                })
                all_scores.append(new_score)

        elif action == "reject":
            new_node_id = row["new_node_id"]
            new_score = row["new_node_score"]

            if pd.isna(new_node_id) or pd.isna(new_score):
                continue

            reject_slot.append({
                "node_id": str(new_node_id)[:8],
                "score": new_score,
                "seq_id": seq_id
            })
            all_scores.append(new_score)

    # Close all open segments
    max_seq = chain_df["local_seq"].max() if len(chain_df) > 0 else 0
    for slot_idx in range(max_slots):
        if slots[slot_idx] and slots[slot_idx][-1]["end_seq"] is None:
            slots[slot_idx][-1]["end_seq"] = max_seq

    chain_best = max(all_scores) if all_scores else 0.0

    return slots, reject_slot, all_scores, max_seq, chain_best


def plot_elite_history(csv_path: str, precision_digits: int = 4) -> None:
    """Generate elite history slot timeline plot and chains.csv statistics.

    Args:
        csv_path: Path to elite_history.csv
        precision_digits: Number of decimal places for color scale and CSV (default: 4)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    output_dir = csv_path.parent

    # Get all unique chains
    all_chains = sorted(df["chain_idx"].unique())
    num_chains = len(all_chains)

    if num_chains == 0:
        return

    # Auto-detect max pool size
    max_slots = int(df["pool_size"].max())
    if max_slots == 0:
        max_slots = 30  # fallback

    # Process all chains
    all_scores_global = []
    chain_data = {}
    chain_bests = []

    for chain_idx in all_chains:
        slots, reject_slot, scores, max_seq, chain_best = process_chain(chain_idx, df, max_slots)
        chain_data[chain_idx] = {
            "slots": slots,
            "reject_slot": reject_slot,
            "scores": scores,
            "max_seq": max_seq,
            "chain_best": chain_best
        }
        all_scores_global.extend(scores)
        chain_bests.append({"chain_idx": int(chain_idx), "best": chain_best})

    # Save chains.csv: summary statistics of chain best scores
    if chain_bests:
        bests_arr = np.array([c["best"] for c in chain_bests])
        summary = {
            "num_chains": len(bests_arr),
            "max": round(float(np.max(bests_arr)), precision_digits),
            "min": round(float(np.min(bests_arr)), precision_digits),
            "mean": round(float(np.mean(bests_arr)), precision_digits),
            "median": round(float(np.median(bests_arr)), precision_digits),
            "std": round(float(np.std(bests_arr)), precision_digits),
            "q25": round(float(np.percentile(bests_arr, 25)), precision_digits),
            "q75": round(float(np.percentile(bests_arr, 75)), precision_digits),
        }
        pd.DataFrame([summary]).to_csv(output_dir / "chains.csv", index=False)

    if not all_scores_global:
        return

    # Global best score
    global_best = max(all_scores_global)

    def score_to_color(score):
        diff = global_best - score
        min_diff = 10 ** (-precision_digits)
        if diff < min_diff:
            diff = min_diff
        return np.log10(diff)

    all_log_diff = [score_to_color(s) for s in all_scores_global]
    vmin = -precision_digits
    vmax = max(all_log_diff) if all_log_diff else 0

    # Create figure
    ncols = 1
    nrows = max(1, num_chains)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    cmap = plt.cm.Blues_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Sort chains by best score (descending)
    sorted_chains = sorted(all_chains, key=lambda c: chain_data[c]["chain_best"], reverse=True)

    bar_height = 0.9
    bar_offset = bar_height / 2

    for idx, chain_idx in enumerate(sorted_chains):
        ax = axes[idx]
        data = chain_data[chain_idx]
        slots = data["slots"]
        reject_slot = data["reject_slot"]
        chain_max_seq = data["max_seq"]

        # Draw pool slot segments
        for slot_idx in range(max_slots):
            y = max_slots - slot_idx

            for seg in slots[slot_idx]:
                start = seg["start_seq"]
                end = seg["end_seq"] if seg["end_seq"] else chain_max_seq
                score = seg["score"]

                width = end - start
                if width <= 0:
                    width = 1

                log_diff = score_to_color(score)
                color = cmap(norm(log_diff))

                rect = mpatches.Rectangle(
                    (start, y - bar_offset),
                    width,
                    bar_height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3
                )
                ax.add_patch(rect)

                if width > 2:
                    text_color = "white" if norm(log_diff) < 0.5 else "black"
                    ax.text(
                        start + width / 2, y,
                        f"{score:.4f}",
                        ha="center", va="center",
                        fontsize=5, color=text_color
                    )

        # Draw reject slot
        reject_width = 1
        for rej in reject_slot:
            seq_id = rej["seq_id"]
            score = rej["score"]

            log_diff = score_to_color(score)
            color = cmap(norm(log_diff))

            rect = mpatches.Rectangle(
                (seq_id - reject_width / 2, -bar_offset),
                reject_width,
                bar_height,
                facecolor=color,
                edgecolor="red",
                linewidth=0.5
            )
            ax.add_patch(rect)

        # Formatting
        ax.set_xlim(-1, chain_max_seq + 5)
        ax.set_ylim(-1, max_slots + 1)
        ax.set_xlabel("Event Sequence", fontsize=8)
        ax.set_ylabel("Slot", fontsize=8)
        chain_best_score = data["chain_best"]
        ax.set_title(f"Chain {int(chain_idx)} (best: {chain_best_score:.6f})", fontsize=10)

        ax.set_yticks([0] + list(range(1, max_slots + 1, 5)))
        ax.set_yticklabels(["rej"] + [str(i) for i in range(0, max_slots, 5)], fontsize=6)

        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.2, axis="x")

    # Hide unused subplots
    for idx in range(num_chains, len(axes)):
        axes[idx].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="log10(best - score)")

    fig.suptitle(
        f"Elite Pool Slot Timeline - All Chains\nGlobal best: {global_best:.6f} | Pool size: {max_slots}",
        fontsize=14, y=0.995
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.98])

    output_path = output_dir / "elite_history_slots.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize elite pool slot timelines")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to db_state directory containing elite_history.csv",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places for color scale (default: 4)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "elite_history.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return 1

    plot_elite_history(str(csv_path), precision_digits=args.precision)
    print(f"Generated: {data_dir / 'elite_history_slots.png'}")
    print(f"Generated: {data_dir / 'chains.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
