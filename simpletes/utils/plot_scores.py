"""Plot score trends from checkpoint CSV."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import argparse
import json
from pathlib import Path


def _gap_floor_for_checkpoint(csv_dir: Path) -> float:
    config_path = csv_dir / 'config.json'
    path_tokens = []

    if config_path.exists():
        try:
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
            for key in ('evaluator_path', 'instruction_path', 'init_program'):
                value = config.get(key)
                if value:
                    path_tokens.extend(Path(str(value)).parts)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    path_tokens.extend(csv_dir.parts)
    return 1e-8 if any(token.startswith('circle_packing') for token in path_tokens) else 1e-6


def _log_gap_from_reference(reference_score: float, scores: np.ndarray, gap_floor: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=float)
    gap = np.maximum(np.abs(reference_score - scores), gap_floor)
    return np.log10(gap)


def plot_score_trend(csv_path: str, n_nodes: int = None, verbose: bool = False):
    # Load data
    df = pd.read_csv(csv_path)
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    df = df.sort_values('created_at').reset_index(drop=True)

    if n_nodes is not None:
        df = df.head(n_nodes)

    # Add node index (generation order)
    df['node_idx'] = range(len(df))

    # Compute running best
    df['running_best'] = df['score'].cummax()

    # Build node_scores dict early (used in multiple places)
    node_scores = dict(zip(df['id'], df['score']))

    # Load policy.json for chain info
    csv_dir = Path(csv_path).parent
    gap_floor = _gap_floor_for_checkpoint(csv_dir)
    policy_path = csv_dir / 'policy.json'
    chains_from_policy = None
    node_to_chain = {}  # node_id -> chain_id mapping
    if policy_path.exists():
        with open(policy_path) as f:
            policy_data = json.load(f)
            if 'state' in policy_data:
                if 'chain_history' not in policy_data['state']:
                    raise ValueError("policy.json missing required field: state.chain_history")
                chains_from_policy = policy_data['state']['chain_history']
                chain_source = 'chain_history'

            if chains_from_policy is not None:
                if verbose:
                    print(f"Loaded {len(chains_from_policy)} chains from policy.json state.{chain_source}")
                # Build node_id -> chain_id mapping
                for chain_id, chain_nodes in chains_from_policy.items():
                    for nid in chain_nodes:
                        node_to_chain[nid] = int(chain_id)

    # Add chain_id column from policy.json
    df['chain_id'] = df['id'].map(node_to_chain)

    # Parse parent_ids string into list
    def parse_parent_ids(parent_ids):
        """Parse parent_ids string, return (parent_list, parents_key)"""
        if pd.isna(parent_ids):
            return [], None
        parts = str(parent_ids).split(';')
        # Filter out any legacy markers (for backward compatibility)
        parents = [p for p in parts if p and not p.startswith('__')]
        parents_key = tuple(sorted(parents)) if parents else None
        return parents, parents_key

    # Apply parsing
    parsed = df['parent_ids'].apply(parse_parent_ids)
    df['parent_ids_list'] = parsed.apply(lambda x: x[0])
    df['parents_key'] = parsed.apply(lambda x: x[1])

    # Use gen_id column directly for batch identification (if present)
    # Fall back to parents_key grouping if gen_id not available
    if 'gen_id' in df.columns:
        df['batch_id'] = df['gen_id']
    else:
        df['batch_id'] = df['parents_key'].apply(lambda x: hash(x) if x else None)

    # Find best node in each batch (same gen_id)
    local_best_mask = pd.Series(False, index=df.index)
    for batch_id, group in df.groupby('batch_id'):
        if pd.isna(batch_id):
            continue
        best_idx = group['score'].idxmax()
        local_best_mask[best_idx] = True

    # Root node is also considered "in chain"
    root_mask = df['parent_ids_list'].apply(len) == 0
    local_best_mask = local_best_mask | root_mask

    df['is_local_best'] = local_best_mask
    if verbose:
        print(f"Debug: is_local_best count = {local_best_mask.sum()}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get chain best scores and find best chain
    chain_best_scores = []
    chain_data = {}  # chain_id -> (best_score, node_list)

    if chains_from_policy is not None:
        for chain_id, chain_nodes in chains_from_policy.items():
            chain_scores = [node_scores[nid] for nid in chain_nodes if nid in node_scores]
            if chain_scores:
                best = max(chain_scores)
                chain_best_scores.append(best)
                chain_data[chain_id] = (best, chain_nodes)
    else:
        if verbose:
            print("Warning: policy.json not found, chain analysis will be limited")

    chain_best = np.array(chain_best_scores)
    global_best = chain_best.max() if len(chain_best) > 0 else df['score'].max()
    best_chain_id = max(chain_data.keys(), key=lambda k: chain_data[k][0]) if chain_data else None

    # Plot 1 (top-left): Combined histogram - chain bests + all scores (two y-axes)
    ax1 = axes[0, 0]

    # Compute gaps for chain bests
    log_gaps_chain = _log_gap_from_reference(global_best, chain_best, gap_floor)

    # Compute gaps for all scores
    valid_scores = df.loc[df['score'] > 0, 'score'].values
    log_gaps_all = _log_gap_from_reference(global_best, valid_scores, gap_floor)

    if log_gaps_chain.size > 0 or log_gaps_all.size > 0:
        all_log_gaps = np.concatenate([log_gaps_chain, log_gaps_all])
        log_gap_min = all_log_gaps.min()
        log_gap_max = all_log_gaps.max()
        if np.isclose(log_gap_min, log_gap_max):
            bin_edges = np.linspace(log_gap_min - 0.5, log_gap_max + 0.5, 31)
        else:
            bin_edges = np.linspace(log_gap_min, log_gap_max, 31)

        ax1.hist(log_gaps_chain, bins=bin_edges, alpha=0.7, color='steelblue', edgecolor='black', label='Chain Best')
        ax1.set_xlabel('log₁₀(|Best - Score|)')
        ax1.set_ylabel('Chain Count', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax1_all = ax1.twinx()
        ax1_all.hist(log_gaps_all, bins=bin_edges, alpha=0.5, color='orange', edgecolor='black', label='All Scores')
        ax1_all.set_ylabel('All Scores Count', color='orange')
        ax1_all.tick_params(axis='y', labelcolor='orange')
        ax1_all.set_yscale('log')

        ax1.legend(handles=[Patch(color='steelblue', alpha=0.7, label='Chain Best'),
                            Patch(color='orange', alpha=0.5, label='All Scores')], loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'No valid score data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xlabel('log₁₀(|Best - Score|)')
        ax1.set_ylabel('Chain Count', color='steelblue')

    ax1.set_title(f'Gap Distribution ({len(chain_best)} chains, {len(valid_scores)} scores, best={global_best:.4f})')
    ax1.grid(True, alpha=0.3)

    # Plot 2 (top-right): Per-chain running-best gap to global best
    ax2 = axes[0, 1]
    if chain_data:
        max_chain_len = 0

        ordered_chain_ids = sorted(chain_data.keys(), key=lambda chain_id: (0, int(chain_id)) if str(chain_id).isdigit() else (1, str(chain_id)))
        non_best_chain_ids = [chain_id for chain_id in ordered_chain_ids if chain_id != best_chain_id]
        color_map = plt.get_cmap('tab20', max(len(non_best_chain_ids), 1))
        chain_colors = {
            chain_id: color_map(idx)
            for idx, chain_id in enumerate(non_best_chain_ids)
        }
        if best_chain_id is not None:
            chain_colors[best_chain_id] = 'red'

        for chain_id in ordered_chain_ids:
            _, chain_nodes = chain_data[chain_id]
            chain_nodes_in_csv = [nid for nid in chain_nodes if nid in node_scores]
            if not chain_nodes_in_csv:
                continue

            chain_scores = np.array([node_scores[nid] for nid in chain_nodes_in_csv], dtype=float)
            chain_running_best = np.maximum.accumulate(chain_scores)
            log_gap = _log_gap_from_reference(global_best, chain_running_best, gap_floor)
            x = np.arange(len(chain_nodes_in_csv))
            max_chain_len = max(max_chain_len, len(chain_nodes_in_csv))

            if chain_id == best_chain_id:
                color = chain_colors[chain_id]
                line_width = 3.0
                alpha = 0.95
                legend_label = f'Best Chain #{chain_id}'
            else:
                color = chain_colors[chain_id]
                line_width = 1.5
                alpha = 0.9
                legend_label = f'Chain #{chain_id}'

            ax2.plot(x, log_gap, color=color, linewidth=line_width, alpha=alpha, label=legend_label)

        ax2.axhline(np.log10(gap_floor), color='gray', linestyle=':', linewidth=1.0,
                    alpha=0.8, label=f'Precision Floor ({gap_floor:.0e})')

        ax2.set_xlabel('Node Index in Chain')
        ax2.set_ylabel('log₁₀(|Global Best - Chain Running Best|)')
        ax2.set_title(f'Per-Chain Running-Best Gap ({len(chain_data)} chains, best={global_best:.6f})')
        if max_chain_len > 0:
            ax2.set_xlim(0, max_chain_len - 1)
        ax2.legend(loc='upper right', fontsize=7, ncol=2)
    else:
        ax2.text(0.5, 0.5, 'No chain data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel('Node Index in Chain')
        ax2.set_ylabel('log₁₀(|Global Best - Chain Running Best|)')
        ax2.set_title('Per-Chain Running-Best Gap')

    ax2.grid(True, alpha=0.3)

    # Plot 3 (bottom-left): Best chain's DAG with broken y-axis
    # Remove the original ax3 and create two stacked axes
    axes[1, 0].remove()
    gs = axes[0, 0].get_gridspec()
    ax3_top = fig.add_subplot(gs[1, 0])

    # Find the best chain
    if chain_data:
        best_chain_score, best_chain_nodes = chain_data[best_chain_id]

        # Filter to nodes that exist in the CSV
        chain_nodes_in_csv = [nid for nid in best_chain_nodes if nid in node_scores]

        # Build position and score maps
        node_pos = {nid: i for i, nid in enumerate(chain_nodes_in_csv)}
        chain_scores = [node_scores[nid] for nid in chain_nodes_in_csv]
        score_arr = np.array(chain_scores)

        # Find root score (first non-zero score)
        non_zero_scores = score_arr[score_arr > 0]
        root_score = non_zero_scores[0] if len(non_zero_scores) > 0 else 0

        # Build parent list map from df (only for nodes in chain)
        chain_node_set = set(chain_nodes_in_csv)
        parents_map = {row['id']: row['parent_ids_list']
                       for _, row in df[df['id'].isin(chain_node_set)].iterrows()}

        # Replace zeros and -inf with root_score for plotting
        fail_mask = (score_arr == 0) | (score_arr == -np.inf)
        score_arr_plot = np.where(fail_mask, root_score, score_arr)

        # Draw edges: connect each node to ALL its parents
        edge_count = 0
        for nid in chain_nodes_in_csv:
            nid_pos = node_pos[nid]
            nid_score = score_arr_plot[nid_pos]
            for parent in parents_map.get(nid, []):
                if parent in node_pos:
                    parent_score = score_arr_plot[node_pos[parent]]
                    ax3_top.annotate('', xy=(nid_pos, nid_score),
                                xytext=(node_pos[parent], parent_score),
                                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.5),
                                zorder=1)
                    edge_count += 1

        # Plot nodes: failures (0 or -inf) in red, others in steelblue
        positions = np.array(range(len(chain_nodes_in_csv)))
        ax3_top.scatter(positions[~fail_mask], score_arr_plot[~fail_mask], s=50, c='steelblue', zorder=3)
        if fail_mask.any():
            ax3_top.scatter(positions[fail_mask], score_arr_plot[fail_mask], s=50, c='red', zorder=3, label=f'{fail_mask.sum()} failures')

        # Highlight the best node
        best_idx = np.argmax(score_arr)
        ax3_top.scatter([best_idx], [score_arr_plot[best_idx]],
                   s=100, c='gold', marker='*', zorder=4, label=f'Best: {score_arr[best_idx]:.4f}')

        ax3_top.set_xlabel('Position in Chain')
        ax3_top.set_ylabel('Score')
        ax3_top.set_title(f'Best Chain #{best_chain_id} ({len(chain_nodes_in_csv)} nodes, {edge_count} edges)')
        ax3_top.legend(loc='lower right')
        ax3_top.grid(True, alpha=0.3)

        # Debug: print missing edges
        missing = len(chain_nodes_in_csv) - 1 - edge_count
        if missing > 0 and verbose:
            print(f"Warning: {missing} edges missing in best chain (parent not in chain)")
    else:
        ax3_top.text(0.5, 0.5, 'No chain data available', ha='center', va='center', transform=ax3_top.transAxes)
        ax3_top.set_title('Best Chain DAG')

    ax3_top.grid(True, alpha=0.3)

    # Plot 4: Gap from best over Node Count / Time (chronological order)
    ax4 = axes[1, 1]
    # Compute time elapsed in hours
    start_time = df['created_at'].min()
    hours_elapsed = (df['created_at'] - start_time).dt.total_seconds() / 3600

    # Compute log10(best - running_best), clamped by task-specific floor
    final_best = df['running_best'].iloc[-1]
    log_gap = _log_gap_from_reference(final_best, df['running_best'].values, gap_floor)

    # Plot log gap vs node index
    ax4.plot(df['node_idx'], log_gap, 'b-', linewidth=1.5, label='Gap vs Nodes')
    ax4.set_xlabel('Node Count')
    ax4.set_ylabel('log₁₀(Best - Running Best)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')

    # Secondary x-axis for time
    ax4_time = ax4.twiny()
    ax4_time.plot(hours_elapsed, log_gap, 'r-', linewidth=1.5, alpha=0.7, label='Gap vs Time')
    ax4_time.set_xlabel('Hours Elapsed', color='r')
    ax4_time.tick_params(axis='x', labelcolor='r')

    # Secondary y-axis for failure (0 or -inf) ratio
    ax4_zero = ax4.twinx()
    fail_cumsum = ((df['score'] == 0) | (df['score'] == -np.inf)).cumsum().values
    fail_ratio = fail_cumsum / (df['node_idx'].values + 1)
    ax4_zero.plot(df['node_idx'], fail_ratio, 'g--', linewidth=1.5, alpha=0.7, label='Fail Ratio')
    ax4_zero.set_ylabel('Fail Ratio (0 or -inf)', color='g')
    ax4_zero.tick_params(axis='y', labelcolor='g')
    ax4_zero.set_ylim(0, 1)

    ax4.set_title(f'Gap from Best Over Time (best={final_best:.6f})')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_time.get_legend_handles_labels()
    lines3, labels3 = ax4_zero.get_legend_handles_labels()
    ax4.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='center right')

    # Summary stats
    n_total = len(df)
    # invalid is -inf or 0
    invalid = ((df['score'] == -float("inf")) | (df['score'] == 0)).sum()
    n_local_best = df['is_local_best'].sum()
    best_score = df['score'].max()

    fig.suptitle(f'Score Analysis: {n_total} nodes, {len(chain_best)} chains, {invalid} invalid ({invalid/n_total*100:.1f}%), '
                 f'{n_local_best} local-best, max={best_score:.6f}', fontsize=12)

    plt.tight_layout()

    # Save or show
    output_path = csv_path.replace('.csv', '.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved to {output_path}")
        print(f"\n=== Summary ===")
        print(f"Total nodes: {n_total}")
        print(f"Chains: {len(chain_best)}")
        print(f"Invalid scores: {invalid} ({invalid/n_total*100:.1f}%)")
        print(f"Local-best nodes: {n_local_best}")
        print(f"Best score: {best_score:.6f}")
        print(f"Time span: {hours_elapsed.max():.2f} hours")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot score trends from checkpoint CSV")
    parser.add_argument("csv_path", help="Path to scores CSV file")
    parser.add_argument("--n_nodes", default=None, type=int, help="Number of nodes to plot")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug info")
    args = parser.parse_args()

    plot_score_trend(args.csv_path, args.n_nodes, verbose=args.verbose)
