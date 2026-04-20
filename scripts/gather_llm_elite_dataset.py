#!/usr/bin/env python
import argparse
import gzip
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


LLM_OUTPUT_PATTERN = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>(.*)",
    re.DOTALL,
)


def find_policy_files(root_paths: List[str]) -> List[Path]:
    policy_files: List[Path] = []
    for root in root_paths:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            continue
        if root_path.is_file():
            if root_path.name == "policy.json":
                policy_files.append(root_path)
            continue
        for dirpath, _, filenames in os.walk(root_path):
            if "policy.json" in filenames:
                policy_files.append(Path(dirpath) / "policy.json")
    return policy_files


SUBTASK_LEVEL_FROM_LEAF = 5


def extract_subtask_name(policy_path: Path, subtask_level_from_leaf: int = SUBTASK_LEVEL_FROM_LEAF) -> str:
    """
    Extract subtask name from the absolute path of policy.json.

    We treat the directory containing policy.json as level 0,
    its parent as level 1, and so on. For paths like:

        checkpoints/.../circle_packing/circle_packing_26/openai_gpt-4o/.../policy.json

    a subtask_level_from_leaf of 5 gives "circle_packing_26".
    """
    parts = policy_path.resolve().parts
    idx = len(parts) - (subtask_level_from_leaf + 1)
    if idx < 0 or idx >= len(parts):
        return "unknown"
    return parts[idx]


def extract_run_date_and_instance(policy_path: Path) -> Tuple[str, str]:
    """
    Try to infer run date (YYYY-MM-DD) and instance id ('instance-xxxx')
    from the absolute path of policy.json.
    """
    parts = list(policy_path.resolve().parts)
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    run_date = "unknown_date"
    instance = "unknown_instance"

    for i, p in enumerate(parts):
        if date_pattern.match(p):
            run_date = p
        if p.startswith("instance-"):
            instance = p

    return run_date, instance

def parse_llm_output(
    raw_output: str,
) -> Tuple[str, str, bool]:
    """
    Split llm_output into (thinking, content).

    Returns (thinking, content, ok).
    If the expected pattern does not match, ok is False.

    Also handles responses with multiple thinking blocks by removing
    duplicate patterns and checking if any tokens remain that indicate
    malformed output.
    """
    # Clean up duplicate thinking blocks (thinking twice scenario)
    cleaned_output = re.sub(
        r'<\|end\|><\|start\|>assistant<\|channel\|>analysis<\|message\|>',
        '',
        raw_output
    )

    m = LLM_OUTPUT_PATTERN.fullmatch(cleaned_output)
    if not m:
        return "", "", False
    thinking = m.group(1).strip()
    content = m.group(2).strip()

    # Check if any special tokens remain in thinking or content
    # (indicates malformed output that should be dropped)
    special_tokens = ['<|channel|>', '<|message|>', '<|end|>', '<|start|>', '<|content|>']
    for token in special_tokens:
        if token in thinking or token in content:
            return "", "", False

    return thinking, content, True


def build_messages(llm_input: str, llm_output: str) -> Tuple[List[Dict[str, Any]], bool]:
    thinking, content, ok = parse_llm_output(llm_output)
    if not ok:
        return [], False
    messages = [
        {"role": "user", "content": llm_input},
        {"role": "assistant", "content": content, "thinking": thinking},
    ]
    return messages, True


class Filter:
    """Base filter for elite_history entries."""

    def __call__(self, entry: Dict[str, Any]) -> bool:
        raise NotImplementedError


class NoopFilter(Filter):
    def __call__(self, entry: Dict[str, Any]) -> bool:  # type: ignore[override]
        return True


class ImprovedFilter(Filter):
    """
    Keep entries whose new_node_id is in an allowlist derived from nodes.json.

    We mark a node's parents as allowed if that node's score is >= 0.95 * max
    score of all nodes generated before it.
    """

    def __init__(self, policy_path: Path):
        self._allowed_ids = set()
        db_dir = policy_path.parent
        nodes_gz_path = db_dir / "nodes.json.gz"
        nodes_path = db_dir / "nodes.json"

        try:
            if nodes_gz_path.exists():
                with gzip.open(nodes_gz_path, "rt", encoding="utf-8") as f:
                    nodes = json.load(f)
            elif nodes_path.exists():
                with nodes_path.open("r", encoding="utf-8") as f:
                    nodes = json.load(f)
            else:
                nodes = []
        except Exception:
            nodes = []

        if not isinstance(nodes, list):
            return

        # Ensure nodes are processed in creation order
        nodes_sorted = sorted(
            (nd for nd in nodes if isinstance(nd, dict)),
            key=lambda nd: (
                nd.get("created_at") or "",
                nd.get("gen_id") or 0,
            ),
        )

        running_max: float | None = None
        for nd in nodes_sorted:
            if not isinstance(nd, dict):
                continue
            score = nd.get("score")
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue

            if running_max is None or score_f > running_max:
                running_max = score_f

            if running_max is None:
                continue

            if score_f >= 0.95 * running_max:
                parents = nd.get("parent_ids") or []
                if isinstance(parents, list):
                    for pid in parents:
                        if isinstance(pid, str):
                            self._allowed_ids.add(pid)

    def __call__(self, entry: Dict[str, Any]) -> bool:  # type: ignore[override]
        new_node_id = entry.get("new_node_id")
        if not isinstance(new_node_id, str):
            return False
        return new_node_id in self._allowed_ids


class ChainFilter(Filter):
    """
    Keep entries whose new_node_id is in the *top chains*.

    We rank chains by their best score, which is stored in
    policy.json as negative scores under state["chain_sorted_neg_scores"].
    Scores are guaranteed to be >= 0 or -inf, so neg-scores are <= 0 or +inf.

    Instead of taking top k% of chains by count, we use a score-based threshold:
    find the score at the top k% position, then keep all chains with score >= that threshold.
    This helps with tasks like HM29, KN11, KN13 where many chains reach the best.
    """

    def __init__(self, state: Dict[str, Any]):
        # Hardcoded but easy to modify: change this one line.
        self.top_chain_fraction = 0.25

        self.ranked_chain_count = 0
        self.kept_chain_count = 0
        self._kept_chain_node_ids: set[str] = set()

        chains = state.get("chains")
        neg_scores_by_chain = state.get("chain_sorted_neg_scores")
        if not isinstance(chains, dict) or not isinstance(neg_scores_by_chain, dict):
            return

        best_neg_by_chain: Dict[str, float] = {}
        for chain_id, neg_scores in neg_scores_by_chain.items():
            if not isinstance(chain_id, str):
                continue
            if not isinstance(neg_scores, list) or not neg_scores:
                continue
            best_neg: float | None = None
            for v in neg_scores:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                # score can be -inf => neg-score becomes +inf; ignore for best.
                if not math.isfinite(f):
                    continue
                if best_neg is None or f < best_neg:
                    best_neg = f
            if best_neg is None:
                continue
            best_neg_by_chain[chain_id] = best_neg

        if not best_neg_by_chain:
            return

        # Convert negative scores to actual scores (negate them) and sort ascending
        # (sorted by neg-score ascending = sorted by score descending)
        sorted_chains = sorted(best_neg_by_chain.items(), key=lambda kv: kv[1])
        self.ranked_chain_count = len(sorted_chains)

        # Find the score at the top k% position
        cutoff_idx = max(0, min(
            int(math.ceil(self.ranked_chain_count * self.top_chain_fraction)) - 1,
            self.ranked_chain_count - 1
        ))
        _, cutoff_neg_score = sorted_chains[cutoff_idx]
        cutoff_score = -cutoff_neg_score

        # Keep all chains with score >= cutoff_score
        kept_chain_ids = {
            cid for cid, neg_score in best_neg_by_chain.items()
            if -neg_score >= cutoff_score
        }

        self.kept_chain_count = len(kept_chain_ids)

        for chain_id in kept_chain_ids:
            node_ids = chains.get(chain_id)
            if not isinstance(node_ids, list):
                continue
            for nid in node_ids:
                if isinstance(nid, str):
                    self._kept_chain_node_ids.add(nid)

    def __call__(self, entry: Dict[str, Any]) -> bool:  # type: ignore[override]
        new_node_id = entry.get("new_node_id")
        if not isinstance(new_node_id, str):
            return False
        return new_node_id in self._kept_chain_node_ids


class ChainBeforeBestByNodesFilter(Filter):
    """
    Like ChainFilter(top 25% chains by score threshold), but truncate each kept chain
    using node creation time from nodes.json(.gz) instead of relying on the chain list
    order in policy.json.

    For each kept chain:
    - compute the chain's best score via state["chain_sorted_neg_scores"] (stored as -score)
    - find all nodes in this chain that achieve that best score
    - choose the *earliest* such node by (created_at, gen_id) from nodes.json
    - keep only nodes with (created_at, gen_id) <= that cutoff (drop later nodes)

    Uses score-based threshold: find the score at the top k% position, then keep all
    chains with score >= that threshold. This helps with tasks like HM29, KN11, KN13
    where many chains reach the best.
    """

    def __init__(self, policy_path: Path, state: Dict[str, Any]):
        # Hardcoded but easy to modify: change this one line.
        self.top_chain_fraction = 0.25

        self.ranked_chain_count = 0
        self.kept_chain_count = 0
        self._kept_chain_node_ids: set[str] = set()

        chains = state.get("chains")
        neg_scores_by_chain = state.get("chain_sorted_neg_scores")
        if not isinstance(chains, dict) or not isinstance(neg_scores_by_chain, dict):
            return

        # Load node time + score index from nodes.json(.gz)
        node_time_index: Dict[str, tuple[str, int]] = {}
        node_score_index: Dict[str, float] = {}
        db_dir = policy_path.parent
        nodes_gz_path = db_dir / "nodes.json.gz"
        nodes_path = db_dir / "nodes.json"
        try:
            if nodes_gz_path.exists():
                with gzip.open(nodes_gz_path, "rt", encoding="utf-8") as f:
                    nodes = json.load(f)
            elif nodes_path.exists():
                with nodes_path.open("r", encoding="utf-8") as f:
                    nodes = json.load(f)
            else:
                nodes = []
        except Exception:
            nodes = []

        if isinstance(nodes, list):
            for nd in nodes:
                if not isinstance(nd, dict):
                    continue
                node_id = nd.get("id")
                if not isinstance(node_id, str) or not node_id:
                    continue
                created_at = nd.get("created_at")
                created_at_s = created_at if isinstance(created_at, str) else ""
                gen_id = nd.get("gen_id")
                try:
                    gen_id_i = int(gen_id)
                except (TypeError, ValueError):
                    gen_id_i = 0
                node_time_index[node_id] = (created_at_s, gen_id_i)
                score = nd.get("score")
                try:
                    score_f = float(score) if score is not None else math.nan
                except (TypeError, ValueError):
                    score_f = math.nan
                node_score_index[node_id] = score_f

        missing_time_sentinel = ("~", 2**31 - 1)  # sorts after normal ISO timestamps

        # Only "operations" that can become training data are those that appear in elite_history.
        elite_history = state.get("elite_history") or []
        elite_new_node_ids: set[str] = set()
        if isinstance(elite_history, list):
            for e in elite_history:
                if not isinstance(e, dict):
                    continue
                nid = e.get("new_node_id")
                if isinstance(nid, str):
                    elite_new_node_ids.add(nid)

        # Compute best neg-score per chain (min finite neg-score)
        best_neg_by_chain: Dict[str, float] = {}
        for chain_id, neg_scores in neg_scores_by_chain.items():
            if not isinstance(chain_id, str):
                continue
            if not isinstance(neg_scores, list) or not neg_scores:
                continue
            best_neg: float | None = None
            for v in neg_scores:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(f):
                    continue
                if best_neg is None or f < best_neg:
                    best_neg = f
            if best_neg is None:
                continue
            best_neg_by_chain[chain_id] = best_neg

        if not best_neg_by_chain:
            return

        # Use score-based threshold: find the score at top k% position, keep all chains with score >= that
        sorted_chains = sorted(best_neg_by_chain.items(), key=lambda kv: kv[1])
        self.ranked_chain_count = len(sorted_chains)

        # Find the score at the top k% position
        cutoff_idx = max(0, min(
            int(math.ceil(self.ranked_chain_count * self.top_chain_fraction)) - 1,
            self.ranked_chain_count - 1
        ))
        _, cutoff_neg_score = sorted_chains[cutoff_idx]
        cutoff_score = -cutoff_neg_score

        # Keep all chains with score >= cutoff_score
        kept_chain_ids = {
            cid for cid, neg_score in best_neg_by_chain.items()
            if -neg_score >= cutoff_score
        }

        self.kept_chain_count = len(kept_chain_ids)

        # For each kept chain, truncate after the first time (by created_at/gen_id)
        # that the chain reaches its best *action node* score.
        for chain_id in kept_chain_ids:
            node_ids = chains.get(chain_id)
            if not isinstance(node_ids, list):
                continue

            # Restrict to nodes that appear in elite_history (training actions) and
            # have finite score + valid time info.
            action_nodes: List[str] = []
            for nid in node_ids:
                if not isinstance(nid, str):
                    continue
                if nid not in elite_new_node_ids:
                    continue
                score_f = node_score_index.get(nid, math.nan)
                if not math.isfinite(score_f):
                    continue
                if nid not in node_time_index:
                    continue
                action_nodes.append(nid)

            if not action_nodes:
                continue

            # Best score among action nodes for this chain.
            best_score_action = max(node_score_index[nid] for nid in action_nodes)

            # Choose cutoff as the earliest node (created_at, then gen_id)
            # among nodes that achieve the best score.
            best_action_nodes = [
                nid
                for nid in action_nodes
                if abs(node_score_index[nid] - best_score_action) <= 1e-6
            ]
            cutoff_node = min(
                best_action_nodes,
                key=lambda nid: node_time_index.get(nid, missing_time_sentinel),
            )
            cutoff_key = node_time_index.get(cutoff_node, missing_time_sentinel)

            # Keep only action nodes not later than the cutoff.
            for nid in action_nodes:
                if node_time_index.get(nid, missing_time_sentinel) <= cutoff_key:
                    self._kept_chain_node_ids.add(nid)

    def __call__(self, entry: Dict[str, Any]) -> bool:  # type: ignore[override]
        new_node_id = entry.get("new_node_id")
        if not isinstance(new_node_id, str):
            return False
        return new_node_id in self._kept_chain_node_ids


def to_parquet(
    records: List[Dict[str, Any]],
    train_path: Path,
    val_path: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    if not records:
        return

    rng = random.Random(seed)
    rng.shuffle(records)

    n_total = len(records)
    n_train = int(n_total * train_ratio)
    if n_train == 0 and n_total > 1:
        n_train = n_total - 1

    train_records = records[:n_train] if n_train > 0 else []
    val_records = records[n_train:] if n_train < n_total else []

    def build_table(recs: List[Dict[str, Any]]) -> pa.Table:
        # messages: list of dicts like
        # [{"role": "user", "content": ...},
        #  {"role": "assistant", "content": ..., "thinking": ...}]
        messages_column = []
        subtask_column = []
        source_column = []

        for rec in recs:
            msgs = rec["messages"]
            normalized_msgs = []
            for m in msgs:
                msg = {
                    "role": m["role"],
                    "content": m["content"],
                }
                if "thinking" in m and m["thinking"] is not None:
                    msg["thinking"] = m["thinking"]
                normalized_msgs.append(msg)
            messages_column.append(normalized_msgs)
            subtask_column.append(rec["subtask"])
            source_column.append(str(rec["source_path"]))

        messages_array = pa.array(messages_column)
        subtask_array = pa.array(subtask_column, type=pa.string())
        source_array = pa.array(source_column, type=pa.string())

        return pa.Table.from_arrays(
            [messages_array, subtask_array, source_array],
            names=["messages", "subtask", "source_path"],
        )

    if train_records:
        train_table = build_table(train_records)
        pq.write_table(train_table, train_path)
    if val_records:
        val_table = build_table(val_records)
        pq.write_table(val_table, val_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather LLM elite history data into SFT-ready parquet files.",
    )
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="One or more root paths to search recursively for policy.json files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: first data_path).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data to put into train.parquet (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split shuffling.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "improved", "chain", "chain_before_best"],
        help="Optional filter for elite_history entries (default: none).",
    )

    args = parser.parse_args()

    data_paths = args.data_paths
    if not data_paths:
        raise SystemExit("No data paths provided.")

    output_dir = Path(args.output_dir or data_paths[0]).expanduser().resolve()
    dataset_dir = output_dir / f"llm_elite_dataset_{args.filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    report_path = dataset_dir / "report.txt"
    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "test.parquet"

    policy_files = [p.resolve() for p in find_policy_files(data_paths)]

    # Log discovered policy files by subtask for easier tracking.
    print("Discovered policy.json files:")
    for p in sorted(policy_files):
        print(f"  - {p}")

    total_elite_entries = 0
    total_valid_examples = 0
    total_missing_llm_fields = 0
    total_parse_failures = 0
    total_policy_errors = 0
    total_filtered_entries = 0

    subtask_counter: Counter = Counter()
    subtask_valid_counter: Counter = Counter()
    subtask_filtered_counter: Counter = Counter()
    # (subtask -> (run_date, instance) -> counts)
    per_subtask_instance_counter: Dict[str, Counter] = defaultdict(Counter)
    per_subtask_instance_valid_counter: Dict[str, Counter] = defaultdict(Counter)
    per_subtask_instance_filtered_counter: Dict[str, Counter] = defaultdict(Counter)
    errors_per_policy: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"missing_llm_fields": 0, "parse_failures": 0}
    )

    all_records: List[Dict[str, Any]] = []

    # Process each policy.json in parallel (I/O bound workload).
    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(
                process_single_policy,
                policy_path,
                args.filter,
            ): policy_path
            for policy_path in policy_files
        }

        for fut in as_completed(future_to_path):
            policy_path = future_to_path[fut]
            (
                records,
                stats,
            ) = fut.result()

            print(f"[gather_llm_elite_train_data] Finished {policy_path}")

            all_records.extend(records)

            total_elite_entries += stats["total_elite_entries"]
            total_valid_examples += stats["total_valid_examples"]
            total_missing_llm_fields += stats["total_missing_llm_fields"]
            total_parse_failures += stats["total_parse_failures"]
            total_policy_errors += stats["total_policy_errors"]
            total_filtered_entries += stats["total_filtered_entries"]

            subtask_counter.update(stats["subtask_counter"])
            subtask_valid_counter.update(stats["subtask_valid_counter"])
            subtask_filtered_counter.update(stats["subtask_filtered_counter"])
            for k, v in stats["per_subtask_instance_counter"].items():
                per_subtask_instance_counter[k].update(v)
            for k, v in stats["per_subtask_instance_valid_counter"].items():
                per_subtask_instance_valid_counter[k].update(v)
            for k, v in stats["per_subtask_instance_filtered_counter"].items():
                per_subtask_instance_filtered_counter[k].update(v)
            for policy_key, err_counts in stats["errors_per_policy"].items():
                errors_per_policy[policy_key]["missing_llm_fields"] += err_counts[
                    "missing_llm_fields"
                ]
                errors_per_policy[policy_key]["parse_failures"] += err_counts[
                    "parse_failures"
                ]

    to_parquet(
        all_records,
        train_path=train_path,
        val_path=val_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    report_text = build_report(
        policy_files=policy_files,
        train_path=train_path,
        val_path=val_path,
        report_path=report_path,
        filter_name=args.filter,
        total_elite_entries=total_elite_entries,
        total_valid_examples=total_valid_examples,
        total_missing_llm_fields=total_missing_llm_fields,
        total_parse_failures=total_parse_failures,
        total_policy_errors=total_policy_errors,
        total_filtered_entries=total_filtered_entries,
        subtask_counter=subtask_counter,
        subtask_valid_counter=subtask_valid_counter,
        subtask_filtered_counter=subtask_filtered_counter,
        per_subtask_instance_counter=per_subtask_instance_counter,
        per_subtask_instance_valid_counter=per_subtask_instance_valid_counter,
        per_subtask_instance_filtered_counter=per_subtask_instance_filtered_counter,
        errors_per_policy=errors_per_policy,
    )

    # Also print to stdout so it appears in the terminal.
    print(report_text, end="")


def process_single_policy(
    policy_path: Path,
    filter_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    policy_key = str(policy_path)

    total_elite_entries = 0
    total_valid_examples = 0
    total_missing_llm_fields = 0
    total_parse_failures = 0
    total_policy_errors = 0
    total_filtered_entries = 0

    subtask_counter: Counter = Counter()
    subtask_valid_counter: Counter = Counter()
    subtask_filtered_counter: Counter = Counter()
    per_subtask_instance_counter: Dict[str, Counter] = defaultdict(Counter)
    per_subtask_instance_valid_counter: Dict[str, Counter] = defaultdict(Counter)
    per_subtask_instance_filtered_counter: Dict[str, Counter] = defaultdict(Counter)
    errors_per_policy: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"missing_llm_fields": 0, "parse_failures": 0}
    )

    records: List[Dict[str, Any]] = []

    try:
        with policy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        total_policy_errors += 1
        return records, {
            "total_elite_entries": total_elite_entries,
            "total_valid_examples": total_valid_examples,
            "total_missing_llm_fields": total_missing_llm_fields,
            "total_parse_failures": total_parse_failures,
            "total_policy_errors": total_policy_errors,
            "total_filtered_entries": total_filtered_entries,
            "subtask_counter": subtask_counter,
            "subtask_valid_counter": subtask_valid_counter,
            "subtask_filtered_counter": subtask_filtered_counter,
            "per_subtask_instance_counter": per_subtask_instance_counter,
            "per_subtask_instance_valid_counter": per_subtask_instance_valid_counter,
            "per_subtask_instance_filtered_counter": per_subtask_instance_filtered_counter,
            "errors_per_policy": errors_per_policy,
        }

    state = data.get("state", {})
    elite_history = state.get("elite_history") or []
    if not isinstance(elite_history, list):
        return records, {
            "total_elite_entries": total_elite_entries,
            "total_valid_examples": total_valid_examples,
            "total_missing_llm_fields": total_missing_llm_fields,
            "total_parse_failures": total_parse_failures,
            "total_policy_errors": total_policy_errors,
            "total_filtered_entries": total_filtered_entries,
            "subtask_counter": subtask_counter,
            "subtask_valid_counter": subtask_valid_counter,
            "subtask_filtered_counter": subtask_filtered_counter,
            "per_subtask_instance_counter": per_subtask_instance_counter,
            "per_subtask_instance_valid_counter": per_subtask_instance_valid_counter,
            "per_subtask_instance_filtered_counter": per_subtask_instance_filtered_counter,
            "errors_per_policy": errors_per_policy,
        }

    if filter_name == "improved":
        policy_filter: Filter = ImprovedFilter(policy_path)
    elif filter_name == "chain":
        if isinstance(state, dict):
            policy_filter = ChainFilter(state)
        else:
            policy_filter = ChainFilter({})
    elif filter_name == "chain_before_best":
        if isinstance(state, dict):
            policy_filter = ChainBeforeBestByNodesFilter(policy_path, state)
        else:
            policy_filter = ChainBeforeBestByNodesFilter(policy_path, {})
    else:
        policy_filter = NoopFilter()

    # Log chain filter configuration for traceability.
    if isinstance(policy_filter, (ChainFilter, ChainBeforeBestByNodesFilter)):
        print(
            f"[{filter_name}] top_chain_fraction={policy_filter.top_chain_fraction} "
            f"ranked_chains={policy_filter.ranked_chain_count} "
            f"kept_chains={policy_filter.kept_chain_count} "
            f"policy={policy_path}"
        )

    subtask = extract_subtask_name(policy_path.parent)
    run_date, instance_id = extract_run_date_and_instance(policy_path)

    n_entries = len(elite_history)
    subtask_counter[subtask] += n_entries
    per_subtask_instance_counter[subtask][(run_date, instance_id)] += n_entries
    total_elite_entries += n_entries

    for entry in elite_history:
        if not policy_filter(entry):
            total_filtered_entries += 1
            subtask_filtered_counter[subtask] += 1
            per_subtask_instance_filtered_counter[subtask][
                (run_date, instance_id)
            ] += 1
            continue
        llm_input = entry.get("llm_input")
        llm_output = entry.get("llm_output")
        if not isinstance(llm_input, str) or not isinstance(llm_output, str):
            total_missing_llm_fields += 1
            errors_per_policy[policy_key]["missing_llm_fields"] += 1
            continue

        messages, ok = build_messages(llm_input, llm_output)
        if not ok:
            total_parse_failures += 1
            errors_per_policy[policy_key]["parse_failures"] += 1
            continue

        rec = {
            "messages": messages,
            "subtask": subtask,
            "source_path": str(policy_path),
        }
        records.append(rec)
        subtask_valid_counter[subtask] += 1
        per_subtask_instance_valid_counter[subtask][(run_date, instance_id)] += 1
        total_valid_examples += 1

    stats: Dict[str, Any] = {
        "total_elite_entries": total_elite_entries,
        "total_valid_examples": total_valid_examples,
        "total_missing_llm_fields": total_missing_llm_fields,
        "total_parse_failures": total_parse_failures,
        "total_policy_errors": total_policy_errors,
        "total_filtered_entries": total_filtered_entries,
        "subtask_counter": subtask_counter,
        "subtask_valid_counter": subtask_valid_counter,
        "subtask_filtered_counter": subtask_filtered_counter,
        "per_subtask_instance_counter": per_subtask_instance_counter,
        "per_subtask_instance_valid_counter": per_subtask_instance_valid_counter,
        "per_subtask_instance_filtered_counter": per_subtask_instance_filtered_counter,
        "errors_per_policy": errors_per_policy,
    }

    return records, stats


def build_report(
    *,
    policy_files: List[Path],
    train_path: Path,
    val_path: Path,
    report_path: Path,
    filter_name: str,
    total_elite_entries: int,
    total_valid_examples: int,
    total_missing_llm_fields: int,
    total_parse_failures: int,
    total_policy_errors: int,
    total_filtered_entries: int,
    subtask_counter: Counter,
    subtask_valid_counter: Counter,
    subtask_filtered_counter: Counter,
    per_subtask_instance_counter: Dict[str, Counter],
    per_subtask_instance_valid_counter: Dict[str, Counter],
    per_subtask_instance_filtered_counter: Dict[str, Counter],
    errors_per_policy: Dict[str, Dict[str, int]],
) -> str:
    # Build report content once, then write to file and return text.
    lines: List[str] = []
    lines.append("LLM Elite Train Data Gathering Report")
    lines.append("=====================================")
    lines.append("")
    lines.append(f"Total policy.json files found: {len(policy_files)}")
    lines.append(f"Filter: {filter_name}")
    lines.append(f"Total elite_history entries: {total_elite_entries}")
    lines.append(f"Total entries filtered by '{filter_name}': {total_filtered_entries}")
    kept_after_filter = total_elite_entries - total_filtered_entries
    lines.append(f"Total entries after filter (before parsing): {kept_after_filter}")
    lines.append(f"Total valid SFT examples: {total_valid_examples}")
    lines.append(
        f"Total entries with missing llm_input/llm_output: {total_missing_llm_fields}"
    )
    lines.append(
        f"Total entries with llm_output pattern mismatch: {total_parse_failures}"
    )
    lines.append(f"Total policy.json files failed to load: {total_policy_errors}")
    lines.append("")
    lines.append("Per-subtask statistics (elite_history entries):")

    total_entries_for_ratio = sum(subtask_counter.values()) or 1
    for subtask, count in sorted(subtask_counter.items(), key=lambda x: x[0]):
        ratio = count / total_entries_for_ratio * 100.0
        valid = subtask_valid_counter.get(subtask, 0)
        filtered = subtask_filtered_counter.get(subtask, 0)
        kept = count - filtered
        filtered_pct = (filtered / count * 100.0) if count else 0.0
        lines.append(
            f"  - {subtask}: {count} entries "
            f"({ratio:.2f}% of all), {valid} valid examples, "
            f"{filtered} filtered ({filtered_pct:.2f}% of this subtask, kept={kept})"
        )

        # Instance-level breakdown within this subtask.
        inst_counter = per_subtask_instance_counter.get(subtask, {})
        inst_valid_counter = per_subtask_instance_valid_counter.get(subtask, {})
        inst_filtered_counter = per_subtask_instance_filtered_counter.get(subtask, {})
        for (run_date, instance_id), inst_count in sorted(
            inst_counter.items(), key=lambda x: (x[0][0], x[0][1])
        ):
            inst_valid = inst_valid_counter.get((run_date, instance_id), 0)
            inst_filtered = inst_filtered_counter.get((run_date, instance_id), 0)
            inst_kept = inst_count - inst_filtered
            inst_ratio = inst_count / total_entries_for_ratio * 100.0
            inst_filtered_pct = (
                inst_filtered / inst_count * 100.0 if inst_count else 0.0
            )
            lines.append(
                f"    - {instance_id} (from {run_date}): {inst_count} entries "
                f"({inst_ratio:.2f}% of all), {inst_valid} valid examples, "
                f"{inst_filtered} filtered ({inst_filtered_pct:.2f}% of this instance, kept={inst_kept})"
            )

    lines.append("")
    lines.append("Per-policy error summary:")
    for policy_key, err_counts in sorted(errors_per_policy.items()):
        missing = err_counts["missing_llm_fields"]
        parse_fail = err_counts["parse_failures"]
        lines.append(
            f"  - {policy_key} :: "
            f"missing_llm_fields={missing}, parse_failures={parse_fail}"
        )

    lines.append("")
    lines.append("Output files:")
    lines.append(f"  - Train parquet: {train_path}")
    lines.append(f"  - Val parquet:   {val_path}")

    report_text = "\n".join(lines) + "\n"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


if __name__ == "__main__":
    main()