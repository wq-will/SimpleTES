#!/usr/bin/env python3
"""
Extract run statistics from SimpleTES instance directories.

Usage:
    python scripts/extract_run_stats.py <input_folder> <output_path.csv>

Example:
    python scripts/extract_run_stats.py runs/20260327/ahc/ahc058 results.csv
"""

import argparse
import csv
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def find_latest_db_state(instance_dir: str) -> Optional[str]:
    """Find the latest db_state_* directory in an instance directory."""
    db_states = glob(os.path.join(instance_dir, "db_state_*"))
    if not db_states:
        return None
    # Sort by name (timestamp) and return the latest
    return sorted(db_states)[-1]


def load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file safely, return None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not load {path}: {e}", file=sys.stderr)
        return None


def load_chain_best_scores(db_state_dir: str) -> tuple:
    """Load per-chain best scores from scores_*.csv file.

    Returns (scores_array, has_chain_idx, real_nodes) where:
    - scores_array: array of best scores, one per chain (num_chains values), or None
    - has_chain_idx: True if chain_idx column exists in CSV
    - real_nodes: total number of valid rows in scores CSV
    """
    score_files = glob(os.path.join(db_state_dir, "scores_*.csv"))
    if not score_files:
        return None, False, 0

    # Use the latest scores file
    score_file = sorted(score_files)[-1]

    try:
        # Collect scores per chain
        chain_scores: Dict[int, List[float]] = {}
        has_chain_idx = False
        real_nodes = 0

        with open(score_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Check if chain_idx column exists
            if reader.fieldnames and ("chain_idx" in reader.fieldnames or "chain" in reader.fieldnames):
                has_chain_idx = True

            for row in reader:
                try:
                    # Try different possible column names
                    chain_idx = int(row.get("chain_idx", row.get("chain", -1)))
                    score = float(row.get("score", row.get("Score", "nan")))

                    if chain_idx >= 0 and not np.isnan(score) and score > -1e9:
                        real_nodes += 1
                        if chain_idx not in chain_scores:
                            chain_scores[chain_idx] = []
                        chain_scores[chain_idx].append(score)
                except (ValueError, TypeError):
                    continue

        if not chain_scores:
            return None, has_chain_idx, real_nodes

        # Get best score per chain
        chain_bests = [max(scores) for scores in chain_scores.values()]
        return np.array(chain_bests) if chain_bests else None, has_chain_idx, real_nodes

    except Exception as e:
        print(f"  Warning: Could not load scores from {score_file}: {e}", file=sys.stderr)
        return None, False, 0


def extract_task_subtask_from_config(config: Dict[str, Any]) -> tuple:
    """Extract task and subtask from config's init_program or evaluator_path.

    Example: datasets/ahc/ahc039/init_program.py -> task=ahc, subtask=ahc039
    Example: datasets/sums_diffs/sums_diffs/evaluator.py -> task=sums_diffs, subtask=sums_diffs
    """
    # Try init_program first, then evaluator_path
    for key in ["init_program", "evaluator_path"]:
        path = config.get(key, "")
        if not path:
            continue
        parts = Path(path).parts
        # Look for "datasets" and extract the next two parts
        for i, part in enumerate(parts):
            if part == "datasets" and i + 2 < len(parts):
                return parts[i + 1], parts[i + 2]
        # Fallback: just take parts before the filename
        if len(parts) >= 3:
            return parts[-3], parts[-2]

    return "unknown", "unknown"


def check_include_construction(instance_dir: str, config: Dict[str, Any]) -> str:
    """Check if include_construction was enabled.

    First check config, then check if shared_constructions directory has files.
    """
    # Check config first
    if config.get("include_construction"):
        return "Yes"

    # Check if shared_constructions directory has any files
    shared_dir = os.path.join(instance_dir, "shared_constructions")
    if os.path.isdir(shared_dir):
        files = os.listdir(shared_dir)
        if files:
            return "Yes"

    return "No"


def check_run_completed(instance_dir: str) -> bool:
    """Check if run completed by looking for 'Final Results' in last 100 lines of run.log."""
    log_path = os.path.join(instance_dir, "run.log")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            last_100 = lines[-100:] if len(lines) >= 100 else lines
            return any("Final Results" in line for line in last_100)
    except FileNotFoundError:
        return False


def extract_instance_stats(instance_dir: str, contributor: str = "") -> Optional[Dict[str, Any]]:
    """Extract statistics from a single instance directory."""
    db_state_dir = find_latest_db_state(instance_dir)
    if not db_state_dir:
        print(f"  Skip: no db_state found", file=sys.stderr)
        return None

    # Load config, metadata, and policy
    config = load_json_safe(os.path.join(db_state_dir, "config.json"))
    metadata = load_json_safe(os.path.join(db_state_dir, "metadata.json"))
    policy = load_json_safe(os.path.join(db_state_dir, "policy.json"))
    policy_state = policy.get("state", {}) if policy else {}

    if not config:
        print(f"  Skip: no config.json", file=sys.stderr)
        return None

    # Filter: max_generations must be >= 25600
    max_gen = config.get("max_generations", 0)
    if max_gen < 25600:
        print(f"  Skip: max_generations={max_gen} < 25600", file=sys.stderr)
        return None

    # Filter: run must be completed (Final Results in log)
    if not check_run_completed(instance_dir):
        print(f"  Skip: run not completed (no 'Final Results' in log)", file=sys.stderr)
        return None

    # Filter: must be high reasoning mode (vllm_token_forcing backend)
    llm_backend = config.get("llm_backend", "")
    if llm_backend != "vllm_token_forcing":
        print(f"  Skip: llm_backend={llm_backend} (not vllm_token_forcing)", file=sys.stderr)
        return None

    # Load per-chain best scores
    scores, has_chain_idx, real_nodes = load_chain_best_scores(db_state_dir)

    # Filter: must have chain_idx column in scores CSV
    if not has_chain_idx:
        print(f"  Skip: no chain_idx column in scores CSV", file=sys.stderr)
        return None

    # Filter: real_nodes must be > 75% of max_generations
    if real_nodes < max_gen * 0.75:
        print(f"  Skip: real_nodes={real_nodes} < 75% of max_generations={max_gen}", file=sys.stderr)
        return None

    # Extract task/subtask from config
    task, subtask = extract_task_subtask_from_config(config)

    # Calculate score statistics from per-chain best scores
    # scores array contains the best score from each chain (num_chains values)
    if scores is not None and len(scores) > 0:
        best_score = float(np.max(scores))
        q75_score = float(np.percentile(scores, 75))
        q50_score = float(np.percentile(scores, 50))
        q25_score = float(np.percentile(scores, 25))
        mean_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        num_chains_actual = len(scores)  # Number of chains with scores
    else:
        best_score = metadata.get("best_score", "") if metadata else ""
        q75_score = ""
        q50_score = ""
        q25_score = ""
        mean_score = ""
        min_score = ""
        num_chains_actual = ""

    # Extract run date from path (look for YYYYMMDD pattern)
    run_date = ""
    for part in Path(instance_dir).parts:
        if len(part) == 8 and part.isdigit():
            run_date = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            break

    # Commit time is today's date
    from datetime import date
    commit_time = date.today().isoformat()

    # Determine from_init_or_best
    from_init_or_best = "init"  # Default assumption

    # Build result
    result = {
        "Run Dates": run_date,
        "Commit Time": commit_time,
        "Contributor": contributor,
        "Task": task,
        "Subtask": subtask,
        "From init / best": from_init_or_best,
        "Temperature": config.get("temperature", ""),
        "Model": config.get("model", ""),
        "#inspir": config.get("num_inspirations", ""),
        "include-construction?": check_include_construction(instance_dir, config),
        "Policy": config.get("inspiration_policy", ""),
        "N_chains": policy_state.get("num_chains", config.get("num_chains", "")),
        "K_candidates": policy_state.get("k", config.get("k_candidates", "")),
        "Total Nodes": config.get("max_generations", ""),
        "Real Nodes": real_nodes,
        "Best_Score": best_score,
        "Q75_Score": q75_score,
        "Q50_Score": q50_score,
        "Q25_Score": q25_score,
        "Mean_Score": mean_score,
        "Min_Score": min_score,
        "Notes": "",
        "Abs Path": os.path.abspath(instance_dir),
    }

    return result


def find_instance_dirs(input_folder: str) -> List[str]:
    """Find all instance-* directories recursively."""
    instance_dirs = []

    for root, dirs, files in os.walk(input_folder):
        for d in dirs:
            if d.startswith("instance-"):
                instance_dirs.append(os.path.join(root, d))

    return sorted(instance_dirs)


def main():
    parser = argparse.ArgumentParser(description="Extract run statistics from SimpleTES instances")
    parser.add_argument("input_folder", help="Input folder containing instance-* directories")
    parser.add_argument("output_path", help="Output CSV file path")
    parser.add_argument("--contributor", "-c", default="", help="Contributor name")
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all instance directories
    instance_dirs = find_instance_dirs(args.input_folder)
    print(f"Found {len(instance_dirs)} instance directories")

    if not instance_dirs:
        print("No instance directories found", file=sys.stderr)
        sys.exit(1)

    # Extract stats from each instance
    results = []
    for i, instance_dir in enumerate(instance_dirs):
        print(f"Processing [{i+1}/{len(instance_dirs)}]: {instance_dir}")
        stats = extract_instance_stats(instance_dir, contributor=args.contributor)
        if stats:
            results.append(stats)

    if not results:
        print("No valid results extracted", file=sys.stderr)
        sys.exit(1)

    # Write CSV (append if file exists)
    fieldnames = [
        "Run Dates", "Commit Time", "Contributor", "Task", "Subtask",
        "From init / best", "Temperature", "Model", "#inspir", "include-construction?",
        "Policy", "N_chains", "K_candidates", "Total Nodes", "Real Nodes", "Best_Score",
        "Q75_Score", "Q50_Score", "Q25_Score", "Mean_Score", "Min_Score",
        "Notes", "Abs Path"
    ]

    file_exists = os.path.isfile(args.output_path)
    mode = "a" if file_exists else "w"

    with open(args.output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    action = "Appended" if file_exists else "Written"
    print(f"\n{action} {len(results)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
