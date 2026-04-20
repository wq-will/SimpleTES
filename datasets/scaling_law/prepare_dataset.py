"""Pre-fetch sldbench subsets from HuggingFace into the HF cache.

The evaluators in each scaling_law subtask already fall back to
``datasets.load_dataset("pkuHaowei/sldbench", name=<subtask>, split=<split>)``
when a local ``dataset/`` directory is absent. This script forces the
download up-front so runs start faster and work offline afterwards, and
writes a sentinel file so ``scripts/prepare_task.py`` can tell the fetch
has completed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import datasets

HUB_REPO_ID = "pkuHaowei/sldbench"
SUBTASKS = [
    "data_constrained_scaling_law",
    "domain_mixture_scaling_law",
    "easy_question_scaling_law",
    "lr_bsz_scaling_law",
    "moe_scaling_law",
    "parallel_scaling_law",
    "sft_scaling_law",
    "vocab_scaling_law",
]

SENTINEL = Path(__file__).resolve().parent / ".sldbench_cached"


def main() -> int:
    failures: list[str] = []
    for name in SUBTASKS:
        print(f"  Fetching {HUB_REPO_ID}::{name}")
        try:
            datasets.load_dataset(HUB_REPO_ID, name=name, split="train")
            datasets.load_dataset(HUB_REPO_ID, name=name, split="test")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"    FAILED: {exc}", file=sys.stderr)

    if failures:
        print("\nSome subtasks failed to download:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    SENTINEL.write_text(f"cached from {HUB_REPO_ID}\n")
    print(f"All sldbench subtasks cached. Sentinel: {SENTINEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
