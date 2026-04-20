#!/usr/bin/env python3
"""
Unified task preparation CLI for SimpleTES.

Discovers ``data_manifest.json`` files under ``datasets/`` and runs the
declared prepare commands to download data, create venvs, etc.

Usage:
    python scripts/prepare_task.py              # prepare all tasks
    python scripts/prepare_task.py --task ahc   # prepare only ahc
    python scripts/prepare_task.py --list       # show status of all tasks
    python scripts/prepare_task.py --check      # check only, exit 1 if missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from simpletes.utils.task_prep import check_files, discover_all_manifests, run_prepare

_DATASETS_DIR = _REPO_ROOT / "datasets"


def _status_line(task_dir: Path, manifest: dict) -> str:
    missing = check_files(task_dir, manifest)
    total = len(manifest.get("required_files", []))
    present = total - len(missing)
    mark = "OK" if not missing else "MISSING"
    return f"  [{mark}] {task_dir.name:30s}  {present}/{total} files present"


def cmd_list(manifests: list[tuple[Path, dict]]) -> None:
    print("Task preparation status:")
    for task_dir, manifest in manifests:
        print(_status_line(task_dir, manifest))
        missing = check_files(task_dir, manifest)
        if missing:
            for f in missing:
                print(f"         - {f}")


def cmd_check(manifests: list[tuple[Path, dict]]) -> int:
    any_missing = False
    for task_dir, manifest in manifests:
        missing = check_files(task_dir, manifest)
        if missing:
            any_missing = True
            print(f"MISSING  {task_dir.name}:")
            for f in missing:
                print(f"  - {f}")
    if not any_missing:
        print("All tasks are ready.")
    return 1 if any_missing else 0


def cmd_prepare(manifests: list[tuple[Path, dict]]) -> int:
    all_ok = True
    for task_dir, manifest in manifests:
        missing = check_files(task_dir, manifest)
        if not missing:
            print(f"[OK] {task_dir.name}: all files present, skipping.")
            continue

        task_name = task_dir.name
        print(f"\n=== {task_name} ===")
        print(f"Missing files: {', '.join(missing)}")

        errors = run_prepare(task_dir, manifest)
        if errors:
            print(f"[WARN] {task_name}: some commands failed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)

        still_missing = check_files(task_dir, manifest)
        if still_missing:
            all_ok = False
            print(f"[WARN] {task_name}: still missing after preparation:")
            for f in still_missing:
                print(f"  - {f}")
        else:
            print(f"[OK] {task_name}: all ready.")

    return 0 if all_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare tasks for SimpleTES benchmarks (download data, create venvs, etc.)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Only prepare this task (e.g., ahc, numerical_tasks, scaling_law, open_problems_bio)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="show_list",
        help="List all tasks and their preparation status",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check task readiness only, exit 1 if any files are missing",
    )
    args = parser.parse_args()

    manifests = discover_all_manifests(_DATASETS_DIR)
    if not manifests:
        print(f"No data_manifest.json files found under {_DATASETS_DIR}")
        sys.exit(0)

    # Filter by task name if requested
    if args.task:
        manifests = [(d, m) for d, m in manifests if d.name == args.task]
        if not manifests:
            available = ", ".join(
                d.name for d, _ in discover_all_manifests(_DATASETS_DIR)
            )
            print(f"No manifest found for task '{args.task}'. Available: {available}")
            sys.exit(1)

    if args.show_list:
        cmd_list(manifests)
    elif args.check:
        sys.exit(cmd_check(manifests))
    else:
        sys.exit(cmd_prepare(manifests))


if __name__ == "__main__":
    main()
