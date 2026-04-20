#!/usr/bin/env python3
"""Manage a shared registry of SimpleTES checkpoints via symlinks."""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence


DEFAULT_SHARED_ROOT = os.environ.get("SIMPLETES_SHARED_ROOT", "./shared_registry")
CHECKPOINT_PREFIXES = ("db_state_", "ckpt_")
REGISTRY_DIRNAME = "registry"
ENTRIES_DIRNAME = "entries"
BATCHES_DIRNAME = "batches"
LOCK_FILENAME = ".lock"


@dataclass(frozen=True)
class CheckpointIdentity:
    task: str | None
    subtask: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown"


def is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith(CHECKPOINT_PREFIXES)


def is_instance_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("instance-")


def list_instance_checkpoints(instance_dir: Path) -> list[Path]:
    return sorted(
        [child for child in instance_dir.iterdir() if is_checkpoint_dir(child)],
        key=lambda child: child.name,
    )


def expand_inputs(raw_inputs: Sequence[str], select: str) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()

    for raw_input in raw_inputs:
        candidate = Path(raw_input).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw_input}")

        candidate = candidate.resolve(strict=True)
        checkpoint_dirs: list[Path]

        if is_checkpoint_dir(candidate):
            checkpoint_dirs = [candidate]
        elif is_instance_dir(candidate):
            checkpoint_dirs = list_instance_checkpoints(candidate)
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No db_state_* checkpoints found under instance dir: {candidate}")
            checkpoint_dirs = [checkpoint_dirs[-1]] if select == "latest" else checkpoint_dirs
        else:
            raise ValueError(
                "Each input must be a db_state_*/ckpt_* directory or an instance-* directory: "
                f"{candidate}"
            )

        for checkpoint_dir in checkpoint_dirs:
            realpath = os.path.realpath(checkpoint_dir)
            if realpath in seen:
                continue
            seen.add(realpath)
            resolved.append(checkpoint_dir)

    return resolved


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_checkpoint_files(checkpoint_dir: Path) -> tuple[Path, Path, Path]:
    metadata_path = checkpoint_dir / "metadata.json"
    config_path = checkpoint_dir / "config.json"
    policy_path = checkpoint_dir / "policy.json"
    nodes_path = checkpoint_dir / "nodes.json"
    nodes_gzip_path = checkpoint_dir / "nodes.json.gz"

    missing = [path.name for path in (metadata_path, config_path, policy_path) if not path.exists()]
    if not nodes_path.exists() and not nodes_gzip_path.exists():
        missing.append("nodes.json(.gz)")
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir} is missing required files: {', '.join(missing)}"
        )
    return metadata_path, config_path, policy_path


def infer_identity_from_config(config: dict[str, Any]) -> CheckpointIdentity | None:
    candidates: list[tuple[str | None, str]] = []

    for key in ("init_program", "evaluator_path", "instruction_path"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue

        path = Path(value)
        parts = [part for part in path.parts if part not in {"", "."}]
        if "datasets" in parts:
            datasets_idx = parts.index("datasets")
            if datasets_idx + 2 < len(parts):
                task = parts[datasets_idx + 1]
                subtask = parts[datasets_idx + 2]
                if subtask:
                    candidates.append((task, subtask))
                    continue

        if path.parent.name:
            task = path.parent.parent.name or None
            subtask = path.parent.name
            candidates.append((task, subtask))

    if not candidates:
        return None

    (task, subtask), _count = Counter(candidates).most_common(1)[0]
    return CheckpointIdentity(task=task, subtask=subtask)


def discover_artifacts(checkpoint_dir: Path) -> dict[str, str]:
    artifact_map: dict[str, str] = {}
    fixed_files = {
        "best_program": "best_program.py",
        "metadata": "metadata.json",
        "config": "config.json",
        "policy": "policy.json",
        "reflection": "reflection.json",
        "failure": "failure.json",
        "training_stats": "training_stats.csv",
        "nodes": "nodes.json",
        "nodes_gzip": "nodes.json.gz",
    }
    globbed_files = {
        "scores_csv": "scores_*.csv",
        "scores_plot": "scores_*.png",
        "quantile_plot": "quantile_plot_*.png",
    }

    for key, filename in fixed_files.items():
        file_path = checkpoint_dir / filename
        if file_path.exists():
            artifact_map[key] = filename

    for key, pattern in globbed_files.items():
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            artifact_map[key] = matches[-1].name

    return artifact_map


def relative_href_from_registry(shared_root: Path, target_path: Path) -> str:
    registry_dir = shared_root / REGISTRY_DIRNAME
    return os.path.relpath(target_path, registry_dir).replace(os.sep, "/")


def make_entry_id(source_realpath: str) -> str:
    digest = hashlib.sha256(source_realpath.encode("utf-8")).hexdigest()
    return digest[:16]


def make_batch_id(
    contributor_name: str,
    committed_at: str,
    source_realpaths: Sequence[str],
) -> str:
    digest = hashlib.sha256(
        "\n".join([contributor_name, committed_at, *source_realpaths]).encode("utf-8")
    ).hexdigest()
    timestamp = committed_at.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"batch_{timestamp}_{digest[:10]}"


def infer_git_revision(source_path: Path) -> tuple[str | None, str | None]:
    commands = (
        ["git", "-C", str(source_path), "rev-parse", "--show-toplevel"],
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
    )
    try:
        repo_proc = subprocess.run(
            commands[0],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if repo_proc.returncode != 0:
            return None, None
        repo_root = repo_proc.stdout.strip() or None
        head_proc = subprocess.run(
            commands[1],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        revision = head_proc.stdout.strip() if head_proc.returncode == 0 else None
        return repo_root, revision or None
    except OSError:
        return None, None


def entry_manifest_path(shared_root: Path, entry_id: str) -> Path:
    return shared_root / REGISTRY_DIRNAME / ENTRIES_DIRNAME / f"{entry_id}.json"


def batch_manifest_path(shared_root: Path, batch_id: str) -> Path:
    return shared_root / REGISTRY_DIRNAME / BATCHES_DIRNAME / f"{batch_id}.json"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link_path.with_name(f".{link_path.name}.tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(str(target), str(tmp_link))
    os.replace(tmp_link, link_path)


@contextlib.contextmanager
def registry_lock(shared_root: Path) -> Iterator[None]:
    registry_dir = shared_root / REGISTRY_DIRNAME
    registry_dir.mkdir(parents=True, exist_ok=True)
    lock_path = registry_dir / LOCK_FILENAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_entry_manifest(
    checkpoint_dir: Path,
    *,
    contributor_name: str,
    contributor_slug: str,
    notes: str,
    job_ids: Sequence[str],
    tags: Sequence[str],
    batch_id: str,
    batch_index: int,
    committed_at: str,
    shared_root: Path,
    subtask_override: str | None,
) -> dict[str, Any]:
    metadata_path, config_path, policy_path = require_checkpoint_files(checkpoint_dir)
    metadata = read_json(metadata_path)
    config = read_json(config_path)
    policy = read_json(policy_path)
    identity = infer_identity_from_config(config)

    if subtask_override:
        subtask = subtask_override
        task = identity.task if identity else None
    elif identity:
        subtask = identity.subtask
        task = identity.task
    else:
        raise ValueError(
            f"Unable to infer subtask for checkpoint {checkpoint_dir}. "
            "Pass --subtask to override."
        )

    source_realpath = os.path.realpath(checkpoint_dir)
    source_path = Path(source_realpath)
    instance_path = source_path.parent
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        if instance_path.name.startswith("instance-"):
            instance_id = instance_path.name[len("instance-") :]
        else:
            instance_id = instance_path.name

    entry_id = make_entry_id(source_realpath)
    link_name = f"{source_path.name}__{instance_id}__{contributor_slug}"
    link_relpath = Path(subtask) / link_name
    link_path = shared_root / link_relpath
    repo_root, git_revision = infer_git_revision(source_path)

    model = config.get("model")
    if not isinstance(model, str):
        model = None
    policy_name = policy.get("name")
    if not isinstance(policy_name, str):
        policy_name = config.get("inspiration_policy") if isinstance(config.get("inspiration_policy"), str) else None

    best_score = metadata.get("best_score")
    if isinstance(best_score, (int, float)):
        best_score_value = float(best_score)
    else:
        best_score_value = None

    completed_evaluations = metadata.get("completed_evaluations")
    completed_evaluations_value = int(completed_evaluations) if isinstance(completed_evaluations, int) else None

    artifacts = discover_artifacts(source_path)
    run_log_path = instance_path / "run.log"

    return {
        "entry_id": entry_id,
        "batch_id": batch_id,
        "batch_index": batch_index,
        "committed_at": committed_at,
        "contributor": {
            "name": contributor_name,
            "slug": contributor_slug,
        },
        "notes": notes,
        "job_ids": list(job_ids),
        "tags": list(tags),
        "task": task,
        "subtask": subtask,
        "source": {
            "checkpoint_name": source_path.name,
            "checkpoint_path": str(source_path),
            "checkpoint_realpath": source_realpath,
            "instance_id": instance_id,
            "instance_path": str(instance_path),
            "run_log_path": str(run_log_path) if run_log_path.exists() else None,
            "repo_root": repo_root,
            "git_revision": git_revision,
        },
        "shared": {
            "root": str(shared_root),
            "link_name": link_name,
            "link_relpath": link_relpath.as_posix(),
            "link_path": str(link_path),
        },
        "checkpoint": {
            "best_score": best_score_value,
            "best_node_id": metadata.get("best_node_id"),
            "completed_evaluations": completed_evaluations_value,
            "generation_attempts": metadata.get("generation_attempts"),
            "generation_failures": metadata.get("generation_failures"),
            "generation_cancellations": metadata.get("generation_cancellations"),
            "evaluation_failures": metadata.get("evaluation_failures"),
            "model": model,
            "policy": policy_name,
            "init_program": config.get("init_program"),
            "evaluator_path": config.get("evaluator_path"),
            "instruction_path": config.get("instruction_path"),
            "output_path": config.get("output_path"),
            "artifacts": artifacts,
        },
        "committer": {
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
        },
    }


def render_entry_for_dashboard(shared_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    link_relpath = entry["shared"]["link_relpath"]
    link_path = shared_root / Path(link_relpath)
    symlink_present = link_path.is_symlink()
    status = "active" if symlink_present and link_path.exists() else "broken"

    artifacts = entry.get("checkpoint", {}).get("artifacts", {})
    artifact_links: dict[str, str] = {}
    for key, filename in artifacts.items():
        candidate = link_path / filename
        if symlink_present and candidate.exists():
            artifact_links[key] = relative_href_from_registry(shared_root, candidate)

    return {
        "entry_id": entry["entry_id"],
        "batch_id": entry["batch_id"],
        "committed_at": entry.get("committed_at"),
        "contributor": entry.get("contributor", {}).get("name"),
        "contributor_slug": entry.get("contributor", {}).get("slug"),
        "notes": entry.get("notes", ""),
        "job_ids": entry.get("job_ids", []),
        "tags": entry.get("tags", []),
        "task": entry.get("task"),
        "subtask": entry.get("subtask"),
        "model": entry.get("checkpoint", {}).get("model"),
        "policy": entry.get("checkpoint", {}).get("policy"),
        "best_score": entry.get("checkpoint", {}).get("best_score"),
        "completed_evaluations": entry.get("checkpoint", {}).get("completed_evaluations"),
        "generation_failures": entry.get("checkpoint", {}).get("generation_failures"),
        "evaluation_failures": entry.get("checkpoint", {}).get("evaluation_failures"),
        "source_checkpoint_path": entry.get("source", {}).get("checkpoint_path"),
        "source_instance_id": entry.get("source", {}).get("instance_id"),
        "source_run_log_path": entry.get("source", {}).get("run_log_path"),
        "git_revision": entry.get("source", {}).get("git_revision"),
        "status": status,
        "shared_link_name": entry.get("shared", {}).get("link_name"),
        "shared_link_href": relative_href_from_registry(shared_root, link_path),
        "artifact_links": artifact_links,
    }


def load_entry_manifests(shared_root: Path) -> list[dict[str, Any]]:
    entries_dir = shared_root / REGISTRY_DIRNAME / ENTRIES_DIRNAME
    if not entries_dir.exists():
        return []

    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(entries_dir.glob("*.json")):
        try:
            manifest = read_json(manifest_path)
            manifests.append(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return manifests


def build_dashboard_payload(shared_root: Path, subtask_filter: str | None = None) -> dict[str, Any]:
    rendered_entries = [
        render_entry_for_dashboard(shared_root, manifest)
        for manifest in load_entry_manifests(shared_root)
    ]

    if subtask_filter:
        rendered_entries = [entry for entry in rendered_entries if entry["subtask"] == subtask_filter]

    rendered_entries.sort(key=lambda entry: (entry.get("committed_at") or "", entry["entry_id"]), reverse=True)

    contributors = {entry["contributor"] for entry in rendered_entries if entry.get("contributor")}
    subtasks = {entry["subtask"] for entry in rendered_entries if entry.get("subtask")}
    active_entries = [entry for entry in rendered_entries if entry["status"] == "active"]
    broken_entries = [entry for entry in rendered_entries if entry["status"] == "broken"]

    subtask_stats: list[dict[str, Any]] = []
    for subtask in sorted(subtasks):
        entries = [entry for entry in rendered_entries if entry["subtask"] == subtask]
        subtask_active = [entry for entry in entries if entry["status"] == "active"]
        best_score = None
        numeric_scores = [entry["best_score"] for entry in subtask_active if isinstance(entry.get("best_score"), (int, float))]
        if numeric_scores:
            best_score = max(float(score) for score in numeric_scores)
        subtask_stats.append(
            {
                "subtask": subtask,
                "entries": len(entries),
                "active_entries": len(subtask_active),
                "broken_entries": len(entries) - len(subtask_active),
                "contributors": sorted({entry["contributor"] for entry in entries if entry.get("contributor")}),
                "best_score": best_score,
            }
        )

    return {
        "generated_at": utc_now_iso(),
        "scope": {
            "shared_root": str(shared_root),
            "subtask": subtask_filter,
        },
        "summary": {
            "entries": len(rendered_entries),
            "active_entries": len(active_entries),
            "broken_entries": len(broken_entries),
            "subtasks": len(subtasks),
            "contributors": len(contributors),
        },
        "subtask_stats": subtask_stats,
        "entries": rendered_entries,
    }


def render_dashboard_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    summary = payload["summary"]
    scope = payload["scope"]

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>SimpleTES Shared Registry</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: rgba(255, 252, 247, 0.9);
      --ink: #1e293b;
      --muted: #5b6470;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --danger: #9f1239;
      --danger-soft: #ffe1ea;
      --line: rgba(30, 41, 59, 0.14);
      --shadow: 0 24px 70px rgba(15, 23, 42, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Helvetica Neue\", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 24%),
        linear-gradient(180deg, #f9f7f2 0%, var(--bg) 100%);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    .hero {{
      display: grid;
      gap: 20px;
      grid-template-columns: minmax(0, 1.6fr) minmax(320px, 1fr);
      margin-bottom: 28px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .hero-copy {{
      padding: 28px;
    }}
    .eyebrow {{
      display: inline-flex;
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 10px;
      font-size: clamp(32px, 5vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero-copy p {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      max-width: 62ch;
    }}
    .scope-grid {{
      display: grid;
      gap: 12px;
      padding: 24px;
      align-content: start;
    }}
    .scope-item {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid var(--line);
    }}
    .scope-item strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .summary-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin-bottom: 28px;
    }}
    .summary-card {{
      padding: 18px 20px;
    }}
    .summary-card h2 {{
      margin: 0 0 6px;
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary-card .value {{
      font-size: clamp(24px, 4vw, 36px);
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .summary-card .hint {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .filters {{
      padding: 22px;
      margin-bottom: 24px;
    }}
    .filters h2, .subtasks h2, .table-wrap h2 {{
      margin: 0 0 16px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .filters-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .filter-field {{
      display: grid;
      gap: 6px;
    }}
    label {{
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      font: inherit;
    }}
    .subtasks {{
      padding: 22px;
      margin-bottom: 24px;
    }}
    .subtask-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .subtask-card {{
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      display: grid;
      gap: 10px;
    }}
    .subtask-card h3 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.02em;
    }}
    .subtask-meta {{
      color: var(--muted);
      font-size: 14px;
      display: grid;
      gap: 4px;
    }}
    .table-wrap {{
      padding: 22px;
    }}
    .table-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1024px;
    }}
    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      background: rgba(255, 255, 255, 0.55);
      position: sticky;
      top: 0;
      backdrop-filter: blur(4px);
    }}
    tbody tr:hover {{
      background: rgba(15, 118, 110, 0.05);
    }}
    .mono {{
      font-family: \"IBM Plex Mono\", \"SFMono-Regular\", monospace;
      font-size: 12px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      line-height: 1;
      margin: 0 6px 6px 0;
    }}
    .pill.active {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .pill.broken {{
      background: var(--danger-soft);
      color: var(--danger);
    }}
    .tag {{
      display: inline-block;
      margin: 0 6px 6px 0;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.08);
      font-size: 12px;
      color: var(--muted);
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .links a {{
      display: inline-flex;
      padding: 6px 8px;
      border-radius: 10px;
      background: rgba(15, 118, 110, 0.09);
      font-size: 12px;
      font-weight: 600;
    }}
    .muted {{
      color: var(--muted);
    }}
    .empty {{
      padding: 20px;
      text-align: center;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 20px 14px 48px;
      }}
      .panel {{
        border-radius: 20px;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <div class=\"panel hero-copy\">
        <span class=\"eyebrow\">SimpleTES Shared Registry</span>
        <h1>Track shared trajectories without copying checkpoint trees.</h1>
        <p>
          Contributors commit checkpoint directories as symlinks into a shared workspace.
          This dashboard indexes the committed metadata, links back to the checkpoint
          artifacts, and gives one place to filter by subtask, contributor, model, policy,
          and checkpoint health.
        </p>
      </div>
      <aside class=\"panel scope-grid\">
        <div class=\"scope-item\">
          <strong>Shared Root</strong>
          <span class=\"mono\">{scope['shared_root']}</span>
        </div>
        <div class=\"scope-item\">
          <strong>Generated</strong>
          <span>{payload['generated_at']}</span>
        </div>
        <div class=\"scope-item\">
          <strong>Scope</strong>
          <span>{scope['subtask'] or 'all subtasks'}</span>
        </div>
      </aside>
    </section>

    <section class=\"summary-grid\">
      <div class=\"panel summary-card\">
        <h2>Entries</h2>
        <div class=\"value\" id=\"summary-entries\">{summary['entries']}</div>
        <div class=\"hint\">Committed checkpoint directories in this view.</div>
      </div>
      <div class=\"panel summary-card\">
        <h2>Active</h2>
        <div class=\"value\" id=\"summary-active\">{summary['active_entries']}</div>
        <div class=\"hint\">Symlinks whose target checkpoints still exist.</div>
      </div>
      <div class=\"panel summary-card\">
        <h2>Broken</h2>
        <div class=\"value\" id=\"summary-broken\">{summary['broken_entries']}</div>
        <div class=\"hint\">Symlinks with missing targets.</div>
      </div>
      <div class=\"panel summary-card\">
        <h2>Subtasks</h2>
        <div class=\"value\" id=\"summary-subtasks\">{summary['subtasks']}</div>
        <div class=\"hint\">Unique subtasks represented in this view.</div>
      </div>
      <div class=\"panel summary-card\">
        <h2>Contributors</h2>
        <div class=\"value\" id=\"summary-contributors\">{summary['contributors']}</div>
        <div class=\"hint\">Unique contributor names across the current entries.</div>
      </div>
    </section>

    <section class=\"panel filters\">
      <h2>Filters</h2>
      <div class=\"filters-grid\">
        <div class=\"filter-field\">
          <label for=\"search\">Search</label>
          <input id=\"search\" type=\"search\" placeholder=\"notes, job ids, tags, source path\">
        </div>
        <div class=\"filter-field\">
          <label for=\"status\">Status</label>
          <select id=\"status\"><option value=\"\">All</option></select>
        </div>
        <div class=\"filter-field\">
          <label for=\"subtask\">Subtask</label>
          <select id=\"subtask\"><option value=\"\">All</option></select>
        </div>
        <div class=\"filter-field\">
          <label for=\"contributor\">Contributor</label>
          <select id=\"contributor\"><option value=\"\">All</option></select>
        </div>
        <div class=\"filter-field\">
          <label for=\"model\">Model</label>
          <select id=\"model\"><option value=\"\">All</option></select>
        </div>
        <div class=\"filter-field\">
          <label for=\"policy\">Policy</label>
          <select id=\"policy\"><option value=\"\">All</option></select>
        </div>
      </div>
    </section>

    <section class=\"panel subtasks\">
      <h2>Subtask Snapshot</h2>
      <div class=\"subtask-grid\" id=\"subtask-grid\"></div>
    </section>

    <section class=\"panel table-wrap\">
      <div class=\"table-toolbar\">
        <h2>Checkpoint Entries</h2>
        <div id=\"table-count\">Showing {summary['entries']} entries</div>
      </div>
      <div style=\"overflow:auto;\">
        <table>
          <thead>
            <tr>
              <th data-sort-key=\"committed_at\">Committed</th>
              <th data-sort-key=\"subtask\">Subtask</th>
              <th data-sort-key=\"contributor\">Contributor</th>
              <th data-sort-key=\"model\">Model</th>
              <th data-sort-key=\"policy\">Policy</th>
              <th data-sort-key=\"best_score\">Best Score</th>
              <th data-sort-key=\"completed_evaluations\">Evaluations</th>
              <th data-sort-key=\"status\">Status</th>
              <th data-sort-key=\"notes\">Notes</th>
              <th data-sort-key=\"shared_link_name\">Links</th>
            </tr>
          </thead>
          <tbody id=\"entries-body\"></tbody>
        </table>
        <div id=\"empty-state\" class=\"empty\" hidden>No entries match the current filters.</div>
      </div>
    </section>
  </div>

  <script id=\"dashboard-data\" type=\"application/json\">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("dashboard-data").textContent);
    const entries = payload.entries;
    const subtaskStats = payload.subtask_stats;
    const state = {{
      search: "",
      status: "",
      subtask: "",
      contributor: "",
      model: "",
      policy: "",
      sortKey: "committed_at",
      sortDir: "desc",
    }};

    const filters = {{
      search: document.getElementById("search"),
      status: document.getElementById("status"),
      subtask: document.getElementById("subtask"),
      contributor: document.getElementById("contributor"),
      model: document.getElementById("model"),
      policy: document.getElementById("policy"),
    }};

    const entriesBody = document.getElementById("entries-body");
    const tableCount = document.getElementById("table-count");
    const emptyState = document.getElementById("empty-state");
    const subtaskGrid = document.getElementById("subtask-grid");

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>\"]/g, (char) => {{
        return {{
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
        }}[char];
      }});
    }}

    function formatNumber(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "n/a";
      }}
      const abs = Math.abs(Number(value));
      if (abs === 0) {{
        return "0";
      }}
      if (abs >= 1000000 || abs < 0.001) {{
        return Number(value).toExponential(4);
      }}
      return Number(value).toLocaleString(undefined, {{
        maximumFractionDigits: 4,
      }});
    }}

    function fillSelect(select, values) {{
      for (const value of values) {{
        if (!value) continue;
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }}
    }}

    function renderSubtaskCards(data) {{
      if (!data.length) {{
        subtaskGrid.innerHTML = '<div class="empty" style="grid-column:1 / -1;">No subtask data available.</div>';
        return;
      }}
      subtaskGrid.innerHTML = data.map((item) => {{
        const contributors = item.contributors.length ? item.contributors.join(", ") : "n/a";
        return `
          <article class="subtask-card">
            <h3>${{escapeHtml(item.subtask)}}</h3>
            <div class="subtask-meta">
              <span>${{item.active_entries}} active / ${{item.broken_entries}} broken</span>
              <span>${{item.entries}} committed checkpoints</span>
              <span>Best active score: ${{formatNumber(item.best_score)}}</span>
              <span>Contributors: ${{escapeHtml(contributors)}}</span>
            </div>
          </article>
        `;
      }}).join("");
    }}

    function buildLinks(entry) {{
      const links = [];
      links.push(`<a href="${{escapeHtml(entry.shared_link_href)}}">checkpoint</a>`);
      for (const [key, href] of Object.entries(entry.artifact_links || {{}})) {{
        links.push(`<a href="${{escapeHtml(href)}}">${{escapeHtml(key)}}</a>`);
      }}
      return links.join("");
    }}

    function passesFilter(entry) {{
      if (state.status && entry.status !== state.status) return false;
      if (state.subtask && entry.subtask !== state.subtask) return false;
      if (state.contributor && entry.contributor !== state.contributor) return false;
      if (state.model && entry.model !== state.model) return false;
      if (state.policy && entry.policy !== state.policy) return false;

      if (!state.search) return true;
      const haystack = [
        entry.notes,
        entry.subtask,
        entry.task,
        entry.contributor,
        entry.model,
        entry.policy,
        entry.shared_link_name,
        entry.source_checkpoint_path,
        entry.source_instance_id,
        entry.source_run_log_path,
        entry.git_revision,
        ...(entry.job_ids || []),
        ...(entry.tags || []),
      ].join(" ").toLowerCase();
      return haystack.includes(state.search.toLowerCase());
    }}

    function compareEntries(left, right) {{
      const key = state.sortKey;
      const dir = state.sortDir === "asc" ? 1 : -1;
      const leftValue = left[key];
      const rightValue = right[key];

      const leftNumber = typeof leftValue === "number" ? leftValue : Number.NaN;
      const rightNumber = typeof rightValue === "number" ? rightValue : Number.NaN;
      if (!Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) {{
        return (leftNumber - rightNumber) * dir;
      }}

      return String(leftValue ?? "").localeCompare(String(rightValue ?? "")) * dir;
    }}

    function updateSummary(filtered) {{
      const active = filtered.filter((entry) => entry.status === "active");
      const broken = filtered.filter((entry) => entry.status === "broken");
      document.getElementById("summary-entries").textContent = String(filtered.length);
      document.getElementById("summary-active").textContent = String(active.length);
      document.getElementById("summary-broken").textContent = String(broken.length);
      document.getElementById("summary-subtasks").textContent = String(new Set(filtered.map((entry) => entry.subtask).filter(Boolean)).size);
      document.getElementById("summary-contributors").textContent = String(new Set(filtered.map((entry) => entry.contributor).filter(Boolean)).size);
      tableCount.textContent = `Showing ${{filtered.length}} of ${{entries.length}} entries`;
    }}

    function renderTable() {{
      const filtered = entries.filter(passesFilter).sort(compareEntries);
      updateSummary(filtered);

      if (!filtered.length) {{
        entriesBody.innerHTML = "";
        emptyState.hidden = false;
        renderSubtaskCards([]);
        return;
      }}

      emptyState.hidden = true;
      entriesBody.innerHTML = filtered.map((entry) => {{
        const tags = (entry.tags || []).map((tag) => `<span class="tag">${{escapeHtml(tag)}}</span>`).join("");
        const jobs = (entry.job_ids || []).length
          ? `<div class="muted">Jobs: ${{escapeHtml(entry.job_ids.join(", "))}}</div>`
          : "";
        const runLog = entry.source_run_log_path
          ? `<div class="muted mono">run.log: ${{escapeHtml(entry.source_run_log_path)}}</div>`
          : "";
        const revision = entry.git_revision
          ? `<div class="muted mono">git: ${{escapeHtml(entry.git_revision.slice(0, 12))}}</div>`
          : "";
        return `
          <tr>
            <td>
              <div>${{escapeHtml(entry.committed_at || "n/a")}}</div>
              <div class="muted mono">${{escapeHtml(entry.source_instance_id || "n/a")}}</div>
            </td>
            <td>
              <div><strong>${{escapeHtml(entry.subtask || "n/a")}}</strong></div>
              <div class="muted">${{escapeHtml(entry.task || "")}}</div>
            </td>
            <td>${{escapeHtml(entry.contributor || "n/a")}}</td>
            <td class="mono">${{escapeHtml(entry.model || "n/a")}}</td>
            <td>${{escapeHtml(entry.policy || "n/a")}}</td>
            <td class="mono">${{formatNumber(entry.best_score)}}</td>
            <td class="mono">${{entry.completed_evaluations ?? "n/a"}}</td>
            <td><span class="pill ${{entry.status}}">${{escapeHtml(entry.status)}}</span></td>
            <td>
              <div>${{escapeHtml(entry.notes || "")}}</div>
              ${{jobs}}
              <div>${{tags}}</div>
              ${{revision}}
              ${{runLog}}
            </td>
            <td>
              <div class="links">${{buildLinks(entry)}}</div>
              <div class="muted mono" style="margin-top:8px;">${{escapeHtml(entry.shared_link_name || "")}}</div>
            </td>
          </tr>
        `;
      }}).join("");

      const filteredSubtasks = subtaskStats.filter((item) => filtered.some((entry) => entry.subtask === item.subtask));
      renderSubtaskCards(filteredSubtasks);
    }}

    for (const [key, element] of Object.entries(filters)) {{
      element.addEventListener(key === "search" ? "input" : "change", (event) => {{
        state[key] = event.target.value;
        renderTable();
      }});
    }}

    fillSelect(filters.status, [...new Set(entries.map((entry) => entry.status))].sort());
    fillSelect(filters.subtask, [...new Set(entries.map((entry) => entry.subtask).filter(Boolean))].sort());
    fillSelect(filters.contributor, [...new Set(entries.map((entry) => entry.contributor).filter(Boolean))].sort());
    fillSelect(filters.model, [...new Set(entries.map((entry) => entry.model).filter(Boolean))].sort());
    fillSelect(filters.policy, [...new Set(entries.map((entry) => entry.policy).filter(Boolean))].sort());

    document.querySelectorAll("th[data-sort-key]").forEach((header) => {{
      header.addEventListener("click", () => {{
        const nextKey = header.dataset.sortKey;
        if (state.sortKey === nextKey) {{
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        }} else {{
          state.sortKey = nextKey;
          state.sortDir = nextKey === "committed_at" ? "desc" : "asc";
        }}
        renderTable();
      }});
    }});

    renderSubtaskCards(subtaskStats);
    renderTable();
  </script>
</body>
</html>
"""


def dashboard_output_paths(shared_root: Path, subtask: str | None) -> tuple[Path, Path]:
    if subtask:
        slug = slugify(subtask)
        return (
            shared_root / REGISTRY_DIRNAME / f"dashboard_{slug}.html",
            shared_root / REGISTRY_DIRNAME / f"dashboard_{slug}.json",
        )
    return (
        shared_root / REGISTRY_DIRNAME / "dashboard.html",
        shared_root / REGISTRY_DIRNAME / "dashboard_data.json",
    )


def write_dashboard(shared_root: Path, *, subtask: str | None = None) -> tuple[Path, Path, dict[str, Any]]:
    payload = build_dashboard_payload(shared_root, subtask_filter=subtask)
    html_path, json_path = dashboard_output_paths(shared_root, subtask)
    atomic_write_json(json_path, payload)
    atomic_write_text(html_path, render_dashboard_html(payload))
    return html_path, json_path, payload


def commit_command(args: argparse.Namespace) -> int:
    shared_root = Path(args.shared_root).expanduser()
    checkpoints = expand_inputs(args.inputs, args.select)
    committed_at = utc_now_iso()
    contributor_slug = slugify(args.name)
    batch_id = make_batch_id(
        args.name,
        committed_at,
        [os.path.realpath(checkpoint) for checkpoint in checkpoints],
    )

    new_entries: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []

    with registry_lock(shared_root):
        for batch_index, checkpoint_dir in enumerate(checkpoints):
            entry = build_entry_manifest(
                checkpoint_dir,
                contributor_name=args.name,
                contributor_slug=contributor_slug,
                notes=args.notes,
                job_ids=args.job_id,
                tags=args.tag,
                batch_id=batch_id,
                batch_index=batch_index,
                committed_at=committed_at,
                shared_root=shared_root,
                subtask_override=args.subtask,
            )
            manifest_path = entry_manifest_path(shared_root, entry["entry_id"])
            if manifest_path.exists():
                skipped_duplicates.append(entry)
                continue

            link_path = shared_root / entry["shared"]["link_relpath"]
            if link_path.exists() or link_path.is_symlink():
                raise FileExistsError(
                    f"Shared link path already exists and does not match registry state: {link_path}"
                )
            safe_symlink(Path(entry["source"]["checkpoint_path"]), link_path)
            atomic_write_json(manifest_path, entry)
            new_entries.append(entry)

        html_path = json_path = None
        if new_entries:
            batch_manifest = {
                "batch_id": batch_id,
                "committed_at": committed_at,
                "contributor": {
                    "name": args.name,
                    "slug": contributor_slug,
                },
                "notes": args.notes,
                "job_ids": list(args.job_id),
                "tags": list(args.tag),
                "shared_root": str(shared_root),
                "input_paths": [str(Path(value).expanduser()) for value in args.inputs],
                "resolved_checkpoint_paths": [entry["source"]["checkpoint_path"] for entry in new_entries],
                "entry_ids": [entry["entry_id"] for entry in new_entries],
                "skipped_duplicate_entry_ids": [entry["entry_id"] for entry in skipped_duplicates],
            }
            atomic_write_json(batch_manifest_path(shared_root, batch_id), batch_manifest)

        if not args.no_refresh:
            html_path, json_path, _payload = write_dashboard(shared_root)

    print(
        f"Committed {len(new_entries)} checkpoint(s) into {shared_root}. "
        f"Skipped {len(skipped_duplicates)} duplicate(s)."
    )
    for entry in new_entries:
        print(
            f"  - {entry['subtask']}: {entry['shared']['link_relpath']} -> "
            f"{entry['source']['checkpoint_path']}"
        )
    if skipped_duplicates:
        print("Skipped duplicates:")
        for entry in skipped_duplicates:
            print(f"  - {entry['source']['checkpoint_path']}")
    if html_path and json_path:
        print(f"Dashboard refreshed: {html_path}")
        print(f"Dashboard data: {json_path}")
    return 0


def preview_command(args: argparse.Namespace) -> int:
    shared_root = Path(args.shared_root).expanduser()
    with registry_lock(shared_root):
        html_path, json_path, payload = write_dashboard(shared_root, subtask=args.subtask)
    print(f"Wrote dashboard for {payload['summary']['entries']} entries: {html_path}")
    print(f"Wrote machine-readable data: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit SimpleTES checkpoints into a shared symlink registry and build a static preview.",
    )
    parser.add_argument(
        "--shared-root",
        default=DEFAULT_SHARED_ROOT,
        help=f"Shared registry root (default: {DEFAULT_SHARED_ROOT})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    commit_parser = subparsers.add_parser(
        "commit",
        help="Register one or more checkpoint dirs in the shared registry via symlinks.",
    )
    commit_parser.add_argument("--name", required=True, help="Contributor name.")
    commit_parser.add_argument("--notes", required=True, help="Batch note describing the committed jobs.")
    commit_parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Optional job id. Repeat to record multiple jobs in one batch.",
    )
    commit_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional tag for later filtering. Repeat to record multiple tags.",
    )
    commit_parser.add_argument(
        "--subtask",
        help="Optional subtask override if it cannot be inferred from config.json.",
    )
    commit_parser.add_argument(
        "--select",
        choices=("latest", "all"),
        default="latest",
        help="When an input is an instance-* dir, commit only the latest checkpoint or all child checkpoints.",
    )
    commit_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip rebuilding the dashboard after the commit.",
    )
    commit_parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="+",
        help="db_state_*/ckpt_* dirs or instance-* dirs to register.",
    )
    commit_parser.set_defaults(func=commit_command)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Build the static HTML dashboard and JSON index from registry manifests.",
    )
    preview_parser.add_argument(
        "--subtask",
        help="Optional subtask filter. If set, write dashboard_<subtask>.html/json instead of the global files.",
    )
    preview_parser.set_defaults(func=preview_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
