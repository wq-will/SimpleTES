"""
Checkpoint management for SimpleTES.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, TYPE_CHECKING

import pandas as pd

from simpletes.evaluator import rich_print
from simpletes.utils.log import format_log
from simpletes.node import save_score_statistics, validate_node_for_db

if TYPE_CHECKING:
    from simpletes.config import EngineConfig
    from simpletes.node import NodeDatabase
    from simpletes.policies import Selector


def _format_log(instance_id: str, icon: str = "", msg: str = "") -> str:
    return format_log(icon, msg, prefix=f"[cyan][{instance_id}][/cyan]")


class CheckpointManager:
    """Manages checkpoint creation and loading for SimpleTES."""

    def __init__(self, config: EngineConfig, instance_id: str, checkpoint_dir: str):
        self.config = config
        self.instance_id = instance_id
        self.checkpoint_dir = checkpoint_dir
        self._last_checkpoint_path: str | None = None  # Track for O(1) cleanup
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    async def snapshot(
        self,
        db: NodeDatabase,
        best_node_id: str | None,
        completed_evaluations: int,
        generation_attempts: int,
        generation_failures: int,
        generation_cancellations: int,
        evaluation_failures: int,
        best_score: float,
        selector: Selector,
    ) -> tuple[
        str | None,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """Take a consistent snapshot for checkpointing."""
        best_node = db.nodes.get(best_node_id) if best_node_id else None
        best_code = best_node.code if best_node else None
        nodes_snapshot = [n.to_dict(include_llm_io=self.config.save_llm_io) for n in db.nodes.values()]

        metadata = {
            "instance_id": self.instance_id,
            "completed_evaluations": completed_evaluations,
            "generation_attempts": generation_attempts,
            "generation_failures": generation_failures,
            "generation_cancellations": generation_cancellations,
            "evaluation_failures": evaluation_failures,
            "best_score": best_score,
            "best_node_id": best_node_id,
        }
        config = self._config_to_dict()
        policy = {"name": selector.name, "state": selector.state_dict()}
        return best_code, metadata, config, policy, nodes_snapshot

    def write_sync(
        self,
        best_code: str | None,
        metadata: dict[str, Any],
        config: dict[str, Any],
        policy: dict[str, Any],
        nodes: list[dict[str, Any]],
        failure_records: list[dict[str, Any]],
        shared_construction_state: dict[str, Any] | None = None,
    ) -> None:
        """Write checkpoint to disk (thread-safe).

        Overwrites previous checkpoint since checkpoints are monotonically growing.
        """
        ts = datetime.now().strftime("%H%M%S")
        path = os.path.join(self.checkpoint_dir, f"db_state_{ts}")

        # Remove previous checkpoint (tracked in memory, O(1) instead of O(n) dir scan)
        if self._last_checkpoint_path and os.path.isdir(self._last_checkpoint_path):
            shutil.rmtree(self._last_checkpoint_path)

        os.makedirs(path, exist_ok=True)

        if best_code is not None:
            self._atomic_write(os.path.join(path, "best_program.py"), best_code.encode("utf-8"))

        metadata_to_write = dict(metadata)
        if shared_construction_state and shared_construction_state.get("by_chain"):
            metadata_to_write["shared_constructions"] = {
                "by_chain": shared_construction_state.get("by_chain", {})
            }

        self._atomic_write(
            os.path.join(path, "metadata.json"),
            json.dumps(metadata_to_write, indent=2).encode("utf-8"),
        )
        self._atomic_write(
            os.path.join(path, "config.json"),
            json.dumps(config, indent=2).encode("utf-8"),
        )
        self._atomic_write(
            os.path.join(path, "policy.json"),
            json.dumps(policy, indent=2).encode("utf-8"),
        )

        reflections = []
        for node in nodes:
            reflection = node.get("reflection")
            if not reflection:
                continue
            reflections.append(
                {
                    "node_id": node.get("id"),
                    "parent_ids": node.get("parent_ids", []),
                    "metrics": node.get("metrics"),
                    "score": node.get("score"),
                    "reflection": reflection,
                    "created_at": node.get("created_at"),
                }
            )
        self._atomic_write(
            os.path.join(path, "reflection.json"),
            json.dumps(reflections, indent=2).encode("utf-8"),
        )

        nodes_bytes = json.dumps(nodes, indent=2).encode("utf-8")
        if self.config.use_gzip:
            nodes_bytes = gzip.compress(nodes_bytes)
            self._atomic_write(os.path.join(path, "nodes.json.gz"), nodes_bytes)
        else:
            self._atomic_write(os.path.join(path, "nodes.json"), nodes_bytes)

        # Save score statistics (CSV + quantile plot)
        save_score_statistics(
            nodes,
            path,
            metadata.get("completed_evaluations", 0),
            metadata.get("best_score", 0.0),
        )

        if failure_records:
            self._atomic_write(
                os.path.join(path, "failure.json"),
                json.dumps(failure_records, indent=2).encode("utf-8"),
            )

        # Write elite_history.csv if available (from llm_elite policy)
        elite_history = policy.get("state", {}).get("elite_history", [])
        if elite_history:
            csv_path = os.path.join(path, "elite_history.csv")
            df = pd.DataFrame(elite_history)
            # Reorder columns for readability
            columns = [
                "timestamp", "gen_id", "chain_idx", "action",
                "new_node_id", "new_node_score",
                "removed_node_id", "removed_node_score", "removed_index",
                "llm_reason", "used_fallback",
                "pool_size", "pool_avg_score", "pool_max_score",
            ]
            # Only include columns that exist
            columns = [c for c in columns if c in df.columns]
            df = df[columns]
            df.to_csv(csv_path, index=False)

            # Generate elite history plot and chains.csv
            try:
                import subprocess
                script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "plot_elite_history.py")
                subprocess.run(
                    ["python", script_path, "--data-dir", path],
                    capture_output=True,
                    timeout=60,
                )
            except Exception as e:
                import sys
                print(f"Warning: Failed to generate elite history plot: {e}", file=sys.stderr)

        # Save shared construction files
        files = shared_construction_state.get("files", []) if shared_construction_state else []
        if files:
            shared_dir = os.path.join(path, "shared_constructions")
            os.makedirs(shared_dir, exist_ok=True)
            for entry in files:
                source_path = entry.get("source_path")
                filename = entry.get("filename")
                if not source_path or not filename or not os.path.exists(source_path):
                    continue
                shutil.copy2(source_path, os.path.join(shared_dir, filename))

        self._last_checkpoint_path = path
        rich_print(_format_log(self.instance_id, "✓", f"[green][dim]Checkpoint:[/dim][/green] [cyan]{path}[/cyan]"))

    def _atomic_write(self, path: str, content: bytes) -> None:
        """
        Write file atomically using temp file + rename.

        This ensures that readers never see a partially-written file:
        1. Write content to a temporary file in the same directory
        2. Rename (move) the temp file to the target path

        The rename is atomic on POSIX systems, so the file either exists
        with full content or doesn't exist at all.
        """
        dir_path = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _config_to_dict(self) -> dict[str, Any]:
        """Serialize config (excluding sensitive data)."""
        c = self.config
        return {
            "init_program": c.init_program, "evaluator_path": c.evaluator_path,
            "eval_venv": getattr(c, "eval_venv", None),
            "eval_python": c.eval_python,
            "instruction_path": c.instruction_path, "max_generations": c.max_generations,
            "num_inspirations": c.num_inspirations,
            "min_inspirations_cnt": c.min_inspirations_cnt,
            "max_inspirations_cnt": c.max_inspirations_cnt,
            "eval_concurrency": c.eval_concurrency,
            "eval_timeout": c.eval_timeout, "gen_concurrency": c.gen_concurrency,
            "init_eval_repeats": c.init_eval_repeats,
            "init_eval_reduce": "max",
            "restart_every_n": c.restart_every_n,
            "log_interval": c.log_interval, "early_stop_score": c.early_stop_score,
            "output_path": c.output_path, "save_llm_io": c.save_llm_io,
            "selector": c.selector, "model": c.model,
            "temperature": c.temperature, "max_tokens": c.max_tokens,
            "api_base": c.api_base, "retry": c.retry, "timeout": c.timeout,
            "llm_backend": c.llm_backend,
            "context_window": c.context_window,
            "reasoning_budget": c.reasoning_budget,
            "response_budget": c.response_budget,
            "reflection_mode": c.reflection_mode,
            "llm_policy_model": c.llm_policy_model if c.llm_policy_model is not None else c.model,
            "llm_policy_api_base": c.llm_policy_api_base if c.llm_policy_api_base is not None else c.api_base,
        }

    def load(self, path: str, db: NodeDatabase, selector: Selector) -> dict[str, Any]:
        """
        Load checkpoint from path and return restored state.

        Auto-selection logic:
        - If path points to a checkpoint directory (e.g., "ckpt_20240101_120000"),
          load directly from that checkpoint
        - If path points to an instance directory (e.g., "instance_abc123"),
          automatically select the latest checkpoint by sorting ckpt_* directories

        Returns a dict with: instance_id, completed_evaluations, generation_attempts,
        generation_failures, generation_cancellations, best_score, best_node_id
        """
        from simpletes.node import Node

        # Validate path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

        # Auto-select latest checkpoint if directory given
        basename = os.path.basename(path)
        if not basename.startswith("ckpt_") and not basename.startswith("db_state_"):
            if not os.path.isdir(path):
                raise ValueError(f"Expected directory for checkpoint auto-selection: {path}")
            # Support both old (ckpt_) and new (db_state_) checkpoint formats
            ckpts = sorted([d for d in os.listdir(path) if d.startswith("ckpt_") or d.startswith("db_state_")])
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in: {path}")
            path = os.path.join(path, ckpts[-1])

        metadata_path = os.path.join(path, "metadata.json")
        config_path = os.path.join(path, "config.json")
        policy_path = os.path.join(path, "policy.json")
        nodes_gz_path = os.path.join(path, "nodes.json.gz")
        nodes_path = os.path.join(path, "nodes.json")

        missing = [
            p for p in (metadata_path, config_path, policy_path)
            if not os.path.exists(p)
        ]
        if missing:
            missing_str = ", ".join(os.path.basename(p) for p in missing)
            raise FileNotFoundError(f"Checkpoint missing required files: {missing_str}")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        with open(config_path, encoding="utf-8") as f:
            _ = json.load(f)
        with open(policy_path, encoding="utf-8") as f:
            policy_state = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain a JSON object")
        if not isinstance(policy_state, dict):
            policy_state = {}

        if os.path.exists(nodes_gz_path):
            with gzip.open(nodes_gz_path, "rt", encoding="utf-8") as f:
                nodes = json.load(f)
        elif os.path.exists(nodes_path):
            with open(nodes_path, encoding="utf-8") as f:
                nodes = json.load(f)
        else:
            raise FileNotFoundError("Checkpoint missing required file: nodes.json(.gz)")

        if not isinstance(nodes, list):
            raise ValueError("nodes.json must contain a JSON list")

        self.checkpoint_dir = os.path.dirname(path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        invalid_nodes = 0
        for nd in nodes:
            node = Node.from_dict(nd)
            try:
                validate_node_for_db(node)
            except ValueError:
                invalid_nodes += 1
                continue
            db.add(node)

        # Restore policy state
        if policy_state:
            saved_name = policy_state.get("name")
            if saved_name and saved_name != selector.name:
                rich_print(_format_log(
                    self.instance_id, "⚠",
                    f"[yellow]Policy mismatch: '{saved_name}' vs '{selector.name}'[/yellow]"
                ))
            selector.load_state_dict(policy_state.get("state", policy_state.get("policy_state", {})))

        selector.reconcile_with_db(db)

        if invalid_nodes:
            rich_print(_format_log(
                self.instance_id,
                "⚠",
                f"[yellow]Skipped {invalid_nodes} invalid nodes while loading checkpoint[/yellow]",
            ))

        best_node = db.best()
        best_score = best_node.score if best_node else -float("inf")
        best_node_id = best_node.id if best_node else None

        failure_path = os.path.join(path, "failure.json")
        if os.path.exists(failure_path):
            with open(failure_path, encoding="utf-8") as f:
                failure_records = json.load(f)
            assert isinstance(failure_records, list), "failure.json must contain a JSON list"
        else:
            failure_records = []

        runtime_shared_dir = os.path.join(self.checkpoint_dir, "shared_constructions")
        os.makedirs(runtime_shared_dir, exist_ok=True)
        shared_meta = metadata.get("shared_constructions", {})
        shared_dir = os.path.join(path, "shared_constructions")
        shared_state: dict[str, dict[str, str] | None] = {}
        for chain_idx, ref in dict(shared_meta.get("by_chain", {})).items():
            if not ref:
                shared_state[str(chain_idx)] = None
                continue
            filename = ref.get("filename")
            snapshot_id = ref.get("snapshot_id")
            summary = ref.get("summary", "")
            if not filename or not snapshot_id:
                shared_state[str(chain_idx)] = None
                continue
            source_path = os.path.join(shared_dir, filename)
            runtime_path = os.path.join(runtime_shared_dir, filename)
            if os.path.exists(source_path):
                shutil.copy2(source_path, runtime_path)
                shared_state[str(chain_idx)] = {
                    "snapshot_id": str(snapshot_id),
                    "summary": str(summary),
                    "path": runtime_path,
                }
            else:
                shared_state[str(chain_idx)] = None

        return {
            "instance_id": metadata.get("instance_id", self.instance_id),
            "completed_evaluations": int(metadata.get("completed_evaluations", 0)),
            "generation_attempts": int(metadata.get("generation_attempts", 0)),
            "generation_failures": int(metadata.get("generation_failures", 0)),
            "generation_cancellations": int(metadata.get("generation_cancellations", 0)),
            "evaluation_failures": int(metadata.get("evaluation_failures", 0)),
            "gen_id_counter": int(metadata.get("gen_id_counter", 0)),
            "best_score": float(best_score),
            "best_node_id": best_node_id,
            "failure_records": failure_records,
            "shared_constructions": shared_state,
            "chain_best_scores": metadata.get("chain_best_scores", {}),
        }
