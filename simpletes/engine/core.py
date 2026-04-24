"""
SimpleTES core evolution engine.

Architecture:
    Scheduler (policy) -> gen_queue -> gen_consumer -> LLM -> (nodes) -> eval_queue -> eval_consumer -> evaluator -> DB
"""
from __future__ import annotations

# ==========================================================================
# Queue Sizing Constants
# ==========================================================================
# Multiplier ensures queues buffer enough work during scheduler pauses.
# 8x provides headroom for bursty generation patterns.
_QUEUE_SIZE_MULTIPLIER = 8
_MIN_QUEUE_SIZE = 4096
_INITIAL_SHARED_CONSTRUCTION_ENV = "SIMPLETES_INITIAL_SHARED_CONSTRUCTION_PATH"

from datetime import datetime
import asyncio
from functools import lru_cache
import importlib.util
import os
import signal
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from rich.panel import Panel

from simpletes.evaluator import rich_print
from simpletes.utils.log import format_log, install_tee_logger

from simpletes.config import (
    EngineConfig,
    task_dir_from_evaluator_path,
)
from simpletes.engine.runtime import LocalRuntime, RuntimeBase
from simpletes.engine.checkpoint import CheckpointManager
from simpletes.engine.scheduler import SchedulerMixin
from simpletes.evaluator import EvaluationOutcome, EvaluatorWorker
from simpletes.node import (
    EvolveBlockContext,
    Node,
    NodeDatabase,
    NodeDatabaseSnapshot,
    Status,
    score_from_metrics,
    validate_node_for_db,
)
from simpletes.generator import Generator, GenerationTask
from simpletes.llm import LLMCallError
from simpletes.policies import Selector, create_selector
from simpletes.construction import summarize_construction_payload, write_payload
from simpletes.utils.task_prep import load_task_requirements
from simpletes.utils.text import DEFAULT_METRICS_ERROR_MAX_CHARS, truncate_error_in_metrics


# ============================================================================
# Main Engine
# ============================================================================

class SimpleTESEngine(SchedulerMixin):
    """
    Main evolution engine with optimized concurrency.

    Uses fine-grained locking for high-throughput scenarios (256+ concurrent workers).
    """

    def __init__(self, config: EngineConfig, runtime: RuntimeBase | None = None, resume_path: str | None = None):
        self.config = config
        self.runtime: RuntimeBase = runtime or LocalRuntime()
        self.instance_id = str(uuid.uuid4())[:8]
        self.db = NodeDatabase()
        self._resume_path = resume_path  # Store for later use in run()

        # Lock order to avoid circular waits: counter (short critical sections)
        # before db (longer). Policy callbacks are invoked OUTSIDE both locks
        # since policies have their own internal locks.
        self._counter_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()

        # Qubit-routing evaluator slot namespace/caching env.
        self._qubit_routing_slot_workspace = None
        if _is_qubit_routing_evaluator(config.evaluator_path):
            slot_workspace_module = _load_qubit_routing_slot_workspace_module()
            self._qubit_routing_slot_workspace = slot_workspace_module.QubitRoutingSlotWorkspace(
                evaluator_path=config.evaluator_path,
                instance_id=self.instance_id,
                eval_concurrency=config.eval_concurrency,
                eval_timeout=config.eval_timeout,
                log_message=self._log,
                printer=rich_print,
            )

        # Load instruction
        with open(config.instruction_path, encoding="utf-8") as f:
            self.instruction = f.read()

        self.worker = EvaluatorWorker(
            config.evaluator_path,
            timeout=config.eval_timeout,
            python_executable=config.eval_python,
            target_filename=(
                f"program{Path(config.init_program).suffix}"
                if Path(config.init_program).suffix
                else "program.py"
            ),
        )

        # Create policy
        self.selector: Selector = create_selector(
            config.selector,
            exploitation_ratio=config.exploitation_ratio,
            exploration_ratio=config.exploration_ratio,
            elite_ratio=config.elite_ratio,
            num_chains=config.num_chains,
            max_generations=config.max_generations,
            k=config.k_candidates,
            restart_every_n=config.restart_every_n,
            c=config.puct_c,
            gamma=config.rpucg_gamma,
            min_inspirations_cnt=config.min_inspirations_cnt,
            max_inspirations_cnt=config.max_inspirations_cnt,
            reflection_mode=config.reflection_mode,
            llm_policy_model=config.llm_policy_model if config.llm_policy_model is not None else config.model,
            llm_policy_api_base=config.llm_policy_api_base if config.llm_policy_api_base is not None else config.api_base,
            llm_policy_api_key=config.llm_policy_api_key if config.llm_policy_api_key is not None else config.api_key,
            llm_policy_pool_size=config.llm_policy_pool_size,
            elite_selection_strategy=config.elite_selection_strategy,
            task_instruction=self.instruction,
        )

        # Queues with bounded size to prevent memory explosion under high concurrency
        gen_queue_size = max(_MIN_QUEUE_SIZE, config.gen_concurrency * _QUEUE_SIZE_MULTIPLIER)
        eval_queue_size = max(_MIN_QUEUE_SIZE, config.eval_concurrency * _QUEUE_SIZE_MULTIPLIER)
        self.gen_queue: asyncio.Queue[GenerationTask] = asyncio.Queue(maxsize=gen_queue_size)
        self.eval_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=eval_queue_size)
        
        # Pending nodes awaiting evaluation (not yet in DB)
        self._pending_nodes: dict[str, Node] = {}

        # Runtime state
        self.completed_evaluations: int = 0
        self.generation_attempts: int = 0
        self.generation_failures: int = 0
        self.generation_cancellations: int = 0
        self.evaluation_failures: int = 0
        self._failure_records: list[dict[str, Any]] = []
        self._gen_inflight: int = 0
        self._eval_inflight: int = 0
        self.best_score: float = -float("inf")
        self.best_node_id: str | None = None
        self._early_stop_logged: bool = False
        self._last_db_display_interval: int = 0  # Track last shown db_show_interval threshold
        self._chain_shared_constructions: dict[int, dict[str, str] | None] = {
            i: None for i in range(max(1, config.num_chains))
        }
        self._chain_best_scores: dict[int, float] = {
            i: -float("inf") for i in range(max(1, config.num_chains))
        }
        # Cached DB snapshot for high-concurrency optimization.
        # The scheduler calls select() frequently, and creating a new snapshot each time
        # is expensive (O(n) copy). By caching the snapshot and tracking the DB version,
        # we can reuse it when the DB hasn't changed, reducing overhead significantly
        # when many scheduler iterations occur between evaluations completing.
        self._cached_snapshot: NodeDatabaseSnapshot | None = None
        self._cached_snapshot_version: int = -1

        # Static prompt mode: reuse init_program only when num_inspirations == 0.
        self._static_prompt_mode = (self.config.num_inspirations == 0)
        if self._static_prompt_mode:
            if self.config.num_chains != 1:
                raise ValueError("static prompt mode requires num_chains == 1")
        self._static_prompt: str | None = None
        self._static_inspiration_ids: list[str] | None = None
        self._static_shared_construction_id: str | None = None

        task_dir = task_dir_from_evaluator_path(config.evaluator_path)
        self._requirements_path, self._available_packages = load_task_requirements(task_dir)
        
        # EVOLVE-BLOCK context for extracting only the evolving part
        with open(config.init_program, encoding="utf-8") as f:
            init_code = f.read()
        self._evolve_context = EvolveBlockContext.from_program(init_code)
        if not self._evolve_context.has_markers:
            rich_print(
                f"[yellow]Warning:[/yellow] init_program missing EVOLVE-BLOCK markers: "
                f"[cyan]{config.init_program}[/cyan]"
            )
            raise SystemExit(2)

        # Generator handles LLM lifecycle, prompt building, code extraction
        self.generator = Generator(
            config=config,
            instruction=self.instruction,
            evolve_context=self._evolve_context,
            available_packages=self._available_packages,
        )
        self._stop_event = asyncio.Event()
        self._progress_event = asyncio.Event()
        self._shutdown_requested = False
        
        # Worker tasks
        self._gen_workers: list[asyncio.Task] = []
        self._eval_workers: list[asyncio.Task] = []

        # Checkpoints - use date-based directory structure
        # If resuming, use the instance directory from resume_path
        if self._resume_path:
            self.checkpoint_dir = self._resolve_resume_checkpoint_dir(self._resume_path)
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.checkpoint_dir = os.path.join(config.output_path, date_str, f"instance-{self.instance_id}")
        self._shared_construction_dir = os.path.join(self.checkpoint_dir, "shared_constructions")
        os.makedirs(self._shared_construction_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            config=config, instance_id=self.instance_id, checkpoint_dir=self.checkpoint_dir,
        )

        # Install tee logger: duplicate all stdout to run.log in checkpoint dir
        self._tee = install_tee_logger(os.path.join(self.checkpoint_dir, "run.log"))

        init_info = self.runtime.decorate_init_info(self._build_init_info())
        rich_print(Panel(init_info, border_style="cyan", title="[bold]Initialization[/bold]"))

    def _policy_summary(self) -> str:
        c = self.config
        policy = c.selector
        if policy not in {"balance", "puct", "rpucg"}:
            return policy

        chains = max(1, c.num_chains)
        total_gens = max(0, c.max_generations)
        k = max(1, c.k_candidates)
        base = total_gens // chains
        rem = total_gens % chains
        base_prompt = (base + k - 1) // k if base > 0 else 0
        extra_prompt = ((base + 1 + k - 1) // k) - base_prompt if rem else 0
        extra = f" (+1 for {rem} chains)" if extra_prompt else ""
        core = f"chains={chains}, local_bon(k)={k}, prompts/chain={base_prompt}{extra}"
        if policy == "puct":
            return f"puct({core}, c={c.puct_c:g})"
        if policy == "rpucg":
            return f"rpucg({core}, c={c.puct_c:g}, γ={c.rpucg_gamma:g})"
        return f"balance({core})"

    def _build_init_info(self) -> str:
        """Build initialization panel text."""
        c = self.config
        policy = self._policy_summary()

        if c.num_inspirations == 0:
            insp_desc = "0 (static init_program)"
        elif c.min_inspirations_cnt is not None and c.max_inspirations_cnt is not None:
            insp_desc = f"{c.min_inspirations_cnt}-{c.max_inspirations_cnt} (sampled)"
        else:
            insp_desc = str(c.num_inspirations)

        lines = [
            f"[bold cyan]SimpleTES[/bold cyan] [dim]({self.instance_id})[/dim]",
            "",
            "[bold]── Task ──[/bold]",
            f"[dim]Init program:[/dim] [cyan]{c.init_program}[/cyan]",
            f"[dim]Evaluator:[/dim] [cyan]{c.evaluator_path}[/cyan]",
            f"[dim]Instruction:[/dim] [cyan]{c.instruction_path}[/cyan]",
            "",
            "[bold]── Policy ──[/bold]",
            f"[dim]Selector:[/dim] [cyan]{policy}[/cyan]",
            f"[dim]Inspirations:[/dim] {insp_desc} | [dim]failure_patterns:[/dim] {'ON' if c.include_failure_patterns else 'OFF'}",
        ]

        # Policy-specific parameters from get_info()
        policy_info = self.selector.get_info()
        if policy_info:
            info_parts = [f"{k}={v}" for k, v in policy_info.items()]
            lines.append(f"[dim]Params:[/dim] {' | '.join(info_parts)}")

        lines.append(f"[dim]Reflection:[/dim] {'[green]ON[/green]' if c.reflection_mode else '[dim]OFF[/dim]'}")

        lines.extend([
            "",
            "[bold]── Generator ──[/bold]",
            f"[dim]Model:[/dim] [cyan]{c.model}[/cyan]",
            f"[dim]Backend:[/dim] [cyan]{c.llm_backend}[/cyan]" +
            (f" | tokenizer={c.tokenizer_path}" if c.tokenizer_path else ""),
            f"[dim]Params:[/dim] temp={c.temperature:g} | max_tokens={c.max_tokens}" +
            (f" | max_total_tokens={c.max_total_tokens}" if c.max_total_tokens else "") +
            f" | retry={c.retry}" +
            (f" | timeout={c.timeout:g}s" if c.timeout else ""),
        ])
        if c.api_base:
            lines.append(f"[dim]API base:[/dim] [cyan]{c.api_base}[/cyan]")
        if c.api_key:
            lines.append(f"[dim]API key:[/dim] ***")

        lines.extend([
            "",
            "[bold]── Concurrency & Budget ──[/bold]",
            f"[dim]Workers:[/dim] gen={c.gen_concurrency} | eval={c.eval_concurrency}" +
            f" | stream_k={'ON' if c.stream_k_candidates else 'OFF'}",
            f"[dim]Init eval:[/dim] repeats={c.init_eval_repeats} | reduce=max",
            f"[dim]Generation:[/dim] max_generations={c.max_generations} | restart_every_n={c.restart_every_n}",
            f"[dim]Timeouts:[/dim] eval={c.eval_timeout:g}s | backpressure={c.backpressure_multiplier:g}",
            f"[dim]Budget:[/dim] max_generations={c.max_generations}" +
            (f" | early_stop={c.early_stop_score:.6f}" if c.early_stop_score is not None else ""),
            "",
            "[bold]── Checkpoint ──[/bold]",
            f"[dim]Output:[/dim] [cyan]{c.output_path}[/cyan]",
            f"[dim]Interval:[/dim] every {c.log_interval} evals | db_show={c.db_show_interval}" +
            f" | gzip={'ON' if c.use_gzip else 'OFF'}" +
            f" | save_llm_io={'ON' if c.save_llm_io else 'OFF'}",
        ])

        # Eval environment section
        lines.extend(["", "[bold]── Eval Environment ──[/bold]"])
        eval_python_used = c.eval_python or sys.executable
        lines.append(
            f"[dim]Python:[/dim] [cyan]{eval_python_used}[/cyan]"
            + (" [dim](override)[/dim]" if c.eval_python else " [dim](sys.executable)[/dim]")
        )

        task_dir = task_dir_from_evaluator_path(c.evaluator_path)
        auto_venv_dir = task_dir / "venv"
        explicit_venv = getattr(c, "eval_venv", None)
        if explicit_venv:
            explicit_venv_dir = Path(explicit_venv).expanduser().resolve()
            suffix = "[dim](explicit)[/dim]" if c.eval_python else "[dim](explicit; not used)[/dim]"
            lines.append(f"[dim]Venv:[/dim] [cyan]{explicit_venv_dir}[/cyan] {suffix}")
        elif c.eval_python:
            lines.append(f"[dim]Venv:[/dim] [cyan]{auto_venv_dir}[/cyan] [dim](auto)[/dim]")
        else:
            lock_path = task_dir / "uv.lock"
            if lock_path.is_file():
                lines.append(
                    "[yellow]Warning:[/yellow] "
                    f"Found lockfile at [cyan]{lock_path}[/cyan] but no venv at [cyan]{auto_venv_dir}[/cyan]"
                )

        if self._requirements_path is not None:
            lines.append(
                f"[dim]Requirements:[/dim] [cyan]{self._requirements_path}[/cyan] | "
                f"packages={len(self._available_packages)}"
            )

        return "\n".join(lines)

    def _log(self, icon: str = "", msg: str = "", status: str = "") -> str:
        return format_log(icon, msg, status)

    # ---------- Scheduler helpers ----------

    def _db_snapshot_locked(self) -> NodeDatabaseSnapshot:
        """Return a cached DB snapshot (must be called under _db_lock)."""
        if self._cached_snapshot_version == self.db._version and self._cached_snapshot is not None:
            return self._cached_snapshot
        snapshot = self.db.snapshot()
        self._cached_snapshot = snapshot
        self._cached_snapshot_version = self.db._version
        return snapshot

    def _queue_stats_locked(self) -> dict[str, int]:
        """Return queue stats for backpressure (called under _db_lock)."""
        return self.runtime.queue_stats(self)

    def _shared_construction_refs_locked(self) -> dict[int, dict[str, str] | None]:
        return {
            chain_idx: (dict(ref) if ref is not None else None)
            for chain_idx, ref in self._chain_shared_constructions.items()
        }

    def _shared_construction_checkpoint_state_locked(self) -> dict[str, Any]:
        by_chain: dict[str, dict[str, str] | None] = {}
        files: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for chain_idx, ref in self._chain_shared_constructions.items():
            if ref is None:
                by_chain[str(chain_idx)] = None
                continue
            by_chain[str(chain_idx)] = {
                "snapshot_id": ref["snapshot_id"],
                "summary": ref["summary"],
                "filename": os.path.basename(ref["path"]),
            }
            snapshot_id = ref["snapshot_id"]
            if snapshot_id in seen_ids:
                continue
            seen_ids.add(snapshot_id)
            files.append({
                "snapshot_id": snapshot_id,
                "filename": os.path.basename(ref["path"]),
                "source_path": ref["path"],
            })
        return {"by_chain": by_chain, "files": files}

    def _set_shared_construction_for_chain_locked(
        self,
        chain_idx: int,
        snapshot_id: str,
        payload: Any,
    ) -> None:
        path = os.path.join(self._shared_construction_dir, f"{snapshot_id}.json")
        if not os.path.exists(path):
            write_payload(path, payload)
        summary = summarize_construction_payload(payload)
        ref = {"snapshot_id": snapshot_id, "path": path, "summary": summary}
        self._chain_shared_constructions[chain_idx] = ref
        self._static_prompt = None
        self._static_shared_construction_id = None

    def _set_shared_construction_for_all_chains_locked(
        self,
        snapshot_id: str,
        payload: Any,
    ) -> None:
        for chain_idx in self._chain_shared_constructions:
            self._set_shared_construction_for_chain_locked(chain_idx, snapshot_id, payload)

    def _clear_shared_construction_for_chain_locked(self, chain_idx: int) -> None:
        self._chain_shared_constructions[chain_idx] = None
        self._static_prompt = None
        self._static_shared_construction_id = None

    def _clear_shared_constructions_locked(self) -> None:
        for chain_idx in self._chain_shared_constructions:
            self._clear_shared_construction_for_chain_locked(chain_idx)

    async def _evaluate_code(
        self,
        code: str,
        *,
        shared_construction_id: str | None = None,
        shared_construction_path: str | None = None,
    ) -> EvaluationOutcome:
        """Evaluate code and return metrics, score, and any captured payload."""
        shared_path = shared_construction_path
        if shared_path is None and shared_construction_id is not None:
            shared_path = os.path.join(self._shared_construction_dir, f"{shared_construction_id}.json")
        if shared_path is not None and not os.path.exists(shared_path):
            shared_path = None
        try:
            outcome = await self.worker.evaluate(
                code,
                shared_construction_path=shared_path,
            )
            metrics = outcome.metrics
            truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
            return EvaluationOutcome(
                metrics=metrics,
                captured_construction_payload=outcome.captured_construction_payload,
            )
        except Exception as e:
            metrics = {"error": str(e), "combined_score": -float("inf")}
            truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
            return EvaluationOutcome(metrics=metrics, captured_construction_payload=None)

    async def _handle_invalid_evaluation(
        self,
        node: Node,
        reason: str,
        *,
        from_pending: bool,
    ) -> None:
        """Handle invalid evaluation results without inserting into the DB."""
        if from_pending:
            async with self._db_lock:
                self._pending_nodes.pop(node.id, None)
        async with self._counter_lock:
            self.evaluation_failures += 1
            self._failure_records.append({
                "type": "evaluation",
                "reason": reason,
                "llm_input": node.llm_input,
                "llm_output": node.llm_output,
                "code": node.code,
                "metrics": node.metrics,
                "score": node.score,
                "gen_id": node.gen_id,
                "chain_idx": node.chain_idx,
                "shared_construction_id": node.shared_construction_id,
                "created_at": node.created_at,
            })
        assert node.gen_id is not None, "node missing gen_id in _handle_invalid_evaluation"
        completion = self.selector.on_generation_failed(node.gen_id)
        await self._handle_pending_finalize(completion)
        short_reason = reason[:200]
        rich_print(self._log("⚠", f"[yellow]Eval result rejected: {short_reason}[/yellow]"))

    # ---------- Initialization ----------

    async def _initialize_from_scratch(self) -> None:
        """Evaluate the initial program and add to DB."""
        with open(self.config.init_program, encoding="utf-8") as f:
            code = f.read()

        node = Node(id=uuid.uuid4().hex, code=code, parent_ids=[], status=Status.EVAL_PENDING)
        initial_shared_path = os.environ.get(_INITIAL_SHARED_CONSTRUCTION_ENV)

        repeats = self.config.init_eval_repeats
        outcomes = await asyncio.gather(*[
            self._evaluate_code(code, shared_construction_path=initial_shared_path)
            for _ in range(repeats)
        ])

        scores = [score_from_metrics(outcome.metrics) for outcome in outcomes]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        outcome = outcomes[best_idx]
        metrics = outcome.metrics
        score = scores[best_idx]

        node.metrics, node.score, node.status = metrics, score, Status.DONE
        try:
            validate_node_for_db(node)
        except ValueError as e:
            raise ValueError(f"Initial program evaluation invalid: {e}") from e

        async with self._db_lock:
            self.db.add(node)
            self.best_score, self.best_node_id = node.score, node.id
            self.completed_evaluations += 1
            for i in self._chain_best_scores:
                self._chain_best_scores[i] = node.score
            if self.config.include_construction and outcome.captured_construction_payload is not None:
                self._set_shared_construction_for_all_chains_locked(
                    snapshot_id=node.id,
                    payload=outcome.captured_construction_payload,
                )

        rich_print(self._log("", (
            f"[bold]Initial score:[/bold] [green]{score:.6f}[/green] "
            f"[dim](reduce=max, repeats={repeats}[/dim]"
        )))

        if self.config.early_stop_score is not None and score >= self.config.early_stop_score:
            rich_print(self._log("✓", "[bold green]Early stop satisfied by initial program[/bold green]"))
            self._stop_event.set()

    # ---------- Checkpoint helpers ----------

    async def _snapshot_for_checkpoint(self) -> tuple[
        str | None,
        dict[str, object],
        dict[str, object],
        dict[str, object],
        list[dict[str, object]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        async with self._db_lock:
            snapshot = await self.checkpoint_manager.snapshot(
                db=self.db,
                best_node_id=self.best_node_id,
                completed_evaluations=self.completed_evaluations,
                generation_attempts=self.generation_attempts,
                generation_failures=self.generation_failures,
                generation_cancellations=self.generation_cancellations,
                evaluation_failures=self.evaluation_failures,
                best_score=self.best_score,
                selector=self.selector,
            )
            shared_snapshot = self._shared_construction_checkpoint_state_locked()
        async with self._counter_lock:
            failure_snapshot = list(self._failure_records)
        return (*snapshot, failure_snapshot, shared_snapshot)

    async def _write_checkpoint(self) -> None:
        best_code, metadata, config, policy, nodes, failure_records, shared_snapshot = await self._snapshot_for_checkpoint()
        # Save gen_id counter for resume
        metadata["gen_id_counter"] = self.generator.get_gen_id_counter()
        # Save per-chain best scores
        metadata["chain_best_scores"] = {
            str(k): v for k, v in self._chain_best_scores.items()
        }
        await asyncio.to_thread(
            self.checkpoint_manager.write_sync,
            best_code,
            metadata,
            config,
            policy,
            nodes,
            failure_records,
            shared_snapshot,
        )

    def _final_results_panel(self) -> Panel:
        return Panel(
            f"[bold green]Evolution Complete![/bold green]\n"
            f"[dim]Instance:[/dim] [cyan]{self.instance_id}[/cyan]\n"
            f"[dim]Gen fails:[/dim] {self.generation_failures}\n"
            f"[dim]Gen cancels:[/dim] {self.generation_cancellations}\n"
            f"[dim]Eval rejects:[/dim] {self.evaluation_failures}\n"
            f"[bold]Best score:[/bold] [green]{self.best_score:.8f}[/green]",
            border_style="green",
            title="[bold]Final Results[/bold]",
        )

    async def _finalize_run(self) -> None:
        """Write final checkpoint and print summary."""
        try:
            await self._write_checkpoint()
        except Exception as e:
            rich_print(self._log("⚠", f"[yellow]Checkpoint failed: {e}[/yellow]"))

        self.generator.close()

        rich_print(self._final_results_panel())

    # ---------- Generation pipeline ----------

    async def _generate_batch(self, task: GenerationTask) -> None:
        """Generate code for one queued task (task.k candidates from one prompt)."""
        track_io = self.config.save_llm_io or self.config.reflection_mode
        successes = 0
        cancelled = False

        async with self._counter_lock:
            self._gen_inflight += 1

        try:
            results = await self.generator.generate(task, self.instance_id, track_io)
        except asyncio.CancelledError:
            cancelled = True
            rich_print(self._log("⚠", "[yellow]Generation cancelled[/yellow]"))
            async with self._counter_lock:
                self.generation_cancellations += 1
            results = []
        except LLMCallError as e:
            rich_print(self._log("⚠", "[red]Generation failed[/red]"))
            rich_print(f"   [dim]Model:    [/dim]{e.model}")
            rich_print(f"   [dim]API base: [/dim]{e.api_base or '(provider default)'}")
            rich_print(f"   [dim]Error:    [/dim][red]{e.error_type}[/red]")
            rich_print(f"   [dim]Message:  [/dim]{e.message}")
            async with self._counter_lock:
                self.generation_failures += 1
            results = []
        except Exception as e:
            rich_print(self._log("⚠", f"[red]Generation failed: {type(e).__name__}: {str(e)}[/red]"))
            async with self._counter_lock:
                self.generation_failures += 1
            results = []
        for result in results:
            if result.success:
                successes += 1
                await self.runtime.handle_generation_output(self, task, result, track_io)
            else:
                async with self._counter_lock:
                    self._failure_records.append({
                        "type": "generation",
                        "reason": result.reason,
                        "llm_input": task.prompt,
                        "llm_output": result.llm_output,
                        "gen_id": task.gen_id,
                        "chain_idx": task.chain_idx,
                        "shared_construction_id": task.shared_construction_id,
                        "created_at": datetime.utcnow().isoformat(),
                    })
                rich_print(self._log("⚠", f"[yellow]Generation rejected: {result.reason}[/yellow]"))
                completion = self.selector.on_generation_failed(task.gen_id)
                await self._handle_pending_finalize(completion)

        # Notify policy for missing results (LLM returned fewer than k)
        for _ in range(task.k - len(results)):
            completion = self.selector.on_generation_failed(task.gen_id)
            await self._handle_pending_finalize(completion)

        async with self._counter_lock:
            self._gen_inflight -= 1

        if cancelled:
            raise asyncio.CancelledError()

    async def _gen_worker(self, wid: int) -> None:
        """Generation worker with clean cancellation."""
        try:
            while True:
                task = await self.gen_queue.get()
                try:
                    await self._generate_batch(task)
                    self.gen_queue.task_done()
                    self._progress_event.set()
                except asyncio.CancelledError:
                    self.gen_queue.task_done()
                    # Note: _generate_batch's finally block already calls on_generation_failed
                    # so we don't need to call it again here (avoids double-call)
                    raise
                except Exception as e:
                    self.gen_queue.task_done()
                    rich_print(self._log("⚠", f"[yellow]Gen worker error: {type(e).__name__}: {str(e)}[/yellow]"))
                    rich_print(f"[dim]{traceback.format_exc()}[/dim]")
                    raise
        except asyncio.CancelledError:
            # If cancelled while waiting on get(), there's nothing to clean up
            pass

    # ---------- Evaluation pipeline ----------

    async def _evaluate_one(self, node_id: str) -> None:
        """Evaluate a single pending node and commit it to DB.
        
        Policy callbacks are called outside locks to avoid blocking the event loop.
        Data needed for callbacks is collected inside the lock, then callback is invoked outside.
        """
        async with self._db_lock:
            node = self._pending_nodes.get(node_id)
            assert node is not None, f"pending node {node_id} missing from _pending_nodes"

        outcome = await self._evaluate_code(
            node.code,
            shared_construction_id=node.shared_construction_id,
        )
        metrics = outcome.metrics
        score = score_from_metrics(metrics)
        node.metrics, node.score, node.status = metrics, score, Status.DONE
        try:
            validate_node_for_db(node)
        except ValueError as e:
            await self._handle_invalid_evaluation(node, str(e), from_pending=True)
            return
        await self._commit_node(node, outcome.captured_construction_payload)

    async def _commit_node(
        self,
        node: Node,
        captured_construction_payload: Any | None = None,
    ) -> None:
        """Commit a finished node to DB and trigger bookkeeping."""
        should_ckpt = False
        should_show = False
        parents: list[Node] = []
        improved = False
        old_best_score = -float("inf")
        early_stop_just_triggered = False

        async with self._db_lock:
            self.db.add(node)
            self._pending_nodes.pop(node.id, None)

            # Collect parents inside lock to avoid race condition
            parents = [self.db.nodes[pid] for pid in node.parent_ids if pid in self.db.nodes]

            # Global best tracking (logging, early stop, checkpoint)
            if node.score > self.best_score:
                old_best_score = self.best_score
                self.best_score = node.score
                self.best_node_id = node.id
                improved = True

            # Per-chain construction update
            if self.config.include_construction and node.chain_idx is not None:
                chain_idx = node.chain_idx
                if chain_idx in self._chain_best_scores and node.score > self._chain_best_scores[chain_idx]:
                    self._chain_best_scores[chain_idx] = node.score
                    if captured_construction_payload is not None:
                        self._set_shared_construction_for_chain_locked(
                            chain_idx, snapshot_id=node.id, payload=captured_construction_payload,
                        )
                    else:
                        self._clear_shared_construction_for_chain_locked(chain_idx)

            self.completed_evaluations += 1
            ce = self.completed_evaluations

            should_ckpt = self.config.log_interval > 0 and ce % self.config.log_interval == 0

            # Threshold-based db_show_interval to avoid skips under high concurrency.
            if self.config.db_show_interval > 0:
                current_interval = ce // self.config.db_show_interval
                if current_interval > self._last_db_display_interval:
                    self._last_db_display_interval = current_interval
                    should_show = True

            if self.config.early_stop_score is not None and self.best_score >= self.config.early_stop_score:
                if not self._early_stop_logged:
                    self._early_stop_logged = True
                    early_stop_just_triggered = True
                self._stop_event.set()

        # Log outside lock
        if improved:
            rich_print(self._log("🏆", f"[green]New best score: {node.score:.6f} at chain {node.chain_idx} (prev: {old_best_score:.6f}, node: {node.id[:8]} at chain {node.chain_idx})[/green]"))
        if early_stop_just_triggered:
            rich_print(self._log("✓", f"[bold green]Early stop triggered! score={self.best_score:.6f} >= threshold={self.config.early_stop_score:.6f}[/bold green]"))

        # Notify policy outside lock
        completion = self.selector.on_child_done(node, parents)
        await self._handle_pending_finalize(completion)

        if should_show:
            await self._print_status()
        if should_ckpt:
            await self._write_checkpoint()
        self._progress_event.set()

    async def _ingest_result(self, result: Any) -> None:
        """Ingest an evaluation result from an external runtime."""
        metrics = result.metrics
        truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
        node = Node(
            id=result.node_id,
            code=result.code,
            parent_ids=result.parent_ids,
            gen_id=result.gen_id,
            chain_idx=result.chain_idx,
            shared_construction_id=getattr(result, "shared_construction_id", None),
            metrics=metrics,
            score=result.score,
            status=Status.DONE,
            llm_input=getattr(result, "llm_input", None),
            llm_output=getattr(result, "llm_output", None),
            token_usage=getattr(result, "token_usage", None),
        )

        try:
            validate_node_for_db(node)
        except ValueError as e:
            await self._handle_invalid_evaluation(node, str(e), from_pending=False)
            return

        await self._commit_node(
            node,
            getattr(result, "captured_construction_payload", None),
        )

    async def _eval_worker(self, wid: int) -> None:
        """Evaluation worker with clean cancellation."""
        current_node_id: str | None = None
        try:
            while True:
                current_node_id = await self.eval_queue.get()
                eval_started = False
                try:
                    async with self._counter_lock:
                        self._eval_inflight += 1
                    eval_started = True
                    await self._evaluate_one(current_node_id)
                    self.eval_queue.task_done()
                    self._progress_event.set()
                    current_node_id = None  # Mark as processed
                except asyncio.CancelledError:
                    self.eval_queue.task_done()
                    # Notify policy about the evaluation that was in progress
                    if current_node_id is not None:
                        async with self._db_lock:
                            node = self._pending_nodes.pop(current_node_id, None)
                        if node is not None:
                            assert node.gen_id is not None, "pending node missing gen_id"
                            completion = self.selector.on_generation_failed(node.gen_id)
                            await self._handle_pending_finalize(completion)
                    raise
                except Exception as e:
                    self.eval_queue.task_done()
                    rich_print(self._log("⚠", f"[yellow]Eval worker error: {type(e).__name__}: {str(e)}[/yellow]"))
                    rich_print(f"[dim]{traceback.format_exc()}[/dim]")
                    current_node_id = None
                    raise
                finally:
                    if eval_started:
                        async with self._counter_lock:
                            self._eval_inflight -= 1
        except asyncio.CancelledError:
            # If cancelled while waiting on get(), current_node_id is None (nothing to clean up)
            pass

    # ---------- Status ----------

    async def _print_status(self) -> None:
        """Print compact status."""
        async with self._counter_lock:
            gen_failures = self.generation_failures
            eval_failures = self.evaluation_failures
            gen_inflight = self._gen_inflight
        async with self._db_lock:
            total = len(self.db)
            best_score = self.best_score if total > 0 else None
            stats = self._queue_stats_locked()
        gq = stats["gen_queue"]
        eq = stats["eval_queue"]
        eval_inflight = self.runtime.eval_active(self)

        msg_parts = [
            f"finished_nodes={total}",
            f"gen_fail={gen_failures} gen_inflight={gen_inflight} llm_q={gq}",
            f"eval_fail={eval_failures} eval_inflight={eval_inflight} eval_q={eq}",
        ]
        if best_score is not None and best_score != -float("inf"):
            msg_parts.append(f"best={best_score:.6f}")
        msg = " | ".join(msg_parts)
        rich_print(self._log("📊", msg))

    # ---------- Main run ----------

    async def run(self) -> None:
        """Run the evolution loop."""
        # Signal handlers
        def handle_signal(sig: signal.Signals) -> None:
            if not self._shutdown_requested:
                self._shutdown_requested = True
                rich_print(f"\n{self._log('⚠', f'[yellow]Received {sig.name}, shutting down...[/yellow]')}")
                self._stop_event.set()

        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
        except (ValueError, RuntimeError):
            pass

        try:
            await self.runtime.run(self)
        finally:
            if self._qubit_routing_slot_workspace is not None:
                self._qubit_routing_slot_workspace.cleanup()

    # ---------- Checkpoint loading ----------

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path."""
        restored = self.checkpoint_manager.load(path, self.db, self.selector)
        # Restore gen_id counter
        if "gen_id_counter" in restored:
            self.generator.set_gen_id_counter(restored["gen_id_counter"])
        self.instance_id = restored["instance_id"]
        self.completed_evaluations = restored["completed_evaluations"]
        self.generation_attempts = restored["generation_attempts"]
        self.generation_failures = restored.get("generation_failures", 0)
        self.generation_cancellations = restored.get("generation_cancellations", 0)
        self.evaluation_failures = restored.get("evaluation_failures", 0)
        self._failure_records = restored.get("failure_records", [])
        self.best_score = restored["best_score"]
        self.best_node_id = restored["best_node_id"]
        self.checkpoint_dir = self.checkpoint_manager.checkpoint_dir
        self._shared_construction_dir = os.path.join(self.checkpoint_dir, "shared_constructions")
        os.makedirs(self._shared_construction_dir, exist_ok=True)
        restored_shared = restored.get("shared_constructions", {})
        self._chain_shared_constructions = {
            i: None for i in range(max(1, self.config.num_chains))
        }
        for chain_idx_str, ref in restored_shared.items():
            try:
                chain_idx = int(chain_idx_str)
            except (TypeError, ValueError):
                continue
            if 0 <= chain_idx < len(self._chain_shared_constructions):
                self._chain_shared_constructions[chain_idx] = dict(ref) if ref is not None else None

        # Restore per-chain best scores
        restored_chain_best = restored.get("chain_best_scores", {})
        if restored_chain_best:
            for chain_idx_str, score in restored_chain_best.items():
                try:
                    chain_idx = int(chain_idx_str)
                except (TypeError, ValueError):
                    continue
                if 0 <= chain_idx < len(self._chain_best_scores):
                    self._chain_best_scores[chain_idx] = float(score)
        else:
            # Backward compat: reconstruct from policy chain data
            policy_state = self.selector.state_dict()
            chains = policy_state.get("chains", {})
            for chain_idx_str, node_ids in chains.items():
                try:
                    chain_idx = int(chain_idx_str)
                except (TypeError, ValueError):
                    continue
                if 0 <= chain_idx < len(self._chain_best_scores):
                    best_in_chain = max(
                        (self.db.nodes[nid].score for nid in node_ids if nid in self.db.nodes),
                        default=-float("inf"),
                    )
                    self._chain_best_scores[chain_idx] = best_in_chain

        self._static_prompt = None
        self._static_inspiration_ids = None
        self._static_shared_construction_id = None

        # Restore interval tracker for db_show_interval
        if self.config.db_show_interval > 0:
            self._last_db_display_interval = self.completed_evaluations // self.config.db_show_interval

        # Gather DB and policy stats
        db_node_count = len(self.db.nodes)
        policy_state = self.selector.state_dict()
        num_chains = policy_state.get("num_chains", 0)
        chains = policy_state.get("chains", {})
        chain_prompt_count = policy_state.get("chain_prompt_count", {})
        prompt_budget = policy_state.get("prompt_budget", {})
        chains_with_nodes = sum(1 for c in chains.values() if c)
        total_budget_remaining = sum(
            prompt_budget.get(str(i), 0) - chain_prompt_count.get(str(i), 0)
            for i in range(num_chains)
        )

        rich_print(Panel(
            f"[bold cyan]Checkpoint Loaded[/bold cyan]\n"
            f"[dim]Instance:[/dim] [cyan]{self.instance_id}[/cyan]\n"
            f"[dim]Path:[/dim] [cyan]{path}[/cyan]\n"
            f"[dim]Attempts:[/dim] {self.generation_attempts} | "
            f"[dim]Gen fails:[/dim] {self.generation_failures} | "
            f"[dim]Gen cancels:[/dim] {self.generation_cancellations} | "
            f"[dim]Eval rejects:[/dim] {self.evaluation_failures} | "
            f"[dim]Evals:[/dim] {self.completed_evaluations} | "
            f"[dim]Best:[/dim] [green]{self.best_score:.8f}[/green]\n"
            f"[dim]DB nodes:[/dim] {db_node_count} | "
            f"[dim]Chains:[/dim] {chains_with_nodes}/{num_chains} with nodes | "
            f"[dim]Budget remaining:[/dim] {total_budget_remaining} prompts",
            border_style="cyan", title="[bold]Resume[/bold]",
        ))

    @staticmethod
    def _resolve_resume_checkpoint_dir(resume_path: str) -> str:
        """Resolve the instance directory from a resume path.

        Handles both:
        - instance-xxx/ (returns as-is)
        - instance-xxx/db_state_xxx/ (returns parent instance-xxx/)
        """
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path does not exist: {resume_path}")

        basename = os.path.basename(resume_path.rstrip("/"))
        if basename.startswith("db_state_") or basename.startswith("ckpt_"):
            # Path points to a checkpoint dir, go up to instance dir
            return os.path.dirname(resume_path)
        else:
            # Path is already the instance dir
            return resume_path


# ============================================================================
# Task-local helpers
# ============================================================================

@lru_cache(maxsize=1)
def _load_qubit_routing_slot_workspace_module():
    """Load the qubit-routing slot workspace helper from the dataset tree."""
    module_path = Path(__file__).resolve().parents[2] / "datasets" / "qubit_routing" / "slot_workspace.py"
    spec = importlib.util.spec_from_file_location(
        "simpletes_qubit_routing_slot_workspace",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load qubit-routing slot workspace helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_qubit_routing_evaluator(evaluator_path: str) -> bool:
    """Return whether *evaluator_path* belongs to the built-in qubit-routing task."""
    qubit_routing_root = (
        Path(__file__).resolve().parents[2] / "datasets" / "qubit_routing"
    ).resolve()
    configured_evaluator = Path(evaluator_path).expanduser()
    if not configured_evaluator.is_absolute():
        configured_evaluator = (Path.cwd() / configured_evaluator).resolve()
    else:
        configured_evaluator = configured_evaluator.resolve()
    return configured_evaluator.is_relative_to(qubit_routing_root)
