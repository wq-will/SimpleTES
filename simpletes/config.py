"""
Configuration dataclass for SimpleTES.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simpletes.evaluator import rich_print


def python_in_venv(venv_dir: Path) -> Path | None:
    """
    Return the Python executable inside a virtual environment directory.

    Supports both POSIX venvs (<venv>/bin/python) and Windows venvs
    (<venv>/Scripts/python.exe).
    """
    # Prefer Windows layout if present.
    for p in (venv_dir / "Scripts" / "python.exe", venv_dir / "bin" / "python"):
        if p.is_file():
            return p
    return None


def venv_dir_from_python(python_executable: Path) -> Path | None:
    """
    Infer the virtualenv directory from a Python executable path.

    We treat a path as "in a venv" if it matches common layouts and the venv root
    contains a pyvenv.cfg:
      - <venv>/bin/python
      - <venv>/Scripts/python.exe
    """
    try:
        py = python_executable.expanduser().resolve()
    except OSError:
        py = python_executable

    parent = py.parent
    if parent.name not in {"bin", "Scripts"}:
        return None

    venv_dir = parent.parent
    return venv_dir if (venv_dir / "pyvenv.cfg").is_file() else None


def task_dir_from_evaluator_path(evaluator_path: str) -> Path:
    """
    Infer the "task directory" from an evaluator path.

    SimpleTES's built-in datasets are typically laid out as:

        datasets/<task>/<subtask>/evaluator.py

    In that case, we treat the task directory as `datasets/<task>` so a single
    venv can be shared across all subtasks.

    For custom layouts (no `datasets/<task>/...` in the path), we fall back to
    the directory containing the evaluator script.
    """
    eval_path = Path(evaluator_path).expanduser().resolve()

    # Walk up to find the nearest ancestor named "datasets".
    datasets_dir: Path | None = None
    for parent in [eval_path.parent, *eval_path.parents]:
        if parent.name == "datasets":
            datasets_dir = parent
            break

    if datasets_dir is not None:
        try:
            rel = eval_path.relative_to(datasets_dir)
        except ValueError:
            rel = None

        if rel is not None and rel.parts:
            candidate = datasets_dir / rel.parts[0]
            # Guard against pathological layouts like datasets/evaluator.py.
            try:
                if candidate.is_dir():
                    return candidate
            except OSError:
                pass

    return eval_path.parent


def resolve_eval_python(*, evaluator_path: str, eval_venv: str | None) -> str | None:
    """
    Resolve the interpreter to use for evaluation subprocesses.

    Precedence:
      - Explicit venv passed via --eval-venv
      - Auto-detected task venv at <task_dir>/venv
      - None (caller should fall back to sys.executable)
    """
    if eval_venv:
        venv_dir = Path(eval_venv).expanduser().resolve()
        py = python_in_venv(venv_dir)
        return str(py) if py else None

    task_dir = task_dir_from_evaluator_path(evaluator_path)
    auto_venv_dir = task_dir / "venv"
    py = python_in_venv(auto_venv_dir)
    return str(py) if py else None


@dataclass
class EngineConfig:
    """Typed configuration for SimpleTESEngine."""
    init_program: str
    evaluator_path: str
    instruction_path: str
    
    # Budget / termination
    max_generations: int = 100
    early_stop_score: float | None = None
    
    # Prompt shape
    num_inspirations: int = 5
    min_inspirations_cnt: int | None = None
    max_inspirations_cnt: int | None = None
    
    # Concurrency
    eval_concurrency: int = 4
    eval_timeout: float = 3000.0
    init_eval_repeats: int = 16  # Number of concurrent init-program evaluations; keep max score
    eval_venv: str | None = field(default=None, kw_only=True)  # Venv dir requested for evaluations (if any)
    eval_python: str | None = field(default=None, kw_only=True)  # Optional Python executable for eval subprocesses
    gen_concurrency: int = 1
    
    # LLM
    model: str = "gemini/gemini-2.0-flash"
    tokenizer_path: str | None = None
    temperature: float = 0.7
    max_tokens: int = 32768
    api_base: str | None = None
    api_key: str | None = None
    retry: int = 0
    timeout: float | None = 3000
    max_total_tokens: int | None = None  # Total token budget (prompt + completion). If set, caps completion tokens.
    reasoning_effort: str = "medium"  # Reasoning effort level for supported models (low, medium, high)
    llm_backend: str = "litellm"  # litellm | vllm_token_forcing

    # Token forcing (vllm_token_forcing backend)
    reasoning_budget: int | None = 32768  # Phase 1 total budget (prompt + reasoning). None = auto from context_window - response_budget
    response_budget: int = 16384             # Reserved tokens for phase 2 final response
    context_window: int = 49152             # vLLM server context window size
    reflection_mode: bool = False  # Enable reflection summaries (uses llm_policy config)
    
    # Checkpoints
    output_path: str = "checkpoints"
    log_interval: int = 1024
    db_show_interval: int = 16  # Interval for printing database state
    save_llm_io: bool = True  # Save full LLM input/output and token usage (large files)
    use_gzip: bool = False  # Use gzip compression for checkpoint nodes file
    
    # Policy
    selector: str = "balance"
    exploitation_ratio: float = 0.7
    exploration_ratio: float = 0.2
    elite_ratio: float = 0.2
    
    # Chain-based policy parameters (balance, puct)
    num_chains: int = 4  # Number of independent chains
    k_candidates: int = 4  # (balance/puct) Candidates per prompt, keep only best
    stream_k_candidates: bool = True  # Dispatch k candidates as independent k=1 jobs
    restart_every_n: int = 50  # Reset chain to its best node after every N kept nodes
    include_failure_patterns: bool = True  # Include failure patterns in the prompt
    debug_prompt_lines: int = 0  # Print first N lines of generator prompt (0 = disabled)
    include_construction: bool = False  # Enable per-chain shared construction: save and provide GLOBAL_BEST_CONSTRUCTION from each chain's best solution to the LLM prompt

    # PUCT policy parameters
    puct_c: float = 0.5  # Exploration coefficient (higher = more exploration)
    rpucg_gamma: float = 0.8  # Decay factor for RPUCG DAG-aware value backpropagation

    # LLM policy parameters (used by llm_elite policy)
    llm_policy_model: str | None = None
    llm_policy_api_base: str | None = None
    llm_policy_api_key: str | None = None
    llm_policy_pool_size: int | None = None  # Elite pool size, only for llm_elite
    elite_selection_strategy: str = "linear_rank"  # "linear_rank" or "balance"

    # Backpressure
    backpressure_multiplier: float = 0  # Multiplier for backpressure threshold (0 = no backpressure)


# ============================================================================
# Argument examination and warning hints
# ============================================================================

def build_config_from_args(args: Any) -> EngineConfig:
    """Build EngineConfig from argparse Namespace. Every field below is
    registered with a default in cli.build_parser, so we rely on argparse
    rather than duplicating defaults via getattr fallbacks."""
    return EngineConfig(
        init_program=args.init_program,
        evaluator_path=args.evaluator,
        instruction_path=args.instruction,
        eval_venv=args.eval_venv,
        eval_python=resolve_eval_python(evaluator_path=args.evaluator, eval_venv=args.eval_venv),
        max_generations=args.max_generations,
        early_stop_score=args.early_stop_score,
        num_inspirations=args.num_inspirations,
        min_inspirations_cnt=args.min_inspirations_cnt,
        max_inspirations_cnt=args.max_inspirations_cnt,
        eval_concurrency=args.eval_concurrency,
        eval_timeout=args.eval_timeout,
        init_eval_repeats=args.init_eval_repeats,
        gen_concurrency=args.gen_concurrency,
        model=args.model,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=args.api_base,
        api_key=args.api_key,
        retry=args.retry,
        timeout=args.timeout,
        max_total_tokens=args.max_total_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_budget=args.reasoning_budget,
        response_budget=args.response_budget,
        context_window=args.context_window,
        llm_backend=resolve_llm_backend(args),
        reflection_mode=not args.disable_reflection,
        output_path=args.output_path,
        log_interval=args.log_interval,
        db_show_interval=args.db_show_interval,
        save_llm_io=args.save_llm_io,
        use_gzip=args.gzip,
        selector=args.selector,
        exploitation_ratio=args.exploitation_ratio,
        exploration_ratio=args.exploration_ratio,
        elite_ratio=args.elite_ratio,
        num_chains=args.num_chains,
        k_candidates=args.k_candidates,
        restart_every_n=args.restart_every_n,
        stream_k_candidates=args.stream_k_candidates,
        debug_prompt_lines=args.debug_prompt_lines,
        include_construction=args.include_construction,
        puct_c=args.puct_c,
        rpucg_gamma=args.rpucg_gamma,
        llm_policy_model=args.llm_policy_model,
        llm_policy_api_base=args.llm_policy_api_base,
        llm_policy_api_key=args.llm_policy_api_key,
        llm_policy_pool_size=args.llm_policy_pool_size,
        elite_selection_strategy=args.elite_selection_strategy,
        backpressure_multiplier=args.backpressure_multiplier,
    )


def resolve_llm_backend(args: Any) -> str:
    return getattr(args, "llm_backend", None) or "litellm"

def _warn(msg: str) -> None:
    rich_print(f"[yellow]WARN[/yellow] {msg}")


def _maybe_warn_path(label: str, path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.exists():
        _warn(f"{label} path does not exist: {path_value}")


def _maybe_warn_nonpositive(label: str, value: float | None, hint: str) -> None:
    if value is None:
        return
    try:
        if value <= 0:
            _warn(f"{label} {hint} (got {value})")
    except TypeError:
        _warn(f"{label} {hint} (got {value})")


def _maybe_warn_out_of_range(label: str, value: float | None, low: float, high: float) -> None:
    if value is None:
        return
    try:
        if value < low or value > high:
            _warn(f"{label} is out of range [{low}, {high}] (got {value})")
    except TypeError:
        _warn(f"{label} is out of range [{low}, {high}] (got {value})")


def examine_args(args, *, mode: str = "single", policies: set[str]) -> None:
    """
    Emit warnings for unreasonable or risky argument combinations.

    Args:
        args: argparse.Namespace
        mode: must be "single"
        policies: set of valid policy names (for policy-specific checks)
    """
    if mode != "single":
        raise ValueError(f"Unknown mode: {mode}")
    cpu_count = os.cpu_count() or 1

    # Paths
    _maybe_warn_path("--init-program", getattr(args, "init_program", None))
    _maybe_warn_path("--evaluator", getattr(args, "evaluator", None))
    _maybe_warn_path("--instruction", getattr(args, "instruction", None))
    _maybe_warn_path("--tokenizer-path", getattr(args, "tokenizer_path", None))
    _maybe_warn_path("--resume", getattr(args, "resume", None))

    output_path = getattr(args, "output_path", None)
    if output_path:
        output_dir = Path(output_path)
        if output_dir.exists() and output_dir.is_file():
            _warn(f"--output-path points to a file, expected a directory: {output_path}")

    # Concurrency
    gen_concurrency = getattr(args, "gen_concurrency", None)
    eval_concurrency = getattr(args, "eval_concurrency", None)
    _maybe_warn_nonpositive("--gen-concurrency", gen_concurrency, "must be > 0; no LLM workers will start")
    _maybe_warn_nonpositive("--eval-concurrency", eval_concurrency, "must be > 0; no eval workers will start")

    if isinstance(gen_concurrency, int) and isinstance(eval_concurrency, int):
        total_workers = gen_concurrency + eval_concurrency
        if total_workers > cpu_count:
            _warn(
                f"gen+eval concurrency ({total_workers}) exceeds CPU count ({cpu_count}); "
                "this may oversubscribe the node"
            )
        if gen_concurrency > cpu_count:
            _warn(f"--gen-concurrency ({gen_concurrency}) exceeds CPU count ({cpu_count})")
        if eval_concurrency > cpu_count:
            _warn(f"--eval-concurrency ({eval_concurrency}) exceeds CPU count ({cpu_count})")

    # Budget / timeouts
    _maybe_warn_nonpositive("--max-generations", getattr(args, "max_generations", None),
                            "must be > 0; no generations will be scheduled")
    _maybe_warn_nonpositive("--eval-timeout", getattr(args, "eval_timeout", None),
                            "must be > 0; evaluations will immediately timeout")
    timeout = getattr(args, "timeout", None)
    if timeout is not None:
        _maybe_warn_nonpositive("--timeout", timeout, "must be > 0; LLM calls may fail immediately")

    # LLM params
    _maybe_warn_nonpositive("--max-tokens", getattr(args, "max_tokens", None),
                            "must be > 0; LLM output will be empty")
    temperature = getattr(args, "temperature", None)
    _maybe_warn_out_of_range("--temperature", temperature, 0.0, 2.0)
    retry = getattr(args, "retry", None)
    if isinstance(retry, int) and retry < 0:
        _warn(f"--retry is negative (got {retry}); it will be clamped to 0")

    # Prompt shaping
    num_insp = getattr(args, "num_inspirations", None)
    if isinstance(num_insp, int) and num_insp < 0:
        _warn("--num-inspirations must be >= 0; prompts will have no inspirations")
    min_insp = getattr(args, "min_inspirations_cnt", None)
    max_insp = getattr(args, "max_inspirations_cnt", None)
    if min_insp is not None:
        _maybe_warn_nonpositive("--min-inspirations-cnt", min_insp, "must be > 0")
    if max_insp is not None:
        _maybe_warn_nonpositive("--max-inspirations-cnt", max_insp, "must be > 0")
    if (min_insp is None) != (max_insp is None):
        _warn("--min-inspirations-cnt and --max-inspirations-cnt should be set together to enable sampling")
    if isinstance(min_insp, int) and isinstance(max_insp, int) and min_insp > max_insp:
        _warn(
            f"--min-inspirations-cnt ({min_insp}) > --max-inspirations-cnt ({max_insp}); "
            "values will be swapped at runtime"
        )

    # Policy parameters
    policy = getattr(args, "selector", None)
    if policy and policy in policies:
        if policy == "balance":
            exploitation_ratio = getattr(args, "exploitation_ratio", None)
            exploration_ratio = getattr(args, "exploration_ratio", None)
            elite_ratio = getattr(args, "elite_ratio", None)

            _maybe_warn_out_of_range("--exploitation-ratio", exploitation_ratio, 0.0, 1.0)
            _maybe_warn_out_of_range("--exploration-ratio", exploration_ratio, 0.0, 1.0)
            _maybe_warn_out_of_range("--elite-ratio", elite_ratio, 0.0, 1.0)

            if isinstance(exploitation_ratio, (int, float)) and isinstance(exploration_ratio, (int, float)):
                if exploitation_ratio + exploration_ratio > 1.0:
                    _warn(
                        "exploitation+exploration ratio exceeds 1.0; "
                        "sampling probabilities may be invalid"
                    )

        if policy in {"balance", "puct", "rpucg"}:
            num_chains = getattr(args, "num_chains", None)
            _maybe_warn_nonpositive("--num-chains", num_chains, "must be > 0; will be clamped to 1")
            max_generations = getattr(args, "max_generations", None)
            if isinstance(num_chains, int) and isinstance(max_generations, int):
                if max_generations < num_chains:
                    _warn(
                        f"--max-generations ({max_generations}) < --num-chains ({num_chains}); "
                        "some chains will have zero budget"
                    )
            _maybe_warn_nonpositive("--k-candidates", getattr(args, "k_candidates", None),
                                    "must be > 0; no candidates will be kept")

        if policy == "puct":
            _maybe_warn_nonpositive("--puct-c", getattr(args, "puct_c", None),
                                    "must be > 0; exploration will be disabled")

        if policy == "rpucg":
            _maybe_warn_nonpositive("--puct-c", getattr(args, "puct_c", None),
                                    "must be > 0; exploration will be disabled")
            _maybe_warn_out_of_range("--rpucg-gamma", getattr(args, "rpucg_gamma", None), 0.0, 1.0)

    # Checkpointing hints
    log_interval = getattr(args, "log_interval", None)
    if isinstance(log_interval, int) and log_interval <= 0:
        _warn("--log-interval <= 0 disables periodic checkpoints")
    db_show_interval = getattr(args, "db_show_interval", None)
    if isinstance(db_show_interval, int) and db_show_interval <= 0:
        _warn("--db-show-interval <= 0 disables DB state printing")

