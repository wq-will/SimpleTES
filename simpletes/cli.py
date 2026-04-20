"""
Shared CLI builder for SimpleTES.
"""
from __future__ import annotations

import argparse
import sys

from simpletes.config import EngineConfig
from simpletes.policies import available_policies


def build_parser(*, mode: str = "single") -> argparse.ArgumentParser:
    if mode != "single":
        raise ValueError(f"Unknown mode: {mode}")

    parser = argparse.ArgumentParser(
        description="SimpleTES: LLM-driven code evolution"
    )

    # List policies option (must be early to allow early exit)
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List all available inspiration policies and exit",
    )

    # Required arguments (not required if --list-policies is used)
    parser.add_argument("--init-program", help="Path to initial program")
    parser.add_argument("--evaluator", help="Path to evaluator module")
    parser.add_argument("--instruction", help="Path to instruction file")

    # Budget / termination
    parser.add_argument(
        "--max-generations",
        type=int,
        default=EngineConfig.max_generations,
        help=f"Number of generation prompts to schedule (default: {EngineConfig.max_generations})",
    )
    parser.add_argument(
        "--early-stop-score",
        type=float,
        default=None,
        help="Stop when best score reaches this threshold",
    )

    # Prompt shape
    parser.add_argument(
        "--num-inspirations",
        type=int,
        default=EngineConfig.num_inspirations,
        help=(
            "Number of in-context inspirations per prompt "
            f"(0 = init_program only; default: {EngineConfig.num_inspirations})"
        ),
    )
    parser.add_argument(
        "--min-inspirations-cnt",
        type=int,
        default=EngineConfig.min_inspirations_cnt,
        help="Minimum inspirations per prompt (enable sampling when paired with --max-inspirations-cnt)",
    )
    parser.add_argument(
        "--max-inspirations-cnt",
        type=int,
        default=EngineConfig.max_inspirations_cnt,
        help="Maximum inspirations per prompt (enable sampling when paired with --min-inspirations-cnt)",
    )

    # Concurrency
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=EngineConfig.eval_concurrency,
        help=f"Max concurrent evaluations (default: {EngineConfig.eval_concurrency})",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=EngineConfig.eval_timeout,
        help=f"Evaluation timeout in seconds (default: {EngineConfig.eval_timeout:g})",
    )
    parser.add_argument(
        "--init-eval-repeats",
        type=int,
        default=EngineConfig.init_eval_repeats,
        help=(
            "Number of concurrent evaluations for init_program; "
            f"best score is kept (default: {EngineConfig.init_eval_repeats})"
        ),
    )
    parser.add_argument(
        "--eval-venv",
        type=str,
        default=None,
        help=(
            "Virtualenv directory to use for running evaluations. "
            "If unset, auto-uses <task_dir>/venv when present "
            "(for built-in datasets: datasets/<task>/venv). "
            "If <task_dir>/uv.lock exists but the venv is missing, SimpleTES will warn and fall back to the main Python environment."
        ),
    )
    parser.add_argument(
        "--gen-concurrency",
        type=int,
        default=EngineConfig.gen_concurrency,
        help=f"Number of concurrent LLM generation workers (default: {EngineConfig.gen_concurrency})",
    )

    # Inspiration policy
    parser.add_argument(
        "--selector",
        type=str,
        default=EngineConfig.selector,
        help=f"Inspiration selection policy name (default: {EngineConfig.selector}). Use --list-policies to see options.",
    )
    parser.add_argument(
        "--exploitation-ratio",
        type=float,
        default=EngineConfig.exploitation_ratio,
        help=f"(balance) Probability of sampling from elite (default: {EngineConfig.exploitation_ratio:g})",
    )
    parser.add_argument(
        "--exploration-ratio",
        type=float,
        default=EngineConfig.exploration_ratio,
        help=f"(balance) Probability of sampling from mid-tier (default: {EngineConfig.exploration_ratio:g})",
    )
    parser.add_argument(
        "--elite-ratio",
        type=float,
        default=EngineConfig.elite_ratio,
        help=f"(balance) Top ratio considered elite (default: {EngineConfig.elite_ratio:g})",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=EngineConfig.num_chains,
        help=f"(balance/puct) Number of independent evolution chains (default: {EngineConfig.num_chains})",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=EngineConfig.k_candidates,
        help=f"(balance/puct) Candidates per prompt, only best is kept (default: {EngineConfig.k_candidates})",
    )
    parser.add_argument(
        "--no-stream-k-candidates",
        action="store_false",
        dest="stream_k_candidates",
        default=True,
        help="Disable streaming k candidates as independent jobs (default: enabled)",
    )
    parser.add_argument(
        "--debug-prompt-lines",
        type=int,
        default=0,
        help="Print first N lines of generator prompt for debugging (default: 0 = disabled)",
    )
    parser.add_argument(
        "--restart-every-n",
        type=int,
        default=EngineConfig.restart_every_n,
        help=(
            "(chain policies) Reset a chain to its best node after every N kept nodes "
            f"(default: {EngineConfig.restart_every_n})"
        ),
    )
    parser.add_argument(
        "--stream-k-candidates",
        action="store_true",
        help="Dispatch k candidates as independent generation jobs (k=1 per job) while preserving local-best batch semantics",
    )
    parser.add_argument(
        "--include-construction",
        action="store_true",
        help="Enable per-chain shared construction: save and provide GLOBAL_BEST_CONSTRUCTION from each chain's best solution to the LLM prompt (default: disabled)",
    )

    # PUCT policy parameters
    parser.add_argument(
        "--puct-c",
        type=float,
        default=EngineConfig.puct_c,
        help=f"(puct) Exploration coefficient, higher = more exploration (default: {EngineConfig.puct_c:g})",
    )

    # RPUCG policy parameters
    parser.add_argument(
        "--rpucg-gamma",
        type=float,
        default=EngineConfig.rpucg_gamma,
        help=f"(rpucg) Decay factor for value backpropagation (default: {EngineConfig.rpucg_gamma:g})",
    )

    # LLM policy parameters
    parser.add_argument(
        "--llm-policy-model",
        type=str,
        default=None,
        help="Model for llm-policy (default: --model)",
    )
    parser.add_argument(
        "--llm-policy-api-base",
        type=str,
        default=None,
        help="Custom API base URL for llm-policy model (default: --api-base)",
    )
    parser.add_argument(
        "--llm-policy-api-key",
        type=str,
        default=None,
        help="API key override for llm-policy model (default: --api-key)",
    )
    parser.add_argument(
        "--llm-policy-pool-size",
        type=int,
        default=None,
        help="(llm_elite only) Elite pool size per chain (default: 15)",
    )
    parser.add_argument(
        "--elite-selection-strategy",
        type=str,
        default="linear_rank",
        choices=["linear_rank", "balance", "all"],
        help="(llm_elite only) Selection strategy: linear_rank, balance, or all (default: linear_rank)",
    )

    # Backpressure
    parser.add_argument(
        "--backpressure-multiplier",
        type=float,
        default=EngineConfig.backpressure_multiplier,
        help=f"Multiplier for backpressure threshold (default: {EngineConfig.backpressure_multiplier:g})",
    )

    # Checkpoints
    parser.add_argument(
        "--log-interval",
        type=int,
        default=EngineConfig.log_interval,
        help=f"Checkpoint save interval in evaluations (default: {EngineConfig.log_interval})",
    )
    parser.add_argument(
        "--db-show-interval",
        type=int,
        default=EngineConfig.db_show_interval,
        help=f"Interval for printing database state (default: {EngineConfig.db_show_interval})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=EngineConfig.output_path,
        help=f"Output path for checkpoints (default: {EngineConfig.output_path})",
    )
    parser.add_argument(
        "--save-llm-io",
        action="store_true",
        help="Save full LLM input/output and token usage in checkpoints (large files)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Use gzip compression for checkpoint nodes file",
    )

    # LLM configuration
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["litellm", "vllm_token_forcing"],
        default=None,
        help="LLM backend to use (litellm, vllm_token_forcing). Defaults to litellm.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EngineConfig.model,
        help=f"LLM model to use (default: {EngineConfig.model})",
    )
    parser.add_argument(
        "--disable-reflection",
        action="store_true",
        help="Disable reflection summaries for batch-best nodes (uses --llm-policy-* config)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=EngineConfig.tokenizer_path,
        help="Local tokenizer path (vllm_token_forcing backend only)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=EngineConfig.temperature,
        help=f"LLM temperature (default: {EngineConfig.temperature:g})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=EngineConfig.max_tokens,
        help=f"LLM max tokens (default: {EngineConfig.max_tokens})",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=EngineConfig.api_base,
        help="Custom API base URL / proxy base_url for LiteLLM (default: None, use provider/env defaults)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=EngineConfig.api_key,
        help="API key override for LiteLLM/provider (default: None, use provider/env defaults)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=EngineConfig.retry,
        help=f"Number of retries for LLM calls (default: {EngineConfig.retry})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=EngineConfig.timeout,
        help=f"LLM request timeout in seconds (default: {EngineConfig.timeout:g})",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=EngineConfig.max_total_tokens,
        help="Total token budget (prompt + completion). If set, caps completion tokens to fit.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=EngineConfig.reasoning_effort,
        help=f"Reasoning effort level for supported models (default: {EngineConfig.reasoning_effort})",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="(vllm_token_forcing) Phase 1 total budget in tokens (prompt + reasoning). None = auto from context_window - response_budget",
    )
    parser.add_argument(
        "--response-budget",
        type=int,
        default=EngineConfig.response_budget,
        help=f"(vllm_token_forcing) Reserved tokens for phase 2 final response (default: {EngineConfig.response_budget})",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=EngineConfig.context_window,
        help=f"(vllm_token_forcing) vLLM server context window size (default: {EngineConfig.context_window})",
    )
    # Resume
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume from")

    # Preflight
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the 1-token LLM ping at startup (use when your backend is not yet ready)",
    )

    return parser


def _handle_list_policies(args) -> None:
    if getattr(args, "list_policies", False):
        policies = available_policies()
        print("Available inspiration policies:")
        for policy in policies:
            print(f"  - {policy}")
        sys.exit(0)


def _validate_required_args(args, parser: argparse.ArgumentParser, *, mode: str = "single") -> None:
    if mode != "single":
        raise ValueError(f"Unknown mode: {mode}")
    missing_required = not args.init_program or not args.evaluator or not args.instruction
    if missing_required:
        parser.error("--init-program, --evaluator, and --instruction are required (unless using --list-policies)")


def _validate_init_eval_args(args, parser: argparse.ArgumentParser) -> None:
    init_eval_repeats = getattr(args, "init_eval_repeats", EngineConfig.init_eval_repeats)
    if init_eval_repeats <= 0:
        parser.error("--init-eval-repeats must be > 0")


def _validate_eval_venv_args(args, parser: argparse.ArgumentParser) -> None:
    """
    Validate --eval-venv if provided.

    We fail fast if an explicit path is invalid to avoid silently running the
    wrong environment.
    """
    eval_venv = getattr(args, "eval_venv", None)
    if not eval_venv:
        return

    from pathlib import Path

    from simpletes.config import python_in_venv

    venv_dir = Path(eval_venv).expanduser()
    if not venv_dir.exists():
        parser.error(f"--eval-venv path does not exist: {eval_venv}")
    if not venv_dir.is_dir():
        parser.error(f"--eval-venv must be a directory: {eval_venv}")

    py = python_in_venv(venv_dir)
    if py is None:
        parser.error(
            f"--eval-venv does not look like a virtualenv: {eval_venv} "
            "(missing bin/python or Scripts/python.exe)"
        )


def _validate_policy(args, policies, *, show_hint: bool) -> None:
    if args.selector not in policies:
        print(f"Error: Unknown policy '{args.selector}'", file=sys.stderr)
        print(f"Available policies: {', '.join(policies)}", file=sys.stderr)
        if show_hint:
            print("\nUse --list-policies to see all available policies.", file=sys.stderr)
        sys.exit(1)


def _validate_chain_policy_args(args, parser: argparse.ArgumentParser) -> None:
    if args.selector not in {"balance", "puct", "rpucg", "llm_puct", "llm_rpucg", "llm_elite"}:
        return

    restart_every_n = getattr(args, "restart_every_n", EngineConfig.restart_every_n)
    if restart_every_n <= 0:
        parser.error("--restart-every-n must be > 0")

    num_chains = max(1, getattr(args, "num_chains", EngineConfig.num_chains))
    max_generations = max(0, getattr(args, "max_generations", EngineConfig.max_generations))
    k_candidates = max(1, getattr(args, "k_candidates", EngineConfig.k_candidates))

    base = max_generations // num_chains
    remainder = max_generations % num_chains
    invalid = {}
    for chain_idx in range(num_chains):
        chain_budget = base + (1 if chain_idx < remainder else 0)
        prompt_budget = (chain_budget + k_candidates - 1) // k_candidates if chain_budget > 0 else 0
        if prompt_budget > 0 and prompt_budget % restart_every_n != 0:
            invalid[chain_idx] = prompt_budget

    if invalid:
        parser.error(
            "--restart-every-n must divide each non-zero per-chain prompt budget; "
            f"got restart_every_n={restart_every_n}, prompt_budget={invalid}"
        )
