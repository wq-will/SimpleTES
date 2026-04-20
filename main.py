"""
SimpleTES CLI entry point.

Usage:
    python main.py --init-program path/to/init.py --evaluator path/to/evaluator.py --instruction path/to/task.txt
"""
from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpletes.engine import SimpleTESEngine

from simpletes.cli import (
    _handle_list_policies,
    _validate_chain_policy_args,
    _validate_policy,
    _validate_eval_venv_args,
    _validate_required_args,
    _validate_init_eval_args,
    build_parser,
)
from simpletes.config import build_config_from_args, examine_args
from simpletes.engine import SimpleTESEngine
from simpletes.engine.runtime import LocalRuntime
from simpletes.policies import available_policies


def _emergency_save_checkpoint(engine: "SimpleTESEngine") -> None:
    """Save checkpoint on KeyboardInterrupt."""
    from rich import print as rich_print

    print("\n[Ctrl+C] Interrupted - saving checkpoint before exit...")
    try:
        # Create a new event loop since the previous one was interrupted
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(engine._write_checkpoint())
            print("[Ctrl+C] Checkpoint saved.")
            # Show final results panel
            rich_print(engine._final_results_panel())
        finally:
            loop.close()
    except KeyboardInterrupt:
        print("[Ctrl+C] Interrupted again - forcing exit without save.")
    except Exception as e:
        print(f"[Ctrl+C] Failed to save checkpoint: {e}")


def main():
    parser = build_parser(mode="single")
    args = parser.parse_args()

    # Handle --list-policies (early exit, no other args needed)
    _handle_list_policies(args)

    # Validate required arguments
    _validate_required_args(args, parser, mode="single")
    _validate_init_eval_args(args, parser)
    _validate_eval_venv_args(args, parser)

    # Validate policy name before building config
    policies = available_policies()
    _validate_policy(args, policies, show_hint=True)
    _validate_chain_policy_args(args, parser)

    # Check data readiness and auto-download if needed
    from simpletes.utils.task_prep import check_and_prepare_task
    check_and_prepare_task(args.evaluator)

    examine_args(args, mode="single", policies=set(policies))

    # Build config from args
    config = build_config_from_args(args)

    if not getattr(args, "skip_preflight", False):
        _run_preflight(config)

    engine = SimpleTESEngine(config, runtime=LocalRuntime(), resume_path=args.resume)

    if args.resume:
        engine.load_checkpoint(args.resume)

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        # Signal handler should have already triggered graceful shutdown,
        # but if asyncio.run() was interrupted before cleanup, ensure checkpoint is saved
        _emergency_save_checkpoint(engine)


def _run_preflight(config) -> None:
    """Validate LLM credentials with a 1-token ping before starting the engine.

    Only runs for the litellm backend — the vllm_token_forcing client has
    different bring-up semantics and doesn't suffer the same opaque
    BrokenProcessPool failure mode.
    """
    if config.llm_backend != "litellm":
        return

    from rich import print as rich_print

    from simpletes.llm import LLMCallError, LLMClient

    client = LLMClient(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        api_base=config.api_base,
        timeout=config.timeout,
        retry=0,
        pool_size=0,  # preflight runs synchronously in the main process
        reasoning_effort=config.reasoning_effort,
    )
    try:
        client.preflight()
    except LLMCallError as err:
        rich_print("[bold red]✗ LLM preflight failed[/bold red]")
        rich_print(f"  [dim]Model:    [/dim]{err.model}")
        rich_print(f"  [dim]API base: [/dim]{err.api_base or '(provider default)'}")
        rich_print(f"  [dim]Error:    [/dim][red]{err.error_type}[/red]")
        rich_print(f"  [dim]Message:  [/dim]{err.message}")
        rich_print(
            "\n[dim]Fix the credentials or pass --skip-preflight to bypass this check "
            "(e.g. if your local backend is still starting up).[/dim]"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
