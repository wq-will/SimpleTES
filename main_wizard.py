#!/usr/bin/env python3
"""Interactive launcher for SimpleTES.

Discovers tasks under datasets/, walks the user through configuration,
and invokes main.py with the resulting arguments.

Usage:
    python main_wizard.py                       # interactive wizard
    python main_wizard.py --load-profile NAME   # skip wizard, run saved profile
    python main_wizard.py --save-profile NAME   # save the resulting config
    python main_wizard.py --list-profiles       # list saved profiles
    python main_wizard.py --dry-run             # print the command without running

Any trailing args are forwarded to main.py unchanged, e.g.:
    python main_wizard.py -- --num-chains 32 --resume path/to/ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
PROFILES_DIR = REPO_ROOT / "profiles"
KEYS_DIR = REPO_ROOT / "keys"

console = Console()

POLICIES = ["balance", "puct", "rpucg", "llm_puct", "llm_rpucg", "llm_elite"]

# Task-family runtime tips shown after selection (env vars that materially
# affect the run but live outside main.py's arg surface).
FAMILY_TIPS: Dict[str, str] = {
    "ahc": (
        "AHC tasks run the evaluator inside Docker. Recommended env:\n"
        "  export AHC_CACHE_DIR=datasets/ahc/cache\n"
        "  export AHC_CASE_WORKERS=6        # parallel test cases per eval\n"
        "  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
    ),
}


# ==================== Task discovery ====================

@dataclass
class Task:
    family: str
    subtask: str
    init_program: Path
    evaluator: Path
    instruction: Path
    needs_setup: bool = False
    setup_hint: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.family}/{self.subtask}"


_INIT_EXTS = ("py", "rs", "cpp")


def discover_tasks() -> List[Task]:
    tasks: List[Task] = []
    init_paths = {
        p
        for ext in _INIT_EXTS
        for p in DATASETS_DIR.glob(f"*/*/init_program.{ext}")
    }
    for init_path in sorted(init_paths):
        subtask_dir = init_path.parent
        family = subtask_dir.parent.name
        subtask = subtask_dir.name
        evaluator = subtask_dir / "evaluator.py"
        if not evaluator.exists():
            continue

        instruction = subtask_dir / f"{subtask}.txt"
        if not instruction.exists():
            txts = sorted(subtask_dir.glob("*.txt"))
            if not txts:
                continue
            instruction = txts[0]

        needs_setup, hint = _check_setup(family)
        tasks.append(Task(family, subtask, init_path, evaluator, instruction, needs_setup, hint))
    return tasks


def _check_setup(family: str) -> tuple[bool, Optional[str]]:
    manifest = DATASETS_DIR / family / "data_manifest.json"
    if not manifest.exists():
        return False, None
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return False, None
    required = data.get("required_files", [])
    family_dir = DATASETS_DIR / family
    missing = [rel for rel in required if not (family_dir / rel).exists()]
    if not missing:
        return False, None
    return True, f"python scripts/prepare_task.py --task {family}"


# ==================== Wizard ====================

def choose_task(tasks: List[Task]) -> Task:
    families: Dict[str, List[Task]] = {}
    for t in tasks:
        families.setdefault(t.family, []).append(t)

    family_choices = []
    for fam in sorted(families):
        fam_tasks = families[fam]
        needs = sum(1 for t in fam_tasks if t.needs_setup)
        extra = f"  ({len(fam_tasks)} subtask{'s' if len(fam_tasks) > 1 else ''}"
        if needs:
            extra += f", {needs} need setup"
        extra += ")"
        family_choices.append(questionary.Choice(title=fam + extra, value=fam))

    family = _ask(questionary.select("Task family:", choices=family_choices))

    fam_tasks = families[family]
    if len(fam_tasks) == 1:
        return fam_tasks[0]

    subtask_choices = [
        questionary.Choice(title=f"{'⚠' if t.needs_setup else '✓'}  {t.subtask}", value=t)
        for t in sorted(fam_tasks, key=lambda x: x.subtask)
    ]
    return _ask(questionary.select(f"{family} subtask:", choices=subtask_choices))


def _ask(question):
    """Drive a questionary prompt and exit cleanly if the user cancels (Ctrl+C / ESC)."""
    answer = question.ask()
    if answer is None:
        sys.exit(0)
    return answer


def _is_positive_int(s: str) -> bool:
    try:
        return int(s) > 0
    except ValueError:
        return False


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def prompt_int(label: str, default: int) -> int:
    return int(_ask(questionary.text(label, default=str(default), validate=_is_positive_int)))


def prompt_float(label: str, default: float) -> float:
    return float(_ask(questionary.text(label, default=str(default), validate=_is_float)))


def prompt_text(label: str, default: str = "", allow_empty: bool = True) -> str:
    while True:
        resp = _ask(questionary.text(label, default=default)).strip()
        if resp or allow_empty:
            return resp
        console.print("[red]This field is required.[/red]")


def prompt_config(task: Task, defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(defaults)
    cfg["task"] = task.label

    console.print(
        "[dim]Model uses LiteLLM format, e.g. gpt-4o / gemini-2.0-flash / ollama/qwen2.5-coder.\n"
        "Any provider supported by LiteLLM works — you supply the api-base and api-key.[/dim]"
    )
    cfg["model"] = prompt_text(
        "Model:",
        default=cfg.get("model", ""),
        allow_empty=False,
    )
    cfg["api_base"] = prompt_text(
        "API base URL (blank = provider default):",
        default=cfg.get("api_base", _auto_read(KEYS_DIR / "openai_api_base")),
    )
    cfg["api_key"] = prompt_text(
        "API key (blank = rely on env vars like OPENAI_API_KEY):",
        default=cfg.get("api_key", _auto_read(KEYS_DIR / "openai_api_key")),
    )

    cfg["max_generations"] = prompt_int("Max generations:", cfg.get("max_generations", 100))
    cfg["selector"] = _ask(questionary.select(
        "Selector:",
        choices=POLICIES,
        default=cfg.get("selector", "balance"),
    ))

    if _ask(questionary.confirm("Show advanced options?", default=False)):
        cfg["num_chains"] = prompt_int("num_chains:", cfg.get("num_chains", 4))
        cfg["k_candidates"] = prompt_int("k_candidates:", cfg.get("k_candidates", 4))
        cfg["eval_concurrency"] = prompt_int("eval_concurrency:", cfg.get("eval_concurrency", 8))
        cfg["gen_concurrency"] = prompt_int("gen_concurrency:", cfg.get("gen_concurrency", 4))
        cfg["temperature"] = prompt_float("temperature:", cfg.get("temperature", 0.7))

    return cfg


def _auto_read(path: Path) -> str:
    """Return stripped file contents if the file exists, else empty string."""
    if path.exists():
        try:
            return path.read_text().strip()
        except Exception:
            return ""
    return ""


def _compute_safe_restart_every_n(max_generations: int, num_chains: int, k_candidates: int) -> int:
    """Pick a restart_every_n that divides every non-zero per-chain prompt budget.

    cli._validate_chain_policy_args requires prompt_budget % restart_every_n == 0
    for each chain. Returning the minimum non-zero prompt_budget always
    satisfies that and leaves restart effectively a no-op for short runs.
    """
    from simpletes.policies.base import compute_chain_budgets

    budgets = [
        (chain_budget + k_candidates - 1) // k_candidates
        for chain_budget in compute_chain_budgets(max_generations, num_chains).values()
        if chain_budget > 0
    ]
    return min(budgets) if budgets else 1


# ==================== Command construction ====================

def _resolve_task_paths(cfg: Dict[str, Any], tasks: List[Task]) -> Task:
    match = next((t for t in tasks if t.label == cfg["task"]), None)
    if match is None:
        console.print(f"[red]Task not found: {cfg['task']}[/red]")
        sys.exit(1)
    return match


def build_command(cfg: Dict[str, Any], task: Task, extra: List[str]) -> List[str]:
    args: List[str] = [
        sys.executable, "main.py",
        "--init-program", str(task.init_program.relative_to(REPO_ROOT)),
        "--evaluator", str(task.evaluator.relative_to(REPO_ROOT)),
        "--instruction", str(task.instruction.relative_to(REPO_ROOT)),
        "--max-generations", str(cfg["max_generations"]),
        "--model", cfg["model"],
        "--selector", cfg["selector"],
    ]
    for key in ("num_chains", "k_candidates", "eval_concurrency", "gen_concurrency", "temperature"):
        if key in cfg:
            args += [f"--{key.replace('_', '-')}", str(cfg[key])]

    # Auto-compute restart_every_n to satisfy the engine's divisibility check
    # unless the user pinned one via advanced prompts or profile.
    if "restart_every_n" in cfg:
        args += ["--restart-every-n", str(cfg["restart_every_n"])]
    else:
        safe = _compute_safe_restart_every_n(
            cfg["max_generations"],
            cfg.get("num_chains", 4),
            cfg.get("k_candidates", 4),
        )
        args += ["--restart-every-n", str(safe)]

    # Only pass --output-path if the profile or wizard set one; otherwise let
    # EngineConfig.output_path ("checkpoints") apply.
    if cfg.get("output_path"):
        args += ["--output-path", cfg["output_path"]]

    if cfg.get("api_base"):
        args += ["--api-base", cfg["api_base"]]
    if cfg.get("api_key"):
        args += ["--api-key", cfg["api_key"]]

    args += extra
    return args


def show_review(cfg: Dict[str, Any], task: Task, cmd: List[str]) -> None:
    tbl = Table(show_header=False, border_style="cyan", title="Run Configuration")
    tbl.add_column(style="bold cyan", no_wrap=True)
    tbl.add_column()
    for key in ("task", "model", "api_base", "max_generations", "selector",
                "num_chains", "k_candidates", "eval_concurrency", "gen_concurrency",
                "temperature", "restart_every_n"):
        if cfg.get(key) not in (None, ""):
            tbl.add_row(key, str(cfg[key]))
    if cfg.get("api_key"):
        tbl.add_row("api_key", "*** (hidden)")
    console.print(tbl)

    # Mask api-key when displaying
    display = []
    skip_next = False
    for tok in cmd:
        if skip_next:
            display.append("***")
            skip_next = False
        elif tok == "--api-key":
            display.append(tok)
            skip_next = True
        else:
            display.append(tok)
    console.print(Panel(" \\\n  ".join(display), title="Command", border_style="dim"))


# ==================== Profiles ====================

def list_profile_names() -> List[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(name: str) -> Dict[str, Any]:
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        console.print(f"[red]Profile not found: {name}[/red]")
        sys.exit(1)
    return json.loads(path.read_text())


def save_profile(name: str, cfg: Dict[str, Any]) -> None:
    PROFILES_DIR.mkdir(exist_ok=True)
    # Never persist the raw api_key to disk; keep api_base since it's not secret.
    safe = {k: v for k, v in cfg.items() if k != "api_key"}
    (PROFILES_DIR / f"{name}.json").write_text(json.dumps(safe, indent=2, sort_keys=True))


# ==================== Entry ====================

def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Interactive SimpleTES launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-profile", metavar="NAME", help="Run a saved profile (skips wizard)")
    parser.add_argument("--save-profile", metavar="NAME", help="Save the resolved config as a profile")
    parser.add_argument("--list-profiles", action="store_true", help="List saved profiles and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()

    if args.list_profiles:
        names = list_profile_names()
        if names:
            for n in names:
                console.print(n)
        else:
            console.print("[dim](no profiles saved)[/dim]")
        return

    tasks = discover_tasks()
    if not tasks:
        console.print(f"[red]No tasks found under {DATASETS_DIR}[/red]")
        sys.exit(1)

    # ---- Resolve config (profile or wizard) ----
    if args.load_profile:
        cfg = load_profile(args.load_profile)
        console.print(f"[green]Loaded profile:[/green] {args.load_profile}")
    else:
        console.print(Panel.fit("SimpleTES Launcher", style="bold cyan"))

        profiles = list_profile_names()
        starting_cfg: Dict[str, Any] = {}
        if profiles:
            action = _ask(questionary.select(
                "How would you like to start?",
                choices=["Configure a new run", *[f"Start from profile: {p}" for p in profiles]],
            ))
            if action.startswith("Start from profile: "):
                starting_cfg = load_profile(action.removeprefix("Start from profile: "))

        if starting_cfg and "task" in starting_cfg:
            task = next((t for t in tasks if t.label == starting_cfg["task"]), None)
            if task is None:
                console.print(f"[yellow]Profile task unavailable ({starting_cfg['task']}), picking manually[/yellow]")
                task = choose_task(tasks)
        else:
            task = choose_task(tasks)

        if task.needs_setup:
            console.print(f"[yellow]⚠ {task.label} needs setup.  Hint: {task.setup_hint}[/yellow]")
            if not _ask(questionary.confirm("Continue anyway?", default=False)):
                sys.exit(0)

        tip = FAMILY_TIPS.get(task.family)
        if tip:
            console.print(Panel(tip, title=f"{task.family} tip", border_style="yellow"))

        cfg = prompt_config(task, starting_cfg)

    task = _resolve_task_paths(cfg, tasks)
    cmd = build_command(cfg, task, extra)
    show_review(cfg, task, cmd)

    if args.save_profile:
        save_profile(args.save_profile, cfg)
        console.print(f"[green]Saved profile:[/green] {args.save_profile}")

    if args.dry_run:
        return

    if not args.load_profile and not _ask(questionary.confirm("Run now?", default=True)):
        return

    os.chdir(REPO_ROOT)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
