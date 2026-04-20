"""
Task directory utilities: data-manifest-driven preparation plus
``requirements.txt`` parsing for the generator prompt.

Data prep: discovers ``data_manifest.json`` next to task dirs, checks
required files, runs declared prepare commands (download, venv, etc.).

Requirements: ``read_available_packages`` / ``load_task_requirements``
parse a task's ``requirements.txt`` into a de-duplicated package name
list that is surfaced to the LLM in the generation prompt.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from simpletes.config import python_in_venv, task_dir_from_evaluator_path

_MANIFEST_NAME = "data_manifest.json"


# ----- requirements.txt parsing -----
# These helpers feed the generator prompt with the set of packages a task
# allows its evolved code to use. They don't install anything.

_REQ_NAME_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<extras>\[[^\]]+\])?")
_REQ_SKIP_PREFIXES = (
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable",
    "--index-url", "--extra-index-url", "--find-links", "--trusted-host", "--no-index",
)


def _extract_requirement_name(line: str) -> str | None:
    content = line.strip()
    if not content or content.startswith("#"):
        return None
    if content.lower().startswith(_REQ_SKIP_PREFIXES):
        return None
    if ";" in content:
        content = content.split(";", 1)[0].strip()
    match = _REQ_NAME_RE.match(content)
    if not match:
        return None
    return f"{match.group('name')}{match.group('extras') or ''}"


def read_available_packages(requirements_path: Path) -> list[str]:
    """Parse requirements.txt into a de-duplicated list of package names (preserving extras)."""
    try:
        text = requirements_path.read_text(encoding="utf-8")
    except OSError:
        return []

    packages: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        name = _extract_requirement_name(raw_line)
        if not name:
            continue
        base_key = name.split("[", 1)[0].lower()
        if base_key in seen:
            # Upgrade an earlier plain entry if we later see one with extras.
            if "[" in name:
                for i, existing in enumerate(packages):
                    if existing.split("[", 1)[0].lower() == base_key and "[" not in existing:
                        packages[i] = name
                        break
            continue
        seen.add(base_key)
        packages.append(name)
    return packages


def load_task_requirements(task_dir: Path) -> tuple[Path | None, list[str]]:
    """Return ``(requirements_path, package_names)`` for a task dir, or ``(None, [])``."""
    requirements_path = task_dir / "requirements.txt"
    if not requirements_path.is_file():
        return None, []
    return requirements_path, read_available_packages(requirements_path)


def find_manifest(evaluator_path: str) -> tuple[Path, dict] | None:
    """Return ``(task_dir, manifest_dict)`` or *None* if no manifest exists."""
    task_dir = task_dir_from_evaluator_path(evaluator_path)
    manifest_path = task_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path) as f:
        return task_dir, json.load(f)


def discover_all_manifests(datasets_root: Path) -> list[tuple[Path, dict]]:
    """Scan *datasets_root* for all ``data_manifest.json`` files."""
    results: list[tuple[Path, dict]] = []
    for manifest_path in sorted(datasets_root.glob(f"*/{_MANIFEST_NAME}")):
        with open(manifest_path) as f:
            results.append((manifest_path.parent, json.load(f)))
    return results


def check_files(task_dir: Path, manifest: dict) -> list[str]:
    """Return list of required files/dirs that are missing."""
    missing: list[str] = []
    for rel in manifest.get("required_files", []):
        if not (task_dir / rel).exists():
            missing.append(rel)
    return missing


def _resolve_command(task_dir: Path, cmd_spec: dict) -> tuple[list[str], Path] | None:
    """Resolve a prepare command to ``(argv, cwd)`` or *None* on error."""
    command = list(cmd_spec["command"])
    cwd = (task_dir / cmd_spec.get("cwd", ".")).resolve()
    venv_rel = cmd_spec.get("venv")

    if command and command[0] == "python":
        if venv_rel:
            venv_dir = (task_dir / venv_rel).resolve()
            py = python_in_venv(venv_dir)
            if py is None:
                desc = cmd_spec.get("description", " ".join(command))
                print(
                    f"  [skip] venv not found: {venv_dir}\n"
                    f"         Run the task's setup.sh first to create it.\n"
                    f"         Skipping: {desc}",
                )
                return None
            command[0] = str(py)
        else:
            command[0] = sys.executable

    return command, cwd


def run_prepare(task_dir: Path, manifest: dict) -> list[str]:
    """Execute prepare commands sequentially. Return list of error messages."""
    errors: list[str] = []
    for cmd_spec in manifest.get("prepare_commands", []):
        resolved = _resolve_command(task_dir, cmd_spec)
        if resolved is None:
            # venv missing — already warned, skip this command
            continue
        command, cwd = resolved
        desc = cmd_spec.get("description", " ".join(command))
        print(f"  Preparing: {desc}")
        try:
            subprocess.run(command, cwd=cwd, check=True)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{desc}: exit code {exc.returncode}")
        except FileNotFoundError as exc:
            errors.append(f"{desc}: {exc}")
    return errors


def check_and_prepare_task(evaluator_path: str) -> None:
    """Check task readiness for *evaluator_path* and auto-prepare if needed.

    Called from ``main.py`` before the engine starts.
    """
    result = find_manifest(evaluator_path)
    if result is None:
        return  # no manifest — self-contained task or custom evaluator

    task_dir, manifest = result
    missing = check_files(task_dir, manifest)
    if not missing:
        return  # all data present

    task_name = task_dir.name
    print(f"[task_prep] Task '{task_name}' has missing data files:")
    for f in missing:
        print(f"  - {f}")
    print(f"[task_prep] Running prepare commands for '{task_name}'...")

    errors = run_prepare(task_dir, manifest)
    if errors:
        print("[task_prep] Some prepare commands failed:")
        for e in errors:
            print(f"  - {e}")

    # Re-check after preparation
    still_missing = check_files(task_dir, manifest)
    if still_missing:
        print("[task_prep] Warning: some files are still missing after preparation:")
        for f in still_missing:
            print(f"  - {task_dir / f}")
        print(
            f"[task_prep] You may need to run the prepare script manually.\n"
            f"         See: {task_dir / _MANIFEST_NAME}",
        )
