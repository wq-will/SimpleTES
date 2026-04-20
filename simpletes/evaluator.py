"""
Evaluator components for SimpleTES.

Runs evaluations in a subprocess with hard timeout + kill for robustness.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


from rich.console import Console

from simpletes.construction import (
    CAPTURE_CONSTRUCTION_ENV,
    MAX_SNAPSHOT_BYTES_ENV,
    SHARED_CONSTRUCTION_ENV,
    max_snapshot_bytes,
    read_payload,
)
from simpletes.utils.text import DEFAULT_METRICS_ERROR_MAX_CHARS, truncate_error_in_metrics

# Console that writes to sys.__stdout__ (Python's original stdout, unaffected by redirections)
# This allows the engine to print status even when evaluator output is suppressed.
# Let Rich auto-detect terminal capabilities to avoid showing raw escape codes (ESC[33m etc.)
# in terminals that don't support ANSI sequences.
rich_print = Console(file=sys.__stdout__, soft_wrap=True).print

TEMP_EVAL_DIR = tempfile.gettempdir()
REPO_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class EvaluationOutcome:
    """Metrics plus an optional captured shared-construction payload."""

    metrics: dict[str, Any]
    captured_construction_payload: Any | None = None


class Evaluator(Protocol):
    """Protocol for custom evaluators.
    
    Evaluators must be synchronous functions that take a filepath and return metrics.
    The returned dict must contain at least 'combined_score' (float).
    """
    def evaluate(self, filepath: str) -> dict[str, Any]: ...


# Subprocess runner script for isolated code evaluation.
#
# Why subprocess isolation?
#   - Prevents evaluated code from corrupting the main process state
#   - Enables hard timeout via process kill (no cooperative cancellation needed)
#   - Isolates crashes, infinite loops, and resource exhaustion
#   - Each evaluation runs in a fresh Python interpreter
#
# JSON protocol:
#   - The runner prints a single JSON object to stdout on success
#   - On error, it prints {"error": "...", "combined_score": -inf}
#   - We search from the last line backwards for JSON because the evaluated
#     code might print extra output before the result
#
# Process group:
#   - We use start_new_session=True to create a new process group
#   - This allows killpg() to terminate the entire process tree on timeout
_EVAL_RUNNER = textwrap.dedent(r'''
import importlib.util
import json
import sys
import traceback

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Invalid arguments", "combined_score": float("-inf")}))
        sys.exit(1)
    
    evaluator_path = sys.argv[1]
    target_file = sys.argv[2]
    
    try:
        # Load evaluator module
        spec = importlib.util.spec_from_file_location("user_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            print(json.dumps({"error": f"Could not load evaluator: {evaluator_path}", "combined_score": float("-inf")}))
            sys.exit(1)
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Run evaluation
        result = mod.evaluate(target_file)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_msg, "combined_score": float("-inf")}))
        sys.exit(1)

if __name__ == "__main__":
    main()
''').strip()


class EvaluatorWorker:
    """Evaluates generated code using a user-provided evaluator module.

    Spawns a fresh Python subprocess per evaluation, enabling hard timeout via kill.
    """

    def __init__(
        self,
        evaluator_path: str,
        timeout: float = 300.0,
        *,
        python_executable: str | None = None,
    ):
        self.instance_id = str(uuid.uuid4())[:8]
        self.timeout = timeout
        self.evaluator_path = os.path.abspath(evaluator_path)
        self.python_executable = os.path.abspath(python_executable) if python_executable else None

        if not os.path.exists(self.evaluator_path):
            raise ValueError(f"Evaluator module not found: {self.evaluator_path}")
        if self.python_executable and not os.path.exists(self.python_executable):
            raise ValueError(f"Evaluation Python executable not found: {self.python_executable}")

    def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """Kill process and its children."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            # Fallback to killing just the process if killpg fails (e.g., Windows)
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _subprocess_env(self, capture_path: str, shared_construction_path: str | None) -> dict[str, str]:
        env = os.environ.copy()
        env[CAPTURE_CONSTRUCTION_ENV] = capture_path
        env[MAX_SNAPSHOT_BYTES_ENV] = str(max_snapshot_bytes())
        if shared_construction_path:
            env[SHARED_CONSTRUCTION_ENV] = shared_construction_path
        else:
            env.pop(SHARED_CONSTRUCTION_ENV, None)

        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
            if current_pythonpath
            else REPO_ROOT
        )
        return env

    async def evaluate(
        self,
        code: str,
        *,
        shared_construction_path: str | None = None,
    ) -> EvaluationOutcome:
        """Evaluate code and return metrics plus an optional captured payload."""
        # Create unique subdirectory for this evaluation to isolate file operations
        eval_dir = os.path.join(TEMP_EVAL_DIR, f"eval_{self.instance_id}_{uuid.uuid4().hex}")
        os.makedirs(eval_dir, exist_ok=True)
        filename = os.path.join(eval_dir, "program.py")
        capture_path = os.path.join(eval_dir, "captured_construction.json")
        env = self._subprocess_env(capture_path, shared_construction_path)

        try:
            # Write code to temp file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code)

            # Set TMPDIR/TMP/TEMP environment variables so that all tempfile calls
            # (including those in evaluator subprocesses) use the eval_dir.
            # This ensures temp files are isolated per evaluation without modifying
            # individual evaluator code.
            env["TMPDIR"] = eval_dir
            env["TMP"] = eval_dir
            env["TEMP"] = eval_dir

            proc = await asyncio.create_subprocess_exec(
                (self.python_executable or sys.executable),
                "-c",
                _EVAL_RUNNER,
                self.evaluator_path, filename,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=eval_dir,  # Isolate file operations to eval directory
                env=env,  # Use modified environment with TMPDIR pointing to eval_dir
                start_new_session=True,  # Create new process group for killpg
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            except TimeoutError:
                self._kill_process_group(proc)
                await proc.wait()
                rich_print(
                    f"[yellow][{self.instance_id}][/yellow] [yellow]⚠[/yellow] "
                    f"Evaluation timed out after [bold]{self.timeout}s[/bold] (process killed)"
                )
                metrics = {"error": "timeout", "combined_score": -float("inf")}
                truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
                return EvaluationOutcome(metrics=metrics)
            except asyncio.CancelledError:
                self._kill_process_group(proc)
                await proc.wait()
                raise

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                error_msg = stderr_text or stdout_text or f"Process exited with code {proc.returncode}"
                rich_print(
                    f"[red][{self.instance_id}][/red] [red]✗[/red] "
                    f"[bold red]Evaluation error:[/bold red] {error_msg[:200]}"
                )
                metrics = {"error": error_msg, "combined_score": -float("inf")}
                truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
                return EvaluationOutcome(metrics=metrics)

            metrics = self._parse_json_from_output(stdout_text)
            truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
            payload = None
            if os.path.exists(capture_path):
                try:
                    payload = read_payload(capture_path)
                except Exception:
                    payload = None
            return EvaluationOutcome(metrics=metrics, captured_construction_payload=payload)

        finally:
            # Clean up entire eval directory (including any files created by evolved code)
            try:
                shutil.rmtree(eval_dir, ignore_errors=True)
            except Exception:
                pass

    def _parse_json_from_output(self, stdout_text: str) -> dict[str, Any]:
        """
        Parse JSON result from subprocess output.

        The evaluator prints JSON to stdout, but the evaluated code might also
        print output. We search backwards from the last line to find the first
        valid JSON object (which should be the evaluator's result).

        Returns the parsed dict, or an error dict if parsing fails.
        """
        if not stdout_text or not stdout_text.strip():
            metrics = {"error": "Empty evaluator output", "combined_score": -float("inf")}
            truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
            return metrics

        try:
            # Find the last line that looks like JSON
            for line in reversed(stdout_text.split("\n")):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            # Fallback: try parsing the entire output
            return json.loads(stdout_text)
        except json.JSONDecodeError as e:
            preview = stdout_text[:200] if len(stdout_text) > 200 else stdout_text
            rich_print(
                f"[red][{self.instance_id}][/red] [red]✗[/red] "
                f"[bold red]Failed to parse evaluator output:[/bold red] {e}\n"
                f"Output preview: {preview}"
            )
            metrics = {"error": f"Invalid JSON output: {e}", "combined_score": -float("inf")}
            truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)
            return metrics
