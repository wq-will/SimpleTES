"""GPUMode TriMul evaluator (vendored).

We vendor the relevant pieces of `discover/gpu_mode` into this repository so we do NOT depend on an
external discover checkout at runtime.

This evaluator follows the upstream flow:
- load the official `task.yml` via `libkernelbot.task.make_task_definition`
- build a run config via `libkernelbot.task.build_task_config`
- execute via `libkernelbot.run_eval.run_config`
"""

from __future__ import annotations

import json
import os
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

from .code_utils import (
    cuda_sources_to_submission_py,
    normalize_cuda_sources,
    normalize_triton_submission_py,
)
from . import DEFAULT_TRIMUL_TASK_DIR
from .libkernelbot.consts import SubmissionMode
from .libkernelbot.run_eval import FullResult, run_config
from .libkernelbot.task import build_task_config, make_task_definition


@dataclass
class GpuModeEvalSummary:
    status: str
    latency_ms: Optional[float]
    log_excerpt: str
    run_key: Optional[str]
    run_success: bool
    run_passed: bool
    per_benchmark_means_us: list[float]
    raw_result: Optional[dict[str, Any]]


def _extract_benchmark_means_s(run_result_dict: dict[str, Any]) -> list[float]:
    if "benchmark-count" not in run_result_dict:
        return []
    n = int(run_result_dict["benchmark-count"])
    means_s: list[float] = []
    for i in range(n):
        key = f"benchmark.{i}.mean"
        if key not in run_result_dict:
            raise KeyError(f"Missing {key} in run result")
        # Upstream `trimul/eval.py` records benchmark durations in **nanoseconds**
        # (`torch.cuda.Event.elapsed_time` returns ms, then it multiplies by 1e6).
        # So `benchmark.{i}.mean` is in ns, and we convert it to seconds here.
        mean_ns = float(run_result_dict[key])
        means_s.append(mean_ns / 1e9)
    return means_s


def _aggregate_score_s(means_s: list[float], ranking_by: str) -> float:
    if not means_s:
        return float("nan")
    if ranking_by == "last":
        if len(means_s) != 1:
            raise ValueError(f"ranking_by=last expects 1 benchmark, got {len(means_s)}")
        return float(means_s[0])
    if ranking_by == "mean":
        return float(sum(means_s) / len(means_s))
    if ranking_by == "geom":
        prod = 1.0
        for x in means_s:
            prod *= x
        return float(prod ** (1.0 / len(means_s)))
    raise ValueError(f"Unknown ranking_by: {ranking_by}")


def evaluate_trimul_submission(
    *,
    submission_code: Any,
    mode: str = "benchmark",
    language: str = "python",
    task_dir: Path | None = None,
    keep_tmp: bool = False,
    tmpdir: Path | None = None,
    verbose: bool = False,
) -> GpuModeEvalSummary:
    task_dir = (task_dir or DEFAULT_TRIMUL_TASK_DIR).expanduser().resolve()
    task_yaml = task_dir / "task.yml"
    definition = make_task_definition(task_yaml)
    task = definition.task

    mode_enum = SubmissionMode(str(mode or "benchmark"))
    lang = (language or "").strip().lower()
    if lang == "cuda":
        sources = normalize_cuda_sources(submission_code)
        submission_content = cuda_sources_to_submission_py(sources)
    else:
        submission_content = normalize_triton_submission_py(submission_code)

    config = build_task_config(task=task, submission_content=submission_content, arch=None, mode=mode_enum)

    # Match the original local runner behavior (`discover/gpu_mode/run_trimul_local.py`):
    # execute inside a temp directory so eval/submission artifacts don't clutter cwd.
    cleanup = False
    run_dir: str
    if tmpdir is not None:
        run_dir = str(Path(tmpdir).expanduser().resolve())
        os.makedirs(run_dir, exist_ok=True)
        cleanup = False
    else:
        run_dir = tempfile.mkdtemp(prefix="ksearch_trimul_")
        cleanup = not bool(keep_tmp)

    old_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        result: FullResult = run_config(config)
    finally:
        os.chdir(old_cwd)
        if cleanup:
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass

    run_key = None
    if mode == "leaderboard":
        # In upstream GPUMode harness, leaderboard runs `test` first and may early-exit before
        # producing a leaderboard run payload. In that case, surface the test failure logs.
        if "leaderboard" in result.runs:
            run_key = "leaderboard"
        elif "test" in result.runs:
            run_key = "test"
        else:
            run_key = None
    elif mode in ("benchmark", "test", "profile"):
        run_key = mode if mode in result.runs else None

    log_excerpt = ""
    latency_ms = None
    per_bench_us: list[float] = []
    run_success = False
    run_passed = False
    status = "failed"
    raw = dataclasses.asdict(result)

    if not result.success:
        log_excerpt = str(result.error or "")
    elif run_key is None:
        log_excerpt = f"No run results for mode={mode}. Available: {sorted(result.runs.keys())}"
    else:
        eval_payload = result.runs[run_key]
        run = eval_payload.run
        if run is None:
            log_excerpt = f"No run payload for key={run_key}"
        else:
            run_success = bool(run.success)
            run_passed = bool(run.passed)
            stderr = (run.stderr or "").strip()
            stdout = (run.stdout or "").strip()
            combined = (stderr + "\n" + stdout).strip()
            # Only include verbose "failure reason" excerpts when the run did not pass.
            # On PASS, we want prompts/logs to stay compact; perf is communicated via summary lines.
            if (not run_success) or (not run_passed):
                # Prefer structured validator errors when present (e.g. correctness mismatches).
                # Also include a small header even when stderr/stdout is non-empty (warnings can otherwise
                # hide the actual failure mode).
                header_lines: list[str] = []
                try:
                    header_lines.append(
                        f"[gpumode] run_success={run_success} run_passed={run_passed} exit_code={getattr(run, 'exit_code', None)}"
                    )
                except Exception:
                    pass
                try:
                    cmd = str(getattr(run, "command", "") or "").strip()
                    if cmd:
                        header_lines.append(f"command: {cmd}")
                except Exception:
                    pass

                structured: list[str] = []
                try:
                    rr = getattr(run, "result", None)
                    if isinstance(rr, dict) and rr:
                        # Common keys in GPUMode tasks:
                        # - check: pass|fail
                        # - test.0.error / benchmark.0.error: human-readable failure reason
                        for k in ("test.0.error", "benchmark.0.error", "test.error", "benchmark.error"):
                            v = rr.get(k)
                            if isinstance(v, str) and v.strip():
                                structured.append(f"{k}: {v.strip()}")
                except Exception:
                    pass

                rr_dump = ""
                if not structured:
                    # If we didn't find a human-readable error key, include a bounded dump of the
                    # POPCORN result dict (often contains benchmark.<i>.status/error).
                    try:
                        rr = getattr(run, "result", None)
                        if isinstance(rr, dict) and rr:
                            rr_dump = "result:\n" + json.dumps(rr, indent=2, sort_keys=True)[:4000]
                    except Exception:
                        rr_dump = ""

                blocks: list[str] = []
                if header_lines:
                    blocks.append("\n".join([ln for ln in header_lines if str(ln).strip()]).strip())
                if structured:
                    blocks.append("\n".join(structured).strip())
                if rr_dump.strip():
                    blocks.append(rr_dump.strip())
                if combined:
                    blocks.append(combined)
                log_excerpt = "\n\n".join([b for b in blocks if b.strip()]).strip()[:8000]
            else:
                # Passed run: keep excerpt empty so prompts focus on perf only.
                # (stderr/stdout may contain harmless warnings; we intentionally do not feed them back.)
                log_excerpt = ""
            if run_success and run_passed and run_key in ("benchmark", "leaderboard"):
                try:
                    means_s = _extract_benchmark_means_s(run.result)
                    per_bench_us = [m * 1e6 for m in means_s]
                    if means_s:
                        agg_s = _aggregate_score_s(means_s, task.ranking_by.value)
                        latency_ms = agg_s * 1000.0
                    else:
                        latency_ms = None
                    status = "passed"
                except Exception as e:
                    log_excerpt = (log_excerpt + f"\n[score_error] {e}").strip()

    if verbose:
        kept = "" if cleanup else " (kept)"
        print(f"[gpumode] status={status} run_key={run_key} latency_ms={latency_ms} tmpdir={run_dir}{kept}")

    return GpuModeEvalSummary(
        status=status,
        latency_ms=latency_ms,
        log_excerpt=log_excerpt,
        run_key=run_key,
        run_success=run_success,
        run_passed=run_passed,
        per_benchmark_means_us=per_bench_us,
        raw_result=raw,
    )


