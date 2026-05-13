from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_SUPPORTED_TASKS = {"trimul", "cumsum", "asymmetricmatmul"}

_TASK_TO_SERVER_DIR = {
    "trimul": "trimul",
    "cumsum": "cumsum",
    "asymmetricmatmul": "asymmetricmatmul",
}


def _ensure_server_dev_root_on_syspath() -> Path:
    server_root = Path(__file__).resolve().parent.parent
    server_root_str = str(server_root)
    if server_root_str not in sys.path:
        sys.path.insert(0, server_root_str)
    return server_root


def _looks_like_cuda_xml(code: str) -> bool:
    text = str(code or "")
    return (
        '<header_file name="kernel.h">' in text
        and '<cuda_file name="kernel.cu">' in text
        and '<cpp_file name="main.cpp">' in text
    )


def is_gpukernel_task(task: Any, *, backend: str | None = None) -> bool:
    b = str(backend or "").strip().lower()
    t = str(task or "").strip().lower()
    return b in {"triton", "torch", "cuda"} and t in _SUPPORTED_TASKS


def resolve_task_dir(task: Any) -> Path:
    from . import (
        DEFAULT_TRIMUL_TASK_DIR,
        DEFAULT_ASYMMETRICMATMUL_TASK_DIR,
        DEFAULT_CUMSUM_TASK_DIR,
    )

    _dir_map = {
        "trimul": DEFAULT_TRIMUL_TASK_DIR,
        "cumsum": DEFAULT_CUMSUM_TASK_DIR,
        "asymmetricmatmul": DEFAULT_ASYMMETRICMATMUL_TASK_DIR,
    }
    t = str(task or "").strip().lower()
    if t not in _dir_map:
        raise ValueError(f"Unsupported task: {task!r}")
    return _dir_map[t]


def _resolve_mode(task: Any, stage: str) -> str:
    stage_name = str(stage or "full").strip().lower()
    if stage_name == "correctness":
        return "test"
    if stage_name == "performance":
        return "benchmark"
    return "benchmark"


def infer_language(*, backend: str | None, program_src: str, task: Any) -> str:
    b = str(backend or "").strip().lower()
    if b == "cuda":
        return "cuda" if _looks_like_cuda_xml(program_src) else "python"
    if b == "triton":
        return "triton"
    if b == "torch":
        return "python"

    if _looks_like_cuda_xml(program_src):
        return "cuda"
    if "@triton." in str(program_src or ""):
        return "triton"
    return "python"


def _read_submission_code(cache_path: str) -> str:
    with open(cache_path, "r", encoding="utf-8") as f:
        return f.read()


def _summary_to_server_result(
    *,
    summary: Any,
    mode: str,
    language: str,
    request_id: str,
    gpu_id: int,
) -> Dict[str, Any]:
    run_success = bool(getattr(summary, "run_success", False))
    run_passed = bool(getattr(summary, "run_passed", False))
    latency_ms = getattr(summary, "latency_ms", None)
    score = 0.0
    if run_success and run_passed and isinstance(latency_ms, (int, float)) and float(latency_ms) > 0:
        score = 1.0 / float(latency_ms)

    log_excerpt = str(getattr(summary, "log_excerpt", "") or "").strip()

    result: Dict[str, Any] = {
        "compiled": run_success,
        "correctness": bool(run_success and run_passed),
        "combined_score": float(score),
        "metadata": {
            "status": str(getattr(summary, "status", "") or ""),
            "mode": mode,
            "run_success": run_success,
            "run_passed": run_passed,
            "latency_ms": latency_ms,
            "per_benchmark_means_us": list(getattr(summary, "per_benchmark_means_us", []) or []),
        },
    }

    if mode in {"benchmark", "leaderboard"} and isinstance(latency_ms, (int, float)):
        result["runtime_stats"] = {
            "summary": {
                "latency_ms": float(latency_ms),
                "score": float(score),
                "passed": bool(run_success and run_passed),
            },
            "per_benchmark_means_us": list(getattr(summary, "per_benchmark_means_us", []) or []),
        }

    if not (run_success and run_passed):
        result["error"] = log_excerpt or str(getattr(summary, "status", "failed") or "failed")
        result["error_name"] = "GpuKernelEvaluationFailed"

    return result


def run_gpukernel_eval(
    *,
    cache_path: str,
    task: Any,
    stage: str,
    backend: str | None,
    request_id: str,
    gpu_id: int,
) -> Dict[str, Any]:
    try:
        from .evaluator import evaluate_gpukernel_submission

        program_src = _read_submission_code(cache_path)
        language = infer_language(backend=backend, program_src=program_src, task=task)
        mode = _resolve_mode(task, stage)
        task_dir = resolve_task_dir(task)

        summary = evaluate_gpukernel_submission(
            submission_code=program_src,
            mode=mode,
            language=language,
            task_dir=task_dir,
            keep_tmp=False,
            verbose=False,
        )

        return _summary_to_server_result(
            summary=summary,
            mode=mode,
            language=language,
            request_id=request_id,
            gpu_id=gpu_id,
        )
    except Exception as e:
        return {
            "compiled": False,
            "correctness": False,
            "combined_score": 0.0,
            "metadata": {
                "gpukernel": {
                    "stage": str(stage or ""),
                    "backend": str(backend or ""),
                },
            },
            "error": str(e),
            "error_name": type(e).__name__,
        }
