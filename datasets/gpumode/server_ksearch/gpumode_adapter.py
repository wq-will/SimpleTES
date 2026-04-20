from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_server_dev_root_on_syspath() -> Path:
    server_root = Path(__file__).resolve().parent
    server_root_str = str(server_root)
    if server_root_str not in sys.path:
        sys.path.insert(0, server_root_str)
    return server_root


def _parse_task_config(task: Any) -> Dict[str, Any]:
    if isinstance(task, dict):
        return {str(k): v for k, v in task.items()}

    s = str(task or "").strip()
    if not s:
        return {}

    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        except Exception:
            pass

    cfg: Dict[str, Any] = {}
    for part in re.split(r"[;,]", s):
        token = part.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        key = str(k).strip()
        val = str(v).strip()
        if key:
            cfg[key] = val
    return cfg


def _looks_like_cuda_xml(code: str) -> bool:
    text = str(code or "")
    return (
        '<header_file name="kernel.h">' in text
        and '<cuda_file name="kernel.cu">' in text
        and '<cpp_file name="main.cpp">' in text
    )


def is_gpumode_task(
    task: Any,
    *,
    backend: str | None = None,
    level_id: int | None = None,
    task_id: int | None = None,
) -> bool:
    """Contract-first routing.

    Primary expected request contract:
    - backend=triton
    - level_id=0
    - task_id=1
    - task="gpumode"
    """
    b = str(backend or "").strip().lower()
    t = str(task or "").strip().lower()

    if b == "triton" and t == "gpumode" and int(level_id or -1) == 0 and int(task_id or -1) == 1:
        return True

    # Minimal compatibility fallback.
    cfg = _parse_task_config(task)
    if str(cfg.get("task_source", "")).strip().lower() == "gpumode":
        return True

    return t in {"gpumode", "gpu_mode", "trimul"}


def resolve_task_dir(task: Any) -> Path:
    _ensure_server_dev_root_on_syspath()
    from gpu_mode import DEFAULT_TRIMUL_TASK_DIR

    cfg = _parse_task_config(task)
    candidate = cfg.get("task_dir")
    if candidate is not None:
        p = Path(str(candidate)).expanduser().resolve()
        if p.is_file() and p.name == "task.yml":
            return p.parent
        if p.is_dir() and (p / "task.yml").exists():
            return p

    # GPUMode route: fixed default task bundle.
    return DEFAULT_TRIMUL_TASK_DIR


def _resolve_mode(task: Any, stage: str) -> str:
    stage_name = str(stage or "full").strip().lower()
    if stage_name == "correctness":
        return "test"
    if stage_name == "performance":
        return "benchmark"

    cfg = _parse_task_config(task)
    mode = str(cfg.get("mode", "")).strip().lower()
    if mode in {"test", "benchmark", "leaderboard", "profile"}:
        return mode
    return "benchmark"


def infer_language(*, backend: str | None, program_src: str, task: Any) -> str:
    cfg = _parse_task_config(task)
    cfg_lang = str(cfg.get("language", "")).strip().lower()
    if cfg_lang in {"cuda", "triton", "python"}:
        return cfg_lang

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
    task: Any,
    level_id: int,
    task_id: int,
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
        result["error_name"] = "GpuModeEvaluationFailed"

    return result


def run_gpumode_eval(
    *,
    cache_path: str,
    task: Any,
    stage: str,
    backend: str | None,
    level_id: int,
    task_id: int,
    request_id: str,
    gpu_id: int,
) -> Dict[str, Any]:
    try:
        _ensure_server_dev_root_on_syspath()
        from gpu_mode.evaluator import evaluate_trimul_submission

        program_src = _read_submission_code(cache_path)
        language = infer_language(backend=backend, program_src=program_src, task=task)
        mode = _resolve_mode(task, stage)
        task_dir = resolve_task_dir(task)

        summary = evaluate_trimul_submission(
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
            task=task,
            level_id=level_id,
            task_id=task_id,
            request_id=request_id,
            gpu_id=gpu_id,
        )
    except Exception as e:
        return {
            "compiled": False,
            "correctness": False,
            "combined_score": 0.0,
            "metadata": {
                "gpumode": {
                    "stage": str(stage or ""),
                    "backend": str(backend or ""),
                },
            },
            "error": str(e),
            "error_name": type(e).__name__,
        }
