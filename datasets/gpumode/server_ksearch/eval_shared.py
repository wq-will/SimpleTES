"""GPUMode-only evaluation entrypoints used by worker subprocesses."""

from __future__ import annotations

from typing import Any, Dict

from gpumode_adapter import is_gpumode_task, run_gpumode_eval
from server_utils import tprint


def _gpumode_only_failure(
    *,
    stage: str,
    task: Any,
    backend: str,
    level_id: int,
    task_id: int,
    request_id: str,
    gpu_id: int,
) -> Dict[str, Any]:
    error = (
        "server_ksearch is gpumode-only: non-gpumode request is not supported "
        f"(task={task!r}, backend={backend!r}, level_id={level_id}, task_id={task_id})"
    )
    tprint(f"[{request_id}] [GPU{gpu_id}] [REJECT] {error}")
    return {
        "compiled": False,
        "correctness": False,
        "combined_score": 0.0,
        "error": error,
        "error_name": "UnsupportedTask",
        "metadata": {
            "gpumode_only": True,
            "stage": stage,
            "request_context": {
                "task": str(task or ""),
                "backend": str(backend or ""),
                "level_id": int(level_id),
                "task_id": int(task_id),
                "request_id": str(request_id or ""),
                "gpu_id": int(gpu_id),
            },
        },
    }


def evaluate_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate a submission in full mode for gpumode tasks."""
    if not is_gpumode_task(task, backend=backend, level_id=level_id, task_id=task_id):
        return _gpumode_only_failure(
            stage="full",
            task=task,
            backend=backend,
            level_id=level_id,
            task_id=task_id,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpumode_eval(
        cache_path=cache_path,
        task=task,
        stage="full",
        backend=backend,
        level_id=level_id,
        task_id=task_id,
        request_id=request_id,
        gpu_id=gpu_id,
    )


def evaluate_correctness_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate correctness stage for gpumode tasks."""
    if not is_gpumode_task(task, backend=backend, level_id=level_id, task_id=task_id):
        return _gpumode_only_failure(
            stage="correctness",
            task=task,
            backend=backend,
            level_id=level_id,
            task_id=task_id,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpumode_eval(
        cache_path=cache_path,
        task=task,
        stage="correctness",
        backend=backend,
        level_id=level_id,
        task_id=task_id,
        request_id=request_id,
        gpu_id=gpu_id,
    )


def evaluate_performance_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate performance stage for gpumode tasks."""
    if not is_gpumode_task(task, backend=backend, level_id=level_id, task_id=task_id):
        return _gpumode_only_failure(
            stage="performance",
            task=task,
            backend=backend,
            level_id=level_id,
            task_id=task_id,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpumode_eval(
        cache_path=cache_path,
        task=task,
        stage="performance",
        backend=backend,
        level_id=level_id,
        task_id=task_id,
        request_id=request_id,
        gpu_id=gpu_id,
    )
