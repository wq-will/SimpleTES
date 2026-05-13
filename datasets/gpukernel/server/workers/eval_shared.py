"""GPUKernel-only evaluation entrypoints used by worker subprocesses."""

from __future__ import annotations

from typing import Any, Dict

from tasks.adapter import is_gpukernel_task, run_gpukernel_eval
from server_utils import tprint


def _gpukernel_only_failure(
    *,
    stage: str,
    task: Any,
    backend: str,
    request_id: str,
    gpu_id: int,
) -> Dict[str, Any]:
    error = (
        f"Unsupported task {task!r} (backend={backend!r}). "
        "Supported tasks: trimul, cumsum, asymmetricmatmul."
    )
    tprint(f"[{request_id}] [GPU{gpu_id}] [REJECT] {error}")
    return {
        "compiled": False,
        "correctness": False,
        "combined_score": 0.0,
        "error": error,
        "error_name": "UnsupportedTask",
        "metadata": {
            "gpukernel_only": True,
            "stage": stage,
            "request_context": {
                "task": str(task or ""),
                "backend": str(backend or ""),
                "request_id": str(request_id or ""),
                "gpu_id": int(gpu_id),
            },
        },
    }


def evaluate_from_python_file(
    cache_path: str,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate a submission in full mode for gpukernel tasks."""
    if not is_gpukernel_task(task, backend=backend):
        return _gpukernel_only_failure(
            stage="full",
            task=task,
            backend=backend,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpukernel_eval(
        cache_path=cache_path,
        task=task,
        stage="full",
        backend=backend,
        request_id=request_id,
        gpu_id=gpu_id,
    )


def evaluate_correctness_from_python_file(
    cache_path: str,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate correctness stage for gpukernel tasks."""
    if not is_gpukernel_task(task, backend=backend):
        return _gpukernel_only_failure(
            stage="correctness",
            task=task,
            backend=backend,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpukernel_eval(
        cache_path=cache_path,
        task=task,
        stage="correctness",
        backend=backend,
        request_id=request_id,
        gpu_id=gpu_id,
    )


def evaluate_performance_from_python_file(
    cache_path: str,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    backend: str = "",
) -> Dict[str, Any]:
    """Evaluate performance stage for gpukernel tasks."""
    if not is_gpukernel_task(task, backend=backend):
        return _gpukernel_only_failure(
            stage="performance",
            task=task,
            backend=backend,
            request_id=request_id,
            gpu_id=gpu_id,
        )

    return run_gpukernel_eval(
        cache_path=cache_path,
        task=task,
        stage="performance",
        backend=backend,
        request_id=request_id,
        gpu_id=gpu_id,
    )
