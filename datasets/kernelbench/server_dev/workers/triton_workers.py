"""
Triton backend workers: staged compiler-eval (correctness) + perf-eval
"""
import asyncio
import concurrent.futures
import os
import time
import traceback
from queue import Empty
from typing import Dict

from .base import (
    BaseEvalWorker,
    setup_build_env,
    static_check_program,
    validate_program_syntax,
    write_program_to_file,
)


def _default_failed_result_with_metadata(error_key: str, error_msg: str) -> Dict:
    return {
        "compiled": False,
        "correctness": False,
        "metadata": {error_key: error_msg},
        "combined_score": 0.0,
        "error": error_msg,
        "error_name": error_key,
    }


class TritonCompilerEvalWorker(BaseEvalWorker):
    """Triton compiler-eval worker: syntax/static checks + correctness-only stage"""

    def __init__(self, spawn_ctx, gpu_id: int, build_cache_dir: str, timeout: float = 600.0):
        super().__init__(spawn_ctx, gpu_id, build_cache_dir, timeout)
        # Dedicated executor so queue.get() does not contend on asyncio's default thread pool.
        self._queue_wait_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="triton-compiler-eval",
        )

    async def submit(self, task: Dict) -> Dict:
        import sys

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT_DIR = os.path.dirname(CURRENT_DIR)
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)

        from server_utils import tprint

        request_id = task.get("request_id", "unknown")
        result_queue = self.spawn_ctx.Queue()
        process = None

        try:
            process = self.spawn_ctx.Process(
                target=_triton_compiler_eval_worker_subprocess,
                args=(task, result_queue, self.gpu_id, self.build_cache_dir),
            )
            process.start()

            tprint(
                f"[Triton-CompilerEval-{request_id}] Started subprocess "
                f"PID={process.pid} on GPU {self.gpu_id}"
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._queue_wait_executor,
                lambda: result_queue.get(timeout=self.timeout),
            )
            tprint(f"[Triton-CompilerEval-{request_id}] Got result from subprocess")
            return result

        except Empty:
            tprint(f"[Triton-CompilerEval-{request_id}] Timeout after {self.timeout}s")
            return {
                "success": False,
                "result": _default_failed_result_with_metadata(
                    "compiler_eval_timeout", f"CompilerEval timeout ({self.timeout}s)"
                ),
            }
        except Exception as e:
            tprint(f"[Triton-CompilerEval-{request_id}] Error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "result": _default_failed_result_with_metadata(
                    "compiler_eval_client_error", str(e)
                ),
            }
        finally:
            try:
                _cleanup_process_and_queue(process, result_queue, request_id, worker_name="Triton-CompilerEval")
            finally:
                self._queue_wait_executor.shutdown(wait=True)


class TritonEvalWorker(BaseEvalWorker):
    """Triton perf-eval worker: performance-only stage"""

    async def submit(self, task: Dict) -> Dict:
        import sys

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT_DIR = os.path.dirname(CURRENT_DIR)
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)

        from server_utils import tprint

        request_id = task.get("request_id", "unknown")
        result_queue = self.spawn_ctx.Queue()
        process = None

        try:
            process = self.spawn_ctx.Process(
                target=_triton_perf_eval_worker_subprocess,
                args=(task, result_queue, self.gpu_id, self.build_cache_dir),
            )
            process.start()

            tprint(
                f"[Triton-PerfEval-{request_id}] Started subprocess "
                f"PID={process.pid} on GPU {self.gpu_id}"
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: result_queue.get(timeout=self.timeout),
            )
            tprint(f"[Triton-PerfEval-{request_id}] Got result from subprocess")
            return result

        except Empty:
            tprint(f"[Triton-PerfEval-{request_id}] Timeout after {self.timeout}s")
            return {
                "success": False,
                "error": f"PerfEval timeout ({self.timeout}s)",
                "result": self._default_failed_result(),
            }
        except Exception as e:
            tprint(f"[Triton-PerfEval-{request_id}] Error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "result": self._default_failed_result(),
            }
        finally:
            _cleanup_process_and_queue(process, result_queue, request_id, worker_name="Triton-PerfEval")


def _cleanup_process_and_queue(process, result_queue, request_id: str, worker_name: str):
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    from server_utils import tprint

    if process is not None:
        if process.is_alive():
            tprint(f"[{worker_name}-{request_id}] Killing subprocess PID={process.pid}")
            process.kill()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        else:
            process.join(timeout=1)

    if result_queue is not None:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass


def _prepare_task_file(
    task: Dict,
    build_cache_dir: str,
) -> tuple[str, str, str]:
    request_id = task["request_id"]
    level_id = task["level_id"]
    task_id = task["task_id"]

    python_files_dir = os.path.join(build_cache_dir, "python_files")
    request_python_dir = os.path.join(python_files_dir, request_id)
    request_build_dir = os.path.join(build_cache_dir, request_id)
    os.makedirs(request_python_dir, exist_ok=True)
    os.makedirs(request_build_dir, exist_ok=True)

    setup_build_env(request_build_dir)

    cache_filename = f"{request_id}_{level_id}_{task_id}.py"
    cache_path = os.path.join(request_python_dir, cache_filename)
    write_program_to_file(task["program_src"], cache_path)

    return cache_path, request_python_dir, request_build_dir


def _cleanup_request_dirs(request_python_dir: str | None, request_build_dir: str | None):
    import shutil
    # ethan
    try:
        if request_python_dir and os.path.exists(request_python_dir):
            shutil.rmtree(request_python_dir, ignore_errors=True)
        if request_build_dir and os.path.exists(request_build_dir):
            shutil.rmtree(request_build_dir, ignore_errors=True)
    except Exception:
        pass




def _set_process_cpu_affinity(task: Dict, stage: str, request_id: str):
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    from server_utils import tprint

    if stage == "perf":
        cpu_cores = task.get("triton_perf_cpu_cores")
    else:
        cpu_cores = task.get("triton_compiler_cpu_cores")

    if not cpu_cores:
        return

    try:
        os.sched_setaffinity(0, set(int(core) for core in cpu_cores))
        actual = sorted(os.sched_getaffinity(0))
    except Exception as e:
        tprint(
            f"[Triton-{stage.capitalize()}Eval-{request_id}] "
            f"Failed to set CPU affinity ({cpu_cores}): {e}"
        )


def _triton_compiler_eval_worker_subprocess(task: Dict, result_queue, gpu_id: int, build_cache_dir: str):
    """Subprocess target for Triton compiler-eval (correctness-only stage)."""
    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    request_id = task.get("request_id", "unknown")
    backend = task.get("backend", "triton")
    precision = task.get("precision", "float32")

    _set_process_cpu_affinity(task, stage="compiler", request_id=request_id)

    request_python_dir = None
    request_build_dir = None
    keep_artifacts_for_perf = False

    try:
        from server_utils import tprint
        from eval_shared import evaluate_correctness_from_python_file

        if backend != "triton":
            result_queue.put(
                {
                    "success": False,
                    "result": _default_failed_result_with_metadata(
                        "invalid_backend", f"Expected triton, got {backend}"
                    ),
                }
            )
            return

        load_program_time_ms = validate_program_syntax(task["program_src"])
        valid, errors, warnings, bypass, static_check_time_ms = static_check_program(
            task["program_src"], backend, precision
        )

        if not valid:
            tprint(f"[Triton-CompilerEval-{request_id}] Static check FAILED")
            static_metadata = {
                "static_check_failed": True,
                "static_errors": errors,
                "static_warnings": warnings,
                "static_bypass": bypass,
                "load_program_time_ms": load_program_time_ms,
                "static_check_time_ms": static_check_time_ms,
            }

            if bypass:
                tprint(f"[Triton-CompilerEval-{request_id}] Bypass: {bypass}")
                result_queue.put(
                    {
                        "success": False,
                        "result": {
                            "compiled": False,
                            "correctness": False,
                            "metadata": static_metadata,
                            "combined_score": 0.0,
                        },
                    }
                )
                return

            if errors:
                tprint(f"[Triton-CompilerEval-{request_id}] Errors: {errors}")
            if warnings:
                tprint(f"[Triton-CompilerEval-{request_id}] Warnings: {warnings}")

            result = {
                "compiled": False,
                "correctness": False,
                "metadata": static_metadata,
                "combined_score": 0.0,
            }
            if errors:
                result["error"] = errors
                result["error_name"] = "Static check failed"

            result_queue.put({"success": False, "result": result})
            return
        cache_path, request_python_dir, request_build_dir = _prepare_task_file(task, build_cache_dir)

        correctness_result = evaluate_correctness_from_python_file(
            cache_path=cache_path,
            level_id=task["level_id"],
            task_id=task["task_id"],
            task=task["task"],
            seed_num=task["seed_num"],
            precision=precision,
            request_id=request_id,
            gpu_id=gpu_id,
        )

        if "metadata" not in correctness_result:
            correctness_result["metadata"] = {}
        correctness_result["metadata"]["static_warnings"] = warnings
        correctness_result["metadata"]["static_check_time_ms"] = static_check_time_ms
        correctness_result["metadata"]["load_program_time_ms"] = load_program_time_ms

        if not correctness_result.get("correctness", False):
            result_queue.put({"success": False, "result": correctness_result})
            return

        result_queue.put(
            {
                "success": True,
                "result": correctness_result,
                "cache_path": cache_path,
                "request_build_dir": request_build_dir,
                "request_python_dir": request_python_dir,
            }
        )
        keep_artifacts_for_perf = True

    except Exception as e:
        traceback.print_exc()
        result_queue.put(
            {
                "success": False,
                "result": _default_failed_result_with_metadata("compiler_eval_worker_error", str(e)),
            }
        )

    finally:
        if not keep_artifacts_for_perf:
            _cleanup_request_dirs(request_python_dir, request_build_dir)


def _triton_perf_eval_worker_subprocess(task: Dict, result_queue, gpu_id: int, build_cache_dir: str):
    """Subprocess target for Triton performance-only evaluation stage."""
    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    request_id = task.get("request_id", "unknown")
    precision = task.get("precision", "float32")

    _set_process_cpu_affinity(task, stage="perf", request_id=request_id)

    request_python_dir = None
    request_build_dir = None

    try:
        from server_utils import tprint
        from eval_shared import evaluate_performance_from_python_file

        cache_path = task.get("cache_path")
        if cache_path is None:
            cache_path, request_python_dir, request_build_dir = _prepare_task_file(task, build_cache_dir)
        else:
            request_python_dir = task.get("request_python_dir")

            request_build_dir = task.get("request_build_dir")
            if request_build_dir:
                setup_build_env(request_build_dir)

        result = evaluate_performance_from_python_file(
            cache_path=cache_path,
            level_id=task["level_id"],
            task_id=task["task_id"],
            task=task["task"],
            seed_num=task["seed_num"],
            precision=precision,
            request_id=request_id,
            gpu_id=gpu_id,
        )

        result_queue.put({"success": True, "result": result})

        _cleanup_request_dirs(request_python_dir, request_build_dir)

    except Exception as e:
        from server_utils import tprint

        tprint(f"[Triton-PerfEval-{request_id}] Error: {e}")
        traceback.print_exc()

        result_queue.put(
            {
                "success": False,
                "result": {
                    "compiled": False,
                    "correctness": False,
                    "error": str(e),
                    "error_name": type(e).__name__,
                    "metadata": {traceback.format_exc()},
                    "combined_score": 0.0,
                },
            }
        )

        _cleanup_request_dirs(request_python_dir, request_build_dir)
