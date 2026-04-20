"""
CUDA backend workers: requires compilation via load_inline
"""
import asyncio
import os
import time
import traceback
from typing import Dict
from queue import Empty
from .base import (
    BaseCompilerWorker,
    BaseEvalWorker,
    setup_build_env,
    validate_program_syntax,
    static_check_program,
    write_program_to_file,
)


class CUDACompilerWorker(BaseCompilerWorker):
    """CUDA compiler worker - handles load_inline compilation"""

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
                target=_cuda_compile_worker_subprocess,
                args=(task, result_queue, self.python_files_dir, self.build_cache_dir)
            )
            process.start()

            tprint(f"[CUDA-Compile-{request_id}] Started subprocess PID={process.pid}")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: result_queue.get(timeout=self.timeout)
            )
            tprint(f"[CUDA-Compile-{request_id}] Got result from subprocess")
            return result

        except Empty:
            tprint(f"[CUDA-Compile-{request_id}] Timeout after {self.timeout}s")
            return {
                "compiled": False,
                "cache_path": None,
                "error": f"Compile timeout ({self.timeout}s)",
                "error_name": "Compile Timeout",
            }
        except Exception as e:
            tprint(f"[CUDA-Compile-{request_id}] Error: {e}")
            traceback.print_exc()
            return {
                "compiled": False,
                "cache_path": None,
                "error": str(e),
                "error_name": type(e).__name__,
            }
        finally:
            self._cleanup(process, result_queue, request_id)

    def _cleanup(self, process, result_queue, request_id: str):
        import sys
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT_DIR = os.path.dirname(CURRENT_DIR)
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)

        from server_utils import tprint

        if process is not None:
            if process.is_alive():
                tprint(f"[CUDA-Compile-{request_id}] Killing subprocess PID={process.pid}")
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


def _cuda_compile_worker_subprocess(task: Dict, result_queue, python_files_dir: str, build_cache_dir: str):
    """Subprocess target for CUDA compilation"""
    try:
        result = _cuda_compile_and_cache(task, python_files_dir, build_cache_dir)
        result_queue.put(result)
    except Exception as e:
        traceback.print_exc()
        result_queue.put({
            "compiled": False,
            "cache_path": None,
            "error": str(e),
            "error_name": type(e).__name__,
            "metadata": {
                "traceback": traceback.format_exc(),
            }
        })


def _cuda_compile_and_cache(task: Dict, python_files_dir: str, build_cache_dir: str) -> Dict:
    """
    CUDA-specific compilation: syntax check + static check + file write + load_inline
    """
    import sys
    import shutil
    import importlib.util

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    from server_utils import tprint

    request_id = task["request_id"]
    program_src = task["program_src"]
    level_id = task["level_id"]
    task_id = task["task_id"]
    backend = task.get("backend", "cuda")
    precision = task.get("precision", "float32")

    start_time = time.time()
    request_python_dir = None
    request_build_dir = None
    keep_artifacts_for_eval = False

    try:
        # Phase 1: Validation
        validate_time_ms = validate_program_syntax(program_src)
        valid, errors, warnings, bypass, static_check_time_ms = static_check_program(program_src, backend, precision)

        if not valid:
            tprint(f"[CUDA-{request_id}] Static check FAILED (time={static_check_time_ms:.1f}ms)")
            static_metadata = {
                "static_check_failed": True,
                "static_errors": errors,
                "static_warnings": warnings,
                "static_bypass": bypass,
                "validate_time_ms": validate_time_ms,
                "static_check_time_ms": static_check_time_ms,
                "total_time_ms": (time.time() - start_time) * 1000,
            }

            if bypass:
                tprint(f"[CUDA-{request_id}] Bypass: {bypass}")
                return {
                    "compiled": False,
                    "cache_path": None,
                    "metadata": static_metadata,
                }

            if errors:
                tprint(f"[CUDA-{request_id}] Errors: {errors}")
            if warnings:
                tprint(f"[CUDA-{request_id}] Warnings: {warnings}")

            result = {
                "compiled": False,
                "cache_path": None,
                "metadata": static_metadata,
            }
            if errors:
                result["error"] = errors
                result["error_name"] = "Static check failed"
            return result
        if warnings:
            tprint(f"[CUDA-{request_id}] Static check PASSED with warnings (time={static_check_time_ms:.1f}ms)")
            tprint(f"[CUDA-{request_id}] Warnings: {warnings}")
        else:
            tprint(f"[CUDA-{request_id}] Static check PASSED (time={static_check_time_ms:.1f}ms)")

        # Phase 2: File preparation
        t0 = time.time()
        request_python_dir = os.path.join(python_files_dir, request_id)
        request_build_dir = os.path.join(build_cache_dir, request_id)
        os.makedirs(request_python_dir, exist_ok=True)
        os.makedirs(request_build_dir, exist_ok=True)

        setup_build_env(request_build_dir)

        cache_filename = f"{request_id}_{level_id}_{task_id}.py"
        cache_path = os.path.join(request_python_dir, cache_filename)
        write_program_to_file(program_src, cache_path)
        write_time_ms = (time.time() - t0) * 1000

        # Phase 3: Module import (triggers load_inline - CUDA specific!)
        t0 = time.time()
        spec = importlib.util.spec_from_file_location("custom_module", cache_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        custom_kernel = getattr(custom_module, "custom_kernel", None)
        if not callable(custom_kernel):
            raise AttributeError("Cannot find callable custom_kernel in custom code")

        import_time_ms = (time.time() - t0) * 1000
        total_time_ms = (time.time() - start_time) * 1000

        tprint(f"[CUDA-{request_id}] Compiled successfully → {cache_path} "
              f"(validate={validate_time_ms:.1f}ms, static_check={static_check_time_ms:.1f}ms, "
              f"write={write_time_ms:.1f}ms, import={import_time_ms:.1f}ms, total={total_time_ms:.1f}ms)")

        keep_artifacts_for_eval = True
        return {
            "compiled": True,
            "cache_path": cache_path,
            "request_build_dir": request_build_dir,
            "metadata": {
                "validate_time_ms": validate_time_ms,
                "static_check_time_ms": static_check_time_ms,
                "static_warnings": warnings,
                "write_time_ms": write_time_ms,
                "import_time_ms": import_time_ms,
                "total_time_ms": total_time_ms,
            },
        }

    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        error_traceback = traceback.format_exc()
        tprint(f"[CUDA-{request_id}] Compile failed: {e} (time={total_time_ms:.1f}ms)")
        tprint(f"[CUDA-{request_id}] Traceback:\n{error_traceback}")

        return {
            "compiled": False,
            "cache_path": None,
            "error": str(e),
            "error_name": type(e).__name__,
            "error_traceback": error_traceback,
            "total_time_ms": total_time_ms,
            "metadata": {
                "error": str(e),
                "error_name": type(e).__name__,
                "error_traceback": error_traceback,
                "total_time_ms": total_time_ms
            },
        }

    finally:
        import torch.utils.cpp_extension as ce
        ce.JIT_EXTENSION_VERSIONER.entries.clear()

        if not keep_artifacts_for_eval:
            try:
                if request_python_dir and os.path.exists(request_python_dir):
                    shutil.rmtree(request_python_dir, ignore_errors=True)
                if request_build_dir and os.path.exists(request_build_dir):
                    shutil.rmtree(request_build_dir, ignore_errors=True)
            except Exception:
                pass


class CUDAEvalWorker(BaseEvalWorker):
    """CUDA eval worker - evaluates compiled CUDA kernels"""

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
                target=_cuda_eval_worker_subprocess,
                args=(task, result_queue, self.gpu_id, self.build_cache_dir)
            )
            process.start()

            tprint(f"[CUDA-Eval-{request_id}] Started subprocess PID={process.pid} on GPU {self.gpu_id}")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: result_queue.get(timeout=self.timeout)
            )
            tprint(f"[CUDA-Eval-{request_id}] Got result from subprocess")
            return result

        except Empty:
            tprint(f"[CUDA-Eval-{request_id}] Timeout after {self.timeout}s")
            return {
                "success": False,
                "result": self._default_failed_result()
            }
        except Exception as e:
            tprint(f"[CUDA-Eval-{request_id}] Error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "result": self._default_failed_result()
            }
        finally:
            self._cleanup(process, result_queue, request_id)

    def _cleanup(self, process, result_queue, request_id: str):
        import sys
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT_DIR = os.path.dirname(CURRENT_DIR)
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)

        from server_utils import tprint

        if process is not None:
            if process.is_alive():
                tprint(f"[CUDA-Eval-{request_id}] Killing subprocess PID={process.pid}")
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


def _cuda_eval_worker_subprocess(task: Dict, result_queue, gpu_id: int, build_cache_dir: str):
    """Subprocess target for CUDA evaluation"""
    import os
    import sys
    import shutil

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    request_id = task.get("request_id", "unknown")
    request_build_dir = task.get("request_build_dir", "")

    cache_path = task.get("cache_path")
    request_python_dir = os.path.dirname(cache_path) if cache_path else None

    try:
        if request_build_dir:
            setup_build_env(request_build_dir)

        from eval_shared import evaluate_from_python_file
        from server_utils import tprint

        tprint(f"[CUDA-Eval-{request_id}] Processing on GPU {gpu_id}")

        result = evaluate_from_python_file(
            cache_path=task["cache_path"],
            level_id=task["level_id"],
            task_id=task["task_id"],
            task=task["task"],
            seed_num=task["seed_num"],
            precision=task["precision"],
            request_id=request_id,
            gpu_id=gpu_id,
            backend=task.get("backend", "cuda"),
        )

        tprint(f"[CUDA-Eval-{request_id}] Evaluation completed")
        result_queue.put({"success": True, "result": result})

    except Exception as e:
        from server_utils import tprint
        tprint(f"[CUDA-Eval-{request_id}] Error: {e}")
        traceback.print_exc()

        result_queue.put({
            "success": False,
            "result": {
                "error": str(e),
                "error_name": type(e).__name__,
                "compiled": False,
                "correctness": False,
                "combined_score": 0.0,
            }
        })
    finally:
        try:
            import torch.utils.cpp_extension as ce
            ce.JIT_EXTENSION_VERSIONER.entries.clear()

            if request_python_dir and os.path.exists(request_python_dir):
                shutil.rmtree(request_python_dir, ignore_errors=True)

            if request_build_dir and os.path.exists(request_build_dir):
                shutil.rmtree(request_build_dir, ignore_errors=True)
        except Exception as cleanup_error:
            from server_utils import tprint
            tprint(f"[CUDA-Eval-{request_id}] Cleanup warning: {cleanup_error}")
