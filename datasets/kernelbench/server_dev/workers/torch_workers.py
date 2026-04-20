"""
Torch backend workers: no compilation needed, pure PyTorch evaluation
"""
import asyncio
import os
import time
import traceback
from typing import Dict
from queue import Empty
from .base import (
    BaseEvalWorker,
    setup_build_env,
    validate_program_syntax,
    static_check_program,
    write_program_to_file,
)


class TorchEvalWorker(BaseEvalWorker):
    """Torch eval worker - no compilation, direct evaluation"""

    async def submit(self, task: Dict) -> Dict:
        """
        Torch doesn't need separate compilation step.
        We do: syntax check + static check + write file + eval
        """
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
                target=_torch_eval_worker_subprocess,
                args=(task, result_queue, self.gpu_id, self.build_cache_dir)
            )
            process.start()

            tprint(f"[Torch-Eval-{request_id}] Started subprocess PID={process.pid} on GPU {self.gpu_id}")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: result_queue.get(timeout=self.timeout)
            )
            tprint(f"[Torch-Eval-{request_id}] Got result from subprocess")
            return result

        except Empty:
            tprint(f"[Torch-Eval-{request_id}] Timeout after {self.timeout}s")
            return {
                "success": False,
                "result": self._default_failed_result()
            }
        except Exception as e:
            tprint(f"[Torch-Eval-{request_id}] Error: {e}")
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
                tprint(f"[Torch-Eval-{request_id}] Killing subprocess PID={process.pid}")
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


def _torch_eval_worker_subprocess(task: Dict, result_queue, gpu_id: int, build_cache_dir: str):
    """Subprocess target for Torch evaluation (no compilation)"""
    import os
    import sys
    import shutil

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    request_id = task.get("request_id", "unknown")
    program_src = task["program_src"]
    level_id = task["level_id"]
    task_id = task["task_id"]
    backend = task.get("backend", "torch")
    precision = task.get("precision", "float32")

    request_python_dir = None
    request_build_dir = None

    start_time = time.time()

    try:
        from server_utils import tprint

        # Phase 1: Validation (no compilation for Torch)
        compile_time_ms = validate_program_syntax(program_src)
        valid, errors, warnings, bypass, static_check_time_ms = static_check_program(program_src, backend, precision)

        if not valid:
            tprint(f"[Torch-{request_id}] Static check FAILED")
            static_metadata = {
                "static_check_failed": True,
                "static_errors": errors,
                "static_warnings": warnings,
                "static_bypass": bypass,
            }

            if bypass:
                tprint(f"[Torch-{request_id}] Bypass: {bypass}")
                result = {
                    "compiled": False,
                    "correctness": False,
                    "metadata": static_metadata,
                    "combined_score": 0.0,
                }
            else:
                if errors:
                    tprint(f"[Torch-{request_id}] Errors: {errors}")
                if warnings:
                    tprint(f"[Torch-{request_id}] Warnings: {warnings}")
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

        # Phase 2: Write file
        python_files_dir = os.path.join(build_cache_dir, "python_files")
        request_python_dir = os.path.join(python_files_dir, request_id)
        request_build_dir = os.path.join(build_cache_dir, request_id)
        os.makedirs(request_python_dir, exist_ok=True)
        os.makedirs(request_build_dir, exist_ok=True)

        setup_build_env(request_build_dir)

        cache_filename = f"{request_id}_{level_id}_{task_id}.py"
        cache_path = os.path.join(request_python_dir, cache_filename)
        write_program_to_file(program_src, cache_path)

        # Phase 3: Evaluate directly (pure PyTorch)
        from eval_shared import evaluate_from_python_file

        tprint(f"[Torch-Eval-{request_id}] Processing on GPU {gpu_id}")

        result = evaluate_from_python_file(
            cache_path=cache_path,
            level_id=level_id,
            task_id=task_id,
            task=task["task"],
            seed_num=task["seed_num"],
            precision=precision,
            request_id=request_id,
            gpu_id=gpu_id,
        )

        # Add static check metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["static_warnings"] = warnings
        result["metadata"]["static_check_time_ms"] = static_check_time_ms

        tprint(f"[Torch-Eval-{request_id}] Evaluation completed")
        result_queue.put({"success": True, "result": result})

    except Exception as e:
        from server_utils import tprint
        tprint(f"[Torch-Eval-{request_id}] Error: {e}")
        traceback.print_exc()

        result_queue.put({
            "success": False,
            "result": {
                "compiled": False,
                "correctness": False,
                "metadata": {"worker_error": str(e)},
                "error": str(e),
                "error_name": type(e).__name__,
                "combined_score": 0.0,
            }
        })
    finally:
        try:
            if request_python_dir and os.path.exists(request_python_dir):
                shutil.rmtree(request_python_dir, ignore_errors=True)
            if request_build_dir and os.path.exists(request_build_dir):
                shutil.rmtree(request_build_dir, ignore_errors=True)
        except Exception as cleanup_error:
            from server_utils import tprint
            tprint(f"[Torch-Eval-{request_id}] Cleanup warning: {cleanup_error}")
