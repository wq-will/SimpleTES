import asyncio
import os
import tempfile
import time
from multiprocessing import get_context
from typing import Dict, List

from server_utils import tprint
from workers import (
    CUDACompilerWorker,
    CUDAEvalWorker,
    TorchEvalWorker,
    TritonCompilerEvalWorker,
    TritonEvalWorker,
)


class GPUAllocator:
    def __init__(
        self,
        num_gpus: int | None = None,
        gpu_ids: List[int] | None = None,
        capacity_per_gpu: int = 1,
    ):
        if gpu_ids is None:
            if num_gpus is None:
                raise ValueError("Either num_gpus or gpu_ids must be provided")
            gpu_ids = list(range(num_gpus))

        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")
        if capacity_per_gpu <= 0:
            raise ValueError("capacity_per_gpu must be > 0")

        self.gpu_ids = list(gpu_ids)
        self.capacity_per_gpu = capacity_per_gpu
        self.num_gpus = len(self.gpu_ids)
        self.total_slots = self.num_gpus * self.capacity_per_gpu

        self._initialized = False
        self._token_queue: asyncio.Queue | None = None
        self._in_use_counts: Dict[int, int] = {gpu_id: 0 for gpu_id in self.gpu_ids}

    def _ensure_initialized(self):
        if self._initialized:
            return

        self._token_queue = asyncio.Queue(maxsize=self.total_slots)
        for _ in range(self.capacity_per_gpu):
            for gpu_id in self.gpu_ids:
                self._token_queue.put_nowait(gpu_id)

        self._initialized = True

    async def acquire(self) -> int:
        self._ensure_initialized()
        assert self._token_queue is not None

        gpu_id = await self._token_queue.get()
        self._in_use_counts[gpu_id] += 1
        tprint(
            f"[GPUAllocator] Acquired GPU {gpu_id} "
            f"(in_use={self._in_use_counts[gpu_id]}/{self.capacity_per_gpu})"
        )
        return gpu_id

    def release(self, gpu_id: int):
        self._ensure_initialized()
        assert self._token_queue is not None

        if gpu_id not in self._in_use_counts:
            return
        if self._in_use_counts[gpu_id] <= 0:
            return

        self._in_use_counts[gpu_id] -= 1
        self._token_queue.put_nowait(gpu_id)
        tprint(
            f"[GPUAllocator] Released GPU {gpu_id} "
            f"(in_use={self._in_use_counts[gpu_id]}/{self.capacity_per_gpu})"
        )

    def get_status(self) -> Dict:
        self._ensure_initialized()
        return {
            "gpu_ids": self.gpu_ids,
            "num_gpus": self.num_gpus,
            "capacity_per_gpu": self.capacity_per_gpu,
            "total_slots": self.total_slots,
            "gpu_status": [
                {
                    "gpu_id": gpu_id,
                    "in_use": self._in_use_counts[gpu_id],
                    "capacity": self.capacity_per_gpu,
                    "busy": self._in_use_counts[gpu_id] >= self.capacity_per_gpu,
                }
                for gpu_id in self.gpu_ids
            ],
        }


class RobustSubprocessPool:
    def __init__(
        self,
        num_compile_workers: int,
        num_gpus: int,
        seed_num: int,
        precision: str,
        timeout: float,
        backend: str,
        extra_args: Dict | None = None,
    ):
        self.num_compile_workers = num_compile_workers
        self.num_gpus = num_gpus
        self.seed_num = seed_num
        self.precision = precision
        self.timeout = timeout
        self.backend = backend
        self.extra_args = extra_args or {}

        if backend == "triton":
            self.triton_compiler_gpus = self.extra_args.get("triton_compiler_gpus", [0, 1, 2, 3])
            self.triton_eval_gpus = self.extra_args.get("triton_eval_gpus", [4, 5, 6, 7])
            self.triton_compiler_gpu_concurrency = self.extra_args.get("triton_compiler_gpu_concurrency", 12)
            self.triton_perf_cpu_per_gpu = self.extra_args.get("triton_perf_cpu_per_gpu", 4)
            self.triton_perf_cpu_cores, self.triton_compiler_cpu_cores = self._derive_triton_cpu_cores()
        else:
            self.triton_compiler_gpus = None
            self.triton_eval_gpus = None
            self.triton_compiler_gpu_concurrency = None
            self.triton_perf_cpu_per_gpu = None
            self.triton_perf_cpu_cores = None
            self.triton_compiler_cpu_cores = None

        # Select worker classes based on backend
        if backend == "cuda":
            self.compile_worker_class = CUDACompilerWorker
            self.eval_worker_class = CUDAEvalWorker
            self.triton_compiler_worker_class = None
            self.triton_perf_worker_class = None
        elif backend == "triton":
            self.compile_worker_class = None
            self.eval_worker_class = None
            self.triton_compiler_worker_class = TritonCompilerEvalWorker
            self.triton_perf_worker_class = TritonEvalWorker
        elif backend == "torch":
            self.compile_worker_class = None
            self.eval_worker_class = TorchEvalWorker
            self.triton_compiler_worker_class = None
            self.triton_perf_worker_class = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.spawn_ctx = get_context("spawn")

        if self.backend == "triton":
            self.triton_compiler_allocator = GPUAllocator(
                gpu_ids=self.triton_compiler_gpus,
                capacity_per_gpu=self.triton_compiler_gpu_concurrency,
            )
            self.triton_eval_allocator = GPUAllocator(
                gpu_ids=self.triton_eval_gpus,
                capacity_per_gpu=1,
            )
            self.gpu_allocator = self.triton_eval_allocator
        else:
            self.triton_compiler_allocator = None
            self.triton_eval_allocator = None
            self.gpu_allocator = GPUAllocator(num_gpus=num_gpus, capacity_per_gpu=1)

        # Create cache directories
        self.python_files_dir = os.path.join(tempfile.gettempdir(), "gpu_kernel_eval", "python_files")
        self.build_cache_dir = os.path.join(tempfile.gettempdir(), "gpu_kernel_eval", "build_cache")
        os.makedirs(self.python_files_dir, exist_ok=True)
        os.makedirs(self.build_cache_dir, exist_ok=True)

        # Semaphore for compile concurrency (only for CUDA)
        self.compile_sem = None

        # Statistics
        self.requests_total = 0
        self.requests_processed = 0
        self.requests_failed = 0
        self.compile_tasks_completed = 0
        self.compile_tasks_failed = 0
        self.eval_tasks_completed = 0
        self.eval_tasks_failed = 0

        if self.backend == "triton":
            self.triton_compiler_eval_tasks_completed = 0
            self.triton_compiler_eval_tasks_failed = 0
            self.triton_perf_eval_tasks_completed = 0
            self.triton_perf_eval_tasks_failed = 0

        tprint("[Pool] Configuration (On-Demand Mode):")
        if self.backend == "cuda":
            tprint(f"  Max Compile Concurrency: {num_compile_workers}")
        elif self.backend == "triton":
            tprint("  Triton staged mode: compiler/correctness then perf")
            tprint(f"  Triton Compiler GPUs: {self.triton_compiler_gpus}")
            tprint(f"  Triton Eval GPUs: {self.triton_eval_gpus}")
            tprint(f"  Triton Compiler GPU Concurrency: {self.triton_compiler_gpu_concurrency}")
            tprint(f"  Triton Perf CPU Cores: {self.triton_perf_cpu_cores}")
            tprint(f"  Triton Compiler CPU Cores: {self.triton_compiler_cpu_cores}")
        else:
            tprint(f"  Compilation: Not needed for {backend}")

        if self.backend == "triton":
            tprint(
                f"  Max Triton CompilerEval Concurrency: "
                f"{len(self.triton_compiler_gpus)} x {self.triton_compiler_gpu_concurrency}"
            )
            tprint(f"  Max Triton Perf Eval Concurrency: {len(self.triton_eval_gpus)} (1 per GPU)")
        else:
            tprint(f"  Max Eval Concurrency: {num_gpus} (1 per GPU)")

        tprint(f"  Python Files Dir: {self.python_files_dir}")
        tprint(f"  Build Cache Dir: {self.build_cache_dir}")
        tprint(f"  Seed: {seed_num}")
        tprint(f"  Precision: {precision}")
        tprint(f"  Timeout: {timeout}s")
        tprint(f"  Backend: {backend}")

    def _derive_triton_cpu_cores(self) -> tuple[list[int] | None, list[int] | None]:
        total_cores = os.cpu_count()
        if total_cores is None or total_cores <= 0:
            return None, None

        perf_core_count = min(total_cores, len(self.triton_eval_gpus) * self.triton_perf_cpu_per_gpu)
        perf_cpu_cores = list(range(perf_core_count))

        compiler_cpu_cores = list(range(perf_core_count, total_cores))
        if not compiler_cpu_cores:
            compiler_cpu_cores = perf_cpu_cores.copy()

        return perf_cpu_cores, compiler_cpu_cores

    async def initialize(self):
        """Initialize pool (no pre-started workers in on-demand mode)"""
        tprint("[Pool] Initializing on-demand pool...")

        # Initialize semaphore in async context (only for CUDA)
        if self.backend == "cuda":
            self.compile_sem = asyncio.Semaphore(self.num_compile_workers)


    async def shutdown(self):
        """Shutdown pool and cleanup"""
        tprint("[Pool] Shutting down...")

        # Cleanup cache directories
        import shutil

        try:
            shutil.rmtree(self.python_files_dir, ignore_errors=True)
            tprint(f"[Pool] Cleaned up python files dir: {self.python_files_dir}")
        except Exception:
            pass

        try:
            shutil.rmtree(self.build_cache_dir, ignore_errors=True)
            tprint(f"[Pool] Cleaned up build cache dir: {self.build_cache_dir}")
        except Exception:
            pass

        tprint("[Pool] Shutdown complete. Stats:")
        tprint(
            f"  Requests - Total: {self.requests_total}, "
            f"Processed: {self.requests_processed}, Failed: {self.requests_failed}"
        )

        if self.backend == "cuda":
            tprint(
                f"  Compile - Completed: {self.compile_tasks_completed}, "
                f"Failed: {self.compile_tasks_failed}"
            )

        if self.backend == "triton":
            tprint(
                "  Triton CompilerEval - "
                f"Completed: {self.triton_compiler_eval_tasks_completed}, "
                f"Failed: {self.triton_compiler_eval_tasks_failed}"
            )
            tprint(
                "  Triton Perf Eval - "
                f"Completed: {self.triton_perf_eval_tasks_completed}, "
                f"Failed: {self.triton_perf_eval_tasks_failed}"
            )
        else:
            tprint(
                f"  Eval - Completed: {self.eval_tasks_completed}, "
                f"Failed: {self.eval_tasks_failed}"
            )

    async def process_request(self, program_src: str, level_id: int, task_id: int, task: str) -> Dict:
        """Process a complete evaluation request"""
        import uuid

        request_id = str(uuid.uuid4())[:8]

        self.requests_total += 1

        tprint(f"[{request_id}] START level{level_id}_{task_id} (backend={self.backend})")

        task = {
            "request_id": request_id,
            "program_src": program_src,
            "level_id": level_id,
            "task_id": task_id,
            "backend": self.backend,
            "precision": self.precision,
            "seed_num": self.seed_num,
            "task": task,
        }

        if self.backend == "triton":
            task["triton_perf_cpu_cores"] = self.triton_perf_cpu_cores
            task["triton_compiler_cpu_cores"] = self.triton_compiler_cpu_cores

        # CUDA: compile then eval
        if self.backend == "cuda":
            compile_result = await self._compile_phase(task)

            if not compile_result["compiled"]:
                self.requests_failed += 1
                tprint(f"[{request_id}] COMPILE FAILED")
                return {
                    "success": False,
                    "result": {
                        "compiled": False,
                        "correctness": False,
                        "metadata": compile_result.get("metadata", {}),
                        "error": compile_result.get("error", "Compilation failed"),
                        "error_name": compile_result.get("error_name", "CompilationFailed"),
                    },
                }

            tprint(f"[{request_id}] COMPILE SUCCESS")

            # Phase 2: Eval
            task["cache_path"] = compile_result["cache_path"]
            task["request_build_dir"] = compile_result.get("request_build_dir")

            eval_result = await self._eval_phase(task)

            # Merge compile timing into result metadata
            if eval_result.get("success") and eval_result.get("result"):
                if "metadata" not in eval_result["result"]:
                    eval_result["result"]["metadata"] = {}
                eval_result["result"]["metadata"]["compile_phase"] = compile_result.get("metadata", {})

        elif self.backend == "triton":
            compiler_eval_result = await self._triton_compiler_eval_phase(task)

            if not compiler_eval_result.get("success"):
                eval_result = compiler_eval_result
            elif not compiler_eval_result.get("result", {}).get("correctness"):
                tprint(f"[{request_id}] TRITON CORRECTNESS FAILED (skip perf)")
                eval_result = compiler_eval_result
            else:
                if compiler_eval_result.get("cache_path"):
                    task["cache_path"] = compiler_eval_result["cache_path"]
                if compiler_eval_result.get("request_python_dir"):
                    task["request_python_dir"] = compiler_eval_result["request_python_dir"]
                if compiler_eval_result.get("request_build_dir"):
                    task["request_build_dir"] = compiler_eval_result["request_build_dir"]

                eval_result = await self._triton_perf_eval_phase(task)

                if eval_result.get("result") is not None:
                    if "metadata" not in eval_result["result"]:
                        eval_result["result"]["metadata"] = {}
                    eval_result["result"]["metadata"]["triton_compiler_eval_phase"] = (
                        compiler_eval_result.get("result", {}).get("metadata", {})
                    )

        # Torch: eval only
        else:
            assert self.backend == "torch"
            eval_result = await self._eval_phase(task)

        if eval_result.get("success") and eval_result["result"].get("correctness"):
            self.requests_processed += 1
        else:
            self.requests_failed += 1

        tprint(f"[{request_id}] COMPLETE")

        return eval_result

    async def _compile_phase(self, task: Dict) -> Dict:
        """Compile phase: validate and cache code (CUDA only)"""
        request_id = task["request_id"]

        async with self.compile_sem:
            worker = self.compile_worker_class(
                spawn_ctx=self.spawn_ctx,
                python_files_dir=self.python_files_dir,
                build_cache_dir=self.build_cache_dir,
                timeout=self.timeout,
            )

            t0 = time.time()
            result = await worker.submit(task)
            elapsed_ms = (time.time() - t0) * 1000

            if result["compiled"]:
                self.compile_tasks_completed += 1
                tprint(f"[{request_id}] Compile phase: {elapsed_ms:.1f}ms")
            else:
                self.compile_tasks_failed += 1

            return result

    async def _eval_phase(self, task: Dict) -> Dict:
        """Eval phase: evaluate on GPU (CUDA/Torch)"""
        request_id = task["request_id"]

        gpu_id = await self.gpu_allocator.acquire()

        try:
            worker = self.eval_worker_class(
                spawn_ctx=self.spawn_ctx,
                gpu_id=gpu_id,
                build_cache_dir=self.build_cache_dir,
                timeout=self.timeout,
            )

            t0 = time.time()
            result = await worker.submit(task)
            elapsed_ms = (time.time() - t0) * 1000

            if result.get("success"):
                self.eval_tasks_completed += 1
                tprint(f"[{request_id}] Eval phase: {elapsed_ms:.1f}ms")
            else:
                self.eval_tasks_failed += 1

            return result
        except Exception as e:
            tprint(f"[{request_id}] Error: {e}")
            self.eval_tasks_failed += 1
            return {
                "success": False,
                "result": {
                    "error": str(e),
                    "error_name": type(e).__name__,
                    "compiled": False,
                    "correctness": False,
                    "metadata": {"eval_worker_error": str(e)},
                    "combined_score": 0.0,
                },
            }
        finally:
            self.gpu_allocator.release(gpu_id)

    async def _triton_compiler_eval_phase(self, task: Dict) -> Dict:
        request_id = task["request_id"]

        assert self.triton_compiler_allocator is not None
        gpu_id = await self.triton_compiler_allocator.acquire()

        try:
            worker = self.triton_compiler_worker_class(
                spawn_ctx=self.spawn_ctx,
                gpu_id=gpu_id,
                build_cache_dir=self.build_cache_dir,
                timeout=self.timeout,
            )

            t0 = time.time()
            result = await worker.submit(task)
            elapsed_ms = (time.time() - t0) * 1000

            if result.get("success"):
                self.triton_compiler_eval_tasks_completed += 1
                tprint(f"[{request_id}] Triton compiler-eval phase: {elapsed_ms:.1f}ms")
            else:
                self.triton_compiler_eval_tasks_failed += 1

            return result
        except Exception as e:
            tprint(f"[{request_id}] Triton compiler-eval error: {e}")
            self.triton_compiler_eval_tasks_failed += 1
            return {
                "success": False,
                "result": {
                    "compiled": False,
                    "correctness": False,
                    "error": str(e),
                    "error_name": type(e).__name__,
                    "combined_score": 0.0,
                },
            }
        finally:
            self.triton_compiler_allocator.release(gpu_id)

    async def _triton_perf_eval_phase(self, task: Dict) -> Dict:
        request_id = task["request_id"]

        assert self.triton_eval_allocator is not None
        gpu_id = await self.triton_eval_allocator.acquire()

        try:
            worker = self.triton_perf_worker_class(
                spawn_ctx=self.spawn_ctx,
                gpu_id=gpu_id,
                build_cache_dir=self.build_cache_dir,
                timeout=self.timeout,
            )

            t0 = time.time()
            result = await worker.submit(task)
            elapsed_ms = (time.time() - t0) * 1000

            if result.get("success"):
                self.triton_perf_eval_tasks_completed += 1
                tprint(f"[{request_id}] Triton perf-eval phase: {elapsed_ms:.1f}ms")
            else:
                self.triton_perf_eval_tasks_failed += 1

            return result
        except Exception as e:
            tprint(f"[{request_id}] Triton perf-eval error: {e}")
            self.triton_perf_eval_tasks_failed += 1
            return {
                "success": False,
                "error": str(e),
                "result": {
                    "compiled": False,
                    "correctness": False,
                    "metadata": {"triton_perf_eval_worker_error": str(e)},
                    "runtime_stats": {},
                    "ref_runtime_stats": {},
                    "combined_score": 0.0,
                },
            }
        finally:
            self.triton_eval_allocator.release(gpu_id)

    def get_stats(self) -> Dict:
        """Get pool statistics"""
        stats = {
            "requests_total": self.requests_total,
            "requests_processed": self.requests_processed,
            "requests_failed": self.requests_failed,
        }

        if self.backend == "triton":
            stats["triton_compiler_eval"] = {
                "gpu_ids": self.triton_compiler_gpus,
                "capacity_per_gpu": self.triton_compiler_gpu_concurrency,
                "cpu_cores": self.triton_compiler_cpu_cores,
                "tasks_completed": self.triton_compiler_eval_tasks_completed,
                "tasks_failed": self.triton_compiler_eval_tasks_failed,
                "gpu_status": self.triton_compiler_allocator.get_status()
                if self.triton_compiler_allocator and self.triton_compiler_allocator._initialized
                else {},
            }
            stats["triton_perf_eval"] = {
                "gpu_ids": self.triton_eval_gpus,
                "capacity_per_gpu": 1,
                "cpu_cores": self.triton_perf_cpu_cores,
                "tasks_completed": self.triton_perf_eval_tasks_completed,
                "tasks_failed": self.triton_perf_eval_tasks_failed,
                "gpu_status": self.triton_eval_allocator.get_status()
                if self.triton_eval_allocator and self.triton_eval_allocator._initialized
                else {},
            }
        else:
            stats["eval_workers"] = {
                "num_gpus": self.num_gpus,
                "tasks_completed": self.eval_tasks_completed,
                "tasks_failed": self.eval_tasks_failed,
            }
            stats["gpu_status"] = self.gpu_allocator.get_status() if self.gpu_allocator._initialized else {}

        if self.backend == "cuda":
            stats["compile_workers"] = {
                "max_concurrency": self.num_compile_workers,
                "tasks_completed": self.compile_tasks_completed,
                "tasks_failed": self.compile_tasks_failed,
            }

        return stats

    def get_worker_status(self) -> Dict:
        """Get worker status (simplified for on-demand mode)"""
        status = {
            "mode": "on-demand",
            "backend": self.backend,
        }

        if self.backend == "triton":
            status["triton_compiler_eval_allocator"] = (
                self.triton_compiler_allocator.get_status()
                if self.triton_compiler_allocator and self.triton_compiler_allocator._initialized
                else {}
            )
            status["triton_perf_eval_allocator"] = (
                self.triton_eval_allocator.get_status()
                if self.triton_eval_allocator and self.triton_eval_allocator._initialized
                else {}
            )
            status["triton_compiler_cpu_cores"] = self.triton_compiler_cpu_cores
            status["triton_perf_cpu_cores"] = self.triton_perf_cpu_cores
        else:
            status["gpu_allocator"] = self.gpu_allocator.get_status() if self.gpu_allocator._initialized else {}

        if self.backend == "cuda":
            status["compile_semaphore"] = {
                "max": self.num_compile_workers,
                "available": self.compile_sem._value if self.compile_sem else self.num_compile_workers,
            }

        return status
