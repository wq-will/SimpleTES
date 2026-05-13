"""
Base classes and shared utilities for all backend workers
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import os
import time


def write_program_to_file(program_src: str, cache_path: str):
    """Write program source to file"""
    with open(cache_path, "w") as f:
        f.write(program_src)


def setup_build_env(request_build_dir: str):
    """Setup build cache environment variables"""
    os.environ["TORCH_EXTENSIONS_DIR"] = request_build_dir
    os.environ["XDG_CACHE_HOME"] = request_build_dir
    os.environ["CUDA_CACHE_PATH"] = os.path.join(request_build_dir, "cuda")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(request_build_dir, "triton")
    os.environ["TILELANG_CACHE_DIR"] = os.path.join(request_build_dir, "tilelang")

    # Keep legacy cache var aligned for code that reads this key.
    os.environ["FIB_CACHE_PATH"] = request_build_dir


def validate_program_syntax(program_src: str) -> float:
    """Validate Python syntax, return compile time in ms"""
    t0 = time.time()
    compile(program_src, "<string>", "exec")
    return (time.time() - t0) * 1000


def static_check_program(program_src: str, backend: str, precision: str) -> Tuple[bool, list, list, list, float]:
    """Run static checks, return (valid, errors, warnings, bypass, time_ms)"""
    import sys
    import os

    # Add parent dir to path for kernel_static_checker import
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    from kernel_static_checker import validate_kernel_static

    t0 = time.time()
    valid, errors, warnings, bypass = validate_kernel_static(
        code=program_src,
        backend=backend,
        precision=precision,
    )
    check_time_ms = (time.time() - t0) * 1000
    return valid, errors, warnings, bypass, check_time_ms


class BaseCompilerWorker(ABC):
    """Base class for compiler workers (backends that need compilation)"""

    def __init__(self, spawn_ctx, python_files_dir: str, build_cache_dir: str, timeout: float = 600.0):
        self.spawn_ctx = spawn_ctx
        self.python_files_dir = python_files_dir
        self.build_cache_dir = build_cache_dir
        self.timeout = timeout

    @abstractmethod
    async def submit(self, task: Dict) -> Dict:
        """Submit compilation task"""
        pass


class BaseEvalWorker(ABC):
    """Base class for eval workers"""

    def __init__(self, spawn_ctx, gpu_id: int, build_cache_dir: str, timeout: float = 600.0):
        self.spawn_ctx = spawn_ctx
        self.gpu_id = gpu_id
        self.build_cache_dir = build_cache_dir
        self.timeout = timeout

    @abstractmethod
    async def submit(self, task: Dict) -> Dict:
        """Submit evaluation task"""
        pass

    def _default_failed_result(self) -> Dict:
        return {
            "compiled": False,
            "correctness": False,
            "error": "Worker crashed or timeout",
            "error_name": "Worker Crashed or Timeout",
            "runtime_stats": {},
            "ref_runtime_stats": {},
            "combined_score": 0.0,
        }
