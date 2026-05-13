"""
Utility functions for evaluation server
"""

import os
import subprocess
import asyncio
import torch
from datetime import datetime

def get_cpu_count() -> int:
    return os.cpu_count() or 96

def get_gpu_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def tprint(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs, flush=True)

def get_error_name(e: Exception) -> str:
    """
    Get the error name, for logging purposes
    """
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def get_torch_dtype_from_string(precision: str) -> torch.dtype:
    """
    Get the torch dtype for specific precision
    """
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else: # future, FP8, FP4, etc. support?
        raise ValueError(f"Invalid precision not supported: {precision}")


def get_tolerance_for_precision(precision: str | torch.dtype) -> float:
    """
    Get the tolerance from a string representing the percision.
    These tolerances are inspired by torchbench (PyTorch Benchmarking Suite): 
    Reference:
    https://github.com/pytorch/benchmark/blob/cfd835c35d04513ced9a59bd074eeb21dc8187d7/torchbenchmark/util/env_check.py#L519
    """
    if isinstance(precision, str):
        precision = get_torch_dtype_from_string(precision)

    PRECISION_TOLERANCES = {
        # By default for fp32, 1e-4 is used according to torchbench.
        torch.float32: 1e-4,
        # torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if it's too strict. 
        # @todo: Let user configure own tolerance as an option
        torch.float16: 1e-2, 
        torch.bfloat16: 1e-2,
    }
    assert precision in PRECISION_TOLERANCES, f"Invalid precision not supported: {precision}"
    return PRECISION_TOLERANCES[precision]


def clear_l2_cache(device: torch.device | str = "cuda"):
    """
    Clear L2 Cache line by thrashing with a large tensor
    Acknowledge GPU mode reference kernel repo:
    https://github.com/gpu-mode/reference-kernels/commit/7c15075a39286e88939d99d3f3a60be88b8e6223#diff-3a30a71cbf8db2badd224f4d92f9a2546925a5b522632a31d353526b7a5f3338R158-R163
    """
    # don't reserve space for persisting lines
    # cp.cuda.runtime.cudaDeviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
    
    # Thrash L2 cache by creating a larger dummy tensor, effectively flushing the cache
    # 32 * 1024 * 1024 * 8B = 256MB 
    # NOTE: we can make this more adaptive based on device
    # L2 cache sizes: A100=40MB, H100=50MB, H200=90MB, RTX4090=72MB, L40S=48MB, Blackwell≈192MB → overwrite >200MB to fully thrash L2
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    # write to tensor with inplace fill
    dummy.fill_(42) 
    del dummy


def clear_l2_cache_triton(cache=None, device: str = "cuda"):
    """
    Thrash the cache by making a large dummy tensor, using triton runtime's functionality
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    from triton import runtime as triton_runtime
    with torch.cuda.device(device):
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()
        # this effectively thrashes L2 cache under the hood too
        triton_runtime.driver.active.clear_cache(cache)


def check_gpu_utilization(device_id: int) -> float:
    """
    Check GPU utilization percentage using nvidia-smi.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        GPU utilization percentage (0-100), or -1 if check failed
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print(f"[Server] Warning: nvidia-smi failed for GPU {device_id}")
            return -1
        
        # Parse output: "0, 5\n1, 10\n..."
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 2:
                gpu_idx = int(parts[0].strip())
                if gpu_idx == device_id:
                    utilization = float(parts[1].strip())
                    return utilization
        
        print(f"[Server] Warning: GPU {device_id} not found in nvidia-smi output")
        return -1
    except subprocess.TimeoutExpired:
        print(f"[Server] Warning: nvidia-smi timeout for GPU {device_id}")
        return -1
    except Exception as e:
        print(f"[Server] Warning: Failed to check GPU {device_id} utilization: {e}")
        return -1


async def check_gpu_available(device_id: int, num_checks: int = 3, check_interval: float = 1.0, max_utilization: float = 5.0) -> bool:
    """
    Check if GPU is available by monitoring utilization over multiple checks.
    
    Args:
        device_id: GPU device ID
        num_checks: Number of times to check (default: 3)
        check_interval: Seconds between checks (default: 1.0)
        max_utilization: Maximum allowed utilization percentage (default: 5.0%)
        
    Returns:
        True if GPU is available (all checks show low utilization), False otherwise
    """
    print(f"[Server] Checking GPU {device_id} availability...")
    
    for check_idx in range(num_checks):
        utilization = check_gpu_utilization(device_id)
        
        if utilization < 0:
            # If check failed, assume GPU is not available to be safe
            print(f"[Server] GPU {device_id} check {check_idx + 1}/{num_checks}: Check failed, skipping GPU")
            return False
        
        print(f"[Server] GPU {device_id} check {check_idx + 1}/{num_checks}: Utilization = {utilization:.1f}%")
        
        if utilization > max_utilization:
            print(f"[Server] GPU {device_id} has high utilization ({utilization:.1f}% > {max_utilization}%), skipping")
            return False
        
        # Wait before next check (except for the last one)
        if check_idx < num_checks - 1:
            await asyncio.sleep(check_interval)
    
    print(f"[Server] GPU {device_id} passed all availability checks")
    return True
