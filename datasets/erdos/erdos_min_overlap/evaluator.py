"""
Evaluator for Erdős minimum overlap problem

Minimizing C₅ = max_k ∫ h(x)(1 - h(x+k)) dx
where h is a step function on [0, 2] → [0, 1] with ∫h = 1.

HACK-PROOF DESIGN:
- Programs return (h_values, c5_bound, n_points)
- Evaluator INDEPENDENTLY recomputes C5 from h_values
- Evaluator validates reported C5 matches computed C5 (tolerance: 1e-4)
- If mismatch detected → validation fails
- Programs cannot fake their scores!

This follows the same pattern as autocorrelation_second and matches
the original task.py + verifier.py design from raw_erdos.
"""

import numpy as np

from simpletes.construction import capture_construction_if_requested
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle
import psutil
import resource


# ============================================================================
# CONFIGURATION
# ============================================================================

def _env_int(key, default):
    """Get integer from environment variable or use default."""
    return int(os.environ.get(key, default))

def _env_float(key, default):
    """Get float from environment variable or use default."""
    return float(os.environ.get(key, default))

CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 64)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 1100)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class EvaluatorTimeoutError(Exception):
    """Raised when program execution exceeds the time limit."""
    pass


class MemoryLimitExceededError(Exception):
    """Raised when program execution exceeds the memory limit."""
    pass


# ============================================================================
# VALIDATION (from original verifier.py)
# ============================================================================

def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    """
    Verify the solution matches ttt_discover's verification logic.

    Args:
        h_values: np.ndarray of step heights
        c5_achieved: The C5 value claimed by the program
        n_points: Number of discretization points

    Returns:
        float: The verified C5 value

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    # Strict bounds check matching ttt_discover (no tolerance)
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    # Exact equality check matching ttt_discover
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    """
    Evaluate the Erdős solution (matches original verifier.py).
    
    Args:
        h_values: Step function values
        c5_bound: Claimed C5 bound
        n_points: Number of points
    
    Returns:
        float: The verified C5 value
    """
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


# ============================================================================
# SUBPROCESS EXECUTION
# ============================================================================

def _get_memory_limit_bytes():
    """Calculate per-process memory limit based on system resources."""
    total_mem = psutil.virtual_memory().total
    safe_mem = total_mem * (1.0 - OS_BUFFER_PERCENT)
    return int(safe_mem / CONCURRENT_PROCESSES)


def run_with_timeout(program_path, timeout_seconds=None):
    """
    Run the user program in a subprocess with timeout and memory limits.
    
    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time
    
    Returns:
        dict: {"solution": (h_values, c5_bound, n_points)}
    
    Raises:
        EvaluatorTimeoutError: If execution times out
        MemoryLimitExceededError: If memory limit exceeded
        RuntimeError: For other execution errors
    """
    if timeout_seconds is None:
        timeout_seconds = TIMEOUT_SECONDS
    
    limit_bytes = _get_memory_limit_bytes()
    limit_mb = limit_bytes / (1024 * 1024)
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f'''
import resource

def limit_memory():
    try:
        soft, hard = {limit_bytes}, {limit_bytes}
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
    except ValueError:
        pass

limit_memory()

import sys, os, pickle, traceback, numpy as np, importlib.util

sys.path.insert(0, os.path.dirname('{program_path}'))

def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mod = _load('{program_path}')
    out = mod.run_code() if hasattr(mod, 'run_code') and callable(getattr(mod, 'run_code')) else None
    if out is None:
        raise RuntimeError('Program must define run_code().')
    
    # Expect output format: (h_values, c5_bound, n_points)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        h_values = np.asarray(out[0], dtype=float)
        c5_bound = float(out[1])
        n_points = int(out[2])
        solution = (h_values, c5_bound, n_points)
    else:
        raise RuntimeError(f'Invalid output format. Expected (h_values, c5_bound, n_points), got {{type(out)}} with len={{len(out) if hasattr(out, "__len__") else "N/A"}}')
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"solution": solution}}, f)

except MemoryError:
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": "Memory limit exceeded (MemoryError caught)"}}, f)
except Exception as e:
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
'''
        temp_file.write(script.encode())
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        child_env = os.environ.copy()
        child_env["OMP_NUM_THREADS"] = "4"
        child_env["OPENBLAS_NUM_THREADS"] = "4"
        child_env["MKL_NUM_THREADS"] = "4"
        child_env["NUMEXPR_NUM_THREADS"] = "4"
        child_env["VECLIB_MAXIMUM_THREADS"] = "4"
        child_env["BLIS_NUM_THREADS"] = "4"

        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=child_env,
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
            
            if exit_code in (-9, -11):
                raise MemoryLimitExceededError(
                    f"Process killed by OS (likely OOM). Limit was {limit_mb:.2f}MB"
                )
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        results = pickle.load(f)
                    if "error" in results:
                        err_msg = results["error"]
                        if "MemoryError" in err_msg:
                            raise MemoryLimitExceededError(err_msg)
                        raise RuntimeError(f"Program execution failed: {err_msg}")
                    return results
                except (pickle.UnpicklingError, EOFError):
                    raise RuntimeError(
                        "Failed to read results file (possibly truncated due to crash)."
                    )
            else:
                if exit_code != 0:
                    err_out = stderr.decode()
                    if "MemoryError" in err_out:
                        raise MemoryLimitExceededError("Memory limit exceeded (stderr)")
                    raise RuntimeError(
                        f"Process exited with code {exit_code}. Stderr: {err_out}"
                    )
                raise RuntimeError("Results file not found but process exited 0.")
        
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(
                f"Process timed out after {timeout_seconds} seconds"
            )
    
    finally:
        for path in (temp_file_path, results_path):
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# MAIN EVALUATE FUNCTION
# ============================================================================

def _make_error_result(error_msg, eval_time=0.0):
    """Create a standardized error result dict."""
    return {
        "c5": float('inf'),
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """
    Evaluate the program by running it and computing the C₅ value.
    
    Args:
        program_path: Path to the program file
    
    Returns:
        dict: Evaluation metrics including:
            - c5: The C₅ overlap value (lower is better)
            - validity: 1.0 if valid, 0.0 otherwise
            - eval_time: Execution time in seconds
            - combined_score: 1/(1e-8 + c5) for valid solutions (higher is better)
            - error: (optional) Error message if evaluation failed
    """
    try:
        start_time = time.time()
        
        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)
        
        eval_time = time.time() - start_time
        
        solution = res.get("solution")
        if not isinstance(solution, (tuple, list)) or len(solution) != 3:
            return _make_error_result("Invalid solution format. Expected (h_values, c5_bound, n_points)", eval_time)
        
        h_values, c5_bound, n_points = solution
        h_values = np.asarray(h_values, dtype=float)
        c5_bound = float(c5_bound)
        n_points = int(n_points)
        
        # Use the original evaluation logic from verifier.py
        try:
            c5_value = evaluate_erdos_solution(h_values, c5_bound, n_points)
        except ValueError as e:
            print(f"Validation failed: {e}")
            return {
                "c5": float('inf'),
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
                "error": str(e),
            }
        
        # Combined score - matches original task.py line 39: return float(1.0 / (1e-8 + c5_bound))
        capture_construction_if_requested(solution)

        combined_score = 1.0 / (1e-8 + c5_value)
        
        print(f"Evaluation: valid=True, c5={c5_value:.10f}, score={combined_score:.10f}, time={eval_time:.2f}s")
        
        return {
            "c5": float(c5_value),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }
    
    except MemoryLimitExceededError as e:
        print(f"Evaluation failed due to memory limit: {e}")
        return _make_error_result(f"Memory limit exceeded: {e}")
    
    except EvaluatorTimeoutError as e:
        print(f"Evaluation failed due to timeout: {e}")
        return _make_error_result(f"Timeout: {e}")
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return _make_error_result(f"{type(e).__name__}: {e}")
