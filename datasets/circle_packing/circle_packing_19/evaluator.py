"""
Evaluator for circle packing problem (n=19)

Packs n non-overlapping circles in a unit square, maximizing sum of radii.
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
import resource  # Required for hard memory limits (Linux only)


# ============================================================================
# CONFIGURATION (can be overridden via environment variables)
# ============================================================================

def _env_int(key, default):
    """Get integer from environment variable or use default."""
    return int(os.environ.get(key, default))

def _env_float(key, default):
    """Get float from environment variable or use default."""
    return float(os.environ.get(key, default))

# Concurrency & memory settings
CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 64)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)

# Timeout settings
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 530)

# Problem-specific constants
N = 19


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
# VALIDATION
# ============================================================================

def validate_solution(circles):
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        circles: np.array of shape (n, 3) with (x, y, r) for each circle

    Returns:
        tuple: (is_valid: bool, error_msg: str or None)
    """
    if circles is None:
        return False, "Circles is None"

    # Check shape
    if circles.ndim != 2 or circles.shape[1] != 3 or circles.shape[0] != N:
        return False, f"Invalid circles shape: {circles.shape}, expected ({N}, 3)"

    # Check for NaN values
    if np.isnan(circles).any():
        return False, "NaN values detected in circles"

    # Check if radii are nonnegative
    for i in range(N):
        x, y, r = circles[i]
        if r < 0:
            return False, f"Circle {i} has negative radius {r}"
        elif np.isnan(r):
            return False, f"Circle {i} has nan radius"

    # Check if circles are inside the unit square
    for i in range(N):
        x, y, r = circles[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False, f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"

    # Check for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((circles[i, 0] - circles[j, 0]) ** 2 + (circles[i, 1] - circles[j, 1]) ** 2)
            if dist < circles[i, 2] + circles[j, 2] - 1e-12:
                return False, f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i, 2]+circles[j, 2]}"

    return True, None


# ============================================================================
# SCORING
# ============================================================================

def compute_score(circles):
    """
    Compute the sum of radii.

    Args:
        circles: A validated circles array

    Returns:
        float: Sum of all radii
    """
    return float(np.sum(circles[:, 2]))


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
        timeout_seconds: Maximum execution time (defaults to TIMEOUT_SECONDS)

    Returns:
        dict: {"solution": np.ndarray, "reported": float or None}

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

    if isinstance(out, (tuple, list)) and len(out) >= 1:
        solution = np.asarray(out[0], dtype=float)
        reported = float(out[1]) if len(out) >= 2 and np.isscalar(out[1]) else None
    else:
        solution = np.asarray(out, dtype=float)
        reported = None

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"solution": solution, "reported": reported}}, f)

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
        "sum_radii": 0.0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """
    Evaluate the program by running it and checking the sum of radii.

    Args:
        program_path: Path to the program file

    Returns:
        dict: Evaluation metrics including:
            - sum_radii: Sum of all circle radii
            - validity: 1.0 if valid, 0.0 otherwise
            - eval_time: Execution time in seconds
            - combined_score: Same as sum_radii for valid solutions
            - error: (optional) Error message if evaluation failed
    """
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        circles = np.asarray(res.get("solution"), dtype=float)

        # Validate solution
        is_valid, error_msg = validate_solution(circles)

        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return {
                "sum_radii": 0.0,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
                "error": error_msg,
            }

        # Compute score
        capture_construction_if_requested(circles)
        sum_radii = compute_score(circles)

        # Check reported value if provided
        reported = res.get("reported")
        if reported is not None and abs(sum_radii - reported) > 1e-6:
            print(f"Warning: Reported sum {reported} doesn't match calculated sum {sum_radii}")

        print(f"Evaluation: valid=True, sum_radii={sum_radii:.6f}, time={eval_time:.2f}s")

        return {
            "sum_radii": float(sum_radii),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(sum_radii),
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
