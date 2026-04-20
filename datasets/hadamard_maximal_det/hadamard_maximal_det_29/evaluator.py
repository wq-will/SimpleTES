"""
Evaluator for Hadamard matrix optimization (n=29)

Finds ±1 matrices with maximum determinant. The theoretical maximum
for n=29 is 29^(29/2) ≈ 1.27×10^21.
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
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 350)

# Problem-specific constants
MATRIX_SIZE = 29
# Theoretical maximum determinant for n=29 Hadamard matrix: 29^(29/2)
THEORETICAL_MAX = 1270698346568170340352


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

def validate_solution(matrix):
    """
    Validate that matrix is a proper ±1 matrix of size 29x29.

    Args:
        matrix: np.array of shape (29, 29) with entries ±1

    Returns:
        tuple: (is_valid: bool, error_msg: str or None)
    """
    if matrix is None:
        return False, "Matrix is None"

    # Check shape
    if matrix.ndim != 2 or matrix.shape[0] != MATRIX_SIZE or matrix.shape[1] != MATRIX_SIZE:
        return False, f"Invalid matrix shape: {matrix.shape}, expected ({MATRIX_SIZE}, {MATRIX_SIZE})"

    # Check for NaN values
    if np.isnan(matrix).any():
        return False, "NaN values detected in matrix"

    # Check if all entries are ±1
    if not np.all(np.isin(matrix, [-1, 1])):
        return False, "Matrix entries must be +1 or -1"

    return True, None


# ============================================================================
# SCORING
# ============================================================================

def det_bareiss(A):
    """
    Bareiss algorithm for exact integer determinant calculation.
    
    Args:
        A: List of lists representing an integer matrix
        
    Returns:
        int: The determinant
    """
    n = len(A)
    if n == 0:
        return 1
    M = [row.copy() for row in A]
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return M[-1][-1]


def compute_score(matrix):
    """
    Compute the determinant ratio for the matrix.

    Args:
        matrix: A validated ±1 matrix

    Returns:
        tuple: (abs_determinant, determinant_ratio)
    """
    int_matrix = matrix.astype(int).tolist()
    det_exact = det_bareiss(int_matrix)
    abs_det = abs(det_exact)
    det_ratio = abs_det / THEORETICAL_MAX if THEORETICAL_MAX > 0 else 0.0
    return abs_det, det_ratio


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
    else:
        solution = np.asarray(out, dtype=float)

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
        "determinant_ratio": 0.0,
        "abs_determinant": 0.0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """
    Evaluate the program by running it and computing the determinant ratio.

    Args:
        program_path: Path to the program file

    Returns:
        dict: Evaluation metrics including:
            - determinant_ratio: |det(M)| / theoretical_max
            - abs_determinant: Absolute value of determinant
            - validity: 1.0 if valid, 0.0 otherwise
            - eval_time: Execution time in seconds
            - combined_score: Same as determinant_ratio for valid solutions
            - error: (optional) Error message if evaluation failed
    """
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        matrix = np.asarray(res.get("solution"), dtype=float)

        # Validate solution
        is_valid, error_msg = validate_solution(matrix)

        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return {
                "determinant_ratio": 0.0,
                "abs_determinant": 0.0,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
                "error": error_msg,
            }

        # Compute score
        capture_construction_if_requested(matrix)
        abs_det, det_ratio = compute_score(matrix)

        print(
            f"Evaluation: valid=True, det_ratio={det_ratio:.6f}, "
            f"abs_det={abs_det:.0f}, time={eval_time:.2f}s"
        )

        return {
            "determinant_ratio": float(det_ratio),
            "abs_determinant": float(abs_det),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(det_ratio),
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
