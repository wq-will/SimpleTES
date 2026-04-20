"""
Evaluator for third autocorrelation inequality (C₃ minimization)

Minimizing C₃ = 2·n·max(|conv(f,f)|) / (∑f)²
Lower values are better.
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
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 70)

# Problem-specific constants
# Known best bound (lower is better for C₃)
BENCHMARK = 1.4556427953745406

# Validation constants
MAX_CHECK_VALUE = 1e10
INTEGRAL_F_SQ_EPS = 1e-9
REPORTED_C3_ATOL = 1e-3


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

def validate_solution(heights, n_points):
    """
    Validate that heights array is valid for C₃ computation.

    Args:
        heights: np.array of shape (n,) with function values
        n_points: expected discretization size

    Returns:
        np.ndarray: normalized 1D float array

    Raises:
        ValueError: If validation fails
    """
    if heights is None:
        raise ValueError("Heights is None")

    try:
        heights = np.asarray(heights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert heights to numpy array: {exc}") from exc

    if heights.ndim != 1 or heights.size == 0:
        raise ValueError(f"Invalid heights shape: {heights.shape}, expected 1D non-empty array")

    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")

    if heights.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {heights.shape}.")

    if not np.all(np.isfinite(heights)):
        raise ValueError("Heights contain NaN or infinite values")

    max_height_abs = float(np.max(np.abs(heights)))
    if max_height_abs > MAX_CHECK_VALUE:
        raise ValueError(f"Heights contain extreme values (|h| > {MAX_CHECK_VALUE})")

    return heights


# ============================================================================
# SCORING
# ============================================================================

def compute_score(heights, n_points):
    """
    Compute the C₃ autoconvolution constant for given function values.

    C₃ = max |f ★ f| / (∫f)² on [-1/4, 1/4], discretized with n_points.

    Args:
        heights: np.array of shape (n,) with function values
        n_points: Number of discretization points

    Returns:
        float: The C₃ value
    """
    dx = 0.5 / int(n_points)
    integral_f_sq = (np.sum(heights) * dx) ** 2

    if integral_f_sq < INTEGRAL_F_SQ_EPS:
        raise ValueError("Function integral is close to zero, ratio is unstable.")

    conv_full = np.convolve(heights, heights, mode="full")
    scaled_conv = conv_full * dx
    max_abs_conv = np.max(np.abs(scaled_conv))
    c3 = max_abs_conv / integral_f_sq

    if not np.isfinite(c3):
        raise ValueError(f"Computed C3 is not finite: {c3}")

    return float(c3)


def verify_c3_solution(heights, reported_c3=None, n_points=None):
    """
    Verify the solution using the same logic as skydiscover.

    Args:
        heights: np.array of shape (n,) with function values
        reported_c3: Optional reported C3 value from the program
        n_points: Expected discretization size. Defaults to len(heights).

    Returns:
        float: Verified C3 value recomputed from heights
    """
    if n_points is None:
        if heights is None:
            raise ValueError("Heights is None")
        n_points = len(heights)

    heights = validate_solution(heights, n_points)
    computed_c3 = compute_score(heights, n_points)

    if reported_c3 is not None:
        try:
            reported_c3 = float(reported_c3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reported C3 is not a float: {reported_c3}") from exc

        if not np.isfinite(reported_c3):
            raise ValueError(f"reported C3 is not finite: {reported_c3}")

        delta = abs(computed_c3 - reported_c3)
        if delta > REPORTED_C3_ATOL:
            raise ValueError(
                f"C3 mismatch: reported {reported_c3:.6f}, "
                f"computed {computed_c3:.6f}, delta: {delta:.6f}"
            )

    return computed_c3


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
        dict: {"solution": np.ndarray, "reported": float or None, "loss": float or None, "n_points": int or None}

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

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError('run_code() returned an empty tuple/list.')
        solution = np.asarray(out[0], dtype=float)
        reported = float(out[1]) if len(out) >= 2 and np.isscalar(out[1]) else None
        loss = float(out[2]) if len(out) >= 3 and np.isscalar(out[2]) else None
        n_points = int(out[3]) if len(out) >= 4 and np.isscalar(out[3]) else None
    else:
        solution = np.asarray(out, dtype=float)
        reported = None
        loss = None
        n_points = None

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(
            {{
                "solution": solution,
                "reported": reported,
                "loss": loss,
                "n_points": n_points,
            }},
            f,
        )

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
        "c3": float("inf"),
        "n_points": 0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """
    Evaluate the program by running it and computing the C₃ value.

    Args:
        program_path: Path to the program file

    Returns:
        dict: Evaluation metrics including:
            - c3: The C₃ autoconvolution value (lower is better)
            - n_points: Number of points in the solution
            - validity: 1.0 if valid, 0.0 otherwise
            - eval_time: Execution time in seconds
            - combined_score: BENCHMARK / c3 for valid solutions (higher is better)
            - error: (optional) Error message if evaluation failed
    """
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        heights = np.asarray(res.get("solution"), dtype=float)
        n_points_value = res.get("n_points")
        n_points = int(n_points_value) if n_points_value is not None else len(heights)
        reported = res.get("reported")

        verified_c3 = verify_c3_solution(heights, reported_c3=reported, n_points=n_points)
        capture_construction_if_requested(heights)
        c3_value = float(reported) if reported is not None else float(verified_c3)

        # Combined score - higher is better (BENCHMARK / c3_value since lower C₃ is better)
        combined_score = (BENCHMARK / c3_value) if c3_value > 0 else 0.0

        print(
            f"Evaluation: valid=True, c3={c3_value:.6f}, "
            f"score={combined_score:.6f}, time={eval_time:.2f}s"
        )

        result = {
            "c3": float(c3_value),
            "n_points": int(n_points),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }
        loss = res.get("loss")
        if loss is not None:
            result["loss"] = float(loss)
        return result

    except MemoryLimitExceededError as e:
        eval_time = time.time() - start_time if "start_time" in locals() else 0.0
        print(f"Evaluation failed due to memory limit: {e}")
        return _make_error_result(f"Memory limit exceeded: {e}", eval_time=eval_time)

    except EvaluatorTimeoutError as e:
        eval_time = time.time() - start_time if "start_time" in locals() else 0.0
        print(f"Evaluation failed due to timeout: {e}")
        return _make_error_result(f"Timeout: {e}", eval_time=eval_time)

    except ValueError as e:
        eval_time = time.time() - start_time if "start_time" in locals() else 0.0
        print(f"Validation failed: {e}")
        return _make_error_result(f"ValueError: {e}", eval_time=eval_time)

    except Exception as e:
        eval_time = time.time() - start_time if "start_time" in locals() else 0.0
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return _make_error_result(f"{type(e).__name__}: {e}", eval_time=eval_time)
