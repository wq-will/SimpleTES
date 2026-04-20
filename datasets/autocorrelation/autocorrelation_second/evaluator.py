"""
Evaluator for second autocorrelation inequality problem (AC2).

This evaluator mirrors the pasted AC2 logic:
- verify outputs with evaluate_sequence_ac2()
- optimize by maximizing C2 lower bound R(f)
- combined_score is the raw C2 value (no benchmark normalization)
"""

import os
import pickle
import resource  # Linux memory caps
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from simpletes.construction import capture_construction_if_requested
import psutil


# ============================================================================
# CONFIGURATION (can be overridden via environment variables)
# ============================================================================


def _env_int(key, default):
    """Get integer from environment variable or use default."""
    return int(os.environ.get(key, default))


# Concurrency and memory settings
CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 64)
OS_BUFFER_PERCENT = float(os.environ.get("EVALUATOR_OS_BUFFER_PERCENT", 0.10))

# Timeout settings
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 1100)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class EvaluatorTimeoutError(Exception):
    """Raised when program execution exceeds the time limit."""


class MemoryLimitExceededError(Exception):
    """Raised when program execution exceeds the memory limit."""


# ============================================================================
# AC2 VERIFIER LOGIC (mirrors pasted code)
# ============================================================================


def evaluate_sequence_ac2(sequence):
    """
    AC2 verifier logic from the pasted code.

    Raises:
        ValueError: for invalid sequence types/values.
    """
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)  # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


def validate_solution(solution):
    """Match AC2 validity semantics from the pasted env verifier."""
    try:
        value = evaluate_sequence_ac2(solution)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", float("inf")

    if value == float("inf"):
        return False, "Invalid solution.", value

    return True, None, float(value)


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

    Returns:
        dict: {"solution": object, "reported": float or None}
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

import importlib.util
import os
import pickle
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname("{program_path}"))


def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    mod = _load("{program_path}")
    out = mod.run_code() if hasattr(mod, "run_code") and callable(getattr(mod, "run_code")) else None
    if out is None:
        raise RuntimeError("Program must define run_code().")

    if isinstance(out, (tuple, list)) and len(out) >= 1:
        solution = out[0]
        reported = float(out[1]) if len(out) >= 2 and np.isscalar(out[1]) else None
    else:
        solution = out
        reported = None

    with open("{temp_file.name}.results", "wb") as f:
        pickle.dump({{"solution": solution, "reported": reported}}, f)

except MemoryError:
    with open("{temp_file.name}.results", "wb") as f:
        pickle.dump({{"error": "Memory limit exceeded (MemoryError caught)"}}, f)
except Exception as e:
    traceback.print_exc()
    with open("{temp_file.name}.results", "wb") as f:
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

            if exit_code != 0:
                err_out = stderr.decode()
                if "MemoryError" in err_out:
                    raise MemoryLimitExceededError("Memory limit exceeded (stderr)")
                raise RuntimeError(f"Process exited with code {exit_code}. Stderr: {err_out}")
            raise RuntimeError("Results file not found but process exited 0.")

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        for path in (temp_file_path, results_path):
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# MAIN EVALUATE FUNCTION
# ============================================================================


def _make_invalid_result(msg, eval_time=0.0):
    """Invalid construction result (logic failure, not infrastructure failure)."""
    return {
        "c2": 0.0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "msg": msg,
    }


def _make_error_result(error_msg, eval_time=0.0):
    """Infrastructure/runtime error result."""
    return {
        "c2": 0.0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """Evaluate the program and compute AC2 metrics."""
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        solution = res.get("solution")

        is_valid, error_msg, c2_value = validate_solution(solution)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return _make_invalid_result(error_msg, eval_time=eval_time)

        capture_construction_if_requested(solution)

        reported = res.get("reported")
        if reported is not None and np.isfinite(reported) and abs(c2_value - reported) > 1e-4:
            print(f"Warning: Reported C2 {reported} doesn't match calculated {c2_value}")

        combined_score = c2_value

        print(
            f"Evaluation: valid=True, c2={c2_value:.6f}, "
            f"score={combined_score:.6f}, time={eval_time:.2f}s"
        )

        return {
            "c2": float(c2_value),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except MemoryLimitExceededError as exc:
        print(f"Evaluation failed due to memory limit: {exc}")
        return _make_error_result(f"Memory limit exceeded: {exc}")

    except EvaluatorTimeoutError as exc:
        print(f"Evaluation failed due to timeout: {exc}")
        return _make_error_result(f"Timeout: {exc}")

    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        return _make_error_result(f"{type(exc).__name__}: {exc}")
