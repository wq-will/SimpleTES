"""
Evaluator for first autocorrelation inequality problem (C1 minimization).

This evaluator verifies outputs with evaluate_sequence_c1(),
optimizes by minimizing C1, and sets
reward/combined_score = 1 / (1e-8 + C1).
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
# C1 VERIFIER LOGIC
# ============================================================================


def evaluate_sequence_c1(sequence):
    """
    C1 verifier logic for the first autocorrelation task.

    Returns +inf if the input is invalid.
    """
    if not isinstance(sequence, list):
        return float("inf")

    if not sequence:
        return float("inf")

    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return float("inf")
        if np.isnan(x) or np.isinf(x):
            return float("inf")

    sequence = [float(x) for x in sequence]
    sequence = [max(0.0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    if sum_a < 0.01:
        return float("inf")

    return float(2 * n * max_b / (sum_a**2))


def validate_solution(solution):
    """Validate a candidate solution for the C1 objective."""
    try:
        value = evaluate_sequence_c1(solution)
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
        "c1": float("inf"),
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "msg": msg,
    }


def _make_error_result(error_msg, eval_time=0.0):
    """Infrastructure/runtime error result."""
    return {
        "c1": float("inf"),
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """Evaluate the program and compute C1 metrics."""
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        solution = res.get("solution")

        is_valid, error_msg, c1_value = validate_solution(solution)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return _make_invalid_result(error_msg, eval_time=eval_time)

        capture_construction_if_requested(solution)

        reported = res.get("reported")
        if reported is not None and np.isfinite(reported) and abs(c1_value - reported) > 1e-4:
            print(f"Warning: Reported C1 {reported} does not match calculated {c1_value}")

        combined_score = 1.0 / (1e-8 + c1_value)

        print(
            f"Evaluation: valid=True, c1={c1_value:.6f}, "
            f"score={combined_score:.6f}, time={eval_time:.2f}s"
        )

        return {
            "c1": float(c1_value),
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
