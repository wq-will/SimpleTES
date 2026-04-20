"""Evaluator for the sums/differences lower-bound task."""

import math
import os
import pickle
import psutil
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from simpletes.construction import capture_construction_if_requested


def _env_int(key, default):
    return int(os.environ.get(key, default))


def _env_float(key, default):
    return float(os.environ.get(key, default))


CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 64)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 180)

MIN_SET_SIZE = 2
MAX_SET_SIZE = 512
MIN_INT = -1_000_000
MAX_INT = 1_000_000
INTEGER_ATOL = _env_float("EVALUATOR_INTEGER_ATOL", 1e-9)
REPORTED_ATOL = _env_float("EVALUATOR_REPORTED_ATOL", 1e-9)


class EvaluatorTimeoutError(Exception):
    pass


class MemoryLimitExceededError(Exception):
    pass


def _normalize_candidate(values):
    """Convert user output to a validated sorted unique integer list."""
    if values is None:
        raise ValueError("Candidate set is None")

    if isinstance(values, np.ndarray):
        raw = values.reshape(-1).tolist()
    else:
        try:
            raw = list(values)
        except TypeError as e:
            raise ValueError(f"Candidate set is not iterable: {e}")

    ints = []
    for idx, x in enumerate(raw):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"Element {idx} is not numeric: {x}")
        if not math.isfinite(xf):
            raise ValueError(f"Element {idx} is not finite: {xf}")

        xr = round(xf)
        if abs(xf - xr) > INTEGER_ATOL:
            raise ValueError(f"Element {idx} is not an integer: {xf}")

        xi = int(xr)
        if xi < MIN_INT or xi > MAX_INT:
            raise ValueError(
                f"Element {idx}={xi} is outside [{MIN_INT}, {MAX_INT}]"
            )
        ints.append(xi)

    unique_vals = sorted(set(ints))

    if len(unique_vals) < MIN_SET_SIZE:
        raise ValueError(
            f"Need at least {MIN_SET_SIZE} distinct integers, got {len(unique_vals)}"
        )

    if len(unique_vals) > MAX_SET_SIZE:
        raise ValueError(
            f"Too many distinct integers: {len(unique_vals)} > {MAX_SET_SIZE}"
        )

    return unique_vals


def _compute_stats(values):
    """Compute |A+A|, |A-A|, and C(A)."""
    n = len(values)

    sumset = {a + b for a in values for b in values}
    diffset = {a - b for a in values for b in values}

    sumset_size = len(sumset)
    diffset_size = len(diffset)

    sum_ratio = sumset_size / n
    diff_ratio = diffset_size / n

    if sum_ratio <= 1.0 or diff_ratio <= 1.0:
        raise ValueError(
            "Invalid ratios: both |A+A|/|A| and |A-A|/|A| must be > 1"
        )

    c_value = math.log(sum_ratio) / math.log(diff_ratio)
    if not math.isfinite(c_value):
        raise ValueError(f"Computed C(A) is not finite: {c_value}")

    return {
        "set_size": int(n),
        "sumset_size": int(sumset_size),
        "diffset_size": int(diffset_size),
        "sum_ratio": float(sum_ratio),
        "diff_ratio": float(diff_ratio),
        "c_value": float(c_value),
    }


def evaluate_sums_diffs_solution(candidate, reported_c=None):
    values = _normalize_candidate(candidate)
    stats = _compute_stats(values)

    if reported_c is not None:
        try:
            rep = float(reported_c)
        except (TypeError, ValueError):
            raise ValueError(f"reported_c is not a float: {reported_c}")
        if not math.isfinite(rep):
            raise ValueError(f"reported_c is not finite: {rep}")
        if abs(rep - stats["c_value"]) > REPORTED_ATOL:
            raise ValueError(
                f"Reported C(A) mismatch: reported {rep:.12f}, "
                f"computed {stats['c_value']:.12f}"
            )

    return stats


def _get_memory_limit_bytes():
    total_mem = psutil.virtual_memory().total
    safe_mem = total_mem * (1.0 - OS_BUFFER_PERCENT)
    return int(safe_mem / CONCURRENT_PROCESSES)


def run_with_timeout(program_path, timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = TIMEOUT_SECONDS

    limit_bytes = _get_memory_limit_bytes()
    limit_mb = limit_bytes / (1024 * 1024)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
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
sys.path.insert(0, os.path.dirname({program_path!r}))

def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mod = _load({program_path!r})
    out = mod.run_code() if hasattr(mod, 'run_code') and callable(getattr(mod, 'run_code')) else None
    if out is None:
        raise RuntimeError('Program must define run_code().')

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError('run_code() returned an empty tuple/list.')
        solution = out[0]
        reported = float(out[1]) if len(out) >= 2 and np.isscalar(out[1]) else None
    else:
        solution = out
        reported = None

    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"solution": solution, "reported": reported}}, f)

except MemoryError:
    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"error": "Memory limit exceeded (MemoryError caught)"}}, f)
except Exception as e:
    traceback.print_exc()
    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
"""
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
            _stdout, stderr = process.communicate(timeout=timeout_seconds)
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
                    raise RuntimeError("Failed to read results file.")

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


def _make_error_result(error_msg, eval_time=0.0):
    return {
        "c_value": 0.0,
        "set_size": 0,
        "sumset_size": 0,
        "diffset_size": 0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """Run and score a candidate program."""
    try:
        start_time = time.time()
        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)
        eval_time = time.time() - start_time

        stats = evaluate_sums_diffs_solution(
            candidate=res.get("solution"),
            reported_c=res.get("reported"),
        )

        capture_construction_if_requested(res.get("solution"))
        c_value = stats["c_value"]
        print(
            "Evaluation: "
            f"valid=True, |A|={stats['set_size']}, "
            f"|A+A|={stats['sumset_size']}, |A-A|={stats['diffset_size']}, "
            f"C(A)={c_value:.10f}, time={eval_time:.2f}s"
        )
        
        return {
            "c_value": float(c_value),
            "set_size": int(stats["set_size"]),
            "sumset_size": int(stats["sumset_size"]),
            "diffset_size": int(stats["diffset_size"]),
            "sum_ratio": float(stats["sum_ratio"]),
            "diff_ratio": float(stats["diff_ratio"]),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(c_value),
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
