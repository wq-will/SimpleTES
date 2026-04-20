"""
Evaluator for circle packing problem (n=26)

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
import resource


def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_float(key, default):
    return float(os.environ.get(key, default))

CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 64)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 530)
N = 26


class EvaluatorTimeoutError(Exception):
    pass

class MemoryLimitExceededError(Exception):
    pass


def validate_solution(circles):
    if circles is None:
        return False, "Circles is None"
    if circles.ndim != 2 or circles.shape[1] != 3 or circles.shape[0] != N:
        return False, f"Invalid circles shape: {circles.shape}, expected ({N}, 3)"
    if np.isnan(circles).any():
        return False, "NaN values detected in circles"
    for i in range(N):
        x, y, r = circles[i]
        if r < 0:
            return False, f"Circle {i} has negative radius {r}"
        elif np.isnan(r):
            return False, f"Circle {i} has nan radius"
    for i in range(N):
        x, y, r = circles[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False, f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((circles[i, 0] - circles[j, 0]) ** 2 + (circles[i, 1] - circles[j, 1]) ** 2)
            if dist < circles[i, 2] + circles[j, 2] - 1e-12:
                return False, f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i, 2]+circles[j, 2]}"
    return True, None


def compute_score(circles):
    return float(np.sum(circles[:, 2]))


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
                raise MemoryLimitExceededError(f"Process killed by OS (likely OOM). Limit was {limit_mb:.2f}MB")
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
            else:
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


def _make_error_result(error_msg, eval_time=0.0):
    return {"sum_radii": 0.0, "validity": 0.0, "eval_time": float(eval_time), "combined_score": 0.0, "error": error_msg}


def evaluate(program_path):
    try:
        start_time = time.time()
        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)
        eval_time = time.time() - start_time
        circles = np.asarray(res.get("solution"), dtype=float)
        is_valid, error_msg = validate_solution(circles)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return {"sum_radii": 0.0, "validity": 0.0, "eval_time": float(eval_time), "combined_score": 0.0, "error": error_msg}
        capture_construction_if_requested(circles)
        sum_radii = compute_score(circles)
        reported = res.get("reported")
        if reported is not None and abs(sum_radii - reported) > 1e-6:
            print(f"Warning: Reported sum {reported} doesn't match calculated sum {sum_radii}")
        print(f"Evaluation: valid=True, sum_radii={sum_radii:.6f}, time={eval_time:.2f}s")
        return {"sum_radii": float(sum_radii), "validity": 1.0, "eval_time": float(eval_time), "combined_score": float(sum_radii)}
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
