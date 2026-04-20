"""
Evaluator for single-cell RNA-seq denoising problem
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

def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Torch-free Poisson negative log-likelihood used by this task.

    This matches `molecular_cross_validation.mcv_sweep.poisson_nll_loss` exactly.
    """
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


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
CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 256)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)

# Timeout settings
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 400)

# Problem-specific constants: baseline scores per dataset
BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
    "pbmc": {
        "baseline_mse": 0.270945,
        "baseline_poisson": 0.300447,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.043569,
    },
    "tabula": {
        "baseline_mse": 0.261763,
        "baseline_poisson": 0.206542,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.026961,
    },
}

# Poisson constraint threshold
POISSON_NORM_THRESHOLD = 0.97

# Data directory for pre-cached train/test splits
import os as _os
DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "denoising_datasets", "pancreas")

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

def validate_solution(result):
    """
    Validate that the denoising result is well-formed.

    Args:
        result: tuple of (mse, poisson) from run_code

    Returns:
        tuple: (is_valid: bool, error_msg: str or None)
    """
    if result is None:
        return False, "Result is None"

    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return False, f"Result must be a tuple of (mse, poisson), got {type(result)}"

    mse, poisson = result[0], result[1]

    # Check for finite values
    if not np.isfinite(mse):
        return False, f"MSE is not finite: {mse}"
    if not np.isfinite(poisson):
        return False, f"Poisson loss is not finite: {poisson}"

    # Check for negative values
    if mse < 0:
        return False, f"MSE is negative: {mse}"
    if poisson < 0:
        return False, f"Poisson loss is negative: {poisson}"

    # Check Poisson is not suspiciously below the theoretical minimum
    baseline = BASELINES["pancreas"]
    if poisson < baseline["perfect_poisson"]:
        return False, (
            f"Poisson loss {poisson:.6f} is below theoretical minimum "
            f"{baseline['perfect_poisson']:.6f} - likely an error"
        )

    return True, None


# ============================================================================
# SCORING
# ============================================================================

def compute_scores(result, dataset="pancreas"):
    """
    Compute normalized scores from raw MSE and Poisson loss values.

    Args:
        result: tuple of (mse, poisson)
        dataset: which baseline to use

    Returns:
        dict with raw and normalized scores
    """
    mse, poisson = result[0], result[1]
    baseline = BASELINES[dataset]

    # Normalize Poisson score
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    if poisson_range > 0:
        poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range
    else:
        poisson_norm = 0.0

    # Normalize MSE score
    mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
    if mse_range > 0:
        mse_norm = (baseline["baseline_mse"] - mse) / mse_range
    else:
        mse_norm = 0.0
    mse_norm = max(0.0, min(1.0, mse_norm))

    # Poisson is a hard constraint
    poisson_pass = poisson_norm >= POISSON_NORM_THRESHOLD

    return {
        "mse_raw": float(mse),
        "poisson_raw": float(poisson),
        "mse_norm": float(mse_norm),
        "poisson_norm": float(poisson_norm),
        "poisson_pass": poisson_pass,
    }


# ============================================================================
# SUBPROCESS EXECUTION
# ============================================================================

def _get_memory_limit_bytes():
    """Calculate per-process memory limit based on system resources."""
    total_mem = psutil.virtual_memory().total
    safe_mem = total_mem * (1.0 - OS_BUFFER_PERCENT)
    return int(safe_mem / CONCURRENT_PROCESSES)



def _compute_metrics(Y_denoised, X_train, X_test):
    """Compute MSE and Poisson metrics. Runs in outer process only."""
    import anndata
    import scanpy as sc
    import sklearn.metrics
    import scprep

    test_X = scprep.utils.toarray(X_test).copy()
    denoised_X = np.asarray(Y_denoised).copy()

    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)

    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)

    mse = sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)

    test_X_poisson = scprep.utils.toarray(X_test)
    denoised_X_poisson = np.asarray(Y_denoised).copy()
    initial_sum = X_train.sum()
    target_sum = test_X_poisson.sum()
    denoised_scaled = denoised_X_poisson * target_sum / initial_sum
    poisson = poisson_nll_loss(test_X_poisson, denoised_scaled)

    return (float(mse), float(poisson))


def run_with_timeout(program_path, timeout_seconds=None):
    """
    Run the user program in a subprocess with timeout and memory limits.

    The evolved code subprocess only receives X_train and writes Y_denoised.
    X_test and metric computation happen in the outer process after the
    subprocess exits, so the evolved code can never access the test set.
    """
    if timeout_seconds is None:
        timeout_seconds = TIMEOUT_SECONDS

    limit_bytes = _get_memory_limit_bytes()
    limit_mb = limit_bytes / (1024 * 1024)
    x_train_path = repr(os.path.join(DATA_DIR, "pancreas_train_seed42.npy"))

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
import types

# Torch-free shim
mcv_pkg = types.ModuleType("molecular_cross_validation")
mcv_sweep_mod = types.ModuleType("molecular_cross_validation.mcv_sweep")
def _poisson_nll_loss(y_pred, y_true):
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()
mcv_sweep_mod.poisson_nll_loss = _poisson_nll_loss
mcv_pkg.mcv_sweep = mcv_sweep_mod
sys.modules.setdefault("molecular_cross_validation", mcv_pkg)
sys.modules.setdefault("molecular_cross_validation.mcv_sweep", mcv_sweep_mod)

sys.path.insert(0, os.path.dirname('{program_path}'))

def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mod = _load('{program_path}')

    if not hasattr(mod, 'magic_denoise') or not callable(getattr(mod, 'magic_denoise')):
        raise RuntimeError('Program must define magic_denoise().')

    # Load X_train only — X_test is never present in this process
    X_train = np.load({x_train_path})

    Y_denoised = mod.magic_denoise(X_train, random_state=42)

    if not np.isfinite(Y_denoised).all():
        raise ValueError("Y_denoised contains non-finite values")
    if np.any(Y_denoised < 0):
        raise ValueError("Y_denoised contains negative values")
    if Y_denoised.max() > X_train.sum():
        raise ValueError("Y_denoised max exceeds X_train total count")

    with open('{temp_file.name}.denoised', 'wb') as f:
        pickle.dump(Y_denoised, f)

except MemoryError:
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": "Memory limit exceeded (MemoryError caught)"}}, f)
except Exception as e:
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"
    denoised_path = f"{temp_file_path}.denoised"

    try:
        child_env = os.environ.copy()
        child_env["OMP_NUM_THREADS"] = "4"
        child_env["OPENBLAS_NUM_THREADS"] = "4"
        child_env["MKL_NUM_THREADS"] = "4"
        child_env["NUMEXPR_NUM_THREADS"] = "4"

        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=True
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            if exit_code in (-9, -11):
                raise MemoryLimitExceededError(
                    f"Process killed by OS (likely OOM). Limit was {limit_mb:.2f}MB"
                )

            # Check if subprocess wrote an error result
            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        results = pickle.load(f)
                    if "error" in results:
                        err_msg = results["error"]
                        if "MemoryError" in err_msg:
                            raise MemoryLimitExceededError(err_msg)
                        raise RuntimeError(f"Program execution failed: {err_msg}")
                except (pickle.UnpicklingError, EOFError):
                    raise RuntimeError(
                        "Failed to read results file (possibly truncated due to crash)."
                    )

            # Check subprocess wrote Y_denoised
            if not os.path.exists(denoised_path):
                if exit_code != 0:
                    err_out = stderr.decode()
                    if "MemoryError" in err_out:
                        raise MemoryLimitExceededError("Memory limit exceeded (stderr)")
                    raise RuntimeError(
                        f"Process exited with code {exit_code}. Stderr: {err_out}"
                    )
                raise RuntimeError("Denoised output not found but process exited 0.")

            # Load Y_denoised and compute metrics — X_test never entered the subprocess
            try:
                with open(denoised_path, "rb") as f:
                    Y_denoised = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                raise RuntimeError("Failed to read denoised output (possibly truncated).")

            X_train = np.load(os.path.join(DATA_DIR, "pancreas_train_seed42.npy"))
            X_test = np.load(os.path.join(DATA_DIR, "pancreas_test_seed42.npy"))
            solution = _compute_metrics(Y_denoised, X_train, X_test)
            return {"solution": solution}

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(
                f"Process timed out after {timeout_seconds} seconds"
            )

    finally:
        for path in (temp_file_path, results_path, denoised_path):
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# MAIN EVALUATE FUNCTION
# ============================================================================

def _make_error_result(error_msg, eval_time=0.0):
    """Create a standardized error result dict."""
    return {
        "mse_raw": 0.0,
        "poisson_raw": 0.0,
        "mse_norm": 0.0,
        "poisson_norm": 0.0,
        "poisson_pass": False,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": 0.0,
        "error": error_msg,
    }


def evaluate(program_path):
    """
    Evaluate the denoising program by running it and scoring the results.

    Args:
        program_path: Path to the program file

    Returns:
        dict: Evaluation metrics including:
            - mse_raw: Raw MSE in log-normalized space
            - poisson_raw: Raw Poisson negative log-likelihood
            - mse_norm: Normalized MSE score (0.0 to 1.0, higher is better)
            - poisson_norm: Normalized Poisson score (higher is better)
            - poisson_pass: Whether Poisson constraint is satisfied
            - validity: 1.0 if valid, 0.0 otherwise
            - eval_time: Execution time in seconds
            - combined_score: mse_norm if Poisson constraint passes, else 0.0
            - error: (optional) Error message if evaluation failed
    """
    try:
        start_time = time.time()

        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)

        eval_time = time.time() - start_time
        solution = res.get("solution")

        # Validate solution
        is_valid, error_msg = validate_solution(solution)

        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return {
                "mse_raw": 0.0,
                "poisson_raw": 0.0,
                "mse_norm": 0.0,
                "poisson_norm": 0.0,
                "poisson_pass": False,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
                "error": error_msg,
            }

        # Compute scores
        capture_construction_if_requested(solution)
        scores = compute_scores(solution)

        # Combined score: MSE norm if Poisson constraint passes, else 0
        if scores["poisson_pass"]:
            combined_score = scores["mse_norm"]
        else:
            combined_score = 0.0

        print(
            f"Evaluation: valid=True, "
            f"mse={scores['mse_raw']:.6f} (norm={scores['mse_norm']:.4f}), "
            f"poisson={scores['poisson_raw']:.6f} (norm={scores['poisson_norm']:.4f}), "
            f"poisson_pass={scores['poisson_pass']}, "
            f"combined={combined_score:.4f}, "
            f"time={eval_time:.2f}s"
        )

        return {
            "mse_raw": scores["mse_raw"],
            "poisson_raw": scores["poisson_raw"],
            "mse_norm": scores["mse_norm"],
            "poisson_norm": scores["poisson_norm"],
            "poisson_pass": scores["poisson_pass"],
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