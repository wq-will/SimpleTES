"""
Evaluator for Lasso Path task.
"""

import numpy as np
import time
import os
import signal
import struct
import hashlib
import subprocess
import tempfile
import traceback
import sys
import pickle
import psutil


# ============================================================================
# CONFIGURATION
# ============================================================================

def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_float(key, default):
    return float(os.environ.get(key, default))

CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 256)
OS_BUFFER_PERCENT    = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)
TIMEOUT_SECONDS      = _env_int("EVALUATOR_TIMEOUT_SECONDS", 600)

# ============================================================================
# TEST PROBLEM SIZES
# ============================================================================

PROBLEM_SIZES = [
    (200,  100,  "gaussian"),   # small baseline
    (200,  500,  "gaussian"),   # medium
    (500,  1000, "gaussian"),   # standard
    (200,  2000, "gaussian"),   # wide (p >> n)
    (50,   3000, "gaussian"),   # extreme p >> n 
    (2000, 500,  "gaussian"),   # large n, moderate p
    (100,  5000, "gaussian"),   # ultrawide
    (1000, 500,  "sparse"),     # sparse design, n > p
    (500,  1000, "sparse"),     # sparse design, n ~ p
    (500,  1000, "dense_sol"),  # Gaussian, dense true solution (many active features)
    (500,  30,   "gaussian"),   # very small p, dense active set
    (1000, 50,   "gaussian"),   # small p, larger n
    (2000, 3000, "sparse"),     # large n and p, sparse design
    (500,  200,  "corr_low"),   # moderate correlation (rho=0.5), moderate p
    (500,  500,  "corr_high"),  # high correlation (rho=0.9), stress convergence
    (200,  1000, "corr_low"),   # wide + correlated
    (1000, 1000, "corr_high"),   # larger correlated
]

N_TIMING_RUNS  = 3
N_PROBLEMS     = len(PROBLEM_SIZES)

N_ALPHAS = 50
EPS      = 1e-2

TOL_OBJECTIVE = 1e-6


# ============================================================================
# EXCEPTIONS
# ============================================================================

class EvaluatorTimeoutError(Exception):
    pass

class MemoryLimitExceededError(Exception):
    pass


def _generate(n, p, seed):
    """iid Gaussian design, sparse true coefficients (p//20 nonzeros)."""
    rng = np.random.RandomState(seed)
    s = max(1, p // 20)
    X = rng.randn(n, p)
    w_true = np.zeros(p)
    idx = rng.choice(p, s, replace=False)
    w_true[idx] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y


def _generate_sparse(n, p, seed):
    """Sparse column design: ~20 nonzeros per row, vectorized."""
    rng = np.random.RandomState(seed)
    nnz = min(20, p)
    # Build row/col indices for all nonzeros at once — no Python loop over rows.
    row_idx = np.repeat(np.arange(n), nnz)
    col_idx = np.array([rng.choice(p, nnz, replace=False) for _ in range(n)]).ravel()
    vals    = rng.randn(n * nnz)
    X = np.zeros((n, p))
    np.add.at(X, (row_idx, col_idx), vals)
    col_std = X.std(axis=0)
    X /= np.where(col_std > 1e-8, col_std, 1.0)
    s = max(1, p // 100)
    w_true = np.zeros(p)
    w_true[rng.choice(p, s, replace=False)] = rng.randn(s) * 2
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y


def _generate_dense_sol(n, p, seed):
    """iid Gaussian design, dense true solution (p//5 nonzeros)."""
    rng = np.random.RandomState(seed)
    s = max(1, p // 5)
    X = rng.randn(n, p)
    w_true = np.zeros(p)
    idx = rng.choice(p, s, replace=False)
    w_true[idx] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y


def _generate_corr(n, p, rho, seed):
    """Toeplitz-correlated Gaussian design: X_ij ~ N(0,1) with cov(j,k)=rho^|j-k|.
    Sparse true solution (p//20 nonzeros). Tests CD convergence under correlation."""
    rng = np.random.RandomState(seed)
    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L   = np.linalg.cholesky(cov)
    X   = rng.randn(n, p) @ L.T
    s   = max(1, p // 20)
    w_true = np.zeros(p)
    w_true[rng.choice(p, s, replace=False)] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y


def _generate_problem(n, p, gen, seed):
    """Dispatch to the right generator based on gen tag."""
    if gen == "gaussian":
        return _generate(n, p, seed)
    elif gen == "sparse":
        return _generate_sparse(n, p, seed)
    elif gen == "dense_sol":
        return _generate_dense_sol(n, p, seed)
    elif gen == "corr_low":
        return _generate_corr(n, p, rho=0.5, seed=seed)
    elif gen == "corr_high":
        return _generate_corr(n, p, rho=0.9, seed=seed)
    else:
        raise ValueError(f"Unknown generator: {gen}")


def _obj(X, y, w, lam):
    n = X.shape[0]
    r = y - X @ w
    return (1.0 / (2.0 * n)) * np.dot(r, r) + lam * np.sum(np.abs(w))


# ============================================================================
# VALIDATION & SCORING
# ============================================================================

def validate_solution(result):
    if result is None:
        return False, "Result is None"
    if not isinstance(result, (list, tuple)):
        return False, f"Expected list of per-problem results, got {type(result)}"
    if len(result) != N_PROBLEMS:
        return False, f"Expected {N_PROBLEMS} results, got {len(result)}"
    for i, entry in enumerate(result):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            return False, f"Problem {i}: expected (valid, sol_ms, max_gap[, err]), got {entry}"
        valid_flag, sol_ms, _ = entry[:3]
        if not np.isfinite(sol_ms):
            err_detail = entry[3] if len(entry) > 3 else "unknown"
            return False, f"Problem {i}: sol_time_ms is not finite. Error: {err_detail}"
    return True, None


def compute_scores(result):
    sol_times  = []
    n_scorable = 0
    problems   = []

    for i, entry in enumerate(result):
        valid_flag, sol_ms, max_gap = entry[:3]
        err_str = entry[3] if len(entry) > 3 else None
        valid   = (valid_flag == 1.0)
        problems.append({
            "problem_idx": i,
            "sol_time_ms": float(sol_ms),
            "max_gap":     float(max_gap),
            "valid":       valid,
            "error":       err_str,
        })
        n_scorable += 1
        if valid:
                sol_times.append(sol_ms)

    if n_scorable > 0 and len(sol_times) == n_scorable:
        geo_mean = float(np.exp(np.mean(np.log(sol_times))))
        score    = 1.0 / geo_mean
    else:
        geo_mean = float("inf")
        score    = 0.0

    return {
        "problems":        problems,
        "n_valid":         len(sol_times),
        "n_scorable":      n_scorable,
        "n_total":         N_PROBLEMS,
        "geo_mean_sol_ms": geo_mean,
        "raw_speed_score": score,
    }


# ============================================================================
# SUBPROCESS EXECUTION
# ============================================================================

def _get_memory_limit_bytes():
    total_mem = psutil.virtual_memory().total
    safe_mem  = total_mem * (1.0 - OS_BUFFER_PERCENT)
    return int(safe_mem / CONCURRENT_PROCESSES)


def run_with_timeout(program_path, timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = TIMEOUT_SECONDS

    limit_bytes = _get_memory_limit_bytes()

    # Generate all seeds upfront: N_TIMING_RUNS timing seeds + 1 correctness
    # seed per problem size — all distinct and randomized per evaluation run.
    eval_rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    all_seeds = {}
    for (n, p, gen) in PROBLEM_SIZES:
        timing_seeds     = [int(eval_rng.randint(1, 2**30)) for _ in range(N_TIMING_RUNS)]
        correctness_seed = int(eval_rng.randint(1, 2**30))
        all_seeds[(n, p, gen)] = {
            "timing":      timing_seeds,
            "correctness": correctness_seed,
        }

    task_dir = os.path.dirname(os.path.abspath(__file__))

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tf:
        tf.write(f'''
import resource, sys, os, pickle, traceback, numpy as np, importlib.util, time
import struct, hashlib, subprocess
from sklearn.linear_model import lasso_path as sk_lasso_path

def _limit_memory():
    try:
        resource.setrlimit(resource.RLIMIT_AS,   ({limit_bytes}, {limit_bytes}))
        resource.setrlimit(resource.RLIMIT_DATA, ({limit_bytes}, {limit_bytes}))
    except ValueError:
        pass

_limit_memory()

TOL_OBJECTIVE  = {TOL_OBJECTIVE}
N_ALPHAS       = {N_ALPHAS}
EPS            = {EPS}
N_TIMING_RUNS  = {N_TIMING_RUNS}
PROBLEM_SIZES  = {PROBLEM_SIZES}
ALL_SEEDS      = {all_seeds}
TASK_DIR       = {repr(task_dir)}

def _generate(n, p, seed):
    rng = np.random.RandomState(seed)
    s = max(1, p // 20)
    X = rng.randn(n, p)
    w_true = np.zeros(p)
    idx = rng.choice(p, s, replace=False)
    w_true[idx] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y

def _generate_sparse(n, p, seed):
    rng = np.random.RandomState(seed)
    nnz = min(20, p)
    row_idx = np.repeat(np.arange(n), nnz)
    col_idx = np.array([rng.choice(p, nnz, replace=False) for _ in range(n)]).ravel()
    vals    = rng.randn(n * nnz)
    X = np.zeros((n, p))
    np.add.at(X, (row_idx, col_idx), vals)
    col_std = X.std(axis=0)
    X /= np.where(col_std > 1e-8, col_std, 1.0)
    s = max(1, p // 100)
    w_true = np.zeros(p)
    w_true[rng.choice(p, s, replace=False)] = rng.randn(s) * 2
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y

def _generate_dense_sol(n, p, seed):
    rng = np.random.RandomState(seed)
    s = max(1, p // 5)
    X = rng.randn(n, p)
    w_true = np.zeros(p)
    idx = rng.choice(p, s, replace=False)
    w_true[idx] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y

def _generate_corr(n, p, rho, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L   = np.linalg.cholesky(cov)
    X   = rng.randn(n, p) @ L.T
    s   = max(1, p // 20)
    w_true = np.zeros(p)
    w_true[rng.choice(p, s, replace=False)] = rng.randn(s)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y

def _generate_problem(n, p, gen, seed):
    if gen == "gaussian":    return _generate(n, p, seed)
    elif gen == "sparse":    return _generate_sparse(n, p, seed)
    elif gen == "dense_sol": return _generate_dense_sol(n, p, seed)
    elif gen == "corr_low":  return _generate_corr(n, p, rho=0.5, seed=seed)
    elif gen == "corr_high": return _generate_corr(n, p, rho=0.9, seed=seed)
    else: raise ValueError(f"Unknown generator: {{gen}}")

def _obj(X, y, w, lam):
    n = X.shape[0]
    r = y - X @ w
    return (1.0 / (2.0 * n)) * np.dot(r, r) + lam * np.sum(np.abs(w))

def _load_cpp_code(path):
    spec = importlib.util.spec_from_file_location("evolved_program", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CPP_CODE"):
        raise RuntimeError("Evolved program must define CPP_CODE.")
    cpp_code    = mod.CPP_CODE
    extra_flags = list(getattr(mod, "COMPILE_FLAGS", []))
    return cpp_code, extra_flags

def _get_binary(cpp_code, extra_flags):
    """Compile CPP_CODE, cache by content hash."""
    code_hash   = hashlib.md5(cpp_code.encode()).hexdigest()[:12]
    cache_dir   = os.path.join({repr(tempfile.gettempdir())}, "lasso_path_cpp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    binary_path = os.path.join(cache_dir, f"lasso_solver_{{code_hash}}")

    if not os.path.exists(binary_path):
        src_path = binary_path + ".cpp"
        with open(src_path, "w") as f:
            f.write(cpp_code)

        eigen_local = os.path.join(TASK_DIR, "eigen")
        eigen_flag  = f"-I{{eigen_local}}" if os.path.isdir(eigen_local) else "-I/usr/include/eigen3"

        base_flags  = ["g++", "-O3", "-march=native", "-std=c++17",
                       eigen_flag, src_path, "-o", binary_path]
        compile_cmd = base_flags[:5] + extra_flags + base_flags[5:]

        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\\n{{result.stderr}}")

    return binary_path

def _run_binary(binary_path, X, y, lambda_path, call_timeout):
    """Run solver binary on a single problem, return coef_path."""
    n, p     = X.shape
    n_lambda = len(lambda_path)
    header   = struct.pack("iii", n, p, n_lambda)
    payload  = (header
                + np.asarray(X, dtype=np.float64, order="C").tobytes()
                + np.asarray(y, dtype=np.float64, order="C").tobytes()
                + np.asarray(lambda_path, dtype=np.float64, order="C").tobytes())

    proc = subprocess.run([binary_path], input=payload, capture_output=True,
                          timeout=call_timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Solver crashed (code {{proc.returncode}}):\\n"
            f"{{proc.stderr.decode(errors='ignore')[:500]}}"
        )
    expected = p * n_lambda * 8
    if len(proc.stdout) != expected:
        raise RuntimeError(f"Output size mismatch: got {{len(proc.stdout)}}, expected {{expected}}")
    return np.frombuffer(proc.stdout, dtype=np.float64).reshape((p, n_lambda), order="F")

# Per-call timeout: divide budget evenly across all calls with a cap of 60s.
# Each problem has N_TIMING_RUNS timing calls + 1 correctness call.
# We also reserve ~30s for compilation and sklearn overhead.
_CALLS_TOTAL   = len(PROBLEM_SIZES) * (N_TIMING_RUNS + 1)
_PROCESS_BUDGET = {timeout_seconds} - 30
CALL_TIMEOUT   = max(5, min(60, int(_PROCESS_BUDGET / _CALLS_TOTAL)))

try:
    cpp_code, extra_flags = _load_cpp_code("{program_path}")
    binary_path = _get_binary(cpp_code, extra_flags)

    results = []
    for (n, p, gen) in PROBLEM_SIZES:

        # ── Synthetic dataset handling ────────────────────────────────────────
        seeds = ALL_SEEDS[(n, p, gen)]

        timing_ms = []
        timing_error = None
        for seed in seeds["timing"]:
            X, y = _generate_problem(n, p, gen, seed)
            alphas_sk, _, _ = sk_lasso_path(X, y, n_alphas=N_ALPHAS, eps=EPS)
            try:
                t0        = time.perf_counter()
                coef_path = _run_binary(binary_path, X, y, alphas_sk, CALL_TIMEOUT)
                elapsed   = (time.perf_counter() - t0) * 1000
            except Exception as e:
                import traceback as _tb
                timing_error = f"{{type(e).__name__}}: {{e}}\\n{{_tb.format_exc()}}"
                break
            timing_ms.append(elapsed)

        if timing_error is not None:
            results.append((0.0, float("inf"), float("inf"), timing_error))
            continue

        min_ms = min(timing_ms)

        c_seed = seeds["correctness"]
        X_c, y_c = _generate_problem(n, p, gen, c_seed)
        alphas_c, coefs_sk_c, _ = sk_lasso_path(X_c, y_c, n_alphas=N_ALPHAS, eps=EPS)

        try:
            coef_path_c = _run_binary(binary_path, X_c, y_c, alphas_c, CALL_TIMEOUT)
        except Exception as e:
            import traceback as _tb
            err_str = f"{{type(e).__name__}}: {{e}}\\n{{_tb.format_exc()}}"
            results.append((0.0, min_ms, float("inf"), err_str))
            continue

        coef_path_c = np.asarray(coef_path_c, dtype=np.float64)
        if coef_path_c.shape != (p, len(alphas_c)):
            results.append((0.0, min_ms, float("inf"),
                            f"Shape mismatch: got {{coef_path_c.shape}}, expected {{(p, len(alphas_c))}}"))
            continue

        max_gap = max(
            _obj(X_c, y_c, coef_path_c[:, k], alphas_c[k])
            - _obj(X_c, y_c, coefs_sk_c[:, k], alphas_c[k])
            for k in range(len(alphas_c))
        )
        valid = (max_gap <= TOL_OBJECTIVE) and np.all(np.isfinite(coef_path_c))
        results.append((1.0 if valid else 0.0, min_ms, float(max_gap)))

    with open("{tf.name}.results", "wb") as f:
        pickle.dump({{"solution": results}}, f)

    # Report summary to stderr for diagnosis
    for i, ((n_i, p_i, gen_i), entry) in enumerate(zip(PROBLEM_SIZES, results)):
        vf = entry[0]
        if vf == 2.0:
            print(f"[eval] problem {{i}} {{gen_i}}: SKIPPED", file=sys.stderr)
        elif vf == 1.0:
            print(f"[eval] problem {{i}} {{gen_i}}: OK  gap={{entry[2]:.2e}}  ms={{entry[1]:.1f}}", file=sys.stderr)
        else:
            err = entry[3] if len(entry) > 3 else "no detail"
            print(f"[eval] problem {{i}} {{gen_i}}: FAIL  gap={{entry[2]:.2e}}  err={{str(err)[:80]}}", file=sys.stderr)

except MemoryError:
    with open("{tf.name}.results", "wb") as f:
        pickle.dump({{"error": "Memory limit exceeded"}}, f)
except Exception as e:
    traceback.print_exc()
    with open("{tf.name}.results", "wb") as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
''')
        temp_path = tf.name

    results_path = temp_path + ".results"

    try:
        child_env = os.environ.copy()
        child_env["OMP_NUM_THREADS"]      = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"]      = "1"
        child_env["NUMEXPR_NUM_THREADS"]  = "1"
        child_env["LASSO_TASK_DIR"] = os.path.dirname(os.path.abspath(__file__))

        process = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            if exit_code in (-9, -11):
                raise MemoryLimitExceededError("Process killed by OS (likely OOM)")

            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        data = pickle.load(f)
                    if "error" in data:
                        err = data["error"]
                        if "Memory" in err:
                            raise MemoryLimitExceededError(err)
                        raise RuntimeError(f"Program execution failed: {err}")
                    return data
                except (pickle.UnpicklingError, EOFError):
                    raise RuntimeError("Failed to read results (truncated/crashed)")
            else:
                if exit_code != 0:
                    raise RuntimeError(
                        f"Process exited {exit_code}. Stderr: {stderr.decode()[:2000]}"
                    )
                raise RuntimeError("Results file not found but process exited 0")

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(f"Timed out after {timeout_seconds}s")

    finally:
        for path in (temp_path, results_path):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ============================================================================
# MAIN EVALUATE FUNCTION
# ============================================================================

def _make_error_result(error_msg, eval_time=0.0):
    return {
        "n_valid":         0,
        "n_total":         N_PROBLEMS,
        "geo_mean_sol_ms": float("inf"),
        "raw_speed_score": 0.0,
        "validity":        0.0,
        "eval_time":       float(eval_time),
        "combined_score":  0.0,
        "error":           error_msg,
    }


def evaluate(program_path):
    """
    Evaluate a lasso path solver.

    The program must define:
        CPP_CODE : str   — C++ source for the solver binary
        COMPILE_FLAGS : list[str]  (optional) — extra compiler flags e.g. ["-fopenmp"]

    The C++ binary must read the binary wire format from stdin and write
    coef_path (column-major float64) to stdout.

    Timing: N_TIMING_RUNS runs per problem size, each with a different random
    seed, minimum time reported. Caching across calls cannot inflate the score
    because each call receives different data.

    Correctness: checked on a separate fresh problem not used for timing.

    Returns:
        dict with combined_score = 1 / geo_mean(solve_time_ms)
    """
    try:
        t0  = time.time()
        res = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)
        eval_time = time.time() - t0

        solution = res.get("solution")
        is_valid, err_msg = validate_solution(solution)

        if not is_valid:
            print(f"Validation failed: {err_msg}")
            r = _make_error_result(err_msg, eval_time)
            r["validity"] = 0.0
            return r

        scores = compute_scores(solution)
        combined_score = scores["raw_speed_score"]

        print(f"\n{'Problem':<25} {'Sol(ms)':>10} {'MaxGap':>12} {'Valid':>6}")
        print("-" * 57)
        size_labels = [
            f"n{n}_p{p}_{gen}"
            for (n, p, gen) in PROBLEM_SIZES
        ]
        for i, prob in enumerate(scores["problems"]):
            label  = size_labels[i] if i < len(size_labels) else f"prob_{i+1}"
            status  = "OK" if prob["valid"] else "FAIL"
            sol_str = f"{prob['sol_time_ms']:>10.2f}" if np.isfinite(prob['sol_time_ms']) else "     ERROR"
            print(
                f"{label:<25} "
                f"{sol_str} "
                f"{prob['max_gap']:>12.2e} "
                f"{status:>6}"
            )
            if prob.get("error"):
                first_line = prob["error"].split("\n")[0]
                print(f"  ↳ {first_line}")
        print("-" * 57)
        n_sc = scores.get("n_scorable", scores["n_total"])
        print(
            f"Valid: {scores['n_valid']}/{n_sc} (scorable), "
            f"GeoMean: {scores['geo_mean_sol_ms']:.2f}ms, "
            f"Score: {combined_score:.6f}, "
            f"EvalTime: {eval_time:.1f}s"
        )

        return {
            "problems":        scores["problems"],
            "n_valid":         scores["n_valid"],
            "n_total":         scores["n_total"],
            "geo_mean_sol_ms": scores["geo_mean_sol_ms"],
            "raw_speed_score": scores["raw_speed_score"],
            "validity":        1.0,
            "eval_time":       float(eval_time),
            "combined_score":  float(combined_score),
        }

    except MemoryLimitExceededError as e:
        print(f"Memory limit exceeded: {e}")
        return _make_error_result(f"Memory limit exceeded: {e}")

    except EvaluatorTimeoutError as e:
        print(f"Evaluation timed out: {e}")
        return _make_error_result(f"Timeout: {e}")

    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return _make_error_result(f"{type(e).__name__}: {e}")