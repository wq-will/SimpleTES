"""Evaluate ZAPBench forecasting candidates."""

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import psutil

# SimpleTES is optional on worker environments.
try:
    from simpletes.construction import capture_construction_if_requested
except ImportError:
    def capture_construction_if_requested(_Y_pred):
        return None


# ============================================================================
# CONFIGURATION (overridable via env)
# ============================================================================

def _env_int(key, default):
    return int(os.environ.get(key, default))


def _env_float(key, default):
    return float(os.environ.get(key, default))


CONCURRENT_PROCESSES = _env_int("EVALUATOR_CONCURRENT_PROCESSES", 1)
OS_BUFFER_PERCENT = _env_float("EVALUATOR_OS_BUFFER_PERCENT", 0.10)
TIMEOUT_SECONDS = _env_int("EVALUATOR_TIMEOUT_SECONDS", 9000)  # 2.5 hr / candidate

# CUDA_VISIBLE_DEVICES handling:
# - Under SLURM, leave it alone — SLURM has already pinned the worker to its
#   allocated GPU(s), and `os.environ.copy()` propagates that to the child.

SANDBOX_CUDA_DEVICES = os.environ.get("ZAPBENCH_SANDBOX_CUDA")

# Where setup.sh stages the precomputed inputs.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zapbench_cache"
)

_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
_TRAIN_LENGTHS_PATH = os.path.join(DATA_DIR, "X_train_condition_lengths.npy")
_VAL_INPUTS_PATH = os.path.join(DATA_DIR, "val_predict_inputs.npy")
_VAL_TARGETS_PATH = os.path.join(DATA_DIR, "val_targets.npy")
_VAL_BASELINE_PATH = os.path.join(DATA_DIR, "mean_baseline_mae_val.json")


# ============================================================================
# EXCEPTIONS
# ============================================================================

class EvaluatorTimeoutError(Exception):
    pass


class MemoryLimitExceededError(Exception):
    pass


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_pred(Y_pred, expected_shape):
    if Y_pred is None:
        return False, "Y_pred is None"
    if not isinstance(Y_pred, np.ndarray):
        return False, f"Y_pred must be np.ndarray, got {type(Y_pred).__name__}"
    if Y_pred.shape != expected_shape:
        return False, f"Y_pred shape {Y_pred.shape} != expected {expected_shape}"
    if not np.isfinite(Y_pred).all():
        return False, "Y_pred contains non-finite values"
    return True, None


# ============================================================================
# SCORING
# ============================================================================

def _mae(Y_pred, Y_true):
    return float(np.mean(np.abs(Y_pred.astype(np.float64) - Y_true.astype(np.float64))))


def _combined_score(mae_candidate):
    """Negated raw MAE; higher is better. ERA-comparable.

    Unlike the previous `max(0, 1 - MAE/baseline)` skill score, this preserves
    the full ordering of below-baseline candidates (the `max(0, ...)` clamp
    used to collapse them all to 0.0) and removes the dependency on the
    cached `mae_mean_baseline` for the search signal itself.
    """
    return -float(mae_candidate)


# ============================================================================
# SUBPROCESS EXECUTION
# ============================================================================

def _get_memory_limit_bytes():
    # Two cases to handle distinctly:
    #
    # (a) Running inside a SLURM worker (SLURM_MEM_PER_NODE set) — the
    #     cgroup is sized for ONE eval process. EVALUATOR_CONCURRENT_PROCESSES
    #     (which defends against many parallel evals sharing host RAM in
    #     LOCAL-fallback mode) does not apply here; if we divide the worker's
    #     cgroup by 14, we end up with ~6 GB per process and small numpy
    #     allocations fail at RLIMIT_AS. Force concurrent=1 for workers.
    #
    # (b) Running in LOCAL-fallback mode (no SLURM cgroup) — the host's
    #     full RAM is shared among CONCURRENT_PROCESSES parallel evals.
    #     Divide accordingly.
    slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem_mb:
        try:
            total_mem = int(slurm_mem_mb) * 1024 * 1024
            concurrent = 1     # one eval per worker cgroup
        except ValueError:
            total_mem = psutil.virtual_memory().total
            concurrent = CONCURRENT_PROCESSES
    else:
        total_mem = psutil.virtual_memory().total
        concurrent = CONCURRENT_PROCESSES
    safe_mem = total_mem * (1.0 - OS_BUFFER_PERCENT)
    return int(safe_mem / concurrent)


def run_with_timeout(program_path, timeout_seconds=None):
    """Run the candidate program in a subprocess.

    Returns the subprocess's Y_pred as a numpy array, loaded by the outer
    process after the subprocess exits. Y_true is never touched here.
    """
    if timeout_seconds is None:
        timeout_seconds = TIMEOUT_SECONDS

    limit_bytes = _get_memory_limit_bytes()

    # Resolve absolute paths in the parent — the subprocess may run with a
    # different cwd or os module shim.
    train_path = repr(os.path.abspath(_TRAIN_PATH))
    train_lengths_path = repr(os.path.abspath(_TRAIN_LENGTHS_PATH))
    val_inputs_path = repr(os.path.abspath(_VAL_INPUTS_PATH))

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # The subprocess receives X_train + predict_inputs only; Y_true stays
        # in the outer process.
        script = f"""
import resource, sys

# --- diagnostic: log inherited + applied memory limits to stderr so the
# worker log shows what actually happened. Cheap to print, invaluable to
# debug the "MemoryError caught" mystery.
_inh_as_soft, _inh_as_hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"[wrapper] computed limit_bytes = {limit_bytes / 1e9:.2f} GiB", file=sys.stderr, flush=True)
print(f"[wrapper] inherited RLIMIT_AS soft={{_inh_as_soft}} hard={{_inh_as_hard}}", file=sys.stderr, flush=True)
print(f"[wrapper] SLURM_MEM_PER_NODE = {{repr(__import__('os').environ.get('SLURM_MEM_PER_NODE'))}}", file=sys.stderr, flush=True)

def _limit_memory():
    try:
        soft, hard = {limit_bytes}, {limit_bytes}
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
    except ValueError as _e:
        print(f"[wrapper] setrlimit RLIMIT_AS to {{soft}} FAILED: {{_e}}", file=sys.stderr, flush=True)

_limit_memory()

_post_as_soft, _post_as_hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"[wrapper] post-setrlimit RLIMIT_AS soft={{_post_as_soft}} hard={{_post_as_hard}}", file=sys.stderr, flush=True)

import os, pickle, traceback, importlib.util
print(f"[wrapper] stdlib imported, about to import numpy", file=sys.stderr, flush=True)
import numpy as np
print(f"[wrapper] numpy {{np.__version__}} imported OK", file=sys.stderr, flush=True)

sys.path.insert(0, os.path.dirname('{program_path}'))


def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    mod = _load('{program_path}')

    if not hasattr(mod, 'forecast') or not callable(getattr(mod, 'forecast')):
        raise RuntimeError('Program must define forecast().')

    X_train = np.load({train_path})
    X_train_condition_lengths = np.load({train_lengths_path})
    predict_inputs = np.load({val_inputs_path})

    Y_pred = mod.forecast(
        X_train,
        predict_inputs,
        X_train_condition_lengths=X_train_condition_lengths,
        random_state=42,
        time_budget_s={timeout_seconds},
        num_train_conditions=int(X_train_condition_lengths.shape[0]),
    )

    if not isinstance(Y_pred, np.ndarray):
        Y_pred = np.asarray(Y_pred)
    Y_pred = Y_pred.astype(np.float32)

    if not np.isfinite(Y_pred).all():
        raise ValueError('Y_pred contains non-finite values')

    with open('{temp_file.name}.pred', 'wb') as f:
        pickle.dump(Y_pred, f)

except MemoryError as _me:
    # Print the traceback so we can see WHERE the MemoryError fired
    # (import time vs Nlinear fit vs forecast vs Y_pred allocation).
    traceback.print_exc()
    _fail_as_soft, _fail_as_hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f"[wrapper] MemoryError at RLIMIT_AS soft={{_fail_as_soft}} hard={{_fail_as_hard}}", file=sys.stderr, flush=True)
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": f"Memory limit exceeded (MemoryError caught): {{_me}}"}}, f)
except Exception as e:
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"
    pred_path = f"{temp_file_path}.pred"

    try:
        child_env = os.environ.copy()
        # Only override CUDA_VISIBLE_DEVICES when ZAPBENCH_SANDBOX_CUDA was
        # explicitly set (single-host fallback). Under SLURM, inherit the
        # allocation set by the scheduler.
        if SANDBOX_CUDA_DEVICES is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = SANDBOX_CUDA_DEVICES
        # Avoid JAX preallocating the full GPU; lets two candidates coexist if
        # eval_concurrency > 1.
        child_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        child_env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
        # Same idea for PyTorch.
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Numeric library threading: match the SLURM CPU allocation
        # (ZAPBENCH_WORKER_CPUS -> --cpus-per-task -> SLURM_CPUS_PER_TASK) so
        # each candidate uses all its allocated cores. Falls back to 4 when
        # off-SLURM (local/smoke) to avoid oversubscribing a shared driver.
        _eval_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
        child_env["OMP_NUM_THREADS"] = _eval_threads
        child_env["OPENBLAS_NUM_THREADS"] = _eval_threads
        child_env["MKL_NUM_THREADS"] = _eval_threads
        child_env["NUMEXPR_NUM_THREADS"] = _eval_threads

        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            if exit_code in (-9, -11):
                raise MemoryLimitExceededError(
                    f"Process killed by OS (likely OOM). Limit was {limit_bytes / (1024 ** 3):.2f} GB"
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
                except (pickle.UnpicklingError, EOFError):
                    raise RuntimeError(
                        "Failed to read results file (possibly truncated due to crash)."
                    )

            if not os.path.exists(pred_path):
                if exit_code != 0:
                    err_out = stderr.decode(errors="replace")
                    if "MemoryError" in err_out:
                        raise MemoryLimitExceededError("Memory limit exceeded (stderr)")
                    raise RuntimeError(
                        f"Process exited with code {exit_code}. Stderr: {err_out[-2000:]}"
                    )
                raise RuntimeError("Y_pred output not found but process exited 0.")

            try:
                with open(pred_path, "rb") as f:
                    Y_pred = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                raise RuntimeError("Failed to read Y_pred output (possibly truncated).")

            return Y_pred

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(
                f"Process timed out after {timeout_seconds} seconds"
            )

    finally:
        for p in (temp_file_path, results_path, pred_path):
            if os.path.exists(p):
                os.unlink(p)


# ============================================================================
# MAIN EVALUATE
# ============================================================================

# Failure-sentinel score: must be (a) finite (SimpleTES's
# validate_node_for_db rejects nan/inf), and (b) below any plausible real
# combined_score. Real MAE is bounded by the data range (~1.5), so real
# combined_score = -MAE is in roughly [-1.5, 0]. -10.0 sits 6× below the
# worst-case real score — clearly a failure marker, finite, JSON-safe.
_FAILURE_SCORE = -10.0


def _make_error_result(error_msg, eval_time=0.0):
    return {
        "mae_raw": float("inf"),
        "mae_mean_baseline": 0.0,
        "validity": 0.0,
        "eval_time": float(eval_time),
        "combined_score": _FAILURE_SCORE,
        "error": error_msg,
    }


def _evaluate_local(program_path):
    """Run the candidate locally on this node (subprocess-isolated) and score.

    Returns the same dict shape evaluate() does — see evaluate() docstring.
    """
    try:
        start = time.time()
        Y_pred = run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS)
        eval_time = time.time() - start

        Y_true = np.load(_VAL_TARGETS_PATH)
        expected_shape = Y_true.shape  # (n_windows, 32, 71721)

        ok, err = _validate_pred(Y_pred, expected_shape)
        if not ok:
            return _make_error_result(err, eval_time)

        with open(_VAL_BASELINE_PATH) as f:
            mae_mean_baseline = float(json.load(f)["mae"])

        mae_candidate = _mae(Y_pred, Y_true)
        score = _combined_score(mae_candidate)

        capture_construction_if_requested(Y_pred)

        print(
            f"Evaluation: valid=True, mae={mae_candidate:.6f} "
            f"(baseline_mae={mae_mean_baseline:.6f}), "
            f"combined={score:.4f}, time={eval_time:.1f}s"
        )

        return {
            "mae_raw": mae_candidate,
            "mae_mean_baseline": mae_mean_baseline,
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(score),
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


# ============================================================================
# SLURM DISPATCH (Path A)
# ============================================================================

def _evaluate_via_slurm(program_path):
    """Dispatch a single candidate to a remote SLURM worker via `sbatch --wait`.

    Env-driven configuration:
        ZAPBENCH_STAGE_DIR        NFS-shared dir for per-eval result + log
                                  files. Default: ~/zap_dispatch.
        ZAPBENCH_WORKER_WRAPPER   sbatch wrapper script. Default: the repo's
                                  scripts/slurm/zapbench_eval_worker.sh.
        ZAPBENCH_WORKER_PARTITION SLURM partition to submit to.
        ZAPBENCH_WORKER_ACCOUNT   --account value. Required on clusters that
                                  enforce account attribution. Find yours with:
                                    sacctmgr show user $USER -s format=user,account,defaultaccount,partition
                                  Omitted from sbatch if unset.
        ZAPBENCH_WORKER_QOS       --qos value. Optional; omitted if unset.
        ZAPBENCH_WORKER_GRES      --gres value. Default: gpu:1. Restrict to a
                                  specific GPU model with e.g. gpu:<model>:1.
        ZAPBENCH_WORKER_EXCLUDE   --exclude node list, in SLURM range syntax,
                                  to keep candidates off unsuitable nodes.
                                  Omitted if unset.
        ZAPBENCH_WORKER_CONSTRAINT  --constraint value (feature expression).
                                  Omitted if unset.
        ZAPBENCH_WORKER_CPUS      --cpus-per-task. Default: 4.
        ZAPBENCH_WORKER_MEM       --mem. Default: 24G.
        ZAPBENCH_WORKER_TIME      --time. Default: 04:30:00.
        ZAPBENCH_KEEP_RESULTS     If "1", keep result/log files after read
                                  (default: clean up).
    """
    stage_dir = Path(os.environ.get("ZAPBENCH_STAGE_DIR", "~/zap_dispatch")).expanduser()
    stage_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    default_wrapper = repo_root / "scripts" / "slurm" / "zapbench_eval_worker.sh"
    wrapper = Path(os.environ.get("ZAPBENCH_WORKER_WRAPPER", default_wrapper)).resolve()
    if not wrapper.exists():
        return _make_error_result(f"Worker wrapper not found: {wrapper}")

    job_id = uuid.uuid4().hex[:12]
    result_path = stage_dir / f"result_{job_id}.json"
    log_path = stage_dir / f"worker_{job_id}.out"
    staged_program = stage_dir / f"program_{job_id}.py"

    # SimpleTES writes each candidate to a driver-local /tmp/user/$UID/eval_*/
    # dir. SLURM workers run on different nodes and can't see that path. Copy
    # the program to a shared filesystem (stage_dir, which must be visible from
    # the worker nodes) so the worker can read it.
    try:
        shutil.copyfile(os.path.abspath(program_path), staged_program)
    except (OSError, IOError) as e:
        return _make_error_result(
            f"Failed to stage program to shared FS: {type(e).__name__}: {e}"
        )

    # Pass repo root explicitly — sbatch copies the script to a spool dir, so
    # BASH_SOURCE[0]-based discovery inside the wrapper resolves to the wrong
    # place. --export=ALL keeps the user's submit env (PATH, venv, etc.) AND
    # sets our marker.
    sbatch_cmd = [
        "sbatch", "--wait",
        f"--export=ALL,ZAPBENCH_REPO_ROOT={repo_root}",
        "--partition", os.environ.get("ZAPBENCH_WORKER_PARTITION", "atlas"),
        "--gres", os.environ.get("ZAPBENCH_WORKER_GRES", "gpu:1"),
        "--cpus-per-task", os.environ.get("ZAPBENCH_WORKER_CPUS", "4"),
        "--mem", os.environ.get("ZAPBENCH_WORKER_MEM", "24G"),
        "--time", os.environ.get("ZAPBENCH_WORKER_TIME", "04:30:00"),
        "--output", str(log_path),
        "--job-name", f"zap-eval-{job_id[:6]}",
    ]
    account = os.environ.get("ZAPBENCH_WORKER_ACCOUNT")
    if account:
        sbatch_cmd += ["--account", account]
    qos = os.environ.get("ZAPBENCH_WORKER_QOS")
    if qos:
        sbatch_cmd += ["--qos", qos]
    exclude = os.environ.get("ZAPBENCH_WORKER_EXCLUDE")
    if exclude:
        sbatch_cmd += ["--exclude", exclude]
    constraint = os.environ.get("ZAPBENCH_WORKER_CONSTRAINT")
    if constraint:
        sbatch_cmd += ["--constraint", constraint]
    sbatch_cmd += [
        str(wrapper),
        str(staged_program),
        str(result_path),
    ]

    keep = os.environ.get("ZAPBENCH_KEEP_RESULTS") == "1"
    start = time.time()

    sbatch_rc = None
    sbatch_err = ""
    try:
        proc = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        sbatch_rc = proc.returncode
        sbatch_err = proc.stderr or ""
    except Exception as e:
        return _make_error_result(
            f"sbatch invocation failed: {type(e).__name__}: {e}",
            time.time() - start,
        )

    wall = time.time() - start

    # Result file is the authoritative success signal — it may exist even when
    # sbatch returns non-zero (worker wrote it then exited badly later).
    if result_path.exists():
        try:
            with open(result_path) as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as e:
            return _make_error_result(
                f"Result file present but unreadable: {type(e).__name__}: {e}",
                wall,
            )
        finally:
            if not keep:
                for p in (result_path, log_path, staged_program):
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass

    log_tail = ""
    if log_path.exists():
        try:
            log_tail = log_path.read_text(errors="replace")[-2000:]
        except OSError:
            pass
        if not keep:
            for p in (log_path, staged_program):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    msg = f"sbatch rc={sbatch_rc}; no result file."
    if sbatch_err.strip():
        msg += f" stderr: {sbatch_err.strip()[:400]}"
    if log_tail:
        msg += f" log_tail: {log_tail}"
    return _make_error_result(msg[:2400], wall)


def evaluate(program_path):
    """Run the candidate program and score it against the inner-loop val split.

    Dispatches to a remote SLURM worker when ZAPBENCH_DISPATCH_MODE=slurm is
    set; otherwise runs locally (subprocess-isolated on this host). Both
    paths return the same dict shape:

        mae_raw: float, candidate's MAE on val targets
        mae_mean_baseline: float, cached mean-baseline MAE on the same split
        combined_score: -MAE_candidate (higher better; -inf on failure)
        validity: 1.0 if valid, 0.0 otherwise
        eval_time: wall-clock seconds spent inside the candidate subprocess
        error: optional, present when validity==0
    """
    if os.environ.get("ZAPBENCH_DISPATCH_MODE") == "slurm":
        return _evaluate_via_slurm(program_path)
    return _evaluate_local(program_path)


# ============================================================================
# WORKER CLI (invoked by the sbatch wrapper on a remote node)
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "ZAPBench evaluator CLI. Two modes:\n"
            "  (default)  — runs the candidate LOCALLY on this node and writes\n"
            "               the result JSON. This is the worker mode invoked\n"
            "               by the sbatch wrapper; defensively pops\n"
            "               ZAPBENCH_DISPATCH_MODE to prevent recursion.\n"
            "  --via-dispatcher — calls evaluate(), which honors\n"
            "               ZAPBENCH_DISPATCH_MODE. Use for diagnostic tests\n"
            "               of the SLURM-dispatch pipeline (e.g., from a\n"
            "               login node or any non-worker shell)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("program_path", help="Path to candidate program (.py)")
    parser.add_argument("result_path", help="Path to write result JSON")
    parser.add_argument(
        "--via-dispatcher", action="store_true",
        help="Route through evaluate() so ZAPBENCH_DISPATCH_MODE=slurm "
             "actually dispatches via sbatch. Diagnostic use only.",
    )
    args = parser.parse_args()

    if args.via_dispatcher:
        result = evaluate(args.program_path)
    else:
        # Worker mode — defensively pop DISPATCH_MODE so the worker never
        # recurses into sbatch on itself.
        os.environ.pop("ZAPBENCH_DISPATCH_MODE", None)
        result = _evaluate_local(args.program_path)

    tmp = args.result_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, args.result_path)
