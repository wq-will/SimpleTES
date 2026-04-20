"""
Evaluator for AHC039 (Purse Seine Fishing) — SimpleTES task.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROBLEM_ID = "ahc039"
NUM_CASES = 150
TIME_LIMIT = 2.0
NORM_FACTOR = 1500.0
N_EVAL_RUNS = 3
AGGREGATION = "mean"

# Docker image
DOCKER_IMAGE = os.environ.get("AHC_DOCKER_IMAGE", "yimjk/ale-bench:cpp20-202301")

# Paths on the HOST — these get mounted into the container
# Cache directory containing public_inputs_150/ and tester_binaries/
CACHE_DIR = Path(os.environ.get(
    "AHC_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / "cache")
))

RUNNER_SCRIPT = Path(os.environ.get(
    "AHC_RUNNER_SCRIPT",
    str(Path(__file__).resolve().parent.parent / "docker_runner.py")
))

# Parallelism inside the container
NUM_CASE_WORKERS = int(os.environ.get("AHC_CASE_WORKERS", 12))
DOCKER_TIMEOUT = int(os.environ.get("AHC_DOCKER_TIMEOUT", 180))
_PIN_CORES = os.environ.get("AHC_PIN_CORES", "1") == "1"
_WORKER_ID = int(os.environ.get("SIMPLETES_WORKER_ID", "0"))


# ============================================================================
# MAIN EVALUATE FUNCTION
# ============================================================================

def evaluate(program_path):
    """
    Evaluate a C++ program for AHC039.

    SimpleTES calls this with the path to a Python file whose EVOLVE-BLOCK
    contains CPP_CODE = '''...'''. We extract the C++ code, write it to a temp
    file, and invoke Docker to compile + run + score all 150 test cases.

    The evaluation is repeated N_EVAL_RUNS times. The final combined_score is
    either the mean or max across runs, according to AGGREGATION.

    Args:
        program_path: Path to the Python program file

    Returns:
        dict with at least 'combined_score'
    """
    overall_start = time.time()

    try:
        # --- Validate paths ---
        input_dir = CACHE_DIR / "public_inputs_150" / f"{PROBLEM_ID}_inputs"
        tester_bin = CACHE_DIR / "tester_binaries" / f"{PROBLEM_ID}_tester"

        if not input_dir.exists():
            return _error(f"Input cache not found: {input_dir}. Run get_cache.sh first.")
        if not tester_bin.exists():
            return _error(f"Tester not found: {tester_bin}. Run get_cache.sh first.")
        if not RUNNER_SCRIPT.exists():
            return _error(f"docker_runner.py not found: {RUNNER_SCRIPT}")

        # --- Extract C++ code ---
        cpp_code = _extract_cpp_code(program_path)
        if cpp_code is None:
            return _error("Failed to extract C++ code from program file.")

        # --- Write C++ to temp dir (reused across all runs) ---
        tmp_dir = Path(tempfile.mkdtemp(prefix="ahc039_eval_"))
        try:
            cpp_file = tmp_dir / "Main.cpp"
            cpp_file.write_text(cpp_code)

            # Build the docker command once (same across all runs)
            docker_cmd = [
                "docker", "run", "--rm",
                "--network=none",
            ]

            if _PIN_CORES:
                start_core = _WORKER_ID * NUM_CASE_WORKERS
                end_core   = start_core + NUM_CASE_WORKERS - 1
                docker_cmd += ["--cpuset-cpus", f"{start_core}-{end_core}"]

            docker_cmd += [
                "-v", f"{tmp_dir}:/work:ro",
                "-v", f"{CACHE_DIR}:/cache:ro",
                "-v", f"{RUNNER_SCRIPT}:/runner.py:ro",
                DOCKER_IMAGE,
                "python3", "/runner.py",
                "/work/Main.cpp",
                f"/cache/public_inputs_150/{PROBLEM_ID}_inputs",
                f"/cache/tester_binaries/{PROBLEM_ID}_tester",
                str(NUM_CASES),
                str(NUM_CASE_WORKERS),
                str(TIME_LIMIT),
            ]

            # --- Run N_EVAL_RUNS times ---
            run_results = []
            for run_idx in range(N_EVAL_RUNS):
                run_result = _single_run(docker_cmd, run_idx, time.time() - overall_start)
                if run_result is None:
                    # Hard failure (Docker crash, timeout, etc.) — propagate immediately
                    return _error(f"Run {run_idx + 1}/{N_EVAL_RUNS} failed fatally.")
                run_results.append(run_result)
                print(
                    f"[Run {run_idx + 1}/{N_EVAL_RUNS}] "
                    f"AC: {run_result['num_accepted']}/{NUM_CASES}. "
                    f"Total: {run_result['total_score']}. "
                    f"Combined: {run_result['combined_score']:.8f}."
                )

            # --- Aggregate across runs ---
            combined_scores = [r["combined_score"] for r in run_results]
            total_scores    = [r["total_score"]    for r in run_results]
            raw_scores      = [r["raw_score"]      for r in run_results]

            if AGGREGATION == "max":
                best_idx       = combined_scores.index(max(combined_scores))
                final_combined = combined_scores[best_idx]
                final_total    = total_scores[best_idx]
                final_raw      = raw_scores[best_idx]
            else:
                final_combined = sum(combined_scores) / len(combined_scores)
                final_total    = sum(total_scores)    / len(total_scores)
                final_raw      = sum(raw_scores)      / len(raw_scores)

            eval_time = time.time() - overall_start
            print(
                f"[{AGGREGATION.upper()} over {N_EVAL_RUNS} run(s)] "
                f"Total: {final_total:.1f}. "
                f"Combined: {final_combined:.8f}. "
                f"Time: {eval_time:.1f}s"
            )

            # Use validity and error from last run for reporting
            last = run_results[-1]
            result = {
                "combined_score": float(final_combined),
                "raw_score":      float(final_raw),
                "total_score":    int(round(final_total)),
                "num_accepted":   last["num_accepted"],
                "num_cases":      NUM_CASES,
                "validity":       last["validity"],
                "eval_time":      float(eval_time),
                "n_eval_runs":    N_EVAL_RUNS,
                "aggregation":    AGGREGATION,
            }
            if last.get("error"):
                result["error"] = last["error"]
            return result

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        traceback.print_exc()
        return _error(f"{type(e).__name__}: {e}")


# ============================================================================
# SINGLE RUN HELPER
# ============================================================================

def _single_run(docker_cmd, run_idx, elapsed_so_far):
    """
    Execute one full Docker evaluation run.

    Returns a dict with run-level scores, or None on a fatal error.
    """
    try:
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"ERROR: Docker timed out on run {run_idx + 1} after {DOCKER_TIMEOUT}s")
            return None

        if result.returncode != 0:
            stderr = result.stderr[:500] if result.stderr else ""
            print(f"ERROR: Docker exited {result.returncode} on run {run_idx + 1}: {stderr}")
            return None

        stdout = result.stdout.strip()
        if not stdout:
            print(f"ERROR: Docker produced no output on run {run_idx + 1}. stderr: {result.stderr[:300]}")
            return None

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON from Docker on run {run_idx + 1}: {e}. Output: {stdout[:200]}")
            return None

        if "error" in data:
            print(f"ERROR: Runner error on run {run_idx + 1}: {data['error']}")
            return None

        case_results  = data["case_results"]
        num_accepted  = sum(1 for r in case_results if r["judge"] == "AC")
        total_score   = sum(r["score"] for r in case_results if r["judge"] == "AC")
        any_failed    = num_accepted < NUM_CASES
        raw_score     = total_score / NUM_CASES
        combined_score = (raw_score / NORM_FACTOR) if not any_failed else 0.0

        failures = [r for r in case_results if r["judge"] != "AC"]
        fail_summary = ""
        if failures:
            from collections import Counter
            f = failures[0]
            fail_summary = f"First fail: case {f['case_idx']} [{f['judge']}]: {f['msg'][:120]}"
            fail_counts  = Counter(r["judge"] for r in failures)
            fail_summary += f" | Failures: {dict(fail_counts)}"

        run_result = {
            "combined_score": float(combined_score),
            "raw_score":      float(raw_score),
            "total_score":    int(total_score),
            "num_accepted":   num_accepted,
            "validity":       1.0 if not any_failed else 0.0,
        }
        if fail_summary:
            run_result["error"] = fail_summary
        return run_result

    except Exception as e:
        print(f"ERROR: Exception during run {run_idx + 1}: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# HELPERS
# ============================================================================

def _extract_cpp_code(program_path):
    """
    Extract CPP_CODE from the program file by exec'ing it.
    
    The LLM writes C++ escape sequences like \\n inside Python triple-quoted
    strings. When exec'd, Python interprets \\n as the two-character sequence
    backslash-n, which is exactly what the C++ compiler needs to see.
    
    For example, the LLM writes:  CPP_CODE = '''cout << "\\n";'''
    After exec(), CPP_CODE = 'cout << "\\n";'  (backslash + n)
    Written to .cpp file: cout << "\n";  — correct C++ escape sequence.
    """
    try:
        prog_globals = {}
        with open(program_path, "r") as f:
            exec(f.read(), prog_globals)
        cpp_code = prog_globals.get("CPP_CODE")
        if cpp_code and isinstance(cpp_code, str) and len(cpp_code.strip()) > 0:
            return cpp_code.strip()
        return None
    except Exception as e:
        print(f"Failed to extract C++ code: {e}")
        traceback.print_exc()
        return None


def _error(msg, eval_time=0.0):
    """Standardized error result."""
    print(f"ERROR: {msg}")
    return {
        "combined_score": 0.0,
        "raw_score":      0.0,
        "total_score":    0,
        "num_accepted":   0,
        "num_cases":      NUM_CASES,
        "validity":       0.0,
        "eval_time":      float(eval_time),
        "error":          msg,
    }