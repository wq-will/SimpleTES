"""Shared SimpleTES evaluator for the five impulsive MGA mission tasks.

Each task-level ``evaluator.py`` passes one explicit ``instance.json`` path to
``evaluate_instance``.  This keeps mission selection deterministic even when
candidate programs are executed from temporary directories.

Node format (unified across all types):
    {type, time, planet_id, r, v_before, v_after}
"""

import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from trajectory_schema import (
    NODE_START,
    NODE_END,
    NODE_GA,
    NODE_DSM,
    SchemaError,
    validate_schema,
)
from problem_config import (
    MU,
    RADIUS,
    build_problem,
    load_instance,
    default_flyby_altitude,
)
from tools_wrapper import Tools

# Construction capture is optional in task-local evaluator environments. Those
# environments intentionally install only the scientific dependencies needed by
# the task, not the full SimpleTES application dependency set.
try:
    from simpletes.construction import capture_construction_if_requested
except ImportError:

    def capture_construction_if_requested(_trajectory):
        return None


# ------------------------------------------------------------------ config

INSTANCE_MEMORY_LIMIT_BYTES = 10 * 1024 ** 3
DAY = 86400.0  # s

_EVAL_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EVAL_DIR, "..", ".."))

SAVE_TRAJECTORIES = os.environ.get("IMPULSE_MGA_SAVE_TRAJECTORIES", "") == "1"
_RAW_TRAJ_SAVE_DIR = os.environ.get(
    "IMPULSE_MGA_TRAJ_SAVE_DIR",
    os.path.join(tempfile.gettempdir(), "impulse_mga_trajectories"),
)
TRAJ_SAVE_DIR = os.path.abspath(_RAW_TRAJ_SAVE_DIR)

PROBLEM_BASE = {"planet_mu": dict(MU), "planet_radius": dict(RADIUS)}
TOOLS = Tools()

DEFAULT_TOLS = {
    "planet_pos_tol_km": 1e4,
    "segment_pos_tol_km": 1e4,
    "segment_vel_tol_kms": 1e-2,
    "vinf_match_tol_kms": 1e-3,
    "exact_time_tol_days": 0.01,
}

_PLANET_NAME: dict[str, str] = {
    "1": "Mercury", "2": "Venus", "3": "Earth", "4": "Mars",
    "5": "Jupiter", "6": "Saturn", "7": "Uranus", "8": "Neptune",
}


# ------------------------------------------------------------------ exceptions

class _Infeasible(Exception):
    def __init__(self, reason, failed_node=None):
        super().__init__(reason)
        self.reason = reason
        self.failed_node = failed_node


class EvaluatorTimeoutError(Exception):
    pass


class MemoryLimitExceededError(Exception):
    pass


# ------------------------------------------------------------------ unified state helpers

def _departure_state(node):
    r = np.asarray(node["r"], dtype=float)
    v = np.asarray(node["v_after"], dtype=float)
    return r, v


def _arrival_state(node):
    r = np.asarray(node["r"], dtype=float)
    v = np.asarray(node["v_before"], dtype=float)
    return r, v


# ------------------------------------------------------------------ piecewise linear

def _piecewise_linear(x: float, breakpoints) -> float:
    """Interpolate y = f(x) from sorted piecewise linear breakpoints."""
    bp = sorted(breakpoints, key=lambda p: float(p[0]))
    if x <= float(bp[0][0]):
        return float(bp[0][1])
    if x >= float(bp[-1][0]):
        return float(bp[-1][1])
    for i in range(len(bp) - 1):
        x0, y0 = float(bp[i][0]), float(bp[i][1])
        x1, y1 = float(bp[i + 1][0]), float(bp[i + 1][1])
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(bp[-1][1])


# ------------------------------------------------------------------ planet position check

def _check_planet_positions(traj, instance, tools, tols):
    tol = tols["planet_pos_tol_km"]
    for i, node in enumerate(traj):
        pid = str(node["planet_id"])
        if pid == "0":
            continue
        r_p, _v_p = tools.ephem(pid, float(node["time"]))
        err = float(np.linalg.norm(np.asarray(node["r"], dtype=float) - r_p))
        if err > tol:
            raise _Infeasible(
                f"planet position mismatch node {i} (planet {pid}): err={err:.3e} km",
                failed_node=i,
            )


# ------------------------------------------------------------------ two-body segment check

def _check_two_body_segments(traj, instance, tools, tols):
    mu_sun = float(instance["mu_sun"])
    pos_tol = tols["segment_pos_tol_km"]
    vel_tol = tols["segment_vel_tol_kms"]
    for i in range(len(traj) - 1):
        r0, v0 = _departure_state(traj[i])
        dt = (float(traj[i + 1]["time"]) - float(traj[i]["time"])) * DAY
        if dt <= 0:
            continue
        r_prop, v_prop = tools.propagate_two_body(r0, v0, dt, mu_sun)
        r_arr, v_arr = _arrival_state(traj[i + 1])
        err_r = float(np.linalg.norm(r_prop - r_arr))
        err_v = float(np.linalg.norm(v_prop - v_arr))
        if err_r > pos_tol:
            raise _Infeasible(
                f"two-body position mismatch segment {i}->{i + 1}: "
                f"err_r={err_r:.3e} km (tol={pos_tol:.3e})",
                failed_node=i,
            )
        if err_v > vel_tol:
            raise _Infeasible(
                f"two-body velocity mismatch segment {i}->{i + 1}: "
                f"err_v={err_v:.3e} km/s (tol={vel_tol:.3e})",
                failed_node=i,
            )


# ------------------------------------------------------------------ gravity assist check

def _check_gravity_assists(traj, instance, tools, tols):
    """Validate GA nodes via powered-flyby model.

    Returns list of (node_index, r_p, dv_flyby) for each GA node.
    """
    ga_results = []
    for i, node in enumerate(traj):
        if node["type"] != NODE_GA:
            continue
        pid = str(node["planet_id"])
        _r_pl, v_pl = tools.ephem(pid, float(node["time"]))
        mu_p = float(instance["planet_mu"][pid])
        min_alt = default_flyby_altitude(instance, pid)
        r_p_min = float(instance["planet_radius"][pid]) + min_alt

        r_p, dv_flyby, feasible = tools.powered_flyby(
            np.asarray(node["v_before"], dtype=float),
            np.asarray(node["v_after"], dtype=float),
            v_pl, mu_p, r_p_min,
        )

        if not feasible:
            alt = r_p - float(instance["planet_radius"][pid])
            raise _Infeasible(
                f"GA node {i} (planet {pid}): required periapsis "
                f"r={r_p:.1f} km < min={r_p_min:.1f} km "
                f"(altitude {alt:.1f} < {min_alt})",
                failed_node=i,
            )

        ga_results.append((i, r_p, float(dv_flyby)))

    return ga_results


# ------------------------------------------------------------------ boundary checks

def _check_boundary_time(node, boundary_spec, tols):
    t = float(node["time"])
    ts = boundary_spec["time"]
    if ts["kind"] == "window":
        lo, hi = float(ts["lo"]), float(ts["hi"])
        if not (lo <= t <= hi):
            raise _Infeasible(
                f"boundary time {t} outside window [{lo}, {hi}]"
            )
    elif ts["kind"] == "exact":
        ref = float(ts["value"])
        if abs(t - ref) > tols["exact_time_tol_days"]:
            raise _Infeasible(
                f"boundary time {t} != exact {ref} "
                f"(tol={tols['exact_time_tol_days']} days)"
            )


def _check_boundary_reference_velocity(node, boundary_spec, tools, tols):
    pid = str(boundary_spec["planet_id"])
    vel_tol = tols["segment_vel_tol_kms"]

    if pid == "0":
        # state reference: verify against boundary's state_v
        ref_v = np.array(boundary_spec["state_v"], dtype=float)
    else:
        _r_pl, v_pl = tools.ephem(pid, float(node["time"]))
        ref_v = v_pl

    if node["type"] == NODE_START:
        node_ref_v = np.asarray(node["v_before"], dtype=float)
    else:
        node_ref_v = np.asarray(node["v_after"], dtype=float)

    err = float(np.linalg.norm(node_ref_v - ref_v))
    if err > vel_tol:
        raise _Infeasible(
            f"boundary reference velocity mismatch: err={err:.3e} km/s"
        )


def _check_boundary_state_position(node, boundary_spec, tols):
    """Verify that a state-mode boundary's position matches the node r."""
    pid = str(boundary_spec["planet_id"])
    if pid != "0":
        return
    ref_r = np.array(boundary_spec["state_r"], dtype=float)
    node_r = np.asarray(node["r"], dtype=float)
    err = float(np.linalg.norm(node_r - ref_r))
    if err > tols["planet_pos_tol_km"]:
        raise _Infeasible(
            f"boundary state position mismatch: err={err:.3e} km"
        )


# ------------------------------------------------------------------ boundary dv computation

def _compute_boundary_dv(node, boundary_spec, tools):
    btype = boundary_spec["type"]

    if btype == "piecewise_linear":
        dv_mag = float(np.linalg.norm(
            np.asarray(node["v_after"], dtype=float)
            - np.asarray(node["v_before"], dtype=float)
        ))
        return _piecewise_linear(dv_mag, boundary_spec["breakpoints"])

    elif btype == "periapsis_maneuver":
        pid = str(boundary_spec["planet_id"])
        mu_p = float(PROBLEM_BASE["planet_mu"][pid])
        R_p = float(PROBLEM_BASE["planet_radius"][pid])
        h_factor = float(boundary_spec["h_factor"])
        T_days = float(boundary_spec["T_days"])
        r_peri = R_p * (1.0 + h_factor)
        T_secs = T_days * DAY

        _r_pl, v_pl = tools.ephem(pid, float(node["time"]))
        if node["type"] == NODE_START:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float) - v_pl
            ))
        else:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_before"], dtype=float) - v_pl
            ))

        two_mu_over_r = 2.0 * mu_p / r_peri
        v_peri_hyp_sq = vinf * vinf + two_mu_over_r
        term = (4.0 * np.pi**2 * mu_p**2 / T_secs**2) ** (1.0 / 3.0)
        v_peri_ell_sq = max(two_mu_over_r - term, 0.0)
        return float(np.sqrt(v_peri_hyp_sq) - np.sqrt(v_peri_ell_sq))

    else:
        raise ValueError(f"Unknown boundary type: {btype}")


# ------------------------------------------------------------------ total dv

def _compute_total_dv(traj, instance, tools, ga_results=None):
    """Return (total_dv, start_dv, end_dv, ga_dv_total, dsm_dv_total).

    ga_results: list of (node_index, r_p, dv_flyby) from _check_gravity_assists.
    """
    total = 0.0

    start_dv = _compute_boundary_dv(traj[0], instance["start"], tools)
    total += start_dv

    ga_dv_total = 0.0
    if ga_results:
        for _idx, _r_p, dv in ga_results:
            ga_dv_total += float(dv)
    total += ga_dv_total

    dsm_dv_total = 0.0
    for node in traj:
        if node["type"] == NODE_DSM:
            dsm_dv_total += float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float)
                - np.asarray(node["v_before"], dtype=float)
            ))
    total += dsm_dv_total

    end_dv = _compute_boundary_dv(traj[-1], instance["end"], tools)
    total += end_dv

    return total, float(start_dv), float(end_dv), float(ga_dv_total), float(dsm_dv_total)


_PLANET_SHORT: dict[str, str] = {
    "0": "st", "1": "Me", "2": "V", "3": "E", "4": "M",
    "5": "J", "6": "S", "7": "U", "8": "N",
}


# ------------------------------------------------------------------ trajectory summary

def _format_boundary_info(node, boundary_spec, tools):
    """Generate type-specific boundary info string for a start/end node."""
    btype = boundary_spec["type"]
    pid = str(boundary_spec["planet_id"])

    if pid != "0":
        _r, v_pl = tools.ephem(pid, float(node["time"]))
        if node["type"] == NODE_START:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float) - v_pl
            ))
        else:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_before"], dtype=float) - v_pl
            ))
    else:
        if node["type"] == NODE_START:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float)
                - np.asarray(node["v_before"], dtype=float)
            ))
        else:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_before"], dtype=float)
                - np.asarray(node["v_after"], dtype=float)
            ))

    if btype == "piecewise_linear":
        return f"vinf={vinf:.2f}"
    elif btype == "periapsis_maneuver":
        h = float(boundary_spec["h_factor"])
        Td = float(boundary_spec["T_days"])
        return f"vinf={vinf:.2f} h={h:.1f}R T={Td:.0f}d"
    return f"vinf={vinf:.2f}"


def _format_trajectory_summary(trajectory, inst_problem, tools):
    """Build a compact per-node summary string (one line per node) with dv."""
    lines = []
    for i, node in enumerate(trajectory):
        ntype = node["type"]
        pid = str(node["planet_id"])
        t = float(node["time"])
        short = _PLANET_SHORT.get(pid, f"id={pid}")

        if ntype == NODE_START:
            dv = _compute_boundary_dv(node, inst_problem["start"], tools)
            info = _format_boundary_info(node, inst_problem["start"], tools)
            lines.append(f"  {i}:{short}({pid}) start t={t:.1f} dv={dv:.4f} {info}")
        elif ntype == NODE_END:
            dv = _compute_boundary_dv(node, inst_problem["end"], tools)
            info = _format_boundary_info(node, inst_problem["end"], tools)
            lines.append(f"  {i}:{short}({pid}) end   t={t:.1f} dv={dv:.4f} {info}")
        elif ntype == NODE_GA:
            _r, v_pl = tools.ephem(pid, t)
            mu_p = float(inst_problem["planet_mu"][pid])
            min_alt = default_flyby_altitude(inst_problem, pid)
            r_p_min = float(inst_problem["planet_radius"][pid]) + min_alt
            _rp, dv_ga, _feas = tools.powered_flyby(
                np.asarray(node["v_before"], dtype=float),
                np.asarray(node["v_after"], dtype=float),
                v_pl, mu_p, r_p_min,
            )
            hp = _rp - float(inst_problem["planet_radius"].get(pid, 0))
            lines.append(
                f"  {i}:{short}({pid}) GA    t={t:.1f} dv={dv_ga:.4f} hp={hp:.0f}"
            )
        elif ntype == NODE_DSM:
            dv = float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float)
                - np.asarray(node["v_before"], dtype=float)
            ))
            lines.append(f"  {i}:DSM    t={t:.1f} dv={dv:.4f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ result helpers

def _make_error_result(error_msg, eval_time=0.0):
    return {
        "score": 0.0,
        "total_dv": float("inf"),
        "error": error_msg,
        "eval_time": float(eval_time),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ------------------------------------------------------------------ memory limit

def _get_memory_limit_bytes():
    return INSTANCE_MEMORY_LIMIT_BYTES


# ------------------------------------------------------------------ subprocess runner

def run_with_timeout(program_path, instance_problem, instance_path, timeout_seconds):
    """Run the candidate program for ONE instance.

    Overrides mod.problem with *instance_problem* before calling run_code().
    Returns {"solution": trajectory} or {"error": msg}.
    """
    limit_bytes = _get_memory_limit_bytes()
    limit_mb = limit_bytes / (1024 * 1024)

    instance_json = json.dumps(instance_problem)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file_path = temp_file.name
        script = f"""
import resource

def limit_memory():
    try:
        soft, hard = {limit_bytes}, {limit_bytes}
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
    except (ValueError, OSError):
        pass

limit_memory()

import sys, os, pickle, traceback as _tb, json, numpy as np, importlib.util

# Ensure repo root and case dir are on sys.path
_repo_root = {_REPO_ROOT!r}
_case_dir = os.path.join(_repo_root, "datasets", "astrodynamics")
for _p in (_repo_root, _case_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.dirname({program_path!r}))

_instance_overrides = json.loads({instance_json!r})

from recorder import Record

def _load(path):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mod = _load({program_path!r})
    # Override the module's 'problem' with the instance-specific problem
    mod.problem = _instance_overrides
    mod.record = Record()
    if not hasattr(mod, 'run_code') or not callable(getattr(mod, 'run_code')):
        raise RuntimeError('Program must define run_code().')
    out = mod.run_code()
    if out is None:
        raise RuntimeError('run_code() returned None (no feasible trajectory found).')

    try:
        rec_text = mod.record.to_string() if hasattr(mod, 'record') else ''
    except Exception:
        rec_text = ''
    if len(rec_text) > 3000:
        rec_text = rec_text[:3000] + '...'

    if isinstance(out, (list,)):
        trajectory = list(out)
    else:
        raise RuntimeError(f'Expected list, got {{type(out).__name__}}')

    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"solution": trajectory, "diagnostics": rec_text}}, f)

except MemoryError:
    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"error": "Memory limit exceeded (MemoryError caught)"}}, f)
except Exception as e:
    with open({(str(temp_file.name) + '.results')!r}, 'wb') as f:
        pickle.dump({{"error": f"{{type(e).__name__}}: {{e}}"}}, f)
"""
        temp_file.write(script.encode())

    results_path = f"{temp_file_path}.results"

    try:
        child_env = os.environ.copy()
        child_env["OMP_NUM_THREADS"] = "4"
        child_env["OPENBLAS_NUM_THREADS"] = "4"
        child_env["MKL_NUM_THREADS"] = "4"
        child_env["NUMEXPR_NUM_THREADS"] = "4"
        child_env["VECLIB_MAXIMUM_THREADS"] = "4"
        child_env["BLIS_NUM_THREADS"] = "4"
        child_env["SIMPLETES_ASTRODYNAMICS_INSTANCE"] = str(instance_path)

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

            stderr_text = stderr.decode(errors="replace")

            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        results = pickle.load(f)
                    if "error" in results:
                        err_msg = results["error"]
                        if "MemoryError" in err_msg or "Memory limit exceeded" in err_msg:
                            raise MemoryLimitExceededError(err_msg)
                        if stderr_text.strip():
                            err_msg = f"{err_msg}\n[stderr]: {stderr_text[-2000:]}"
                        return {"error": err_msg}
                    return results
                except (pickle.UnpicklingError, EOFError):
                    raise RuntimeError(
                        f"Failed to read results file. stderr: {stderr_text[-1000:]}"
                    )

            if exit_code != 0:
                raise RuntimeError(
                    f"Process exited with code {exit_code}. Stderr: {stderr_text[-2000:]}"
                )
            raise RuntimeError(
                f"Results file not found but process exited 0."
            )

        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise EvaluatorTimeoutError(
                f"Process timed out after {timeout_seconds} seconds"
            )

    finally:
        for path in (temp_file_path, results_path):
            if os.path.exists(path):
                os.unlink(path)


# ------------------------------------------------------------------ trajectory persistence

def _serialize_node(node):
    d = {}
    for k, v in node.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
        else:
            d[k] = v
    return d


def _save_trajectory_json(result, program_path):
    """Save selected-mission trajectory + evaluation results to JSON.

    Output: ``<combined_score:.4f>-<isotime>.json`` under TRAJ_SAVE_DIR.
    """
    import datetime as _dt

    try:
        os.makedirs(TRAJ_SAVE_DIR, exist_ok=True)
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        score = result.get("combined_score", 0.0)
        fname = f"{score:.4f}-{ts}.json"
        dest = os.path.join(TRAJ_SAVE_DIR, fname)

        # Build per-instance entries with serialized trajectories
        instances_out = []
        for r in result.get("per_instance", []):
            entry = {k: v for k, v in r.items()}
            traj = entry.pop("_trajectory", None)
            if isinstance(traj, list):
                entry["trajectory"] = [_serialize_node(n) for n in traj]
            instances_out.append(entry)

        payload = {
            "program_path": program_path,
            "combined_score": result.get("combined_score"),
            "validity": result.get("validity"),
            "mean_total_dv": result.get("mean_total_dv"),
            "mean_boundary_dv": result.get("mean_boundary_dv"),
            "mean_ga_dv": result.get("mean_ga_dv"),
            "mean_dsm_dv": result.get("mean_dsm_dv"),
            "failed_instances": result.get("failed_instances"),
            "num_instances": result.get("num_instances"),
            "eval_time": result.get("eval_time"),
            "node_id": result.get("node_id"),
            "selected_mission": result.get("selected_mission"),
            "timestamp_utc": ts,
            "per_instance": instances_out,
        }

        tmp = dest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, default=_json_default, indent=2)
        os.rename(tmp, dest)
    except Exception:
        traceback.print_exc()


# ------------------------------------------------------------------ main entry

def evaluate_instance(program_path: str, instance_path: str | Path) -> dict:
    """Evaluate one candidate against one explicit mission instance.

    Task-level evaluator wrappers preserve SimpleTES's required
    ``evaluate(program_path)`` signature and call this function with the
    sibling ``instance.json`` path.
    """
    instance_path = Path(instance_path).expanduser().resolve()
    instances = [load_instance(str(instance_path))]
    per_instance_results = []
    total_start = time.time()

    for i, instance in enumerate(instances):
        inst_id = instance["id"]
        timeout = float(instance.get("timeout_seconds", 60))

        # Build single-instance problem
        inst_problem = build_problem(instance)

        # Run program for this instance
        inst_start = time.time()
        try:
            res = run_with_timeout(
                program_path, inst_problem, instance_path, timeout
            )
        except EvaluatorTimeoutError:
            per_instance_results.append({
                "instance_id": inst_id,
                "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": f"timeout after {timeout}s",
                "strategy_diagnostics": "",
                "eval_time": timeout,
                "eval_time_limit": timeout,
            })
            continue
        except MemoryLimitExceededError as e:
            per_instance_results.append({
                "instance_id": inst_id,
                "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": "memory_limit_exceeded",
                "memory_error": str(e)[:200],
                "strategy_diagnostics": "",
                "eval_time": time.time() - inst_start,
                "eval_time_limit": timeout,
            })
            continue
        except Exception as e:
            per_instance_results.append({
                "instance_id": inst_id,
                "score": 0.0,
                "total_dv": float("inf"),
                "error": f"{type(e).__name__}: {e}",
                "strategy_diagnostics": "",
                "eval_time": time.time() - inst_start,
                "eval_time_limit": timeout,
            })
            continue

        inst_elapsed = time.time() - inst_start

        if "error" in res:
            per_instance_results.append({
                "instance_id": inst_id,
                "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": str(res["error"]),
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })
            continue

        trajectory = res.get("solution")
        if not isinstance(trajectory, list):
            per_instance_results.append({
                "instance_id": inst_id,
                "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": "Invalid solution format",
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })
            continue

        # Validate and score
        try:
            validate_schema(trajectory, inst_problem)
            _check_planet_positions(trajectory, inst_problem, TOOLS, DEFAULT_TOLS)
            _check_two_body_segments(trajectory, inst_problem, TOOLS, DEFAULT_TOLS)
            ga_results = _check_gravity_assists(trajectory, inst_problem, TOOLS, DEFAULT_TOLS)

            # Boundary checks
            _check_boundary_time(trajectory[0], inst_problem["start"], DEFAULT_TOLS)
            _check_boundary_time(trajectory[-1], inst_problem["end"], DEFAULT_TOLS)
            _check_boundary_reference_velocity(
                trajectory[0], inst_problem["start"], TOOLS, DEFAULT_TOLS
            )
            _check_boundary_reference_velocity(
                trajectory[-1], inst_problem["end"], TOOLS, DEFAULT_TOLS
            )
            _check_boundary_state_position(
                trajectory[0], inst_problem["start"], DEFAULT_TOLS
            )
            _check_boundary_state_position(
                trajectory[-1], inst_problem["end"], DEFAULT_TOLS
            )

            total_dv, start_dv, end_dv, ga_dv, dsm_dv = _compute_total_dv(
                trajectory, inst_problem, TOOLS, ga_results
            )
            target_dv = float(inst_problem.get("target_dv", 1.0))
            score_i = 2.0 * target_dv / (target_dv + max(total_dv, 0.0))

            summary = _format_trajectory_summary(trajectory, inst_problem, TOOLS)
            sequence = _build_sequence(trajectory)

            # Per-instance aggregate stats
            num_GA = sum(1 for n in trajectory if n["type"] == NODE_GA)
            num_DSM = sum(1 for n in trajectory if n["type"] == NODE_DSM)
            duration_days = (
                float(trajectory[-1]["time"]) - float(trajectory[0]["time"])
            )

            # Capture best construction
            capture_construction_if_requested(trajectory)

            per_instance_results.append({
                "instance_id": inst_id,
                    "score": float(score_i),
                "total_dv": float(total_dv),
                "boundary_dv": float(start_dv + end_dv),
                "ga_dv": float(ga_dv),
                "dsm_dv": float(dsm_dv),
                "sequence": sequence,
                "summary": summary,
                "num_GA": num_GA,
                "num_DSM": num_DSM,
                "duration_days": duration_days,
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "_trajectory": trajectory,
                "violations": [],
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })

        except SchemaError as e:
            per_instance_results.append({
                "instance_id": inst_id,
                    "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": f"Schema: {e}",
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })
        except _Infeasible as e:
            per_instance_results.append({
                "instance_id": inst_id,
                    "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": str(e.reason),
                "failed_node": e.failed_node,
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })
        except Exception as e:
            per_instance_results.append({
                "instance_id": inst_id,
                    "score": 0.0,
                "total_dv": float("inf"),
                "boundary_dv": 0.0,
                "ga_dv": 0.0,
                "dsm_dv": 0.0,
                "error": f"Eval: {type(e).__name__}: {e}",
                "strategy_diagnostics": str(res.get("diagnostics", "")),
                "eval_time": inst_elapsed,
                "eval_time_limit": timeout,
            })

    total_elapsed = time.time() - total_start
    scores = [r["score"] for r in per_instance_results]
    combined_score = float(np.mean(scores)) if scores else 0.0

    total_dvs = [r.get("total_dv", float("inf")) for r in per_instance_results
                 if r.get("total_dv", float("inf")) != float("inf")]

    finite_dvs = [d for d in total_dvs if d != float("inf")]
    boundary_dvs = [r.get("boundary_dv", 0) for r in per_instance_results
                    if r.get("score", 0) > 0]
    ga_dvs = [r.get("ga_dv", 0) for r in per_instance_results
              if r.get("score", 0) > 0]
    dsm_dvs = [r.get("dsm_dv", 0) for r in per_instance_results
               if r.get("score", 0) > 0]
    failed = sum(1 for r in per_instance_results if r.get("score", 0) == 0)

    node_id = os.environ.get("SE_NODE_ID", "unknown")

    result = {
        "combined_score": combined_score,
        "validity": sum(1 for r in per_instance_results if r.get("score", 0) > 0)
                    / max(len(per_instance_results), 1),
        "mission_success_rate": sum(1 for r in per_instance_results if r.get("score", 0) > 0)
                                / max(len(per_instance_results), 1),
        "selected_mission": instances[0]["id"] if len(instances) == 1 else "multiple",
        "mean_total_dv": float(np.mean(total_dvs)) if total_dvs else float("inf"),
        "mean_boundary_dv": float(np.mean(boundary_dvs)) if boundary_dvs else 0.0,
        "mean_ga_dv": float(np.mean(ga_dvs)) if ga_dvs else 0.0,
        "mean_dsm_dv": float(np.mean(dsm_dvs)) if dsm_dvs else 0.0,
        "failed_instances": failed,
        "num_instances": len(instances),
        "eval_time": float(total_elapsed),
        "per_instance": per_instance_results,
        "node_id": node_id,
    }

    if SAVE_TRAJECTORIES:
        _save_trajectory_json(result, program_path)

    # Strip numpy _trajectory from the engine-facing result
    for r in per_instance_results:
        r.pop("_trajectory", None)

    return result


def _build_sequence(trajectory):
    parts = []
    for node in trajectory:
        pid = str(node["planet_id"])
        if node["type"] == NODE_START:
            parts.append(_PLANET_NAME.get(pid, f"id={pid}")[0])
        elif node["type"] == NODE_GA:
            parts.append(f"->{_PLANET_NAME.get(pid, f'id={pid}')[0]}")
        elif node["type"] == NODE_END:
            parts.append(f"->{_PLANET_NAME.get(pid, f'id={pid}')[0]}")
    return "".join(parts)
