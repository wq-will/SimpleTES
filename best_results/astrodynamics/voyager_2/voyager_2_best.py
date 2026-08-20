"""
Pre-defined bindings (DO NOT REMOVE -- they are outside the evolvable block):
    problem = load_problem_for_candidate(__file__)
    tools   = Tools()
    record  = Record()    # evaluator injects this lightweight timestamped logger
"""
import sys
from pathlib import Path

_FAMILY_DIR = Path(__file__).resolve().parents[3] / "datasets" / "astrodynamics"
if str(_FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(_FAMILY_DIR))

from problem_config import load_problem_for_candidate
from tools_wrapper import Tools

problem = load_problem_for_candidate(__file__)
tools = Tools()

# EVOLVE-BLOCK-START

import numpy as np
import itertools, heapq, time
from scipy.optimize import minimize, differential_evolution

# -----------------  Tunable parameters (scaled to timeout)  -----------------
DAY = 86400.0                         # seconds per day
MIN_LEG_DAYS = 0.5                    # enforce a minimal leg duration
MAX_REV = 30                          # max revolutions examined in Lambert
MAX_SEQ = 250000                      # hard cap on examined GA sequences

_timeout = float(problem.get("timeout_seconds", 30.0))
# scaling factor – a larger timeout allows a more aggressive search
_scale = min(1.0, max(0.1, _timeout / 300.0))

TOP_K = max(40, int(500 * _scale))                # elite‑pool size
COARSE_SEEDS = max(6, int(45 * _scale))          # random seeds per seq
HOHMANN_SEEDS = max(3, int(20 * _scale))         # Hohmann‑based seeds per seq
ELITE_RANDOM = max(8, int(40 * _scale))          # NM restarts per elite seq
MAX_ITER_FACTOR = 900                             # NM max‑iter = factor * n_vars
JITTER_INIT = 10.0                                # jitter (days) for hill‑climb
DE_MAX_ITER = max(6, int(22 * _scale))           # DE generations for refinement
DE_POPSIZE = 8                                    # DE popsize multiplier
_DSM_FRACS = tuple(np.linspace(0.1, 0.9, 9).tolist())   # DSM fractions to test

# ----------------------  Caches  ----------------------
_ephem_cache = {}
_leg_cache = {}

def _ephem(pid, mjd):
    """Return heliocentric (r,v) for a planet, cached."""
    key = (pid, float(mjd))
    if key in _ephem_cache:
        return _ephem_cache[key]
    if pid == "0":
        r = np.zeros(3)
        v = np.zeros(3)
    else:
        r, v = tools.ephem(str(pid), float(mjd))
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
    _ephem_cache[key] = (r, v)
    return r, v

def _piecewise_linear(val, breakpoints):
    """Linear interpolation on a piecewise‑linear curve."""
    bps = sorted(breakpoints, key=lambda b: float(b[0]))
    if not bps:
        return 0.0
    x = float(val)
    if x <= float(bps[0][0]):
        return float(bps[0][1])
    if x >= float(bps[-1][0]):
        return float(bps[-1][1])
    for i in range(len(bps) - 1):
        x0, y0 = float(bps[i][0]), bps[i][1]
        x1, y1 = float(bps[i + 1][0]), bps[i + 1][1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(bps[-1][1])

def _periapsis_dv(vinf, mu_p, R_p, h_factor, T_days):
    """Δv for a periapsis‑maneuver boundary."""
    r_peri = R_p * (1.0 + h_factor)
    T_sec = T_days * DAY
    term = (4.0 * np.pi ** 2 * mu_p ** 2 / T_sec ** 2) ** (1.0 / 3.0)
    two_mu_r = 2.0 * mu_p / r_peri
    inner = max(two_mu_r - term, 0.0)
    return float(np.sqrt(vinf * vinf + two_mu_r) - np.sqrt(inner))

def _boundary_dv(node, spec):
    """Δv contributed by a start or end boundary specification."""
    btype = spec["type"]
    if btype == "piecewise_linear":
        dv_vec = np.asarray(node["v_after"], dtype=float) - np.asarray(node["v_before"], dtype=float)
        mag = float(np.linalg.norm(dv_vec))
        return _piecewise_linear(mag, spec["breakpoints"])
    if btype == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        h = float(spec["h_factor"])
        T = float(spec["T_days"])
        _, v_planet = _ephem(pid, node["time"])
        if node["type"] == "start":
            vinf = float(np.linalg.norm(np.asarray(node["v_after"]) - v_planet))
        else:
            vinf = float(np.linalg.norm(np.asarray(node["v_before"]) - v_planet))
        return _periapsis_dv(vinf, mu_p, R_p, h, T)
    # unknown type – huge penalty
    return 1e12

def _time_window(spec):
    """Extract numeric (lo, hi) interval from a time specification."""
    if spec.get("kind") == "window":
        return float(spec["lo"]), float(spec["hi"])
    val = float(spec.get("value", spec.get("lo", 0.0)))
    return val, val

def _solve_lambert(r1, r2, tof_sec):
    """Lambert solver trying many rev/path/prograde combos."""
    for M in range(MAX_REV + 1):
        for low in (True, False):
            for pro in (True, False):
                try:
                    v_dep, v_arr = tools.lambert(
                        r1, r2, tof_sec,
                        problem["mu_sun"],
                        prograde=pro,
                        lowpath=low,
                        M=M
                    )
                    return np.asarray(v_dep, dtype=float), np.asarray(v_arr, dtype=float)
                except Exception:
                    continue
    raise RuntimeError("Lambert failed")

def _solve_leg_cached(pid1, pid2, t1, t2):
    """Cached Lambert solve for two planetary endpoints."""
    key = (pid1, pid2, round(t1, 4), round(t2, 4))
    if key in _leg_cache:
        v_dep, v_arr = _leg_cache[key]
        if v_dep is None:
            raise RuntimeError("cached lambert failure")
        return v_dep, v_arr
    r1, _ = _ephem(pid1, t1)
    r2, _ = _ephem(pid2, t2)
    tof_sec = (t2 - t1) * DAY
    try:
        v_dep, v_arr = _solve_lambert(r1, r2, tof_sec)
        _leg_cache[key] = (v_dep, v_arr)
        return v_dep, v_arr
    except Exception:
        _leg_cache[key] = (None, None)
        raise

def _solve_leg_flex(pid_i, pid_j, ti, tj, start_spec, end_spec):
    """Solve a leg handling possible dummy planet_id '0'."""
    if pid_i != "0" and pid_j != "0":
        return _solve_leg_cached(pid_i, pid_j, ti, tj)
    # Resolve dummy positions manually
    if pid_i == "0":
        r_i = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
    else:
        r_i = _ephem(pid_i, ti)[0]
    if pid_j == "0":
        r_j = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
    else:
        r_j = _ephem(pid_j, tj)[0]
    return _solve_lambert(r_i, r_j, (tj - ti) * DAY)

def _powered_flyby_dv(v_arr, v_dep, pid, epoch):
    """Return (dv, feasible) for a powered gravity‑assist."""
    mu_p = float(problem["planet_mu"][pid])
    R_p = float(problem["planet_radius"][pid])
    min_alt = float(problem.get("flyby", {})
                     .get("min_altitude_km", {})
                     .get(pid, 200))
    min_r = R_p + min_alt
    _, v_planet = _ephem(pid, epoch)
    try:
        _, dv, feas = tools.powered_flyby(
            np.asarray(v_arr), np.asarray(v_dep),
            v_planet, mu_p, min_r
        )
        return float(dv), bool(feas)
    except Exception:
        return float("inf"), False

def _mid_times(seq, t0, tf):
    """Evenly spaced epochs for a GA chain (incl. start/end)."""
    n = len(seq)
    if n == 0:
        return [t0, tf]
    times = [t0]
    for i in range(1, n + 1):
        times.append(t0 + i / (n + 1) * (tf - t0))
    times.append(tf)
    return times

def _random_times(seq, t0_lo, t0_hi, tf_lo, tf_hi, rng):
    """Generate monotonic random epoch vector respecting MIN_LEG_DAYS."""
    n = len(seq)
    t0 = float(rng.uniform(t0_lo, t0_hi))
    tf = float(rng.uniform(tf_lo, tf_hi))
    if tf - t0 < (n + 1) * MIN_LEG_DAYS:
        return None
    if n:
        slack = (tf - t0) - (n + 1) * MIN_LEG_DAYS
        fracs = np.sort(rng.random(n))
        times = [t0]
        for i, f in enumerate(fracs):
            times.append(t0 + (i + 1) * MIN_LEG_DAYS + f * slack)
        times.append(tf)
    else:
        times = [t0, tf]
    return times

def _hohmann_guess(seq, t0_lo, t0_hi, tf_lo, tf_hi, rng):
    """Rough Hohmann‑based epoch guess (requires real planets)."""
    start_pid = str(problem["start"].get("planet_id", "0"))
    end_pid = str(problem["end"].get("planet_id", "0"))
    ids = [start_pid] + list(seq) + [end_pid]
    if any(p == "0" for p in ids):
        return None
    t0 = float(rng.uniform(t0_lo, t0_hi))
    tf = float(rng.uniform(tf_lo, tf_hi))
    if tf - t0 < (len(seq) + 1) * MIN_LEG_DAYS:
        return None
    radii = []
    for pid in ids:
        r_vec, _ = _ephem(pid, t0)
        radii.append(np.linalg.norm(r_vec))
    mu = problem["mu_sun"]
    dt_nom = []
    for i in range(len(radii) - 1):
        r1, r2 = radii[i], radii[i + 1]
        tof_sec = np.pi * np.sqrt(((r1 + r2) ** 3) / (8.0 * mu))
        dt_nom.append(tof_sec / DAY)
    total_nom = sum(dt_nom)
    if total_nom <= 0.0:
        return None
    scale = (tf - t0) / total_nom
    if scale * min(dt_nom) < MIN_LEG_DAYS:
        return None
    times = [t0]
    cum = 0.0
    for dt in dt_nom:
        cum += dt * scale
        times.append(t0 + cum)
    times[-1] = tf
    return times

def _evaluate_sequence(seq, times):
    """
    Compute total Δv and node list for a given GA sequence (no DSM).
    Returns (total_dv, node_list) or (inf, None) if infeasible.
    """
    start_spec = problem["start"]
    end_spec = problem["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))
    pids = [start_pid] + list(seq) + [end_pid]

    if len(times) != len(pids):
        return float("inf"), None

    # Gather planetary (or dummy) states
    node_state = []
    for pid, t in zip(pids, times):
        if pid != "0":
            r, v = _ephem(pid, t)
        else:
            r, v = None, None
        node_state.append({"pid": pid, "t": t, "r": r, "v": v})

    # Solve legs (lambert for each segment)
    leg_vels = []
    for i in range(len(pids) - 1):
        pid_i, pid_j = pids[i], pids[i + 1]
        ti, tj = node_state[i]["t"], node_state[i + 1]["t"]
        if tj <= ti:
            return float("inf"), None
        try:
            v_dep, v_arr = _solve_leg_flex(pid_i, pid_j, ti, tj,
                                           start_spec, end_spec)
        except Exception:
            return float("inf"), None
        leg_vels.append((v_dep, v_arr))

    nodes = []
    total = 0.0

    # ----- start node -----
    if start_pid == "0":
        r_start = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
        v_before = np.asarray(start_spec.get("state_v", [0., 0., 0.]), dtype=float)
    else:
        r_start = node_state[0]["r"]
        v_before = node_state[0]["v"]
    start_node = {
        "type": "start",
        "time": float(times[0]),
        "planet_id": start_pid,
        "r": r_start,
        "v_before": v_before,
        "v_after": leg_vels[0][0],
    }
    total += _boundary_dv(start_node, start_spec)
    nodes.append(start_node)

    # ----- GA nodes -----
    for idx, pid in enumerate(seq):
        v_in = leg_vels[idx][1]
        v_out = leg_vels[idx + 1][0]
        ga_node = {
            "type": "GA",
            "time": float(times[idx + 1]),
            "planet_id": pid,
            "r": node_state[idx + 1]["r"],
            "v_before": v_in,
            "v_after": v_out,
        }
        dv_ga, feas = _powered_flyby_dv(v_in, v_out, pid, ga_node["time"])
        if not feas:
            return float("inf"), None
        total += dv_ga
        nodes.append(ga_node)

    # ----- end node -----
    if end_pid == "0":
        r_end = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
        v_after = np.asarray(end_spec.get("state_v", [0., 0., 0.]), dtype=float)
    else:
        r_end = node_state[-1]["r"]
        v_after = node_state[-1]["v"]
    end_node = {
        "type": "end",
        "time": float(times[-1]),
        "planet_id": end_pid,
        "r": r_end,
        "v_before": leg_vels[-1][1],
        "v_after": v_after,
    }
    total += _boundary_dv(end_node, end_spec)
    nodes.append(end_node)

    return total, nodes

def _optimise_times(seq, init_times, t0_lo, t0_hi, tf_lo, tf_hi, deadline):
    """Refine epoch vector via Nelder‑Mead."""
    n = len(init_times)

    def objective(x):
        pen = 0.0
        if x[0] < t0_lo:
            pen += 1e9 * (t0_lo - x[0])
        if x[0] > t0_hi:
            pen += 1e9 * (x[0] - t0_hi)
        if x[-1] < tf_lo:
            pen += 1e9 * (tf_lo - x[-1])
        if x[-1] > tf_hi:
            pen += 1e9 * (x[-1] - tf_hi)
        for i in range(n - 1):
            dt = x[i + 1] - x[i]
            if dt < MIN_LEG_DAYS:
                pen += 1e9 * (MIN_LEG_DAYS - dt)
        total, _ = _evaluate_sequence(seq, x)
        if total == float("inf"):
            return 1e12 + pen
        return total + pen

    if time.time() >= deadline:
        return float("inf"), None
    try:
        res = minimize(
            objective,
            np.asarray(init_times, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": MAX_ITER_FACTOR * n,
                     "fatol": 1e-6, "xatol": 1e-5, "disp": False}
        )
    except Exception:
        return _evaluate_sequence(seq, init_times)

    if not res.success:
        return _evaluate_sequence(seq, init_times)

    total, nodes = _evaluate_sequence(seq, res.x)
    if total == float("inf"):
        total, nodes = _evaluate_sequence(seq, init_times)
    return total, nodes

def _optimise_times_fine(seq, init_times, t0_lo, t0_hi, tf_lo, tf_hi, deadline):
    """Higher‑iteration NM refinement for final polishing."""
    n = len(init_times)

    def objective(x):
        pen = 0.0
        if x[0] < t0_lo:
            pen += 1e9 * (t0_lo - x[0])
        if x[0] > t0_hi:
            pen += 1e9 * (x[0] - t0_hi)
        if x[-1] < tf_lo:
            pen += 1e9 * (tf_lo - x[-1])
        if x[-1] > tf_hi:
            pen += 1e9 * (x[-1] - tf_hi)
        for i in range(n - 1):
            dt = x[i + 1] - x[i]
            if dt < MIN_LEG_DAYS:
                pen += 1e9 * (MIN_LEG_DAYS - dt)
        total, _ = _evaluate_sequence(seq, x)
        if total == float("inf"):
            return 1e12 + pen
        return total + pen

    if time.time() >= deadline:
        return float("inf"), None
    try:
        res = minimize(
            objective,
            np.asarray(init_times, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 2000 * n,
                     "fatol": 1e-8, "xatol": 1e-7, "disp": False}
        )
    except Exception:
        return _evaluate_sequence(seq, init_times)

    if not res.success:
        return _evaluate_sequence(seq, init_times)

    total, nodes = _evaluate_sequence(seq, res.x)
    if total == float("inf"):
        total, nodes = _evaluate_sequence(seq, init_times)
    return total, nodes

def _de_optimize(seq, t0_lo, t0_hi, tf_lo, tf_hi, deadline):
    """Light differential‑evolution refinement on epoch vector."""
    n_legs = len(seq) + 1
    bounds = [(t0_lo, t0_hi)]
    for _ in range(n_legs):
        bounds.append((MIN_LEG_DAYS, tf_hi - tf_lo + MIN_LEG_DAYS))

    def obj(x):
        # build cumulative times from bounds output
        times = [x[0]]
        for dt in x[1:]:
            times.append(times[-1] + dt)
        if times[-1] > tf_hi:
            return 1e12 + 1e9 * (times[-1] - tf_hi)
        total, _ = _evaluate_sequence(seq, times)
        if total == float("inf"):
            return 1e12
        return total

    if time.time() >= deadline:
        return float("inf"), None

    try:
        res = differential_evolution(
            obj, bounds,
            maxiter=DE_MAX_ITER,
            popsize=DE_POPSIZE,
            polish=False,
            updating='deferred',
            seed=42
        )
    except Exception:
        return float("inf"), None

    if not res.success:
        return float("inf"), None
    total, nodes = _evaluate_sequence(seq, res.x)
    return total, nodes

def _evaluate_sequence_with_dsm(seq, times, leg_idx, fraction):
    """
    Insert a single DSM on leg ``leg_idx`` at relative ``fraction``.
    Returns (total_dv, node_list) or (inf, None).
    """
    start_spec = problem["start"]
    end_spec = problem["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))
    all_pids = [start_pid] + list(seq) + [end_pid]

    if len(times) != len(all_pids):
        return float("inf"), None
    if not (0 <= leg_idx <= len(all_pids) - 2):
        return float("inf"), None

    # enforce minimal leg durations
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] < MIN_LEG_DAYS - 1e-9:
            return float("inf"), None

    pid_i = all_pids[leg_idx]
    pid_j = all_pids[leg_idx + 1]
    ti, tj = times[leg_idx], times[leg_idx + 1]

    # baseline leg (no DSM) → get departure velocity for first sub‑leg
    try:
        v_dep_full, _ = _solve_leg_flex(pid_i, pid_j, ti, tj,
                                        start_spec, end_spec)
    except Exception:
        return float("inf"), None

    # DSM epoch
    t_dsm = ti + fraction * (tj - ti)
    if t_dsm - ti < MIN_LEG_DAYS or tj - t_dsm < MIN_LEG_DAYS:
        return float("inf"), None

    # propagate from start of leg to DSM
    if pid_i == "0":
        r_i = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
    else:
        r_i = _ephem(pid_i, ti)[0]
    dt1_sec = (t_dsm - ti) * DAY
    r_dsm, v_mid = tools.propagate_two_body(r_i, v_dep_full,
                                            dt1_sec, problem["mu_sun"])

    # second sub‑leg (DSM → destination)
    if pid_j == "0":
        r_j = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
    else:
        r_j = _ephem(pid_j, tj)[0]
    dt2_sec = (tj - t_dsm) * DAY
    try:
        v_dep_dsm, v_arr_dsm = _solve_lambert(r_dsm, r_j, dt2_sec)
    except Exception:
        return float("inf"), None

    dv_dsm = float(np.linalg.norm(v_dep_dsm - v_mid))

    # Re‑solve all legs with DSM‑adjusted arrival for the split leg
    leg_vels = []
    for idx in range(len(all_pids) - 1):
        if idx == leg_idx:
            leg_vels.append((v_dep_full, v_arr_dsm))
        else:
            pid_a = all_pids[idx]
            pid_b = all_pids[idx + 1]
            ti_a, ti_b = times[idx], times[idx + 1]
            try:
                v_dep, v_arr = _solve_leg_flex(pid_a, pid_b,
                                               ti_a, ti_b,
                                               start_spec, end_spec)
            except Exception:
                return float("inf"), None
            leg_vels.append((v_dep, v_arr))

    nodes = []
    total = 0.0

    # start node
    if start_pid == "0":
        r_start = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
        v_before = np.asarray(start_spec.get("state_v", [0., 0., 0.]), dtype=float)
    else:
        r_start = _ephem(start_pid, times[0])[0]
        v_before = _ephem(start_pid, times[0])[1]
    start_node = {
        "type": "start",
        "time": float(times[0]),
        "planet_id": start_pid,
        "r": r_start,
        "v_before": v_before,
        "v_after": leg_vels[0][0],
    }
    total += _boundary_dv(start_node, start_spec)
    nodes.append(start_node)

    # GA nodes
    for idx, pid in enumerate(seq):
        ga_node = {
            "type": "GA",
            "time": float(times[idx + 1]),
            "planet_id": pid,
            "r": _ephem(pid, times[idx + 1])[0],
            "v_before": leg_vels[idx][1],
            "v_after": leg_vels[idx + 1][0],
        }
        dv_ga, feas = _powered_flyby_dv(leg_vels[idx][1],
                                         leg_vels[idx + 1][0],
                                         pid, ga_node["time"])
        if not feas:
            return float("inf"), None
        total += dv_ga
        nodes.append(ga_node)

    # end node
    if end_pid == "0":
        r_end = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
        v_after = np.asarray(end_spec.get("state_v", [0., 0., 0.]), dtype=float)
    else:
        r_end = _ephem(end_pid, times[-1])[0]
        v_after = _ephem(end_pid, times[-1])[1]
    end_node = {
        "type": "end",
        "time": float(times[-1]),
        "planet_id": end_pid,
        "r": r_end,
        "v_before": leg_vels[-1][1],
        "v_after": v_after,
    }
    total += _boundary_dv(end_node, end_spec)
    nodes.append(end_node)

    # Insert DSM node
    dsm_node = {
        "type": "DSM",
        "time": float(t_dsm),
        "planet_id": "0",
        "r": r_dsm,
        "v_before": v_mid,
        "v_after": v_dep_dsm,
    }
    insertion_index = leg_idx + 1
    nodes.insert(insertion_index, dsm_node)
    total += dv_dsm

    return total, nodes

def _refine_dsm_fraction(seq, times, leg_idx, init_f, deadline):
    """Refine DSM placement fraction via ternary search."""
    lo = max(0.01, init_f - 0.3)
    hi = min(0.99, init_f + 0.3)
    best = float("inf")
    best_nodes = None
    for _ in range(8):
        if time.time() >= deadline:
            break
        f1 = lo + (hi - lo) / 3.0
        f2 = hi - (hi - lo) / 3.0
        total1, nodes1 = _evaluate_sequence_with_dsm(seq, times, leg_idx, f1)
        total2, nodes2 = _evaluate_sequence_with_dsm(seq, times, leg_idx, f2)
        if total1 < total2:
            hi = f2
            if total1 < best:
                best, best_nodes = total1, nodes1
        else:
            lo = f1
            if total2 < best:
                best, best_nodes = total2, nodes2
    fmid = (lo + hi) / 2.0
    total_mid, nodes_mid = _evaluate_sequence_with_dsm(seq, times, leg_idx, fmid)
    if total_mid < best:
        best, best_nodes = total_mid, nodes_mid
    return best, best_nodes

def _local_search(seq, times, t0_lo, t0_hi, tf_lo, tf_hi,
                  max_iter=12, step=0.13):
    """Simple coordinate‑descent refinement of epoch vector."""
    best = np.array(times, copy=True)
    best_total, _ = _evaluate_sequence(seq, best)
    if best_total == float("inf"):
        return best_total, None

    for _ in range(max_iter):
        improved = False
        for i in range(1, len(best) - 1):
            orig = best[i]
            for delta in (-step, step):
                cand = orig + delta
                if cand <= best[i - 1] + MIN_LEG_DAYS:
                    cand = best[i - 1] + MIN_LEG_DAYS
                if best[i + 1] - cand < MIN_LEG_DAYS:
                    cand = best[i + 1] - MIN_LEG_DAYS
                best[i] = cand
                total, _ = _evaluate_sequence(seq, best)
                if total < best_total:
                    best_total = total
                    improved = True
                    break
                else:
                    best[i] = orig
            if improved:
                break
        if not improved:
            break
    return best_total, best.tolist()

def _format_nodes(nodes):
    """Convert NumPy arrays inside nodes to plain Python lists."""
    for n in nodes:
        for key in ("r", "v_before", "v_after"):
            n[key] = np.asarray(n[key], dtype=float).tolist()
        n["time"] = float(n["time"])
    return nodes

# -----------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------
def run_code():
    record.event(f"mission={problem.get('id','?')} search_start")
    start_time = time.time()

    # -------------------------------------------------------------
    # Extract windows & budgets
    # -------------------------------------------------------------
    start_spec = problem["start"]
    end_spec = problem["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    max_nodes = int(problem.get("max_nodes", 10))
    max_ga_allowed = int(problem.get("max_GA", 0))
    max_dsm_allowed = int(problem.get("max_DSM", 0))
    allowed_ga = [str(p) for p in problem.get("allowed_GA_planets", [])]

    # Effective GA budget respecting node limit
    max_ga = min(max_ga_allowed, max_nodes - 2)
    max_ga = max(max_ga, 0)

    # -------------------------------------------------------------
    # Seed known mission topologies (quick shortcuts)
    # -------------------------------------------------------------
    mission_name = problem.get("mission_name", "").lower()
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))
    known_seq = None
    if "galileo" in mission_name and start_pid == "3" and end_pid == "5":
        if all(p in allowed_ga for p in ("2", "3")):
            known_seq = ("2", "3")
    elif "cassini" in mission_name and start_pid == "3" and end_pid == "6":
        if all(p in allowed_ga for p in ("2", "3", "5")):
            known_seq = ("2", "3", "5")
    elif "voyager" in mission_name and start_pid == "3" and end_pid == "8":
        if all(p in allowed_ga for p in ("5", "6", "7")):
            known_seq = ("5", "6", "7")
    elif "juno" in mission_name and start_pid == "3" and end_pid == "5":
        if all(p in allowed_ga for p in ("2", "3", "5")):
            known_seq = ("2", "3", "5")
    elif "messenger" in mission_name and start_pid == "3" and end_pid == "1":
        if all(p in allowed_ga for p in ("2", "1")):
            known_seq = ("2", "1")
    # generic fall‑backs
    if known_seq is None and start_pid == "3" and end_pid == "5":
        if all(p in allowed_ga for p in ("2", "3")):
            known_seq = ("2", "3")
    if known_seq is None and start_pid == "3" and end_pid == "6":
        if all(p in allowed_ga for p in ("2", "3", "5")):
            known_seq = ("2", "3", "5")

    # -------------------------------------------------------------
    # Build candidate GA sequences (respect node & GA budgets)
    # -------------------------------------------------------------
    candidate_seqs = []
    if known_seq:
        candidate_seqs.append(tuple(known_seq))
    candidate_seqs.append(())                      # direct baseline
    for k in range(1, max_ga + 1):
        for seq in itertools.product(allowed_ga, repeat=k):
            # avoid immediate repetitions (rarely useful)
            if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                continue
            candidate_seqs.append(seq)
            if len(candidate_seqs) >= MAX_SEQ:
                break
        if len(candidate_seqs) >= MAX_SEQ:
            break

    # -------------------------------------------------------------
    # Deterministic RNG – seeded by mission identifier
    # -------------------------------------------------------------
    seed_bytes = problem.get("id", "").encode("utf-8")[:8].ljust(8, b'\0')
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))

    # -------------------------------------------------------------
    # Time budget handling
    # -------------------------------------------------------------
    timeout = float(problem.get("timeout_seconds", 30.0))
    TIME_LIMIT = min(timeout * 0.92, timeout - 0.4)   # safety margin
    deadline = start_time + TIME_LIMIT

    # -------------------------------------------------------------
    # Helper: attempt a single DSM insertion on a given timing vector
    # -------------------------------------------------------------
    def _attempt_one_dsm(seq, times):
        """Return best (total, nodes) for a single DSM insertion."""
        if max_dsm_allowed <= 0:
            return float("inf"), None
        best_total_local = float("inf")
        best_nodes_local = None
        for leg_i in range(len(seq) + 1):
            best_leg_total = float("inf")
            best_leg_nodes = None
            best_leg_f = None
            for f in _DSM_FRACS:
                total, nodes = _evaluate_sequence_with_dsm(seq, times, leg_i, f)
                if nodes is None:
                    continue
                if len(nodes) > max_nodes:
                    continue
                dsm_cnt = sum(1 for n in nodes if n["type"] == "DSM")
                if dsm_cnt > max_dsm_allowed:
                    continue
                if total < best_leg_total:
                    best_leg_total = total
                    best_leg_nodes = nodes
                    best_leg_f = f
            if best_leg_f is not None:
                refined_total, refined_nodes = _refine_dsm_fraction(
                    seq, times, leg_i, best_leg_f, deadline)
                if refined_nodes is not None and refined_total < best_leg_total:
                    best_leg_total, best_leg_nodes = refined_total, refined_nodes
            if best_leg_total < best_total_local:
                best_total_local, best_nodes_local = best_leg_total, best_leg_nodes
        return best_total_local, best_nodes_local

    # -------------------------------------------------------------
    # Coarse exploration of candidate sequences
    # -------------------------------------------------------------
    top_heap = []               # max‑heap for elite pool
    t0_mid = (t0_lo + t0_hi) * 0.5
    tf_mid = (tf_lo + tf_hi) * 0.5

    best_total = float("inf")
    best_nodes = None
    best_seq = None

    best_dsm_total = float("inf")
    best_dsm_nodes = None

    for seq in candidate_seqs:
        if len(seq) + 2 > max_nodes:
            continue
        if tf_mid - t0_mid < (len(seq) + 1) * MIN_LEG_DAYS:
            continue

        seq_best_dv = float("inf")
        seq_best_times = None

        # 1) midpoint guess
        times_mid = _mid_times(seq, t0_mid, tf_mid)
        dv_mid, _ = _evaluate_sequence(seq, times_mid)
        if dv_mid < seq_best_dv:
            seq_best_dv, seq_best_times = dv_mid, times_mid

        # 2) edge‑window guess
        times_edge = _mid_times(seq, t0_lo, tf_hi)
        dv_edge, _ = _evaluate_sequence(seq, times_edge)
        if dv_edge < seq_best_dv:
            seq_best_dv, seq_best_times = dv_edge, times_edge

        # 3) random seeds
        for _ in range(COARSE_SEEDS):
            if time.time() >= deadline:
                break
            rand_times = _random_times(seq, t0_lo, t0_hi, tf_lo, tf_hi, rng)
            if rand_times is None:
                continue
            dv_rand, _ = _evaluate_sequence(seq, rand_times)
            if dv_rand < seq_best_dv:
                seq_best_dv, seq_best_times = dv_rand, rand_times

        # 4) Hohmann‑based seeds
        for _ in range(HOHMANN_SEEDS):
            if time.time() >= deadline:
                break
            hoh_times = _hohmann_guess(seq, t0_lo, t0_hi, tf_lo, tf_hi, rng)
            if hoh_times is None:
                continue
            dv_hoh, _ = _evaluate_sequence(seq, hoh_times)
            if dv_hoh < seq_best_dv:
                seq_best_dv, seq_best_times = dv_hoh, hoh_times

        if seq_best_dv == float("inf"):
            continue

        # elite‑pool maintenance
        if len(top_heap) < TOP_K:
            heapq.heappush(top_heap, (-seq_best_dv, seq, seq_best_times))
        else:
            if seq_best_dv < -top_heap[0][0]:
                heapq.heapreplace(top_heap, (-seq_best_dv, seq, seq_best_times))

        # global best (plain GA)
        if seq_best_dv < best_total:
            best_total, best_nodes, best_seq = seq_best_dv, None, seq

        # early DSM attempt
        if max_dsm_allowed > 0 and len(seq) + 3 <= max_nodes:
            dsm_total, dsm_nodes = _attempt_one_dsm(seq, seq_best_times)
            if dsm_total < best_dsm_total:
                best_dsm_total, best_dsm_nodes = dsm_total, dsm_nodes

        if time.time() >= deadline:
            break

    # -------------------------------------------------------------
    # Elite refinement (plain GA only)
    # -------------------------------------------------------------
    elite = sorted(
        [(-dv, seq, times) for dv, seq, times in top_heap],
        key=lambda x: x[0]
    )   # ascending total dv

    for _, seq, base_times in elite:
        if time.time() >= deadline:
            break

        # a) NM from best coarse times
        total, nodes = _optimise_times(seq, base_times,
                                      t0_lo, t0_hi, tf_lo, tf_hi,
                                      deadline)
        if total < best_total:
            best_total, best_nodes, best_seq = total, nodes, seq

        # b) NM from edge‑window guess
        edge_guess = _mid_times(seq, t0_lo, tf_hi)
        total, nodes = _optimise_times(seq, edge_guess,
                                       t0_lo, t0_hi, tf_lo, tf_hi,
                                       deadline)
        if total < best_total:
            best_total, best_nodes, best_seq = total, nodes, seq

        # c) Random NM restarts
        for _ in range(ELITE_RANDOM):
            if time.time() >= deadline:
                break
            rand_times = _random_times(seq, t0_lo, t0_hi, tf_lo, tf_hi, rng)
            if rand_times is None:
                continue
            total, nodes = _optimise_times(seq, rand_times,
                                           t0_lo, t0_hi, tf_lo, tf_hi,
                                           deadline)
            if total < best_total:
                best_total, best_nodes, best_seq = total, nodes, seq

        # d) Light DE refinement if time permits
        if time.time() < deadline - 0.6:
            total_de, nodes_de = _de_optimize(seq,
                                              t0_lo, t0_hi,
                                              tf_lo, tf_hi,
                                              deadline)
            if total_de < best_total:
                best_total, best_nodes, best_seq = total_de, nodes_de, seq

    # -------------------------------------------------------------
    # Jitter‑based hill‑climb on the current best plain solution
    # -------------------------------------------------------------
    if best_nodes is not None and best_seq is not None:
        jitter = JITTER_INIT
        while time.time() < deadline - 0.5:
            base_times = [node["time"] for node in best_nodes]
            cand = base_times.copy()
            n = len(cand)
            for i in range(1, n - 1):
                delta = (rng.random() - 0.5) * jitter
                new_t = cand[i] + delta
                if new_t - cand[i - 1] < MIN_LEG_DAYS:
                    new_t = cand[i - 1] + MIN_LEG_DAYS
                if cand[i + 1] - new_t < MIN_LEG_DAYS:
                    new_t = cand[i + 1] - MIN_LEG_DAYS
                cand[i] = new_t
            total, nodes = _optimise_times(best_seq, cand,
                                           t0_lo, t0_hi, tf_lo, tf_hi,
                                           deadline)
            if total < best_total:
                best_total, best_nodes = total, nodes
                jitter = max(jitter * 0.85, 0.4)
            else:
                jitter = max(jitter * 0.96, 0.4)

    # -------------------------------------------------------------
    # Fine‑grid coordinate descent
    # -------------------------------------------------------------
    if best_nodes is not None and best_seq is not None:
        total_grid, nodes_grid = _local_search(best_seq,
                                               [node["time"] for node in best_nodes],
                                               t0_lo, t0_hi, tf_lo, tf_hi,
                                               max_iter=16, step=0.14)
        if total_grid < best_total:
            best_total, best_nodes = total_grid, nodes_grid

    # -------------------------------------------------------------
    # High‑resolution NM polishing (if time remains)
    # -------------------------------------------------------------
    if best_nodes is not None and best_seq is not None and time.time() < deadline - 0.3:
        fine_total, fine_nodes = _optimise_times_fine(best_seq,
                                                     [node["time"] for node in best_nodes],
                                                     t0_lo, t0_hi, tf_lo, tf_hi,
                                                     deadline)
        if fine_total < best_total:
            best_total, best_nodes = fine_total, fine_nodes

    # -------------------------------------------------------------
    # Greedy GA insertion (if node budget permits)
    # -------------------------------------------------------------
    if best_nodes is not None and max_ga_allowed > 0:
        ga_improved = True
        while ga_improved and time.time() < deadline:
            ga_improved = False
            cur_seq = list(best_seq)
            cur_times = [node["time"] for node in best_nodes]
            for insert_pos in range(len(cur_seq) + 1):
                if time.time() >= deadline:
                    break
                for p in allowed_ga:
                    if (insert_pos > 0 and cur_seq[insert_pos - 1] == p) or \
                       (insert_pos < len(cur_seq) and cur_seq[insert_pos] == p):
                        continue
                    new_seq = cur_seq.copy()
                    new_seq.insert(insert_pos, p)
                    if len(new_seq) + 2 > max_nodes:
                        continue
                    t_before = cur_times[insert_pos]
                    t_after = cur_times[insert_pos + 1]
                    t_new = (t_before + t_after) * 0.5
                    new_times = cur_times.copy()
                    new_times.insert(insert_pos + 1, t_new)
                    total, nodes = _optimise_times(new_seq, new_times,
                                                  t0_lo, t0_hi, tf_lo, tf_hi,
                                                  deadline)
                    if total < best_total - 1e-6:
                        best_total, best_nodes, best_seq = total, nodes, tuple(new_seq)
                        ga_improved = True
                        break
                if ga_improved:
                    break

    # -------------------------------------------------------------
    # Greedy GA removal (prune unnecessary assists)
    # -------------------------------------------------------------
    if best_nodes is not None and best_seq is not None:
        removed = True
        while removed and time.time() < deadline:
            removed = False
            cur_seq = list(best_seq)
            cur_times = [node["time"] for node in best_nodes if node["type"] != "DSM"]
            for idx in range(len(cur_seq)):
                if time.time() >= deadline:
                    break
                new_seq = cur_seq.copy()
                del new_seq[idx]
                new_times = cur_times[:idx + 1] + cur_times[idx + 2:]
                total, nodes = _optimise_times(new_seq, new_times,
                                              t0_lo, t0_hi, tf_lo, tf_hi,
                                              deadline)
                if total < best_total - 1e-6:
                    best_total, best_nodes, best_seq = total, nodes, tuple(new_seq)
                    removed = True
                    break

    # -------------------------------------------------------------
    # Greedy DSM insertion (if budget permits)
    # -------------------------------------------------------------
    if best_nodes is not None and max_dsm_allowed > 0:
        remaining = max_dsm_allowed
        improved = True
        while improved and remaining > 0 and time.time() < deadline:
            improved = False
            cur_times = [node["time"] for node in best_nodes if node["type"] != "DSM"]
            for leg_i in range(len(best_seq) + 1):
                if time.time() >= deadline:
                    break
                for f in _DSM_FRACS:
                    total, nodes = _evaluate_sequence_with_dsm(best_seq,
                                                              cur_times,
                                                              leg_i, f)
                    if nodes is None:
                        continue
                    if len(nodes) > max_nodes:
                        continue
                    dsm_cnt = sum(1 for n in nodes if n["type"] == "DSM")
                    if dsm_cnt > max_dsm_allowed:
                        continue
                    if total < best_total - 1e-6:
                        best_total, best_nodes = total, nodes
                        remaining -= dsm_cnt
                        improved = True
                        break
                if improved:
                    break

    # -------------------------------------------------------------
    # Final thorough DSM refinement (if time allows)
    # -------------------------------------------------------------
    if best_nodes is not None and best_seq is not None and max_dsm_allowed > 0 and time.time() < deadline:
        cur_seq = best_seq
        cur_times = [node["time"] for node in best_nodes if node["type"] != "DSM"]
        for leg_i in range(len(cur_seq) + 1):
            if time.time() >= deadline:
                break
            refined_total, refined_nodes = _refine_dsm_fraction(
                cur_seq, cur_times, leg_i, 0.5, deadline)
            if refined_nodes is None:
                continue
            if len(refined_nodes) > max_nodes:
                continue
            dsm_cnt = sum(1 for n in refined_nodes if n["type"] == "DSM")
            if dsm_cnt > max_dsm_allowed:
                continue
            if refined_total < best_total - 1e-6:
                best_total, best_nodes = refined_total, refined_nodes

    # -------------------------------------------------------------
    # Fallback direct transfer if nothing feasible yet
    # -------------------------------------------------------------
    if best_nodes is None:
        fallback_start = (t0_lo + t0_hi) * 0.5
        fallback_end = (tf_lo + tf_hi) * 0.5
        best_total, best_nodes = _evaluate_sequence((), [fallback_start, fallback_end])

    # -------------------------------------------------------------
    # Prefer DSM candidate if it beats plain best
    # -------------------------------------------------------------
    if best_dsm_total < best_total:
        best_total = best_dsm_total
        best_nodes = best_dsm_nodes
        best_seq = None   # indicate DSM‑based solution

    record.set("final_nodes", len(best_nodes) if best_nodes else 0)
    record.event("solution_ready")
    return _format_nodes(best_nodes)
# EVOLVE-BLOCK-END
