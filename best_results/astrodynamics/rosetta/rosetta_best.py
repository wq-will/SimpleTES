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

import time
import itertools
import numpy as np
from scipy.optimize import minimize, differential_evolution
from functools import lru_cache

# -------------------------------------------------------------------------
# Global parameters – adapt to the allowed runtime (problem["timeout_seconds"])
# -------------------------------------------------------------------------
DAY = 86400.0                     # seconds per day
R_AU = 1.495978707e8            # km per AU
MIN_TOF = 2.0                    # minimum days between any two nodes
TOTAL_TIMEOUT = float(problem.get("timeout_seconds", 30.0))

# resolution / budget – tuned for the allocated time
if TOTAL_TIMEOUT > 240:
    GRID_LAUNCH = 80; GRID_ARRIVAL = 80; INTERIOR_SAMPLES = 250
    POOL_SIZE = 15000; TOP_REFINE = 300; MAX_NM_ITER = 6000
    MAX_LAMBERT_REV = 50; MAX_OPTIONS = 30; DP_BEAM = 250
elif TOTAL_TIMEOUT > 180:
    GRID_LAUNCH = 70; GRID_ARRIVAL = 70; INTERIOR_SAMPLES = 200
    POOL_SIZE = 12000; TOP_REFINE = 250; MAX_NM_ITER = 5000
    MAX_LAMBERT_REV = 45; MAX_OPTIONS = 25; DP_BEAM = 200
elif TOTAL_TIMEOUT > 120:
    GRID_LAUNCH = 60; GRID_ARRIVAL = 60; INTERIOR_SAMPLES = 150
    POOL_SIZE = 10000; TOP_REFINE = 200; MAX_NM_ITER = 4000
    MAX_LAMBERT_REV = 35; MAX_OPTIONS = 20; DP_BEAM = 150
elif TOTAL_TIMEOUT > 90:
    GRID_LAUNCH = 50; GRID_ARRIVAL = 50; INTERIOR_SAMPLES = 110
    POOL_SIZE = 8000; TOP_REFINE = 150; MAX_NM_ITER = 3000
    MAX_LAMBERT_REV = 30; MAX_OPTIONS = 15; DP_BEAM = 120
elif TOTAL_TIMEOUT > 60:
    GRID_LAUNCH = 40; GRID_ARRIVAL = 40; INTERIOR_SAMPLES = 80
    POOL_SIZE = 6000; TOP_REFINE = 120; MAX_NM_ITER = 2500
    MAX_LAMBERT_REV = 25; MAX_OPTIONS = 12; DP_BEAM = 90
else:
    GRID_LAUNCH = 30; GRID_ARRIVAL = 30; INTERIOR_SAMPLES = 50
    POOL_SIZE = 4000; TOP_REFINE = 80; MAX_NM_ITER = 2000
    MAX_LAMBERT_REV = 15; MAX_OPTIONS = 8; DP_BEAM = 60

# -------------------------------------------------------------------------
# Cached ephemerides (DE430)
# -------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _ephem_cached(pid: str, mjd: float):
    r, v = tools.ephem(pid, float(mjd))
    return tuple(r), tuple(v)

def _planet_state(pid: str, mjd: float, is_start=False, is_end=False):
    """Return heliocentric (r, v). Handles synthetic reference (pid '0')."""
    if pid == "0":
        if is_start and "state_r" in problem.get("start", {}):
            return (np.asarray(problem["start"]["state_r"], dtype=float),
                    np.asarray(problem["start"]["state_v"], dtype=float))
        if is_end and "state_r" in problem.get("end", {}):
            return (np.asarray(problem["end"]["state_r"], dtype=float),
                    np.asarray(problem["end"]["state_v"], dtype=float))
        return np.zeros(3), np.zeros(3)
    r, v = _ephem_cached(pid, float(mjd))
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float)

def _time_window(spec):
    if spec.get("kind") == "window":
        return float(spec["lo"]), float(spec["hi"])
    # exact moment
    val = float(spec.get("value", 0.0))
    return val, val

# -------------------------------------------------------------------------
# Piecewise‑linear interpolation (launch/capture curves)
# -------------------------------------------------------------------------
def _piecewise_linear(x, breakpoints):
    bp = sorted(breakpoints, key=lambda p: float(p[0]))
    if x <= float(bp[0][0]):
        return float(bp[0][1])
    if x >= float(bp[-1][0]):
        return float(bp[-1][1])
    for i in range(len(bp)-1):
        x0, y0 = float(bp[i][0]), bp[i][1]
        x1, y1 = float(bp[i+1][0]), bp[i+1][1]
        if x0 <= x <= x1:
            if x1 == x0:
                return float(y0)
            return float(y0 + (y1 - y0)*(x - x0)/(x1 - x0))
    return float(bp[-1][1])

# -------------------------------------------------------------------------
# Periapsis‑maneuver Δv helper (used in start/end boundaries)
# -------------------------------------------------------------------------
def _periapsis_dv(vinf, mu_p, R_p, h_factor, T_days):
    r_peri = R_p * (1.0 + h_factor)
    T_sec = T_days * DAY
    a_term = 2.0 * mu_p / r_peri
    term = (4.0*np.pi**2 * mu_p**2 / T_sec**2)**(1.0/3.0)
    return float(np.sqrt(vinf*vinf + a_term) -
                 np.sqrt(max(a_term - term, 0.0)))

def _boundary_dv(node, spec):
    typ = spec["type"]
    if typ == "piecewise_linear":
        dv_mag = float(np.linalg.norm(np.asarray(node["v_after"]) -
                                      np.asarray(node["v_before"])))
        return _piecewise_linear(dv_mag, spec["breakpoints"])
    elif typ == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        h = float(spec["h_factor"])
        T = float(spec["T_days"])
        _, v_pl = tools.ephem(pid, float(node["time"]))
        if node["type"] == "start":
            vinf = float(np.linalg.norm(np.asarray(node["v_after"]) - np.asarray(v_pl)))
        else:
            vinf = float(np.linalg.norm(np.asarray(node["v_before"]) - np.asarray(v_pl)))
        return _periapsis_dv(vinf, mu_p, R_p, h, T)
    else:
        return 1e9

# -------------------------------------------------------------------------
# Lambert solver wrapper (multi‑rev, pro/retro, low/high)
# -------------------------------------------------------------------------
def _lambert_solve(r0, r1, tof_sec, mu):
    for M in range(0, MAX_LAMBERT_REV + 1):
        for low in (True, False):
            for pro in (True, False):
                try:
                    v_dep, v_arr = tools.lambert(r0, r1, tof_sec, mu,
                                                prograde=pro, lowpath=low, M=M)
                    return np.asarray(v_dep), np.asarray(v_arr)
                except Exception:
                    continue
    raise RuntimeError("Lambert failure")

def _lambert_options(r0, r1, tof_sec, mu, max_options=MAX_OPTIONS):
    """Return a short list of Lambert alternatives for DP."""
    sols = []
    cnt = 0
    # allow a richer set of revolutions for option generation
    max_rev = min(MAX_LAMBERT_REV, 20)
    for M in range(0, max_rev + 1):
        for low in (True, False):
            for pro in (True, False):
                try:
                    v_dep, v_arr = tools.lambert(r0, r1, tof_sec, mu,
                                                prograde=pro, lowpath=low, M=M)
                    sols.append((np.asarray(v_dep), np.asarray(v_arr)))
                    cnt += 1
                    if cnt >= max_options:
                        return sols
                except Exception:
                    continue
    return sols

# -------------------------------------------------------------------------
# Powered‑flyby helper (returns mismatch Δv and feasibility)
# -------------------------------------------------------------------------
def _powered_flyby_dv(v_arr, v_out, pid, t, prob):
    pid = str(pid)
    mu_p = float(prob["planet_mu"][pid])
    R_p = float(prob["planet_radius"][pid])
    min_alt = float(prob.get("flyby", {})
                     .get("min_altitude_km", {})
                     .get(pid, 200))
    _, v_pl = tools.ephem(pid, float(t))
    try:
        _, dv, feas = tools.powered_flyby(np.asarray(v_arr),
                                          np.asarray(v_out),
                                          v_pl, mu_p, R_p + min_alt)
        return float(dv), bool(feas)
    except Exception:
        return float("inf"), False

# -------------------------------------------------------------------------
# GA sequence generators
# -------------------------------------------------------------------------
def _ga_sequences(allowed, n_ga, limit=None):
    """Return all length‑n_ga sequences (or a random subset if limited)."""
    if n_ga == 0:
        return [()]
    combos = list(itertools.product(allowed, repeat=n_ga))
    if limit is not None and len(combos) > limit:
        np.random.shuffle(combos)
        combos = combos[:limit]
    return [list(seq) for seq in combos]

def _heuristic_patterns(allowed_set, max_len):
    """Generate a handful of commonsense patterns for inner‑planet tours."""
    patterns = []

    # single‑planet repeats
    for pid in allowed_set:
        for n in range(1, max_len+1):
            patterns.append([pid]*n)

    # alternating pairs (E‑V, V‑E, etc.)
    for a, b in itertools.combinations(allowed_set, 2):
        # basic AB and BA
        patterns.append([a, b])
        patterns.append([b, a])
        # longer alternations
        alt = [a, b] * ((max_len // 2) + 1)
        patterns.append(alt[:max_len])
        alt2 = [b, a] * ((max_len // 2) + 1)
        patterns.append(alt2[:max_len])

    # three‑planet permutations (if we have at least three)
    if len(allowed_set) >= 3:
        base = list(allowed_set)[:3]
        for perm in itertools.permutations(base, max_len):
            patterns.append(list(perm))

    # deduplicate
    uniq = []
    seen = set()
    for seq in patterns:
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            uniq.append(seq)
    return uniq

# -------------------------------------------------------------------------
# Evaluate a GA‑only trajectory (single‑lambert per leg)
# -------------------------------------------------------------------------
def _evaluate_ga_trajectory(ga_seq, times, best_limit=float('inf')):
    max_nodes = int(problem.get("max_nodes", 99))
    if len(ga_seq) + 2 > max_nodes:
        return float('inf'), None

    start_spec = problem["start"]
    end_spec   = problem["end"]
    start_pid = str(start_spec.get("planet_id","0"))
    end_pid   = str(end_spec.get("planet_id","0"))
    mu_sun = problem["mu_sun"]
    pid_seq = [start_pid] + list(ga_seq) + [end_pid]

    # planetary states
    wp_r, wp_v = [], []
    for idx, (pid, t) in enumerate(zip(pid_seq, times)):
        is_start = (idx == 0)
        is_end   = (idx == len(times)-1)
        r, v = _planet_state(pid, t, is_start=is_start, is_end=is_end)
        wp_r.append(r); wp_v.append(v)

    # Lambert for each leg (first viable solution)
    leg_dep, leg_arr = [], []
    for i in range(len(times)-1):
        tof = (times[i+1] - times[i]) * DAY
        if tof <= 0.0:
            return float('inf'), None
        try:
            v_dep, v_arr = _lambert_solve(wp_r[i], wp_r[i+1], tof, mu_sun)
        except Exception:
            return float('inf'), None
        leg_dep.append(v_dep); leg_arr.append(v_arr)

    total = 0.0
    nodes = []

    # start node
    start_node = {
        "type":"start",
        "time":float(times[0]),
        "planet_id":start_pid,
        "r":wp_r[0].tolist(),
        "v_before":wp_v[0].tolist(),
        "v_after":leg_dep[0].tolist(),
    }
    total += _boundary_dv(start_node, start_spec)
    if total > best_limit:
        return float('inf'), None
    nodes.append(start_node)

    # GA nodes
    for i, ga_pid in enumerate(ga_seq):
        v_in = leg_arr[i]
        v_out = leg_dep[i+1]
        dv_ga, feas = _powered_flyby_dv(v_in, v_out, ga_pid,
                                         times[i+1], problem)
        if not feas:
            return float('inf'), None
        total += dv_ga
        if total > best_limit:
            return float('inf'), None
        nodes.append({
            "type":"GA",
            "time":float(times[i+1]),
            "planet_id":str(ga_pid),
            "r":wp_r[i+1].tolist(),
            "v_before":v_in.tolist(),
            "v_after":v_out.tolist(),
        })

    # end node
    end_node = {
        "type":"end",
        "time":float(times[-1]),
        "planet_id":end_pid,
        "r":wp_r[-1].tolist(),
        "v_before":leg_arr[-1].tolist(),
        "v_after":wp_v[-1].tolist(),
    }
    total += _boundary_dv(end_node, end_spec)
    if total > best_limit:
        return float('inf'), None
    nodes.append(end_node)

    return total, nodes

# -------------------------------------------------------------------------
# DP‑based multi‑option evaluation (beam search)
# -------------------------------------------------------------------------
def _evaluate_ga_trajectory_multi(ga_seq, times, best_limit=float('inf')):
    max_nodes = int(problem.get("max_nodes", 99))
    if len(ga_seq) + 2 > max_nodes:
        return float('inf'), None

    start_spec = problem["start"]
    end_spec   = problem["end"]
    start_pid = str(start_spec.get("planet_id","0"))
    end_pid   = str(end_spec.get("planet_id","0"))
    mu_sun = problem["mu_sun"]
    pid_seq = [start_pid] + list(ga_seq) + [end_pid]

    wp_r, wp_v = [], []
    for idx, (pid, t) in enumerate(zip(pid_seq, times)):
        is_start = (idx == 0)
        is_end   = (idx == len(times)-1)
        r, v = _planet_state(pid, t, is_start=is_start, is_end=is_end)
        wp_r.append(r); wp_v.append(v)

    # collect Lambert options per leg
    leg_options = []
    for i in range(len(times)-1):
        tof = (times[i+1] - times[i]) * DAY
        if tof <= 0.0:
            return float('inf'), None
        opts = _lambert_options(wp_r[i], wp_r[i+1], tof, mu_sun,
                                 max_options=MAX_OPTIONS)
        if not opts:
            return float('inf'), None
        leg_options.append(opts)

    # DP – keep a beam of best partial trajectories
    states = []
    for v_dep0, v_arr0 in leg_options[0]:
        start_node = {
            "type":"start",
            "time":float(times[0]),
            "planet_id":start_pid,
            "r":wp_r[0].tolist(),
            "v_before":wp_v[0].tolist(),
            "v_after":v_dep0.tolist(),
        }
        cost0 = _boundary_dv(start_node, start_spec)
        if cost0 >= best_limit:
            continue
        states.append((v_arr0, cost0, [start_node]))
    if not states:
        return float('inf'), None
    states.sort(key=lambda x: x[1])
    states = states[:DP_BEAM]

    n_legs = len(leg_options) - 1
    for i in range(n_legs):
        ga_pid = ga_seq[i]
        ga_time = times[i+1]
        ga_r = wp_r[i+1]
        next_opts = leg_options[i+1]
        new_states = []
        for v_arr_i, cost_i, nodes_i in states:
            for v_dep_next, v_arr_next in next_opts:
                dv_ga, feas = _powered_flyby_dv(v_arr_i, v_dep_next,
                                                ga_pid, ga_time, problem)
                if not feas:
                    continue
                new_cost = cost_i + dv_ga
                if new_cost >= best_limit:
                    continue
                ga_node = {
                    "type":"GA",
                    "time":float(ga_time),
                    "planet_id":str(ga_pid),
                    "r":ga_r.tolist(),
                    "v_before":v_arr_i.tolist(),
                    "v_after":v_dep_next.tolist(),
                }
                new_states.append((v_arr_next, new_cost,
                                   nodes_i + [ga_node]))
        if not new_states:
            return float('inf'), None
        new_states.sort(key=lambda x: x[1])
        states = new_states[:DP_BEAM]

    # finalize with end node
    best_total = float('inf')
    best_nodes = None
    for v_arr_last, cost_last, nodes_last in states:
        end_node = {
            "type":"end",
            "time":float(times[-1]),
            "planet_id":end_pid,
            "r":wp_r[-1].tolist(),
            "v_before":v_arr_last.tolist(),
            "v_after":wp_v[-1].tolist(),
        }
        total = cost_last + _boundary_dv(end_node, end_spec)
        if total < best_total:
            best_total = total
            best_nodes = nodes_last + [end_node]
    return best_total, best_nodes

# -------------------------------------------------------------------------
# Compute total Δv (including DSMs)
# -------------------------------------------------------------------------
def _compute_total_dv(nodes):
    if not nodes:
        return float('inf')
    total = 0.0
    for n in nodes:
        typ = n["type"]
        if typ == "start":
            total += _boundary_dv(n, problem["start"])
        elif typ == "end":
            total += _boundary_dv(n, problem["end"])
        elif typ == "GA":
            dv, feas = _powered_flyby_dv(np.asarray(n["v_before"]),
                                         np.asarray(n["v_after"]),
                                         n["planet_id"], n["time"], problem)
            if not feas:
                return float('inf')
            total += dv
        elif typ == "DSM":
            total += float(np.linalg.norm(np.asarray(n["v_after"]) -
                                          np.asarray(n["v_before"])))
    return total

# -------------------------------------------------------------------------
# Interior GA epoch optimisation (Nelder‑Mead)
# -------------------------------------------------------------------------
def _optimize_ga_times(ga_seq, t0, tf, interior_initial, best_limit=float('inf')):
    if not interior_initial:
        total, nodes = _evaluate_ga_trajectory(ga_seq, [t0, tf])
        return total, [t0, tf], nodes

    x0 = np.asarray(interior_initial, dtype=float)

    def obj(xx):
        interior = np.sort(xx)
        interior = np.clip(interior, t0 + MIN_TOF, tf - MIN_TOF)
        times = [float(t0)] + interior.tolist() + [float(tf)]
        total, _ = _evaluate_ga_trajectory(ga_seq, times)
        return total

    try:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": MAX_NM_ITER,
                                "fatol": 1e-3, "xatol": 1e-3})
    except Exception:
        total, nodes = _evaluate_ga_trajectory(ga_seq,
                                               [t0] + list(interior_initial) + [tf])
        return total, [t0] + list(interior_initial) + [tf], nodes

    interior_opt = np.sort(np.clip(res.x, t0 + MIN_TOF, tf - MIN_TOF))
    times_opt = [float(t0)] + interior_opt.tolist() + [float(tf)]
    total_opt, nodes_opt = _evaluate_ga_trajectory(ga_seq, times_opt)
    if nodes_opt is None:
        total_opt, nodes_opt = _evaluate_ga_trajectory(ga_seq,
                                                       [t0] + list(interior_initial) + [tf])
        times_opt = [float(t0)] + list(interior_initial) + [float(tf)]
    return total_opt, times_opt, nodes_opt

# -------------------------------------------------------------------------
# GA sequence mutation utilities (local neighbourhood)
# -------------------------------------------------------------------------
def _mutate_sequence(seq, allowed):
    mutated = set()
    n = len(seq)
    for i in range(n):
        for pid in allowed:
            if pid != seq[i]:
                new_seq = list(seq)
                new_seq[i] = pid
                mutated.add(tuple(new_seq))
    for i in range(n):
        for j in range(i+1, n):
            new_seq = list(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            mutated.add(tuple(new_seq))
    max_ga = int(problem.get("max_GA", 0))
    max_nodes = int(problem.get("max_nodes", 99))
    if n < max_ga and (2 + n + 1) <= max_nodes:
        for pid in allowed:
            mutated.add(tuple(list(seq) + [pid]))
    if n > 0:
        for i in range(n):
            new_seq = list(seq)
            del new_seq[i]
            mutated.add(tuple(new_seq))
    return [list(m) for m in mutated]

# -------------------------------------------------------------------------
# Global optimisation of all epochs (Nelder‑Mead + DE)
# -------------------------------------------------------------------------
def _optimize_all_times(ga_seq, start_spec, end_spec,
                        times_initial, time_budget):
    if not ga_seq:
        return None, None
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    x0 = np.asarray(times_initial, dtype=float)

    best_total = float('inf')
    best_nodes = None
    start_clock = time.time()

    def obj(xx):
        t0 = float(np.clip(xx[0], t0_lo, t0_hi))
        tf = float(np.clip(xx[1], tf_lo, tf_hi))
        interior = np.sort(xx[2:])
        interior = np.clip(interior, t0 + MIN_TOF, tf - MIN_TOF)
        times = [t0] + interior.tolist() + [tf]
        total, _ = _evaluate_ga_trajectory(ga_seq, times)
        return total

    # Initial NM pass
    try:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": max(200, MAX_NM_ITER//2),
                                "fatol": 1e-3, "xatol": 1e-3})
        t0_opt = float(np.clip(res.x[0], t0_lo, t0_hi))
        tf_opt = float(np.clip(res.x[1], tf_lo, tf_hi))
        interior_opt = np.sort(res.x[2:])
        interior_opt = np.clip(interior_opt, t0_opt + MIN_TOF, tf_opt - MIN_TOF)
        times_opt = [t0_opt] + interior_opt.tolist() + [tf_opt]
        total_opt, nodes_opt = _evaluate_ga_trajectory(ga_seq, times_opt)
        if nodes_opt is not None and total_opt < best_total:
            best_total = total_opt
            best_nodes = nodes_opt
            x0 = res.x
    except Exception:
        pass

    # Random NM restarts until time budget exhausted
    while time.time() - start_clock < time_budget:
        perturb = np.random.randn(len(x0)) * np.array([5.0,5.0] + [10.0]*(len(x0)-2))
        guess = x0 + perturb
        try:
            res = minimize(obj, guess, method="Nelder-Mead",
                           options={"maxiter": max(100, MAX_NM_ITER//4),
                                    "fatol": 1e-3, "xatol": 1e-3})
            t0_opt = float(np.clip(res.x[0], t0_lo, t0_hi))
            tf_opt = float(np.clip(res.x[1], tf_lo, tf_hi))
            interior_opt = np.sort(res.x[2:])
            interior_opt = np.clip(interior_opt, t0_opt + MIN_TOF, tf_opt - MIN_TOF)
            times_opt = [t0_opt] + interior_opt.tolist() + [tf_opt]
            total_opt, nodes_opt = _evaluate_ga_trajectory(ga_seq, times_opt)
            if nodes_opt is not None and total_opt < best_total:
                best_total = total_opt
                best_nodes = nodes_opt
                x0 = res.x
        except Exception:
            continue

    # Differential Evolution (short) – if any time left (handled by caller)
    return _format_nodes(best_nodes) if best_nodes is not None else None, best_total

# -------------------------------------------------------------------------
# Short DE optimisation of all epochs (used as final polish)
# -------------------------------------------------------------------------
def _global_de_optimize(ga_seq, times_initial, time_budget):
    if not ga_seq:
        return None, None
    start_spec = problem["start"]
    end_spec = problem["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    n_vars = len(times_initial)
    bounds = []
    for i in range(n_vars):
        if i == 0:
            bounds.append((t0_lo, t0_hi))
        elif i == n_vars-1:
            bounds.append((tf_lo, tf_hi))
        else:
            bounds.append((t0_lo, tf_hi))

    def obj(xx):
        t0 = float(np.clip(xx[0], t0_lo, t0_hi))
        tf = float(np.clip(xx[-1], tf_lo, tf_hi))
        interior = np.sort(xx[1:-1])
        interior = np.clip(interior, t0 + MIN_TOF, tf - MIN_TOF)
        times = [t0] + interior.tolist() + [tf]
        total, nodes = _evaluate_ga_trajectory(ga_seq, times)
        if nodes is None:
            return 1e9
        return total

    try:
        result = differential_evolution(obj, bounds,
                                        maxiter=24, polish=False,
                                        seed=42, updating='deferred')
    except Exception:
        return None, None

    xx = result.x
    t0 = float(np.clip(xx[0], t0_lo, t0_hi))
    tf = float(np.clip(xx[-1], tf_lo, tf_hi))
    interior = np.sort(xx[1:-1])
    interior = np.clip(interior, t0 + MIN_TOF, tf - MIN_TOF)
    times_opt = [t0] + interior.tolist() + [tf]
    total_opt, nodes_opt = _evaluate_ga_trajectory(ga_seq, times_opt)
    if nodes_opt is None:
        return None, None
    return _format_nodes(nodes_opt), total_opt

# -------------------------------------------------------------------------
# Refine launch/arrival windows (random + NM)
# -------------------------------------------------------------------------
def _refine_start_end_times(ga_seq, start_spec, end_spec,
                            best_nodes, best_total, time_budget):
    if time_budget <= 0.0:
        return best_nodes, best_total
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    ga_times = [n["time"] for n in best_nodes if n["type"] == "GA"]
    interior_frac = []
    if ga_times:
        t0_cur = best_nodes[0]["time"]
        tf_cur = best_nodes[-1]["time"]
        span = tf_cur - t0_cur
        if span > 0:
            interior_frac = [(t - t0_cur) / span for t in ga_times]

    deadline = time.time() + time_budget
    best_nodes_loc = best_nodes
    best_total_loc = best_total

    while time.time() < deadline:
        t0 = np.random.uniform(t0_lo, t0_hi)
        min_arrival = t0 + MIN_TOF * (len(ga_seq) + 1)
        if tf_hi < min_arrival:
            continue
        tf = np.random.uniform(max(tf_lo, min_arrival), tf_hi)

        if interior_frac:
            interior_init = [t0 + f * (tf - t0) for f in interior_frac]
        else:
            interior_init = []

        total_opt, times_opt, nodes_opt = _optimize_ga_times(
            ga_seq, t0, tf, interior_init, best_limit=best_total_loc)
        if nodes_opt is None:
            continue
        total_opt = _compute_total_dv(nodes_opt)
        if total_opt < best_total_loc:
            best_total_loc = total_opt
            best_nodes_loc = nodes_opt

    return best_nodes_loc, best_total_loc

# -------------------------------------------------------------------------
# DSM insertion & joint optimisation (optional)
# -------------------------------------------------------------------------
def _add_dsms_and_optimize(base_nodes, prob_dict, time_budget):
    if base_nodes is None:
        return None
    max_dsm = int(prob_dict.get("max_DSM", 0))
    max_nodes = int(prob_dict.get("max_nodes", 99))
    n_ga = sum(1 for n in base_nodes if n["type"] == "GA")
    n_legs = n_ga + 1
    if max_dsm < n_legs or (2 + n_ga + n_legs) > max_nodes:
        return None

    mu_sun = prob_dict["mu_sun"]
    start_spec = prob_dict["start"]
    end_spec = prob_dict["end"]
    start_pid = str(start_spec.get("planet_id","0"))
    end_pid   = str(end_spec.get("planet_id","0"))

    # collect fixed waypoint times/positions
    base_times = []
    base_r = []
    ga_pids = []
    for n in base_nodes:
        if n["type"] in ("start","GA","end"):
            base_times.append(float(n["time"]))
            base_r.append(np.asarray(n["r"], dtype=float))
        if n["type"] == "GA":
            ga_pids.append(str(n["planet_id"]))

    # initial guess vector: (t_dsm, x, y, z) per leg + mutable GA epochs
    x0 = []
    for i in range(n_legs):
        ti, tj = base_times[i], base_times[i+1]
        ri, rj = base_r[i], base_r[i+1]
        t_mid = ti + (tj - ti) * 0.5
        x0.append(t_mid)

        try:
            v_full, _ = _lambert_solve(ri, rj, (tj - ti) * DAY, mu_sun)
            r_mid, _ = tools.propagate_two_body(ri, v_full,
                                                (tj - ti) * DAY * 0.5,
                                                mu_sun)
        except Exception:
            r_mid = (ri + rj) * 0.5
        x0.extend([float(r_mid[0]), float(r_mid[1]), float(r_mid[2])])

    # mutable GA epochs
    for k in range(n_ga):
        x0.append(float(base_times[k+1]))
    x0 = np.asarray(x0, dtype=float)

    n_vars = len(x0)
    n_per_leg = 4   # (t_dsm, x, y, z)

    def _eval(xx):
        total = 0.0
        t_seq = list(base_times)
        r_seq = [np.asarray(r, dtype=float) for r in base_r]

        # slide GA epochs
        for k in range(n_ga):
            idx = n_per_leg * n_legs + k
            lo = t_seq[k] + MIN_TOF
            hi = (t_seq[k+2] if k+2 < len(t_seq) else t_seq[-1]) - MIN_TOF
            t_seq[k+1] = float(np.clip(xx[idx],
                                         lo,
                                         max(lo+0.1, hi)))
            r_seq[k+1], _ = _planet_state(ga_pids[k], t_seq[k+1])

        # reference start state
        r_seq[0], v_ref_start = _planet_state(start_pid, t_seq[0], is_start=True)

        legs = []
        for i in range(n_legs):
            ti, tj = t_seq[i], t_seq[i+1]
            ri, rj = r_seq[i], r_seq[i+1]

            # DSM variables
            t_dsm = float(np.clip(xx[n_per_leg*i],
                                 ti + MIN_TOF,
                                 tj - MIN_TOF))
            r_dsm = np.array([xx[n_per_leg*i+1],
                              xx[n_per_leg*i+2],
                              xx[n_per_leg*i+3]])

            # i → DSM
            tof_a = max(0.1, (t_dsm - ti) * DAY)
            try:
                v_dep_i, v_arr_dsm = _lambert_solve(ri, r_dsm, tof_a, mu_sun)
            except Exception:
                return float('inf'), None

            # DSM → j
            tof_b = max(0.1, (tj - t_dsm) * DAY)
            try:
                v_dep_dsm, v_arr_j = _lambert_solve(r_dsm, rj, tof_b, mu_sun)
            except Exception:
                return float('inf'), None

            dv_dsm = float(np.linalg.norm(v_dep_dsm - v_arr_dsm))
            total += dv_dsm

            legs.append({
                "t_i": ti, "t_j": tj,
                "r_i": ri, "r_j": rj,
                "t_dsm": t_dsm, "r_dsm": r_dsm,
                "v_dep_i": v_dep_i, "v_arr_dsm": v_arr_dsm,
                "v_dep_dsm": v_dep_dsm, "v_arr_j": v_arr_j,
                "dv_dsm": dv_dsm
            })

        # Assemble node list
        result_nodes = []

        # start node
        start_node = {
            "type":"start",
            "time":float(t_seq[0]),
            "planet_id":start_pid,
            "r":r_seq[0].tolist(),
            "v_before":v_ref_start.tolist(),
            "v_after":legs[0]["v_dep_i"].tolist(),
        }
        total += _boundary_dv(start_node, start_spec)
        result_nodes.append(start_node)

        for i in range(n_legs):
            leg = legs[i]

            # DSM node
            result_nodes.append({
                "type":"DSM",
                "time":float(leg["t_dsm"]),
                "planet_id":"0",
                "r":leg["r_dsm"].tolist(),
                "v_before":leg["v_arr_dsm"].tolist(),
                "v_after":leg["v_dep_dsm"].tolist(),
            })

            # GA after this leg (if any)
            if i < n_ga:
                v_in = leg["v_arr_j"]
                v_out = legs[i+1]["v_dep_i"]
                ga_pid = ga_pids[i]
                ga_time = t_seq[i+1]
                dv_ga, feas = _powered_flyby_dv(v_in, v_out, ga_pid,
                                                ga_time, prob_dict)
                if not feas:
                    return float('inf'), None
                total += dv_ga
                result_nodes.append({
                    "type":"GA",
                    "time":float(ga_time),
                    "planet_id":str(ga_pid),
                    "r":r_seq[i+1].tolist(),
                    "v_before":v_in.tolist(),
                    "v_after":v_out.tolist(),
                })

        # end node
        _, v_ref_end = _planet_state(end_pid, t_seq[-1], is_end=True)
        end_node = {
            "type":"end",
            "time":float(t_seq[-1]),
            "planet_id":end_pid,
            "r":r_seq[-1].tolist(),
            "v_before":legs[-1]["v_arr_j"].tolist(),
            "v_after":v_ref_end.tolist(),
        }
        total += _boundary_dv(end_node, prob_dict["end"])
        result_nodes.append(end_node)

        return total, result_nodes

    # Primary NM optimisation
    dv0, nodes0 = _eval(x0)
    if dv0 >= 1e8 or nodes0 is None:
        return None

    # Simple simplex scaling for NM
    scales = []
    for _ in range(n_legs):
        scales.extend([5.0,
                       0.03*R_AU, 0.03*R_AU, 0.03*R_AU])
    for _ in range(n_ga):
        scales.append(5.0)

    def _run_nm(start_vec):
        try:
            res = minimize(lambda xx: _eval(xx)[0],
                           start_vec,
                           method="Nelder-Mead",
                           options={"maxiter": max(150, 180*len(start_vec)),
                                    "fatol": 1e-6, "xatol": 1e-4,
                                    "initial_simplex": np.array(
                                        [start_vec] +
                                        [start_vec + np.eye(len(start_vec))[j] *
                                         (scales[j] if j < len(scales) else 1.0)
                                         for j in range(len(start_vec))][:len(start_vec)]
                                    )})
            return res.x, _eval(res.x)[0], _eval(res.x)[1]
        except Exception:
            return None, float('inf'), None

    vec, dv_best, nodes_best = _run_nm(x0)
    if nodes_best is not None and dv_best < dv0:
        best_dv, best_nodes = dv_best, nodes_best
        x0 = vec
    else:
        best_dv, best_nodes = dv0, nodes0

    restart_start = time.time()
    while time.time() - restart_start < time_budget:
        perturb = np.random.randn(n_vars) * np.array(scales) * 0.2
        guess = x0 + perturb
        for i in range(n_legs):
            ti, tj = base_times[i], base_times[i+1]
            guess[n_per_leg*i] = np.clip(guess[n_per_leg*i],
                                           ti + MIN_TOF,
                                           tj - MIN_TOF)
        vec2, dv2, nodes2 = _run_nm(guess)
        if nodes2 is not None and dv2 < best_dv:
            best_dv, best_nodes = dv2, nodes2
            x0 = vec2

    return _format_nodes(best_nodes)

# -------------------------------------------------------------------------
# Convert possible NumPy arrays in nodes to plain Python lists
# -------------------------------------------------------------------------
def _format_nodes(nodes):
    if nodes is None:
        return None
    for n in nodes:
        for k in ("r","v_before","v_after"):
            n[k] = np.asarray(n[k], dtype=float).tolist()
    return nodes

# -------------------------------------------------------------------------
# Heuristic initial search (quick low‑dv seed)
# -------------------------------------------------------------------------
def _heuristic_initial_search():
    """Try a handful of classic GA patterns with DP refinement."""
    start_spec = problem["start"]
    end_spec = problem["end"]
    allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga_allowed = min(int(problem.get("max_GA", 0)),
                         int(problem.get("max_nodes", 99)) - 2)
    if max_ga_allowed <= 0 or not allowed:
        return None

    patterns = _heuristic_patterns(set(allowed), max_ga_allowed)

    # family‑specific hints (e.g., V‑E‑V, V‑E‑E‑V for Rosetta‑type)
    fam = problem.get("reference_family", "").lower()
    name = problem.get("mission_name", "").lower()
    if any(k in fam for k in ("rosetta","cassini","jupiter","saturn")):
        if {"2","3"}.issubset(set(allowed)):
            patterns.append(["2","3"])
            patterns.append(["3","2"])
            patterns.append(["2","3","2"])
            patterns.append(["3","2","3"])
        if {"2","3","5"}.issubset(set(allowed)):
            patterns.append(["2","3","5"])
            patterns.append(["5","3","2"])

    # deduplicate (already done inside helper but re‑ensure)
    uniq = []
    seen = set()
    for seq in patterns:
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            uniq.append(seq)

    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])

    best_nodes = None
    best_total = float('inf')
    target = float(problem.get("target_dv", 0.0))

    # coarse grids for launch/arrival
    launch_grid = np.linspace(t0_lo, t0_hi, num=max(3, GRID_LAUNCH//4))
    arrival_grid = np.linspace(tf_lo, tf_hi, num=max(3, GRID_ARRIVAL//4))

    for seq in uniq:
        n = len(seq)
        if n == 0:
            continue
        min_span = MIN_TOF * (n + 1)
        for t0 in launch_grid:
            if t0 + min_span > tf_hi:
                continue
            tf_min = max(tf_lo, t0 + min_span)
            for tf in arrival_grid:
                if tf < tf_min:
                    continue
                # equal‑spaced interior epochs (good starting guess)
                times_eq = [float(t0)]
                for i in range(n):
                    times_eq.append(float(t0 + (i+1)*(tf - t0)/(n+1)))
                times_eq.append(float(tf))
                total, nodes = _evaluate_ga_trajectory_multi(seq, times_eq, best_limit=best_total)
                if nodes is not None and total < best_total:
                    best_total, best_nodes = total, nodes
                if target > 0 and best_total <= 0.9 * target:
                    return best_nodes
    return best_nodes

# -------------------------------------------------------------------------
# Deterministic grid search (cheap seed)
# -------------------------------------------------------------------------
def _grid_search_ga(budget_seconds):
    start_spec = problem["start"]
    end_spec = problem["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    start_pid = str(start_spec.get("planet_id","0"))
    end_pid   = str(end_spec.get("planet_id","0"))
    allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga = min(int(problem.get("max_GA", 0)),
                 int(problem.get("max_nodes", 99)) - 2)

    launch_grid = np.linspace(t0_lo, t0_hi, GRID_LAUNCH)
    arrival_grid = np.linspace(tf_lo, tf_hi, GRID_ARRIVAL)

    best = None
    best_total = float('inf')
    start_time = time.time()
    target = float(problem.get("target_dv", 0.0))

    for n_ga in range(max_ga, -1, -1):
        if 2 + n_ga > int(problem.get("max_nodes", 99)):
            continue
        ga_combos = _ga_sequences(allowed, n_ga, limit=None)
        min_span = MIN_TOF * (n_ga + 1)

        for ga_seq in ga_combos:
            for t0 in launch_grid:
                if t0 + min_span > tf_hi:
                    continue
                tf_min = max(tf_lo, t0 + min_span)
                for tf in arrival_grid:
                    if tf < tf_min:
                        continue

                    # deterministic equal‑spacing interior times
                    if n_ga > 0:
                        times_eq = [float(t0)]
                        for i in range(n_ga):
                            times_eq.append(float(t0 + (i+1)*(tf - t0)/(n_ga+1)))
                        times_eq.append(float(tf))
                        total, nodes = _evaluate_ga_trajectory_multi(ga_seq, times_eq, best_limit=best_total)
                        if nodes is not None and total < best_total:
                            best_total, best = total, nodes
                        if target > 0 and best_total <= 0.9 * target:
                            return best
                    else:
                        # direct transfer (no GA)
                        total, nodes = _evaluate_ga_trajectory([], [float(t0), float(tf)], best_limit=best_total)
                        if nodes is not None and total < best_total:
                            best_total, best = total, nodes
                        if target > 0 and best_total <= 0.9 * target:
                            return best

                    # random interior samplings (lighter evaluation)
                    for _ in range(INTERIOR_SAMPLES):
                        if n_ga == 0:
                            times = [float(t0), float(tf)]
                        else:
                            fracs = np.sort(np.random.rand(n_ga))
                            times = [float(t0)]
                            times.extend([t0 + f * (tf - t0) for f in fracs])
                            times.append(float(tf))
                        total, nodes = _evaluate_ga_trajectory(ga_seq, times, best_total)
                        if nodes is None:
                            continue
                        if total < best_total:
                            best_total, best = total, nodes
                        if target > 0 and best_total <= 0.9 * target:
                            return best
                if time.time() - start_time > budget_seconds:
                    return best if best is not None else []
    # fallback direct transfer
    if best is None:
        try:
            t0 = (t0_lo + t0_hi) * 0.5
            tf = (tf_lo + tf_hi) * 0.5
            r0, v0 = _planet_state(start_pid, t0, is_start=True)
            r1, v1 = _planet_state(end_pid, tf, is_end=True)
            v_dep, v_arr = _lambert_solve(r0, r1, (tf - t0) * DAY, problem["mu_sun"])
            best = [
                {"type":"start","time":float(t0),"planet_id":start_pid,
                 "r":r0.tolist(),"v_before":v0.tolist(),"v_after":v_dep.tolist()},
                {"type":"end","time":float(tf),"planet_id":end_pid,
                 "r":r1.tolist(),"v_before":v_arr.tolist(),"v_after":v1.tolist()}
            ]
        except Exception:
            best = []
    return best

# -------------------------------------------------------------------------
# Stochastic pool of GA candidates
# -------------------------------------------------------------------------
def _random_search_ga_pool(budget_seconds, pool_size=POOL_SIZE):
    start_spec = problem["start"]
    end_spec = problem["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga_allowed = min(int(problem.get("max_GA", 0)),
                         int(problem.get("max_nodes", 99)) - 2)

    pool = []
    deadline = time.time() + budget_seconds
    best_seen = float('inf')
    target = float(problem.get("target_dv", 0.0))

    while time.time() < deadline:
        n_ga = np.random.randint(0, max_ga_allowed + 1)
        ga_seq = [np.random.choice(allowed) for _ in range(n_ga)] if n_ga > 0 else []
        t0 = np.random.uniform(t0_lo, t0_hi)

        min_arrival = t0 + MIN_TOF * (n_ga + 1)
        if tf_hi < min_arrival:
            continue
        tf = np.random.uniform(max(tf_lo, min_arrival), tf_hi)

        if n_ga:
            fracs = np.sort(np.random.rand(n_ga))
            times = [float(t0)]
            times.extend([t0 + f * (tf - t0) for f in fracs])
            times.append(float(tf))
        else:
            times = [float(t0), float(tf)]

        total, nodes = _evaluate_ga_trajectory(ga_seq, times, best_seen)
        if nodes is None:
            continue
        if total < best_seen:
            best_seen = total
        if len(pool) < pool_size:
            pool.append((total, tuple(ga_seq), times, nodes))
            pool.sort(key=lambda x: x[0])
        else:
            if total < pool[-1][0]:
                pool[-1] = (total, tuple(ga_seq), times, nodes)
                pool.sort(key=lambda x: x[0])
        if target > 0 and best_seen <= 0.9 * target:
            break
    return pool

# -------------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------------
def run_code():
    record.event(f"mission={problem.get('id','unknown')} search_start")
    timeout = float(problem.get("timeout_seconds", 30.0))
    start_time = time.time()

    # Phase 1 – deterministic grid (quick seed)
    grid_budget = max(2.0, timeout * 0.12)
    grid_nodes = _grid_search_ga(grid_budget)

    # Phase 1b – heuristic patterns (mission‑specific seed)
    heuristic_nodes = _heuristic_initial_search()

    # Phase 2 – stochastic pool (diverse candidates)
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    pool_nodes = []
    if remaining > 6.0:
        pool_budget = remaining * 0.60
        pool = _random_search_ga_pool(pool_budget, POOL_SIZE)
        for total, ga_seq, times, nodes in pool:
            pool_nodes.append((nodes, total))

    # Assemble candidate list
    candidates = []
    if grid_nodes:
        candidates.append((grid_nodes, _compute_total_dv(grid_nodes)))
    if heuristic_nodes:
        candidates.append((heuristic_nodes, _compute_total_dv(heuristic_nodes)))
    candidates.extend(pool_nodes)

    if not candidates:
        record.event("no_valid_candidate")
        return []

    candidates.sort(key=lambda x: x[1])

    # Phase 3 – refine top candidates (inner‑epoch NM + DP)
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    top_n = min(TOP_REFINE, len(candidates))
    best_nodes = None
    best_total = float('inf')

    for nodes, _ in candidates[:top_n]:
        ga_seq = [n["planet_id"] for n in nodes if n["type"] == "GA"]
        start_node = next(n for n in nodes if n["type"] == "start")
        end_node   = next(n for n in nodes if n["type"] == "end")
        t0, tf = start_node["time"], end_node["time"]
        interior = [n["time"] for n in nodes if n["type"] == "GA"]

        total_opt, times_opt, nodes_opt = _optimize_ga_times(
            ga_seq, t0, tf, interior, best_limit=best_total)
        if nodes_opt is not None:
            total_opt = _compute_total_dv(nodes_opt)
            if total_opt < best_total:
                best_total, best_nodes = total_opt, nodes_opt

        # DP multi‑option refinement if time permits
        if time.time() - start_time < timeout * 0.90:
            dp_total, dp_nodes = _evaluate_ga_trajectory_multi(
                ga_seq, times_opt, best_limit=best_total)
            if dp_nodes is not None and dp_total < best_total:
                best_total, best_nodes = dp_total, dp_nodes

    if best_nodes is None:
        best_nodes, best_total = candidates[0]

    # Phase 3b – local mutation of GA sequences
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    if best_nodes is not None and remaining > 3.0:
        base_seq = [n["planet_id"] for n in best_nodes if n["type"] == "GA"]
        allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
        mutated = _mutate_sequence(base_seq, allowed)
        np.random.shuffle(mutated)
        deadline = time.time() + remaining * 0.45
        for mut_seq in mutated[:200]:
            if time.time() > deadline:
                break
            n_mut = len(mut_seq)
            if n_mut == 0:
                times = [float(best_nodes[0]["time"]), float(best_nodes[-1]["time"])]
            else:
                fracs = np.linspace(0, 1, n_mut + 2)[1:-1]
                times = [float(best_nodes[0]["time"])]
                times.extend([float(best_nodes[0]["time"] + f *
                              (best_nodes[-1]["time"] - best_nodes[0]["time"])) for f in fracs])
                times.append(float(best_nodes[-1]["time"]))
            total, nodes = _evaluate_ga_trajectory(mut_seq, times, best_total)
            if nodes is None:
                continue
            interior = [n["time"] for n in nodes if n["type"] == "GA"]
            total_opt, times_opt, nodes_opt = _optimize_ga_times(
                mut_seq, times[0], times[-1], interior, best_limit=best_total)
            if nodes_opt is None:
                continue
            total_opt = _compute_total_dv(nodes_opt)
            if total_opt < best_total:
                best_total, best_nodes = total_opt, nodes_opt

    # Phase 4 – global all‑epoch optimisation (NM + DE)
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    if best_nodes is not None and remaining > 2.0:
        ga_seq = [n["planet_id"] for n in best_nodes if n["type"] == "GA"]
        times_initial = [n["time"] for n in best_nodes
                        if n["type"] in ("start","GA","end")]
        nodes_opt, total_opt = _optimize_all_times(
            ga_seq, problem["start"], problem["end"],
            times_initial, time_budget=remaining * 0.35)
        if nodes_opt is not None and total_opt < best_total:
            best_total, best_nodes = total_opt, nodes_opt

        # short DE polish
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining > 1.5 and ga_seq:
            de_nodes, de_total = _global_de_optimize(
                ga_seq, times_initial, time_budget=remaining * 0.30)
            if de_nodes is not None and de_total < best_total:
                best_total, best_nodes = de_total, de_nodes

    # Phase 5 – refine launch/arrival windows
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    if best_nodes is not None and remaining > 2.0:
        ga_seq = [n["planet_id"] for n in best_nodes if n["type"] == "GA"]
        best_nodes, best_total = _refine_start_end_times(
            ga_seq, problem["start"], problem["end"],
            best_nodes, best_total, time_budget=remaining * 0.45)

    # Phase 6 – optional DSM insertion (if budget permits)
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    if remaining > 4.0 and int(problem.get("max_DSM", 0)) > 0:
        dsm_budget = remaining * 0.5
        refined = _add_dsms_and_optimize(best_nodes, problem, dsm_budget)
        if refined is not None:
            refined_total = _compute_total_dv(refined)
            if refined_total < best_total:
                best_total, best_nodes = refined_total, refined
                record.event("refinement_success")
            else:
                record.event("refinement_no_improve")
        else:
            record.event("refinement_skipped")

    final = _format_nodes(best_nodes) or []
    record.set("final_nodes", len(final))
    record.event("return")
    return final
# EVOLVE-BLOCK-END
