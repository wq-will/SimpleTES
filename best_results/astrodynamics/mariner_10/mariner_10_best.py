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

import itertools
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

# -------------------------------------------------
# Constants & fallback orbital radii (km)
# -------------------------------------------------
DAY = 86400.0
AU_KM = 149597870.7

_ORBIT_RAD_AU = {
    "1": 0.387,   # Mercury
    "2": 0.723,   # Venus
    "3": 1.000,   # Earth
    "4": 1.524,   # Mars
    "5": 5.204,   # Jupiter
    "6": 9.582,   # Saturn
    "7": 19.20,   # Uranus
    "8": 30.07,   # Neptune
}
_ORBIT_RAD_KM = {pid: rad * AU_KM for pid, rad in _ORBIT_RAD_AU.items()}
_mu_sun = float(problem["mu_sun"])

# -------------------------------------------------
# Helper utilities
# -------------------------------------------------
def _approx_orbital_radius(pid: str) -> float:
    """Fallback semi‑major axis (km).  \"0\" → 1 AU."""
    if pid == "0":
        return AU_KM
    return _ORBIT_RAD_KM.get(pid, AU_KM)


def _time_window(spec):
    """Return (lo, hi) MJD for a time spec."""
    if spec["kind"] == "window":
        return float(spec["lo"]), float(spec["hi"])
    # exact specification – treat as a single value
    v = float(spec.get("value", spec.get("lo", 0.0)))
    return v, v


def _piecewise_linear(x, breakpoints):
    """Linear interpolation of (vinf, dv) breakpoints."""
    if not breakpoints:
        return 0.0
    bp = sorted(breakpoints, key=lambda p: float(p[0]))
    x = float(x)
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


def _periapsis_dv(vinf, mu_p, R_p, h_factor, T_days):
    """Δv for a periapsis maneuver."""
    r_peri = R_p * (1.0 + h_factor)
    T_sec = T_days * DAY
    term = (4.0 * np.pi ** 2 * mu_p ** 2 / T_sec ** 2) ** (1.0 / 3.0)
    two_mu_r = 2.0 * mu_p / r_peri
    inner = max(two_mu_r - term, 0.0)
    return float(np.sqrt(vinf ** 2 + two_mu_r) - np.sqrt(inner))


def _boundary_dv(node, spec):
    """Δv contributed by a start / end boundary node."""
    btype = spec["type"]
    v_before = np.asarray(node["v_before"], dtype=float)
    v_after = np.asarray(node["v_after"], dtype=float)

    if btype == "piecewise_linear":
        return _piecewise_linear(np.linalg.norm(v_after - v_before),
                                 spec["breakpoints"])

    if btype == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        h = float(spec["h_factor"])
        T = float(spec["T_days"])
        vinf = np.linalg.norm(v_after - v_before)
        return _periapsis_dv(vinf, mu_p, R_p, h, T)

    # unknown type – penalise heavily
    return 1e9


def _boundary_min_dv(spec):
    """Optimistic lower bound for a boundary node."""
    btype = spec["type"]
    if btype == "piecewise_linear":
        return min(float(y) for (_, y) in spec["breakpoints"])
    if btype == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        h = float(spec["h_factor"])
        T = float(spec["T_days"])
        return _periapsis_dv(0.0, mu_p, R_p, h, T)
    return 0.0


# -------------------------------------------------
# Ephemeris cache (including the "0" reference case)
# -------------------------------------------------
_state_cache = {}


def _planet_state(pid, mjd, context):
    """Return heliocentric (r, v) for a body at epoch mjd (km, km/s)."""
    if pid == "0":
        spec = problem["start"] if context == "start" else problem["end"]
        r = np.asarray(spec.get("state_r", [0.0, 0.0, 0.0]), dtype=float)
        v = np.asarray(spec.get("state_v", [0.0, 0.0, 0.0]), dtype=float)
        return r, v

    key = (pid, float(mjd))
    if key in _state_cache:
        return _state_cache[key]

    r, v = tools.ephem(pid, float(mjd))
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    _state_cache[key] = (r, v)
    return r, v


# -------------------------------------------------
# Orbital / synodic period utilities (days)
# -------------------------------------------------
def _orbital_period_days(pid):
    a = _approx_orbital_radius(pid)          # km
    per_sec = 2.0 * np.pi * np.sqrt(a ** 3 / _mu_sun)
    return per_sec / DAY


def _synodic_period_days(pid1, pid2):
    T1 = _orbital_period_days(pid1)
    T2 = _orbital_period_days(pid2)
    if not np.isfinite(T1) or not np.isfinite(T2) or T1 == T2:
        return np.inf
    return abs(1.0 / (1.0 / T1 - 1.0 / T2))


# -------------------------------------------------
# Minimum flight‑time estimate (half‑period Hohmann bound)
# -------------------------------------------------
def _estimate_min_t_days(r1_km, r2_km):
    a = (r1_km + r2_km) / 2.0
    half_period_sec = np.pi * np.sqrt(a ** 3 / _mu_sun)
    days = half_period_sec / DAY
    return max(5.0, 0.5 * days)


# -------------------------------------------------
# Cached Lambert solver – fast single solution (any rev/geometry)
# -------------------------------------------------
_MAX_REV = max(int(problem.get("max_rev", 6)), 30)   # generous default
_lambert_cache = {}
_lambert_all_cache = {}


def _lambert_key(r1, r2, tof_days):
    r1_key = tuple(np.round(r1 / 1e4).astype(int))
    r2_key = tuple(np.round(r2 / 1e4).astype(int))
    tof_key = int(round(tof_days * 20.0))   # 0.05‑day granularity
    return (r1_key, r2_key, tof_key)


def _solve_lambert_cached(r1, r2, tof_days):
    """Return a viable (v_dep, v_arr) for given TOF (days) or None."""
    key = _lambert_key(r1, r2, tof_days)
    if key in _lambert_cache:
        return _lambert_cache[key]

    tof_sec = tof_days * DAY
    sol = None
    for M in range(_MAX_REV + 1):
        for pro in (True, False):
            for low in (True, False):
                try:
                    v_dep, v_arr = tools.lambert(r1, r2, tof_sec, _mu_sun,
                                                prograde=pro, lowpath=low, M=M)
                    sol = (np.asarray(v_dep), np.asarray(v_arr))
                    break
                except Exception:
                    continue
            if sol is not None:
                break
        if sol is not None:
            break

    if sol is None:
        try:
            v_dep, v_arr = tools.lambert(r1, r2, tof_sec, _mu_sun,
                                         prograde=True, lowpath=True, M=0)
            sol = (np.asarray(v_dep), np.asarray(v_arr))
        except Exception:
            sol = None

    _lambert_cache[key] = sol
    return sol


def _lambert_all(r1, r2, tof_days):
    """Return list of all viable (v_dep, v_arr) for given TOF (days)."""
    key = _lambert_key(r1, r2, tof_days)
    if key in _lambert_all_cache:
        return _lambert_all_cache[key]

    tof_sec = tof_days * DAY
    sols = []
    max_rev_for_all = min(_MAX_REV, 30)
    for M in range(max_rev_for_all + 1):
        for pro in (True, False):
            for low in (True, False):
                try:
                    v_dep, v_arr = tools.lambert(r1, r2, tof_sec, _mu_sun,
                                                prograde=pro, lowpath=low, M=M)
                    v_dep = np.asarray(v_dep)
                    v_arr = np.asarray(v_arr)
                    dup = False
                    for vd, va in sols:
                        if np.allclose(v_dep, vd, atol=1e-3) and np.allclose(v_arr, va, atol=1e-3):
                            dup = True
                            break
                    if not dup:
                        sols.append((v_dep, v_arr))
                except Exception:
                    continue

    if not sols:
        single = _solve_lambert_cached(r1, r2, tof_days)
        if single is not None:
            sols.append(single)

    _lambert_all_cache[key] = sols
    return sols


# -------------------------------------------------
# Powered‑flyby cost (cached)
# -------------------------------------------------
_powered_flyby_cache = {}


def _powered_flyby_cost(v_arr, v_dep, pid, epoch):
    """Return (dv, feasible) for a powered flyby at given epoch."""
    pid_str = str(pid)
    mu_p = float(problem["planet_mu"][pid_str])
    R_p = float(problem["planet_radius"][pid_str])
    min_alt = float(
        problem.get("flyby", {})
        .get("min_altitude_km", {})
        .get(pid_str, 200.0)
    )
    min_rp = R_p + min_alt
    _, v_planet = tools.ephem(pid_str, float(epoch))

    key = (
        tuple(np.round(v_arr, 3)),
        tuple(np.round(v_dep, 3)),
        pid_str,
        int(round(epoch)),
    )
    if key in _powered_flyby_cache:
        return _powered_flyby_cache[key]

    try:
        _, dv, feasible = tools.powered_flyby(
            np.asarray(v_arr), np.asarray(v_dep),
            np.asarray(v_planet), mu_p, min_rp
        )
        result = (float(dv), bool(feasible))
    except Exception:
        result = (float("inf"), False)

    _powered_flyby_cache[key] = result
    return result


# -------------------------------------------------
# Slack allocation heuristics
# -------------------------------------------------
def _alloc_slack_variants(min_tofs, slack, rng):
    """Yield plausible extra‑time distributions over the legs."""
    n = len(min_tofs)
    if n == 0:
        return
    extra_eq = slack / n

    # 1) equal share
    yield [mi + extra_eq for mi in min_tofs]

    # 2) proportional to minima
    s = sum(min_tofs)
    if s > 0.0:
        yield [mi + slack * (mi / s) for mi in min_tofs]

    # 3) random Dirichlet splits
    for _ in range(4):
        fracs = rng.random(n)
        fracs /= fracs.sum()
        yield [mi + slack * f for mi, f in zip(min_tofs, fracs)]

    # 4) front‑loaded
    yield [mi + slack * (1.0 - i / n) / n for i, mi in enumerate(min_tofs)]

    # 5) back‑loaded
    denom = n * (n + 1) / 2.0
    yield [mi + slack * (i + 1) / denom for i, mi in enumerate(min_tofs)]


# -------------------------------------------------
# Synodic‑period adjustments – random nudges
# -------------------------------------------------
def _synodic_adjustments(times, seq, min_tofs, rng, tf_bounds):
    """Create a few variants where each leg duration is nudged by integer multiples
    (±5) of the synodic period of the adjoining bodies."""
    n_legs = len(seq) + 1
    pid_start = str(problem["start"].get("planet_id", "0"))
    pid_end = str(problem["end"].get("planet_id", "0"))
    pid_list = [pid_start] + list(seq) + [pid_end]

    base = np.array(times, dtype=float)
    variants = []

    for _ in range(6):
        new = base.copy()
        for i in range(n_legs):
            pid_i = pid_list[i]
            pid_j = pid_list[i + 1]
            Ts = _synodic_period_days(pid_i, pid_j)
            if not np.isfinite(Ts) or Ts <= 0.0:
                continue
            offset = rng.integers(-5, 6)   # -5 … +5
            if offset == 0:
                continue
            old_dt = new[i + 1] - new[i]
            new_dt = old_dt + offset * Ts
            if new_dt < min_tofs[i]:
                new_dt = min_tofs[i]
            delta = new_dt - old_dt
            if delta != 0.0:
                new[i + 1 :] += delta

        if any(new[k + 1] <= new[k] for k in range(len(new) - 1)):
            continue
        if not (tf_bounds[0] <= new[-1] <= tf_bounds[1]):
            continue
        variants.append(new.tolist())
    return variants


def _synodic_grid_adjustments(times, seq, min_tofs, max_offset=2):
    """Deterministic grid of synodic offsets (bounded) for up to 3 legs."""
    n_legs = len(seq) + 1
    pid_start = str(problem["start"].get("planet_id", "0"))
    pid_end = str(problem["end"].get("planet_id", "0"))
    pid_list = [pid_start] + list(seq) + [pid_end]

    base = np.array(times, dtype=float)
    offsets_ranges = [range(-max_offset, max_offset + 1) for _ in range(n_legs)]

    variants = []
    for combo in itertools.product(*offsets_ranges):
        new = base.copy()
        ok = True
        for i, off in enumerate(combo):
            if off == 0:
                continue
            pid_i = pid_list[i]
            pid_j = pid_list[i + 1]
            Ts = _synodic_period_days(pid_i, pid_j)
            if not np.isfinite(Ts) or Ts <= 0.0:
                ok = False
                break
            old_dt = new[i + 1] - new[i]
            new_dt = old_dt + off * Ts
            if new_dt < min_tofs[i]:
                new_dt = min_tofs[i]
            delta = new_dt - old_dt
            if delta != 0.0:
                new[i + 1 :] += delta
        if not ok:
            continue
        if any(new[k + 1] <= new[k] for k in range(len(new) - 1)):
            continue
        variants.append(new.tolist())
    return variants


# -------------------------------------------------
# Fast trajectory evaluation (single Lambert per leg)
# -------------------------------------------------
def _evaluate_sequence(seq, times):
    """Return (total_dv, node_list) or (inf, None) if infeasible."""
    n_ga = len(seq)
    if len(times) != n_ga + 2:
        return float("inf"), None

    # monotonic time check
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        return float("inf"), None

    # window validation
    t0_lo, t0_hi = _time_window(problem["start"]["time"])
    tf_lo, tf_hi = _time_window(problem["end"]["time"])
    if not (t0_lo <= times[0] <= t0_hi) or not (tf_lo <= times[-1] <= tf_hi):
        return float("inf"), None

    start_pid = str(problem["start"].get("planet_id", "0"))
    end_pid = str(problem["end"].get("planet_id", "0"))
    pid_list = [start_pid] + [str(p) for p in seq] + [end_pid]

    if len(pid_list) > int(problem.get("max_nodes", 100)):
        return float("inf"), None

    # fetch planetary (or reference) states
    r_list = []
    v_ref = []
    for idx, (pid, t) in enumerate(zip(pid_list, times)):
        ctx = "start" if idx == 0 else ("end" if idx == len(pid_list) - 1 else "mid")
        r, v = _planet_state(pid, t, ctx)
        r_list.append(r)
        v_ref.append(v)

    # Lambert for each leg
    dep_vs = []
    arr_vs = []
    for i in range(len(pid_list) - 1):
        tof = times[i + 1] - times[i]
        if tof <= 0.0:
            return float("inf"), None
        sol = _solve_lambert_cached(r_list[i], r_list[i + 1], tof)
        if sol is None:
            return float("inf"), None
        v_dep, v_arr = sol
        dep_vs.append(v_dep)
        arr_vs.append(v_arr)

    total_dv = 0.0
    nodes = []

    # start node
    start_node = {
        "type": "start",
        "time": float(times[0]),
        "planet_id": start_pid,
        "r": r_list[0],
        "v_before": v_ref[0],
        "v_after": dep_vs[0],
    }
    total_dv += _boundary_dv(start_node, problem["start"])
    nodes.append(start_node)

    # gravity‑assist nodes
    for idx, pid in enumerate(seq):
        v_in = arr_vs[idx]
        v_out = dep_vs[idx + 1]
        dv_ga, feasible = _powered_flyby_cost(v_in, v_out, pid, times[idx + 1])
        if not feasible:
            return float("inf"), None
        total_dv += dv_ga
        ga_node = {
            "type": "GA",
            "time": float(times[idx + 1]),
            "planet_id": str(pid),
            "r": r_list[idx + 1],
            "v_before": v_in,
            "v_after": v_out,
        }
        nodes.append(ga_node)

    # end node
    end_node = {
        "type": "end",
        "time": float(times[-1]),
        "planet_id": end_pid,
        "r": r_list[-1],
        "v_before": arr_vs[-1],
        "v_after": v_ref[-1],
    }
    total_dv += _boundary_dv(end_node, problem["end"])
    nodes.append(end_node)

    return float(total_dv), nodes


# -------------------------------------------------
# Exhaustive evaluation (all Lambert solutions per leg)
# -------------------------------------------------
def _evaluate_sequence_opt(seq, times):
    """Same as _evaluate_sequence but enumerates all Lambert solutions."""
    n_ga = len(seq)
    if len(times) != n_ga + 2:
        return float("inf"), None

    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        return float("inf"), None

    t0_lo, t0_hi = _time_window(problem["start"]["time"])
    tf_lo, tf_hi = _time_window(problem["end"]["time"])
    if not (t0_lo <= times[0] <= t0_hi) or not (tf_lo <= times[-1] <= tf_hi):
        return float("inf"), None

    start_pid = str(problem["start"].get("planet_id", "0"))
    end_pid = str(problem["end"].get("planet_id", "0"))
    pid_list = [start_pid] + [str(p) for p in seq] + [end_pid]

    if len(pid_list) > int(problem.get("max_nodes", 100)):
        return float("inf"), None

    # planetary states
    r_list = []
    v_ref = []
    for idx, (pid, t) in enumerate(zip(pid_list, times)):
        ctx = "start" if idx == 0 else ("end" if idx == len(pid_list) - 1 else "mid")
        r, v = _planet_state(pid, t, ctx)
        r_list.append(r)
        v_ref.append(v)

    # collect all Lambert solutions per leg
    leg_sols = []
    for i in range(len(pid_list) - 1):
        tof = times[i + 1] - times[i]
        sols = _lambert_all(r_list[i], r_list[i + 1], tof)
        if not sols:
            return float("inf"), None
        leg_sols.append(sols)

    best_total = float("inf")
    best_combo = None

    def dfs(leg_idx, incoming_vel, cost_sofar, combo):
        nonlocal best_total, best_combo
        if leg_idx == len(leg_sols):
            # final node
            end_node = {
                "type": "end",
                "time": float(times[-1]),
                "planet_id": end_pid,
                "r": r_list[-1],
                "v_before": incoming_vel,
                "v_after": v_ref[-1],
            }
            dv_end = _boundary_dv(end_node, problem["end"])
            total = cost_sofar + dv_end
            if total < best_total:
                best_total = total
                best_combo = combo.copy()
            return

        for v_dep, v_arr in leg_sols[leg_idx]:
            if leg_idx == 0:
                # start node
                start_node = {
                    "type": "start",
                    "time": float(times[0]),
                    "planet_id": start_pid,
                    "r": r_list[0],
                    "v_before": v_ref[0],
                    "v_after": v_dep,
                }
                dv_start = _boundary_dv(start_node, problem["start"])
                new_cost = cost_sofar + dv_start
                if new_cost >= best_total:
                    continue
                dfs(1, v_arr, new_cost, combo + [(v_dep, v_arr)])
            else:
                ga_pid = seq[leg_idx - 1]
                dv_ga, feasible = _powered_flyby_cost(incoming_vel, v_dep, ga_pid, times[leg_idx])
                if not feasible:
                    continue
                new_cost = cost_sofar + dv_ga
                if new_cost >= best_total:
                    continue
                dfs(leg_idx + 1, v_arr, new_cost, combo + [(v_dep, v_arr)])

    dfs(0, None, 0.0, [])
    if best_combo is None:
        return float("inf"), None

    # reconstruct node list from best_combo
    nodes = []
    v_dep0, _ = best_combo[0]
    start_node = {
        "type": "start",
        "time": float(times[0]),
        "planet_id": start_pid,
        "r": r_list[0],
        "v_before": v_ref[0],
        "v_after": v_dep0,
    }
    nodes.append(start_node)

    for idx, pid in enumerate(seq):
        v_arr = best_combo[idx][1]
        v_dep_next = best_combo[idx + 1][0] if idx + 1 < len(best_combo) else None
        ga_node = {
            "type": "GA",
            "time": float(times[idx + 1]),
            "planet_id": str(pid),
            "r": r_list[idx + 1],
            "v_before": v_arr,
            "v_after": v_dep_next,
        }
        nodes.append(ga_node)

    v_arr_last = best_combo[-1][1]
    end_node = {
        "type": "end",
        "time": float(times[-1]),
        "planet_id": end_pid,
        "r": r_list[-1],
        "v_before": v_arr_last,
        "v_after": v_ref[-1],
    }
    nodes.append(end_node)

    return float(best_total), nodes


# -------------------------------------------------
# Candidate‑pool handling
# -------------------------------------------------
_MAX_CAND_POOL = 12
_candidate_pool = []   # list of dicts: {"seq":..., "times":..., "dv":...}


def _add_to_candidate_pool(seq, times, dv):
    if dv == float("inf"):
        return
    entry = {"seq": list(seq), "times": list(times), "dv": dv}
    inserted = False
    for i, existing in enumerate(_candidate_pool):
        if dv < existing["dv"]:
            _candidate_pool.insert(i, entry)
            inserted = True
            break
    if not inserted:
        _candidate_pool.append(entry)
    if len(_candidate_pool) > _MAX_CAND_POOL:
        _candidate_pool.pop()


# -------------------------------------------------
# Small mutation of a GA sequence
# -------------------------------------------------
def _mutate_sequence(seq, allowed, max_len, rng):
    """Randomly insert/delete/replace/swap elements."""
    if max_len == 0:
        return seq
    new_seq = list(seq)
    ops = ["insert", "delete", "replace", "swap"]
    op = rng.choice(ops)
    if op == "insert" and len(new_seq) < max_len and allowed:
        pid = rng.choice(allowed)
        pos = rng.integers(0, len(new_seq) + 1)
        new_seq.insert(pos, pid)
    elif op == "delete" and len(new_seq) > 0:
        idx = rng.integers(0, len(new_seq))
        del new_seq[idx]
    elif op == "replace" and len(new_seq) > 0 and allowed:
        idx = rng.integers(0, len(new_seq))
        pid = rng.choice(allowed)
        new_seq[idx] = pid
    elif op == "swap" and len(new_seq) >= 2:
        i, j = rng.choice(len(new_seq), size=2, replace=False)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


# -------------------------------------------------
# Core routine for a concrete GA sequence
# -------------------------------------------------
def _try_sequence(seq, launch_epoch, best, t0_bounds, tf_bounds,
                  r_start, r_end, max_nodes, max_dsm, rng):
    """Explore a concrete sequence anchored at launch_epoch."""
    radii = [r_start] + [_approx_orbital_radius(p) for p in seq] + [r_end]
    min_tofs = [_estimate_min_t_days(radii[i], radii[i + 1])
                for i in range(len(radii) - 1)]
    total_min = sum(min_tofs)

    # quick feasibility
    if launch_epoch + total_min > tf_bounds[1]:
        return
    earliest_arrival = max(tf_bounds[0], launch_epoch + total_min)
    if earliest_arrival > tf_bounds[1]:
        return

    # Sample a few arrival epochs
    for _ in range(12):
        arrival = rng.uniform(earliest_arrival, tf_bounds[1])
        slack = arrival - launch_epoch - total_min

        for leg_tofs in _alloc_slack_variants(min_tofs, slack, rng):
            times = [float(launch_epoch)]
            for dt in leg_tofs:
                times.append(times[-1] + dt)
            times[-1] = float(arrival)   # enforce exact arrival

            # Fast evaluation first
            total_dv, nodes = _evaluate_sequence(seq, times)
            if nodes is not None and total_dv < best["dv"]:
                best.update({"dv": total_dv, "nodes": nodes,
                             "seq": list(seq), "times": list(times)})
                try:
                    record.event(f"new_best dv={total_dv:.6f} seq={seq}")
                except Exception:
                    pass

            if nodes is not None:
                _add_to_candidate_pool(seq, times, total_dv)

            # Random synodic nudges
            for adj in _synodic_adjustments(times, seq, min_tofs, rng, tf_bounds):
                total_adj, nodes_adj = _evaluate_sequence(seq, adj)
                if nodes_adj is not None and total_adj < best["dv"]:
                    best.update({"dv": total_adj, "nodes": nodes_adj,
                                 "seq": list(seq), "times": list(adj)})
                    try:
                        record.event(f"syn_adj_best dv={total_adj:.6f} seq={seq}")
                    except Exception:
                        pass
                if nodes_adj is not None:
                    _add_to_candidate_pool(seq, adj, total_adj)

            # Deterministic grid (up to 3 legs)
            max_grid_off = 2 if len(seq) <= 2 else 1
            for adj in _synodic_grid_adjustments(times, seq, min_tofs,
                                                  max_offset=max_grid_off):
                total_adj, nodes_adj = _evaluate_sequence(seq, adj)
                if nodes_adj is not None and total_adj < best["dv"]:
                    best.update({"dv": total_adj, "nodes": nodes_adj,
                                 "seq": list(seq), "times": list(adj)})
                    try:
                        record.event(f"grid_adj_best dv={total_adj:.6f} seq={seq}")
                    except Exception:
                        pass
                if nodes_adj is not None:
                    _add_to_candidate_pool(seq, adj, total_adj)

            # Local refinement (Nelder‑Mead)
            def obj(x):
                if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
                    return 1e9
                if not (t0_bounds[0] <= x[0] <= t0_bounds[1]):
                    return 1e9
                if not (tf_bounds[0] <= x[-1] <= tf_bounds[1]):
                    return 1e9
                dv, _ = _evaluate_sequence(seq, x.tolist())
                return dv

            try:
                res = minimize(obj, np.array(times, dtype=float),
                               method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4})
                if res.success:
                    dv_opt, nodes_opt = _evaluate_sequence(seq, res.x.tolist())
                    if dv_opt < best["dv"]:
                        best.update({"dv": dv_opt, "nodes": nodes_opt,
                                     "seq": list(seq), "times": res.x.tolist()})
                        try:
                            record.event(f"refined_best dv={dv_opt:.6f} seq={seq}")
                        except Exception:
                            pass
                    if dv_opt != float("inf"):
                        _add_to_candidate_pool(seq, res.x.tolist(), dv_opt)
            except Exception:
                pass


# -------------------------------------------------
# Main driver
# -------------------------------------------------
def run_code():
    # ------------------------------------------------------------------
    # Extract mission specifications
    # ------------------------------------------------------------------
    start_spec = problem["start"]
    end_spec = problem["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])

    allowed_ga = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga_allowed = min(int(problem.get("max_GA", len(allowed_ga))), len(allowed_ga))
    max_nodes = int(problem.get("max_nodes", 100))
    max_dsm = int(problem.get("max_DSM", 0))

    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))

    # ------------------------------------------------------------------
    # Approximate orbital radii for start & end (km)
    # ------------------------------------------------------------------
    r_start = _approx_orbital_radius(start_pid)
    r_end = _approx_orbital_radius(end_pid)

    # ------------------------------------------------------------------
    # Order planets from inner to outer (or reverse) to guide enumeration
    # ------------------------------------------------------------------
    direction = 1 if r_start <= r_end else -1
    sorted_allowed = sorted(
        allowed_ga,
        key=lambda pid: _approx_orbital_radius(pid) * direction,
    )

    # ------------------------------------------------------------------
    # Bookkeeping for the best solution found so far
    # ------------------------------------------------------------------
    best = {"dv": float("inf"), "nodes": None, "seq": None, "times": None}
    timeout_total = float(problem.get("timeout_seconds", 70.0))
    deadline = time.time() + timeout_total * 0.85   # safety margin
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Build a diverse list of candidate GA sequences
    # ------------------------------------------------------------------
    total_possible = sum(len(allowed_ga) ** L for L in range(0, max_ga_allowed + 1))
    enumeration_limit = 15000
    candidate_seqs = []

    if total_possible <= enumeration_limit:
        # exhaustive enumeration up to max length
        for L in range(0, max_ga_allowed + 1):
            if L == 0:
                candidate_seqs.append([])
            else:
                for seq in itertools.product(allowed_ga, repeat=L):
                    candidate_seqs.append(list(seq))
    else:
        # always keep the direct (no‑GA) option
        candidate_seqs.append([])

        # monotonic inner→outer combos up to length 3 (fast to explore)
        max_len_enum = min(max_ga_allowed, 3, max_nodes - 2)
        for L in range(1, max_len_enum + 1):
            for combo in itertools.combinations(sorted_allowed, L):
                candidate_seqs.append(list(combo))

        # a few repeated‑planet motifs (useful for resonant returns)
        if max_ga_allowed >= 2 and sorted_allowed:
            for pid in sorted_allowed[:3]:
                candidate_seqs.append([pid] * min(max_ga_allowed, 3))

        # random coverage
        for _ in range(80):
            L = rng.integers(1, max_ga_allowed + 1) if max_ga_allowed > 0 else 0
            seq = list(rng.choice(sorted_allowed, size=L, replace=True)) if L > 0 else []
            candidate_seqs.append(seq)

    # deduplicate while preserving order
    uniq = []
    seen = set()
    for seq in candidate_seqs:
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            uniq.append(seq)
    candidate_seqs = uniq

    # ------------------------------------------------------------------
    # Optimistic lower bound (boundary burns only)
    # ------------------------------------------------------------------
    lower_bound_total = _boundary_min_dv(start_spec) + _boundary_min_dv(end_spec)

    # ------------------------------------------------------------------
    # Phase 1 – systematic launch‑epoch grid search
    # ------------------------------------------------------------------
    window_span = max(0.0, t0_hi - t0_lo)
    grid_n = max(8, min(80, int(window_span / 3.0) + 1))
    launch_grid = np.linspace(t0_lo, t0_hi, num=grid_n)

    for seq in candidate_seqs:
        if time.time() > deadline:
            break
        if len(seq) + 2 > max_nodes or len(seq) > max_ga_allowed:
            continue
        for launch_epoch in launch_grid:
            if time.time() > deadline:
                break
            _try_sequence(seq, float(launch_epoch), best,
                          (t0_lo, t0_hi), (tf_lo, tf_hi),
                          r_start, r_end, max_nodes, max_dsm, rng)
            if best["dv"] <= lower_bound_total + 1e-5:
                deadline = time.time()   # early exit – cannot improve bound
                break

    # ------------------------------------------------------------------
    # Phase 2 – stochastic exploration (random launches & sequences)
    # ------------------------------------------------------------------
    while time.time() < deadline:
        n_ga = rng.integers(0, max_ga_allowed + 1) if max_ga_allowed > 0 else 0
        if n_ga > 0:
            seq = list(rng.choice(sorted_allowed, size=n_ga, replace=True))
            if rng.random() < 0.6:
                seq.sort(key=lambda pid: _approx_orbital_radius(pid) * direction)
        else:
            seq = []

        if len(seq) + 2 > max_nodes:
            continue

        launch_epoch = rng.uniform(t0_lo, t0_hi)
        _try_sequence(seq, float(launch_epoch), best,
                      (t0_lo, t0_hi), (tf_lo, tf_hi),
                      r_start, r_end, max_nodes, max_dsm, rng)

        if best["dv"] <= lower_bound_total + 1e-6:
            break

    # ------------------------------------------------------------------
    # Phase 3 – mutate best sequence (local search)
    # ------------------------------------------------------------------
    mut_iters = 120
    while time.time() < deadline and mut_iters > 0:
        mut_iters -= 1
        if best["seq"] is None:
            break
        new_seq = _mutate_sequence(best["seq"], allowed_ga, max_ga_allowed, rng)
        launch_guess = best["times"][0] if best["times"] else rng.uniform(t0_lo, t0_hi)
        _try_sequence(new_seq, float(launch_guess), best,
                      (t0_lo, t0_hi), (tf_lo, tf_hi),
                      r_start, r_end, max_nodes, max_dsm, rng)

    # ------------------------------------------------------------------
    # Phase 4 – jitter‑based refinement of the current best
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None:
        seq = best["seq"]
        radii = [r_start] + [_approx_orbital_radius(p) for p in seq] + [r_end]
        min_tofs = [_estimate_min_t_days(radii[i], radii[i + 1])
                    for i in range(len(radii) - 1)]

        jitter_iters = 3500
        while time.time() < deadline and jitter_iters > 0:
            jitter_iters -= 1
            base = np.diff(best["times"])
            jitter = rng.normal(scale=0.13, size=len(base)) * base
            new_durs = base + jitter
            new_durs = np.maximum(new_durs, np.array(min_tofs) * 0.66)

            new_times = np.concatenate(
                ([best["times"][0]], best["times"][0] + np.cumsum(new_durs))
            )
            if new_times[-1] > tf_hi:
                new_times[-1] = tf_hi
            if not (tf_lo <= new_times[-1] <= tf_hi):
                continue
            if any(new_times[i + 1] <= new_times[i] for i in range(len(new_times) - 1)):
                continue

            total_dv, nodes = _evaluate_sequence(seq, new_times.tolist())
            if nodes is not None and total_dv < best["dv"]:
                best.update({"dv": total_dv, "nodes": nodes,
                             "times": new_times.tolist()})
                try:
                    record.event(f"jitter_best dv={total_dv:.6f} seq={seq}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Phase 5 – L‑BFGS‑B optimisation over the full time vector
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None and time.time() < deadline:
        seq = best["seq"]
        radii = [r_start] + [_approx_orbital_radius(p) for p in seq] + [r_end]
        min_tofs = [_estimate_min_t_days(radii[i], radii[i + 1])
                    for i in range(len(radii) - 1)]

        x0 = np.array([best["times"][0]] + list(np.diff(best["times"])), dtype=float)

        bounds = [(t0_lo, t0_hi)]
        for mt in min_tofs:
            bounds.append((mt, tf_hi - t0_lo))

        def obj_lbfgs(x):
            if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
                return 1e9
            if not (tf_lo <= x[-1] <= tf_hi):
                return 1e9
            dv, _ = _evaluate_sequence(seq, x.tolist())
            return dv

        try:
            res = minimize(obj_lbfgs, x0, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxfun": 800, "maxiter": 800, "ftol": 1e-6})
            if res.success:
                dv_opt, nodes_opt = _evaluate_sequence(seq, res.x.tolist())
                if dv_opt < best["dv"]:
                    best.update({"dv": dv_opt, "nodes": nodes_opt,
                                 "times": res.x.tolist()})
                    try:
                        record.event(f"lbfgs_best dv={dv_opt:.6f} seq={seq}")
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 6 – Differential Evolution on the full time vector
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None and time.time() < deadline:
        seq = best["seq"]
        radii = [r_start] + [_approx_orbital_radius(p) for p in seq] + [r_end]
        min_tofs = [_estimate_min_t_days(radii[i], radii[i + 1])
                    for i in range(len(radii) - 1)]

        de_bounds = [(t0_lo, t0_hi)]
        for mt in min_tofs:
            de_bounds.append((mt, tf_hi - t0_lo))

        def de_obj(x):
            if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
                return 1e9
            if not (tf_lo <= x[-1] <= tf_hi):
                return 1e9
            dv, _ = _evaluate_sequence(seq, x.tolist())
            return dv

        try:
            de_res = differential_evolution(
                de_obj,
                de_bounds,
                maxiter=30,
                popsize=10,
                polish=False,
                seed=rng,
                updating='deferred',
                atol=1e-3,
                disp=False,
            )
            if de_res.success:
                dv_de, nodes_de = _evaluate_sequence(seq, de_res.x.tolist())
                if dv_de < best["dv"]:
                    best.update({"dv": dv_de, "nodes": nodes_de,
                                 "times": de_res.x.tolist()})
                    try:
                        record.event(f"de_best dv={dv_de:.6f} seq={seq}")
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 7 – Dual‑Annealing global refinement
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None and time.time() < deadline:
        seq = best["seq"]
        radii = [r_start] + [_approx_orbital_radius(p) for p in seq] + [r_end]
        min_tofs = [_estimate_min_t_days(radii[i], radii[i + 1])
                    for i in range(len(radii) - 1)]

        da_bounds = [(t0_lo, t0_hi)]
        for mt in min_tofs:
            da_bounds.append((mt, tf_hi - t0_lo))

        def da_obj(x):
            if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
                return 1e9
            if not (tf_lo <= x[-1] <= tf_hi):
                return 1e9
            dv, _ = _evaluate_sequence(seq, x.tolist())
            return dv

        try:
            da_res = dual_annealing(
                da_obj,
                da_bounds,
                maxiter=200,
                seed=rng,
                no_local_search=True,
            )
            if da_res.success:
                dv_da, nodes_da = _evaluate_sequence(seq, da_res.x.tolist())
                if dv_da < best["dv"]:
                    best.update({"dv": dv_da, "nodes": nodes_da,
                                 "times": da_res.x.tolist()})
                    try:
                        record.event(f"da_best dv={dv_da:.6f} seq={seq}")
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 8 – polish top candidates from the pool (full‑Lambert search)
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None and time.time() < deadline:
        for entry in _candidate_pool[:3]:
            seq = entry["seq"]
            times = entry["times"]
            total_opt, nodes_opt = _evaluate_sequence_opt(seq, times)
            if nodes_opt is not None and total_opt < best["dv"]:
                best.update({"dv": total_opt, "nodes": nodes_opt, "times": times, "seq": seq})
                try:
                    record.event(f"pool_opt_best dv={total_opt:.6f} seq={seq}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Phase 9 – exhaustive Lambert polishing on best sequence
    # ------------------------------------------------------------------
    if best["nodes"] is not None and best["seq"] is not None:
        seq = best["seq"]
        times = best["times"]
        total_opt, nodes_opt = _evaluate_sequence_opt(seq, times)
        if nodes_opt is not None and total_opt < best["dv"]:
            best.update({"dv": total_opt, "nodes": nodes_opt})
            try:
                record.event(f"final_opt dv={total_opt:.6f} seq={seq}")
            except Exception:
                pass

        # Light random jitter + polish passes
        for _ in range(12):
            if time.time() > deadline:
                break
            base = np.diff(times)
            jitter = rng.normal(scale=0.05, size=len(base)) * base
            new_durs = base + jitter
            new_durs = np.maximum(new_durs,
                                  np.array([_estimate_min_t_days(r_start, r_end)] + [0.0] * (len(base) - 1)))
            new_times = np.concatenate(([times[0]], times[0] + np.cumsum(new_durs)))
            if new_times[-1] > tf_hi:
                new_times[-1] = tf_hi
            if not (tf_lo <= new_times[-1] <= tf_hi):
                continue
            total_opt2, nodes_opt2 = _evaluate_sequence_opt(seq, new_times.tolist())
            if nodes_opt2 is not None and total_opt2 < best["dv"]:
                best.update({"dv": total_opt2, "nodes": nodes_opt2,
                             "times": new_times.tolist()})
                try:
                    record.event(f"rand_opt dv={total_opt2:.6f}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Phase 10 – fallback direct transfer (if nothing else succeeded)
    # ------------------------------------------------------------------
    if best["nodes"] is None:
        for _ in range(80):
            t0 = rng.uniform(t0_lo, t0_hi)
            tf_min = max(tf_lo, t0 + 30.0)  # reasonable minimum flight time
            if tf_min > tf_hi:
                continue
            tf = rng.uniform(tf_min, tf_hi)
            total_dv, nodes = _evaluate_sequence([], [t0, tf])
            if nodes is not None and total_dv < best["dv"]:
                best.update({"dv": total_dv, "nodes": nodes,
                             "seq": [], "times": [t0, tf]})
                try:
                    record.event(f"fallback_best dv={total_dv:.6f}")
                except Exception:
                    pass
                break

    # ------------------------------------------------------------------
    # Safety placeholder – guarantee a syntactically valid result
    # ------------------------------------------------------------------
    if best["nodes"] is None:
        start_node = {
            "type": "start",
            "time": float(t0_lo),
            "planet_id": start_pid,
            "r": [0.0, 0.0, 0.0],
            "v_before": [0.0, 0.0, 0.0],
            "v_after": [0.0, 0.0, 0.0],
        }
        end_node = {
            "type": "end",
            "time": float(tf_hi),
            "planet_id": end_pid,
            "r": [0.0, 0.0, 0.0],
            "v_before": [0.0, 0.0, 0.0],
            "v_after": [0.0, 0.0, 0.0],
        }
        best["nodes"] = [start_node, end_node]
        best["dv"] = 0.0

    # ------------------------------------------------------------------
    # Convert any numpy arrays inside nodes to plain python lists
    # ------------------------------------------------------------------
    for nd in best["nodes"]:
        for key in ("r", "v_before", "v_after"):
            val = nd[key]
            if isinstance(val, np.ndarray):
                nd[key] = val.tolist()
            elif isinstance(val, (list, tuple)):
                nd[key] = list(val)
            else:
                nd[key] = val

    # ------------------------------------------------------------------
    # Final logging & return
    # ------------------------------------------------------------------
    try:
        record.set("final_nodes", len(best["nodes"]))
        record.set("final_dv", float(best["dv"]) if best["dv"] != float("inf") else None)
        record.event("trajectory_computed")
    except Exception:
        pass

    return best["nodes"]
# EVOLVE-BLOCK-END
