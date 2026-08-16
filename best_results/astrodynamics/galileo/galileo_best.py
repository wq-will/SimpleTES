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
import hashlib
import numpy as np
from scipy.optimize import minimize, differential_evolution

# -------------------------------------------------------------------------
# Global constants & configurable parameters (tuned for higher quality)
# -------------------------------------------------------------------------
DAY = 86400.0                     # seconds per day
MIN_TOF = 0.5                     # minimal leg duration (days)
TIME_MARGIN = 0.985               # reserve a fraction of timeout for safety
MAX_REV_GLOBAL = 120              # max multi‑revolution Lambert solves
MAX_CAND_RAW = 800                # raw Lambert candidates per leg before pruning
PRIORITY_FACTOR = 8.0             # boost for mission‑specific GA patterns

# Sampling budget per GA count (more aggressive for low‑order sequences)
SAMPLE_PLAN = {
    0: 15000,
    1: 12000,
    2: 120000,
    3: 60000,
    4: 30000,
    5: 24000,
    6: 16000,
    7: 10000,
    8: 6000,
}

# -------------------------------------------------------------------------
# Deterministic RNG (seed derived from mission id)
# -------------------------------------------------------------------------
_SEED = int(
    hashlib.sha256(str(problem.get("id", "0")).encode()).hexdigest(),
    16
) % (2**32)
_rng = np.random.default_rng(_SEED)

# -------------------------------------------------------------------------
# Caching helpers
# -------------------------------------------------------------------------
_ephem_cache = {}
def _ephem(pid, mjd):
    """Cached heliocentric planetary state."""
    key = (pid, float(mjd))
    if key not in _ephem_cache:
        _ephem_cache[key] = tools.ephem(pid, float(mjd))
    return _ephem_cache[key]

_lambert_cache = {}
def _lambert_candidates(r0, r1, tof_sec, mu):
    """Generate up to MAX_CAND_RAW Lambert solutions."""
    tof_days = tof_sec / DAY
    max_rev = int(tof_days / 140)          # ~1 rev per 140 days (looser)
    max_rev = max(0, min(max_rev, MAX_REV_GLOBAL))
    key = (tuple(np.round(r0, 3)), tuple(np.round(r1, 3)), int(tof_sec), max_rev)
    if key in _lambert_cache:
        return _lambert_cache[key]

    cands = []
    for M in range(0, max_rev + 1):
        for lowpath in (True, False):
            try:
                v_dep, v_arr = tools.lambert(
                    r0, r1, tof_sec, mu,
                    prograde=True, lowpath=lowpath, M=M
                )
                cands.append((np.asarray(v_dep), np.asarray(v_arr)))
                if len(cands) >= MAX_CAND_RAW:
                    break
            except Exception:
                continue
        if len(cands) >= MAX_CAND_RAW:
            break

    # keep the ~200 lowest‑norm candidates if many are produced
    if len(cands) > 200:
        mags = [np.linalg.norm(v[0]) for v in cands]
        idx = np.argsort(mags)[:200]
        cands = [cands[i] for i in idx]

    _lambert_cache[key] = cands
    return cands

_flyby_cache = {}
def _powered_flyby_dv(v_arr, v_dep, pid, t, prob):
    """Powered‑flyby mismatch Δv and feasibility."""
    pid_str = str(pid)
    key = (tuple(np.round(v_arr, 6)), tuple(np.round(v_dep, 6)), pid_str, float(t))
    if key in _flyby_cache:
        return _flyby_cache[key]

    mu_p = float(prob["planet_mu"][pid_str])
    R_p = float(prob["planet_radius"][pid_str])
    min_alt = float(prob.get("flyby", {})
                     .get("min_altitude_km", {})
                     .get(pid_str, 200))

    _, v_planet = _ephem(pid_str, float(t))
    try:
        _, dv, feas = tools.powered_flyby(
            np.asarray(v_arr), np.asarray(v_dep), v_planet,
            mu_p, R_p + min_alt
        )
        out = (float(dv), bool(feas))
    except Exception:
        out = (float("inf"), False)

    _flyby_cache[key] = out
    return out

# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------
def _time_window(spec):
    """Return (low, high) MJD from a boundary spec."""
    if spec.get("kind") == "window":
        return float(spec["lo"]), float(spec["hi"])
    # fall back to a single value
    val = float(spec.get("value", spec.get("lo", 0.0)))
    return val, val

def _piecewise_linear(vmag, breakpoints):
    """Linear interpolation of Δv from breakpoints."""
    bp = sorted(breakpoints, key=lambda p: float(p[0]))
    if vmag <= float(bp[0][0]):
        return float(bp[0][1])
    if vmag >= float(bp[-1][0]):
        return float(bp[-1][1])
    for i in range(len(bp) - 1):
        x0, y0 = float(bp[i][0]), bp[i][1]
        x1, y1 = float(bp[i + 1][0]), bp[i + 1][1]
        if x0 <= vmag <= x1:
            if x1 == x0:
                return float(y0)
            return y0 + (y1 - y0) * (vmag - x0) / (x1 - x0)
    return float(bp[-1][1])

def _periapsis_dv(vinf, mu_p, R_p, h_factor, T_days):
    """Δv for a periapsis‑maneuver boundary node."""
    r_peri = R_p * (1.0 + h_factor)
    T_sec = T_days * DAY
    two_mu_r = 2.0 * mu_p / r_peri
    term = (4.0 * np.pi**2 * mu_p**2 / T_sec**2) ** (1.0 / 3.0)
    return float(np.sqrt(vinf * vinf + two_mu_r) -
                 np.sqrt(max(two_mu_r - term, 0.0)))

def _boundary_dv(node, spec):
    """Δv contributed by a start or end boundary node."""
    if spec["type"] == "piecewise_linear":
        dv_vec = np.asarray(node["v_after"], dtype=float) - \
                 np.asarray(node["v_before"], dtype=float)
        dvmag = float(np.linalg.norm(dv_vec))
        return _piecewise_linear(dvmag, spec["breakpoints"])
    elif spec["type"] == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        h = float(spec["h_factor"])
        T = float(spec["T_days"])
        _, v_planet = _ephem(pid, float(node["time"]))
        if node["type"] == "start":
            vinf = float(np.linalg.norm(np.asarray(node["v_after"]) - v_planet))
        else:  # end
            vinf = float(np.linalg.norm(np.asarray(node["v_before"]) - v_planet))
        return _periapsis_dv(vinf, mu_p, R_p, h, T)
    else:
        return 1.0e9

# orbital radii for Hohmann‑time estimate (km)
_AU_KM = 149597870.7
_ORBIT_RADII_KM = {
    "1": 0.38709893 * _AU_KM,   # Mercury
    "2": 0.72333199 * _AU_KM,   # Venus
    "3": 1.0 * _AU_KM,          # Earth
    "4": 1.52366231 * _AU_KM,   # Mars
    "5": 5.20336301 * _AU_KM,   # Jupiter
    "6": 9.53707032 * _AU_KM,   # Saturn
    "7": 19.19126393 * _AU_KM,  # Uranus
    "8": 30.06896348 * _AU_KM,  # Neptune
}

def _hohmann_time(pid_i, pid_j):
    """Approximate Hohmann transfer time (days) between two planets."""
    ri = _ORBIT_RADII_KM.get(pid_i, _ORBIT_RADII_KM["3"])
    rj = _ORBIT_RADII_KM.get(pid_j, _ORBIT_RADII_KM["3"])
    a = 0.5 * (ri + rj)
    tof_sec = np.pi * np.sqrt(a ** 3 / problem["mu_sun"])
    return float(tof_sec / DAY)

def _guided_times(t0, tf, ga_seq, start_pid, end_pid):
    """Generate plausible epoch schedule using scaled Hohmann times + jitter."""
    pids = [start_pid] + list(ga_seq) + [end_pid]
    base_tofs = [_hohmann_time(pids[i], pids[i + 1]) for i in range(len(pids) - 1)]
    if not base_tofs:
        return None
    jitter = _rng.uniform(0.9, 1.1, size=len(base_tofs))
    scaled = [max(MIN_TOF, bt * j) for bt, j in zip(base_tofs, jitter)]
    total = sum(scaled)
    factor = (tf - t0) / total if total > 0 else 0.0
    times = [t0]
    cum = 0.0
    for dt in scaled:
        cum += dt
        times.append(t0 + factor * cum)
    times[-1] = tf
    return times

def _prune_leg_candidates(leg_cands, pid_list, times,
                          r_nodes, v_refs,
                          start_spec, end_spec,
                          start_keep=200, internal_keep=400, end_keep=200):
    """Trim candidate sets while keeping promising options."""
    pruned = []
    n_legs = len(leg_cands)
    for i, cands in enumerate(leg_cands):
        if not cands:
            pruned.append([])
            continue

        pid_i = pid_list[i]
        pid_j = pid_list[i + 1]
        v_ref_i = v_refs[i] if pid_i != "0" else np.zeros(3)
        v_ref_j = v_refs[i + 1] if pid_j != "0" else np.zeros(3)

        scores = []
        if i == 0:                     # start leg
            for (v_dep, _) in cands:
                node = {"type":"start","time":times[0],
                        "planet_id":pid_i,"r":r_nodes[0],
                        "v_before":v_ref_i,"v_after":v_dep}
                scores.append(_boundary_dv(node, start_spec))
            keep = start_keep
        elif i == n_legs-1:         # final leg
            for (_, v_arr) in cands:
                node = {"type":"end","time":times[-1],
                         "planet_id":pid_j,"r":r_nodes[-1],
                         "v_before":v_arr,"v_after":v_ref_j}
                scores.append(_boundary_dv(node, end_spec))
            keep = end_keep
        else:                          # intermediate GA leg
            for (v_dep, v_arr) in cands:
                scores.append(np.linalg.norm(v_dep - v_ref_i) +
                              np.linalg.norm(v_arr - v_ref_j))
            keep = internal_keep

        if len(cands) > keep:
            idx = np.argsort(scores)[:keep]
            pruned.append([cands[k] for k in idx])
        else:
            pruned.append(cands)
    return pruned

def _evaluate_trajectory(ga_seq, times, start_spec, end_spec, prob,
                         best_sofar=float('inf')):
    """Return total Δv and node list for a specific GA order and epoch vector."""
    mu_sun = prob["mu_sun"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid   = str(end_spec.get("planet_id", "0"))
    pid_list = [start_pid] + list(ga_seq) + [end_pid]

    # Resolve planetary (or reference) states
    r_nodes, v_refs = [], []
    for idx, pid in enumerate(pid_list):
        t = float(times[idx])
        if pid == "0":
            # Reference node – use explicit state from the corresponding spec
            if idx == 0:
                r = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
                v = np.asarray(start_spec.get("state_v", [0., 0., 0.]), dtype=float)
            else:
                r = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
                v = np.asarray(end_spec.get("state_v", [0., 0., 0.]), dtype=float)
        else:
            r, v = _ephem(pid, t)
            r = np.asarray(r, dtype=float)
            v = np.asarray(v, dtype=float)
        r_nodes.append(r)
        v_refs.append(v)

    # Build Lambert candidate sets per leg
    leg_cands = []
    for i in range(len(times) - 1):
        tof_sec = (float(times[i + 1]) - float(times[i])) * DAY
        if tof_sec <= 0.0:
            return float('inf'), None
        cands = _lambert_candidates(r_nodes[i], r_nodes[i + 1], tof_sec, mu_sun)
        if not cands:
            return float('inf'), None
        leg_cands.append(cands)

    # Prune candidates
    leg_cands = _prune_leg_candidates(
        leg_cands, pid_list, times,
        r_nodes, v_refs,
        start_spec, end_spec,
        start_keep=200, internal_keep=400, end_keep=200)

    n_legs = len(leg_cands)
    if n_legs == 0:
        return float('inf'), None

    # -----------------------------------------------------------------
    # Dynamic programming over legs
    # -----------------------------------------------------------------
    best_costs = []      # cost for each candidate of each leg
    back_ptr = []        # predecessor index for each candidate

    # leg 0 – start
    cost0 = []
    back0 = [None] * len(leg_cands[0])
    for j, (v_dep, _) in enumerate(leg_cands[0]):
        start_node = {"type": "start", "time": float(times[0]),
                      "planet_id": start_pid, "r": r_nodes[0],
                      "v_before": v_refs[0], "v_after": v_dep}
        c = _boundary_dv(start_node, start_spec)
        if c >= best_sofar:
            c = float('inf')
        cost0.append(c)
    best_costs.append(cost0)
    back_ptr.append(back0)

    # intermediate GA legs
    for i in range(1, n_legs):
        cur_cands = leg_cands[i]
        cur_costs = [float('inf')] * len(cur_cands)
        cur_back = [None] * len(cur_cands)

        ga_pid = ga_seq[i - 1]               # planet where this GA occurs
        ga_time = float(times[i])            # epoch of that GA

        prev_cands = leg_cands[i - 1]
        for cur_j, (v_dep_cur, _) in enumerate(cur_cands):
            best_local = float('inf')
            best_prev = None
            for prev_j, prev_cost in enumerate(best_costs[i - 1]):
                if prev_cost >= best_local or prev_cost >= best_sofar:
                    continue
                v_arr_prev = prev_cands[prev_j][1]   # inbound to GA
                dv_flyby, ok = _powered_flyby_dv(v_arr_prev, v_dep_cur,
                                                ga_pid, ga_time, prob)
                if not ok:
                    continue
                total = prev_cost + dv_flyby
                if total < best_local:
                    best_local = total
                    best_prev = prev_j
            if best_local < cur_costs[cur_j]:
                cur_costs[cur_j] = best_local
                cur_back[cur_j] = best_prev

        best_costs.append(cur_costs)
        back_ptr.append(cur_back)

    # final leg – add end boundary cost
    final_cands = leg_cands[-1]
    best_total = float('inf')
    best_last_idx = None
    for idx_last, cost_up_to_arr in enumerate(best_costs[-1]):
        if cost_up_to_arr >= best_sofar or cost_up_to_arr >= best_total:
            continue
        v_arr_last = final_cands[idx_last][1]
        end_node = {"type": "end", "time": float(times[-1]),
                    "planet_id": end_pid, "r": r_nodes[-1],
                    "v_before": v_arr_last, "v_after": v_refs[-1]}
        end_dv = _boundary_dv(end_node, end_spec)
        total = cost_up_to_arr + end_dv
        if total < best_total:
            best_total = total
            best_last_idx = idx_last

    if best_last_idx is None:
        return float('inf'), None

    # -----------------------------------------------------------------
    # Reconstruct optimal node list
    # -----------------------------------------------------------------
    chosen_idx = [None] * n_legs
    chosen_idx[-1] = best_last_idx
    for i in range(n_legs - 1, 0, -1):
        chosen_idx[i - 1] = back_ptr[i][chosen_idx[i]]

    nodes = []

    # start node
    v_dep0 = leg_cands[0][chosen_idx[0]][0]
    start_node = {"type": "start", "time": float(times[0]),
                  "planet_id": start_pid, "r": r_nodes[0],
                  "v_before": v_refs[0], "v_after": v_dep0}
    nodes.append(start_node)

    # GA nodes
    for i, ga_pid in enumerate(ga_seq):
        v_arr_i = leg_cands[i][chosen_idx[i]][1]
        v_dep_next = leg_cands[i + 1][chosen_idx[i + 1]][0]
        ga_node = {"type": "GA", "time": float(times[i + 1]),
                   "planet_id": str(ga_pid), "r": r_nodes[i + 1],
                   "v_before": v_arr_i, "v_after": v_dep_next}
        nodes.append(ga_node)

    # end node
    v_arr_last = leg_cands[-1][best_last_idx][1]
    end_node = {"type": "end", "time": float(times[-1]),
                "planet_id": end_pid, "r": r_nodes[-1],
                "v_before": v_arr_last, "v_after": v_refs[-1]}
    nodes.append(end_node)

    return best_total, nodes

# -------------------------------------------------------------------------
# Epoch refinement (deterministic + stochastic)
# -------------------------------------------------------------------------
def _refine_times(ga_seq, times0, start_spec, end_spec, prob):
    """Nelder‑Mead refinement of launch/flyby/arrival epochs."""
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])

    def obj(x):
        if np.any(x[1:] - x[:-1] <= 1e-9):
            return 1e9
        if not (t0_lo <= x[0] <= t0_hi) or not (tf_lo <= x[-1] <= tf_hi):
            return 1e9
        if np.any(x[1:] - x[:-1] < MIN_TOF):
            return 1e9
        total, _ = _evaluate_trajectory(ga_seq, list(x),
                                        start_spec, end_spec, prob)
        return total

    x0 = np.array(times0, dtype=float)
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 1500,
                                'xatol': 4e-4,
                                'fatol': 4e-4,
                                'disp': False})
    except Exception:
        return None, None
    if not res.success:
        return None, None
    total, nodes = _evaluate_trajectory(ga_seq, list(res.x),
                                        start_spec, end_spec, prob)
    return total, nodes

def _random_refine_times(ga_seq, times0, start_spec, end_spec, prob,
                         best_sofar, max_iter=12000):
    """Stochastic hill‑climbing around a good epoch vector."""
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    n = len(times0)
    best_total = best_sofar
    best_times = np.asarray(times0, dtype=float)
    best_nodes = None

    total_window = tf_hi - t0_lo
    noise_scale = max(total_window * 0.012, 1.5)   # at least ~1.5 days

    for _ in range(max_iter):
        proposal = best_times + _rng.normal(scale=noise_scale, size=n)
        proposal = np.sort(proposal)
        proposal[0] = np.clip(proposal[0], t0_lo, t0_hi)
        proposal[-1] = np.clip(proposal[-1], tf_lo, tf_hi)

        # enforce minimal leg duration
        for i in range(n - 1):
            if proposal[i + 1] - proposal[i] < MIN_TOF:
                proposal[i + 1] = proposal[i] + MIN_TOF
        if proposal[-1] > tf_hi:
            proposal[-1] = tf_hi

        total, nodes = _evaluate_trajectory(
            ga_seq, proposal.tolist(),
            start_spec, end_spec, prob,
            best_sofar=best_total)
        if total < best_total:
            best_total = total
            best_times = proposal
            best_nodes = nodes

    return best_total, best_nodes

def _refine_and_random(ga_seq, times0, start_spec, end_spec, prob, best_sofar):
    """Combine deterministic refinement and stochastic hill‑climbing."""
    det_total, det_nodes = _refine_times(ga_seq, times0,
                                          start_spec, end_spec, prob)
    if det_nodes is not None:
        best_total = det_total
        best_nodes = det_nodes
        best_times = [n["time"] for n in det_nodes]
    else:
        best_total = best_sofar
        best_nodes = None
        best_times = times0

    rand_total, rand_nodes = _random_refine_times(
        ga_seq, best_times, start_spec, end_spec, prob,
        best_sofar=best_total, max_iter=12000)

    if rand_nodes is not None and rand_total < best_total:
        return rand_total, rand_nodes
    else:
        return best_total, best_nodes

# -------------------------------------------------------------------------
# Mission‑specific GA sequence hints
# -------------------------------------------------------------------------
def _priority_ga_sequences(start_pid, end_pid, allowed_ga):
    """Generate a short list of plausible GA sequences per mission family."""
    name = problem.get("mission_name", "").lower()
    max_ga = int(problem.get("max_GA", 0))
    allowed_set = set(allowed_ga)

    seqs = []

    # Galileo VEEGA (Earth‑Venus‑Earth‑Jupiter)
    if "galileo" in name and start_pid == "3" and end_pid == "5":
        if {"2", "3"}.issubset(allowed_set):
            seqs.append(("2", "3"))                         # V‑E loop
            for repeats in range(2, max_ga // 2 + 3):
                seq = tuple(["2", "3"] * repeats)
                if len(seq) <= max_ga:
                    seqs.append(seq)

    # Cassini‑style (Earth‑Venus‑Earth‑Jupiter‑Saturn)
    if "cassini" in name and start_pid == "3" and end_pid == "6":
        base = ("2", "3", "5")
        if all(p in allowed_set for p in base) and len(base) <= max_ga:
            seqs.append(base)
        if {"2", "3"}.issubset(allowed_set):
            for k in range(1, max_ga // 2 + 2):
                prefix = tuple(["2", "3"] * k)
                if len(prefix) + len(base) <= max_ga:
                    seqs.append(prefix + base)

    # Voyager‑like outer‑planet tour
    if "voyager" in name and start_pid == "3":
        outer = [p for p in ["5", "6", "7", "8"] if p in allowed_set]
        for L in range(1, len(outer) + 1):
            seqs.append(tuple(outer[:L]))
        if {"2", "3"}.issubset(allowed_set):
            for k in range(1, max_ga // 2 + 2):
                prefix = tuple(["2", "3"] * k)
                for L in range(1, len(outer) + 1):
                    seq = prefix + tuple(outer[:L])
                    if len(seq) <= max_ga:
                        seqs.append(seq)

    # generic fallback – single flyby of each allowed planet
    for p in allowed_ga:
        seqs.append((p,))

    # deduplicate while preserving order
    seen = set()
    uniq = []
    for s in seqs:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

# -------------------------------------------------------------------------
# Neighbor sequence stochastic optimisation
# -------------------------------------------------------------------------
def _neighbor_optimize(best_seq, best_times, start_spec, end_spec, prob,
                       current_best_total, start_time, wall_limit):
    """Local search that perturbs GA sequence locally and re‑optimises."""
    allowed_ga = [str(p) for p in prob.get("allowed_GA_planets", [])]
    max_ga_allowed = int(prob.get("max_GA", 0))
    max_nodes = int(prob.get("max_nodes", 100))

    best_seq = list(best_seq) if best_seq else []
    best_times = list(best_times) if best_times else []

    attempts = 0
    while True:
        if time.time() - start_time >= wall_limit - 4.0:
            break
        if attempts >= 5000:
            break
        attempts += 1

        seq = list(best_seq)
        times = list(best_times)
        n_ga = len(seq)

        # choose operation
        ops = []
        if n_ga > 0:
            ops.extend(['replace', 'swap'])
        if n_ga < max_ga_allowed and n_ga + 2 <= max_nodes:
            ops.append('insert')
        if n_ga > 0:
            ops.append('delete')
        if not ops:
            break
        op = _rng.choice(ops)

        if op == 'replace':
            idx = _rng.integers(0, n_ga)
            seq[idx] = _rng.choice(allowed_ga)
        elif op == 'swap':
            i, j = _rng.choice(n_ga, size=2, replace=False)
            seq[i], seq[j] = seq[j], seq[i]
        elif op == 'insert':
            pos = _rng.integers(0, n_ga + 1)
            seq.insert(pos, _rng.choice(allowed_ga))
            # Insert a new time between neighbours
            lo = times[pos] + MIN_TOF
            hi = times[pos + 1] - MIN_TOF
            if lo >= hi:
                continue
            new_t = float(_rng.uniform(lo, hi))
            times.insert(pos + 1, new_t)
        elif op == 'delete':
            del_idx = _rng.integers(0, n_ga)
            seq.pop(del_idx)
            times.pop(del_idx + 1)

        # check windows and monotonicity
        t0_lo, t0_hi = _time_window(start_spec["time"])
        tf_lo, tf_hi = _time_window(end_spec["time"])
        if not (t0_lo <= times[0] <= t0_hi) or not (tf_lo <= times[-1] <= tf_hi):
            continue
        illegal = False
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] < MIN_TOF:
                illegal = True
                break
        if illegal:
            continue

        total, _ = _evaluate_trajectory(seq, times,
                                         start_spec, end_spec, prob,
                                         best_sofar=current_best_total)
        if total >= current_best_total:
            continue

        refined_total, refined_nodes = _refine_and_random(
            seq, times, start_spec, end_spec, prob, current_best_total)
        if refined_nodes is not None and refined_total < current_best_total:
            current_best_total = refined_total
            best_seq = seq
            best_times = [n["time"] for n in refined_nodes]

    return current_best_total, best_seq, best_times

# -------------------------------------------------------------------------
# Differential‑Evolution global optimisation for epoch schedule
# -------------------------------------------------------------------------
def _de_optimize(ga_seq, start_spec, end_spec, prob,
                 best_sofar, time_limit):
    """Global DE optimisation of launch/flyby/arrival epochs (constrained)."""
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])
    n_ga = len(ga_seq)

    # bounds: launch time, interior fractions, arrival time
    bounds = [(t0_lo, t0_hi)] + [(0.0, 1.0)] * n_ga + [(tf_lo, tf_hi)]
    start = time.time()

    def obj(x):
        if time.time() - start > time_limit:
            return 1e9
        t0 = float(x[0])
        tf = float(x[-1])
        if tf <= t0:
            return 1e9
        fracs = np.sort(x[1:-1])
        times = [t0] + list(t0 + fracs * (tf - t0)) + [tf]
        total, _ = _evaluate_trajectory(ga_seq, times,
                                        start_spec, end_spec, prob,
                                        best_sofar=best_sofar)
        if total == float('inf'):
            return 1e9
        return total

    try:
        res = differential_evolution(
            obj, bounds,
            maxiter=80, popsize=30,
            mutation=(0.5, 1.0), recombination=0.7,
            polish=True, seed=_SEED, updating='deferred')
    except Exception:
        return best_sofar, None
    total, nodes = _evaluate_trajectory(ga_seq, list([float(res.x[0])] +
                                                    list(res.x[1:-1] * (float(res.x[-1]) - float(res.x[0]))) +
                                                    [float(res.x[-1])]),
                                        start_spec, end_spec, prob)
    if total < best_sofar:
        return total, nodes
    else:
        return best_sofar, None

# -------------------------------------------------------------------------
# Main topology search
# -------------------------------------------------------------------------
def _search_best_trajectory(prob):
    start_spec = prob["start"]
    end_spec   = prob["end"]
    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])

    allowed_ga = [str(p) for p in prob.get("allowed_GA_planets", [])]
    max_ga_allowed = int(prob.get("max_GA", 0))
    max_nodes = int(prob.get("max_nodes", 100))

    max_ga = min(max_ga_allowed, len(allowed_ga), max_nodes - 2)
    max_ga = max(0, max_ga)

    best_total = float('inf')
    best_nodes = None
    best_seq   = None
    best_times = None

    wall_limit = float(prob.get("timeout_seconds", 300.0)) * TIME_MARGIN
    start_time = time.time()
    target_dv = float(prob.get("target_dv", 0.0))

    # ----------- mission‑specific hints -------------
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid   = str(end_spec.get("planet_id", "0"))
    priority_seqs = _priority_ga_sequences(start_pid, end_pid, allowed_ga)
    priority_seqs = [seq for seq in priority_seqs if len(seq) <= max_ga]

    examined = set()

    def _process_ga_sequence(ga_seq, is_priority):
        nonlocal best_total, best_nodes, best_seq, best_times
        n_ga = len(ga_seq)
        if n_ga + 2 > max_nodes:
            return

        base_samples = SAMPLE_PLAN.get(n_ga, 2000)
        if is_priority:
            base_samples = int(min(base_samples * PRIORITY_FACTOR, 60000))
        else:
            if n_ga >= 4:
                base_samples = int(base_samples * 0.6)

        for _ in range(base_samples):
            if time.time() - start_time > wall_limit:
                return

            # random launch / arrival epochs within windows
            t0 = float(_rng.uniform(t0_lo, t0_hi))
            tf = float(_rng.uniform(tf_lo, tf_hi))
            if tf - t0 < MIN_TOF * (n_ga + 1):
                continue

            # epoch vector (guided or uniform)
            if n_ga and _rng.random() < 0.75:
                times = _guided_times(t0, tf, ga_seq, start_pid, end_pid)
                if times is None:
                    continue
            else:
                fracs = np.sort(_rng.random(n_ga))
                times = [t0] + [t0 + f * (tf - t0) for f in fracs] + [tf]

            # sanity check
            if any(times[i + 1] - times[i] < MIN_TOF for i in range(len(times) - 1)):
                continue

            total, nodes = _evaluate_trajectory(
                ga_seq, times,
                start_spec, end_spec, prob,
                best_sofar=best_total)
            if total < best_total:
                refined_total, refined_nodes = _refine_and_random(
                    ga_seq, times, start_spec, end_spec, prob, best_total)
                if refined_nodes is not None and refined_total < best_total:
                    best_total = refined_total
                    best_nodes = refined_nodes
                    best_seq = ga_seq
                    best_times = [n["time"] for n in refined_nodes]
                else:
                    best_total = total
                    best_nodes = nodes
                    best_seq = ga_seq
                    best_times = times

                if target_dv > 0.0 and best_total <= target_dv:
                    return  # early success

    # ---------- Phase 1 – priority sequences ----------
    for seq in priority_seqs:
        tup = tuple(seq)
        if tup in examined:
            continue
        examined.add(tup)
        _process_ga_sequence(seq, is_priority=True)

    # ---------- Phase 2 – exhaustive short sequences (≤3) ----------
    if max_ga >= 1:
        max_ex_len = min(3, max_ga)
        _orig_sample_plan = SAMPLE_PLAN.copy()
        for n in range(1, max_ex_len + 1):
            SAMPLE_PLAN[n] = 2500   # moderate budget for exhaustive loops
        for n in range(1, max_ex_len + 1):
            for seq in itertools.product(allowed_ga, repeat=n):
                tup = tuple(seq)
                if tup in examined:
                    continue
                examined.add(tup)
                _process_ga_sequence(seq, is_priority=False)
                if time.time() - start_time > wall_limit:
                    break
            if time.time() - start_time > wall_limit:
                break
        SAMPLE_PLAN.update(_orig_sample_plan)

    # ---------- Phase 3 – generic enumeration (longer first) ----------
    for n_ga in range(max_ga, -1, -1):
        if time.time() - start_time > wall_limit:
            break

        if n_ga == 0:
            ga_sequences = [()]
        else:
            total_comb = len(allowed_ga) ** n_ga
            if total_comb <= 300:
                ga_sequences = list(itertools.product(allowed_ga, repeat=n_ga))
            else:
                ga_sequences = [tuple(_rng.choice(allowed_ga, size=n_ga))
                                for _ in range(300)]

        for ga_seq in ga_sequences:
            tup = tuple(ga_seq)
            if tup in examined:
                continue
            examined.add(tup)

            _process_ga_sequence(ga_seq, is_priority=False)

            if time.time() - start_time > wall_limit:
                break

    # ---------- Phase 4 – final polishing ----------
    if best_nodes is not None and best_seq is not None and best_times is not None:
        remaining = wall_limit - (time.time() - start_time)
        if remaining > 5.0:
            final_total, final_nodes = _refine_and_random(
                best_seq, best_times, start_spec, end_spec, prob, best_total)
            if final_nodes is not None and final_total < best_total:
                best_total = final_total
                best_nodes = final_nodes
                best_times = [n["time"] for n in final_nodes]

        remaining = wall_limit - (time.time() - start_time)
        if remaining > 7.0:
            neigh_total, neigh_seq, neigh_times = _neighbor_optimize(
                best_seq, best_times,
                start_spec, end_spec, prob,
                best_total, start_time, wall_limit)
            if neigh_total < best_total:
                total2, nodes2 = _refine_and_random(
                    neigh_seq, neigh_times,
                    start_spec, end_spec, prob, neigh_total)
                if nodes2 is not None and total2 < neigh_total:
                    best_total = total2
                    best_nodes = nodes2
                    best_seq = neigh_seq
                    best_times = neigh_times
                else:
                    best_total = neigh_total
                    best_nodes = _evaluate_trajectory(
                        neigh_seq, neigh_times, start_spec, end_spec, prob)[1]
                    best_seq = neigh_seq
                    best_times = neigh_times

        remaining = wall_limit - (time.time() - start_time)
        if remaining > 5.0 and best_seq is not None:
            de_total, de_nodes = _de_optimize(
                best_seq, start_spec, end_spec, prob,
                best_total, remaining)
            if de_nodes is not None and de_total < best_total:
                best_total = de_total
                best_nodes = de_nodes
                best_times = [n["time"] for n in de_nodes]

    return best_nodes

# -------------------------------------------------------------------------
# Optional DSM insertion (kept lightweight)
# -------------------------------------------------------------------------
def _insert_dsms_and_optimize(nodes, prob):
    """Greedy insertion of up to max_DSM deep‑space maneuvers."""
    if nodes is None:
        return None
    max_dsm = int(prob.get("max_DSM", 0))
    max_nodes = int(prob.get("max_nodes", 100))
    if max_dsm <= 0 or len(nodes) + max_dsm > max_nodes:
        return nodes

    mu_sun = prob["mu_sun"]
    start_spec = prob["start"]
    end_spec   = prob["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid   = str(end_spec.get("planet_id", "0"))

    # pure (non‑DSM) nodes
    pure_nodes = [n for n in nodes if n["type"] in ("start", "GA", "end")]
    ga_nodes = [n for n in pure_nodes if n["type"] == "GA"]
    n_ga = len(ga_nodes)
    n_legs = n_ga + 1

    t_nodes = np.array([float(n["time"]) for n in pure_nodes])
    r_nodes = np.array([np.asarray(n["r"], dtype=float) for n in pure_nodes])

    def _total_with_dsms(dsm_info):
        cur_t = list(t_nodes)
        cur_r = [np.asarray(r, dtype=float) for r in r_nodes]

        # insert DSMs
        for d in sorted(dsm_info, key=lambda x: x["leg"]):
            leg = d["leg"]
            cur_t.insert(leg + 1, d["t"])
            cur_r.insert(leg + 1, np.asarray(d["r"], dtype=float))

        seg_vs = []
        for i in range(len(cur_t) - 1):
            ti, ri = cur_t[i], cur_r[i]
            tj, rj = cur_t[i + 1], cur_r[i + 1]
            tof = max(1e-3, (tj - ti) * DAY)
            try:
                v_dep, v_arr = tools.lambert(
                    ri, rj, tof, mu_sun,
                    prograde=True, lowpath=True, M=0)
            except Exception:
                return float('inf')
            seg_vs.append((np.asarray(v_dep), np.asarray(v_arr)))

        total = 0.0
        out_nodes = []

        # start node
        if start_pid != "0":
            v_before_start = _ephem(start_pid, float(cur_t[0]))[1]
        else:
            v_before_start = np.asarray(start_spec.get("state_v", [0., 0., 0.]))
        start_node = {"type": "start",
                     "time": float(cur_t[0]),
                     "planet_id": start_pid,
                     "r": cur_r[0],
                     "v_before": v_before_start,
                     "v_after": seg_vs[0][0]}
        total += _boundary_dv(start_node, start_spec)
        out_nodes.append(start_node)

        pure_idx = 0
        # interior nodes (GA or DSM)
        for i in range(1, len(cur_t) - 1):
            v_arr = seg_vs[i - 1][1]
            if (pure_idx + 1 < len(pure_nodes) and
                abs(cur_t[i] - pure_nodes[pure_idx + 1]["time"]) < 1e-6):
                pure_node = pure_nodes[pure_idx + 1]
                v_before = v_arr
                v_after = seg_vs[i][0]
                dv_flyby, feas = _powered_flyby_dv(v_before, v_after,
                                                  pure_node["planet_id"],
                                                  pure_node["time"], prob)
                if not feas:
                    return float('inf')
                total += dv_flyby
                out_nodes.append({"type": "GA",
                                 "time": pure_node["time"],
                                 "planet_id": pure_node["planet_id"],
                                 "r": pure_node["r"],
                                 "v_before": v_before,
                                 "v_after": v_after})
                pure_idx += 1
            else:
                out_nodes.append({"type": "DSM",
                                 "time": float(cur_t[i]),
                                 "planet_id": "0",
                                 "r": cur_r[i].tolist(),
                                 "v_before": v_arr,
                                 "v_after": seg_vs[i][0].tolist()})
                total += float(np.linalg.norm(seg_vs[i][0] - v_arr))

        # End node
        if end_pid != "0":
            v_after_end = _ephem(end_pid, float(cur_t[-1]))[1]
        else:
            v_after_end = np.asarray(end_spec.get("state_v", [0., 0., 0.]))
        end_node = {"type": "end",
                    "time": float(cur_t[-1]),
                    "planet_id": end_pid,
                    "r": cur_r[-1],
                    "v_before": seg_vs[-1][1],
                    "v_after": v_after_end}
        total += _boundary_dv(end_node, end_spec)
        out_nodes.append(end_node)

        _total_with_dsms.last_nodes = out_nodes
        return total

    # greedy insertion loop
    current_dsms = []
    current_total = _total_with_dsms(current_dsms)
    remaining = max_dsm

    while remaining > 0:
        best_improve = 0.0
        best_candidate = None

        for leg_idx in range(n_legs):
            if any(d["leg"] == leg_idx for d in current_dsms):
                continue
            ti, ri = t_nodes[leg_idx], r_nodes[leg_idx]
            tj, rj = t_nodes[leg_idx + 1], r_nodes[leg_idx + 1]
            t_guess = ti + (tj - ti) * 0.5
            r_guess = (ri + rj) * 0.5

            def obj(x):
                t = float(np.clip(x[0], ti + MIN_TOF, tj - MIN_TOF))
                r = np.array([x[1], x[2], x[3]])
                trial = current_dsms + [{"leg": leg_idx, "t": t, "r": r}]
                return _total_with_dsms(trial)

            x0 = np.concatenate(([t_guess], r_guess))
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 350,
                                        'xatol': 9e-4,
                                        'fatol': 9e-4,
                                        'disp': False})
            except Exception:
                continue
            if not res.success:
                continue
            new_total = obj(res.x)
            improve = current_total - new_total
            if improve > best_improve:
                best_improve = improve
                best_candidate = {"leg": leg_idx,
                                  "t": float(res.x[0]),
                                  "r": np.array(res.x[1:4])}
        if best_candidate is None or best_improve <= 1e-6:
            break
        current_dsms.append(best_candidate)
        current_total -= best_improve
        remaining -= 1

    final_total = _total_with_dsms(current_dsms)
    final_nodes = getattr(_total_with_dsms, "last_nodes", None)
    pure_total = _compute_total_dv(pure_nodes, prob)

    if final_nodes is not None and final_total < pure_total:
        return _format_nodes(final_nodes)
    else:
        return _format_nodes(pure_nodes)

# -------------------------------------------------------------------------
# Utility: total Δv for a node list
# -------------------------------------------------------------------------
def _compute_total_dv(nodes, prob):
    total = 0.0
    start_spec = prob["start"]
    end_spec   = prob["end"]
    for n in nodes:
        if n["type"] == "start":
            total += _boundary_dv(n, start_spec)
        elif n["type"] == "end":
            total += _boundary_dv(n, end_spec)
        elif n["type"] == "GA":
            dv, _ = _powered_flyby_dv(
                np.asarray(n["v_before"]),
                np.asarray(n["v_after"]),
                n["planet_id"], n["time"], prob)
            total += dv
        elif n["type"] == "DSM":
            total += float(np.linalg.norm(np.asarray(n["v_after"]) -
                                         np.asarray(n["v_before"])))
    return total

# -------------------------------------------------------------------------
# Convert NumPy arrays in nodes to plain Python lists
# -------------------------------------------------------------------------
def _format_nodes(nodes):
    if nodes is None:
        return None
    for n in nodes:
        for key in ("r", "v_before", "v_after"):
            n[key] = np.asarray(n[key], dtype=float).tolist()
    return nodes

# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
def run_code():
    record.event(f"mission={problem.get('id')} search_start")
    best_nodes = _search_best_trajectory(problem)

    # -----------------------------------------------------------------
    # Fallback: simple direct transfer if nothing was found
    # -----------------------------------------------------------------
    if best_nodes is None:
        record.event("fallback_no_solution")
        start_spec = problem["start"]
        end_spec   = problem["end"]
        t0_lo, _ = _time_window(start_spec["time"])
        tf_lo, _ = _time_window(end_spec["time"])
        start_pid = str(start_spec.get("planet_id", "0"))
        end_pid   = str(end_spec.get("planet_id", "0"))

        if start_pid != "0":
            r0, v0_ref = tools.ephem(start_pid, float(t0_lo))
        else:
            r0 = np.asarray(start_spec.get("state_r", [0., 0., 0.]), dtype=float)
            v0_ref = np.asarray(start_spec.get("state_v", [0., 0., 0.]), dtype=float)

        if end_pid != "0":
            r1, v1_ref = tools.ephem(end_pid, float(tf_lo))
        else:
            r1 = np.asarray(end_spec.get("state_r", [0., 0., 0.]), dtype=float)
            v1_ref = np.asarray(end_spec.get("state_v", [0., 0., 0.]), dtype=float)

        try:
            v_dep, v_arr = tools.lambert(
                r0, r1, (tf_lo - t0_lo) * DAY,
                problem["mu_sun"], prograde=True, lowpath=True, M=0)
        except Exception:
            v_dep = np.zeros(3)
            v_arr = np.zeros(3)

        best_nodes = [
            {"type": "start",
             "time": float(t0_lo),
             "planet_id": start_pid,
             "r": r0.tolist(),
             "v_before": v0_ref.tolist(),
             "v_after": np.asarray(v_dep).tolist()},
            {"type": "end",
             "time": float(tf_lo),
             "planet_id": end_pid,
             "r": r1.tolist(),
             "v_before": np.asarray(v_arr).tolist(),
             "v_after": v1_ref.tolist()}
        ]

    # -----------------------------------------------------------------
    # Optional DSM insertion (if budget permits)
    # -----------------------------------------------------------------
    max_dsm = int(problem.get("max_DSM", 0))
    max_nodes = int(problem.get("max_nodes", 100))
    if max_dsm > 0 and best_nodes is not None:
        refined = _insert_dsms_and_optimize(best_nodes, problem)
        if refined is not None and len(refined) <= max_nodes:
            if _compute_total_dv(refined, problem) < _compute_total_dv(best_nodes, problem):
                record.event("refinement_success")
                best_nodes = refined
            else:
                record.event("refinement_skipped")
        else:
            record.event("refinement_skipped")
    else:
        record.event("refinement_skipped")

    record.event("search_complete")
    return _format_nodes(best_nodes)
# EVOLVE-BLOCK-END
