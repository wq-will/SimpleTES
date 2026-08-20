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
import itertools, time, math
from scipy.optimize import minimize, differential_evolution, basinhopping
from functools import lru_cache

# ----------------------------------------------------------------------
# Global constants & deterministic RNG
# ----------------------------------------------------------------------
DAY = 86400.0                         # seconds per day
MIN_TOF_DAYS = 0.5                    # minimum leg duration (days)
MAX_REV = 400                         # max revolutions for Lambert solver
RNG_SEED = 20231201                   # deterministic seed
_rng = np.random.default_rng(RNG_SEED)

_START_TIME = time.time()
_TIMEOUT = float(problem.get("timeout_seconds", 300.0))

def _time_remaining():
    """Seconds left before the global timeout."""
    return max(0.0, _TIMEOUT - (time.time() - _START_TIME))

# ----------------------------------------------------------------------
# Cached ephemerides and two‑body propagation
# ----------------------------------------------------------------------
@lru_cache(maxsize=8000)
def _planet_state(pid, mjd):
    pid = str(pid)
    if pid == "0" or int(pid) == 0:
        for spec in (problem.get("start", {}), problem.get("end", {})):
            if "state_r" in spec and "state_v" in spec:
                return (np.asarray(spec["state_r"], dtype=float),
                        np.asarray(spec["state_v"], dtype=float))
        return np.zeros(3), np.zeros(3)
    r, v = tools.ephem(pid, float(mjd))
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float)

@lru_cache(maxsize=250000)
def _lambert_cache(r0_bytes, r1_bytes, tof):
    r0 = np.frombuffer(r0_bytes, dtype=np.float64)
    r1 = np.frombuffer(r1_bytes, dtype=np.float64)
    mu = problem["mu_sun"]
    for M in range(0, MAX_REV + 1):
        for low in (True, False):
            for pro in (True, False):
                try:
                    v0, v1 = tools.lambert(r0, r1, tof, mu,
                                            prograde=pro,
                                            lowpath=low,
                                            M=M)
                    return (np.asarray(v0, dtype=float),
                            np.asarray(v1, dtype=float))
                except Exception:
                    continue
    return (None, None)

def _lambert_solve(r0, r1, tof_sec):
    """Lambert wrapper with caching."""
    v0, v1 = _lambert_cache(r0.tobytes(), r1.tobytes(), float(tof_sec))
    return v0, v1

# ----------------------------------------------------------------------
# Δv helpers & windows
# ----------------------------------------------------------------------
def _powered_flyby_dv(v_arr, v_dep, pid, t, prob):
    pid = str(pid)
    mu_p = float(prob["planet_mu"][pid])
    R_p = float(prob["planet_radius"][pid])
    min_alt_cfg = prob.get("flyby", {}).get("min_altitude_km", {})
    min_alt = float(min_alt_cfg.get(pid, 200.0))
    try:
        _, dv, feas = tools.powered_flyby(np.asarray(v_arr),
                                          np.asarray(v_dep),
                                          tools.ephem(pid, float(t))[1],
                                          mu_p,
                                          R_p + min_alt)
        return float(dv), bool(feas)
    except Exception:
        return float("inf"), False

def _periapsis_dv(vinf, mu_p, R_p, h_factor, T_days):
    r_peri = R_p * (1.0 + h_factor)
    T_sec = float(T_days) * DAY
    two_mu_r = 2.0 * mu_p / r_peri
    term = (4.0 * np.pi**2 * mu_p**2 / T_sec**2) ** (1.0 / 3.0)
    inner = max(two_mu_r - term, 0.0)
    return float(np.sqrt(vinf * vinf + two_mu_r) - np.sqrt(inner))

def _piecewise_linear(x, breakpoints):
    bps = sorted(breakpoints, key=lambda p: float(p[0]))
    x = float(x)
    if x <= float(bps[0][0]): return float(bps[0][1])
    if x >= float(bps[-1][0]): return float(bps[-1][1])
    for i in range(len(bps) - 1):
        x0, y0 = float(bps[i][0]), bps[i][1]
        x1, y1 = float(bps[i + 1][0]), bps[i + 1][1]
        if x0 <= x <= x1:
            if x1 == x0: return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(bps[-1][1])

def _boundary_dv(node, spec):
    typ = spec["type"]
    if typ == "piecewise_linear":
        dv = float(np.linalg.norm(np.asarray(node["v_after"], float) -
                                 np.asarray(node["v_before"], float)))
        return _piecewise_linear(dv, spec["breakpoints"])
    if typ == "periapsis_maneuver":
        pid = str(spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        hf = float(spec["h_factor"])
        Td = float(spec["T_days"])
        _, v_planet = tools.ephem(pid, float(node["time"]))
        if node["type"] == "start":
            vinf = float(np.linalg.norm(np.asarray(node["v_after"], float) - v_planet))
        else:
            vinf = float(np.linalg.norm(np.asarray(node["v_before"], float) - v_planet))
        return _periapsis_dv(vinf, mu_p, R_p, hf, Td)
    return 1e12

def _time_window(spec):
    if spec["kind"] == "window":
        return float(spec["lo"]), float(spec["hi"])
    return float(spec["value"]), float(spec["value"])

# ----------------------------------------------------------------------
# Total Δv for a whole node list
# ----------------------------------------------------------------------
def _total_dv(nodes, limit=None):
    if not nodes:
        return float("inf")
    total = _boundary_dv(nodes[0], problem["start"])
    if limit is not None and total > limit:
        return total
    for nd in nodes[1:-1]:
        if nd["type"] == "GA":
            dv, feas = _powered_flyby_dv(nd["v_before"], nd["v_after"],
                                         nd["planet_id"], nd["time"], problem)
            if not feas:
                return float("inf")
            total += dv
        elif nd["type"] == "DSM":
            total += float(np.linalg.norm(np.asarray(nd["v_before"], float) -
                                         np.asarray(nd["v_after"], float)))
        if limit is not None and total > limit:
            return total
    total += _boundary_dv(nodes[-1], problem["end"])
    return total

def _format_nodes(nodes):
    for n in nodes:
        for k in ("r", "v_before", "v_after"):
            n[k] = np.asarray(n[k], dtype=float).tolist()
    return nodes

# ----------------------------------------------------------------------
# Simple time‑sequence generators
# ----------------------------------------------------------------------
def _linear_spaced_times(lo_start, hi_start, lo_end, hi_end, n_ga):
    times = [float(lo_start)]
    total_len = n_ga + 2
    for i in range(1, n_ga + 1):
        frac = i / (total_len - 1)
        ti = lo_start + frac * (hi_end - lo_start)
        times.append(float(ti))
    times.append(float(hi_end))
    for i in range(1, len(times)):
        if times[i] - times[i-1] < MIN_TOF_DAYS:
            times[i] = times[i-1] + MIN_TOF_DAYS
    if times[-1] > hi_end:
        times[-1] = hi_end
    return times

def _random_times(lo_start, hi_start, lo_end, hi_end, n_ga):
    n_legs = n_ga + 1
    min_total = MIN_TOF_DAYS * n_legs
    for _ in range(12000):
        t0 = _rng.uniform(lo_start, hi_start)
        if t0 + min_total > hi_end:
            continue
        tf = _rng.uniform(max(lo_end, t0 + min_total), hi_end)
        if n_ga == 0:
            return [float(t0), float(tf)]
        fracs = np.sort(_rng.random(n_ga))
        inner = t0 + (tf - t0) * fracs
        prev = t0
        ok = True
        for ti in inner:
            if ti - prev < MIN_TOF_DAYS:
                ok = False
                break
            prev = ti
        if not ok or (tf - prev) < MIN_TOF_DAYS:
            continue
        return [float(t0)] + inner.tolist() + [float(tf)]
    return None

# ----------------------------------------------------------------------
# Evaluate a pure GA sequence (no DSM)
# ----------------------------------------------------------------------
def _evaluate_ga_sequence(ga_seq, times, best_limit=None):
    start_spec = problem["start"]
    end_spec   = problem["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid   = str(end_spec.get("planet_id", "0"))
    pid_list = [start_pid] + list(ga_seq) + [end_pid]

    r_vecs, v_ref = [], []
    for pid, t in zip(pid_list, times):
        r, v = _planet_state(pid, t)
        r_vecs.append(r)
        v_ref.append(v)

    dep, arr = [], []
    for i in range(len(times) - 1):
        tof = (times[i+1] - times[i]) * DAY
        if tof <= 0:
            return None, None
        v_dep, v_arr = _lambert_solve(r_vecs[i], r_vecs[i+1], tof)
        if v_dep is None:
            return None, None
        dep.append(v_dep); arr.append(v_arr)

    nodes = []

    nodes.append({
        "type": "start",
        "time": float(times[0]),
        "planet_id": start_pid,
        "r": r_vecs[0],
        "v_before": v_ref[0],
        "v_after": dep[0]
    })
    for idx, pid in enumerate(ga_seq):
        nodes.append({
            "type": "GA",
            "time": float(times[idx+1]),
            "planet_id": pid,
            "r": r_vecs[idx+1],
            "v_before": arr[idx],
            "v_after": dep[idx+1]
        })
    nodes.append({
        "type": "end",
        "time": float(times[-1]),
        "planet_id": end_pid,
        "r": r_vecs[-1],
        "v_before": arr[-1],
        "v_after": v_ref[-1]
    })
    total = _total_dv(nodes, best_limit)
    if best_limit is not None and total > best_limit:
        return None, None
    return nodes, total

# ----------------------------------------------------------------------
# Refine GA times (local Nelder‑Mead)
# ----------------------------------------------------------------------
def _refine_ga_times(pure_nodes, prob):
    if pure_nodes is None:
        return None
    lo_s, hi_s = _time_window(prob["start"]["time"])
    lo_e, hi_e = _time_window(prob["end"]["time"])
    ga_seq = [str(node["planet_id"]) for node in pure_nodes if node["type"] == "GA"]
    times0 = [float(node["time"]) for node in pure_nodes]

    def _obj(x):
        x = np.asarray(x, dtype=float)
        x[0] = np.clip(x[0], lo_s, hi_s)
        x[-1] = np.clip(x[-1], lo_e, hi_e)
        if np.any(x[1:] - x[:-1] < MIN_TOF_DAYS):
            return 1e9
        nd, tot = _evaluate_ga_sequence(tuple(ga_seq), x.tolist())
        if nd is None or tot is None or not np.isfinite(tot):
            return 1e9
        return tot

    x0 = np.array(times0, dtype=float)
    scales = [5.0] * len(x0)
    simplex = [x0]
    for j in range(len(x0)):
        dx = np.zeros_like(x0)
        dx[j] = scales[j]
        simplex.append(x0 + dx)
    try:
        res = minimize(_obj, x0, method="Nelder-Mead",
                       options={"maxiter": 3000 * len(x0),
                                "xatol": 1e-4, "fatol": 1e-5,
                                "initial_simplex": np.array(simplex[:len(x0)+1])})
    except Exception:
        return pure_nodes
    if not res.success:
        return pure_nodes
    nd_opt, tot_opt = _evaluate_ga_sequence(tuple(ga_seq), res.x.tolist())
    if nd_opt is None:
        return pure_nodes
    if tot_opt < _total_dv(pure_nodes) - 1e-8:
        return nd_opt
    return pure_nodes

# ----------------------------------------------------------------------
# Global DE for GA timings
# ----------------------------------------------------------------------
def _de_optimize_seq(seq, init_times, start_win, end_win):
    n = len(seq)
    bounds = [(start_win[0], start_win[1])]
    for _ in range(n):
        bounds.append((start_win[0], end_win[1]))
    bounds.append((end_win[0], end_win[1]))

    def _obj(x):
        if np.any(x[1:] - x[:-1] < MIN_TOF_DAYS):
            return 1e9
        nd, tot = _evaluate_ga_sequence(tuple(seq), x.tolist())
        if nd is None or tot is None or not np.isfinite(tot):
            return 1e9
        return tot

    # allocate a generous budget – will be trimmed by remaining time
    if _time_remaining() > 120:
        maxiter, popsize = 800, 140
    elif _time_remaining() > 80:
        maxiter, popsize = 600, 110
    elif _time_remaining() > 40:
        maxiter, popsize = 400, 80
    else:
        maxiter, popsize = 250, 45

    try:
        res = differential_evolution(_obj, bounds,
                                     maxiter=maxiter, popsize=popsize,
                                     seed=RNG_SEED, polish=True,
                                     disp=False, workers=1)
    except Exception:
        return None, None
    if not res.success:
        return None, None
    nd_opt, ndv = _evaluate_ga_sequence(tuple(seq), res.x.tolist())
    return nd_opt, ndv

# ----------------------------------------------------------------------
# Build all admissible GA sequences (respect node budget)
# ----------------------------------------------------------------------
def _generate_all_sequences():
    max_nodes = int(problem.get("max_nodes", 10))
    max_ga_allowed = int(problem.get("max_GA", len(problem.get("allowed_GA_planets", []))))
    allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga_by_nodes = max_nodes - 2
    max_ga = min(max_ga_allowed, max_ga_by_nodes, len(allowed))
    seqs = []
    for n in range(max_ga + 1):
        if n == 0:
            seqs.append(())
        else:
            seqs.extend(itertools.product(allowed, repeat=n))
    return seqs

# ----------------------------------------------------------------------
# Known‑mission templates (quick shortcuts)
# ----------------------------------------------------------------------
def _lookup_known():
    key = (problem.get("id", "") or problem.get("reference_family", "")).lower()
    if "cassini" in key:
        return {"candidates": [
            (("2","2","3","5"), [50745.4,50941.2,51355.0,51408.9,51927.6,53349.9]),
            (("2","3","5"), None),
            (("3","5"), None)
        ]}
    if "galileo" in key:
        return {"candidates": [
            (("2","3","5"), None),
            (("2","2","3","5"), None)
        ]}
    if "juno" in key:
        return {"candidates": [ (("5",), None) ]}
    if "voyager" in key:
        return {"candidates": [
            (("2","2","3","5","6","7","8"), None),
            (("2","3","5","6","7","8"), None),
            (("3","5","6","7","8"), None),
            (("2","5","6","7","8"), None)
        ]}
    if "new_horizons" in key or "nh" in key:
        return {"candidates": [ (("5","7"), None) ]}
    if "messenger" in key:
        return {"candidates": [
            (("2","2","5","4"), None),
            (("2","5","4"), None)
        ]}
    if "inner" in key:
        return {"candidates": [
            (("2","3","4"), None),
            (("2","3"), None)
        ]}
    if "outer" in key:
        return {"candidates": [
            (("5","6","7","8"), None),
            (("5","6","7"), None),
            (("5","6"), None)
        ]}
    return None

# ----------------------------------------------------------------------
# DSM utilities
# ----------------------------------------------------------------------
def _leg_weights_from_nodes(pure_nodes):
    w = [0.0] * (len(pure_nodes) - 1)
    for i, nd in enumerate(pure_nodes):
        if nd["type"] != "GA": continue
        dv, feas = _powered_flyby_dv(nd["v_before"], nd["v_after"],
                                      nd["planet_id"], nd["time"], problem)
        if not feas: continue
        if i-1 >= 0: w[i-1] += dv
        if i < len(pure_nodes)-1: w[i] += dv
    return w

def _generate_dsm_masks(n_legs, max_dsm, leg_weights=None):
    if max_dsm <= 0:
        return [[False] * n_legs]
    total = sum(math.comb(n_legs, k) for k in range(0, max_dsm+1))
    if total <= 5000:
        masks = []
        for k in range(max_dsm+1):
            for combo in itertools.combinations(range(n_legs), k):
                m=[False]*n_legs
                for i in combo: m[i]=True
                masks.append(m)
        return masks
    # Weighted reduction when combinatorial explosion
    if leg_weights is None:
        leg_weights = [0.0]*n_legs
    ranked = sorted(range(n_legs), key=lambda i: leg_weights[i], reverse=True)
    pool = ranked[:min(n_legs, max_dsm*4)]
    masks = [[False]*n_legs]
    for k in range(1, max_dsm+1):
        for combo in itertools.combinations(pool, k):
            m=[False]*n_legs
            for i in combo: m[i]=True
            masks.append(m)
    return masks

def _build_initial_guess(pure_ga_nodes, mask):
    clean = [n for n in pure_ga_nodes if n["type"] in ("start","GA","end")]
    t_nodes = np.array([float(n["time"]) for n in clean])
    r_nodes = np.array([np.asarray(n["r"],float) for n in clean])
    n_legs = len(t_nodes)-1
    x = [float(t_nodes[0])]
    for i in range(n_legs):
        if mask[i]:
            ti,tj = t_nodes[i], t_nodes[i+1]
            ri,rj = r_nodes[i], r_nodes[i+1]
            t_mid = ti + 0.5*(tj-ti)
            x.append(float(t_mid))
            tof_full = (tj-ti)*DAY
            v_full,_ = _lambert_solve(ri, rj, tof_full)
            if v_full is not None:
                r_mid,_ = tools.propagate_two_body(ri, v_full,
                                                   tof_full*0.5,
                                                   problem["mu_sun"])
                x.extend(r_mid.tolist())
            else:
                x.extend(((ri+rj)*0.5).tolist())
    for gt in t_nodes[1:-1]:
        x.append(float(gt))
    x.append(float(t_nodes[-1]))
    return np.array(x,dtype=float)

def _eval_dsm_ga_vector(x, ga_pids, mask, start_win, end_win):
    idx = 0
    t_start = float(np.clip(x[idx], *start_win)); idx+=1
    n_legs = len(ga_pids)+1
    dsm_data = []
    for i in range(n_legs):
        if mask[i]:
            t_raw = float(x[idx]); idx+=1
            r_vec = np.array([x[idx], x[idx+1], x[idx+2]], dtype=float)
            idx+=3
            dsm_data.append((t_raw, r_vec))
        else:
            dsm_data.append(None)
    ga_times = []
    for _ in ga_pids:
        ga_times.append(float(x[idx])); idx+=1
    t_end = float(np.clip(x[idx], *end_win))
    cur_t = [t_start] + ga_times + [t_end]
    for k in range(1, len(cur_t)):
        if cur_t[k] - cur_t[k-1] < MIN_TOF_DAYS:
            return float("inf"), None
    pid_seq = [str(problem["start"].get("planet_id","0"))] + ga_pids + [str(problem["end"].get("planet_id","0"))]
    r_nodes_full = [_planet_state(pid, t)[0] for pid,t in zip(pid_seq, cur_t)]
    total = 0.0
    leg_info = []
    for i in range(n_legs):
        ti,tj = cur_t[i], cur_t[i+1]
        ri,rj = r_nodes_full[i], r_nodes_full[i+1]
        dsm = dsm_data[i]
        if dsm is not None:
            t_raw,r_dsm = dsm
            t_dsm = float(np.clip(t_raw, ti+MIN_TOF_DAYS, tj-MIN_TOF_DAYS))
            tof_a = max(1e-8, (t_dsm-ti)*DAY)
            v_dep_i, v_arr_dsm = _lambert_solve(ri, r_dsm, tof_a)
            if v_dep_i is None:
                return float("inf"), None
            tof_b = max(1e-8, (tj-t_dsm)*DAY)
            v_dep_dsm, v_arr_j = _lambert_solve(r_dsm, rj, tof_b)
            if v_dep_dsm is None:
                return float("inf"), None
            dv_dsm = np.linalg.norm(np.asarray(v_dep_dsm)-np.asarray(v_arr_dsm))
            total += dv_dsm
            leg_info.append({"v_dep_i":np.asarray(v_dep_i),
                             "v_arr_dsm":np.asarray(v_arr_dsm),
                             "v_dep_dsm":np.asarray(v_dep_dsm),
                             "v_arr_j":np.asarray(v_arr_j),
                             "t_dsm":t_dsm,
                             "r_dsm":r_dsm})
        else:
            tof = max(1e-8, (tj-ti)*DAY)
            v_dep_i, v_arr_j = _lambert_solve(ri, rj, tof)
            if v_dep_i is None:
                return float("inf"), None
            leg_info.append({"v_dep_i":np.asarray(v_dep_i),
                             "v_arr_j":np.asarray(v_arr_j),
                             "t_dsm":None,
                             "r_dsm":None})
    # start node
    _, v_ref_start = _planet_state(str(problem["start"].get("planet_id","0")), cur_t[0])
    start_node = {"type":"start","time":float(cur_t[0]),
                  "planet_id":str(problem["start"].get("planet_id","0")),
                  "r":np.asarray(r_nodes_full[0]),
                  "v_before":v_ref_start,
                  "v_after":leg_info[0]["v_dep_i"]}
    total += _boundary_dv(start_node, problem["start"])
    full_nodes = [start_node]
    # intermingle DSM and GA nodes
    for i in range(n_legs):
        if dsm_data[i] is not None:
            leg = leg_info[i]
            full_nodes.append({"type":"DSM","time":float(leg["t_dsm"]),
                               "planet_id":"0","r":leg["r_dsm"],
                               "v_before":leg["v_arr_dsm"],
                               "v_after":leg["v_dep_dsm"]})
        if i < len(ga_pids):
            inbound = leg_info[i]["v_arr_j"]
            outbound = leg_info[i+1]["v_dep_i"]
            t_ga = cur_t[i+1]
            ga_pid = ga_pids[i]
            dv_ga, feas = _powered_flyby_dv(inbound, outbound,
                                            ga_pid, t_ga, problem)
            if not feas:
                return float("inf"), None
            total += dv_ga
            full_nodes.append({"type":"GA","time":float(t_ga),
                               "planet_id":ga_pid,
                               "r":np.asarray(r_nodes_full[i+1]),
                               "v_before":inbound,
                               "v_after":outbound})
    # end node
    _, v_ref_end = _planet_state(str(problem["end"].get("planet_id","0")), cur_t[-1])
    end_node = {"type":"end","time":float(cur_t[-1]),
                "planet_id":str(problem["end"].get("planet_id","0")),
                "r":np.asarray(r_nodes_full[-1]),
                "v_before":leg_info[-1]["v_arr_j"],
                "v_after":v_ref_end}
    total += _boundary_dv(end_node, problem["end"])
    full_nodes.append(end_node)
    return total, full_nodes

def _insert_dsms_and_optimize(pure_ga_nodes, prob):
    if pure_ga_nodes is None:
        return None
    max_nodes = int(prob.get("max_nodes",10))
    max_DSM_allowed = int(prob.get("max_DSM",0))
    clean = [n for n in pure_ga_nodes if n["type"] in ("start","GA","end")]
    n_ga = sum(1 for n in clean if n["type"]=="GA")
    ga_pids = [str(n["planet_id"]) for n in clean if n["type"]=="GA"]
    n_legs = n_ga + 1
    max_dsm_by_nodes = max_nodes - (2 + n_ga)
    max_dsm = min(max_DSM_allowed, max_dsm_by_nodes)
    if max_dsm <= 0:
        return None
    leg_weights = _leg_weights_from_nodes(pure_ga_nodes)
    all_masks = _generate_dsm_masks(n_legs, max_dsm, leg_weights)
    start_win = _time_window(prob["start"]["time"])
    end_win   = _time_window(prob["end"]["time"])
    mask_scores = []
    for mask in all_masks:
        x0 = _build_initial_guess(pure_ga_nodes, mask)
        dv0, nodes0 = _eval_dsm_ga_vector(x0, ga_pids, mask, start_win, end_win)
        if nodes0 is None or not np.isfinite(dv0):
            continue
        mask_scores.append((dv0, mask, x0))
    if not mask_scores:
        return None
    mask_scores.sort(key=lambda x: x[0])
    best_nodes = None
    best_dv = float("inf")
    deadline = time.time() + 0.55 * _TIMEOUT
    target = prob.get("target_dv")
    early_thr = target*0.85 if target is not None else None
    max_masks = min(len(mask_scores), 25000)
    if _time_remaining() < 30:
        max_masks = min(max_masks, 4000)
    for dv0, mask, x0 in mask_scores[:max_masks]:
        if time.time()>deadline: break
        for attempt in range(60):
            if time.time()>deadline: break
            x = x0.copy()
            if attempt>0:
                jitter = 4.0
                x[0] += _rng.uniform(-jitter, jitter)
                idx = 1
                for has in mask:
                    if has:
                        x[idx] += _rng.uniform(-jitter, jitter)
                        idx+=1
                        x[idx:idx+3] += _rng.normal(scale=0.02*2e8, size=3)
                        idx+=3
                ga_off = 1 + sum(4 if h else 0 for h in mask)
                for i in range(len(ga_pids)):
                    x[ga_off+i] += _rng.uniform(-6.0,6.0)
                x[-1] += _rng.uniform(-jitter, jitter)
            # NM simplex
            scales = [5.0]
            for has in mask:
                if has:
                    scales.extend([5.0,0.02*2e8,0.02*2e8,0.02*2e8])
            for _ in ga_pids:
                scales.append(7.0)
            scales.append(5.0)
            n_vars = len(x)
            simplex = [x]
            for j in range(n_vars):
                dx = np.zeros(n_vars)
                dx[j] = scales[j] if j < len(scales) else 1.0
                simplex.append(x+dx)
            try:
                res = minimize(lambda zz: _eval_dsm_ga_vector(
                                   zz, ga_pids, mask,
                                   start_win, end_win)[0],
                               x, method="Nelder-Mead",
                               options={"maxiter":2500*n_vars,
                                        "xatol":1e-4,"fatol":1e-5,
                                        "initial_simplex":np.array(simplex[:n_vars+1])})
            except Exception:
                continue
            dv_opt, nodes_opt = _eval_dsm_ga_vector(res.x, ga_pids, mask,
                                                    start_win, end_win)
            if nodes_opt is None: continue
            if dv_opt < best_dv-1e-8:
                best_dv, best_nodes = dv_opt, nodes_opt
                if early_thr is not None and best_dv <= early_thr:
                    deadline = time.time() + 0.05*_TIMEOUT
    if best_nodes is not None:
        pure_dv = _total_dv(pure_ga_nodes)
        if pure_dv < best_dv-1e-8:
            best_nodes, best_dv = pure_ga_nodes, pure_dv
    return best_nodes

# ----------------------------------------------------------------------
# Encode / decode full trajectory vectors
# ----------------------------------------------------------------------
def _extract_vector_and_mask(nodes):
    pure = [n for n in nodes if n["type"] in ("start","GA","end")]
    pure_times = [float(n["time"]) for n in pure]
    ga_pids = [str(n["planet_id"]) for n in pure if n["type"]=="GA"]
    n_legs = len(pure_times)-1
    mask = []
    dsm_nodes = [n for n in nodes if n["type"]=="DSM"]
    for i in range(n_legs):
        ts, te = pure_times[i], pure_times[i+1]
        mask.append(any(ts < float(d["time"]) < te for d in dsm_nodes))
    x = [pure_times[0]]
    for i in range(n_legs):
        if mask[i]:
            for nd in dsm_nodes:
                if pure_times[i] < float(nd["time"]) < pure_times[i+1]:
                    x.append(float(nd["time"]))
                    x.extend(np.asarray(nd["r"],float).tolist())
                    break
    for t in pure_times[1:-1]:
        x.append(float(t))
    x.append(pure_times[-1])
    return np.array(x,float), mask, ga_pids

def _refine_full_trajectory(nodes):
    if nodes is None: return None
    x0, mask, ga_pids = _extract_vector_and_mask(nodes)
    if x0.size==0: return None
    start_win = _time_window(problem["start"]["time"])
    end_win   = _time_window(problem["end"]["time"])
    def _obj(x):
        total,_ = _eval_dsm_ga_vector(x, ga_pids, mask,
                                      start_win, end_win)
        return total
    scales = [5.0]
    for has in mask:
        if has: scales.extend([5.0,0.02*2e8,0.02*2e8,0.02*2e8])
    for _ in ga_pids:
        scales.append(7.0)
    scales.append(5.0)
    n_vars = len(x0)
    simplex = [x0]
    for i in range(n_vars):
        dx = np.zeros(n_vars)
        dx[i] = scales[i] if i < len(scales) else 1.0
        simplex.append(x0+dx)
    try:
        res = minimize(_obj, x0, method="Nelder-Mead",
                       options={"maxiter":3000*n_vars,
                                "xatol":1e-4,"fatol":1e-5,
                                "initial_simplex":np.array(simplex[:n_vars+1])})
    except Exception:
        return None
    if not res.success:
        return None
    total_opt, nodes_opt = _eval_dsm_ga_vector(res.x, ga_pids, mask,
                                             start_win, end_win)
    if nodes_opt is None:
        return None
    if total_opt < _total_dv(nodes) - 1e-8:
        return nodes_opt
    return None

def _local_random_search(nodes, max_iters=1800):
    if nodes is None: return None
    x0, mask, ga_pids = _extract_vector_and_mask(nodes)
    if x0.size==0:
        return nodes
    start_win = _time_window(problem["start"]["time"])
    end_win   = _time_window(problem["end"]["time"])
    best_dv = _total_dv(nodes)
    best_x = x0.copy()
    best_nodes = nodes
    for _ in range(max_iters):
        if _time_remaining()<0.1: break
        x = best_x.copy()
        jitter = 2.0
        x[0] += _rng.uniform(-jitter, jitter)
        idx = 1
        for has in mask:
            if has:
                x[idx] += _rng.uniform(-jitter, jitter)
                idx += 1
                x[idx:idx+3] += _rng.normal(scale=1e6,size=3)
                idx += 3
        for i in range(len(ga_pids)):
            x[idx] += _rng.uniform(-2.0,2.0)
            idx+=1
        x[-1] += _rng.uniform(-jitter, jitter)
        dv, nd = _eval_dsm_ga_vector(x, ga_pids, mask,
                                      start_win, end_win)
        if nd is None or not np.isfinite(dv):
            continue
        if dv < best_dv - 1e-8:
            best_dv, best_x, best_nodes = dv, x.copy(), nd
    return best_nodes

def _basinhopping_refine(nodes):
    if nodes is None: return None
    x0, mask, ga_pids = _extract_vector_and_mask(nodes)
    if x0.size==0: return nodes
    start_win = _time_window(problem["start"]["time"])
    end_win   = _time_window(problem["end"]["time"])
    def _obj(x):
        tot,_ = _eval_dsm_ga_vector(x, ga_pids, mask,
                                    start_win, end_win)
        return tot
    bh = basinhopping(_obj, x0, niter=12, stepsize=0.8,
                      minimizer_kwargs={"method":"Nelder-Mead",
                                        "options":{"maxiter":1500,
                                                  "xatol":1e-4,
                                                  "fatol":1e-5}})
    if bh is None or not hasattr(bh,"x"):
        return nodes
    tot_bh, nodes_bh = _eval_dsm_ga_vector(bh.x, ga_pids, mask,
                                          start_win, end_win)
    if nodes_bh is not None and tot_bh < _total_dv(nodes)-1e-8:
        return nodes_bh
    return nodes

def _final_de_optimize(nodes):
    if nodes is None: return None
    x0, mask, ga_pids = _extract_vector_and_mask(nodes)
    if x0.size==0: return nodes
    start_win = _time_window(problem["start"]["time"])
    end_win   = _time_window(problem["end"]["time"])
    bounds = []
    bounds.append(start_win)                     # start time
    for has in mask:
        if has:
            bounds.append((start_win[0], end_win[1]))   # DSM time
            bounds.extend([(-1e9,1e9),(-1e9,1e9),(-1e9,1e9)])  # DSM pos
    for _ in ga_pids:
        bounds.append((start_win[0], end_win[1]))      # GA times
    bounds.append(end_win)                       # end time
    def _obj(x):
        total,_ = _eval_dsm_ga_vector(x, ga_pids, mask,
                                      start_win, end_win)
        return total
    tleft = _time_remaining()
    if tleft>60:
        maxiter,popsize = 600,100
    elif tleft>30:
        maxiter,popsize = 400,80
    elif tleft>15:
        maxiter,popsize = 250,60
    else:
        maxiter,popsize = 150,40
    try:
        res = differential_evolution(_obj, bounds,
                                     maxiter=maxiter, popsize=popsize,
                                     seed=RNG_SEED, polish=True,
                                     disp=False, workers=1)
    except Exception:
        return nodes
    if not res.success:
        return nodes
    total_opt, nodes_opt = _eval_dsm_ga_vector(res.x, ga_pids, mask,
                                             start_win, end_win)
    if nodes_opt is None:
        return nodes
    if total_opt < _total_dv(nodes)-1e-8:
        return nodes_opt
    return nodes

def _coordinate_descent(nodes):
    if nodes is None: return None
    x0, mask, ga_pids = _extract_vector_and_mask(nodes)
    if x0.size==0: return nodes
    start_win = _time_window(problem["start"]["time"])
    end_win   = _time_window(problem["end"]["time"])
    best_x = x0.copy()
    best_dv = _total_dv(nodes)
    deltas = np.array([0.1,-0.1,0.5,-0.5,1.0,-1.0])
    while _time_remaining()>0.3:
        improved = False
        for i in range(len(best_x)):
            lo,hi = (start_win if i==0 else (end_win if i==len(best_x)-1 else (-np.inf,np.inf)))
            for d in deltas:
                x = best_x.copy()
                x[i] = float(np.clip(x[i]+d, lo, hi))
                total, nd = _eval_dsm_ga_vector(x, ga_pids, mask,
                                               start_win, end_win)
                if nd is None: continue
                if total < best_dv - 1e-9:
                    best_dv, best_x, improved = total, x.copy(), True
                    break
            if improved: break
        if not improved: break
    total, nd = _eval_dsm_ga_vector(best_x, ga_pids, mask,
                                   start_win, end_win)
    if nd is not None and total < _total_dv(nodes)-1e-8:
        return nd
    return None

# ----------------------------------------------------------------------
# Random GA‑sequence sampler (diversification)
# ----------------------------------------------------------------------
def _random_ga_search(elite, max_iter=5000):
    max_nodes = int(problem.get("max_nodes",10))
    max_ga_allowed = int(problem.get("max_GA", len(problem.get("allowed_GA_planets",[]))))
    allowed = [str(p) for p in problem.get("allowed_GA_planets", [])]
    lo_s, hi_s = _time_window(problem["start"]["time"])
    lo_e, hi_e = _time_window(problem["end"]["time"])
    def _push(dv, nodes, seq):
        if not np.isfinite(dv): return
        if len(elite) < 120000:
            elite.append((dv,nodes,seq))
            elite.sort(key=lambda x:x[0])
        elif dv < elite[-1][0]:
            elite[-1] = (dv,nodes,seq)
            elite.sort(key=lambda x:x[0])
    attempts = 0
    target = problem.get("target_dv")
    early_thr = target*0.85 if target is not None else None
    while attempts < max_iter and _time_remaining()>1.2:
        max_len = min(max_ga_allowed, max_nodes-2)
        n = _rng.integers(0, max_len+1)
        seq = () if n==0 else tuple(_rng.choice(allowed,size=n,replace=True))
        times = _random_times(lo_s, hi_s, lo_e, hi_e, len(seq))
        if times is None:
            attempts+=1
            continue
        nodes,total = _evaluate_ga_sequence(seq, times)
        if nodes is None:
            attempts+=1
            continue
        refined = _refine_ga_times(nodes, problem)
        if refined is not None:
            nodes = refined
            total = _total_dv(nodes)
        _push(total, nodes, seq)
        if early_thr is not None and total <= early_thr:
            break
        attempts+=1

# ----------------------------------------------------------------------
# MAIN driver
# ----------------------------------------------------------------------
def run_code():
    try:
        record.event(f"mission={problem.get('id','unknown')} search_start")
    except Exception:
        pass

    lo_s, hi_s = _time_window(problem["start"]["time"])
    lo_e, hi_e = _time_window(problem["end"]["time"])

    # --------------------------------------------------
    # 1 – Process known templates first
    # --------------------------------------------------
    known = _lookup_known()
    elite = []                     # (dv, nodes, seq)
    K_KEEP = 120000               # maximum elite size
    target = problem.get("target_dv")
    early_thr = target*0.85 if target is not None else None

    def _push_elite(dv, nodes, seq):
        if not np.isfinite(dv): return
        if len(elite) < K_KEEP:
            elite.append((dv,nodes,seq))
            elite.sort(key=lambda x:x[0])
        elif dv < elite[-1][0]:
            elite[-1] = (dv,nodes,seq)
            elite.sort(key=lambda x:x[0])

    if known is not None:
        for seq, times in known.get("candidates", []):
            if _time_remaining()<1.0: break
            if times is None:
                times_seq = _linear_spaced_times(lo_s, hi_s, lo_e, hi_e, len(seq))
            else:
                times_seq = times
            nodes,total = _evaluate_ga_sequence(seq, times_seq)
            if nodes is None: continue
            refined = _refine_ga_times(nodes, problem)
            if refined is not None:
                nodes = refined
                total = _total_dv(nodes)
            _push_elite(total, nodes, seq)

    # --------------------------------------------------
    # 2 – Exhaustive enumeration (quick linear times)
    # --------------------------------------------------
    all_seq = _generate_all_sequences()
    for seq in all_seq:
        if _time_remaining()<1.0: break
        lin_times = _linear_spaced_times(lo_s, hi_s, lo_e, hi_e, len(seq))
        nodes,total = _evaluate_ga_sequence(seq, lin_times)
        if nodes is None: continue
        refined = _refine_ga_times(nodes, problem)
        if refined is not None:
            nodes = refined
            total = _total_dv(nodes)
        _push_elite(total, nodes, seq)

    # --------------------------------------------------
    # 3 – Random timings for promising sequences
    # --------------------------------------------------
    if _time_remaining()>5.0:
        # take top N sequences for random probing
        top_seqs = [seq for _,_,seq in elite[:min(200, len(elite))]]
        for seq in top_seqs:
            if _time_remaining()<1.0: break
            for _ in range(2):
                rand_times = _random_times(lo_s, hi_s, lo_e, hi_e, len(seq))
                if rand_times is None: continue
                nodes,total = _evaluate_ga_sequence(seq, rand_times)
                if nodes is None: continue
                refined = _refine_ga_times(nodes, problem)
                if refined is not None:
                    nodes = refined
                    total = _total_dv(nodes)
                _push_elite(total, nodes, seq)

    # --------------------------------------------------
    # 4 – Global DE refinement on top elite candidates
    # --------------------------------------------------
    if elite and _time_remaining()>5.0:
        top_elite = elite[:min(3000, len(elite))]
        for dv, nodes, seq in top_elite:
            if _time_remaining()<2.0: break
            times0 = [float(n["time"]) for n in nodes]
            nd, ndv = _de_optimize_seq(seq, times0,
                                       (lo_s, hi_s), (lo_e, hi_e))
            if nd is not None and ndv is not None and ndv < dv:
                _push_elite(ndv, nd, seq)

    # --------------------------------------------------
    # 5 – Pick best pure‑GA solution
    # --------------------------------------------------
    elite.sort(key=lambda x:x[0])
    best_pure = elite[0][1] if elite else None

    # --------------------------------------------------
    # 6 – Insert DSMs (if budget permits)
    # --------------------------------------------------
    final = best_pure
    if final is not None and _time_remaining()>8.0:
        dsm_opt = _insert_dsms_and_optimize(final, problem)
        if dsm_opt is not None and _total_dv(dsm_opt) < _total_dv(final)-1e-8:
            final = dsm_opt

    # --------------------------------------------------
    # 7 – Full‑trajectory Nelder‑Mead polish
    # --------------------------------------------------
    if final is not None and _time_remaining()>2.0:
        refined = _refine_full_trajectory(final)
        if refined is not None:
            final = refined
        while _time_remaining()>1.5:
            nxt = _refine_full_trajectory(final)
            if nxt is not None and _total_dv(nxt) < _total_dv(final)-1e-8:
                final = nxt
            else:
                break

    # --------------------------------------------------
    # 8 – Stochastic local hill‑climb
    # --------------------------------------------------
    if final is not None and _time_remaining()>1.0:
        final = _local_random_search(final, max_iters=1800)

    # --------------------------------------------------
    # 9 – Light basinhopping polish
    # --------------------------------------------------
    if final is not None and _time_remaining()>1.0:
        final = _basinhopping_refine(final)

    # --------------------------------------------------
    # 10 – Final DE polish on the whole vector
    # --------------------------------------------------
    if final is not None and _time_remaining()>5.0:
        final = _final_de_optimize(final)

    # --------------------------------------------------
    # 11 – Tiny coordinate‑descent final sweep
    # --------------------------------------------------
    if final is not None and _time_remaining()>0.5:
        cd = _coordinate_descent(final)
        if cd is not None:
            final = cd

    # --------------------------------------------------
    # 12 – Fallback safety net
    # --------------------------------------------------
    if final is None:
        final = best_pure if best_pure is not None else []

    try:
        record.set("final_nodes", len(final))
    except Exception:
        pass

    return _format_nodes(final)
# EVOLVE-BLOCK-END
