"""
Pre-defined bindings (DO NOT REMOVE -- they are outside the evolvable block):
    problem = load_problem_for_candidate(__file__)
    tools   = Tools()
    record  = Record()    # evaluator injects this lightweight timestamped logger
"""
import sys
from pathlib import Path

_FAMILY_DIR = Path(__file__).resolve().parent.parent
if str(_FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(_FAMILY_DIR))

from problem_config import load_problem_for_candidate
from tools_wrapper import Tools

problem = load_problem_for_candidate(__file__)
tools = Tools()

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

DAY = 86400.0
MIN_TOF = 10.0  # days
_AU = 1.495978707e8
RNG_SEED = 20260625


def _piecewise_linear(x: float, breakpoints) -> float:
    bp = sorted(breakpoints, key=lambda p: float(p[0]))
    if x <= float(bp[0][0]):
        return float(bp[0][1])
    if x >= float(bp[-1][0]):
        return float(bp[-1][1])
    for i in range(len(bp) - 1):
        x0, y0 = float(bp[i][0]), float(bp[i][1])
        x1, y1 = float(bp[i + 1][0]), float(bp[i + 1][1])
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0
    return float(bp[-1][1])


def _periapsis_dv(vinf: float, mu_p: float, R_p: float,
                  h_factor: float, T_days: float) -> float:
    r_peri = R_p * (1.0 + h_factor)
    T_secs = T_days * DAY
    two_mu_over_r = 2.0 * mu_p / r_peri
    term = (4.0 * np.pi**2 * mu_p**2 / T_secs**2) ** (1.0 / 3.0)
    return float(np.sqrt(vinf * vinf + two_mu_over_r)
                 - np.sqrt(max(two_mu_over_r - term, 0.0)))


def _r_periapsis(r, v, mu):
    r_mag = float(np.linalg.norm(np.asarray(r)))
    v_mag = float(np.linalg.norm(np.asarray(v)))
    eps = 0.5 * v_mag * v_mag - mu / r_mag
    h_vec = np.cross(np.asarray(r), np.asarray(v))
    h_sq = float(np.dot(h_vec, h_vec))
    if h_sq < 1e-30:
        return r_mag
    e_sq = 1.0 + 2.0 * eps * h_sq / (mu * mu)
    if e_sq < 0:
        return r_mag
    return float(h_sq / mu / (1.0 + np.sqrt(e_sq)))


def _boundary_dv(node: dict, boundary_spec: dict) -> float:
    btype = boundary_spec["type"]
    if btype == "piecewise_linear":
        dv_mag = float(np.linalg.norm(
            np.asarray(node["v_after"], dtype=float)
            - np.asarray(node["v_before"], dtype=float)))
        return _piecewise_linear(dv_mag, boundary_spec["breakpoints"])
    if btype == "periapsis_maneuver":
        pid = str(boundary_spec["planet_id"])
        mu_p = float(problem["planet_mu"][pid])
        R_p = float(problem["planet_radius"][pid])
        hf = float(boundary_spec["h_factor"])
        Td = float(boundary_spec["T_days"])
        _, v_pl = tools.ephem(pid, float(node["time"]))
        if node["type"] == "start":
            vinf = float(np.linalg.norm(
                np.asarray(node["v_after"], dtype=float) - v_pl))
        else:
            vinf = float(np.linalg.norm(
                np.asarray(node["v_before"], dtype=float) - v_pl))
        return _periapsis_dv(vinf, mu_p, R_p, hf, Td)
    return 1.0e9


def _time_window(spec):
    if spec["kind"] == "window":
        return float(spec["lo"]), float(spec["hi"])
    v = float(spec["value"])
    return v, v


def _planet_state(pid, t, problem):
    if pid == "0" or int(pid) == 0:
        state_spec = problem["start"]
        if "state_r" not in state_spec or "state_v" not in state_spec:
            state_spec = problem["end"]
        r = np.array(state_spec["state_r"], dtype=float)
        v = np.array(state_spec["state_v"], dtype=float)
        return r, v
    r, v = tools.ephem(str(pid), float(t))
    return np.asarray(r), np.asarray(v)


def _powered_flyby_dv(v_arr, v_dep, ga_pid, t, problem):
    ga_str = str(ga_pid)
    mu_p = float(problem["planet_mu"][ga_str])
    R_p = float(problem["planet_radius"][ga_str])
    flyby = problem.get("flyby", {}).get("min_altitude_km", {})
    min_alt = float(flyby.get(ga_str, 200))
    _, v_pl = tools.ephem(ga_str, float(t))
    try:
        _, dv_ga, feas = tools.powered_flyby(
            np.asarray(v_arr), np.asarray(v_dep), v_pl,
            mu_p, R_p + min_alt)
        return float(dv_ga), bool(feas)
    except Exception:
        return float("inf"), False


# ---------------------------------------------------------------------------
# Phase 1: Grid search GA trajectory
# ---------------------------------------------------------------------------

def _grid_search_ga(problem):
    start_spec = problem["start"]
    end_spec = problem["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))
    mu_sun = problem["mu_sun"]
    rng = np.random.default_rng(RNG_SEED)

    t0_lo, t0_hi = _time_window(start_spec["time"])
    tf_lo, tf_hi = _time_window(end_spec["time"])

    allowed_ga = [str(p) for p in problem.get("allowed_GA_planets", [])]
    max_ga = min(int(problem.get("max_GA", 0)), len(allowed_ga))

    best_dv = float("inf")
    best_nodes = None

    for n_ga in range(max_ga + 1):
        ga_combos = [()] if n_ga == 0 else list(itertools.permutations(allowed_ga, n_ga))
        n_grid = 5 if n_ga == 0 else 3
        n_legs = n_ga + 1

        for ga_pids in ga_combos:
            t0_pts = np.linspace(t0_lo, t0_hi, n_grid)
            tf_pts = np.linspace(max(tf_lo, t0_hi + MIN_TOF * n_legs), tf_hi, n_grid)

            for t0 in t0_pts:
                for tf in tf_pts:
                    span = max(0.1, tf - t0 - MIN_TOF * n_legs)
                    for _ in range(6):
                        times = [float(t0)]
                        fracs = sorted(rng.uniform(0.0, 1.0, n_ga))
                        for k in range(n_ga):
                            times.append(times[-1] + MIN_TOF + fracs[k] * span / n_legs)
                        times.append(float(tf))
                        if any(times[j + 1] <= times[j] for j in range(len(times) - 1)):
                            continue

                        # Evaluate trajectory
                        total = 0.0
                        nodes = []
                        # Get positions
                        wp_r = []
                        wp_v = []
                        for j in range(n_legs + 1):
                            pid = start_pid if j == 0 else (
                                str(ga_pids[j - 1]) if j <= n_ga else end_pid)
                            r, v = _planet_state(pid, times[j], problem)
                            wp_r.append(r)
                            wp_v.append(v)

                        # Lambert legs
                        legs_vd = []
                        legs_va = []
                        for j in range(n_legs):
                            tof = times[j + 1] - times[j]
                            try:
                                vd, va = tools.lambert(
                                    wp_r[j], wp_r[j + 1], tof * DAY, mu_sun, prograde=True)
                            except Exception:
                                total = float("inf")
                                break
                            legs_vd.append(np.asarray(vd))
                            legs_va.append(np.asarray(va))
                        if total >= 1e8:
                            continue

                        # Start node
                        start_node = {"type": "start", "time": float(times[0]),
                                      "planet_id": start_pid,
                                      "r": wp_r[0], "v_before": wp_v[0],
                                      "v_after": legs_vd[0]}
                        total += _boundary_dv(start_node, start_spec)
                        nodes.append(start_node)

                        # GA nodes
                        for j in range(n_ga):
                            ga_dv, feas = _powered_flyby_dv(
                                legs_va[j], legs_vd[j + 1], ga_pids[j], times[j + 1], problem)
                            if not feas:
                                total = float("inf")
                                break
                            total += ga_dv
                            nodes.append({"type": "GA", "time": float(times[j + 1]),
                                          "planet_id": str(ga_pids[j]),
                                          "r": wp_r[j + 1],
                                          "v_before": legs_va[j],
                                          "v_after": legs_vd[j + 1]})
                        if total >= 1e8:
                            continue

                        # Periapsis penalties
                        for j in range(n_legs):
                            r_p = _r_periapsis(wp_r[j], legs_vd[j], mu_sun)
                            if r_p < 0.1 * _AU:
                                total += 1e6 * (0.1 * _AU / r_p - 1.0)
                            if j == n_legs - 1 and end_pid != "0":
                                mu_tgt = problem["planet_mu"].get(end_pid, 0)
                                if mu_tgt > 0:
                                    r_tgt_min = 1.1 * problem["planet_radius"].get(end_pid, 0)
                                    r_rel = wp_r[j + 1] - wp_r[j]
                                    v_rel = legs_va[j] - wp_v[j + 1]
                                    r_p_t = _r_periapsis(r_rel, v_rel, mu_tgt)
                                    if r_p_t < r_tgt_min:
                                        total += 1e6 * (r_tgt_min / r_p_t - 1.0)

                        # End node
                        end_node = {"type": "end", "time": float(times[-1]),
                                    "planet_id": end_pid,
                                    "r": wp_r[-1],
                                    "v_before": legs_va[-1],
                                    "v_after": wp_v[-1]}
                        total += _boundary_dv(end_node, end_spec)
                        nodes.append(end_node)

                        if total < best_dv:
                            best_dv = total
                            best_nodes = nodes

    # Fallback: direct Lambert
    if best_nodes is None:
        t0_mid = (t0_lo + t0_hi) / 2.0
        tf_mid = (tf_lo + tf_hi) / 2.0
        times = [t0_mid, tf_mid]
        wp_r = [_planet_state(start_pid, t0_mid, problem)[0],
                _planet_state(end_pid, tf_mid, problem)[0]]
        wp_v = [_planet_state(start_pid, t0_mid, problem)[1],
                _planet_state(end_pid, tf_mid, problem)[1]]
        try:
            vd, va = tools.lambert(wp_r[0], wp_r[1], (tf_mid - t0_mid) * DAY,
                                   problem["mu_sun"], prograde=True)
            best_nodes = [
                {"type": "start", "time": float(t0_mid), "planet_id": start_pid,
                 "r": wp_r[0], "v_before": wp_v[0], "v_after": np.asarray(vd)},
                {"type": "end", "time": float(tf_mid), "planet_id": end_pid,
                 "r": wp_r[1], "v_before": np.asarray(va), "v_after": wp_v[1]},
            ]
        except Exception:
            pass

    return best_nodes


# ---------------------------------------------------------------------------
# Phase 2 & 3: Insert DSMs + local optimization
# ---------------------------------------------------------------------------

def _insert_dsms_and_optimize(nodes, problem):
    """Insert DSMs at Lambert midpoints, then Nelder-Mead local optimization.

    x0 = original trajectory with DSMs at exact time/position midpoints
    (dv_dsm = 0, identical to Phase 1).
    Variables: (t_dsm_d, r_dsm_km) per leg + t_ga_d per GA (physical units).
    """
    if nodes is None:
        return None

    mu_sun = problem["mu_sun"]
    start_spec = problem["start"]
    end_spec = problem["end"]
    start_pid = str(start_spec.get("planet_id", "0"))
    end_pid = str(end_spec.get("planet_id", "0"))

    from TrajectoryToolKit.OrbDyn.prop import farnocchia_rv

    nodes_clean = [n for n in nodes if n["type"] in ("start", "GA", "end")]
    n_ga = sum(1 for n in nodes_clean if n["type"] == "GA")
    n_legs = n_ga + 1
    ga_pids = [str(n["planet_id"]) for n in nodes_clean if n["type"] == "GA"]

    t_nodes = np.array([float(n["time"]) for n in nodes_clean])
    r_nodes = np.array([np.asarray(n["r"]) for n in nodes_clean])

    # ---- Build x0: DSMs at Lambert midpoints ----
    x0_list = []
    for i in range(n_legs):
        t_i, r_i = t_nodes[i], r_nodes[i]
        t_j, r_j = t_nodes[i + 1], r_nodes[i + 1]
        tof_days = t_j - t_i

        # t_dsm at midpoint
        x0_list.append(t_i + tof_days / 2.0)

        # r_dsm at Lambert-propagated midpoint
        try:
            vd, _ = tools.lambert(r_i, r_j, tof_days * DAY, mu_sun, prograde=True)
            r_mid, _ = farnocchia_rv(float(mu_sun), np.asarray(r_i),
                                     np.asarray(vd), tof_days * DAY / 2.0)
        except Exception:
            r_mid = (np.asarray(r_i) + np.asarray(r_j)) / 2.0
        x0_list.extend([r_mid[0], r_mid[1], r_mid[2]])

    # GA times
    for k in range(n_ga):
        x0_list.append(t_nodes[k + 1])

    x0 = np.array(x0_list, dtype=float)
    n_vars = len(x0)
    n_per_leg = 4  # t + xyz

    # ---- Objective function (two-pass) ----
    def _eval(x):
        total = 0.0

        # Update GA times & positions
        cur_t = list(t_nodes)
        cur_r = [np.asarray(r) for r in r_nodes]
        ga_vp = {}
        for k in range(n_ga):
            vi = n_per_leg * n_legs + k
            lo = cur_t[k] + MIN_TOF
            hi = (cur_t[k + 2] if k + 2 < len(cur_t) else cur_t[-1]) - MIN_TOF
            cur_t[k + 1] = np.clip(x[vi], lo, max(lo + 0.1, hi))
            cur_r[k + 1], ga_vp[k] = _planet_state(ga_pids[k], cur_t[k + 1], problem)
        cur_r[0], v_ref0 = _planet_state(start_pid, cur_t[0], problem)

        # ---- Pass 1: compute all leg velocities, store for GA reference ----
        legs = []  # each: {t_i, r_i, t_dsm, r_dsm, t_j, r_j, v_dep_i, v_arr_dsm, v_dep_dsm, v_arr_j, dv_dsm}
        for i in range(n_legs):
            t_i, r_i = cur_t[i], cur_r[i]
            t_j, r_j = cur_t[i + 1], cur_r[i + 1]
            t_dsm = np.clip(x[n_per_leg * i], t_i + MIN_TOF, t_j - MIN_TOF)
            r_dsm = np.array([x[n_per_leg * i + 1],
                              x[n_per_leg * i + 2],
                              x[n_per_leg * i + 3]])

            tof_a = max(0.1, (t_dsm - t_i) * DAY)
            try:
                v_dep_i, v_arr_dsm = tools.lambert(
                    np.asarray(r_i), r_dsm, tof_a, mu_sun, prograde=True)
            except Exception:
                return float("inf"), None

            tof_b = max(0.1, (t_j - t_dsm) * DAY)
            try:
                v_dep_dsm, v_arr_j = tools.lambert(
                    r_dsm, np.asarray(r_j), tof_b, mu_sun, prograde=True)
            except Exception:
                return float("inf"), None

            dv_dsm = float(np.linalg.norm(
                np.asarray(v_dep_dsm) - np.asarray(v_arr_dsm)))
            total += dv_dsm

            legs.append({"t_i": t_i, "r_i": r_i, "t_dsm": t_dsm, "r_dsm": r_dsm,
                         "t_j": t_j, "r_j": r_j,
                         "v_dep_i": np.asarray(v_dep_i),
                         "v_arr_dsm": np.asarray(v_arr_dsm),
                         "v_dep_dsm": np.asarray(v_dep_dsm),
                         "v_arr_j": np.asarray(v_arr_j),
                         "dv_dsm": dv_dsm})

            # Periapsis
            for rp_dep, vp_dep in [(np.asarray(r_i), np.asarray(v_dep_i)),
                                    (r_dsm, np.asarray(v_dep_dsm))]:
                rp = _r_periapsis(rp_dep, vp_dep, mu_sun)
                if rp < 0.1 * _AU:
                    total += 1e6 * (0.1 * _AU / rp - 1.0)

        # ---- Pass 2: build nodes with correct GA v_before/v_after ----
        result_nodes = []

        # Start
        l0 = legs[0]
        sn = {"type": "start", "time": float(l0["t_i"]), "planet_id": start_pid,
              "r": np.asarray(l0["r_i"]),
              "v_before": v_ref0, "v_after": l0["v_dep_i"]}
        total += _boundary_dv(sn, start_spec)
        result_nodes.append(sn)

        for i in range(n_legs):
            ld = legs[i]
            # DSM
            result_nodes.append({"type": "DSM", "time": float(ld["t_dsm"]),
                                 "planet_id": "0", "r": ld["r_dsm"],
                                 "v_before": ld["v_arr_dsm"],
                                 "v_after": ld["v_dep_dsm"]})
            # GA after this leg
            if i < n_ga:
                v_arr_ga = ld["v_arr_j"]           # arrival from this leg
                v_dep_ga = legs[i + 1]["v_dep_i"]  # departure to next leg
                ga_dv, feas = _powered_flyby_dv(
                    v_arr_ga, v_dep_ga, ga_pids[i], cur_t[i + 1], problem)
                if not feas:
                    return float("inf"), None
                total += ga_dv
                result_nodes.append({"type": "GA", "time": float(cur_t[i + 1]),
                                     "planet_id": str(ga_pids[i]),
                                     "r": np.asarray(cur_r[i + 1]),
                                     "v_before": v_arr_ga,
                                     "v_after": v_dep_ga})

        # End
        ld_last = legs[-1]
        _, v_ref_end = _planet_state(end_pid, cur_t[-1], problem)
        en = {"type": "end", "time": float(cur_t[-1]), "planet_id": end_pid,
              "r": np.asarray(cur_r[-1]),
              "v_before": ld_last["v_arr_j"], "v_after": v_ref_end}
        total += _boundary_dv(en, end_spec)
        result_nodes.append(en)

        return total, result_nodes

    # ---- Nelder-Mead ----
    dv0, nodes0 = _eval(x0)
    if dv0 >= 1e8:
        return nodes

    # Scales: ~5 days time, ~0.02 AU position, ~10 days GA time
    scales = []
    for i in range(n_legs):
        scales.extend([5.0, 0.02 * _AU, 0.02 * _AU, 0.02 * _AU])
    for k in range(n_ga):
        scales.append(10.0)

    simplex = [x0]
    for j in range(n_vars):
        dx = np.zeros(n_vars)
        dx[j] = scales[j]
        simplex.append(x0 + dx)

    try:
        res = minimize(lambda xx: _eval(xx)[0], x0,
                       method="Nelder-Mead",
                       options={"maxiter": 150 * n_vars, "xatol": 1e-4, "fatol": 1e-6,
                                "initial_simplex": np.array(simplex[:n_vars + 1])})
    except Exception:
        return _format_nodes(nodes0)

    dv_new, nodes_new = _eval(res.x)
    if nodes_new is not None and dv_new < dv0 - 1e-6:
        return _format_nodes(nodes_new)
    return _format_nodes(nodes0)


def _format_nodes(nodes):
    for n in nodes:
        for key in ("r", "v_before", "v_after"):
            n[key] = np.asarray(n[key], dtype=float).tolist()
    return nodes


def run_code():
    record.event(f"mission={problem.get('id')} search_start")
    ga_traj = _grid_search_ga(problem)
    if ga_traj is None:
        record.event("grid_search_failed")
        start_pid = str(problem["start"].get("planet_id", "3"))
        end_pid = str(problem["end"].get("planet_id", "4"))
        t0_lo, t0_hi = _time_window(problem["start"]["time"])
        tf_lo, tf_hi = _time_window(problem["end"]["time"])
        return [{"type": "start", "time": float(t0_lo), "planet_id": start_pid,
                 "r": np.zeros(3).tolist(), "v_before": np.zeros(3).tolist(),
                 "v_after": np.zeros(3).tolist()},
                {"type": "end", "time": float(tf_lo), "planet_id": end_pid,
                 "r": np.zeros(3).tolist(), "v_before": np.zeros(3).tolist(),
                 "v_after": np.zeros(3).tolist()}]

    record.set("phase1_nodes", len(ga_traj))
    refined = _insert_dsms_and_optimize(ga_traj, problem)
    if refined is not None:
        record.set("final_nodes", len(refined))
        record.event("refinement_succeeded")
        return refined
    record.event("refinement_kept_phase1_solution")
    return _format_nodes(ga_traj)

# EVOLVE-BLOCK-END
