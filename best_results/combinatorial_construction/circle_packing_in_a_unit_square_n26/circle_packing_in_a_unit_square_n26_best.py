# EVOLVE-BLOCK-START

import numpy as np
import time
import cvxpy as cp
from scipy.optimize import linprog, differential_evolution

# ----------------------------------------------------------------------
# Problem constants
# ----------------------------------------------------------------------
_N = 26
_PAIR_I, _PAIR_J = np.triu_indices(_N, k=1)
_M = len(_PAIR_I)

# Pre‑computed constraint matrix for the LP (one row per unordered pair)
_A_matrix = np.zeros((_M, _N), dtype=np.float64)
_A_matrix[np.arange(_M), _PAIR_I] = 1.0
_A_matrix[np.arange(_M), _PAIR_J] = 1.0

# ----------------------------------------------------------------------
# Feasibility checker
# ----------------------------------------------------------------------
def _is_valid(circles, tol=1e-9):
    """Return True iff a (26,3) array satisfies all packing constraints."""
    if not isinstance(circles, np.ndarray) or circles.shape != (_N, 3):
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(r).any():
        return False
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    iu = np.triu_indices(_N, k=1)
    if np.any(d2[iu] < (r[iu[0]] + r[iu[1]]) ** 2 - tol):
        return False
    return True

# ----------------------------------------------------------------------
# Core LP solver: given centre coordinates, compute maximal radii
# ----------------------------------------------------------------------
def _solve_lp_for_positions(x, y):
    """Solve a linear program that maximises sum(r) for fixed centres."""
    # Upper bounds from the four borders
    ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    ub = np.maximum(ub, 0.0)   # guard against tiny negatives

    # Pairwise distances (right‑hand side of r_i + r_j ≤ d_ij)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    b = dists[_PAIR_I, _PAIR_J]

    # Linear objective: maximise sum(r) ⇔ minimise -sum(r)
    c = -np.ones(_N, dtype=np.float64)

    # Variable bounds: 0 ≤ r_i ≤ ub_i
    bounds = [(0.0, float(ub_i)) for ub_i in ub]

    # Solve with HiGHS (fast and reliable)
    res = linprog(c, A_ub=_A_matrix, b_ub=b, bounds=bounds, method='highs')
    if res.success and res.x is not None:
        r = np.maximum(res.x, 0.0)
    else:
        r = np.zeros(_N, dtype=np.float64)
    return r

# ----------------------------------------------------------------------
# Fast approximate radii (lower‑bound, cheap to compute)
# ----------------------------------------------------------------------
def _approx_radii(x, y):
    """Lower‑bound radii: min(border, min_j d_ij/2). Cheap to compute."""
    ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.hypot(dx, dy)
    np.fill_diagonal(d, np.inf)
    half_d = np.min(d, axis=1) / 2.0
    r = np.minimum(ub, half_d)
    return r

# ----------------------------------------------------------------------
# Objective for DE (negative sum of approximate radii)
# ----------------------------------------------------------------------
def _objective_de(pos):
    n = _N
    x = pos[:n]
    y = pos[n:]
    return -float(np.sum(_approx_radii(x, y)))

# ----------------------------------------------------------------------
# Optional slack‑exploitation LP (tiny radius boost)
# ----------------------------------------------------------------------
def _apply_slack_lp(x, y, r):
    """Given a feasible layout, enlarge radii via a small LP."""
    # Border slack
    border = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    border_slack = np.maximum(border - r, 0.0)

    # Pairwise slack
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy)
    pair_slack = np.maximum(dist - (r[:, None] + r[None, :]), 0.0)

    delta = cp.Variable(_N)
    constraints = [
        delta >= 0,
        delta <= border_slack,
    ]
    for i, j in zip(_PAIR_I, _PAIR_J):
        constraints.append(delta[i] + delta[j] <= float(pair_slack[i, j]))
    prob = cp.Problem(cp.Maximize(cp.sum(delta)), constraints)
    try:
        prob.solve(solver=cp.ECOS,
                   max_iters=200_000,
                   abstol=1e-12,
                   reltol=1e-12,
                   feastol=1e-12,
                   warm_start=True)
    except Exception:
        return r

    if delta.value is not None:
        delta_val = np.maximum(np.array(delta.value).flatten(), 0.0)
        delta_val = np.minimum(delta_val, border_slack)
        r_new = r + delta_val
        r_new = np.minimum(r_new, border)
        r_new = np.maximum(r_new, 0.0)
        return r_new
    return r

# ----------------------------------------------------------------------
# Main constructor
# ----------------------------------------------------------------------
def construct_circles():
    """Construct 26 circles inside a unit square maximising the sum of radii."""
    n = _N

    # --------------------------------------------------------------
    # 0) Warm‑start from the global best if available.
    # --------------------------------------------------------------
    try:
        gb = GLOBAL_BEST_CONSTRUCTION
        if isinstance(gb, np.ndarray) and gb.shape == (n, 3) and _is_valid(gb):
            best_circles = gb.astype(np.float64, copy=True)
            best_sum = float(np.sum(best_circles[:, 2]))
        else:
            raise Exception
    except Exception:
        # Random initialisation if warm‑start not usable.
        rng0 = np.random.default_rng()
        x0 = rng0.uniform(size=n)
        y0 = rng0.uniform(size=n)
        r0 = _solve_lp_for_positions(x0, y0)
        best_circles = np.column_stack([x0, y0, r0])
        best_sum = float(np.sum(r0))

    start_time = time.time()
    TOTAL_TIME = 520.0   # safety margin below the 530 s limit

    # --------------------------------------------------------------
    # 1) Global search: Differential Evolution on centre coordinates
    # --------------------------------------------------------------
    elapsed = time.time() - start_time
    remaining = TOTAL_TIME - elapsed
    if remaining > 5.0:
        popsize = 8
        maxiter = max(5, min(25, int(remaining / 12)))
        bounds = [(0.0, 1.0)] * (2 * n)

        best_flat = best_circles[:, :2].reshape(-1)
        rng = np.random.default_rng()
        init_pop = np.empty((popsize, 2 * n), dtype=np.float64)
        init_pop[0] = best_flat
        init_pop[1:] = rng.uniform(size=(popsize - 1, 2 * n))

        def _de_callback(xk, convergence):
            # Stop early if overall budget is almost exhausted
            return (time.time() - start_time) > (TOTAL_TIME - 5.0)

        try:
            result = differential_evolution(
                _objective_de,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                init=init_pop,
                polish=False,
                seed=int(elapsed * 1000) % 2**32,
                updating='deferred',
                disp=False,
                callback=_de_callback,
                strategy='best1bin',
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=1,
            )
            de_x = result.x[:n]
            de_y = result.x[n:]
        except Exception:
            de_x = rng.uniform(size=n)
            de_y = rng.uniform(size=n)

        de_r = _solve_lp_for_positions(de_x, de_y)
        de_sum = float(np.sum(de_r))
        if de_sum > best_sum + 1e-12:
            best_circles = np.column_stack([de_x, de_y, de_r])
            best_sum = de_sum

    # --------------------------------------------------------------
    # 2) Local hill‑climbing: single‑circle perturbations
    # --------------------------------------------------------------
    elapsed = time.time() - start_time
    remaining = TOTAL_TIME - elapsed
    if remaining > 5.0:
        attempts = int(min(10000, remaining * 200))
        rng_hc = np.random.default_rng()
        cur_x = best_circles[:, 0].copy()
        cur_y = best_circles[:, 1].copy()
        cur_sum = best_sum

        for it in range(attempts):
            if time.time() - start_time > TOTAL_TIME - 5.0:
                break
            i = rng_hc.integers(0, n)
            step = 0.02 * (1 - it / attempts) + 0.001
            new_xi = np.clip(cur_x[i] + rng_hc.normal(scale=step), 0.0, 1.0)
            new_yi = np.clip(cur_y[i] + rng_hc.normal(scale=step), 0.0, 1.0)

            new_x = cur_x.copy()
            new_y = cur_y.copy()
            new_x[i] = new_xi
            new_y[i] = new_yi

            new_r = _solve_lp_for_positions(new_x, new_y)
            new_sum = float(np.sum(new_r))

            if new_sum > cur_sum + 1e-12:
                cur_x, cur_y, cur_sum = new_x, new_y, new_sum
                if new_sum > best_sum + 1e-12:
                    best_circles = np.column_stack([new_x, new_y, new_r])
                    best_sum = new_sum

    # --------------------------------------------------------------
    # 3) Final slack‑LP to squeeze any remaining slack
    # --------------------------------------------------------------
    try:
        xb = best_circles[:, 0]
        yb = best_circles[:, 1]
        rb = best_circles[:, 2]
        r_new = _apply_slack_lp(xb, yb, rb)
        border = np.minimum.reduce([xb, 1.0 - xb, yb, 1.0 - yb])
        r_new = np.minimum(r_new, border)
        r_new = np.maximum(r_new, 0.0)
        best_circles = np.column_stack([xb, yb, r_new])
    except Exception:
        pass

    # --------------------------------------------------------------
    # Validation and fall‑backs
    # --------------------------------------------------------------
    if best_circles is not None and _is_valid(best_circles):
        return best_circles

    # Fallback to the global best if still valid
    try:
        gb = GLOBAL_BEST_CONSTRUCTION
        if isinstance(gb, np.ndarray) and gb.shape == (n, 3) and _is_valid(gb):
            return gb.astype(np.float64, copy=True)
    except Exception:
        pass

    # As a last resort, return a zero construction
    return np.zeros((n, 3), dtype=np.float64)
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_code():
    """Run the circle packing constructor for n=26"""
    circles = construct_circles()
    sum_radii = float(np.sum(circles[:, 2]))
    return circles, sum_radii


