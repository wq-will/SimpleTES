
# EVOLVE-BLOCK-START

"""
Six‑parameter U‑shaped scaling law (exponential baseline + Gaussian bump).

Model (per task):
    y(x) = a + b·exp(c·x) + d·exp( -½·((x−e)/f)² ),
    where f = exp(log_f) > 0.

Only the three parameters (c, e, log_f) are non‑linear.  For any fixed
(c, e, f) the remaining linear coefficients (a, b, d) are obtained by a
closed‑form least‑squares solve.  This reduces the global optimisation to
three dimensions, keeping the total number of free parameters at six while
allowing a robust search for the characteristic U‑shape.

The implementation emphasises:
  • numerical stability (clipping of exponent arguments);
  • thorough global optimisation (multiple DE runs with generous
    population/iteration settings);
  • heuristic seeding at the observed worst point;
  • two local refinements (soft‑L1 and plain L2) to escape any remaining
    local minima;
  • a clean public API that works for a single task or for many tasks
    simultaneously.
"""

import numpy as np
from scipy.optimize import differential_evolution, least_squares

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _prepare_x(data_points):
    """Extract a 1‑D array of log‑FLOPs from the input.

    Parameters
    ----------
    data_points : array‑like, shape (N,) or (N, D)

    Returns
    -------
    x : ndarray, shape (N,)
    """
    arr = np.asarray(data_points, dtype=float)
    if arr.ndim == 1:
        return arr
    # assume first column contains log_flops
    return arr[:, 0]


def _solve_linear_coeffs(x, y, c, e, f):
    """Solve analytically for the linear coefficients (a, b, d).

    Parameters
    ----------
    x : ndarray (N,)
    y : ndarray (N,)
    c, e, f : float, with f > 0

    Returns
    -------
    coeffs : ndarray (3,)   [a, b, d]
    """
    # safe exponentials
    exp_part = np.exp(np.clip(c * x, -200, 200))
    gauss_part = np.exp(np.clip(-0.5 * ((x - e) / f) ** 2, -200, 200))
    A = np.column_stack((np.ones_like(x), exp_part, gauss_part))
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs  # a, b, d


def _sse_for_triplet(x, y, c, e, log_f):
    """Compute sum‑of‑squared‑errors and the full 6‑parameter vector for a
    given (c, e, log_f)."""
    f = np.exp(log_f)
    a, b, d = _solve_linear_coeffs(x, y, c, e, f)
    exp_term = np.exp(np.clip(c * x, -200, 200))
    gauss_term = np.exp(np.clip(-0.5 * ((x - e) / f) ** 2, -200, 200))
    pred = a + b * exp_term + d * gauss_term
    sse = np.sum((pred - y) ** 2)
    return sse, np.array([a, b, d, c, e, log_f], dtype=float)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def scaling_law_func(data_points, params):
    """Predict Brier‑score from log‑FLOPs using the fitted scaling law.

    Parameters
    ----------
    data_points : array‑like, shape (N,) or (N, D)
        Log10 FLOPs (in 1 E21 units). Only the first column is used.
    params : array‑like, shape (6,) or (T, 6)
        Model parameters ordered as [a, b, d, c, e, log_f].

    Returns
    -------
    preds : ndarray, shape (N,) or (N, T)
        Predicted Brier‑score values (more negative = better).
    """
    x = _prepare_x(data_points)                     # (N,)
    P = np.atleast_2d(np.asarray(params, dtype=float))   # (T,6) or (1,6)

    # Unpack with a broadcasting axis (T,1)
    a = P[:, 0][:, None]
    b = P[:, 1][:, None]
    d = P[:, 2][:, None]
    c = P[:, 3][:, None]
    e = P[:, 4][:, None]
    log_f = P[:, 5][:, None]
    f = np.exp(log_f)               # enforce positivity

    # Vectorise: (1, N) broadcasted to (T, N)
    x_row = x[None, :]              # (1, N)

    # Clip exponent arguments for safety
    exp_arg = np.clip(c * x_row, -200, 200)
    gauss_arg = np.clip(-0.5 * ((x_row - e) / f) ** 2, -200, 200)

    preds = a + b * np.exp(exp_arg) + d * np.exp(gauss_arg)   # (T, N)
    preds = preds.T                                          # (N, T)

    if preds.shape[1] == 1:
        return preds[:, 0]
    return preds


def fit_scaling_law(data_points, loss_values):
    """Fit the six‑parameter U‑shaped scaling law to observed data.

    The optimisation per task consists of:
      1. Multiple differential‑evolution runs on the three non‑linear
         parameters (c, e, log_f), with the linear coefficients solved
         analytically for each candidate.
      2. A heuristic seed that places the Gaussian bump at the worst‑loss
         point.
      3. Two local refinements (soft‑L1 and plain L2) using `least_squares`,
         again solving the linear part analytically.
      4. Selection of the candidate with the lowest SSE.

    Parameters
    ----------
    data_points : array‑like, shape (N,) or (N, D)
        Log10 FLOPs (in 1 E21 units).
    loss_values : array‑like, shape (N,) or (N, T)
        Observed Brier‑score values (negative = better).

    Returns
    -------
    params_opt : ndarray, shape (6,) or (T, 6)
        Optimised parameters per task ordered as [a, b, d, c, e, log_f].
    """
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    x = _prepare_x(data_points)                     # (N,)
    Y = np.asarray(loss_values, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]                               # (N, 1)
    N, T = Y.shape

    rng = np.random.default_rng()
    all_params = []

    # ------------------------------------------------------------------
    # Per‑task optimisation
    # ------------------------------------------------------------------
    for t in range(T):
        y = Y[:, t]

        # Bounds for non‑linear parameters
        x_min, x_max = np.min(x), np.max(x)
        c_bounds = (-30.0, -1e-6)          # decreasing exponential baseline
        e_bounds = (x_min, x_max)         # centre of bump inside the data
        logf_bounds = (-8.0, 8.0)          # width ∈ [≈0.0003, ≈2980] after exp
        nonlin_bounds = [c_bounds, e_bounds, logf_bounds]

        # ------------------------------------------------------------------
        # Helper: compute SSE for a given (c, e, log_f)
        # ------------------------------------------------------------------
        def _candidate_sse(p):
            c, e, log_f = p
            sse, _ = _sse_for_triplet(x, y, c, e, log_f)
            return sse

        # ------------------------------------------------------------------
        # 1) Multiple differential‑evolution runs
        # ------------------------------------------------------------------
        best_sse = np.inf
        best_triplet = None
        n_de_runs = 4
        for _ in range(n_de_runs):
            de_res = differential_evolution(
                _candidate_sse,
                bounds=nonlin_bounds,
                strategy='best1bin',
                maxiter=250,
                popsize=20,
                tol=1e-7,
                polish=False,
                updating='deferred',
                seed=int(rng.integers(2**32 - 1)),
                disp=False,
            )
            sse, params_full = _sse_for_triplet(x, y, *de_res.x)
            if sse < best_sse:
                best_sse = sse
                best_triplet = de_res.x

        # ------------------------------------------------------------------
        # 2) Heuristic seed – centre the bump at the worst observed point
        # ------------------------------------------------------------------
        heuristic_c = -1.0
        heuristic_e = x[np.argmax(y)]                     # location of max loss
        heuristic_f = max((x_max - x_min) / 4.0, 1e-3)    # moderate width
        heuristic_logf = np.log(heuristic_f)

        # Clip to bounds
        heuristic_c = np.clip(heuristic_c, *c_bounds)
        heuristic_e = np.clip(heuristic_e, *e_bounds)
        heuristic_logf = np.clip(heuristic_logf, *logf_bounds)

        sse_h, _ = _sse_for_triplet(x, y, heuristic_c, heuristic_e, heuristic_logf)
        if sse_h < best_sse:
            best_sse = sse_h
            best_triplet = np.array([heuristic_c, heuristic_e, heuristic_logf])

        # ------------------------------------------------------------------
        # 3) Local refinement – soft‑L1 and plain L2
        # ------------------------------------------------------------------
        candidates = []

        # Pre‑refinement candidate
        sse_pre, params_pre = _sse_for_triplet(x, y, *best_triplet)
        candidates.append((sse_pre, params_pre))

        # Residual function for LS (linear part solved analytically each call)
        def _ls_residual(p):
            c, e, log_f = p
            f = np.exp(log_f)
            a, b, d = _solve_linear_coeffs(x, y, c, e, f)
            exp_term = np.exp(np.clip(c * x, -200, 200))
            gauss_term = np.exp(np.clip(-0.5 * ((x - e) / f) ** 2, -200, 200))
            pred = a + b * exp_term + d * gauss_term
            return pred - y

        # Soft‑L1 refinement
        ls_soft = least_squares(
            _ls_residual,
            x0=best_triplet,
            bounds=([c_bounds[0], e_bounds[0], logf_bounds[0]],
                    [c_bounds[1], e_bounds[1], logf_bounds[1]]),
            method='trf',
            loss='soft_l1',
            ftol=1e-9,
            xtol=1e-9,
            max_nfev=3000,
        )
        if ls_soft.success:
            sse_soft, params_soft = _sse_for_triplet(x, y, *ls_soft.x)
            candidates.append((sse_soft, params_soft))

        # Plain L2 refinement
        ls_l2 = least_squares(
            _ls_residual,
            x0=best_triplet,
            bounds=([c_bounds[0], e_bounds[0], logf_bounds[0]],
                    [c_bounds[1], e_bounds[1], logf_bounds[1]]),
            method='trf',
            loss='linear',
            ftol=1e-9,
            xtol=1e-9,
            max_nfev=3000,
        )
        if ls_l2.success:
            sse_l2, params_l2 = _sse_for_triplet(x, y, *ls_l2.x)
            candidates.append((sse_l2, params_l2))

        # Choose the best candidate by SSE
        best_params = min(candidates, key=lambda tup: tup[0])[1]
        all_params.append(best_params)

    # Stack results (T, 6) and flatten for single‑task case
    params_opt = np.vstack(all_params)
    if params_opt.shape[0] == 1:
        return params_opt[0]
    return params_opt
# EVOLVE-BLOCK-END
