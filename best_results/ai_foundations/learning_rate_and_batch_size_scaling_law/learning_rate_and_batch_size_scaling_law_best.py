
# EVOLVE-BLOCK-START

"""
Improved scaling‑law model (≤ 26 parameters).

- Uses a mixture‑of‑monomials formulation with up to five log‑linear terms.
- Implements numerically‑stable log‑sum‑exp aggregation.
- Exponents are bounded in [-exp_bound, exp_bound] (default ±3.0).
- Deterministic initialization is derived from a linear regression on
  log‑loss → yields a sensible starting point for both coefficients and exponents.
- Multiple random restarts (default 6) improve robustness.
- All operations are per‑sample; no dataset‑wide statistics are accessed at
  inference time.
"""

import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _log_sum_exp(log_terms: np.ndarray) -> np.ndarray:
    """
    Numerically stable computation of sum_i exp(log_terms_i) for each row.

    Parameters
    ----------
    log_terms : np.ndarray, shape (N, T)

    Returns
    -------
    np.ndarray, shape (N,)
        Sum of exponentials for each sample.
    """
    max_log = np.max(log_terms, axis=1, keepdims=True)          # (N,1)
    shifted = np.exp(log_terms - max_log)                      # (N,T)
    summed = np.sum(shifted, axis=1)                           # (N,)
    return np.exp(max_log.squeeze()) * summed                  # (N,)


def _prepare_features(data_points: np.ndarray):
    """
    Convert raw hyper‑parameter values to log‑space features.
    Returns four 1‑D arrays (log_lr, log_bsz, log_data, log_param).
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]                     # single sample → (1,4)
    if X.shape[1] != 4:
        raise ValueError("data_points must have exactly 4 columns")
    if np.any(X <= 0):
        raise ValueError("All hyper‑parameter values must be positive")
    lr, bsz, data, param = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.log(lr), np.log(bsz), np.log(data), np.log(param)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def scaling_law_func(data_points, params):
    """
    Predict language‑model loss for a set of hyper‑parameter configurations.

    Parameters
    ----------
    data_points : (N,4) array – columns are [lr, bsz, data_size,
                 non_embedding_param_size].
    params : 1‑D array of up to 26 parameters (see ``fit_scaling_law``).

    Returns
    -------
    preds : (N,) array of predicted loss values (scalar if N == 1).
    """
    # ---- log‑features -------------------------------------------------
    log_lr, log_bsz, log_data, log_par = _prepare_features(data_points)

    # ---- unpack parameters -------------------------------------------
    p = np.asarray(params, dtype=np.float64).ravel()
    # bias is present iff length % 5 == 1 (consistent with original convention)
    if (len(p) % 5) == 1:
        bias = p[-1]
        term_params = p[:-1]
    else:
        bias = 0.0
        term_params = p

    T = len(term_params) // 5
    if T == 0:
        # degenerate case – return zero or bias only
        return np.full_like(log_lr, bias) if bias != 0.0 else np.zeros_like(log_lr)

    term_params = term_params.reshape(T, 5)          # [log_coeff, e_lr, e_bsz, e_data, e_par]
    log_coeffs = term_params[:, 0]                   # (T,)
    exps = term_params[:, 1:]                        # (T,4)

    # ---- compute log‑terms (log of each monomial) --------------------
    logs = np.stack([log_lr, log_bsz, log_data, log_par], axis=1)   # (N,4)
    dot = logs @ exps.T                              # (N,T)
    log_terms = dot + log_coeffs[None, :]            # (N,T)

    LOG_CLIP = 50.0
    log_terms = np.clip(log_terms, -LOG_CLIP, LOG_CLIP)

    # ---- stable aggregation ------------------------------------------
    term_sum = _log_sum_exp(log_terms)               # (N,)

    preds = term_sum + bias

    return preds[0] if preds.shape[0] == 1 else preds


def fit_scaling_law(
    data_points,
    loss_values,
    *,
    n_terms=5,
    n_restarts=6,
    exp_bound=3.0,
    max_nfev=5000,
    random_state=None,
):
    """
    Fit the scaling‑law model to observed (hyper‑parameter, loss) data.

    The optimisation uses ``scipy.optimize.least_squares`` with a robust loss
    and several random restarts.  A deterministic initial guess is obtained
    from a linear regression of ``log(loss)`` against the log‑features,
    providing a sensible starting point for both coefficients and exponents.

    Parameters
    ----------
    data_points : array_like, shape (N, 4)
        Hyper‑parameter configurations.
    loss_values : array_like, shape (N,)
        Observed language‑model cross‑entropy loss values (positive).
    n_terms : int, default 5
        Number of exponential terms (must satisfy 5·n_terms + 1 ≤ 26).
    n_restarts : int, default 6
        Number of optimisation attempts (including a deterministic start).
    exp_bound : float, default 3.0
        Absolute bound for each exponent parameter.
    max_nfev : int, default 5000
        Maximum number of function evaluations per optimisation run.
    random_state : int or np.random.Generator, optional
        Seed/Generator for reproducible random initialisations.

    Returns
    -------
    opt_params : np.ndarray, shape (P,) where P ≤ 26.
        Optimised parameter vector.
    """
    # ---- basic validation ------------------------------------------------
    X = np.asarray(data_points, dtype=np.float64)
    y = np.asarray(loss_values, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("data_points and loss_values must contain the same number of rows")
    if X.shape[1] != 4:
        raise ValueError("data_points must have exactly 4 columns")
    if np.any(X <= 0):
        raise ValueError("All hyper‑parameter values must be strictly positive")
    if np.any(y <= 0):
        raise ValueError("All loss values must be strictly positive")

    T = int(n_terms)
    if not (1 <= T <= 5):
        raise ValueError("n_terms must be between 1 and 5")
    P = T * 5 + 1                     # coeff + 4 exps per term + bias
    if P > 26:
        raise ValueError(f"Parameter budget exceeded: {P} > 26")

    N = X.shape[0]

    # ---- bounds ----------------------------------------------------------
    lo = np.full(P, -np.inf)
    hi = np.full(P,  np.inf)

    for t in range(T):
        base = t * 5
        lo[base + 1 : base + 5] = -exp_bound
        hi[base + 1 : base + 5] =  exp_bound
    # bias left unbounded

    # ---- deterministic initial guess using linear regression -------------
    logs = np.log(X)                  # (N,4)
    y_log = np.log(y)                 # (N,)

    D = np.column_stack([np.ones(N), logs])   # (N,5)
    beta = np.linalg.lstsq(D, y_log, rcond=None)[0]   # shape (5,)
    intercept = beta[0]                # corresponds to log_coeff + log(T)
    exponents_est = beta[1:]           # (4,)

    exponents_est = np.clip(exponents_est, -exp_bound, exp_bound)

    init_det = np.zeros(P, dtype=np.float64)

    log_coeff = intercept - np.log(T)
    init_det[0 : T * 5 : 5] = log_coeff

    for t in range(T):
        start = t * 5 + 1
        init_det[start:start + 4] = exponents_est

    init_det[-1] = np.mean(y)          # bias

    # ---- helper to create randomised initial guess -----------------------
    rng = np.random.default_rng(random_state)

    def _make_initial_guess(deterministic=False):
        if deterministic:
            return init_det.copy()
        p = init_det.copy()
        # perturb log‑coefficients
        p[0 : T * 5 : 5] += rng.normal(scale=0.2, size=T)
        # perturb exponents and enforce bounds
        for t in range(T):
            start = t * 5 + 1
            perturb = rng.uniform(-0.5, 0.5, size=4)
            p[start:start + 4] = np.clip(p[start:start + 4] + perturb,
                                         -exp_bound, exp_bound)
        # perturb bias slightly
        p[-1] += rng.normal(scale=0.05)
        return p

    # ---- residual function -----------------------------------------------
    def _residuals(p):
        pred = scaling_law_func(X, p)   # (N,)
        return pred - y

    # ---- optimisation with multiple restarts -----------------------------
    best_params = None
    best_cost = np.inf

    attempts = [_make_initial_guess(deterministic=True)]
    for _ in range(max(0, n_restarts - 1)):
        attempts.append(_make_initial_guess(deterministic=False))

    for start in attempts:
        res = least_squares(
            _residuals,
            start,
            bounds=(lo, hi),
            method='trf',
            loss='soft_l1',
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        cur_cost = res.cost if hasattr(res, "cost") else np.sum(res.fun ** 2) / 2.0
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_params = res.x.copy()
        if res.success and cur_cost < 1e-12:
            break

    if best_params is None:
        best_params = init_det.copy()

    if best_params.shape[0] > 26:
        best_params = best_params[:26]

    return best_params
# EVOLVE-BLOCK-END
