
# EVOLVE-BLOCK-START

"""
Scaling law for parallel LLM evaluation using a compact 4‑parameter product form:

    loss(N, P) = (a + b * N^{-c}) * (1 + d * P^{-0.5})

where:
 - N : number of model parameters,
 - P : parallel size (number of copies),
 - a ≥ 0, b ≥ 0 are baseline and scaling amplitude,
 - c > 0 controls decay with model size,
 - d ≥ 0 controls the strength of parallelism benefit.

The exponent on P is fixed to ½ (i.e. 1/√P) to reflect diminishing‑returns,
while keeping the total parameter count at four.
"""

import numpy as np
from scipy.optimize import curve_fit


def _product_parallel_model(X, a, b, c, d):
    """
    Vectorised evaluation of the 4‑parameter product law for curve fitting.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Columns are [num_params, parallel_size].
    a, b, c, d : float
        Model parameters.

    Returns
    -------
    loss : ndarray, shape (n,)
        Predicted loss.
    """
    # Guard against zero/negative inputs
    N = np.maximum(X[:, 0], 1.0)
    P = np.maximum(X[:, 1], 1.0)

    # Numerically stable power terms
    term_N = np.exp(-c * np.log(N))   # N^{-c}
    term_P = 1.0 / np.sqrt(P)         # P^{-0.5}

    return (a + b * term_N) * (1.0 + d * term_P)


def _pad_params(params):
    """
    Ensure ``params`` is a 2‑D array with exactly four columns.
    Missing entries are filled with zeros (no contribution).
    """
    p = np.atleast_2d(np.asarray(params, dtype=float))
    if p.shape[1] > 4:
        raise ValueError("Parameter array may contain at most 4 columns.")
    if p.shape[1] < 4:
        pad_width = 4 - p.shape[1]
        p = np.pad(p, ((0, 0), (0, pad_width)), constant_values=0.0)
    return p


def scaling_law_func(data_points, params):
    """
    Predict loss values using the 4‑parameter product scaling law.

    Parameters
    ----------
    data_points : array‑like, shape (N, 2)
        Columns are [num_params, parallel_size].
    params : array‑like
        Either a 1‑D array of up to 4 parameters (a, b, c, d) or a
        2‑D array of shape (K, ≤4) defining K hypotheses.

    Returns
    -------
    preds : ndarray
        Shape (N,) for a single hypothesis or (N, K) for K hypotheses.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    N = np.maximum(X[:, 0], 1.0)
    P = np.maximum(X[:, 1], 1.0)

    log_N = np.log(N)
    term_P = 1.0 / np.sqrt(P)          # shared across all hypotheses

    p_arr = _pad_params(params)        # (K, 4)
    K = p_arr.shape[0]

    if K == 1:
        a, b, c, d = p_arr[0]
        term_N = np.exp(-c * log_N)
        return (a + b * term_N) * (1.0 + d * term_P)

    # Multiple hypotheses – compute each column independently
    preds = np.empty((X.shape[0], K), dtype=float)
    for i in range(K):
        a, b, c, d = p_arr[i]
        term_N = np.exp(-c * log_N)
        preds[:, i] = (a + b * term_N) * (1.0 + d * term_P)

    return preds[:, 0] if K == 1 else preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4‑parameter product law to observed loss data.

    The routine first attempts a non‑linear least‑squares fit via
    ``scipy.optimize.curve_fit``.  If optimisation fails, a deterministic
    linear least‑squares fallback for the additive variant

        loss ≈ a + b·N^{-c_fixed} + d·P^{-0.5}

    is used instead.

    Parameters
    ----------
    data_points : (N, 2) array‑like
        Columns are [num_params, parallel_size].
    loss_values : (N,) array‑like
        Observed language‑modeling loss.

    Returns
    -------
    opt_params : ndarray, shape (4,)
        Optimised parameters [a, b, c, d].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.ravel(np.asarray(loss_values, dtype=float))

    # --------------------------------------------------------------
    # 1️⃣ Primary non‑linear fit
    # --------------------------------------------------------------
    a0 = np.min(y) * 0.95                 # Slightly below smallest loss
    b0 = max(np.max(y) - a0, 1e-6)        # Scale of the N‑dependent term
    c0 = 0.2                               # Typical exponent for model size
    d0 = 0.1                               # Modest parallel benefit
    p0 = np.array([a0, b0, c0, d0], dtype=float)

    lower_bounds = [0.0, 0.0, 0.0, 0.0]    # non‑negative parameters
    upper_bounds = [np.inf, np.inf, 5.0, np.inf]

    try:
        popt, _ = curve_fit(
            _product_parallel_model,
            X,
            y,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
        return popt
    except Exception:
        # --------------------------------------------------------------
        # 2️⃣ Robust fallback – additive variant with fixed P exponent.
        #    loss ≈ a + b·N^{-c_fixed} + d·P^{-0.5}
        # --------------------------------------------------------------
        c_fixed = 0.2
        N = np.maximum(X[:, 0], 1.0)
        P = np.maximum(X[:, 1], 1.0)

        term_N = np.exp(-c_fixed * np.log(N))   # N^{-c_fixed}
        term_P = 1.0 / np.sqrt(P)               # P^{-0.5}

        design = np.column_stack([np.ones_like(N), term_N, term_P])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        a_f, b_f, d_f = coeffs

        return np.array([a_f, b_f, c_fixed, d_f], dtype=float)
# EVOLVE-BLOCK-END
