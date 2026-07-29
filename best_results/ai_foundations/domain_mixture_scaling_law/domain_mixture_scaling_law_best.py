
# EVOLVE-BLOCK-START

"""
35‑parameter scaling‑law for predicting multi‑domain cross‑entropy losses from
domain mixture proportions.  The model consists of:

  * a (5,) – additive post‑exponential bias (output offset)
  * b (5,) – additive pre‑exponential bias (shifts the exponent)
  * rc (5,) – raw exponents (log‑space, c = exp(rc) > 0)
  * rd (5,) – raw gains     (log‑space, d = exp(rd) > 0)
  * w_sym (15,) – upper‑triangular (incl. diagonal) entries of a symmetric
                 5 × 5 mixing matrix W (W = Wᵀ)

For a mixture vector p (∑pᵢ = 1) the forward pass is:

    c = exp(rc)                     # enforce positivity
    d = exp(rd)
    φ = (p ** c) * d                # per‑domain power‑transform + gain
    z = φ @ Wᵀ + b                  # linear mixing + pre‑exp bias
    ŷ = a + exp( clip(z, -30, 30) ) # post‑exp bias + stable exponential
"""

import numpy as np
from scipy.optimize import least_squares


def _sym_from_params(sym_vec):
    """Re‑assemble a 5 × 5 symmetric matrix from its 15‑element upper‑triangular vector."""
    W = np.zeros((5, 5), dtype=float)
    idx = 0
    for i in range(5):
        for j in range(i, 5):
            val = sym_vec[idx]
            W[i, j] = val
            W[j, i] = val
            idx += 1
    return W


def _flatten_sym(W):
    """Extract the 15 upper‑triangular (incl. diagonal) entries of a symmetric 5 × 5 matrix."""
    return np.array([W[i, j] for i in range(5) for j in range(i, 5)], dtype=float)


def scaling_law_func(data_points, params):
    """
    Predict multi‑domain loss values.

    Parameters
    ----------
    data_points : ndarray, shape (N, 5)
        Domain mixture proportions (rows sum to 1).
    params : ndarray, shape (35,)
        Concatenated parameters: [a (5), b (5), rc (5), rd (5), w_sym (15)].

    Returns
    -------
    preds : ndarray, shape (N, 5)
        Predicted loss for each domain.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 35:
        raise ValueError(f'Expected 35 parameters, got {p.size}')

    a = p[:5]          # post‑exp bias
    b = p[5:10]        # pre‑exp bias
    rc = p[10:15]      # log‑exponents
    rd = p[15:20]      # log‑gains
    w_sym = p[20:]     # symmetric mixing matrix entries

    # Positive exponents and gains (log‑parametrisation).
    c = np.exp(rc)
    d = np.exp(rd)

    # Clamp to generous but safe ranges (pre‑empt numerical explosions).
    c = np.clip(c, 0.01, 5.0)
    d = np.clip(d, 0.01, 10.0)

    # Reconstruct symmetric mixing matrix.
    W = _sym_from_params(w_sym)   # (5,5)

    # Power‑transform + per‑input scaling.
    Xc = X ** c                    # (N,5)
    phi = Xc * d                   # (N,5)

    # Linear mixing + pre‑exp bias.
    z = phi @ W.T + b              # (N,5)

    # Clamp before exponentiation for numerical stability.
    z = np.clip(z, -30.0, 30.0)

    # Exponential non‑linearity + post‑exp bias.
    preds = a + np.exp(z)          # (N,5)
    return preds


def _init_params_linear_log(X, Y):
    """
    Initialise the 35‑parameter model using a log‑linear approximation.

    Steps
    -----
    1. Choose a0 slightly below each domain's minimum loss (ensures positivity).
    2. Shift the targets by a0 and take the log → Z.
    3. Solve Z ≈ X @ Wᵀ + b via ordinary least squares.
    4. Symmetrise W and flatten its upper‑triangular part.
    5. Initialise rc = rd = 0 (i.e. c = d = 1).
    """
    eps = 1e-8
    margin = 0.05
    a0 = np.min(Y, axis=0) - margin           # (5,)

    Y_adj = np.clip(Y - a0, eps, None)        # enforce positivity
    Z = np.log(Y_adj)                          # (N,5)

    N = X.shape[0]
    A = np.column_stack([np.ones(N), X])       # (N,6) – intercept + features
    coeffs, *_ = np.linalg.lstsq(A, Z, rcond=None)   # (6,5)

    b0 = coeffs[0]                             # (5,)
    W0 = coeffs[1:].T                          # (5,5)

    # Symmetrise the mixing matrix.
    W_sym0 = (W0 + W0.T) / 2.0
    w_sym0 = _flatten_sym(W_sym0)              # (15,)

    # Log‑space exponents / gains start at zero → c = d = 1.
    rc0 = np.zeros(5, dtype=float)
    rd0 = np.zeros(5, dtype=float)

    return np.concatenate([a0, b0, rc0, rd0, w_sym0])


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 35‑parameter scaling law to observed mixture‑loss data.

    Uses bounded non‑linear least squares with a robust loss function
    and multiple random restarts to avoid poor local minima.

    Parameters
    ----------
    data_points : ndarray, shape (N, 5)
        Domain mixture proportions.
    loss_values : ndarray, shape (N, 5)
        Observed multi‑domain cross‑entropy losses.

    Returns
    -------
    params_opt : ndarray, shape (35,)
        Optimised parameter vector ready for ``scaling_law_func``.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    Y = np.atleast_2d(np.asarray(loss_values, dtype=float))

    if X.shape[1] != 5:
        raise ValueError(f'Expected 5 input dimensions, got {X.shape[1]}')
    if Y.shape != X.shape:
        raise ValueError(f'Loss array must match shape of inputs {X.shape}, got {Y.shape}')

    # ------------------- 1️⃣ Initial guess -------------------
    init_params = _init_params_linear_log(X, Y)

    # ------------------- 2️⃣ Parameter bounds -------------------
    # a, b: unrestricted.
    lower_a = -np.inf * np.ones(5)
    upper_a =  np.inf * np.ones(5)

    lower_b = -np.inf * np.ones(5)
    upper_b =  np.inf * np.ones(5)

    # rc, rd: log‑space bounds that enforce c∈[0.01,5] and d∈[0.01,10].
    lower_rc = np.full(5, np.log(0.01))   # ≈ -4.605
    upper_rc = np.full(5, np.log(5.0))    # ≈ 1.609

    lower_rd = np.full(5, np.log(0.01))   # ≈ -4.605
    upper_rd = np.full(5, np.log(10.0))   # ≈ 2.303

    # w_sym (symmetric mixing matrix): unrestricted.
    lower_w = -np.inf * np.ones(15)
    upper_w =  np.inf * np.ones(15)

    lower = np.concatenate([lower_a, lower_b, lower_rc, lower_rd, lower_w])
    upper = np.concatenate([upper_a, upper_b, upper_rc, upper_rd, upper_w])

    # ------------------- 3️⃣ Residual function -------------------
    # Light L2 regularisation on all parameters except the post‑exp bias a.
    reg_lambda = 1e-12

    def residuals(p):
        pred = scaling_law_func(X, p)          # (N,5)
        data_res = (pred - Y).ravel()          # (N*5,)
        reg = np.sqrt(reg_lambda) * p[5:]      # regularise everything but a
        return np.concatenate([data_res, reg])

    # ------------------- 4️⃣ Primary optimisation -------------------
    best_params = init_params
    result = least_squares(
        residuals,
        x0=init_params,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',               # robust loss against occasional outliers
        f_scale=1.0,
        max_nfev=15000,
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
    )
    if result.success:
        best_params = result.x
        best_cost = result.cost
    else:
        best_cost = np.inf

    # ------------------- 5️⃣ Random restarts -------------------
    rng = np.random.default_rng(12345)   # deterministic seed for reproducibility
    n_restart = 50                        # trade‑off between robustness and runtime
    for _ in range(n_restart):
        # Relative Gaussian perturbation (~20 % of magnitude) to explore the space.
        scale = 0.2
        perturb = best_params + rng.normal(scale=scale,
                                           size=best_params.shape) * np.maximum(np.abs(best_params), 1.0)
        perturb = np.clip(perturb, lower, upper)

        res = least_squares(
            residuals,
            x0=perturb,
            bounds=(lower, upper),
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=8000,
            ftol=1e-12,
            xtol=1e-12,
            verbose=0,
        )
        if res.success and res.cost < best_cost:
            best_params = res.x
            best_cost = res.cost

    return best_params
# EVOLVE-BLOCK-END
