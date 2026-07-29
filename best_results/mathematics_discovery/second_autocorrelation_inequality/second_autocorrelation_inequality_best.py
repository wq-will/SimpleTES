# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize

# ---------- Helper functions ----------
def _simpson_l2sq(conv: np.ndarray):
    """Simpson rule integral of (conv)^2 and its gradient w.r.t. conv."""
    m = conv.size
    if m == 0:
        return 0.0, np.zeros_like(conv)
    dx = 1.0 / (m + 1)                     # interval width used by verifier
    y = np.empty(m + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0
    lhs = y[:-1]
    rhs = y[1:]
    l2_sq = (dx / 3.0) * np.sum(lhs * lhs + lhs * rhs + rhs * rhs)
    grad_y = (dx / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    grad_conv = grad_y[1:-1]
    return float(l2_sq), grad_conv

def _l1(conv: np.ndarray):
    """Average (1‑norm) as used by verifier and its gradient."""
    n = conv.size
    if n == 0:
        return 0.0, np.zeros_like(conv)
    denom = n + 1
    val = np.sum(conv) / denom
    grad = np.full_like(conv, 1.0 / denom)
    return float(val), grad

def _smooth_max_and_grad(conv: np.ndarray, beta: float = 30.0):
    """Smooth approximation of the infinity‑norm (log‑sum‑exp) and its gradient."""
    max_val = np.max(conv)
    shifted = conv - max_val                     # ≤ 0
    exp_vals = np.exp(beta * shifted)           # safe: shifted ≤ 0
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0.0:
        return max_val, np.zeros_like(conv)
    smooth_val = max_val + np.log(sum_exp) / beta
    grad = exp_vals / sum_exp
    return smooth_val, grad

def _conv_full_fft(x: np.ndarray, nfft: int) -> np.ndarray:
    """Linear full convolution via FFT (real inputs)."""
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)
    return conv[: 2 * x.size - 1]

def _grad_f_from_conv_grad_fft(f: np.ndarray, g_conv: np.ndarray, nfft: int) -> np.ndarray:
    """Back‑propagate gradient from convolution to the original vector f."""
    n = f.shape[0]
    G = np.fft.rfft(g_conv, n=nfft)
    F_rev = np.fft.rfft(f[::-1], n=nfft)
    full = np.fft.irfft(G * F_rev, n=nfft)
    return 2.0 * full[n-1:2*n-1]

def _objective_and_grad_conv(conv: np.ndarray, beta: float = 30.0, lambda_var: float = 0.0):
    """Smooth‑max AC2 ratio with optional variance penalty."""
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1_val, g_l1 = _l1(conv)
    linf_val, g_linf = _smooth_max_and_grad(conv, beta=beta)

    if l1_val <= 0.0 or linf_val <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1_val * linf_val
    ratio = l2_sq / denom
    d_denom = g_l1 * linf_val + l1_val * g_linf
    grad_ratio = (g_l2 * denom - l2_sq * d_denom) / (denom * denom)

    if lambda_var != 0.0:
        mean = np.mean(conv)
        var = np.mean((conv - mean) ** 2)
        grad_var = 2.0 * (conv - mean) / conv.size
        ratio = ratio - lambda_var * var
        grad_ratio = grad_ratio - lambda_var * grad_var

    return float(ratio), grad_ratio

def _objective_and_grad_conv_exact(conv: np.ndarray):
    """Exact AC2 ratio (hard max) and its gradient."""
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1_val, g_l1 = _l1(conv)
    linf_val = np.max(conv)
    max_idx = np.argmax(conv)
    g_linf = np.zeros_like(conv)
    g_linf[max_idx] = 1.0
    if l1_val <= 0.0 or linf_val <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1_val * linf_val
    ratio = l2_sq / denom
    d_denom = g_l1 * linf_val + l1_val * g_linf
    grad_ratio = (g_l2 * denom - l2_sq * d_denom) / (denom * denom)
    return float(ratio), grad_ratio

def _ratio_exact(conv: np.ndarray):
    """Exact AC2 ratio (no smoothing)."""
    l2_sq, _ = _simpson_l2sq(conv)
    l1_val, _ = _l1(conv)
    linf_val = np.max(conv)
    if l1_val <= 0.0 or linf_val <= 0.0:
        return 0.0
    return l2_sq / (l1_val * linf_val)

def _upsample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation to a different length."""
    n_old = x.shape[0]
    if n_old == target_len:
        return x.copy()
    x_old = np.linspace(-0.5, 0.5, n_old, endpoint=False)
    x_new = np.linspace(-0.5, 0.5, target_len, endpoint=False)
    return np.interp(x_new, x_old, x)

def _init_from_global_or_spectral(N: int):
    """Warm‑start: use stored best construction if possible, otherwise spectral factorisation."""
    if isinstance(GLOBAL_BEST_CONSTRUCTION, (list, tuple, np.ndarray)):
        try:
            arr = np.array(GLOBAL_BEST_CONSTRUCTION, dtype=np.float64)
            if arr.ndim == 1 and arr.size > 0:
                if arr.size != N:
                    arr = _upsample_linear(arr, N)
                return np.clip(arr, 0.0, 1000.0)
        except Exception:
            pass

    # Spectral factorisation of a flat target autocorrelation (vector of ones).
    M = 2 * N - 1
    target = np.ones(M, dtype=np.float64)
    nfft = 1 << ((M - 1).bit_length())
    G = np.fft.rfft(target, n=nfft)
    H = np.sqrt(G)                     # element‑wise complex sqrt
    h_full = np.fft.irfft(H, n=nfft)
    start = (len(h_full) - N) // 2
    init = h_full[start:start + N]
    init = np.maximum(init, 0.0)
    return np.clip(init, 0.0, 1000.0).astype(np.float64)

def construct_function():
    """
    Construct a non‑negative function on [-1/4,1/4] targeting AC2 ratio > 0.97.
    Uses a very high‑resolution warm‑start and a multi‑stage L‑BFGS‑B optimisation
    with smooth‑max approximations, variance penalties and a final exact‑max refinement.
    """
    global GLOBAL_BEST_CONSTRUCTION

    # High resolution (more degrees of freedom than previous best attempts).
    N = 262144
    f = _init_from_global_or_spectral(N)

    # Small random perturbation to break symmetry.
    rng = np.random.default_rng(20231123)
    f = f + rng.normal(scale=1e-6, size=f.shape)
    f = np.clip(f, 0.0, None)

    # Pre‑compute FFT size for this resolution.
    nfft = 1 << ((2 * N - 1).bit_length())

    # Helper closures using the pre‑computed nfft.
    def conv_fft(x):
        return _conv_full_fft(x, nfft)

    def grad_from_conv(x, g_conv):
        return _grad_f_from_conv_grad_fft(x, g_conv, nfft)

    # Optimisation schedule – gradually increase smooth‑max β and reduce variance penalty.
    schedule = [
        (30.0,    1000, 0.020),
        (100.0,   1000, 0.015),
        (300.0,   1000, 0.010),
        (1000.0,  1000, 0.008),
        (3000.0,  1000, 0.005),
        (10000.0, 1000, 0.003),
        (30000.0, 1000, 0.0015),
        (100000.0,1000, 0.001),
        (300000.0,1000, 0.0005),
        (1e6,     2000, 0.0002),
        (1e7,     3000, 0.0),
        (1e8,     5000, 0.0),
        (1e9,     8000, 0.0)
    ]

    TARGET_RATIO = 0.97

    for beta, maxiter, lambda_var in schedule:
        def fun(x):
            x_clip = np.maximum(x, 0.0)
            conv = conv_fft(x_clip)
            ratio, g_conv = _objective_and_grad_conv(conv, beta=beta, lambda_var=lambda_var)
            grad_f = grad_from_conv(x_clip, g_conv)
            return -ratio, -grad_f

        bounds = [(0.0, 1000.0)] * f.size
        res = optimize.minimize(
            fun,
            f,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        f = np.clip(res.x, 0.0, None)

        # Early exit if target reached.
        current_ratio = _ratio_exact(conv_fft(f))
        if current_ratio > TARGET_RATIO:
            break

    # ------------------------------------------------------------------
    # Final exact‑max refinement (hard max) – push the ratio to its limit.
    # ------------------------------------------------------------------
    def fun_exact(x):
        x_clip = np.maximum(x, 0.0)
        conv = conv_fft(x_clip)
        ratio, g_conv = _objective_and_grad_conv_exact(conv)
        grad_f = grad_from_conv(x_clip, g_conv)
        return -ratio, -grad_f

    bounds = [(0.0, 1000.0)] * f.size
    res = optimize.minimize(
        fun_exact,
        f,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 20000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
    )
    f = np.clip(res.x, 0.0, None)

    heights = np.clip(f, 0.0, None)
    GLOBAL_BEST_CONSTRUCTION = heights.tolist()
    return heights.tolist()
# EVOLVE-BLOCK-END



def evaluate_sequence(sequence):
    """
    AC2 verifier logic from the pasted code.

    Raises:
        ValueError: for invalid sequence types/values.
    """
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)  # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


def run_code():
    """Run the C2 optimization constructor and return (solution, self-reported score)."""
    heights = construct_function()
    c2_value = evaluate_sequence(heights)
    return heights, c2_value
