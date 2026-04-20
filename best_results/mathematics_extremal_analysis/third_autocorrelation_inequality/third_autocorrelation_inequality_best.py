# EVOLVE-BLOCK-START

import numpy as np
import time
import math

def construct_function():
    """
    Construct a 400‑point discrete function on [-1/4, 1/4] that minimises the C₃ constant.
    Warm‑starts from GLOBAL_BEST_CONSTRUCTION (if available) and runs a
    sum‑preserving Adam optimiser that focuses on the worst autocorrelation entries
    (hard‑max top‑K). A quick golden‑section line‑search fine‑tunes the zero‑mean
    direction, and a tiny stochastic hill‑climb exploits any remaining time.
    """
    # --------------------------------------------------------------
    # Problem parameters
    # --------------------------------------------------------------
    n = 400
    dx = 0.5 / n
    target_sum = 2.0 * n                     # Σ f_i = 2·n → integral = 1

    # Baseline constant function (already satisfies the sum)
    baseline = np.full(n, target_sum / n, dtype=np.float64)   # = 2.0 everywhere

    # --------------------------------------------------------------
    # Warm‑start from GLOBAL_BEST_CONSTRUCTION if possible
    # --------------------------------------------------------------
    f = baseline.copy()
    try:
        gbc = GLOBAL_BEST_CONSTRUCTION
        if gbc is not None:
            arr = np.asarray(gbc, dtype=np.float64).ravel()
            if arr.shape[0] == n and np.all(np.isfinite(arr)):
                s = arr.sum()
                if np.abs(s) > 1e-12:
                    arr = arr * (target_sum / s)   # enforce exact sum
                f = arr.copy()
    except Exception:
        pass

    rng = np.random.default_rng()
    # If warm‑start equals baseline, add a tiny random perturbation
    if np.linalg.norm(f - baseline) < 1e-12:
        f += rng.normal(scale=0.02, size=n)
        f -= (np.sum(f) - target_sum) / n

    # --------------------------------------------------------------
    # FFT settings (full linear convolution)
    # --------------------------------------------------------------
    nconv = 2 * n - 1

    def conv_full(v):
        """Full (dx‑scaled) autocorrelation of vector v."""
        V = np.fft.rfft(v, nconv)
        return np.fft.irfft(V * V, nconv) * dx

    # --------------------------------------------------------------
    # Book‑keeping
    # --------------------------------------------------------------
    best_f = f.copy()
    best_val = float(np.max(np.abs(conv_full(best_f))))
    # Indices that map the full convolution gradient back to the original vector
    grad_slice = (2 * n - 2) - np.arange(n)

    # --------------------------------------------------------------
    # Optimisation hyper‑parameters & timing budget
    # --------------------------------------------------------------
    start_time = time.time()
    TIME_LIMIT = 66.0                     # safety margin below the hard 70 s limit
    MAIN_TIME = TIME_LIMIT - 4.0          # reserve a few seconds for line‑search & cleanup

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m = np.zeros_like(f)
    v = np.zeros_like(f)

    # Top‑K schedule (aggressive exponential decay)
    K_start = 250
    K_end = 2
    decay_factor = 5.0            # larger → faster decay
    max_norm = 0.5
    stagnation = 0
    STAGN_LIMIT = 1500

    it = 0
    # --------------------------------------------------------------
    # Main hard‑max top‑K Adam refinement loop
    # --------------------------------------------------------------
    while time.time() - start_time < MAIN_TIME:
        it += 1

        # FFT of current iterate
        f_fft = np.fft.rfft(f, nconv)

        # Full autocorrelation (dx‑scaled)
        conv = np.fft.irfft(f_fft * f_fft, nconv) * dx
        abs_conv = np.abs(conv)
        cur = float(np.max(abs_conv))

        # Track the best solution seen so far
        if cur < best_val:
            best_val = cur
            best_f = f.copy()
            stagnation = 0
        else:
            stagnation += 1

        elapsed = time.time() - start_time
        frac = elapsed / MAIN_TIME

        # Determine K for this iteration (hard‑max on K worst entries)
        K = int(K_start * math.exp(-decay_factor * frac) + K_end)
        K = max(K_end, min(K, K_start))
        if K >= abs_conv.size:
            idxs = np.arange(abs_conv.size)
        else:
            idxs = np.argpartition(abs_conv, -K)[-K:]

        # Weight vector: sign of the K worst autocorrelation entries
        w = np.zeros_like(conv)
        w[idxs] = np.sign(conv[idxs])

        # Gradient of the weighted max‑abs autocorrelation w.r.t. f
        w_rev_fft = np.fft.rfft(w[::-1], nconv)
        grad_full = 2.0 * np.fft.irfft(f_fft * w_rev_fft, nconv) * dx
        grad_f = grad_full[grad_slice]

        # Project gradient onto the zero‑sum subspace (preserve total sum)
        grad_f -= np.mean(grad_f)

        # Gradient clipping
        gn = np.linalg.norm(grad_f)
        if gn > max_norm:
            grad_f = grad_f * (max_norm / gn)

        # Adam update with cosine‑annealed learning rate
        m = beta1 * m + (1.0 - beta1) * grad_f
        v = beta2 * v + (1.0 - beta2) * (grad_f ** 2)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)

        lr_max, lr_min = 0.8, 5e-5
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * frac))

        f = f - lr * m_hat / (np.sqrt(v_hat) + eps)

        # Re‑enforce exact sum (remove tiny drift)
        f -= (np.sum(f) - target_sum) / n

        # Random kick to escape plateaus (probability decays)
        if rng.random() < 0.04 * (1.0 - frac):
            sigma = 0.025 * (1.0 - frac)
            f += rng.normal(scale=sigma, size=n)
            f -= (np.sum(f) - target_sum) / n

        # Restart if stagnating for too long
        if stagnation >= STAGN_LIMIT:
            f = best_f + rng.normal(scale=0.12, size=n)
            f -= (np.sum(f) - target_sum) / n
            m.fill(0.0)
            v.fill(0.0)
            stagnation = 0

    # --------------------------------------------------------------
    # Scalar golden‑section line‑search (optimise overall amplitude)
    # --------------------------------------------------------------
    # The best candidate lives in the affine subspace: baseline + α·p, where p has zero sum.
    p = best_f - baseline
    baseline_fft = np.fft.rfft(baseline, nconv)
    p_fft = np.fft.rfft(p, nconv)

    conv_bb = np.fft.irfft(baseline_fft * baseline_fft, nconv) * dx
    conv_pp = np.fft.irfft(p_fft * p_fft, nconv) * dx
    conv_bp = np.fft.irfft(baseline_fft * p_fft, nconv) * dx

    def max_conv(alpha):
        """Maximum absolute autocorrelation for a given scaling alpha."""
        conv = conv_bb + 2.0 * alpha * conv_bp + (alpha ** 2) * conv_pp
        return float(np.max(np.abs(conv)))

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = -15.0, 15.0
    c = high - phi * (high - low)
    d = low + phi * (high - low)
    fc = max_conv(c)
    fd = max_conv(d)

    ls_iters = 0
    while (time.time() - start_time < TIME_LIMIT - 0.5) and ls_iters < 40:
        if fc < fd:
            high = d
            d = c
            fd = fc
            c = high - phi * (high - low)
            fc = max_conv(c)
        else:
            low = c
            c = d
            fc = fd
            d = low + phi * (high - low)
            fd = max_conv(d)
        ls_iters += 1

    alpha_opt = 0.5 * (low + high)
    best_f = baseline + alpha_opt * p
    best_val = max_conv(alpha_opt)

    # --------------------------------------------------------------
    # Optional hill‑climbing with any leftover time
    # --------------------------------------------------------------
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 0.4:
        step = max(1e-12, np.mean(np.abs(best_f)) * 0.25)
        hill_end = time.time() + remaining
        while time.time() < hill_end:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            delta = rng.normal(scale=step)
            cand = best_f.copy()
            cand[i] += delta
            cand[j] -= delta
            cand -= (np.sum(cand) - target_sum) / n
            conv_cand = conv_full(cand)
            cand_val = float(np.max(np.abs(conv_cand)))
            if cand_val < best_val:
                best_f = cand
                best_val = cand_val
                step *= 1.02
            else:
                step *= 0.995
            if step < 1e-12:
                step = 1e-12

    # --------------------------------------------------------------
    # Final exact‑sum enforcement and safety checks
    # --------------------------------------------------------------
    best_f -= (np.sum(best_f) - target_sum) / n
    if not np.all(np.isfinite(best_f)) or np.any(np.abs(best_f) > 1e5):
        best_f = baseline.copy()

    return best_f


def compute_c3(heights):
    """Compute the C₃ constant for the given discrete function."""
    n = len(heights)
    dx = 0.5 / n
    integral = np.sum(heights) * dx
    if np.abs(integral) < 1e-12:
        raise ValueError("Integral of the function is too close to zero.")
    conv = np.convolve(heights, heights, mode='full') * dx
    return float(np.max(np.abs(conv)) / (integral ** 2))
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_code():
    """Run the C₃ optimization constructor"""
    heights = construct_function()
    c3_value = compute_c3(heights)
    return heights, c3_value, c3_value, len(heights)