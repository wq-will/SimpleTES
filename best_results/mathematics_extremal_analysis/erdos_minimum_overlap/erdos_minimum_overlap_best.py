# EVOLVE-BLOCK-START

import numpy as np
import time

def construct_h():
    """
    Heuristic construction of a step function h on [0,2] discretised to n_fine points.
    The algorithm consists of:
        0) Warm‑start (global best, Paley construction)
        1) Fast stochastic donor‑receiver swaps (exploration)
        2) Adam descent on a smooth‑max surrogate (refinement)
        3) Guided swaps targeting the worst shift (fine‑tuning)
        4) Binary rounding (top‑fraction)
        5) Binary best‑swap optimisation (large zero sample)
        6) Final simulated‑annealing / hill‑climbing polish
    Returns:
        h (np.ndarray), n_fine (int)
    """
    # ------------------------------------------------------------------
    # Basic parameters
    # ------------------------------------------------------------------
    n_fine = 2400                     # number of discretisation points (must be even)
    target_sum = n_fine // 2          # Σ h_i = n_fine/2 (integral = 1)
    eps = 1e-12
    max_time = 1080.0                 # total wall‑clock budget (seconds)
    start_time = time.time()
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Projection onto {0≤x≤1, Σ x = target_sum}
    # ------------------------------------------------------------------
    def _project(v: np.ndarray, s: float) -> np.ndarray:
        """Box‑sum projection via bisection on τ."""
        tau_lo = float(np.min(v) - 1.0)
        tau_hi = float(np.max(v) - 0.0)
        for _ in range(80):
            tau = (tau_lo + tau_hi) / 2.0
            x = np.clip(v - tau, 0.0, 1.0)
            if float(np.sum(x, dtype=np.float64)) > s:
                tau_lo = tau
            else:
                tau_hi = tau
        return np.clip(v - tau_hi, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Full linear correlation of h with (1‑h) via FFT (zero‑padded)
    # ------------------------------------------------------------------
    def _full_corr(h: np.ndarray) -> np.ndarray:
        """Return linear correlation of h with (1‑h); length = 2·n‑1 (real values)."""
        n = h.size
        n_fft = 1 << ((2 * n - 1).bit_length())
        H = np.fft.rfft(h, n=n_fft)
        J = np.fft.rfft((1.0 - h)[::-1], n=n_fft)
        conv = np.fft.irfft(H * J, n=n_fft)[: 2 * n - 1]
        return conv.real

    # ------------------------------------------------------------------
    # Helper: convolution of a weight vector w with h (central n entries)
    # ------------------------------------------------------------------
    def _conv_w_h(w: np.ndarray, hh: np.ndarray) -> np.ndarray:
        """Convolution of w with h; returns the central n entries."""
        n = hh.size
        L = w.size
        n_fft = 1 << ((L + n - 1).bit_length())
        W = np.fft.rfft(w, n=n_fft)
        Hh = np.fft.rfft(hh, n=n_fft)
        conv = np.fft.irfft(W * Hh, n=n_fft)
        start = L - 1
        return conv[start:start + n]

    # ------------------------------------------------------------------
    # Warm‑start: GLOBAL_BEST_CONSTRUCTION → Paley construction
    # ------------------------------------------------------------------
    h = None
    try:
        global GLOBAL_BEST_CONSTRUCTION
        if GLOBAL_BEST_CONSTRUCTION is not None:
            prev_h, _, prev_n = GLOBAL_BEST_CONSTRUCTION[:3]
            prev_h = np.asarray(prev_h, dtype=np.float64).reshape(-1)
            prev_n = int(prev_n)
            if prev_n == n_fine:
                h = prev_h.copy()
            elif prev_n * 2 == n_fine:
                h = np.repeat(prev_h, 2)
    except Exception:
        pass

    if h is None:
        # Paley construction for prime p = 599 (p ≡ 3 (mod 4))
        p = 599
        residues = {(i * i) % p for i in range(p)}   # quadratic residues (incl. 0)
        base = np.zeros(p, dtype=np.float64)
        for idx in residues:
            base[idx] = 1.0
        h = np.concatenate([base, np.zeros(n_fine - p, dtype=np.float64)])

        # Quick search over rotations and complement
        best_h = h.copy()
        best_val = float(np.max(_full_corr(best_h)))
        comp = 1.0 - h
        comp_val = float(np.max(_full_corr(comp)))
        if comp_val < best_val:
            best_h, best_val = comp.copy(), comp_val
        for _ in range(200):
            shift = rng.integers(n_fine)
            rot = np.roll(h, shift)
            val = float(np.max(_full_corr(rot)))
            if val < best_val:
                best_h, best_val = rot.copy(), val
        h = best_h

    # Ensure exact feasibility before optimisation
    h = _project(h, target_sum)

    # ------------------------------------------------------------------
    # Keep track of the best fractional solution
    # ------------------------------------------------------------------
    best_h = h.copy()
    best_val = float(np.max(_full_corr(best_h)))

    # ------------------------------------------------------------------
    # Phase 1 – Fast stochastic donor‑receiver swaps (exploration)
    # ------------------------------------------------------------------
    phase1_deadline = start_time + max_time * 0.05   # ~5 % of budget
    while time.time() < phase1_deadline:
        cur_h = best_h.copy()
        cur_val = best_val
        for _ in range(2000):
            if time.time() > phase1_deadline:
                break
            i, j = rng.choice(n_fine, size=2, replace=False)
            if cur_h[i] <= eps or cur_h[j] >= 1.0 - eps:
                continue
            max_delta = min(cur_h[i], 1.0 - cur_h[j])
            delta = max_delta * rng.random() * 0.5
            if delta < eps:
                continue
            cand = cur_h.copy()
            cand[i] -= delta
            cand[j] += delta
            cand = _project(cand, target_sum)
            cand_val = float(np.max(_full_corr(cand)))
            if cand_val < cur_val - 1e-15:
                cur_h = cand
                cur_val = cand_val
                if cand_val < best_val - 1e-15:
                    best_h = cand.copy()
                    best_val = cand_val

    # ------------------------------------------------------------------
    # Phase 2 – Adam descent on smooth‑max surrogate (refinement)
    # ------------------------------------------------------------------
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    beta_schedule = [
        0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
        40.0, 80.0, 160.0, 320.0, 640.0,
        1280.0, 2560.0, 5120.0, 10240.0,
        20480.0, 40960.0, 81920.0, 163840.0,
        327680.0, 655360.0, 1310720.0
    ]
    steps_per_beta = 8000
    phase2_deadline = start_time + max_time * 0.78   # ~78 % of budget
    h_frac = best_h.copy()
    cur_val = best_val
    m = np.zeros_like(h_frac)
    v = np.zeros_like(h_frac)

    for beta in beta_schedule:
        if time.time() > phase2_deadline:
            break
        m.fill(0.0)
        v.fill(0.0)
        t_iter = 0
        for _ in range(steps_per_beta):
            if time.time() > phase2_deadline:
                break
            t_iter += 1
            O = _full_corr(h_frac)
            O_max = np.max(O)
            exp_term = np.exp(beta * (O - O_max))
            w = exp_term / np.sum(exp_term)
            grad = 1.0 - (_conv_w_h(w, h_frac) + _conv_w_h(w[::-1], h_frac))
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** t_iter)
            v_hat = v / (1.0 - beta2 ** t_iter)
            step = 0.5 / (np.sqrt(v_hat) + eps_adam)
            cand = h_frac - step * m_hat
            cand = _project(cand, target_sum)
            cand_val = float(np.max(_full_corr(cand)))
            if cand_val < cur_val - 1e-15:
                h_frac = cand
                cur_val = cand_val
                if cand_val < best_val - 1e-15:
                    best_h = cand.copy()
                    best_val = cand_val
                continue
            # occasional tiny donor‑receiver tweak
            if rng.random() < 0.006:
                i, j = rng.choice(n_fine, size=2, replace=False)
                if h_frac[i] > eps and h_frac[j] < 1.0 - eps:
                    max_delta = min(h_frac[i], 1.0 - h_frac[j])
                    delta = max_delta * rng.random() * 0.25
                    cand2 = h_frac.copy()
                    cand2[i] -= delta
                    cand2[j] += delta
                    cand2 = _project(cand2, target_sum)
                    cand2_val = float(np.max(_full_corr(cand2)))
                    if cand2_val < cur_val - 1e-15:
                        h_frac = cand2
                        cur_val = cand2_val
                        if cand2_val < best_val - 1e-15:
                            best_h = cand2.copy()
                            best_val = cand2_val

    # ------------------------------------------------------------------
    # Phase 3 – Guided swaps targeting the worst shift (fine‑tuning)
    # ------------------------------------------------------------------
    cur_h = best_h.copy()
    cur_val = best_val
    guided_deadline = start_time + max_time * 0.93   # allocate a bit more before binary phase
    while time.time() < guided_deadline:
        O = _full_corr(cur_h)
        k_max = int(np.argmax(O))
        shift = k_max - (n_fine - 1)

        if shift >= 0:
            start_i = 0
            end_i = n_fine - shift
        else:
            start_i = -shift
            end_i = n_fine
        if start_i >= end_i:
            break
        idx_range = np.arange(start_i, end_i)
        h_i = cur_h[idx_range]
        h_shift = cur_h[idx_range + shift]

        recv_mask = (h_i < 1.0 - eps) & (h_shift < 1.0 - eps)
        recv_idxs = idx_range[recv_mask]
        donor_mask = (h_i > eps) & (h_shift > eps)
        donor_idxs = idx_range[donor_mask]

        if recv_idxs.size == 0 or donor_idxs.size == 0:
            break

        i = int(rng.choice(recv_idxs))
        j = int(rng.choice(donor_idxs))
        if i == j:
            continue

        max_delta = min(1.0 - cur_h[i],
                        1.0 - cur_h[i + shift],
                        cur_h[j],
                        cur_h[j + shift])
        if max_delta <= eps:
            continue
        delta = max_delta * rng.random() * 0.5
        cand = cur_h.copy()
        cand[i] += delta
        cand[i + shift] += delta
        cand[j] -= delta
        cand[j + shift] -= delta
        cand_val = float(np.max(_full_corr(cand)))
        if cand_val < cur_val - 1e-15:
            cur_h = cand
            cur_val = cand_val
            if cand_val < best_val - 1e-15:
                best_h = cand.copy()
                best_val = cand_val

    # Ensure exact feasibility after guided swaps
    best_h = _project(best_h, target_sum)

    # ------------------------------------------------------------------
    # Phase 4 – Binary rounding (top‑fraction)
    # ------------------------------------------------------------------
    binary_h = np.zeros_like(best_h)
    target_sum_int = target_sum
    idx_top = np.argpartition(-best_h, target_sum_int)[:target_sum_int]
    binary_h[idx_top] = 1.0

    # ------------------------------------------------------------------
    # Helper: contribution of a single 1 at index i to O (h with 1‑h)
    # ------------------------------------------------------------------
    def _contrib_one(i: int, h_vec: np.ndarray) -> np.ndarray:
        n_local = h_vec.size
        L_local = 2 * n_local - 1
        contrib = np.zeros(L_local, dtype=np.float64)
        # forward part (i + s)
        fwd_start = -i
        fwd_end = n_local - 1 - i
        s_fwd = np.arange(fwd_start, fwd_end + 1, dtype=np.int64)
        contrib[s_fwd + (n_local - 1)] = h_vec[i + s_fwd]
        # backward part (i - s)
        bwd_start = i - (n_local - 1)
        bwd_end = i
        s_bwd = np.arange(bwd_start, bwd_end + 1, dtype=np.int64)
        contrib[s_bwd + (n_local - 1)] += h_vec[i - s_bwd]
        return contrib

    n = n_fine
    L = 2 * n - 1

    # Initialise correlation and contribution matrix for the binary vector
    O_bin = _full_corr(binary_h)
    cur_val_bin = float(np.max(O_bin))
    best_binary_h = binary_h.copy()
    best_val_bin = cur_val_bin

    M = np.empty((n, L), dtype=np.float64)
    for idx in range(n):
        M[idx] = _contrib_one(idx, binary_h)

    # ------------------------------------------------------------------
    # Phase 5 – Binary best‑swap optimisation (large zero sample)
    # ------------------------------------------------------------------
    binary_deadline = start_time + max_time * 0.97  # allocate most remaining time

    def _binary_optimize(cur_h, cur_O, cur_M, deadline):
        """Best‑swap descent using a limited zero sample."""
        cur_val = float(np.max(cur_O))
        while time.time() < deadline:
            ones_idx = np.flatnonzero(cur_h > 0.5)
            zeros_idx = np.flatnonzero(cur_h < 0.5)
            if ones_idx.size == 0 or zeros_idx.size == 0:
                break

            # Pre‑compute O – M_i for all current ones
            O_minus_M = cur_O[None, :] - cur_M[ones_idx]  # shape (|ones|, L)

            # Sample a (moderately) large subset of zeros
            max_zero_sample = 500
            if zeros_idx.size > max_zero_sample:
                zeros_sample = rng.choice(zeros_idx, size=max_zero_sample, replace=False)
            else:
                zeros_sample = zeros_idx

            best_improve = 0.0          # negative improvement is good
            best_pair = None
            best_new_O = None

            for j in zeros_sample:
                # Candidate O after swapping any current 1 with this zero j
                cand_O = O_minus_M + cur_M[j][None, :]      # shape (|ones|, L)

                # The shift i‑j loses the self‑pair contribution (subtract 1)
                shift_idx = (ones_idx - j) + (n - 1)
                cand_O[np.arange(ones_idx.size), shift_idx] -= 1.0

                # Add the constant “+1” that every 1 contributes to every shift
                cand_O[:, n - 1] += 1.0

                # Determine the worst shift after each possible swap
                cand_max = np.max(cand_O, axis=1)
                i_local = np.argmin(cand_max)          # the i that gives the smallest new max
                cand_val = cand_max[i_local]

                if cand_val < cur_val - 1e-15:
                    improve = cand_val - cur_val        # negative
                    if improve < best_improve:
                        best_improve = improve
                        best_pair = (ones_idx[i_local], j)
                        best_new_O = cand_O[i_local].copy()

            if best_pair is None:
                # No improving swap among the sampled zeros
                break

            # Apply the best improving swap found
            i_swap, j_swap = best_pair
            cur_h[i_swap] = 0.0
            cur_h[j_swap] = 1.0
            cur_O = best_new_O
            cur_val = float(np.max(cur_O))

            # Update contribution rows
            cur_M[i_swap].fill(0.0)                     # i became 0
            cur_M[j_swap] = _contrib_one(j_swap, cur_h) # j became 1

        return cur_h, cur_val, cur_O, cur_M

    binary_h, cur_val_bin, O_bin, M = _binary_optimize(
        binary_h.copy(), O_bin.copy(), M.copy(), binary_deadline
    )
    if cur_val_bin < best_val_bin - 1e-15:
        best_val_bin = cur_val_bin
        best_binary_h = binary_h.copy()

    # ------------------------------------------------------------------
    # Phase 6 – Final simulated‑annealing / hill‑climbing polish
    # ------------------------------------------------------------------
    hill_deadline = start_time + max_time * 0.995
    cur_h = best_binary_h.copy()
    cur_O = _full_corr(cur_h)
    cur_val = float(np.max(cur_O))
    while time.time() < hill_deadline:
        i = rng.integers(n)
        if cur_h[i] == 0.0:
            continue
        j = rng.integers(n)
        if cur_h[j] == 1.0:
            continue
        # incremental update using contributions
        M_i = _contrib_one(i, cur_h)
        M_j = _contrib_one(j, cur_h)
        delta = -M_i + M_j
        delta[n - 1] += 1.0
        shift_idx = (i - j) + (n - 1)
        delta[shift_idx] -= 1.0
        cand_O = cur_O + delta
        cand_val = float(np.max(cand_O))
        if cand_val < cur_val - 1e-15:
            cur_h[i] = 0.0
            cur_h[j] = 1.0
            cur_O = cand_O
            cur_val = cand_val
            if cur_val < best_val_bin - 1e-15:
                best_val_bin = cur_val
                best_binary_h = cur_h.copy()

    # ------------------------------------------------------------------
    # Choose the better of the fractional and binary solutions
    # ------------------------------------------------------------------
    if best_val_bin < best_val - 1e-15:
        return best_binary_h, n_fine
    else:
        return best_h, n_fine
# EVOLVE-BLOCK-END


def run_code():
    """Run the Erdős minimum overlap optimization.
    
    Returns:
        tuple: (h_values, c5_bound, n_points)
            h_values: np.ndarray, shape (n_points,), discretized step function h
            c5_bound: float, max overlap computed from this h_values
            n_points: int, number of bins used to discretize [0, 2]
    """
    h_values, n_points = construct_h()

    n = int(n_points)
    target_sum = n / 2.0

    # Keep post-processing fixed and robust:
    # - cast to float64 (avoid float32 bound spillover)
    # - project to the feasible set {0<=h<=1, sum(h)=n/2}
    h_values = np.asarray(h_values, dtype=np.float64).reshape(-1)
    assert isinstance(n_points, int), TypeError(f"n_points must be an integer, got {type(n_points)}")
    if h_values.shape[0] != n:
        raise ValueError(f"Expected h_values shape ({n},), got {h_values.shape}")

    def _project_box_sum(v: np.ndarray, s: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
        if not np.all(np.isfinite(v)):
            raise ValueError("h_values contain NaN or inf values")
        # Bisection on tau for x = clip(v - tau, lo, hi) such that sum(x)=s.
        tau_lo = float(np.min(v) - hi)
        tau_hi = float(np.max(v) - lo)
        for _ in range(80):
            tau = (tau_lo + tau_hi) / 2.0
            x = np.clip(v - tau, lo, hi)
            if float(np.sum(x, dtype=np.float64)) > s:
                tau_lo = tau
            else:
                tau_hi = tau
        return np.clip(v - tau_hi, lo, hi)

    h_values = _project_box_sum(h_values, target_sum)
    
    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5_bound = np.max(correlation)
    
    return h_values, c5_bound, n_points
