# EVOLVE-BLOCK-START

import time
import numpy as np
from scipy.signal import windows

# ----------------------------------------------------------------------
# Helper: projection onto the probability simplex (non‑negative, sum = 1)
# ----------------------------------------------------------------------
def _simplex(v):
    """Project vector v onto the probability simplex."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / np.arange(1, v.size + 1) > 0)[0]
    if rho.size:
        rho = rho[-1]
        theta = cssv[rho] / (rho + 1)
    else:
        theta = cssv[-1] / v.size
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s > 0:
        w /= s
    else:
        w[:] = 1.0 / w.size
    return w


# ----------------------------------------------------------------------
# Fast full convolution (x * x) using real FFT
# ----------------------------------------------------------------------
def _conv(x):
    """Full linear convolution of x with itself using a real FFT."""
    n = x.shape[0]
    # next power‑of‑two ≥ 2n‑1
    m = 1 << ((2 * n - 1).bit_length())
    X = np.fft.rfft(x, m)
    c = np.fft.irfft(X * X, m)[: 2 * n - 1]
    np.maximum(c, 0.0, out=c)             # clamp tiny negatives
    return c


# ----------------------------------------------------------------------
# Objective (numerator of the C1 metric for a simplex‑normalised vector)
# ----------------------------------------------------------------------
def _score(v):
    n = v.shape[0]
    return float(2.0 * n * _conv(v).max())


# ----------------------------------------------------------------------
# Warm‑start from the best known construction, if usable
# ----------------------------------------------------------------------
def _warm():
    try:
        arr = np.asarray(GLOBAL_BEST_CONSTRUCTION, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        arr = np.clip(arr, 0.0, 1000.0)
        if arr.sum() < 0.01:
            return None
        return _simplex(arr)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Analytic shape factories – each returns a probability vector (sum = 1)
# ----------------------------------------------------------------------
def _uniform(n):
    return np.full(n, 1.0 / n, dtype=float)


def _linear_decrease(n):
    """A deterministic decreasing linear shape (good deterministic seed)."""
    idx = np.arange(1, n + 1, dtype=float)          # 1 … n
    raw = (n + 1 - idx)                            # n … 1
    raw /= raw.sum()
    return raw


def _arcsine(n):
    i = np.arange(1, n + 1, dtype=float)
    raw = 1.0 / np.sqrt(i * (n + 1 - i))
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw /= raw.sum()
    return raw


def _gaussian(n):
    x = np.linspace(-1.0, 1.0, n)
    sigma = np.random.uniform(0.12, 0.35)
    raw = np.exp(-0.5 * (x / sigma) ** 2)
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw /= raw.sum()
    return raw


def _random_spike(n):
    base = np.full(n, 1e-6, dtype=float)
    k = max(2, int(0.02 * n))
    pos = np.random.choice(n, size=k, replace=False)
    spikes = np.random.rand(k) * 0.5 + 0.5
    base[pos] = spikes
    base /= base.sum()
    return base


def _random_block(n):
    nb = max(2, int(np.sqrt(n) * 0.9))
    cuts = np.sort(np.random.choice(np.arange(1, n), size=nb - 1, replace=False))
    cuts = np.concatenate(([0], cuts, [n]))
    heights = np.random.rand(nb) + 0.1
    out = np.empty(n, dtype=float)
    for i in range(nb):
        out[cuts[i]:cuts[i + 1]] = heights[i]
    out += 1e-12
    out /= out.sum()
    return out


def _beta(n):
    a = np.exp(np.random.uniform(np.log(0.05), np.log(5.0)))
    b = np.exp(np.random.uniform(np.log(0.05), np.log(5.0)))
    x = np.linspace(1e-8, 1 - 1e-8, n)
    raw = np.power(x, a - 1) * np.power(1 - x, b - 1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw += 1e-14
    raw /= raw.sum()
    return raw


def _dirichlet(n):
    alpha = np.exp(np.random.uniform(np.log(0.05), np.log(2.5)))
    raw = np.random.dirichlet(np.full(n, alpha))
    raw = np.maximum(raw, 1e-12)
    raw /= raw.sum()
    return raw


def _dpss(n):
    try:
        v = windows.dpss(n, NW=2.5, sym=True, norm=None)
    except Exception:
        return _uniform(n)
    v = np.clip(v, 0.0, None)
    v = np.maximum(v, 1e-12)
    v /= v.sum()
    return v


def _sinusoid(n):
    i = np.arange(n, dtype=float)
    raw = np.sin(np.pi * (i + 0.5) / n)
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw /= raw.sum()
    return raw


def _alternating_high_low(n):
    half = n // 2
    high = np.random.rand(half) * 0.9 + 0.1
    low = np.random.rand(n - half) * 0.1
    out = np.empty(n, dtype=float)
    out[0::2] = high[: ((n + 1) // 2)]
    out[1::2] = low[: (n // 2)]
    out += 1e-12
    out /= out.sum()
    return out


def _tukey(n, alpha=0.5):
    try:
        raw = windows.tukey(n, alpha=alpha)
    except Exception:
        raw = np.hanning(n)
    raw = np.maximum(raw, 0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _triangle(n):
    i = np.arange(n, dtype=float)
    c = (n - 1) / 2.0
    raw = 1.0 - np.abs(i - c) / (c + 1.0)
    raw = np.maximum(raw, 0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _kaiser(n):
    beta = np.random.uniform(0.5, 14.0)
    raw = windows.kaiser(n, beta)
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw = np.maximum(raw, 1e-12)
    raw /= raw.sum()
    return raw


def _edge_power(n):
    a = np.random.uniform(0.5, 2.0)
    i = np.arange(1, n + 1, dtype=float)
    raw = (i * (n + 1 - i)) ** (-a)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _double_edge_spike(n):
    out = np.full(n, 1e-9, dtype=float)
    k = max(2, int(0.015 * n))
    left = np.random.randint(k)
    right = n - 1 - np.random.randint(k)
    out[left] = np.random.rand() * 0.5 + 0.5
    out[right] = np.random.rand() * 0.5 + 0.5
    out /= out.sum()
    return out


def _random_sidonic(n):
    """Construct a random Sidon‑type set (unique pairwise sums)."""
    max_trials = n * 4
    used = set()
    pos = []
    for _ in range(max_trials):
        cand = np.random.randint(0, n)
        conflict = False
        for p in pos:
            if cand + p in used:
                conflict = True
                break
        if conflict or cand + cand in used:
            continue
        for p in pos:
            used.add(cand + p)
            used.add(p + cand)
        used.add(cand + cand)
        pos.append(cand)
        if len(pos) >= int(np.sqrt(2 * n)):
            break
    if not pos:
        return _uniform(n)
    v = np.zeros(n, dtype=float)
    v[pos] = 1.0 / len(pos)
    return v


def _sidon_factory(n):
    """Greedy construction of a Sidon‑type set (unique pairwise sums)."""
    best_set = []
    best_len = 0
    for _ in range(5):
        used = set()
        pos = []
        order = np.random.permutation(n)
        for cand in order:
            conflict = False
            for p in pos:
                if cand + p in used or cand + cand in used:
                    conflict = True
                    break
            if conflict:
                continue
            for p in pos:
                used.add(cand + p)
                used.add(p + cand)
            used.add(cand + cand)
            pos.append(cand)
        if len(pos) > best_len:
            best_set = pos
            best_len = len(pos)
    if best_len == 0:
        return _uniform(n)
    v = np.zeros(n, dtype=float)
    v[best_set] = 1.0 / best_len
    return v


def _exponential_decay(n):
    lam = np.random.uniform(0.001, 0.02)
    idx = np.arange(n, dtype=float)
    raw = np.exp(-lam * idx)
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw /= raw.sum()
    return raw


def _ramp(n):
    raw = np.arange(1, n + 1, dtype=float)
    raw = raw / raw.sum()
    return raw


def _cosine_taper(n):
    raw = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw /= raw.sum()
    return raw


def _hann(n):
    raw = windows.hann(n)
    raw = np.maximum(raw, 0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _hamming(n):
    raw = windows.hamming(n)
    raw = np.maximum(raw, 0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _blackman(n):
    raw = windows.blackman(n)
    raw = np.maximum(raw, 0.0) + 1e-12
    raw /= raw.sum()
    return raw


def _power_law(n):
    alpha = np.random.uniform(0.1, 2.0)
    i = np.arange(1, n + 1, dtype=float)
    raw = i ** (-alpha)
    raw = np.clip(raw, 0.0, None) + 1e-12
    raw /= raw.sum()
    return raw


# ----------------------------------------------------------------------
# Small random perturbations & dedicated moves
# ----------------------------------------------------------------------
def _jitter(v, max_frac=0.025):
    n = v.shape[0]
    i, j = np.random.choice(n, 2, replace=False)
    if v[i] <= 0.0:
        return v
    delta = v[i] * np.random.rand() * max_frac
    v[i] -= delta
    v[j] += delta
    return _simplex(v)


def _break_peak(v, conv, n, frac=0.97):
    k = int(np.argmax(conv))
    i_min = max(0, k - (n - 1))
    i_max = min(n - 1, k)
    idx = np.arange(i_min, i_max + 1, dtype=int)
    contrib = v[idx] * v[k - idx]
    i_star = idx[np.argmax(contrib)]
    j_star = k - i_star
    donor = i_star if v[i_star] >= v[j_star] else j_star
    if v[donor] <= 1e-15:
        return v
    recs = np.setdiff1d(np.arange(n), np.array([i_star, j_star]), assume_unique=True)
    if recs.size == 0:
        return v
    rec = np.random.choice(recs)
    delta = v[donor] * np.random.rand() * frac
    v[donor] -= delta
    v[rec] += delta
    return _simplex(v)


def _break_two(v, conv, n, frac=0.94):
    k = int(np.argmax(conv))
    i_min = max(0, k - (n - 1))
    i_max = min(n - 1, k)
    idx = np.arange(i_min, i_max + 1, dtype=int)
    if idx.size < 2:
        return v
    contrib = v[idx] * v[k - idx]
    order = np.argsort(-contrib)[:2]
    i1, i2 = idx[order[0]], idx[order[1]]
    donors = {i1, i2, k - i1, k - i2}
    recs = np.setdiff1d(np.arange(n), np.array(list(donors)), assume_unique=True)
    if recs.size == 0:
        return v
    for donor in (i1, i2):
        if v[donor] <= 1e-15:
            continue
        rec = np.random.choice(recs)
        delta = v[donor] * np.random.rand() * frac
        v[donor] -= delta
        v[rec] += delta
    return _simplex(v)


def _lowpass(v, cutoff=0.22):
    n = v.shape[0]
    X = np.fft.rfft(v)
    cutoff_idx = int(cutoff * (X.shape[0] - 1))
    X[cutoff_idx + 1:] = 0.0
    v_lp = np.fft.irfft(X, n=n)
    v_lp = np.maximum(v_lp, 0.0)
    return _simplex(v_lp)


def _center_mass(v, frac=0.07):
    n = v.shape[0]
    if n < 3:
        return v
    dl = v[0] * np.random.rand() * frac
    dr = v[-1] * np.random.rand() * frac
    v[0] -= dl
    v[-1] -= dr
    centre = n // 2
    share = (dl + dr) / 2.0
    radius = max(1, int(0.02 * n))
    idx = np.arange(max(0, centre - radius), min(n, centre + radius + 1))
    v[idx] += share / idx.size
    return _simplex(v)


def _flatten(v, frac=0.2):
    n = v.shape[0]
    i = int(np.argmax(v))
    delta = v[i] * np.random.rand() * frac
    v[i] -= delta
    if n > 1:
        v[np.arange(n) != i] += delta / (n - 1)
    return _simplex(v)


def _entropy(v, beta=0.55):
    v = np.power(v, beta)
    return _simplex(v)


def _rotate(v, max_shift=10):
    n = v.shape[0]
    if max_shift <= 0:
        return v
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return v
    v_rot = np.roll(v, shift)
    return _simplex(v_rot)


def _symmetrize(v):
    return _simplex((v + v[::-1]) * 0.5)


def _smooth(v, radius_frac=0.01):
    n = v.shape[0]
    r = max(1, int(radius_frac * n))
    kernel = np.ones(2 * r + 1, dtype=float)
    kernel /= kernel.sum()
    v_s = np.convolve(v, kernel, mode='same')
    v_s = np.maximum(v_s, 0.0)
    return _simplex(v_s)


def _swap(v):
    n = v.shape[0]
    i, j = np.random.choice(n, 2, replace=False)
    v[i], v[j] = v[j], v[i]
    return _simplex(v)


# ----------------------------------------------------------------------
# Gradient‑based refinements
# ----------------------------------------------------------------------
def _grad_soft(v, alpha=30.0, step=0.01):
    """Soft‑max surrogate gradient for the max‑convolution entry."""
    n = v.shape[0]
    conv = _conv(v)
    m = conv.max()
    w = np.exp(alpha * (conv - m))
    w_sum = w.sum()
    if w_sum == 0.0:
        w = np.full_like(w, 1.0 / w.size)
    else:
        w /= w_sum
    corr = np.correlate(w, v, mode='full')
    grad = 2.0 * n * corr[(n - 1) - np.arange(n)]
    v_new = v - step * grad
    return _simplex(v_new)


def _grad_max(v, step=0.015):
    """Exact gradient w.r.t. the currently dominant convolution entry."""
    n = v.shape[0]
    conv = _conv(v)
    k = int(np.argmax(conv))
    i_min = max(0, k - (n - 1))
    i_max = min(n - 1, k)
    grad = np.zeros_like(v)
    idx = np.arange(i_min, i_max + 1, dtype=int)
    grad[idx] = 2.0 * v[k - idx]
    v_new = v - step * grad
    return _simplex(v_new)


# ----------------------------------------------------------------------
# Differential Evolution utilities
# ----------------------------------------------------------------------
def _de_mutate(pop, idx, F=0.5):
    sz = len(pop)
    pool = list(range(sz))
    pool.remove(idx)
    a, b, c = np.random.choice(pool, 3, replace=False)
    donor = pop[a] + F * (pop[b] - pop[c])
    donor = np.maximum(donor, 0.0)
    return donor


def _de_crossover(target, donor, CR=0.9):
    n = target.shape[0]
    mask = np.random.rand(n) < CR
    if not np.any(mask):
        mask[np.random.randint(n)] = True
    trial = np.where(mask, donor, target)
    trial = np.maximum(trial, 0.0)
    return trial


# ----------------------------------------------------------------------
# Main constructor – hybrid DE + aggressive local moves + polishing
# ----------------------------------------------------------------------
def propose_candidate(seed=42, budget_s=1000, **_):
    np.random.seed(seed)
    start = time.time()
    deadline = start + budget_s - 0.5      # safety margin

    # ----- initialise -------------------------------------------------
    warm = _warm()
    if warm is not None:
        n = warm.shape[0]
    else:
        n = 2048                         # high‑resolution grid

    factories = [
        _uniform, _linear_decrease, _arcsine, _gaussian, _random_spike,
        _random_block, _beta, _dirichlet, _dpss,
        _sinusoid, _alternating_high_low, _tukey,
        _triangle, _kaiser, _edge_power, _double_edge_spike,
        _random_sidonic, _sidon_factory, _exponential_decay,
        _ramp, _cosine_taper, _hann, _hamming, _blackman,
        _power_law
    ]

    popsize = 96
    pop = []
    if warm is not None:
        pop.append(warm.copy())
    # deterministic linear‑decrease seed (very strong baseline)
    pop.append(_linear_decrease(n))
    while len(pop) < popsize:
        f1 = np.random.choice(factories)
        cand = f1(n)
        # occasional blend of two factories for extra diversity
        if np.random.rand() < 0.25:
            f2 = np.random.choice(factories)
            cand = 0.5 * cand + 0.5 * f2(n)
        cand = _simplex(cand)
        pop.append(cand)
    pop = np.array(pop, dtype=float)
    pop_score = np.array([_score(v) for v in pop], dtype=float)

    best_idx = int(np.argmin(pop_score))
    best = pop[best_idx].copy()
    best_score = pop_score[best_idx]

    # ----- Phase 1 – Differential Evolution (≈ 55 % of budget) ---------
    total_time = deadline - start
    phase1_end = start + 0.55 * total_time
    F = 0.8
    CR = 0.9

    while time.time() < phase1_end:
        # mutation factor slides between 0.4 and 1.0
        F = 0.4 + 0.6 * np.random.rand()
        for i in range(popsize):
            if time.time() >= phase1_end:
                break
            donor = _de_mutate(pop, i, F=F)
            trial = _de_crossover(pop[i], donor, CR=CR)

            # 10 % chance to add a tiny jitter – maintains diversity
            if np.random.rand() < 0.10:
                trial = _jitter(trial, max_frac=0.03)

            trial = _simplex(trial)
            sc = _score(trial)

            if sc < pop_score[i]:
                pop[i] = trial
                pop_score[i] = sc
                if sc < best_score:
                    best = trial.copy()
                    best_score = sc

        # random injection of fresh individuals (15 % probability)
        if np.random.rand() < 0.15:
            worst = int(np.argmax(pop_score))
            cand = np.random.choice(factories)(n)
            if np.random.rand() < 0.20:
                f2 = np.random.choice(factories)
                cand = 0.5 * cand + 0.5 * f2(n)
            cand = _simplex(cand)
            cand_sc = _score(cand)
            if cand_sc < pop_score[worst]:
                pop[worst] = cand
                pop_score[worst] = cand_sc
                if cand_sc < best_score:
                    best = cand.copy()
                    best_score = cand_sc

    # ----- Phase 2 – Aggressive local refinement (≈ 35 % of budget) ---
    phase2_end = start + 0.90 * total_time
    cur = best.copy()
    cur_score = best_score
    phase2_start = time.time()

    while time.time() < phase2_end:
        conv = _conv(cur)

        # linear annealing temperature for Metropolis acceptance
        temp = max(0.001, (phase2_end - time.time()) / (phase2_end - phase2_start))

        prog = (time.time() - start) / total_time
        # gradually shrink move aggressiveness
        frac = max(0.94 - 0.6 * prog, 0.25)

        move_type = np.random.rand()

        if move_type < 0.12:                               # break dominant peak
            cand = _break_peak(cur.copy(), conv, n, frac=frac)
        elif move_type < 0.22:                             # break two strong contributors
            cand = _break_two(cur.copy(), conv, n, frac=frac * 0.9)
        elif move_type < 0.33:                             # jitter
            cand = _jitter(cur.copy(), max_frac=0.045)
        elif move_type < 0.44:                             # rotate
            cand = _rotate(cur.copy(), max_shift=int(0.12 * n))
        elif move_type < 0.55:                             # low‑pass filter
            cand = _lowpass(cur.copy(), cutoff=0.20)
        elif move_type < 0.66:                             # centre‑mass shift
            cand = _center_mass(cur.copy(), frac=0.10)
        elif move_type < 0.74:                             # flatten a spike
            cand = _flatten(cur.copy(), frac=0.30)
        elif move_type < 0.82:                             # entropy‑style flattening
            beta = np.random.uniform(0.45, 0.65)
            cand = _entropy(cur.copy(), beta=beta)
        elif move_type < 0.90:                             # smooth gently
            cand = _smooth(cur.copy(), radius_frac=0.011)
        else:                                              # force symmetry
            cand = _symmetrize(cur.copy())

        cand_sc = _score(cand)

        # accept if better, otherwise Metropolis probability
        if cand_sc < cur_score or np.random.rand() < np.exp((cur_score - cand_sc) / (temp + 1e-12)):
            cur, cur_score = cand, cand_sc
            if cand_sc < best_score:
                best, best_score = cand, cand_sc

    # ----- Phase 3 – Soft‑max gradient descent polishing (remaining time) -
    while time.time() < deadline:
        prog = (time.time() - start) / total_time
        # anneal temperature – starts low to focus on the current peak,
        # climbs slowly to allow exploration of flatter regions.
        alpha = 5.0 + 95.0 * prog
        # shrink step size as we converge
        step = max(0.00035, 0.017 * (1.0 - prog))

        cand = _grad_soft(best.copy(), alpha=alpha, step=step)
        cand_sc = _score(cand)

        if cand_sc < best_score:
            best, best_score = cand, cand_sc
        else:
            # occasional tiny jitter when stuck
            if np.random.rand() < 0.12:
                cand = _jitter(best.copy(), max_frac=0.012)
                cand_sc = _score(cand)
                if cand_sc < best_score:
                    best, best_score = cand, cand_sc

    # ----- Final deterministic polishing ---------------------------------
    # enforce symmetry – known to never worsen the objective
    sym = _symmetrize(best)
    sym_sc = _score(sym)
    if sym_sc < best_score:
        best, best_score = sym, sym_sc

    # a thorough sweep of exact gradient descent on the currently dominant lag
    for _ in range(40000):
        cand = _grad_max(best.copy(), step=0.0032)
        cand_sc = _score(cand)
        if cand_sc < best_score:
            best, best_score = cand, cand_sc
        else:
            break

    return best.tolist()
# EVOLVE-BLOCK-END


def evaluate_sequence(sequence):
    """
    C1 verifier helper, matching evaluator logic.
    Returns np.inf if the input is invalid.
    """
    if not isinstance(sequence, list):
        return np.inf
    if not sequence:
        return np.inf

    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf

    sequence = [float(x) for x in sequence]
    sequence = [max(0.0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))


def run_code():
    """Run the C1 constructor and return (solution, self-reported score)."""
    heights = propose_candidate(seed=42, budget_s=1000)
    c1_value = evaluate_sequence(heights)
    return heights, c1_value
