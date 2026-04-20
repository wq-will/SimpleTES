# EVOLVE-BLOCK-START

import numpy as np
import random
import time
from scipy.optimize import minimize, linprog, differential_evolution
from sklearn.cluster import KMeans

def _baseline_layout():
    """Deterministic 5×4 grid with filler circles (32 circles)."""
    r_large = 0.10
    cols, rows = 5, 4
    spacing_x = 2.0 * r_large
    spacing_y = (1.0 - 2.0 * r_large) / (rows - 1)

    # large circles (20)
    large = []
    for i in range(cols):
        x = r_large + i * spacing_x
        for j in range(rows):
            y = r_large + j * spacing_y
            large.append((x, y))
    large = np.array(large)  # (20,2)

    # filler circles (12)
    r_filler = np.sqrt(r_large**2 + (spacing_y / 2.0)**2) - r_large
    filler = []
    for i in range(cols - 1):
        x = r_large + (i + 0.5) * spacing_x
        for j in range(rows - 1):
            y = r_large + (j + 0.5) * spacing_y
            filler.append((x, y))
    filler = np.array(filler)  # (12,2)

    radii_large = np.full(large.shape[0], r_large)
    radii_filler = np.full(filler.shape[0], r_filler)

    circles = np.vstack([
        np.column_stack([large, radii_large]),
        np.column_stack([filler, radii_filler])
    ])
    return circles  # shape (32,3)

def _is_feasible(circles, eps=1e-8):
    """Return True iff all packing constraints are satisfied."""
    if circles.shape != (32, 3):
        return False
    if np.isnan(circles).any():
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -eps):
        return False
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps):
        return False
    if np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    sum_r = r[:, None] + r[None, :]
    iu = np.triu_indices(len(x), k=1)
    if np.any(d[iu] < sum_r[iu] - eps):
        return False
    return True

def _repair(circles):
    """Uniformly scale radii downwards until all constraints are satisfied."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    with np.errstate(divide='ignore', invalid='ignore'):
        fx_low = np.where(r > 0, x / r, np.inf)
        fx_high = np.where(r > 0, (1 - x) / r, np.inf)
        fy_low = np.where(r > 0, y / r, np.inf)
        fy_high = np.where(r > 0, (1 - y) / r, np.inf)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    sum_r = r[:, None] + r[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        fpair = np.where(sum_r > 0, d / sum_r, np.inf)

    iu = np.triu_indices(len(x), k=1)
    min_pair = np.min(fpair[iu]) if fpair[iu].size > 0 else np.inf

    factor = min(1.0,
                 fx_low.min(),
                 fx_high.min(),
                 fy_low.min(),
                 fy_high.min(),
                 min_pair)

    r_new = r * factor
    return np.column_stack([x, y, r_new])

def _lp_refine(x, y):
    """Given fixed centres, solve a linear programme to maximise radii."""
    n = len(x)
    if n == 0:
        return None
    ub = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    bounds = [(0.0, float(ub_i)) for ub_i in ub]

    pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pair_idx)
    if m == 0:
        return np.column_stack([x, y, np.zeros(n)])

    A = np.zeros((m, n))
    b = np.zeros(m)
    for k, (i, j) in enumerate(pair_idx):
        A[k, i] = 1.0
        A[k, j] = 1.0
        b[k] = np.hypot(x[i] - x[j], y[i] - y[j])

    c = -np.ones(n)                     # maximise sum(r) -> minimise -sum(r)

    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except Exception:
        return None

    if not res.success:
        return None

    r_opt = res.x
    return np.column_stack([x, y, r_opt])

def _constraints_fun(v):
    """Inequality constraints g(v) >= 0 for SLSQP."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]

    c = np.concatenate([
        x - r,            # x >= r
        1.0 - x - r,      # x <= 1 - r
        y - r,            # y >= r
        1.0 - y - r,      # y <= 1 - r
        r                  # r >= 0
    ])

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    sum_r = r[:, None] + r[None, :]
    iu = np.triu_indices(len(x), k=1)
    pair = d[iu] - sum_r[iu]

    return np.concatenate([c, pair])

def _objective(v):
    """Negative sum of radii (minimise)."""
    return -np.sum(v[2::3])

def _jitter_start(baseline, rng, n, max_shift=0.07, max_r_shift=0.025):
    """Perturb a baseline layout while keeping feasibility by clipping."""
    pert = baseline.copy()
    pert[:, 0] += rng.uniform(-max_shift, max_shift, size=n)
    pert[:, 1] += rng.uniform(-max_shift, max_shift, size=n)
    pert[:, 2] += rng.uniform(-max_r_shift, max_r_shift, size=n)
    pert[:, 2] = np.clip(pert[:, 2], 0.0, None)
    pert[:, 0] = np.clip(pert[:, 0], pert[:, 2], 1.0 - pert[:, 2])
    pert[:, 1] = np.clip(pert[:, 1], pert[:, 2], 1.0 - pert[:, 2])
    return pert.flatten()

def _random_start(rng, n):
    """Generate a random feasible start with modest radii."""
    r = rng.uniform(0.02, 0.07, size=n)
    x = rng.uniform(r, 1.0 - r)
    y = rng.uniform(r, 1.0 - r)
    return np.column_stack([x, y, r]).flatten()

def construct_circles():
    """Return a feasible packing of 32 circles with a large total radius."""
    n = 32
    start_time = time.time()
    max_time = 525.0   # overall budget (leaves a small margin)

    baseline = _baseline_layout()
    best_circles = baseline
    best_sum = float(np.sum(baseline[:, 2]))

    # -----------------------------------------------------------------
    # Seed collection
    # -----------------------------------------------------------------
    seed_vectors = []  # flattened (3*n,) arrays
    seed_sums = []     # corresponding sum of radii

    def _add_seed(arr):
        """Accept a (n,3) array or a flat (3*n,) vector."""
        if arr is None:
            return
        if isinstance(arr, np.ndarray):
            if arr.shape == (n, 3):
                vec = arr.flatten()
                s = float(np.sum(arr[:, 2]))
            elif arr.shape == (3 * n,):
                vec = arr
                s = float(np.sum(vec[2::3]))
            else:
                return
            seed_vectors.append(vec)
            seed_sums.append(s)

    # Warm‑start from GLOBAL_BEST_CONSTRUCTION if present
    gb = globals().get('GLOBAL_BEST_CONSTRUCTION', None)
    if isinstance(gb, np.ndarray) and gb.shape == (n, 3):
        _add_seed(gb)

    _add_seed(baseline)

    rng = np.random.default_rng()

    # Jittered variants of the baseline layout
    for _ in range(1000):
        if time.time() - start_time > max_time * 0.5:
            break
        _add_seed(_jitter_start(baseline, rng, n,
                                max_shift=0.07, max_r_shift=0.025))

    # Random feasible seeds
    for _ in range(1000):
        if time.time() - start_time > max_time * 0.5:
            break
        _add_seed(_random_start(rng, n))

    # Random centre configurations refined by LP (large pool)
    for _ in range(25000):
        if time.time() - start_time > max_time * 0.65:
            break
        x_rand = rng.uniform(0.0, 1.0, size=n)
        y_rand = rng.uniform(0.0, 1.0, size=n)
        lp_cir = _lp_refine(x_rand, y_rand)
        if lp_cir is not None:
            _add_seed(lp_cir)

    # Farthest‑point‑sampling seeds
    for _ in range(2500):
        if time.time() - start_time > max_time * 0.65:
            break
        centres = np.empty((n, 2))
        centres[0] = rng.uniform(0.0, 1.0, size=2)
        for i in range(1, n):
            cand = rng.uniform(0.0, 1.0, size=(300, 2))
            d = np.linalg.norm(cand[:, None, :] - centres[:i][None, :, :], axis=2)
            min_dist = d.min(axis=1)
            best_idx = np.argmax(min_dist)
            centres[i] = cand[best_idx]
        fp_cir = _lp_refine(centres[:, 0], centres[:, 1])
        if fp_cir is not None:
            _add_seed(fp_cir)

    # K‑means based seeds
    for _ in range(600):
        if time.time() - start_time > max_time * 0.65:
            break
        pts = rng.uniform(0.0, 1.0, size=(5000, 2))
        try:
            km = KMeans(n_clusters=n, init='k-means++', n_init=3,
                        max_iter=300,
                        random_state=int(rng.integers(0, 1 << 30)))
            km.fit(pts)
            centres = km.cluster_centers_
        except Exception:
            continue
        km_cir = _lp_refine(centres[:, 0], centres[:, 1])
        if km_cir is not None:
            _add_seed(km_cir)

    # Crossover seeds: combine positions from top seeds
    if len(seed_vectors) >= 2:
        top_k = min(300, len(seed_vectors))
        top_indices = np.argsort(seed_sums)[::-1][:top_k]
        top_vectors = [seed_vectors[i] for i in top_indices]
        for _ in range(600):
            if time.time() - start_time > max_time * 0.65:
                break
            p1, p2 = random.sample(top_vectors, 2)
            x1 = p1[0::3]
            y1 = p1[1::3]
            x2 = p2[0::3]
            y2 = p2[1::3]
            mask = rng.integers(0, 2, size=n)
            child_x = np.where(mask == 0, x1, x2)
            child_y = np.where(mask == 0, y1, y2)
            lp_cir = _lp_refine(child_x, child_y)
            if lp_cir is not None:
                _add_seed(lp_cir)

    # -----------------------------------------------------------------
    # Keep only the top K seeds for the heavy SLSQP phase
    # -----------------------------------------------------------------
    max_seeds = 1500
    if len(seed_vectors) > max_seeds:
        idx = np.argsort(seed_sums)[::-1][:max_seeds]
        seed_vectors = [seed_vectors[i] for i in idx]
        seed_sums = [seed_sums[i] for i in idx]

    # Shuffle order to avoid bias
    combined = list(zip(seed_vectors, seed_sums))
    random.shuffle(combined)
    if combined:
        seed_vectors, seed_sums = zip(*combined)
        seed_vectors = list(seed_vectors)
        seed_sums = list(seed_sums)
    else:
        seed_vectors, seed_sums = [], []

    # -----------------------------------------------------------------
    # SLSQP optimisation from each seed
    # -----------------------------------------------------------------
    constraints = [{'type': 'ineq', 'fun': _constraints_fun}]
    bounds = [(0.0, 1.0) if i % 3 != 2 else (0.0, 0.5) for i in range(3 * n)]
    slsqp_opts = {'maxiter': 3000, 'ftol': 1e-9, 'disp': False}

    for v0 in seed_vectors:
        if time.time() - start_time > max_time * 0.75:
            break
        try:
            res = minimize(_objective, v0, method='SLSQP',
                           constraints=constraints, bounds=bounds,
                           options=slsqp_opts)
        except Exception:
            continue
        if not res.success:
            continue
        circles_opt = np.column_stack([res.x[0::3], res.x[1::3], res.x[2::3]])
        if not _is_feasible(circles_opt):
            circles_opt = _repair(circles_opt)
            if not _is_feasible(circles_opt):
                continue
        sum_opt = float(np.sum(circles_opt[:, 2]))
        if sum_opt > best_sum + 1e-9:
            best_sum = sum_opt
            best_circles = circles_opt

    # -----------------------------------------------------------------
    # Local jitter + LP refinement loop
    # -----------------------------------------------------------------
    step = 0.02
    while time.time() - start_time < max_time - 30:
        pert = best_circles.copy()
        pert[:, 0] += rng.uniform(-step, step, size=n)
        pert[:, 1] += rng.uniform(-step, step, size=n)
        pert[:, 0] = np.clip(pert[:, 0], 0.0, 1.0)
        pert[:, 1] = np.clip(pert[:, 1], 0.0, 1.0)
        lp_cir = _lp_refine(pert[:, 0], pert[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-9:
            best_sum = sum_lp
            best_circles = lp_cir
            step = max(step * 0.9, 0.001)
        else:
            step = min(step * 1.02, 0.05)

    # -----------------------------------------------------------------
    # Single‑circle (and occasional two‑circle) relocation search
    # -----------------------------------------------------------------
    for _ in range(8000):
        if time.time() - start_time > max_time:
            break
        if rng.random() < 0.3:
            # relocate two circles simultaneously
            k1, k2 = rng.choice(n, size=2, replace=False)
            new_pos = rng.uniform(0.0, 1.0, size=(2, 2))
            new_pos = np.clip(new_pos, 0.0, 1.0)
            pert = best_circles.copy()
            pert[k1, 0], pert[k1, 1] = new_pos[0]
            pert[k2, 0], pert[k2, 1] = new_pos[1]
        else:
            # relocate a single circle
            k = rng.integers(0, n)
            new_x, new_y = rng.uniform(0.0, 1.0, size=2)
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            pert = best_circles.copy()
            pert[k, 0], pert[k, 1] = new_x, new_y
        lp_cir = _lp_refine(pert[:, 0], pert[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-9:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Final intensive SLSQP refinement
    # -----------------------------------------------------------------
    try:
        res = minimize(_objective, best_circles.flatten(),
                       method='SLSQP',
                       constraints=constraints, bounds=bounds,
                       options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False})
        if res.success:
            circles_opt = np.column_stack([res.x[0::3], res.x[1::3], res.x[2::3]])
            if _is_feasible(circles_opt):
                sum_opt = float(np.sum(circles_opt[:, 2]))
                if sum_opt > best_sum:
                    best_sum = sum_opt
                    best_circles = circles_opt
    except Exception:
        pass

    # -----------------------------------------------------------------
    # Final LP refinement of radii for the best positions
    # -----------------------------------------------------------------
    try:
        lp_cir = _lp_refine(best_circles[:, 0], best_circles[:, 1])
        if lp_cir is not None and _is_feasible(lp_cir):
            sum_lp = float(np.sum(lp_cir[:, 2]))
            if sum_lp > best_sum:
                best_sum = sum_lp
                best_circles = lp_cir
    except Exception:
        pass

    # -----------------------------------------------------------------
    # Additional small jitter + SLSQP passes
    # -----------------------------------------------------------------
    for _ in range(20):
        if time.time() - start_time > max_time:
            break
        pert = best_circles.copy()
        pert[:, 0] += rng.uniform(-0.005, 0.005, size=n)
        pert[:, 1] += rng.uniform(-0.005, 0.005, size=n)
        pert[:, 0] = np.clip(pert[:, 0], 0.0, 1.0)
        pert[:, 1] = np.clip(pert[:, 1], 0.0, 1.0)
        try:
            res = minimize(_objective, pert.flatten(),
                           method='SLSQP',
                           constraints=constraints, bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
            if res.success:
                circles_opt = np.column_stack([res.x[0::3], res.x[1::3], res.x[2::3]])
                if _is_feasible(circles_opt):
                    sum_opt = float(np.sum(circles_opt[:, 2]))
                    if sum_opt > best_sum:
                        best_sum = sum_opt
                        best_circles = circles_opt
        except Exception:
            continue

    # -----------------------------------------------------------------
    # Fine‑grained Gaussian jitter + LP refinement (tiny step)
    # -----------------------------------------------------------------
    while time.time() - start_time < max_time - 1:
        pert_xy = best_circles[:, :2] + rng.normal(scale=0.001, size=(n, 2))
        pert_xy = np.clip(pert_xy, 0.0, 1.0)
        lp_cir = _lp_refine(pert_xy[:, 0], pert_xy[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-12:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Position‑only optimisation via L‑BFGS‑B (optimise centres, radii via LP)
    # -----------------------------------------------------------------
    if time.time() - start_time < max_time - 5:
        def _pos_objective(pos):
            x = pos[0::2]
            y = pos[1::2]
            lp_c = _lp_refine(x, y)
            if lp_c is None:
                return 1e6
            return -np.sum(lp_c[:, 2])

        bounds_pos = [(0.0, 1.0)] * (2 * n)
        try:
            res_pos = minimize(_pos_objective,
                               best_circles[:, :2].flatten(),
                               method='L-BFGS-B',
                               bounds=bounds_pos,
                               options={'maxiter': 800, 'ftol': 1e-9})
            if res_pos.success:
                lp_c = _lp_refine(res_pos.x[0::2], res_pos.x[1::2])
                if lp_c is not None:
                    sum_lp = float(np.sum(lp_c[:, 2]))
                    if sum_lp > best_sum + 1e-12:
                        best_sum = sum_lp
                        best_circles = lp_c
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Additional small jitter + SLSQP passes (more)
    # -----------------------------------------------------------------
    for _ in range(30):
        if time.time() - start_time > max_time:
            break
        pert = best_circles.copy()
        pert[:, 0] += rng.uniform(-0.005, 0.005, size=n)
        pert[:, 1] += rng.uniform(-0.005, 0.005, size=n)
        pert[:, 0] = np.clip(pert[:, 0], 0.0, 1.0)
        pert[:, 1] = np.clip(pert[:, 1], 0.0, 1.0)
        try:
            res = minimize(_objective, pert.flatten(),
                           method='SLSQP',
                           constraints=constraints, bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
            if res.success:
                circles_opt = np.column_stack([res.x[0::3], res.x[1::3], res.x[2::3]])
                if _is_feasible(circles_opt):
                    sum_opt = float(np.sum(circles_opt[:, 2]))
                    if sum_opt > best_sum:
                        best_sum = sum_opt
                        best_circles = circles_opt
        except Exception:
            continue

    # -----------------------------------------------------------------
    # Fine‑grained Gaussian jitter + LP refinement (again)
    # -----------------------------------------------------------------
    while time.time() - start_time < max_time - 1:
        pert_xy = best_circles[:, :2] + rng.normal(scale=0.001, size=(n, 2))
        pert_xy = np.clip(pert_xy, 0.0, 1.0)
        lp_cir = _lp_refine(pert_xy[:, 0], pert_xy[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-12:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Final random reposition + LP refinement using any remaining time
    # -----------------------------------------------------------------
    while time.time() - start_time < max_time - 1:
        if rng.random() < 0.3:
            # relocate two circles simultaneously
            k1, k2 = rng.choice(n, size=2, replace=False)
            new_pos = rng.uniform(0.0, 1.0, size=(2, 2))
            new_pos = np.clip(new_pos, 0.0, 1.0)
            pert = best_circles.copy()
            pert[k1, 0], pert[k1, 1] = new_pos[0]
            pert[k2, 0], pert[k2, 1] = new_pos[1]
        else:
            # relocate a single circle
            k = rng.integers(0, n)
            new_x, new_y = rng.uniform(0.0, 1.0, size=2)
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            pert = best_circles.copy()
            pert[k, 0], pert[k, 1] = new_x, new_y
        lp_cir = _lp_refine(pert[:, 0], pert[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-9:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Pairwise swap moves (additional refinement)
    # -----------------------------------------------------------------
    while time.time() - start_time < max_time - 1:
        i, j = rng.choice(n, size=2, replace=False)
        pert = best_circles.copy()
        # swap positions of circles i and j
        pert[i, 0], pert[i, 1], pert[j, 0], pert[j, 1] = (
            pert[j, 0], pert[j, 1], pert[i, 0], pert[i, 1]
        )
        lp_cir = _lp_refine(pert[:, 0], pert[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-9:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Differential Evolution global search (if time permits)
    # -----------------------------------------------------------------
    if time.time() - start_time < max_time - 5:
        de_bounds = [(0.0, 1.0)] * (2 * n)

        def _de_obj(v):
            x = v[0::2]
            y = v[1::2]
            lp_c = _lp_refine(x, y)
            if lp_c is None:
                return 1e6
            return -np.sum(lp_c[:, 2])

        try:
            result = differential_evolution(
                _de_obj,
                de_bounds,
                maxiter=30,          # increased iterations
                popsize=12,          # larger population
                polish=False,
                updating='deferred',
                seed=int(rng.integers(0, 1 << 30)),
                disp=False
            )
            if result.success:
                best_x = result.x[0::2]
                best_y = result.x[1::2]
                lp_cir = _lp_refine(best_x, best_y)
                if lp_cir is not None and _is_feasible(lp_cir):
                    sum_lp = float(np.sum(lp_cir[:, 2]))
                    if sum_lp > best_sum:
                        best_sum = sum_lp
                        best_circles = lp_cir
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Final local jitter + LP refinement loop (if any time left)
    # -----------------------------------------------------------------
    step = 0.005
    while time.time() - start_time < max_time - 5:
        pert = best_circles.copy()
        pert[:, 0] += rng.uniform(-step, step, size=n)
        pert[:, 1] += rng.uniform(-step, step, size=n)
        pert[:, 0] = np.clip(pert[:, 0], 0.0, 1.0)
        pert[:, 1] = np.clip(pert[:, 1], 0.0, 1.0)
        lp_cir = _lp_refine(pert[:, 0], pert[:, 1])
        if lp_cir is None:
            continue
        if not _is_feasible(lp_cir):
            lp_cir = _repair(lp_cir)
            if not _is_feasible(lp_cir):
                continue
        sum_lp = float(np.sum(lp_cir[:, 2]))
        if sum_lp > best_sum + 1e-12:
            best_sum = sum_lp
            best_circles = lp_cir

    # -----------------------------------------------------------------
    # Safety fallback
    # -----------------------------------------------------------------
    if not _is_feasible(best_circles):
        best_circles = baseline

    return best_circles
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_code():
    """Run the circle packing constructor for n=32"""
    circles = construct_circles()
    sum_radii = float(np.sum(circles[:, 2]))
    return circles, sum_radii


