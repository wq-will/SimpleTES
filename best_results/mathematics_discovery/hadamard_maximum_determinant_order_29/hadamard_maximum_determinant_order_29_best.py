# EVOLVE-BLOCK-START

import numpy as np
import time

def construct_hadamard_matrix(n=29):
    rng = np.random.default_rng()
    EPS = 1e-12

    def logabs_det(mat):
        sign, logabs = np.linalg.slogdet(mat)
        return -np.inf if sign == 0 else logabs

    # -----------------------------------------------------------------
    # Seed generators
    def quadratic_residue_matrix(p):
        """Circulant ±1 matrix based on quadratic residues modulo p."""
        qr = {(i * i) % p for i in range(1, p)}
        mat = np.empty((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(p):
                if i == j:
                    mat[i, j] = 1
                else:
                    diff = (i - j) % p
                    mat[i, j] = 1 if diff in qr else -1
        return mat

    def sylvester_hadamard(p):
        """Sylvester Hadamard matrix of size 2^k ≥ p, then cropped."""
        H = np.array([[1]], dtype=np.int8)
        while H.shape[0] < p:
            H = np.block([[H, H], [H, -H]])
        return H[:p, :p]

    def orthogonal_sign(p):
        """Sign matrix derived from a random orthogonal matrix."""
        G = rng.standard_normal((p, p))
        Q, _ = np.linalg.qr(G)
        return np.where(Q >= 0, 1, -1).astype(np.int8)

    # -----------------------------------------------------------------
    # Assemble a pool of seed matrices
    seeds = []

    # 1) Global best construction, if available
    gb = globals().get('GLOBAL_BEST_CONSTRUCTION')
    if isinstance(gb, np.ndarray) and gb.shape == (n, n):
        seeds.append(np.where(gb >= 0, 1, -1).astype(np.int8))

    # 2) Quadratic‑residue circulant matrix
    try:
        seeds.append(quadratic_residue_matrix(n))
    except Exception:
        pass

    # 3) Orthogonal‑sign matrix
    try:
        seeds.append(orthogonal_sign(n))
    except Exception:
        pass

    # 4) Cropped Sylvester Hadamard
    try:
        seeds.append(sylvester_hadamard(n))
    except Exception:
        pass

    # 5) Random seeds for diversity
    for _ in range(6):
        seeds.append(rng.choice([-1, 1], size=(n, n)).astype(np.int8))

    if not seeds:
        seeds.append(rng.choice([-1, 1], size=(n, n)).astype(np.int8))

    # -----------------------------------------------------------------
    # Helper: make a matrix invertible by flipping random entries if needed
    def ensure_invertible(int_mat):
        """Flip random entries until the matrix becomes invertible."""
        A = int_mat.astype(np.float64)
        for _ in range(30):
            sign, _ = np.linalg.slogdet(A)
            if sign != 0:
                return int_mat.copy(), A.copy()
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            int_mat[i, j] = -int_mat[i, j]
            A[i, j] = -A[i, j]
        # fallback: fresh random invertible matrix
        while True:
            int_mat = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
            A = int_mat.astype(np.float64)
            sign, _ = np.linalg.slogdet(A)
            if sign != 0:
                return int_mat.copy(), A.copy()

    # -----------------------------------------------------------------
    # Core search: greedy multi‑flip hill‑climb
    def multi_flip_search(start_int, time_budget):
        """Hill‑climb using multi‑flips per iteration."""
        int_mat, A = ensure_invertible(start_int.copy())
        invA = np.linalg.inv(A)
        cur_log = logabs_det(A)
        best_int = int_mat.copy()
        best_log = cur_log

        start = time.time()
        cooldown = np.zeros((n, n), dtype=np.int16)
        COOLDOWN_VAL = 5

        while time.time() - start < time_budget:
            if COOLDOWN_VAL > 0:
                cooldown = np.maximum(cooldown - 1, 0)

            factor = 1.0 - 2.0 * int_mat * invA.T
            abs_factor = np.abs(factor)
            mask = (abs_factor > (1.0 + EPS)) & (cooldown == 0)

            if np.any(mask):
                cand_idx = np.argwhere(mask)
                scores = abs_factor[mask]
                order = np.argsort(-scores)
                selected_rows = set()
                selected_cols = set()
                flips = []
                max_flips = min(5, len(cand_idx))
                for idx in order:
                    i, j = cand_idx[idx]
                    if i in selected_rows or j in selected_cols:
                        continue
                    flips.append((i, j))
                    selected_rows.add(i)
                    selected_cols.add(j)
                    if len(flips) >= max_flips:
                        break
                if flips:
                    for i, j in flips:
                        int_mat[i, j] = -int_mat[i, j]
                        A[i, j] = -A[i, j]
                        cooldown[i, j] = COOLDOWN_VAL
                    try:
                        invA = np.linalg.inv(A)
                        cur_log = logabs_det(A)
                    except np.linalg.LinAlgError:
                        # revert flips if singular
                        for i, j in flips:
                            int_mat[i, j] = -int_mat[i, j]
                            A[i, j] = -A[i, j]
                            cooldown[i, j] = 0
                        invA = np.linalg.inv(A)
                        cur_log = logabs_det(A)
                    if cur_log > best_log + EPS:
                        best_log = cur_log
                        best_int = int_mat.copy()
                    continue

            # No improving multi‑flip – perform a small random perturbation
            saved_int = int_mat.copy()
            saved_A = A.copy()
            saved_inv = invA.copy()
            saved_log = cur_log
            saved_cool = cooldown.copy()

            num_flips = rng.integers(2, 6)  # flip 2–5 entries
            for _ in range(num_flips):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                int_mat[i, j] = -int_mat[i, j]
                A[i, j] = -A[i, j]
                cooldown[i, j] = COOLDOWN_VAL

            try:
                invA = np.linalg.inv(A)
                cur_log = logabs_det(A)
            except np.linalg.LinAlgError:
                int_mat = saved_int
                A = saved_A
                invA = saved_inv
                cur_log = saved_log
                cooldown = saved_cool
                continue

            if cur_log > best_log + EPS:
                best_log = cur_log
                best_int = int_mat.copy()

        return best_int, best_log

    # -----------------------------------------------------------------
    # Main optimisation loop – allocate total time across stages
    TOTAL_TIME = 340.0  # seconds (reserve a small margin for I/O)
    total_start = time.time()
    global_best = None
    global_best_log = -np.inf

    # Short initial climbs from each seed
    for seed in seeds:
        if time.time() - total_start > TOTAL_TIME - 10.0:
            break
        time_left = TOTAL_TIME - (time.time() - total_start)
        budget = min(3.0, time_left * 0.2)
        mat, logdet = multi_flip_search(seed, budget)
        if logdet > global_best_log:
            global_best_log = logdet
            global_best = mat.copy()

    # Deeper focused searches on the current best
    while True:
        elapsed = time.time() - total_start
        time_left = TOTAL_TIME - elapsed
        if time_left < 8.0:
            break
        if global_best is not None and rng.random() < 0.7:
            seed = global_best
        else:
            seed = seeds[rng.integers(0, len(seeds))]
        chunk = min(8.0, time_left * 0.5)
        mat, logdet = multi_flip_search(seed, chunk)
        if logdet > global_best_log:
            global_best_log = logdet
            global_best = mat.copy()

    # Optional simulated‑annealing refinement if time permits
    remaining = TOTAL_TIME - (time.time() - total_start)
    if global_best is not None and remaining > 5.0:
        sign_mat = global_best.copy()
        A = sign_mat.astype(np.float64)
        try:
            invA = np.linalg.inv(A)
            cur_log = logabs_det(A)
        except np.linalg.LinAlgError:
            invA = np.linalg.inv(A + np.eye(n) * 1e-6)
            cur_log = logabs_det(A)
        best_sign = sign_mat.copy()
        best_log = cur_log

        SA_start = time.time()
        T0 = 1.0
        T_end = 0.001
        while time.time() - SA_start < remaining:
            elapsed = time.time() - SA_start
            progress = elapsed / remaining
            T = T0 * (1.0 - progress) + T_end * progress
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            denom = 1.0 - 2.0 * sign_mat[i, j] * invA[j, i]
            if abs(denom) < EPS:
                continue
            log_factor = np.log(abs(denom))
            accept = False
            if log_factor > 0:
                accept = True
            else:
                if np.exp(log_factor / max(T, 1e-12)) > rng.random():
                    accept = True
            if not accept:
                continue
            delta = -2.0 * sign_mat[i, j]
            # Sherman‑Morrison update
            col_i = invA[:, i].copy()
            row_j = invA[j, :].copy()
            invA = invA - (delta / denom) * np.outer(col_i, row_j)
            sign_mat[i, j] = -sign_mat[i, j]
            A[i, j] = -A[i, j]
            cur_log += log_factor
            if cur_log > best_log + EPS:
                best_log = cur_log
                best_sign = sign_mat.copy()
            # occasional full recompute to curb drift
            if rng.random() < 0.01:
                invA = np.linalg.inv(A)
                cur_log = logabs_det(A)
        if best_log > global_best_log:
            global_best = best_sign
            global_best_log = best_log

    # Final sign‑alignment refinement (may give a tiny extra boost)
    if global_best is not None:
        try:
            inv_best = np.linalg.inv(global_best.astype(np.float64))
            aligned = np.where(inv_best.T >= 0, 1, -1).astype(np.int8)
            aligned_log = logabs_det(aligned.astype(np.float64))
            if aligned_log > global_best_log:
                global_best = aligned
        except Exception:
            pass

    # Ensure the result is a proper ±1 integer matrix
    if global_best is None:
        global_best = rng.choice([-1, 1], size=(n, n)).astype(np.int8)
    else:
        global_best = np.where(global_best >= 0, 1, -1).astype(np.int8)

    return global_best
# EVOLVE-BLOCK-END


# Fixed API for evaluator
def run_code():
    """
    Run the Hadamard matrix constructor for n=29.
    
    Returns:
        Tuple of (matrix,) where matrix is an (29, 29) array with entries ±1
    """
    matrix = construct_hadamard_matrix(n=29)
    return (matrix,)


if __name__ == "__main__":
    matrix = run_code()[0]
    print(f"Constructed Hadamard matrix of size {matrix.shape[0]}x{matrix.shape[1]}")
    # Calculate determinant for verification
    det_val = np.linalg.det(matrix.astype(float))
    print(f"Determinant: {abs(det_val):.2e}")
