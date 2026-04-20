import os
import sys
import time
import warnings
import argparse
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

# ── thread pinning so timing is fair ──────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import numpy as np
from sklearn.linear_model import lasso_path as sk_lasso_path

try:
    from scipy import sparse as sp_sparse
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


# ══════════════════════════════════════════════════════════════════════════════
# Config (overridable via CLI)
# ══════════════════════════════════════════════════════════════════════════════
N_REPS        = 5       # timing repetitions (min is reported)
TOL_OBJECTIVE = 1e-6    # max allowed obj gap vs sklearn oracle
N_ALPHAS      = 50      # lambda grid size
EPS           = 1e-2    # lambda_min / lambda_max ratio


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_dense(X):
    if HAS_SCIPY_SPARSE and sp_sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ══════════════════════════════════════════════════════════════════════════════

def make_correlated_data(n_samples, n_features, corr=0.6, n_nonzeros=None,
                         snr=10, random_state=42):
    """Toeplitz-correlated Gaussian design (matches generate_results.py)."""
    rng     = np.random.RandomState(random_state)
    indices = np.arange(n_features)
    cov     = corr ** np.abs(indices[:, None] - indices[None, :])
    L       = np.linalg.cholesky(cov)
    X       = rng.randn(n_samples, n_features) @ L.T
    if n_nonzeros is None:
        n_nonzeros = max(1, n_features // 10)
    w_true  = np.zeros(n_features)
    step = n_features // n_nonzeros; nz_idx = np.arange(0, n_features, step); w_true[nz_idx] = (-1) ** np.arange(len(nz_idx))
    y_sig   = X @ w_true
    noise   = np.linalg.norm(y_sig) / (snr * np.sqrt(n_samples))
    y       = y_sig + noise * rng.randn(n_samples)
    return X, y

def make_glmnet_data(n_samples, n_features, corr=0.5, snr=3.0, random_state=42):
    """
    Equicorrelation design from Friedman, Hastie & Tibshirani (2010), Section 5.1.

    Design matrix: all pairs (X_j, X_k) have the same population correlation rho.
    Coefficients:  beta_j = (-1)^j * exp(-2*(j-1)/20)  — alternating signs,
                   exponentially decaying, dense (all nonzero).
    Noise:         k chosen so signal-to-noise ratio = snr.

    This stresses the covariance method (dense active set from the start)
    and high-correlation regimes where CD convergence slows.
    """
    rng = np.random.RandomState(random_state)
    # Equicorrelation: Sigma = (1-rho)*I + rho*11^T
    # Sample via: X = sqrt(1-rho)*Z + sqrt(rho)*z_common
    Z        = rng.randn(n_samples, n_features)
    z_common = rng.randn(n_samples, 1)
    X        = np.sqrt(1 - corr) * Z + np.sqrt(corr) * z_common
    # Exponentially decaying, alternating-sign coefficients (glmnet paper eq. 31)
    j        = np.arange(n_features)
    beta     = ((-1) ** j) * np.exp(-2 * j / 20)
    y_sig    = X @ beta
    noise    = np.linalg.norm(y_sig) / (snr * np.sqrt(n_samples))
    y        = y_sig + noise * rng.randn(n_samples)
    return X.astype(np.float64), y.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Real-world dataset loader (libsvmdata)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_libsvm(name, **kwargs):
    """Try both libsvmdata API variants."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from libsvmdata import fetch_dataset
            return fetch_dataset(name, **kwargs)
        except Exception:
            from libsvmdata import fetch_libsvm
            return fetch_libsvm(name, **kwargs)


def _load_real(libsvm_name, split=True, test_size=0.15, seed=42, **kwargs):
    X, y = _fetch_libsvm(libsvm_name, **kwargs)
    X, y = _to_dense(X), y.astype(np.float64)
    if split:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X, y
def _fetch_rcv1():
    from sklearn.model_selection import train_test_split
    X, y = _fetch_libsvm("rcv1.binary", min_nnz=3)
    X, y = _to_dense(X), y.astype(np.float64)
    X, _, y, _ = train_test_split(X, y, test_size=0.15, random_state=42)
    return X, y



REAL_DATASET_LOADERS_NONBIO = [
    # ── general ML benchmarks (non-biological) ───────────────────────────────
    ("Gisette",  lambda: _load_real("gisette",  split=True)),
    ("RCV1",     _fetch_rcv1),
]

REAL_DATASET_LOADERS_BIO = [
    # ── biological libsvm datasets ────────────────────────────────────────────
    ("DNA",                lambda: _load_real("dna")),
    ("Leukemia",           lambda: _load_real("leukemia", split=False)),
    ("Colon Cancer",       lambda: _load_real("colon-cancer", split=False)),
    ("Duke Breast Cancer", lambda: _load_real("duke breast-cancer", split=False)),
]


def _build_dataset_list(loaders, verbose=True):
    """Load a list of (name, loader) pairs, skipping failures gracefully."""
    ds = []
    for name, loader in loaders:
        try:
            X, y = loader()
            X = _to_dense(X)
            y = np.asarray(y, dtype=np.float64).ravel()
            ds.append((f"{name} ({X.shape[0]}×{X.shape[1]})", X, y))
            if verbose:
                print(f"  [OK]   {name}  ({X.shape[0]}×{X.shape[1]})")
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {name}: {e}")
    return ds


def build_real(verbose=True):
    """Load real-world datasets. Returns (nonbio_list, bio_list)."""
    try:
        import libsvmdata  # noqa: F401
    except ImportError:
        if verbose:
            print("  [SKIP] libsvmdata not installed — run: pip install libsvmdata")
        return [], []

    if verbose:
        print("  Non-bio:")
    nonbio = _build_dataset_list(REAL_DATASET_LOADERS_NONBIO, verbose=verbose)
    if verbose:
        print("  Bio:")
    bio    = _build_dataset_list(REAL_DATASET_LOADERS_BIO,    verbose=verbose)
    return nonbio, bio


def _genomics_data_dir():
    env = os.environ.get("GENOMICS_DATA_DIR", "")
    if env and os.path.isdir(env):
        return env
    return os.path.dirname(os.path.abspath(__file__))


def _find_file(*candidates):
    """Return first existing path, searching in order:
    GENOMICS_DATA_DIR, <script_dir>/eval_data, <script_dir>, ~/Downloads.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    env  = os.environ.get("GENOMICS_DATA_DIR", "")
    search_dirs = [d for d in [
        env if env else None,
        os.path.join(base, "eval_data"),
        base,
        os.path.expanduser("~/Downloads"),
    ] if d and os.path.isdir(d)]

    for name in candidates:
        for directory in search_dirs:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def _load_kaggle_rna(csv_path, max_samples=None, seed=42):
    """
    Load Kaggle RNA-seq CSV: rows=samples, cols=genes + last col=sample_type_id.
    Returns X (samples x genes, float64), y (binary 0/1, float64).
    """
    import pandas as pd
    df  = pd.read_csv(csv_path)
    y   = df.iloc[:, -1].values.astype(np.float64)   # last col: sample_type_id
    X   = df.iloc[:, :-1].values.astype(np.float64)  # gene expression
    # Drop zero-variance genes
    X   = X[:, X.var(axis=0) > 0]
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]
    return np.ascontiguousarray(X), y


def _load_tcga_brca(expr_path, clinical_path, max_samples=None, seed=42):
    """
    Load TCGA BRCA HiSeqV2.gz (genes x samples) + BRCA_clinicalMatrix.
    Uses Age_at_Initial_Pathologic_Diagnosis as y (continuous regression).
    Returns X (samples x genes, float64), y (age, float64).
    """
    import pandas as pd
    expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
    expr = expr.T
    clin = pd.read_csv(clinical_path, sep='\t', index_col=0)
    common = expr.index.intersection(clin.index)
    expr = expr.loc[common]
    clin = clin.loc[common]
    age_col = next((c for c in clin.columns if "Age" in c and "Diagnosis" in c), None)
    if age_col is None:
        raise ValueError("Age column not found in clinical matrix")
    valid = clin[age_col].notna()
    X = expr.loc[valid].values.astype(np.float64)
    y = clin.loc[valid, age_col].values.astype(np.float64)
    X = X[:, X.var(axis=0) > 0]   # drop zero-variance genes
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]
    y = (y - y.mean()) / y.std()
    return np.ascontiguousarray(X), y


def build_local_genomics(verbose=True, max_samples=500):
    """
    Load locally downloaded genomics files. Skips any file not found.
    max_samples: cap dataset size for benchmarking speed.
    """
    ds = []

    # TCGA BRCA
    expr_path = _find_file("HiSeqV2.gz", "HiSeqV2")
    clin_path = _find_file("BRCA_clinicalMatrix")
    if expr_path and clin_path:
        try:
            X, y = _load_tcga_brca(expr_path, clin_path, max_samples=max_samples)
            ds.append((f"TCGA BRCA ({X.shape[0]}×{X.shape[1]})", X, y))
            if verbose: print(f"  [OK]   TCGA BRCA  ({X.shape[0]}×{X.shape[1]})  y=age")
        except Exception as e:
            if verbose: print(f"  [SKIP] TCGA BRCA: {e}")
    elif verbose:
        print("  [SKIP] TCGA BRCA: HiSeqV2.gz or BRCA_clinicalMatrix not found")

    # Kaggle RNA-seq
    for fname, label in [
        ("Liver RNA Data.csv",    "TCGA Liver RNA"),
        ("Lung RNA Data.csv",     "TCGA Lung RNA"),
        ("Prostate RNA Data.csv", "TCGA Prostate RNA"),
        ("Thyroid RNA Data.csv",  "TCGA Thyroid RNA"),
    ]:
        path = _find_file(fname)
        if path:
            try:
                X, y = _load_kaggle_rna(path, max_samples=max_samples)
                ds.append((f"{label} ({X.shape[0]}×{X.shape[1]})", X, y))
                if verbose: print(f"  [OK]   {label}  ({X.shape[0]}×{X.shape[1]})  y=tumor/normal")
            except Exception as e:
                if verbose: print(f"  [SKIP] {label}: {e}")
        elif verbose:
            print(f"  [SKIP] {label}: {fname} not found — set GENOMICS_DATA_DIR or place in script dir")

    return ds



# ══════════════════════════════════════════════════════════════════════════════
# Synthetic dataset registry
# ══════════════════════════════════════════════════════════════════════════════

def build_synthetic():
    """
    Correlated Gaussian synthetic datasets covering:
    - Three sizes (small/medium/wide) with multiple seeds
    - Varying correlation levels (low=0.3, medium=0.6, high=0.9)
    - Stress tests (large dense, ultrawide) — always included
    """
    from sklearn.model_selection import train_test_split
    ds = []

    # ── Base cases (corr=0.6, three sizes, default seed) ─────────────────────
    for n, p, nz, tag in [(500, 200, 10, "synt_small"),
                           (1000, 500, 25, "synt_medium"),
                           (500, 2000, 50, "synt_wide")]:
        X, y = make_correlated_data(n, p, n_nonzeros=nz)
        X, _, y, _ = train_test_split(X, y, test_size=0.15, random_state=42)
        ds.append((f"{tag} ({X.shape[0]}×{X.shape[1]})", X, y))

    # ── Extra seeds for variance estimate ─────────────────────────────────────
    for seed in [7, 123]:
        X, y = make_correlated_data(425, 200, n_nonzeros=10, random_state=seed)
        ds.append((f"synt_small (425×200, s={seed})", X, y))

    for seed in [7]:
        X, y = make_correlated_data(850, 500, n_nonzeros=25, random_state=seed)
        ds.append((f"synt_medium (850×500, s={seed})", X, y))

    for seed in [7]:
        X, y = make_correlated_data(425, 2000, n_nonzeros=50, random_state=seed)
        ds.append((f"synt_wide (425×2000, s={seed})", X, y))

    # ── Correlation level variants (medium size, varying rho) ─────────────────
    for corr, label in [(0.3, "corr_low"), (0.9, "corr_high")]:
        X, y = make_correlated_data(850, 500, corr=corr, n_nonzeros=25,
                                    random_state=42)
        X, _, y, _ = train_test_split(X, y, test_size=0.15, random_state=42)
        ds.append((f"{label} (850×500, rho={corr})", X, y))

    # ── Stress tests (always included) ────────────────────────────────────────
    X, y = make_correlated_data(2000, 800, corr=0.8, n_nonzeros=80,
                                snr=8, random_state=42)
    ds.append(("stress_dense (2000×800, corr=0.8)", X, y))

    X, y = make_correlated_data(500, 5000, n_nonzeros=100, snr=3,
                                random_state=42)
    ds.append(("stress_ultrawide (500×5000)", X, y))

    return ds


def build_glmnet_synthetic():
    """
    Equicorrelation synthetic datasets matching Friedman et al. (2010) Table 1.

    Problem sizes: (N, p) pairs from the paper, extended to larger p.
    Correlation levels: rho in {0, 0.5, 0.95} — zero, moderate, extreme.
    Signal: dense exponentially decaying coefficients (all p nonzero).
    SNR: 3.0 (matches the paper).

    These complement the Toeplitz datasets: equicorrelation with dense signal
    stresses the covariance method's Gram cache and high-correlation CD convergence.
    """
    ds = []

    # ── Problem sizes from Table 1 of glmnet paper ────────────────────────────
    # (N, p, rho, label)
    configs = [
        (1000, 100,   0.0,  "glmnet_N1000_p100_rho0.0"),
        (1000, 100,   0.5,  "glmnet_N1000_p100_rho0.5"),
        (1000, 100,   0.95, "glmnet_N1000_p100_rho0.95"),
        (100,  1000,  0.0,  "glmnet_N100_p1000_rho0.0"),
        (100,  1000,  0.5,  "glmnet_N100_p1000_rho0.5"),
        (100,  1000,  0.95, "glmnet_N100_p1000_rho0.95"),
        # ── scaled up beyond the paper ────────────────────────────────────────
        (100,  5000,  0.0,  "glmnet_N100_p5000_rho0.0"),
        (100,  5000,  0.5,  "glmnet_N100_p5000_rho0.5"),
        (100,  20000, 0.0,  "glmnet_N100_p20000_rho0.0"),
        (100,  20000, 0.5,  "glmnet_N100_p20000_rho0.5"),
        (100,  50000, 0.0,  "glmnet_N100_p50000_rho0.0"),
    ]

    for n, p, rho, label in configs:
        try:
            X, y = make_glmnet_data(n, p, corr=rho, snr=3.0, random_state=42)
            ds.append((f"{label} ({n}×{p})", X, y))
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")

    return ds



# ══════════════════════════════════════════════════════════════════════════════

import struct
import hashlib
import subprocess as _subprocess
import tempfile as _tempfile

def _compile_binary(cpp_code, extra_flags, task_dir):
    """Compile CPP_CODE to a binary (cached by content hash). Returns binary path."""
    code_hash   = hashlib.md5(cpp_code.encode()).hexdigest()[:12]
    cache_dir   = os.path.join(_tempfile.gettempdir(), "lasso_path_cpp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    binary_path = os.path.join(cache_dir, f"lasso_solver_{code_hash}")

    if not os.path.exists(binary_path):
        src_path = binary_path + ".cpp"
        with open(src_path, "w") as f:
            f.write(cpp_code)

        eigen_local = os.path.join(task_dir, "eigen")
        eigen_flag  = f"-I{eigen_local}" if os.path.isdir(eigen_local) else "-I/usr/include/eigen3"

        base_flags  = ["g++", "-O3", "-march=native", "-std=c++17",
                       eigen_flag, src_path, "-o", binary_path]
        compile_cmd = base_flags[:5] + list(extra_flags) + base_flags[5:]

        result = _subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{result.stderr}")

    return binary_path


def _make_callable(binary_path):
    """Return a lasso_path_solve-compatible callable that invokes the binary."""
    def solver(problem):
        X           = np.asarray(problem["X"],           dtype=np.float64, order="C")
        y           = np.asarray(problem["y"],           dtype=np.float64)
        lambda_path = np.asarray(problem["lambda_path"], dtype=np.float64)
        n, p        = X.shape
        n_lambda    = len(lambda_path)

        payload = (struct.pack("iii", n, p, n_lambda)
                   + X.tobytes()
                   + y.tobytes()
                   + lambda_path.tobytes())

        proc = _subprocess.run([binary_path], input=payload,
                                capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Solver crashed (code {proc.returncode}):\n"
                f"{proc.stderr.decode(errors='ignore')[:500]}"
            )
        expected = p * n_lambda * 8
        if len(proc.stdout) != expected:
            raise RuntimeError(
                f"Output size mismatch: got {len(proc.stdout)}, expected {expected}"
            )
        return np.frombuffer(proc.stdout, dtype=np.float64).reshape((p, n_lambda), order="F")

    return solver


def load_solver(path):
    if not os.path.exists(path):
        return None, f"file not found: {path}"
    try:
        spec = importlib.util.spec_from_file_location("_prog", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if hasattr(mod, "CPP_CODE"):
            task_dir    = os.environ.get("LASSO_TASK_DIR",
                                         os.path.dirname(os.path.abspath(path)))
            extra_flags = list(getattr(mod, "COMPILE_FLAGS", []))
            binary_path = _compile_binary(mod.CPP_CODE, extra_flags, task_dir)
            return _make_callable(binary_path), None

        if hasattr(mod, "lasso_path_solve"):
            return mod.lasso_path_solve, None

        return None, "program must define CPP_CODE or lasso_path_solve()"
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def lasso_obj(X, y, w, lam):
    r = y - X @ w
    return 0.5 / X.shape[0] * np.dot(r, r) + lam * np.abs(w).sum()


def max_obj_gap(X, y, coef_sol, coef_ref, alphas):
    return max(
        lasso_obj(X, y, coef_sol[:, k], alphas[k])
        - lasso_obj(X, y, coef_ref[:, k], alphas[k])
        for k in range(len(alphas))
    )


# ══════════════════════════════════════════════════════════════════════════════
# Timing
# ══════════════════════════════════════════════════════════════════════════════

def time_fn(fn, problem, n_reps):
    def _copy(p):
        return {k: v.copy() if hasattr(v, "copy") else v for k, v in p.items()}
    fn(_copy(problem))
    times, result = [], None
    for _ in range(n_reps):
        t0     = time.perf_counter()
        result = fn(_copy(problem))
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, result


def time_sklearn(X, y, n_reps):
    sk_lasso_path(X.copy(), y.copy(), n_alphas=N_ALPHAS, eps=EPS)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        sk_lasso_path(X.copy(), y.copy(), n_alphas=N_ALPHAS, eps=EPS)
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark one dataset
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(X, y, init_fn, tte_fn, n_reps, tol, dtype=np.float64):
    X = np.ascontiguousarray(X, dtype=dtype)
    y = np.ascontiguousarray(y.ravel(), dtype=dtype)

    alphas, coefs_sk, _ = sk_lasso_path(X, y, n_alphas=N_ALPHAS, eps=EPS)
    sk_ms = time_sklearn(X, y, n_reps)
    prob  = {"X": X, "y": y, "lambda_path": alphas}

    def run(fn):
        if fn is None:
            return None
        try:
            ms, coef = time_fn(fn, prob, n_reps)
            coef     = np.asarray(coef, dtype=np.float64)
            if coef.shape != (X.shape[1], len(alphas)):
                return dict(ms=ms, gap=np.inf, valid=False,
                            err=f"shape {coef.shape}")
            gap   = max_obj_gap(X, y, coef, coefs_sk, alphas)
            valid = gap <= tol
            return dict(ms=ms, gap=gap, valid=valid,
                        spd_sk=sk_ms / ms if ms > 0 else 0.0)
        except Exception as e:
            return dict(ms=np.inf, gap=np.inf, valid=False, err=str(e))

    return sk_ms, run(init_fn), run(tte_fn)


# ══════════════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════════════

NW = 40

def geo_mean(vals):
    vals = [v for v in vals if v and np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")

def _ms(v):  return f"{v:8.1f}" if np.isfinite(v) else "   ERROR"
def _gap(v): return f"{v:10.2e}" if np.isfinite(v) else "       inf"
def _ok(r):  return "OK  " if r and r["valid"] else "FAIL"
def _spd(v): return f"{v:7.2f}x" if (v and np.isfinite(v) and v > 0) else "    n/a"


def _sec_header(label):
    print(f"\n  ── {label} {'─' * max(0, 58 - len(label))}")


def _print_section_summary1(sec_label, spds, n_valid, n_total):
    gm = geo_mean(spds)
    gm_str = f"{gm:.2f}x" if np.isfinite(gm) else "n/a"
    print(f"  → [{sec_label}]  valid: {n_valid}/{n_total}  │  "
          f"geo-mean speedup vs sklearn: {gm_str}")


def _print_section_summary2(sec_label, ratios, n_vi, n_vt, n_total):
    gm = geo_mean(ratios)
    gm_str = f"{gm:.2f}x" if np.isfinite(gm) else "n/a"
    print(f"  → [{sec_label}]  init valid: {n_vi}/{n_total}  │  "
          f"tte valid: {n_vt}/{n_total}  │  tte is {gm_str} faster than init")


def print_table1(sections, tol):
    W = "=" * 96
    print(); print(W)
    print("  TABLE 1  —  sklearn  vs  tte")
    print(f"  Correctness oracle: sklearn lasso_path  │  valid if obj gap ≤ {tol}")
    print(W)
    hdr = (f"  {'Dataset':<{NW}}  {'sklearn(ms)':>11}  │"
           f"  {'tte(ms)':>8}  {'vs sklearn':>10}  {'obj gap':>10}  ok")
    sep = "  " + "─" * (len(hdr) - 2)

    all_spds, all_valid, all_total = [], 0, 0
    sec_summaries = []

    for sec_label, rows in sections:
        if not rows:
            continue
        _sec_header(sec_label)
        print(hdr); print(sep)
        sec_spds, sec_valid, sec_total = [], 0, 0
        for name, sk_ms, init_r, tte_r in rows:
            sec_total += 1
            all_total += 1
            if tte_r is None:
                print(f"  {name:<{NW}}  {sk_ms:>11.1f}  │  {'ERROR':>8}")
                continue
            spd = _spd(tte_r["spd_sk"]) if tte_r["valid"] else "    n/a"
            print(f"  {name:<{NW}}  {sk_ms:>11.1f}  │"
                  f"  {_ms(tte_r['ms']):>8}  {spd:>10}  {_gap(tte_r['gap']):>10}  {_ok(tte_r)}")
            if tte_r["valid"]:
                sec_valid  += 1
                all_valid  += 1
                sec_spds.append(tte_r["spd_sk"])
                all_spds.append(tte_r["spd_sk"])
        sec_summaries.append((sec_label, sec_spds, sec_valid, sec_total))

    # ── Per-section summaries ────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  SUMMARY BY SECTION")
    for label, spds, nv, nt in sec_summaries:
        _print_section_summary1(label, spds, nv, nt)

    # ── Overall summary ──────────────────────────────────────────────────────
    gm_all = geo_mean(all_spds)
    gm_str = f"{gm_all:.2f}x" if np.isfinite(gm_all) else "n/a"
    print(f"  {'─'*60}")
    print(f"  OVERALL  valid: {all_valid}/{all_total}  │  "
          f"geo-mean speedup vs sklearn: {gm_str}")


def print_table2(sections):
    W = "=" * 96
    print(); print(W)
    print("  TABLE 2  —  init  vs  tte")
    print("  Speed ratio = init_ms / tte_ms  (>1x means tte is faster)")
    print(W)
    hdr = (f"  {'Dataset':<{NW}}  {'init(ms)':>8}  │"
           f"  {'tte(ms)':>8}  {'tte/init':>9}  {'init gap':>10}  {'tte gap':>10}"
           f"  {'ok(i)':>5}  {'ok(t)':>5}")
    sep = "  " + "─" * (len(hdr) - 2)

    all_ratios, all_vi, all_vt, all_total = [], 0, 0, 0
    sec_summaries = []

    for sec_label, rows in sections:
        if not rows:
            continue
        _sec_header(sec_label)
        print(hdr); print(sep)
        sec_ratios, sec_vi, sec_vt, sec_total = [], 0, 0, 0
        for name, sk_ms, init_r, tte_r in rows:
            sec_total += 1
            all_total += 1
            if init_r is None:
                print(f"  {name:<{NW}}  {'(init not loaded)':>70}")
                continue
            if tte_r is None:
                print(f"  {name:<{NW}}  {_ms(init_r['ms']):>8}  │  {'ERROR':>8}")
                continue
            ratio = (_spd(init_r["ms"] / tte_r["ms"])
                     if (tte_r["valid"] and tte_r["ms"] > 0
                         and np.isfinite(init_r["ms"]))
                     else "    n/a")
            print(f"  {name:<{NW}}  {_ms(init_r['ms']):>8}  │"
                  f"  {_ms(tte_r['ms']):>8}  {ratio:>9}"
                  f"  {_gap(init_r['gap']):>10}  {_gap(tte_r['gap']):>10}"
                  f"  {_ok(init_r):>5}  {_ok(tte_r):>5}")
            if init_r["valid"]:
                sec_vi += 1; all_vi += 1
            if tte_r["valid"]:
                sec_vt += 1; all_vt += 1
            if (init_r["valid"] and tte_r["valid"]
                    and tte_r["ms"] > 0 and np.isfinite(init_r["ms"])):
                r = init_r["ms"] / tte_r["ms"]
                sec_ratios.append(r)
                all_ratios.append(r)
        sec_summaries.append((sec_label, sec_ratios, sec_vi, sec_vt, sec_total))

    # ── Per-section summaries ────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  SUMMARY BY SECTION")
    for label, ratios, nvi, nvt, nt in sec_summaries:
        _print_section_summary2(label, ratios, nvi, nvt, nt)

    # ── Overall summary ──────────────────────────────────────────────────────
    gm_all = geo_mean(all_ratios)
    gm_str = f"{gm_all:.2f}x" if np.isfinite(gm_all) else "n/a"
    print(f"  {'─'*60}")
    print(f"  OVERALL  init valid: {all_vi}/{all_total}  │  "
          f"tte valid: {all_vt}/{all_total}  │  "
          f"tte is {gm_str} faster than init (geo-mean, both valid)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init",         default="init_program.py",
                        help="Path to init solver (default: init_program.py)")
    parser.add_argument("--tte",          default="best_program.py",
                        help="Path to tte solver  (default: best_program.py)")
    parser.add_argument("--reps",         type=int,   default=N_REPS,
                        help=f"Timing repetitions — default {N_REPS}")
    parser.add_argument("--tol",          type=float, default=TOL_OBJECTIVE,
                        help=f"Obj-gap tolerance — default {TOL_OBJECTIVE}")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Skip all synthetic datasets")
    parser.add_argument("--no-genomics",  action="store_true",
                        help="Skip local genomics datasets (TCGA/Kaggle RNA-seq)")
    parser.add_argument("--genomics-dir",
                        default=os.environ.get(
                            "GENOMICS_DATA_DIR",
                            str(Path(__file__).resolve().parent / "eval_data"),
                        ),
                        help="Directory containing local genomics files "
                             "(HiSeqV2.gz, BRCA_clinicalMatrix, *RNA Data.csv). "
                             "Defaults to lasso_path/eval_data relative to this script, "
                             "overridable via $GENOMICS_DATA_DIR.")
    parser.add_argument("--dtype",        default="float64",
                        choices=["float64", "float32"],
                        help="Dtype for X and y passed to all solvers (default: float64)")
    parser.add_argument("--no-real",      action="store_true",
                        help="Skip real-world libsvm datasets")
    parser.add_argument("--genomics-max-samples", type=int, default=500,
                        help="Max samples per genomics dataset (default 500)")
    args = parser.parse_args()

    # Wire --genomics-dir into the env var so _find_file picks it up
    if args.genomics_dir and os.path.isdir(args.genomics_dir):
        os.environ["GENOMICS_DATA_DIR"] = args.genomics_dir
    elif args.genomics_dir and not os.path.isdir(args.genomics_dir):
        print(f"  ⚠  --genomics-dir '{args.genomics_dir}' not found — genomics datasets will be skipped.")

    os.environ.setdefault("LASSO_TASK_DIR",
                          os.path.dirname(os.path.abspath(__file__)))

    print()
    print("=" * 96)
    print("  tte benchmark  —  lasso path solver comparison")
    print(f"  Config: n_alphas={N_ALPHAS}, eps={EPS}, tol={args.tol}, "
          f"reps={args.reps} (min reported)")
    print("=" * 96)

    # ── Load solvers ─────────────────────────────────────────────────────────
    print("\n  Loading solvers...")
    init_fn, init_err = load_solver(args.init)
    tte_fn,  tte_err  = load_solver(args.tte)
    print(f"  init : {'OK' if init_fn else f'SKIP — {init_err}'}")
    print(f"  tte  : {'OK' if tte_fn  else f'FAIL — {tte_err}'}")
    if tte_fn is None:
        print("\n  tte solver is required. Exiting.")
        sys.exit(1)

    # ── Build dataset list ────────────────────────────────────────────────────
    sections_data = []

    if not args.no_synthetic:
        synt = build_synthetic()
        sections_data.append(("SYNTHETIC  (Toeplitz corr=0.6, snr=10)", synt))
        glmnet_synt = build_glmnet_synthetic()
        sections_data.append(("SYNTHETIC  (glmnet equicorr, snr=3.0)", glmnet_synt))

    if not args.no_real or not args.no_genomics:
        bio_combined = []
        if not args.no_real:
            print("\n  Fetching real-world datasets (libsvmdata)...")
            nonbio, bio = build_real(verbose=True)
            if nonbio:
                sections_data.append(("REAL-WORLD NON-BIO  (libsvm)", nonbio))
            bio_combined.extend(bio)
        if not args.no_genomics:
            print("\n  Loading local genomics datasets (TCGA / Kaggle RNA-seq)...")
            genomics = build_local_genomics(verbose=True,
                                            max_samples=args.genomics_max_samples)
            bio_combined.extend(genomics)
        if bio_combined:
            sections_data.append(("REAL-WORLD BIO  (libsvm + TCGA/Kaggle)", bio_combined))

    total = sum(len(ds) for _, ds in sections_data)
    if total == 0:
        print("\n  No datasets to run. Exiting.")
        sys.exit(0)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\n  Running {total} dataset(s) × {args.reps} reps...\n")

    result_sections = []
    for sec_label, datasets in sections_data:
        rows = []
        for name, X, y in datasets:
            print(f"  {name}...", end=" ", flush=True)
            try:
                sk_ms, init_r, tte_r = benchmark(X, y, init_fn, tte_fn,
                                                  args.reps, args.tol,
                                                  dtype=np.float32 if args.dtype == "float32" else np.float64)
                rows.append((name, sk_ms, init_r, tte_r))
                tte_tag  = ("✓" if tte_r  and tte_r["valid"]
                            else f"✗(gap={tte_r['gap']:.1e})" if tte_r
                            else "✗(ERROR)")
                init_tag = ("✓" if init_r and init_r["valid"]
                            else "—" if init_r is None
                            else f"✗(gap={init_r['gap']:.1e})")
                print(f"tte={tte_tag}  init={init_tag}")
            except Exception as e:
                print(f"SKIP — {e}")
                print(f"  ⚠  [{sec_label}] '{name}' failed and will be excluded from results.")
        result_sections.append((sec_label, rows))

    print_table1(result_sections, args.tol)
    print_table2(result_sections)
    print()


if __name__ == "__main__":
    main()