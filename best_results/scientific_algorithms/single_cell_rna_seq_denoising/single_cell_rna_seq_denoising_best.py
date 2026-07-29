# EVOLVE-BLOCK-START

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import diags
import warnings
import gc
import scprep
import scprep.utils
import scprep.normalize
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
import anndata
import scanpy as sc

# Optional import of magic_impute for optional denoiser
try:
    from magic_impute import MAGIC
    _MAGIC_AVAILABLE = True
except Exception:
    _MAGIC_AVAILABLE = False


def _seed_from_global():
    """Create a deterministic seed from GLOBAL_BEST_CONSTRUCTION, if available."""
    try:
        val = GLOBAL_BEST_CONSTRUCTION
    except NameError:
        val = None
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple, np.ndarray)):
            seed_val = float(val[0]) if len(val) > 0 else 0.0
        else:
            seed_val = float(val)
    except Exception:
        seed_val = 0.0
    return int(round(seed_val * 1_000_000)) % (2 ** 31 - 1)


def _weighted_average_diffusion(data, diff_op, t, decay):
    """Weighted average of t diffusion snapshots."""
    if t < 1:
        raise ValueError("Parameter t must be >= 1")
    current = data.astype(np.float32, copy=True)
    weights = np.array([decay ** (t - 1 - i) for i in range(t)], dtype=np.float32)
    accum = current * weights[0]
    weight_sum = weights[0]
    for i in range(1, t):
        current = diff_op.dot(current)
        accum += current * weights[i]
        weight_sum += weights[i]
    return accum / weight_sum


def _construct_diffusion_operator(X_rep, knn, metric="correlation", n_jobs=8):
    """Build a row‑normalised diffusion operator."""
    n_cells = X_rep.shape[0]
    knn_adj = max(1, min(knn, n_cells - 1))
    nn = NearestNeighbors(
        n_neighbors=knn_adj,
        metric=metric,
        n_jobs=n_jobs,
    )
    nn.fit(X_rep)
    distances, indices = nn.kneighbors(X_rep)
    sigma = (
        np.mean(distances[:, 1:]) + 1e-8 if knn_adj > 1 else 1.0
    )
    weights = np.exp(-(distances ** 2) / sigma ** 2)
    rows = np.repeat(np.arange(n_cells), knn_adj)
    cols = indices.ravel()
    W = sparse.csr_matrix(
        (weights.ravel(), (rows, cols)),
        shape=(n_cells, n_cells),
        dtype=np.float32,
    )
    W_sym = (W + W.T) * 0.5
    row_sums = np.array(W_sym.sum(axis=1)).flatten().astype(np.float32)
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1.0 / row_sums, dtype=np.float32)
    return D_inv.dot(W_sym).tocsr()


def _mse_val_loss(den, obs, val_idx):
    """Mean squared error in log‑normalised space on validation cells."""
    preds = den[val_idx].astype(np.float64)
    true = obs[val_idx].astype(np.float64)
    p_sum = preds.sum(axis=1)
    o_sum = true.sum(axis=1)
    mask = (p_sum > 0) & (o_sum > 0)
    if not np.any(mask):
        return 1e6
    preds_mask = preds[mask]
    true_mask = true[mask]
    preds_norm = preds_mask / p_sum[mask][:, None] * 10000.0
    true_norm = true_mask / o_sum[mask][:, None] * 10000.0
    return np.mean((np.log1p(preds_norm) - np.log1p(true_norm)) ** 2)


def _poisson_val_loss(den, obs, *unused):
    """Poisson negative log‑likelihood."""
    preds = den.astype(np.float64)
    true = obs.astype(np.float64)
    eps = 1e-12
    preds_eps = np.maximum(preds, eps)
    return np.sum(preds_eps - true * np.log(preds_eps))


def _denoise_pipeline(
    cfg,
    X_arr,
    X_norm,
    X_log,
    diff_op,
    diff_op_log,
    libsize_raw,
    libsize_sqrt,
    X_svd_proj,
    svd_components,
    W_nmf,
    H_nmf,
    dropout_frac_gene,
    raw_neighbor_mean_corr,
    raw_neighbor_sum_corr,
    raw_neighbor_mean_euclid,
    raw_neighbor_sum_euclid,
):
    """Apply a single denoising configuration."""
    method = cfg.get("method")

    # raw diffusion
    if method == "raw_diff":
        smoothed = diff_op.dot(X_arr.astype(np.float32))
        return smoothed.astype(np.float32)

    # average of raw neighbor mean (corr + euclid)
    if method == "nn_mean":
        combined = (raw_neighbor_mean_corr + raw_neighbor_mean_euclid) * 0.5
        return combined.astype(np.float32)

    # log diffusion with one step, scaled
    if method == "log_diff":
        t_val = cfg.get("t", 8)
        decay_val = cfg.get("lam", 0.5)
        threshold = cfg.get("threshold", 0.0)

        smoothed_log = _weighted_average_diffusion(X_log, diff_op_log, t_val, decay_val)
        smoothed_counts = np.expm1(smoothed_log).astype(np.float32)
        if threshold > 0.0:
            smoothed_counts[smoothed_counts < threshold] = 0.0

        smoothed_sums = smoothed_counts.sum(axis=1).astype(np.float32)
        raw_scale = np.divide(
            libsize_raw,
            smoothed_sums,
            out=np.zeros_like(smoothed_sums),
            where=smoothed_sums != 0,
        )
        denoised = smoothed_counts * raw_scale[:, None]
        np.maximum(denoised, 0.0, out=denoised)
        return denoised.astype(np.float32)

    # neighbor mean from log adjacency
    if method == "raw_neighbor_mean_log":
        smoothed_log = diff_op_log.dot(X_log).astype(np.float32)
        smoothed_counts = np.expm1(smoothed_log).astype(np.float32)
        smoothed_counts[smoothed_counts < 0] = 0.0
        smoothed_sums = smoothed_counts.sum(axis=1).astype(np.float32)
        raw_scale = np.divide(
            libsize_raw,
            smoothed_sums,
            out=np.zeros_like(smoothed_sums),
            where=smoothed_sums != 0,
        )
        denoised = smoothed_counts * raw_scale[:, None]
        np.maximum(denoised, 0.0, out=denoised)
        return denoised.astype(np.float32)

    # Default weighted diffusion + low‑rank combination
    t_vals = cfg["t"]
    lam_vals = cfg["lam"]
    w_vals = cfg["weights"]
    threshold = cfg.get("threshold", 0.0)

    alpha_base = cfg.get("alpha_base", 0.86)
    beta_base = cfg.get("beta_base", 0.86)
    sigma_low = cfg.get("sigma_low", 0.28)
    sigma_high = cfg.get("sigma_high", 0.82)
    w_svd = cfg.get("w_svd", 0.55)
    w_nmf = cfg.get("w_nmf", 0.45)
    clip_min = cfg.get("clip_min", 0.94)
    clip_max = cfg.get("clip_max", 1.06)

    data_avg1 = _weighted_average_diffusion(X_norm, diff_op, t_vals[0], lam_vals[0])
    data_avg2 = _weighted_average_diffusion(X_norm, diff_op, t_vals[1], lam_vals[1])
    data_avg3 = _weighted_average_diffusion(X_norm, diff_op, t_vals[2], lam_vals[2])
    diff_avg = w_vals[0] * data_avg1 + w_vals[1] * data_avg2 + w_vals[2] * data_avg3
    np.maximum(diff_avg, 0.0, out=diff_avg)

    imputed_counts = (diff_avg ** 2) * libsize_sqrt[:, None]
    if threshold > 0.0:
        imputed_counts[imputed_counts < threshold] = 0.0
    imputed_sums = imputed_counts.sum(axis=1).astype(np.float32)

    raw_scale = np.divide(
        imputed_sums,
        libsize_raw,
        out=np.zeros_like(imputed_sums),
        where=libsize_raw != 0,
    )
    raw_scaled = X_arr * raw_scale[:, None]
    np.maximum(raw_scaled, 0.0, out=raw_scaled)

    raw_neighbor_scaled_corr = np.empty_like(raw_neighbor_mean_corr, dtype=np.float32)
    raw_neighbor_scaled_euclid = np.empty_like(raw_neighbor_mean_euclid, dtype=np.float32)
    mask_nonzero_corr = raw_neighbor_sum_corr > 0.0
    mask_nonzero_euclid = raw_neighbor_sum_euclid > 0.0

    if np.any(mask_nonzero_corr):
        scaling_corr = imputed_sums[mask_nonzero_corr][:, None] / raw_neighbor_sum_corr[mask_nonzero_corr][:, None]
        raw_neighbor_scaled_corr[mask_nonzero_corr] = raw_neighbor_mean_corr[mask_nonzero_corr] * scaling_corr
    raw_neighbor_scaled_corr[~mask_nonzero_corr] = raw_scaled[~mask_nonzero_corr]

    if np.any(mask_nonzero_euclid):
        scaling_euclid = imputed_sums[mask_nonzero_euclid][:, None] / raw_neighbor_sum_euclid[mask_nonzero_euclid][:, None]
        raw_neighbor_scaled_euclid[mask_nonzero_euclid] = raw_neighbor_mean_euclid[mask_nonzero_euclid] * scaling_euclid
    raw_neighbor_scaled_euclid[~mask_nonzero_euclid] = raw_scaled[~mask_nonzero_euclid]

    raw_neighbor_scaled_total = (
        raw_neighbor_scaled_corr + raw_neighbor_scaled_euclid
    ) * 0.5

    sigma_gene = np.clip(1.0 - dropout_frac_gene, sigma_low, sigma_high)
    raw_combined = sigma_gene * raw_scaled + (1.0 - sigma_gene) * raw_neighbor_scaled_total
    np.maximum(raw_combined, 0.0, out=raw_combined)

    rel_lib = libsize_raw / libsize_raw.mean()
    rel_lib = np.clip(rel_lib, 0.0, 1.0)
    alpha_cell = alpha_base + (1.0 - alpha_base) * (1.0 - rel_lib)
    alpha_cell = np.clip(alpha_cell, 0.0, 1.0)
    diff_mix = alpha_cell[:, None] * imputed_counts + (1.0 - alpha_cell)[:, None] * raw_combined
    np.maximum(diff_mix, 0.0, out=diff_mix)

    k_svd = int(min(cfg.get("n_components_svd", 0), svd_components.shape[0]))
    if k_svd > 0:
        proj_k = X_svd_proj[:, :k_svd]
        lr_sqrt = proj_k @ svd_components[:k_svd, :]
        np.maximum(lr_sqrt, 0.0, out=lr_sqrt)
        lr_counts = lr_sqrt ** 2
        lr_sums = lr_counts.sum(axis=1).astype(np.float32)
        lr_scale = np.divide(
            imputed_sums,
            lr_sums,
            out=np.zeros_like(imputed_sums),
            where=lr_sums != 0,
        )
        lr_scaled_svd = lr_counts * lr_scale[:, None]
        np.maximum(lr_scaled_svd, 0.0, out=lr_scaled_svd)
    else:
        lr_scaled_svd = np.zeros_like(imputed_counts, dtype=np.float32)

    k_nmf = int(min(cfg.get("n_components_nmf", 0), W_nmf.shape[1]))
    if k_nmf > 0:
        W_k = W_nmf[:, :k_nmf]
        H_k = H_nmf[:k_nmf, :]
        lr_counts_nmf = (W_k @ H_k) ** 2
        lr_sums_nmf = lr_counts_nmf.sum(axis=1).astype(np.float32)
        lr_scale_nmf = np.divide(
            imputed_sums,
            lr_sums_nmf,
            out=np.zeros_like(imputed_sums),
            where=lr_sums_nmf != 0,
        )
        lr_scaled_nmf = lr_counts_nmf * lr_scale_nmf[:, None]
        np.maximum(lr_scaled_nmf, 0.0, out=lr_scaled_nmf)
    else:
        lr_scaled_nmf = np.zeros_like(imputed_counts, dtype=np.float32)

    denom = w_svd + w_nmf
    if denom > 0:
        lr_mix = (w_svd * lr_scaled_svd + w_nmf * lr_scaled_nmf) / denom
    else:
        lr_mix = np.zeros_like(lr_scaled_svd)
    np.maximum(lr_mix, 0.0, out=lr_mix)

    diff_weight = beta_base + (1.0 - beta_base) * (1.0 - rel_lib)
    diff_weight = np.clip(diff_weight, 0.0, 1.0)
    lr_weight = 1.0 - diff_weight
    denoised = diff_weight[:, None] * diff_mix + lr_weight[:, None] * lr_mix
    np.maximum(denoised, 0.0, out=denoised)

    raw_gene_mean = X_arr.mean(axis=0, dtype=np.float32)
    denoised_gene_mean = denoised.mean(axis=0, dtype=np.float32)
    eps = 1e-12
    scale_factors = raw_gene_mean / (denoised_gene_mean + eps)
    scale_factors = np.clip(scale_factors, clip_min, clip_max)
    denoised = denoised * scale_factors[None, :]
    np.maximum(denoised, 0.0, out=denoised)

    final_sum = denoised.sum(axis=1).astype(np.float32)
    final_scale = np.divide(
        libsize_raw,
        final_sum,
        out=np.ones_like(final_sum),
        where=final_sum != 0,
    )
    final_scale = np.clip(final_scale, 0.95, 1.05)
    denoised = denoised * final_scale[:, None]
    np.maximum(denoised, 0.0, out=denoised)

    return denoised.astype(np.float32)


def _pca_imputed(X_arr, X_svd_proj, svd_components, libsize_raw):
    """Simple PCA‑based imputation with library‑size scaling."""
    imputed_counts = X_svd_proj @ svd_components
    imputed_counts = np.maximum(imputed_counts, 0.0)
    imputed_sums = imputed_counts.sum(axis=1).astype(np.float32)
    mask_zero = imputed_sums == 0
    scaling = np.ones_like(imputed_sums, dtype=np.float32)
    scaling[~mask_zero] = libsize_raw[~mask_zero] / imputed_sums[~mask_zero]
    imputed_scaled = imputed_counts * scaling[:, None]
    return imputed_scaled.astype(np.float32)


def magic_denoise(X, **kwargs):
    """
    Diffusion‑based scRNA‑seq denoiser with ensemble of diffusion, low‑rank,
    and k‑NN smoothers.  The implementation merges several strategies,
    selects a weighted combination guided by Poisson and MSE on a training
    split, refines the best candidate with an extra diffusion step,
    and applies careful library‑size and gene‑wise scaling.
    """
    knn = int(kwargs.get("knn", 20))
    n_pca = int(kwargs.get("n_pca", 200))
    n_svd = int(kwargs.get("n_svd", 80))
    n_nmf = int(kwargs.get("n_nmf", 80))
    n_jobs = int(kwargs.get("n_jobs", 8))
    random_state = kwargs.get("random_state", None)

    exponent_list = kwargs.get("exponent_list", [4, 6, 8, 10])
    refine_weights_list = kwargs.get("refine_weights_list", [0.0, 0.02, 0.04, 0.06, 0.08])

    if random_state is None:
        random_state = _seed_from_global()
    else:
        try:
            random_state = int(random_state)
        except Exception:
            random_state = _seed_from_global()
    if random_state is None:
        random_state = 0

    X_arr = scprep.utils.toarray(X).astype(np.float32, copy=False)
    n_cells, n_genes = X_arr.shape
    if n_cells <= 1 or n_genes == 0:
        return np.empty_like(X_arr, dtype=np.float64)

    libsize_raw = X_arr.sum(axis=1).astype(np.float32)
    libsize_raw[libsize_raw == 0] = 1.0

    X_sqrt = np.sqrt(X_arr, dtype=np.float32)

    X_norm, libsize_sqrt = scprep.normalize.library_size_normalize(
        X_sqrt,
        rescale=1.0,
        return_library_size=True,
    )
    libsize_sqrt = libsize_sqrt.astype(np.float32, copy=False)
    libsize_sqrt[libsize_sqrt == 0] = 1.0
    X_norm = X_norm.astype(np.float32)

    X_log = np.log1p(X_arr).astype(np.float32)

    n_pca_opt = min(n_pca, n_genes, n_cells - 1)
    if n_pca_opt > 0:
        pca_graph = TruncatedSVD(
            n_components=n_pca_opt,
            n_iter=5,
            algorithm="randomized",
            random_state=random_state,
        )
        X_graph = pca_graph.fit_transform(X_sqrt.astype(np.float32)).astype(np.float32)
    else:
        X_graph = X_norm

    diff_op = _construct_diffusion_operator(X_graph, knn, n_jobs=n_jobs)
    raw_neighbor_mean_corr = diff_op.dot(X_arr.astype(np.float32))
    raw_neighbor_sum_corr = raw_neighbor_mean_corr.sum(axis=1).astype(np.float32)

    diff_op_euclid = _construct_diffusion_operator(
        X_sqrt, knn, metric="euclidean", n_jobs=n_jobs
    )
    raw_neighbor_mean_euclid = diff_op_euclid.dot(X_arr.astype(np.float32))
    raw_neighbor_sum_euclid = raw_neighbor_mean_euclid.sum(axis=1).astype(np.float32)

    diff_op_log = _construct_diffusion_operator(X_log, knn, n_jobs=n_jobs)

    max_svd = min(n_svd, n_cells - 1, n_genes)
    svd_low = TruncatedSVD(
        n_components=max_svd,
        n_iter=5,
        algorithm="randomized",
        random_state=random_state,
    )
    X_svd_proj = svd_low.fit_transform(X_sqrt.astype(np.float32)).astype(np.float32)
    svd_components = svd_low.components_.astype(np.float32)

    max_nmf = min(n_nmf, n_cells, n_genes)
    nmf_low = NMF(
        n_components=max_nmf,
        init="nndsvda",
        solver="cd",
        max_iter=200,
        random_state=random_state,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        W_nmf = nmf_low.fit_transform(X_sqrt.astype(np.float32)).astype(np.float32)
        H_nmf = nmf_low.components_.astype(np.float32)

    dropout_frac_gene = (X_arr == 0).astype(np.float32).sum(axis=0) / n_cells

    # Base denoising configurations
    base_configs = [
        {
            "t": [12, 18, 24],
            "lam": [1.6, 1.9, 2.0],
            "weights": [0.05, 0.90, 0.05],
            "threshold": 0.010,
            "alpha_base": 0.86,
            "beta_base": 0.86,
            "sigma_low": 0.28,
            "sigma_high": 0.82,
            "w_svd": 0.55,
            "w_nmf": 0.45,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
        {
            "t": [14, 20, 26],
            "lam": [1.6, 1.9, 2.1],
            "weights": [0.05, 0.90, 0.05],
            "threshold": 0.015,
            "alpha_base": 0.86,
            "beta_base": 0.86,
            "sigma_low": 0.30,
            "sigma_high": 0.84,
            "w_svd": 0.50,
            "w_nmf": 0.50,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
        {
            "t": [10, 15, 20],
            "lam": [1.4, 1.8, 2.0],
            "weights": [0.07, 0.86, 0.07],
            "threshold": 0.020,
            "alpha_base": 0.82,
            "beta_base": 0.84,
            "sigma_low": 0.30,
            "sigma_high": 0.85,
            "w_svd": 0.60,
            "w_nmf": 0.40,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
        {
            "t": [8, 12, 16],
            "lam": [1.3, 1.6, 1.9],
            "weights": [0.10, 0.80, 0.10],
            "threshold": 0.015,
            "alpha_base": 0.88,
            "beta_base": 0.88,
            "sigma_low": 0.35,
            "sigma_high": 0.80,
            "w_svd": 0.60,
            "w_nmf": 0.40,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
        {
            "t": [8, 16, 24],
            "lam": [1.3, 1.7, 2.1],
            "weights": [0.10, 0.80, 0.10],
            "threshold": 0.018,
            "alpha_base": 0.83,
            "beta_base": 0.83,
            "sigma_low": 0.35,
            "sigma_high": 0.81,
            "w_svd": 0.45,
            "w_nmf": 0.55,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
        {"method": "raw_diff"},
        {"method": "nn_mean"},
        {"method": "log_diff", "t": 8, "lam": 0.5, "threshold": 0.01},
        {"method": "raw_neighbor_mean_log"},
        # Additional config with slightly different diff parameters
        {
            "t": [12, 20, 28],
            "lam": [1.6, 1.9, 2.2],
            "weights": [0.08, 0.84, 0.08],
            "threshold": 0.02,
            "alpha_base": 0.86,
            "beta_base": 0.86,
            "sigma_low": 0.30,
            "sigma_high": 0.84,
            "w_svd": 0.50,
            "w_nmf": 0.50,
            "clip_min": 0.94,
            "clip_max": 1.06,
            "n_components_svd": max_svd,
            "n_components_nmf": max_nmf,
        },
    ]

    denoised_list = []
    poisson_losses = []
    mse_losses = []

    magic_matrix = None
    if _MAGIC_AVAILABLE:
        try:
            magic_model = MAGIC()
            magic_matrix = magic_model.fit_transform(X_arr)
            if sparse.issparse(magic_matrix):
                magic_matrix = magic_matrix.toarray()
            magic_matrix = np.asarray(magic_matrix, dtype=np.float32)
        except Exception:
            magic_matrix = None

    rng_val = np.random.RandomState(random_state)
    n_val = max(1, int(np.floor(n_cells * 0.1)))
    val_idx = rng_val.choice(n_cells, size=n_val, replace=False)

    for cfg in base_configs:
        den = _denoise_pipeline(
            cfg,
            X_arr,
            X_norm,
            X_log,
            diff_op,
            diff_op_log,
            libsize_raw,
            libsize_sqrt,
            X_svd_proj,
            svd_components,
            W_nmf,
            H_nmf,
            dropout_frac_gene,
            raw_neighbor_mean_corr,
            raw_neighbor_sum_corr,
            raw_neighbor_mean_euclid,
            raw_neighbor_sum_euclid,
        )
        denoised_list.append(den)

        po_loss = _poisson_val_loss(den, X_arr)
        mse_val = _mse_val_loss(den, X_arr, val_idx)
        poisson_losses.append(po_loss)
        mse_losses.append(mse_val)

        del den
        gc.collect()

    # additional raw diffusion with different k values
    for k in (10, 30):
        diff_op_k = _construct_diffusion_operator(X_graph, k, n_jobs=n_jobs)
        den_k = diff_op_k.dot(X_arr.astype(np.float32))
        den_k = np.maximum(den_k, 0.0).astype(np.float32)
        denoised_list.append(den_k)
        poisson_losses.append(_poisson_val_loss(den_k, X_arr))
        mse_losses.append(_mse_val_loss(den_k, X_arr, val_idx))

    # PCA‑imputed denoiser
    pca_den = _pca_imputed(X_arr, X_svd_proj, svd_components, libsize_raw)
    denoised_list.append(pca_den)
    poisson_losses.append(_poisson_val_loss(pca_den, X_arr))
    mse_losses.append(_mse_val_loss(pca_den, X_arr, val_idx))

    # Optional magic denoiser
    if magic_matrix is not None:
        denoised_list.append(magic_matrix)
        poisson_losses.append(_poisson_val_loss(magic_matrix, X_arr))
        mse_losses.append(_mse_val_loss(magic_matrix, X_arr, val_idx))

    poisson_losses = np.array(poisson_losses, dtype=np.float32)
    mse_losses = np.array(mse_losses, dtype=np.float32)

    eps_w = 1e-12
    poisson_weights = 1.0 / (poisson_losses + eps_w)
    mse_weights = 1.0 / (mse_losses + eps_w)

    threshold_poisson = 0.257575 - 0.97 * (0.257575 - 0.031739)

    raw_gene_mean = X_arr.mean(axis=0, dtype=np.float32)

    def _generate_candidate(exp, refine_ws):
        combined_weights = poisson_weights * (mse_weights ** exp)
        combined_weights = np.maximum(combined_weights, 0.0)
        if combined_weights.sum() > 0:
            combined_weights /= combined_weights.sum()
        else:
            combined_weights = np.ones_like(combined_weights) / len(combined_weights)

        combined = np.zeros_like(X_arr, dtype=np.float32)
        for w, den in zip(combined_weights, denoised_list):
            combined += w * den.astype(np.float32)
        np.maximum(combined, 0.0, out=combined)

        sqrt_comb = np.sqrt(combined + 1e-12).astype(np.float32)
        smooth_sqrt = _weighted_average_diffusion(
            sqrt_comb, diff_op, t=8, decay=2.4,
        )
        refined_counts = (smooth_sqrt ** 2).astype(np.float32)
        combined_output = (1 - 0.04) * combined + 0.04 * refined_counts
        np.maximum(combined_output, 0.0, out=combined_output)

        cand_gene_mean = combined_output.mean(axis=0, dtype=np.float32)
        scale_factors = raw_gene_mean / (cand_gene_mean + 1e-12)
        scale_factors = np.clip(scale_factors, 0.99, 1.01)
        cand_scaled = combined_output * scale_factors[None, :]
        np.maximum(cand_scaled, 0.0, out=cand_scaled)

        best_out = cand_scaled.copy()
        best_mse_val = _mse_val_loss(best_out, X_arr, val_idx)
        best_poisson_val = _poisson_val_loss(best_out, X_arr)

        for w_ref in refine_ws:
            smooth_tmp = _weighted_average_diffusion(
                np.sqrt(cand_scaled + 1e-12).astype(np.float32),
                diff_op,
                t=8,
                decay=2.4,
            )
            refined_tmp = (smooth_tmp ** 2).astype(np.float32)
            cand = (1 - w_ref) * cand_scaled + w_ref * refined_tmp
            np.maximum(cand, 0.0, out=cand)

            cand_gene_mean = cand.mean(axis=0, dtype=np.float32)
            cand_scale_factors = raw_gene_mean / (cand_gene_mean + 1e-12)
            cand_scale_factors = np.clip(cand_scale_factors, 0.99, 1.01)
            cand_scaled = cand * cand_scale_factors[None, :]
            np.maximum(cand_scaled, 0.0, out=cand_scaled)

            cand_mse_val = _mse_val_loss(cand_scaled, X_arr, val_idx)
            cand_poisson_val = _poisson_val_loss(cand_scaled, X_arr)
            if cand_poisson_val <= threshold_poisson and cand_mse_val < best_mse_val:
                best_mse_val = cand_mse_val
                best_out = cand_scaled.copy()
                best_poisson_val = cand_poisson_val
        return best_out, best_mse_val, best_poisson_val

    best_final = None
    best_final_mse = float("inf")

    for exp in exponent_list:
        cand_output, cand_mse, _ = _generate_candidate(exp, refine_weights_list)
        if cand_mse < best_final_mse:
            best_final = cand_output.copy()
            best_final_mse = cand_mse

    if best_final is None:
        best_final, _, _ = _generate_candidate(exponent_list[0], refine_weights_list)

    # Final cell and gene scaling
    cell_scale = libsize_raw / (best_final.sum(axis=1) + 1e-12)
    cell_scale = np.clip(cell_scale, 0.95, 1.05)
    best_final = best_final * cell_scale[:, None]
    np.maximum(best_final, 0.0, out=best_final)

    mask_nonzero = X_arr > 0
    gene_nonzero_counts = mask_nonzero.sum(axis=0).astype(np.float32)
    raw_nonzero_sum = (X_arr * mask_nonzero).sum(axis=0).astype(np.float32)
    raw_nonzero_mean = raw_nonzero_sum / np.maximum(gene_nonzero_counts, 1.0)

    denoised_nonzero_sum = (best_final * mask_nonzero).sum(axis=0).astype(np.float32)
    denoised_nonzero_mean = denoised_nonzero_sum / np.maximum(gene_nonzero_counts, 1.0)

    scale_factors = np.where(
        denoised_nonzero_mean > 1e-6,
        raw_nonzero_mean / denoised_nonzero_mean,
        1.0,
    )
    scale_factors = np.clip(scale_factors, 0.95, 1.05)
    best_final = best_final * scale_factors[None, :]
    np.maximum(best_final, 0.0, out=best_final)

    return best_final.astype(np.float64)
# EVOLVE-BLOCK-END