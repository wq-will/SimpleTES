# EVOLVE-BLOCK-START
import os
import time
import numpy as np

# ---------- optional imports ----------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from sklearn.decomposition import TruncatedSVD
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
# -------------------------------------

def _segment_starts(cond_lengths, context_len, horizon, stride=1):
    """Return start indices of windows that stay fully inside each condition."""
    starts = []
    offset = 0
    for seg_len in cond_lengths:
        seg_len = int(seg_len)
        max_start = seg_len - context_len - horizon + 1
        if max_start > 0:
            starts.append(
                np.arange(offset, offset + max_start, stride, dtype=np.int64)
            )
        offset += seg_len
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def _persistence(predict_inputs, horizon):
    """Fallback: repeat the last observed frame."""
    last = predict_inputs[:, -1:, :]          # (N,1,F)
    return np.repeat(last, horizon, axis=1).astype(np.float32)


def _baseline_trend_tensor(ctx, horizon):
    """
    Linear‑trend baseline.
    ctx: (B, C, F) torch.float32
    Returns: (B, horizon, F) torch.float32
    """
    B, C, F = ctx.shape
    first = ctx[:, 0:1, :]               # (B,1,F)
    last = ctx[:, -1:, :]                # (B,1,F)
    steps = torch.arange(1, horizon + 1, device=ctx.device,
                         dtype=ctx.dtype).view(1, horizon, 1)  # (1,H,1)
    slope = (last - first) / float(C - 1)                 # (B,1,F)
    return first + slope * steps                         # (B, H, F)


class _ResidualBlock(nn.Module):
    """Linear → BatchNorm → Dropout → ReLU (+skip)."""

    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.linear = nn.Linear(units, units, bias=True)
        self.bn = nn.BatchNorm1d(units)
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        h = self.bn(h)
        h = self.dropout(h)
        return self.act(h + x)


class _NeuronForecaster(nn.Module):
    """
    Shared‑weight per‑neuron forecaster.
    Predicts deltas over the horizon.
    """

    def __init__(
        self,
        num_neurons,
        static_dim,
        context_len=4,
        horizon=32,
        global_dim=128,
        embed_dim=64,
        hidden=768,
        n_blocks=6,
        dropout=0.1,
    ):
        super().__init__()
        self.F = num_neurons
        self.C = context_len
        self.H = horizon
        self.global_dim = global_dim
        self.embed_dim = embed_dim
        self.static_dim = static_dim

        # Global brain‑state projection (per time‑step, then mean‑pooled)
        self.global_dense = nn.Linear(num_neurons, global_dim, bias=True)

        # Neuron identity embedding
        self.neuron_embed = nn.Embedding(num_neurons, embed_dim)

        # Number of handcrafted rolling statistics
        self.n_stats = 8

        # Input dimension calculation
        self.in_dim = (
            2 * self.C            # raw + raw_z
            + (self.C - 1)        # first‑order diffs
            + 1                   # slope
            + self.n_stats
            + self.global_dim
            + 2 * self.C          # per‑step global mean & std
            + self.embed_dim
            + self.static_dim
        )
        self.in_bn = nn.BatchNorm1d(self.in_dim)

        self.proj = nn.Linear(self.in_dim, hidden, bias=True)
        self.proj_act = nn.ReLU()
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden, horizon, bias=True)

    @staticmethod
    def _context_stats(raw):
        """Eight simple statistics over the context window (raw shape (M, C))."""
        mean = raw.mean(dim=1, keepdim=True)
        std = raw.std(dim=1, unbiased=False, keepdim=True)
        mn = raw.min(dim=1, keepdim=True).values
        mx = raw.max(dim=1, keepdim=True).values
        last = raw[:, -1:].clone()
        first = raw[:, :1].clone()
        srt = torch.sort(raw, dim=1).values
        c = raw.shape[1]
        if c % 2 == 0:
            median = 0.5 * (srt[:, c // 2 - 1 : c // 2] + srt[:, c // 2 : c // 2 + 1])
        else:
            median = srt[:, c // 2 : c // 2 + 1]
        rng = mx - mn
        return torch.cat([mean, std, mn, mx, last, first, median, rng], dim=1)

    def _global_state(self, ctx):
        """Average‑pooled global brain state."""
        B, C, F = ctx.shape
        x = ctx.reshape(B * C, F)               # (B*C, F)
        g = self.global_dense(x)                # (B*C, G)
        g = g.view(B, C, self.global_dim).mean(dim=1)  # (B, G)
        return g

    def forward(self, ctx, neuron_idx, static_feat):
        """
        ctx: (B, C, F)
        neuron_idx: (F,)
        static_feat: (F, static_dim)
        Returns: (B, F, H) delta predictions.
        """
        B, C, F = ctx.shape

        # raw values per neuron: (B*F, C)
        raw = ctx.permute(0, 2, 1).reshape(B * F, C)

        # Z‑score using static mean/std (first two entries)
        sf = static_feat[neuron_idx]                       # (F, static_dim)
        sf_exp = sf[None, :, :].expand(B, F, self.static_dim)
        sf_exp = sf_exp.reshape(B * F, self.static_dim)
        mean_neu = sf_exp[:, 0:1]
        std_neu = sf_exp[:, 1:2] + 1e-6
        raw_z = (raw - mean_neu) / std_neu                 # (B*F, C)

        # First‑order diffs
        diffs = raw[:, 1:] - raw[:, :-1]                   # (B*F, C-1)

        # Slope of the 4‑step window
        slope = ((raw[:, -1] - raw[:, 0]) / float(C - 1)).unsqueeze(1)  # (B*F,1)

        # Hand‑crafted stats (8)
        stats = self._context_stats(raw)                  # (B*F, 8)

        # Learned global brain‑state broadcast
        g = self._global_state(ctx)                       # (B, G)
        g = g[:, None, :].expand(B, F, self.global_dim)
        g = g.reshape(B * F, self.global_dim)             # (B*F, G)

        # Per‑step global mean & std (2*C)
        g_mean = ctx.mean(dim=2)                          # (B, C)
        g_std = ctx.std(dim=2, unbiased=False)            # (B, C)
        g_stats = torch.cat([g_mean, g_std], dim=1)       # (B, 2*C)
        g_stats = g_stats[:, None, :].expand(B, F, 2 * C)
        g_stats = g_stats.reshape(B * F, 2 * C)           # (B*F, 2*C)

        # Neuron identity embedding
        emb = self.neuron_embed(neuron_idx)               # (F, E)
        emb = emb[None, :, :].expand(B, F, self.embed_dim)
        emb = emb.reshape(B * F, self.embed_dim)          # (B*F, E)

        # Concatenate all features
        feats = torch.cat(
            [raw, raw_z, diffs, slope, stats, g, g_stats, emb, sf_exp],
            dim=1,
        )  # (B*F, in_dim)

        # Feed‑forward tower
        x = self.in_bn(feats)
        x = self.proj_act(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x)                                 # (B*F, H)
        return out.reshape(B, F, self.H)                   # (B, F, H)


def forecast(
    X_train,
    predict_inputs,
    X_train_condition_lengths=None,
    random_state=None,
    time_budget_s=None,
    num_train_conditions=8,
    **kwargs,
):
    """
    Train a shared‑weight per‑neuron MLP forecaster with rich static features,
    a learned global brain‑state and a simple linear‑trend baseline.  Returns
    predictions for the next 32 steps of each window in ``predict_inputs``.
    """
    # ----------------------- hyper‑parameters -----------------------
    CONTEXT_LEN = int(kwargs.get("context_len", 4))
    HORIZON = int(kwargs.get("horizon", 32))

    GLOBAL_DIM = int(kwargs.get("global_dim", 128))
    EMBED_DIM = int(kwargs.get("embed_dim", 64))
    HIDDEN = int(kwargs.get("hidden", 768))
    N_BLOCKS = int(kwargs.get("n_blocks", 6))
    DROPOUT = float(kwargs.get("dropout", 0.1))

    LR = float(kwargs.get("lr", 3e-4))
    WEIGHT_DECAY = float(kwargs.get("weight_decay", 1e-5))
    MAX_EPOCHS = int(kwargs.get("max_epochs", 80))

    WIN_BATCH = int(kwargs.get("win_batch", 4))
    VAL_FRACTION = float(kwargs.get("val_fraction", 0.1))
    EARLY_STOP_PATIENCE = int(kwargs.get("early_stop_patience", 12))
    PRED_WIN_BATCH = int(kwargs.get("pred_win_batch", 8))
    TRAIN_STRIDE = int(kwargs.get("train_stride", 1))
    TRAIN_TIME_FRACTION = float(kwargs.get("train_time_fraction", 0.9))
    MAX_TRAIN_WINDOWS = int(kwargs.get("max_train_windows", 0))   # 0 → no cap

    PCA_COMPONENTS = int(kwargs.get("pca_components", 16))       # cheap low‑dim embedding

    # -------------------- basic preparation --------------------
    rs = 0 if random_state is None else int(random_state)
    rng = np.random.RandomState(rs)
    start_time = time.time()
    budget = float(time_budget_s) if time_budget_s else None

    X_train = np.asarray(X_train, dtype=np.float32)
    predict_inputs = np.asarray(predict_inputs, dtype=np.float32)

    T, F = X_train.shape
    N_pred = predict_inputs.shape[0]

    # simple fallback (always valid)
    fallback = _persistence(predict_inputs, HORIZON)
    if not _TORCH_AVAILABLE:
        return fallback

    # ------------------- condition handling -------------------
    if X_train_condition_lengths is None:
        cond_lengths = np.array([T], dtype=np.int64)
    else:
        cond_lengths = np.asarray(X_train_condition_lengths, dtype=np.int64)

    # ------------------- static neuron features -------------------
    neuron_mean = X_train.mean(axis=0)                # (F,)
    neuron_std = X_train.std(axis=0)                  # (F,)

    # global coupling (correlation with brain‑wide mean)
    global_ts = X_train.mean(axis=1)                  # (T,)
    global_centered = global_ts - global_ts.mean()
    cov = (X_train.T @ global_centered) / T          # (F,)
    var_g = (global_centered ** 2).mean()
    global_corr = cov / (np.sqrt(neuron_std ** 2 * var_g) + 1e-6)  # (F,)

    # optional cheap PCA on neurons (unsupervised)
    if PCA_COMPONENTS > 0 and _SKLEARN_AVAILABLE:
        # sample a subset of time points to keep memory modest
        n_sub = min(2000, T)
        sub_idx = rng.choice(T, size=n_sub, replace=False)
        X_sub = X_train[sub_idx]                      # (n_sub, F)
        svd = TruncatedSVD(n_components=PCA_COMPONENTS, random_state=rs)
        svd.fit(X_sub)
        pca_feat = svd.components_.T.astype(np.float32)   # (F, K)
    else:
        pca_feat = np.empty((F, 0), dtype=np.float32)

    static_feat_np = np.column_stack(
        (neuron_mean, neuron_std, global_corr, pca_feat)
    ).astype(np.float32)                                 # (F, static_dim)

    static_dim = static_feat_np.shape[1]

    # ------------------------ device setup ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(rs)
    np.random.seed(rs)
    try:
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    except Exception:
        pass

    # --------------------- model & optimizer ---------------------
    model = _NeuronForecaster(
        num_neurons=F,
        static_dim=static_dim,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        global_dim=GLOBAL_DIM,
        embed_dim=EMBED_DIM,
        hidden=HIDDEN,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    loss_fn = nn.L1Loss()
    neuron_idx = torch.arange(F, dtype=torch.long, device=device)
    static_feat = torch.from_numpy(static_feat_np).to(device)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ---------------------- helper: time left ----------------------
    def _time_left():
        if budget is None:
            return True
        elapsed = time.time() - start_time
        return elapsed < TRAIN_TIME_FRACTION * budget

    # ------------------- training / validation windows ----------
    starts = _segment_starts(
        cond_lengths, CONTEXT_LEN, HORIZON, stride=TRAIN_STRIDE
    )
    if starts.size == 0:
        return fallback

    if MAX_TRAIN_WINDOWS > 0 and starts.shape[0] > MAX_TRAIN_WINDOWS:
        starts = rng.choice(starts, size=MAX_TRAIN_WINDOWS, replace=False)
    rng.shuffle(starts)

    n_val = max(1, int(len(starts) * VAL_FRACTION)) if len(starts) > 10 else 0
    val_starts = starts[:n_val]
    train_starts = starts[n_val:]

    if train_starts.size == 0:
        return fallback

    offs_c = np.arange(CONTEXT_LEN, dtype=np.int64)                       # (C,)
    offs_h = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON, dtype=np.int64)  # (H,)

    def _batch(starts_arr, i0, i1):
        sel = starts_arr[i0:i1]
        ctx = X_train[sel[:, None] + offs_c]   # (b, C, F)
        tgt = X_train[sel[:, None] + offs_h]   # (b, H, F)
        return ctx, tgt

    # ------------------ validation MAE (full prediction) -------------
    def _val_mae():
        if val_starts.size == 0:
            return None
        model.eval()
        total_err = 0.0
        total_cnt = 0
        with torch.no_grad():
            for i in range(0, len(val_starts), WIN_BATCH):
                ctx_np, tgt_np = _batch(val_starts, i, i + WIN_BATCH)
                ctx = torch.from_numpy(ctx_np).to(device)               # (b, C, F)
                tgt = torch.from_numpy(tgt_np).to(device)               # (b, H, F)

                last = ctx[:, -1:, :]                                   # (b,1,F)

                if scaler:
                    with torch.cuda.amp.autocast():
                        delta_pred = model(ctx, neuron_idx, static_feat)  # (b, F, H)
                else:
                    delta_pred = model(ctx, neuron_idx, static_feat)

                pred = delta_pred + last.squeeze(1).unsqueeze(2)        # (b, F, H)
                pred = pred.permute(0, 2, 1)                           # (b, H, F)

                err = torch.abs(pred - tgt)
                total_err += err.sum().item()
                total_cnt += err.numel()
        return total_err / max(total_cnt, 1)

    # -------------------------- training loop -------------------------
    best_val = float("inf")
    best_state = None
    patience = 0
    n_train = len(train_starts)

    for epoch in range(MAX_EPOCHS):
        if not _time_left():
            break
        model.train()
        perm = rng.permutation(n_train)
        shuffled = train_starts[perm]

        for i in range(0, n_train, WIN_BATCH):
            if not _time_left():
                break
            ctx_np, tgt_np = _batch(shuffled, i, i + WIN_BATCH)
            ctx = torch.from_numpy(ctx_np).to(device)               # (b, C, F)
            tgt = torch.from_numpy(tgt_np).to(device)               # (b, H, F)

            last = ctx[:, -1:, :]                                   # (b,1,F)
            delta_tgt = (tgt - last).permute(0, 2, 1)               # (b, F, H)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    delta_pred = model(ctx, neuron_idx, static_feat)  # (b, F, H)
                    loss = loss_fn(delta_pred, delta_tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                delta_pred = model(ctx, neuron_idx, static_feat)
                loss = loss_fn(delta_pred, delta_tgt)
                loss.backward()
                optimizer.step()

        scheduler.step()
        vm = _val_mae()
        if vm is not None and vm < best_val - 1e-6:
            best_val = vm
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    # ---------------------- load best checkpoint --------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------- compute horizon‑wise blending weight ----------
    if val_starts.size > 0:
        # accumulate numerator & denominator for closed‑form optimal weight
        num = np.zeros(HORIZON, dtype=np.float64)
        den = np.zeros(HORIZON, dtype=np.float64)

        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_starts), WIN_BATCH):
                ctx_np, tgt_np = _batch(val_starts, i, i + WIN_BATCH)
                ctx = torch.from_numpy(ctx_np).to(device)           # (b, C, F)
                tgt = torch.from_numpy(tgt_np).to(device)           # (b, H, F)

                baseline = _baseline_trend_tensor(ctx, HORIZON)     # (b, H, F)

                if scaler:
                    with torch.cuda.amp.autocast():
                        delta_pred = model(ctx, neuron_idx, static_feat)  # (b, F, H)
                else:
                    delta_pred = model(ctx, neuron_idx, static_feat)

                pred = delta_pred + ctx[:, -1:, :].squeeze(1).unsqueeze(2)   # (b, F, H)
                pred = pred.permute(0, 2, 1)                                 # (b, H, F)

                diff = pred - baseline
                y = tgt - baseline

                diff_np = diff.cpu().numpy()
                y_np = y.cpu().numpy()
                num += np.einsum('bhf,bhf->h', diff_np, y_np)
                den += np.einsum('bhf,bhf->h', diff_np, diff_np)

        w_opt = np.where(den > 1e-12, num / den, 0.0)
        w_opt = np.clip(w_opt, 0.0, 1.0).astype(np.float32)   # (H,)
    else:
        w_opt = np.ones(HORIZON, dtype=np.float32)

    # ----------------------------- inference ---------------------------
    model.eval()
    Y_pred = np.empty((N_pred, HORIZON, F), dtype=np.float32)
    w_tensor = torch.from_numpy(w_opt).to(device).view(1, HORIZON, 1)  # (1,H,1)

    with torch.no_grad():
        for i in range(0, N_pred, PRED_WIN_BATCH):
            ctx_np = predict_inputs[i : i + PRED_WIN_BATCH]          # (b, C, F)
            ctx = torch.from_numpy(ctx_np).to(device)                # (b, C, F)

            baseline = _baseline_trend_tensor(ctx, HORIZON)          # (b, H, F)

            if scaler:
                with torch.cuda.amp.autocast():
                    delta_pred = model(ctx, neuron_idx, static_feat)   # (b, F, H)
            else:
                delta_pred = model(ctx, neuron_idx, static_feat)       # (b, F, H)

            pred = delta_pred + ctx[:, -1:, :].squeeze(1).unsqueeze(2)   # (b, F, H)
            pred = pred.permute(0, 2, 1)                                 # (b, H, F)

            blended = baseline + w_tensor * (pred - baseline)           # (b, H, F)
            Y_pred[i : i + blended.shape[0]] = blended.cpu().numpy()

    # ---------------------------- safety net -------------------------
    if not np.isfinite(Y_pred).all():
        return fallback

    return Y_pred.astype(np.float32)
# EVOLVE-BLOCK-END
