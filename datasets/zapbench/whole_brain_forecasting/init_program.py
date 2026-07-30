# EVOLVE-BLOCK-START
"""Initial shared-weight MLP forecaster."""
import os
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

# ----------------------------------------------------------------------
# Window construction (condition-aware)
# ----------------------------------------------------------------------
def _segment_starts(cond_lengths, context_len, horizon, stride=1):
    """Start indices for windows that stay strictly inside each condition."""
    starts = []
    offset = 0
    for seg_len in cond_lengths:
        seg_len = int(seg_len)
        max_start = seg_len - context_len - horizon + 1
        if max_start > 0:
            seg = np.arange(offset, offset + max_start, stride, dtype=np.int64)
            starts.append(seg)
        offset += seg_len
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class _ResidualBlock(nn.Module):
        """Dense -> BN -> Dropout -> (+skip) -> ReLU, input dim == output dim."""

        def __init__(self, units, dropout_rate=0.25):
            super().__init__()
            self.dense = nn.Linear(units, units)
            self.bn = nn.BatchNorm1d(units)
            self.dropout = nn.Dropout(dropout_rate)
            self.act = nn.ReLU()

        def forward(self, x):
            h = self.dense(x)
            h = self.bn(h)
            h = self.dropout(h)
            return self.act(h + x)

    class _SharedNeuronMLP(nn.Module):
        """Shared-weight per-neuron forecaster with a learned global brain state."""

        def __init__(
            self,
            num_neurons,
            context_len=4,
            horizon=32,
            global_dim=64,
            embed_dim=64,
            hidden=256,
            n_blocks=4,
            dropout=0.25,
        ):
            super().__init__()
            self.F = num_neurons
            self.C = context_len
            self.H = horizon
            self.global_dim = global_dim
            self.embed_dim = embed_dim

            # Learned global brain-state: Dense applied to every timestep's full
            # neuron vector, then average-pooled across the C context steps.
            self.global_dense = nn.Linear(num_neurons, global_dim)
            # Per-neuron identity embedding.
            self.neuron_embed = nn.Embedding(num_neurons, embed_dim)

            n_stats = 8
            in_dim = context_len + n_stats + global_dim + embed_dim
            self.in_bn = nn.BatchNorm1d(in_dim)
            self.proj = nn.Linear(in_dim, hidden)
            self.proj_act = nn.ReLU()
            self.blocks = nn.ModuleList(
                [_ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
            )
            self.head = nn.Linear(hidden, horizon)

        @staticmethod
        def _context_stats(ctx_bnc):
            """Rolling stats over the context axis. ctx_bnc: (M, C) -> (M, 8)."""
            mean = ctx_bnc.mean(dim=1, keepdim=True)
            std = ctx_bnc.std(dim=1, unbiased=False, keepdim=True)
            mn = ctx_bnc.min(dim=1, keepdim=True).values
            mx = ctx_bnc.max(dim=1, keepdim=True).values
            last = ctx_bnc[:, -1:]
            first = ctx_bnc[:, :1]
            srt = torch.sort(ctx_bnc, dim=1).values
            c = ctx_bnc.shape[1]
            if c % 2 == 0:
                median = 0.5 * (srt[:, c // 2 - 1:c // 2] + srt[:, c // 2:c // 2 + 1])
            else:
                median = srt[:, c // 2:c // 2 + 1]
            rng = mx - mn
            return torch.cat([mean, std, mn, mx, last, first, median, rng], dim=1)

        def _global_state(self, ctx):
            """ctx: (B, C, F) -> (B, global_dim) average-pooled over C."""
            B, C, F = ctx.shape
            g = self.global_dense(ctx.reshape(B * C, F))      # (B*C, global_dim)
            g = g.reshape(B, C, self.global_dim).mean(dim=1)  # (B, global_dim)
            return g

        def forward(self, ctx, neuron_idx):
            """ctx: (B, C, F) raw context windows.
            neuron_idx: (F,) long. Returns (B, F, H) predictions.
            """
            B, C, F = ctx.shape
            # per-neuron raw context: (B, F, C)
            raw = ctx.permute(0, 2, 1).reshape(B * F, C)
            stats = self._context_stats(raw)                  # (B*F, 8)

            g = self._global_state(ctx)                       # (B, global_dim)
            g = g[:, None, :].expand(B, F, self.global_dim).reshape(B * F, self.global_dim)

            emb = self.neuron_embed(neuron_idx)               # (F, embed_dim)
            emb = emb[None, :, :].expand(B, F, self.embed_dim).reshape(B * F, self.embed_dim)

            feats = torch.cat([raw, stats, g, emb], dim=1)    # (B*F, in_dim)
            x = self.in_bn(feats)
            x = self.proj_act(self.proj(x))
            for blk in self.blocks:
                x = blk(x)
            out = self.head(x)                                # (B*F, H)
            return out.reshape(B, F, self.H)


def _persistence(predict_inputs, horizon):
    """Last-step carry-forward: (N, C, F) -> (N, H, F). Always finite."""
    last = predict_inputs[:, -1:, :]
    return np.repeat(last, horizon, axis=1).astype(np.float32)


def forecast(
    X_train,
    predict_inputs,
    X_train_condition_lengths=None,
    random_state=None,
    time_budget_s=None,
    num_train_conditions=8,
    **kwargs,
):
    """ERA-style shared-weight MLP forecaster.

    Args:
        X_train: (T, F) float32 concatenated training-split slices.
        predict_inputs: (N, 4, F) float32 context windows to forecast forward.
        X_train_condition_lengths: (num_train_conditions,) int — per-condition
            lengths along the time axis; used for boundary-respecting windowing.
        random_state, time_budget_s, num_train_conditions: harness kwargs.

    Returns:
        Y_pred: (N, 32, F) float32.
    """
    CONTEXT_LEN = int(kwargs.get("context_len", 4))
    HORIZON = int(kwargs.get("horizon", 32))

    GLOBAL_DIM = int(kwargs.get("global_dim", 64))
    EMBED_DIM = int(kwargs.get("embed_dim", 64))
    HIDDEN = int(kwargs.get("hidden", 256))
    N_BLOCKS = int(kwargs.get("n_blocks", 4))
    DROPOUT = float(kwargs.get("dropout", 0.25))

    LR = float(kwargs.get("lr", 1e-3))
    WEIGHT_DECAY = float(kwargs.get("weight_decay", 1e-5))
    MAX_EPOCHS = int(kwargs.get("max_epochs", 60))
    # windows per gradient step. Each window expands to n_neurons (~72k)
    # per-neuron samples, and 4 residual blocks retain ~20+ activation tensors
    # of shape (WIN_BATCH*F, hidden) for backward. WIN_BATCH=4 keeps peak VRAM
    # ~7 GB at F=71721 (safe on the 16 GB A4000) and gives ~287k samples/step.
    WIN_BATCH = int(kwargs.get("win_batch", 4))
    MAX_TRAIN_WINDOWS = int(kwargs.get("max_train_windows", 4000))
    TRAIN_STRIDE = int(kwargs.get("train_stride", 1))
    VAL_FRACTION = float(kwargs.get("val_fraction", 0.1))
    EARLY_STOP_PATIENCE = int(kwargs.get("early_stop_patience", 8))
    PRED_WIN_BATCH = int(kwargs.get("pred_win_batch", 8))
    TRAIN_TIME_FRACTION = float(kwargs.get("train_time_fraction", 0.6))

    rs = 0 if random_state is None else int(random_state)
    t_start = time.time()
    budget = float(time_budget_s) if time_budget_s else None

    X_train = np.asarray(X_train, dtype=np.float32)
    predict_inputs = np.asarray(predict_inputs, dtype=np.float32)
    T, F = X_train.shape
    N_pred = predict_inputs.shape[0]

    # ---- guaranteed-valid fallback prepared up front ----
    fallback = _persistence(predict_inputs, HORIZON)

    if not _TORCH_AVAILABLE:
        return fallback

    if X_train_condition_lengths is None:
        cond_lengths = np.array([T], dtype=np.int64)
    else:
        cond_lengths = np.asarray(X_train_condition_lengths, dtype=np.int64)

    try:
        np.random.seed(rs)
        torch.manual_seed(rs)
        try:
            # Use all CPUs the evaluator allocated; OMP_NUM_THREADS tracks
            # --cpus-per-task (8 under SLURM). Fall back to 4 if unset.
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        except Exception:
            pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- build condition-aware training windows ----
        starts = _segment_starts(cond_lengths, CONTEXT_LEN, HORIZON, stride=TRAIN_STRIDE)
        if starts.size == 0:
            return fallback
        rng = np.random.RandomState(rs)
        if starts.shape[0] > MAX_TRAIN_WINDOWS:
            starts = rng.choice(starts, size=MAX_TRAIN_WINDOWS, replace=False)
        rng.shuffle(starts)

        n_val = max(1, int(len(starts) * VAL_FRACTION)) if len(starts) > 10 else 0
        val_starts = starts[:n_val]
        train_starts = starts[n_val:]
        if train_starts.size == 0:
            return fallback

        offs_c = np.arange(CONTEXT_LEN, dtype=np.int64)
        offs_h = np.arange(CONTEXT_LEN, CONTEXT_LEN + HORIZON, dtype=np.int64)

        def _batch(starts_arr, idx0, idx1):
            sel = starts_arr[idx0:idx1]
            ctx = X_train[sel[:, None] + offs_c]   # (b, C, F)
            tgt = X_train[sel[:, None] + offs_h]   # (b, H, F)
            return ctx, tgt

        model = _SharedNeuronMLP(
            num_neurons=F, context_len=CONTEXT_LEN, horizon=HORIZON,
            global_dim=GLOBAL_DIM, embed_dim=EMBED_DIM, hidden=HIDDEN,
            n_blocks=N_BLOCKS, dropout=DROPOUT,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.L1Loss()
        neuron_idx = torch.arange(F, device=device)

        def _time_left():
            if budget is None:
                return True
            return (time.time() - t_start) < TRAIN_TIME_FRACTION * budget

        @torch.no_grad()
        def _val_mae():
            if val_starts.size == 0:
                return None
            model.eval()
            tot, cnt = 0.0, 0
            for i in range(0, len(val_starts), WIN_BATCH):
                ctx_np, tgt_np = _batch(val_starts, i, i + WIN_BATCH)
                ctx = torch.from_numpy(ctx_np).to(device)
                tgt = torch.from_numpy(tgt_np).to(device).permute(0, 2, 1)  # (b,F,H)
                pred = model(ctx, neuron_idx)
                tot += torch.abs(pred - tgt).mean().item() * ctx.shape[0]
                cnt += ctx.shape[0]
            return tot / max(cnt, 1)

        best_val = float("inf")
        best_state = None
        patience = 0
        n_train = len(train_starts)

        for epoch in range(MAX_EPOCHS):
            if not _time_left():
                break
            model.train()
            perm = rng.permutation(n_train)
            ts = train_starts[perm]
            for i in range(0, n_train, WIN_BATCH):
                if not _time_left():
                    break
                ctx_np, tgt_np = _batch(ts, i, i + WIN_BATCH)
                ctx = torch.from_numpy(ctx_np).to(device)
                tgt = torch.from_numpy(tgt_np).to(device).permute(0, 2, 1)  # (b,F,H)
                opt.zero_grad()
                pred = model(ctx, neuron_idx)
                loss = loss_fn(pred, tgt)
                loss.backward()
                opt.step()

            vm = _val_mae()
            if vm is None:
                continue
            if vm < best_val - 1e-6:
                best_val = vm
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- predict ----
        model.eval()
        Y_pred = np.empty((N_pred, HORIZON, F), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, N_pred, PRED_WIN_BATCH):
                ctx = torch.from_numpy(predict_inputs[i:i + PRED_WIN_BATCH]).to(device)
                pred = model(ctx, neuron_idx)          # (b, F, H)
                Y_pred[i:i + PRED_WIN_BATCH] = pred.permute(0, 2, 1).cpu().numpy()

        if not np.isfinite(Y_pred).all():
            return fallback
        return Y_pred.astype(np.float32)

    except Exception:
        return fallback
# EVOLVE-BLOCK-END
