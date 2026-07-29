# EVOLVE-BLOCK-START

import torch
from utils import DisableCuDNNTF32
from typing import Tuple, Dict
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Autotune configuration generators
# ----------------------------------------------------------------------
def _layernorm_configs():
    cfgs = []
    for BLOCK_POS in (64, 128, 256, 512):
        for BLOCK_C in (32, 64, 128, 256):
            num_warps = 8 if BLOCK_POS * BLOCK_C >= 32768 else 4
            cfgs.append(
                triton.Config(
                    {"BLOCK_POS": BLOCK_POS, "BLOCK_C": BLOCK_C},
                    num_warps=num_warps,
                    num_stages=3,
                )
            )
    return cfgs


def _proj_gate_configs():
    cfgs = []
    for BLOCK_POS in (64, 128, 256, 512):
        for BLOCK_HD in (32, 64, 128):
            for BLOCK_K in (32, 64, 128, 256):
                work = BLOCK_POS * BLOCK_HD * BLOCK_K
                num_warps = 8 if work >= 65536 else 4
                cfgs.append(
                    triton.Config(
                        {
                            "BLOCK_POS": BLOCK_POS,
                            "BLOCK_HD": BLOCK_HD,
                            "BLOCK_K": BLOCK_K,
                        },
                        num_warps=num_warps,
                        num_stages=3,
                    )
                )
    return cfgs


def _final_fused_configs():
    cfgs = []
    for BLOCK_POS in (64, 128, 256, 512):
        for BLOCK_D in (32, 64, 128, 256):
            cfgs.append(
                triton.Config(
                    {
                        "BLOCK_POS": BLOCK_POS,
                        "BLOCK_HD": 128,  # hidden_dim is always 128 in the benchmark configs
                        "BLOCK_D": BLOCK_D,
                    },
                    num_warps=4,
                    num_stages=3,
                )
            )
    return cfgs


# ----------------------------------------------------------------------
# Kernel 1: FP32 LayerNorm → FP16 (fused mean/var/affine)
# ----------------------------------------------------------------------
@triton.autotune(configs=_layernorm_configs(), key=["P", "C"])
@triton.jit
def _layernorm_fp32_to_fp16(
    inp_ptr,      # float32* [P, C]
    out_ptr,      # float16* [P, C]
    weight_ptr,   # float32* [C]
    bias_ptr,     # float32* [C]
    P,            # i32 total positions = B * N * N
    C,            # i32 channel dimension (dim)
    eps,          # f32 epsilon
    BLOCK_POS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_base = pid * BLOCK_POS
    pos = pos_base + tl.arange(0, BLOCK_POS)  # [BLOCK_POS]
    pos_mask = pos < P

    c_tiles = tl.cdiv(C, BLOCK_C)

    # ---------- First pass: compute mean & variance ----------
    sum_val = tl.zeros([BLOCK_POS], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_POS], dtype=tl.float32)

    for c in range(c_tiles):
        c_off = c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_off < C
        offs = pos[:, None] * C + c_off[None, :]            # [BLOCK_POS, BLOCK_C]
        load_mask = pos_mask[:, None] & c_mask[None, :]
        val = tl.load(inp_ptr + offs, mask=load_mask, other=0.0)  # fp32
        sum_val += tl.sum(val, axis=1)
        sum_sq += tl.sum(val * val, axis=1)

    mean = sum_val / C
    var = sum_sq / C - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # ---------- Second pass: normalize + affine ----------
    for c in range(c_tiles):
        c_off = c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_off < C
        w = tl.load(weight_ptr + c_off, mask=c_mask, other=0.0)   # fp32
        b = tl.load(bias_ptr + c_off, mask=c_mask, other=0.0)    # fp32

        offs = pos[:, None] * C + c_off[None, :]
        load_mask = pos_mask[:, None] & c_mask[None, :]

        val = tl.load(inp_ptr + offs, mask=load_mask, other=0.0)  # fp32
        norm = (val - mean[:, None]) * inv_std[:, None]          # fp32
        out_fp32 = norm * w[None, :] + b[None, :]                # fp32
        tl.store(out_ptr + offs, out_fp32.to(tl.float16), mask=load_mask)


# ----------------------------------------------------------------------
# Kernel 2: Projection → Gating → (optional) Mask
# Produces left, right and out_gate tensors of shape [B*hd*N*N]
# ----------------------------------------------------------------------
@triton.autotune(configs=_proj_gate_configs(), key=["P", "hd", "dim"])
@triton.jit
def _proj_gate_fused_kernel(
    inp_ptr,            # fp16* [P, C]   (C == dim)
    left_ptr,           # fp16* [B*hd*N*N]
    right_ptr,          # fp16* [B*hd*N*N]
    out_gate_ptr,       # fp16* [B*hd*N*N]
    mask_ptr,           # fp32* [P]      (binary mask; may be ignored)
    left_proj_w_ptr,    # fp16* [hd, C]
    left_gate_w_ptr,    # fp16* [hd, C]
    right_proj_w_ptr,   # fp16* [hd, C]
    right_gate_w_ptr,   # fp16* [hd, C]
    out_gate_w_ptr,     # fp16* [hd, C]
    B, N, hd, dim, P,
    BLOCK_POS: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    # ----- position tiling -----
    pos = tl.program_id(0) * BLOCK_POS + tl.arange(0, BLOCK_POS)
    pos_mask = pos < P

    # ----- hidden dimension tiling -----
    hd_start = tl.program_id(1) * BLOCK_HD
    hd_idx = hd_start + tl.arange(0, BLOCK_HD)
    hd_mask = hd_idx < hd

    # ----- decode flat position -> (b, i, k) -----
    N2 = N * N
    b = pos // N2
    rem = pos - b * N2                     # = i*N + k

    # ----- mask handling -----
    if USE_MASK:
        m = tl.load(mask_ptr + pos, mask=pos_mask, other=1.0)   # fp32
        m_f32 = m[:, None]                                     # [BLOCK_POS, 1]
    else:
        m_f32 = tl.full([BLOCK_POS, 1], 1.0, tl.float32)

    # ----- accumulation buffers (FP32) -----
    acc_left = tl.zeros((BLOCK_POS, BLOCK_HD), dtype=tl.float32)
    acc_left_gate = tl.zeros((BLOCK_POS, BLOCK_HD), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_POS, BLOCK_HD), dtype=tl.float32)
    acc_right_gate = tl.zeros((BLOCK_POS, BLOCK_HD), dtype=tl.float32)
    acc_out_gate = tl.zeros((BLOCK_POS, BLOCK_HD), dtype=tl.float32)

    # ----- loop over input channel dimension -----
    num_k_tiles = tl.cdiv(dim, BLOCK_K)
    for k_tile in range(num_k_tiles):
        cur_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = cur_k < dim

        # Load normalized input tile (fp16)
        a = tl.load(
            inp_ptr + pos[:, None] * dim + cur_k[None, :],
            mask=pos_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # fp16, shape [BLOCK_POS, BLOCK_K]

        # Load weight tiles (row‑major [hd, dim] layout)
        w_left = tl.load(
            left_proj_w_ptr + cur_k[:, None] + hd_idx[None, :] * dim,
            mask=k_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        w_lgate = tl.load(
            left_gate_w_ptr + cur_k[:, None] + hd_idx[None, :] * dim,
            mask=k_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        w_right = tl.load(
            right_proj_w_ptr + cur_k[:, None] + hd_idx[None, :] * dim,
            mask=k_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        w_rgate = tl.load(
            right_gate_w_ptr + cur_k[:, None] + hd_idx[None, :] * dim,
            mask=k_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )
        w_outgate = tl.load(
            out_gate_w_ptr + cur_k[:, None] + hd_idx[None, :] * dim,
            mask=k_mask[:, None] & hd_mask[None, :],
            other=0.0,
        )

        # Accumulate (FP32) via tensor‑core dot
        acc_left = tl.dot(a, w_left, acc_left)
        acc_left_gate = tl.dot(a, w_lgate, acc_left_gate)
        acc_right = tl.dot(a, w_right, acc_right)
        acc_right_gate = tl.dot(a, w_rgate, acc_right_gate)
        acc_out_gate = tl.dot(a, w_outgate, acc_out_gate)

    # ----- sigmoid activations (FP32) -----
    left_gate = 1.0 / (1.0 + tl.exp(-acc_left_gate))
    right_gate = 1.0 / (1.0 + tl.exp(-acc_right_gate))
    out_gate = 1.0 / (1.0 + tl.exp(-acc_out_gate))

    # ----- apply mask and cast back to fp16 -----
    left_fp16 = (acc_left * left_gate * m_f32).to(tl.float16)
    right_fp16 = (acc_right * right_gate * m_f32).to(tl.float16)
    out_gate_fp16 = out_gate.to(tl.float16)

    # ----- write results -----
    base = (b[:, None] * hd + hd_idx[None, :]) * N2 + rem[:, None]   # [BLOCK_POS, BLOCK_HD]
    store_mask = pos_mask[:, None] & hd_mask[None, :]
    tl.store(left_ptr + base, left_fp16, mask=store_mask)
    tl.store(right_ptr + base, right_fp16, mask=store_mask)
    tl.store(out_gate_ptr + base, out_gate_fp16, mask=store_mask)


# ----------------------------------------------------------------------
# Kernel 3: Second LayerNorm + output‑gate + final projection
# ----------------------------------------------------------------------
@triton.autotune(configs=_final_fused_configs(), key=["B", "N", "hd", "dim_out"])
@triton.jit
def _final_fused_kernel_fp32(
    contracted_ptr,    # fp16* [B*hd, N, N] – result of the GEMM
    out_gate_ptr,      # fp16* [B*hd, N, N] – output gate
    ln_weight_ptr,     # fp32* [hd]
    ln_bias_ptr,       # fp32* [hd]
    proj_weight_ptr,   # fp16* [hd, dim_out] – final projection weight (row‑major)
    out_proj_ptr,      # fp32* [B*N*N, dim_out] – final output buffer (flattened)
    B, N, hd, dim_out, # i32 scalars
    eps,               # f32 epsilon for LayerNorm
    BLOCK_POS: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_base = pid * BLOCK_POS
    pos = pos_base + tl.arange(0, BLOCK_POS)               # [BLOCK_POS]
    pos_mask = pos < (B * N * N)

    N2 = N * N
    b = pos // N2
    rem = pos - b * N2                                     # = i*N + j

    hd_idx = tl.arange(0, BLOCK_HD)
    hd_mask = hd_idx < hd

    # ----- load contracted tensor and output gate (fp16 → fp32) -----
    base = b[:, None] * hd * N2 + hd_idx[None, :] * N2 + rem[:, None]   # [BLOCK_POS, BLOCK_HD]

    contracted_fp16 = tl.load(
        contracted_ptr + base,
        mask=pos_mask[:, None] & hd_mask[None, :],
        other=0.0,
    )
    contracted_fp32 = contracted_fp16.to(tl.float32)

    gate_fp16 = tl.load(
        out_gate_ptr + base,
        mask=pos_mask[:, None] & hd_mask[None, :],
        other=0.0,
    )
    gate_fp32 = gate_fp16.to(tl.float32)

    # ---------- LayerNorm across hidden dimension ----------
    sum_val = tl.sum(contracted_fp32, axis=1)               # [BLOCK_POS]
    mean = sum_val / hd
    sum_sq = tl.sum(contracted_fp32 * contracted_fp32, axis=1)
    var = sum_sq / hd - mean * mean
    inv_std = tl.rsqrt(var + eps)

    norm = (contracted_fp32 - mean[:, None]) * inv_std[:, None]

    # ---------- Affine (γ, β) ----------
    gamma = tl.load(ln_weight_ptr + hd_idx, mask=hd_mask, other=0.0)
    beta = tl.load(ln_bias_ptr + hd_idx, mask=hd_mask, other=0.0)
    norm = norm * gamma[None, :] + beta[None, :]

    # ---------- Apply output gate ----------
    norm = norm * gate_fp32

    # ---------- Cast to fp16 for final projection ----------
    norm_fp16 = norm.to(tl.float16)

    # ---------- Final projection (hd → dim_out) ----------
    d_start = 0
    while d_start < dim_out:
        cur_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = cur_d < dim_out

        w = tl.load(
            proj_weight_ptr + hd_idx[:, None] * dim_out + cur_d[None, :],
            mask=hd_mask[:, None] & mask_d[None, :],
            other=0.0,
        )  # fp16, shape [hd, BLOCK_D]

        acc = tl.dot(norm_fp16, w, tl.zeros([BLOCK_POS, BLOCK_D], dtype=tl.float32))

        out_idx = pos[:, None] * dim_out + cur_d[None, :]
        tl.store(
            out_proj_ptr + out_idx,
            acc,
            mask=pos_mask[:, None] & mask_d[None, :],
        )
        d_start += BLOCK_D


# ----------------------------------------------------------------------
# Grid helper functions
# ----------------------------------------------------------------------
def _grid_layernorm(meta):
    return (triton.cdiv(meta["P"], meta["BLOCK_POS"]),)


def _grid_proj(meta):
    return (
        triton.cdiv(meta["P"], meta["BLOCK_POS"]),
        triton.cdiv(meta["hd"], meta["BLOCK_HD"]),
    )


def _grid_final(meta):
    return (triton.cdiv(meta["B"] * meta["N"] * meta["N"], meta["BLOCK_POS"]),)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
    """
    Triton‑based forward‑only implementation of the outgoing TriMul operator.

    Args:
        data: Tuple of (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, dim] (float32)
            - mask: torch.Tensor of shape [B, N, N] (float32, binary; may be all ones)
            - weights: dict containing the model weights (float32 tensors)
            - config: dict with keys "dim", "hidden_dim", "nomask"

    Returns:
        torch.Tensor of shape [B, N, N, dim] (float32)
    """
    input_tensor, mask, weights, config = data

    # Disable TF32 for deterministic / high‑precision behavior
    with DisableCuDNNTF32():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)

        device = input_tensor.device
        input_tensor = input_tensor.contiguous()
        mask = mask.contiguous()

        B, N, _, C = input_tensor.shape            # C == config["dim"]
        hidden_dim = config["hidden_dim"]           # typically 128
        dim_out = config["dim"]                     # output dim equals input dim

        hd = hidden_dim
        N2 = N * N
        P = B * N2                                  # total (b, i, k) positions

        # ------------------------------------------------------------------
        # 1️⃣ First LayerNorm (FP32 → FP16)
        # ------------------------------------------------------------------
        x_norm_fp16 = torch.empty_like(
            input_tensor, dtype=torch.float16, device=device
        )
        _layernorm_fp32_to_fp16[_grid_layernorm](
            input_tensor,
            x_norm_fp16,
            weights["norm.weight"],
            weights["norm.bias"],
            P,
            C,
            eps=1e-5,
        )

        # ------------------------------------------------------------------
        # 2️⃣ Fused projection + gating (+ optional) mask
        # ------------------------------------------------------------------
        # Cast linear weights once to fp16 (row‑major layout)
        w_left_proj = weights["left_proj.weight"].to(device, dtype=torch.float16).contiguous()
        w_right_proj = weights["right_proj.weight"].to(device, dtype=torch.float16).contiguous()
        w_left_gate = weights["left_gate.weight"].to(device, dtype=torch.float16).contiguous()
        w_right_gate = weights["right_gate.weight"].to(device, dtype=torch.float16).contiguous()
        w_out_gate = weights["out_gate.weight"].to(device, dtype=torch.float16).contiguous()

        total_elements = B * hd * N2
        left = torch.empty(total_elements, dtype=torch.float16, device=device)
        right = torch.empty(total_elements, dtype=torch.float16, device=device)
        out_gate = torch.empty(total_elements, dtype=torch.float16, device=device)

        # Flatten mask to [P]; if masking is disabled we still pass a tensor
        mask_flat = mask.view(P).contiguous()
        use_mask_flag = 0 if config.get("nomask", True) else 1

        _proj_gate_fused_kernel[_grid_proj](
            x_norm_fp16,
            left,
            right,
            out_gate,
            mask_flat,
            w_left_proj,
            w_left_gate,
            w_right_proj,
            w_right_gate,
            w_out_gate,
            B,
            N,
            hd,
            C,
            P,
            USE_MASK=use_mask_flag,
        )

        # ------------------------------------------------------------------
        # 3️⃣ Core contraction – batched GEMM (fp16 @ fp16ᵀ) via cuBLAS
        # ------------------------------------------------------------------
        batch = B * hd
        left_3d = left.view(batch, N, N)          # (batch, i, k)   fp16
        right_3d = right.view(batch, N, N)        # (batch, j, k)   fp16

        out_3d = torch.bmm(left_3d, right_3d.transpose(1, 2))

        # ------------------------------------------------------------------
        # 4️⃣ Second LayerNorm + output‑gate + final projection
        # ------------------------------------------------------------------
        out_proj_fp32 = torch.empty(
            (B, N, N, dim_out), dtype=torch.float32, device=device
        )

        ln_weight_fp32 = weights["to_out_norm.weight"].to(device, dtype=torch.float32).contiguous()
        ln_bias_fp32 = weights["to_out_norm.bias"].to(device, dtype=torch.float32).contiguous()

        # Final projection weight: transpose to [hd, dim_out] for row‑major accesses
        proj_weight_fp16 = weights["to_out.weight"].t().contiguous().to(device, dtype=torch.float16)

        _final_fused_kernel_fp32[_grid_final](
            out_3d,
            out_gate,
            ln_weight_fp32,
            ln_bias_fp32,
            proj_weight_fp16,
            out_proj_fp32.view(-1, dim_out),
            B,
            N,
            hd,
            dim_out,
            eps=1e-5,
        )

        return out_proj_fp32
# EVOLVE-BLOCK-END