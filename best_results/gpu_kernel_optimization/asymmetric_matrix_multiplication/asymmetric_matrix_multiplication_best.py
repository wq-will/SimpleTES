# EVOLVE-BLOCK-START

import torch
import triton
import triton.language as tl
from typing import Tuple

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
SMALL_K_MAX = 64                       # K ≤ this uses the small‑K kernel
SMEM_LIMIT = 200 * 1024                # Target shared‑memory usage per block (bytes)
MAX_REG_ELEMENTS = 16384               # Approximate per‑block register budget (elements)
MIN_TOTAL_BLOCKS = 128                 # Minimal number of thread‑blocks to keep GPU busy


# -------------------------------------------------------------------------
# Large‑K kernel (general case)
# -------------------------------------------------------------------------
@triton.jit
def _sfm_large_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_m = tl.program_id(1)  # block row
    pid_n = tl.program_id(0)  # block column

    # -----------------------------------------------------------------
    # Offsets for the current tile
    # -----------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # -----------------------------------------------------------------
    # Accumulator (FP32)
    # -----------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------------
    # Main reduction loop over K
    # -----------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)
        mask_k = cur_k < K

        # Load A tile (BLOCK_M × BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak)
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load B tile (BLOCK_K × BLOCK_N)
        b_ptrs = b_ptr + (cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Tensor‑core multiply‑accumulate
        if BLOCK_K >= 16:
            a = a.to(tl.float16)
            b = b.to(tl.float16)
            acc = acc + tl.dot(a, b).to(tl.float32)
        else:
            # For very small K we fall back to the default FP32 dot
            acc = acc + tl.dot(a, b)

    # -----------------------------------------------------------------
    # Store the result tile
    # -----------------------------------------------------------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# -------------------------------------------------------------------------
# Small‑K kernel (B‑tile reused across several row groups)
# -------------------------------------------------------------------------
@triton.jit
def _sfm_smallK_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # == K (the whole reduction fits in one tile)
    GROUP_M: tl.constexpr,
):
    pid_n = tl.program_id(0)    # column block
    pid_mg = tl.program_id(1)   # group of rows

    # -----------------------------------------------------------------
    # Column offsets – shared across all row groups handled by this block
    # -----------------------------------------------------------------
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # -----------------------------------------------------------------
    # Whole K dimension (tiny) – load once per block
    # -----------------------------------------------------------------
    cur_k = tl.arange(0, BLOCK_K)   # BLOCK_K == K
    mask_k = cur_k < K

    b_ptrs = b_ptr + (cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    b = tl.load(
        b_ptrs,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    )
    if BLOCK_K >= 16:
        b = b.to(tl.float16)

    # -----------------------------------------------------------------
    # Loop over GROUP_M row‑tiles, re‑using the same B tile
    # -----------------------------------------------------------------
    for grp in range(0, GROUP_M):
        offs_m = pid_mg * BLOCK_M * GROUP_M + grp * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak)
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if BLOCK_K >= 16:
            a = a.to(tl.float16)
            acc = tl.dot(a, b).to(tl.float32)
        else:
            acc = tl.dot(a, b)

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# -------------------------------------------------------------------------
# Helper: pick a GROUP_M for the small‑K kernel that yields enough blocks
# -------------------------------------------------------------------------
def _choose_group_m(M, BLOCK_M, BLOCK_N, N, min_total_blocks=MIN_TOTAL_BLOCKS):
    """
    Pick a reuse factor (GROUP_M) that guarantees a minimum number of
    thread blocks while keeping each block reasonably sized.
    """
    if M >= BLOCK_M * 16:
        GROUP_M = 16
    elif M >= BLOCK_M * 8:
        GROUP_M = 8
    elif M >= BLOCK_M * 4:
        GROUP_M = 4
    else:
        GROUP_M = 1

    grid_x = triton.cdiv(N, BLOCK_N)
    grid_y = triton.cdiv(M, BLOCK_M * GROUP_M)

    # Reduce GROUP_M if we still have too few blocks
    while grid_x * grid_y < min_total_blocks and GROUP_M > 1:
        GROUP_M //= 2
        grid_y = triton.cdiv(M, BLOCK_M * GROUP_M)

    return GROUP_M


# -------------------------------------------------------------------------
# Tile selector for the large‑K regime (K > SMALL_K_MAX)
# -------------------------------------------------------------------------
def _select_tile_config_largeK(M: int, N: int, K: int, min_total_blocks=MIN_TOTAL_BLOCKS):
    """
    Heuristic chooser for BLOCK_M, BLOCK_N, BLOCK_K, number of warps,
    and pipeline depth (num_stages) when the reduction dimension is large.
    """
    # ----- Choose BLOCK_K and pipeline depth ---------------------------------
    if K >= 12288:
        BLOCK_K = 128
        num_stages = 4
    elif K >= 8192:
        BLOCK_K = 128
        num_stages = 4
    elif K >= 4096:
        BLOCK_K = 64
        num_stages = 4
    elif K >= 2048:
        BLOCK_K = 64
        num_stages = 3
    elif K >= 1024:
        BLOCK_K = 32
        num_stages = 3
    else:
        BLOCK_K = 32
        num_stages = 2
    BLOCK_K = min(BLOCK_K, K)

    # ----- Choose BLOCK_M (row tile) ----------------------------------------
    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 64

    # ----- Choose BLOCK_N (column tile) respecting SMEM & regs ---------------
    candidates = (256, 128, 64, 32, 16)
    BLOCK_N = 16  # fallback

    for cand in candidates:
        cand = min(cand, N)

        # Register‑budget check (very rough)
        if BLOCK_M * cand > MAX_REG_ELEMENTS:
            continue

        # Shared‑memory budget estimate (FP32 tiles)
        per_stage = (BLOCK_M * BLOCK_K + BLOCK_K * cand) * 4
        if per_stage * num_stages > SMEM_LIMIT:
            continue

        # Occupancy check – ensure enough total blocks
        total_blocks = triton.cdiv(N, cand) * triton.cdiv(M, BLOCK_M)
        if total_blocks < min_total_blocks:
            continue

        BLOCK_N = cand
        break

    # ----- Choose number of warps per block ----------------------------------
    area = BLOCK_M * BLOCK_N
    num_warps = 4 if area <= 64 * 64 else 8
    num_warps = max(4, min(32, num_warps))

    return BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages


# -------------------------------------------------------------------------
# Tile selector for the tiny‑K regime (K ≤ SMALL_K_MAX)
# -------------------------------------------------------------------------
def _select_tile_config_smallK(M: int, N: int, K: int):
    """
    Heuristic for the small‑K case where the entire reduction fits into
    a single tile (B‑reuse kernel).
    """
    BLOCK_K = K  # entire reduction dimension

    # ----- Row tile size ----------------------------------------------------
    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32
    elif M <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 64
    BLOCK_M = min(BLOCK_M, M)

    # ----- Column tile size limited by register budget -----------------------
    max_n = min(N, MAX_REG_ELEMENTS // BLOCK_M)
    for cand in (256, 128, 64, 32, 16):
        if cand <= max_n:
            BLOCK_N = cand
            break
    else:
        BLOCK_N = min(16, N)

    # ----- Warps per block ---------------------------------------------------
    area = BLOCK_M * BLOCK_N
    num_warps = 4 if area <= 64 * 64 else 8
    num_warps = max(4, min(32, num_warps))

    return BLOCK_M, BLOCK_N, BLOCK_K, num_warps


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Triton‑accelerated asymmetric GEMM (C = A @ B).

    Parameters
    ----------
    data : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (a, b, c) – where ``a`` has shape [M, K], ``b`` has shape [K, N],
        and ``c`` is a pre‑allocated output buffer of shape [M, N].

    The function writes the product into ``c`` and returns it.
    """
    a, b, c = data

    # -----------------------------------------------------------------
    # Basic validation
    # -----------------------------------------------------------------
    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (a.dtype == b.dtype == c.dtype == torch.float32):
        raise RuntimeError("Only torch.float32 tensors are supported.")
    if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous.")
    if a.dim() != 2 or b.dim() != 2 or c.dim() != 2:
        raise RuntimeError("All tensors must be 2‑D matrices.")

    M, K = a.shape
    Kb, N = b.shape
    if K != Kb:
        raise ValueError(f"Incompatible inner dimensions: {K} vs {Kb}")
    if c.shape != (M, N):
        raise ValueError(f"Output shape mismatch: expected ({M}, {N}) got {c.shape}")

    # -----------------------------------------------------------------
    # Strides (row‑major, contiguous)
    # -----------------------------------------------------------------
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    # -----------------------------------------------------------------
    # Choose tile configuration and launch appropriate kernel
    # -----------------------------------------------------------------
    if K <= SMALL_K_MAX:
        # ------------------- Small‑K specialization -------------------
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = _select_tile_config_smallK(M, N, K)
        GROUP_M = _choose_group_m(M, BLOCK_M, BLOCK_N, N, MIN_TOTAL_BLOCKS)

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M * GROUP_M))
        _sfm_smallK_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=num_warps,   # compile‑time hint
        )
    else:
        # ------------------- Large‑K path ---------------------------
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _select_tile_config_largeK(
            M, N, K, MIN_TOTAL_BLOCKS
        )
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
        _sfm_large_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_stages=num_stages,
            num_warps=num_warps,   # compile‑time hint
        )

    return c
# EVOLVE-BLOCK-END
