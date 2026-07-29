# EVOLVE-BLOCK-START

import torch
import triton
import triton.language as tl
from typing import Tuple

# ----------------------------------------------------------------------
# Fast‑path kernel: the whole row fits into a single tile.
# ----------------------------------------------------------------------
@triton.jit
def cumsum_one_tile_kernel(
    x_ptr,          # *float32 input
    out_ptr,        # *float32 output
    N,              # row length (runtime scalar)
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    row = tl.program_id(0)               # row index
    row_offset = row * N                  # start offset of the row

    # Offsets for each thread's vector
    thread_idx = tl.arange(0, BLOCK_SIZE) * VEC          # [BLOCK_SIZE]
    vec_idx = tl.arange(0, VEC)                         # [VEC]
    offsets = thread_idx[:, None] + vec_idx[None, :]    # [BLOCK_SIZE, VEC]
    mask = offsets < N

    # Load the row (masked load for the tail)
    x = tl.load(
        x_ptr + row_offset + offsets,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    )   # shape: [BLOCK_SIZE, VEC]

    # Inclusive prefix inside each thread's vector
    thread_prefix = tl.cumsum(x, axis=1)                # [BLOCK_SIZE, VEC]

    # Sum of each thread's vector
    thread_sum = tl.sum(x, axis=1)                      # [BLOCK_SIZE]

    # Exclusive prefix across threads (block‑level scan)
    thread_exclusive = tl.cumsum(thread_sum, axis=0) - thread_sum   # [BLOCK_SIZE]

    # Combine intra‑thread prefix with thread‑level exclusive offsets
    out = thread_prefix + thread_exclusive[:, None]     # [BLOCK_SIZE, VEC]

    # Write the result (masked store)
    tl.store(
        out_ptr + row_offset + offsets,
        out,
        mask=mask,
        cache_modifier=".wt",
    )

# ----------------------------------------------------------------------
# General multi‑tile kernel (handles rows longer than one tile)
# ----------------------------------------------------------------------
@triton.jit
def cumsum_multi_tile_kernel(
    x_ptr,          # *float32 input
    out_ptr,        # *float32 output
    N,              # row length (runtime scalar)
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    row = tl.program_id(0)               # row index
    row_offset = row * N                  # start offset of the row

    # Scalar accumulator for the sum of all previously processed tiles
    running_sum = tl.zeros([], dtype=tl.float32)

    # Pre‑compute per‑thread base offsets and intra‑thread vector offsets
    thread_start = tl.arange(0, BLOCK_SIZE) * VEC   # [BLOCK_SIZE]
    vec_range = tl.arange(0, VEC)                   # [VEC]

    tile_size = BLOCK_SIZE * VEC
    offset = 0

    while offset < N:
        # Absolute offsets for the current tile
        thread_base = thread_start + offset                     # [BLOCK_SIZE]
        offsets = thread_base[:, None] + vec_range[None, :]    # [BLOCK_SIZE, VEC]
        mask = offsets < N

        # Load the tile (masked load for the tail)
        x = tl.load(
            x_ptr + row_offset + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".ca",
        )   # shape: [BLOCK_SIZE, VEC]

        # Intra‑thread inclusive prefix
        thread_prefix = tl.cumsum(x, axis=1)                 # [BLOCK_SIZE, VEC]

        # Sum of each thread's vector
        thread_sum = tl.sum(x, axis=1)                       # [BLOCK_SIZE]

        # Exclusive prefix across threads inside the tile
        thread_exclusive = tl.cumsum(thread_sum, axis=0) - thread_sum   # [BLOCK_SIZE]

        # Combine intra‑thread prefix, thread‑level offset and the running sum
        out = thread_prefix + thread_exclusive[:, None] + running_sum   # broadcast scalar

        # Write the result for the current tile (masked store)
        tl.store(
            out_ptr + row_offset + offsets,
            out,
            mask=mask,
            cache_modifier=".wt",
        )

        # Update the running sum with the total of this tile
        tile_sum = tl.sum(thread_sum)          # scalar
        running_sum = running_sum + tile_sum

        offset += tile_size


def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Batched inclusive cumulative sum (cumsum) along dimension 1.
    Uses a fast‑path kernel when a row fits in a single tile; otherwise
    processes the row in vectorized tiles while keeping a scalar running sum
    across tiles.
    """
    x, output = data

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    assert isinstance(x, torch.Tensor) and isinstance(output, torch.Tensor), \
        "Inputs must be torch tensors."
    assert x.is_cuda and output.is_cuda, "Both tensors must be CUDA tensors."
    assert x.is_contiguous() and output.is_contiguous(), \
        "Tensors must be contiguous."
    assert x.dim() == 2 and output.shape == x.shape, \
        "Input must be a 2‑D tensor with matching shapes."
    assert x.dtype == torch.float32 and output.dtype == torch.float32, \
        "Only float32 dtype is supported."

    bsz, n = x.shape

    # ------------------------------------------------------------------
    # Choose vector width (elements per thread).  Use 64 for very long rows
    # to halve the number of tiles and reduce loop overhead.
    # ------------------------------------------------------------------
    VEC = 64 if n > 65536 else 32

    # ------------------------------------------------------------------
    # Determine the number of logical threads needed per block.
    # We cap the block size at 512 to stay within the per‑SM register budget.
    # ------------------------------------------------------------------
    threads_needed = (n + VEC - 1) // VEC          # ceil division
    MAX_BLOCK = 512
    if threads_needed <= MAX_BLOCK:
        # Next power‑of‑two >= threads_needed, but at least 32.
        BLOCK_SIZE = max(32, 1 << ((threads_needed - 1).bit_length()))
        BLOCK_SIZE = min(BLOCK_SIZE, MAX_BLOCK)
    else:
        BLOCK_SIZE = MAX_BLOCK

    tile_size = BLOCK_SIZE * VEC

    # ------------------------------------------------------------------
    # Launch configuration
    # ------------------------------------------------------------------
    grid = (bsz,)
    num_warps = max(1, BLOCK_SIZE // 32)

    # ------------------------------------------------------------------
    # Fast path for rows that fit into a single tile
    # ------------------------------------------------------------------
    if n <= tile_size:
        cumsum_one_tile_kernel[grid](
            x,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            num_warps=num_warps,
            num_stages=2,          # shallow pipeline for short rows
        )
    else:
        # General tiled scan for long rows
        cumsum_multi_tile_kernel[grid](
            x,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            num_warps=num_warps,
            num_stages=4,          # deeper pipeline to hide memory latency
        )

    return output
# EVOLVE-BLOCK-END
