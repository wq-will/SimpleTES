# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def custom_kernel(data: Tuple[torch.Tensor, int, int, int, int, int, int]) -> torch.Tensor:
    X, batch_size, seq_length, n_heads, d_head, d_state, block_len = data
    # Your optimized implementation

# Reference code in PyTorch
class Model(nn.Module):
    def __init__(
        self,
        batch_size: int,
        seq_length: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        block_len: int = 64,
    ):
        super().__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def segsum(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(
        self,
        X: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        L = torch.exp(self.segsum(A_blocks))
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks,
            B_blocks,
            L,
            X_blocks,
        )

        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_blocks,
            decay_states,
            X_blocks,
        )

        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]


def ref_kernel(
    data: Tuple[
        torch.Tensor,
        int,
        int,
        int,
        int,
        int,
        int,
    ]
) -> torch.Tensor:
    (
        X,
        batch_size,
        seq_length,
        n_heads,
        d_head,
        d_state,
        block_len,
    ) = data

    model = Model(
        batch_size=batch_size,
        seq_length=seq_length,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
        block_len=block_len,
    ).to(X.device)

    return model(X)


def generate_input(
    batch_size: int,
    seq_length: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    block_len: int,
    seed: int,
    distribution: str = "uniform",
    dtype: str = "float32",
) -> Tuple[
    torch.Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    shape = (batch_size, seq_length, n_heads, d_head)

    if distribution == "uniform":
        X = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "normal":
        X = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        X = torch.distributions.Cauchy(0, 2).sample(shape).to(
            device=device, dtype=torch_dtype
        ).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (
        X,
        batch_size,
        seq_length,
        n_heads,
        d_head,
        d_state,
        block_len,
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END