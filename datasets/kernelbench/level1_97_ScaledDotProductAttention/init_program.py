# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    Q, K, V = data
    # Your optimized implementation

# Reference code in PyTorch
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(Q, K, V)


def ref_kernel(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    Q, K, V = data
    model = Model().to(Q.device)
    return model(Q, K, V)


def generate_input(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    embedding_dimension: int,
    seed: int = 17717,
    distribution: str = "uniform",
    dtype: str = "float32",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate inputs for the scaled dot-product attention task.
    """
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    shape = (batch_size, num_heads, sequence_length, embedding_dimension)

    if distribution == "uniform":
        Q = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
        K = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
        V = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "normal":
        Q = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
        K = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
        V = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        Q = torch.distributions.Cauchy(0, 2).sample(shape).to(device=device, dtype=torch_dtype).contiguous()
        K = torch.distributions.Cauchy(0, 2).sample(shape).to(device=device, dtype=torch_dtype).contiguous()
        V = torch.distributions.Cauchy(0, 2).sample(shape).to(device=device, dtype=torch_dtype).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (Q, K, V)


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END