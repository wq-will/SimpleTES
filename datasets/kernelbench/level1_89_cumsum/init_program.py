# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple

import torch
from torch import nn

def custom_kernel(data: Tuple[torch.Tensor, int]) -> torch.Tensor:
    x, dim = data
    # Your optimized implementation

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, dim=self.dim)


def ref_kernel(data: Tuple[torch.Tensor, int]) -> torch.Tensor:
    x, dim = data
    model = Model(dim).to(x.device)
    return model(x)


def generate_input(
    batch_size: int,
    input_shape: Tuple[int, ...],
    dim: int,
    seed: int = 17717,
    distribution: str = "uniform",
    dtype: str = "float32",
) -> Tuple[torch.Tensor]:
    """
    Generate inputs for the scan / cumsum task.

    Args:
        batch_size: batch size
        input_shape: shape after batch dimension
        seed: random seed
        distribution: supports "uniform", "normal", "cauchy"
        dtype: supports "float32", "float16", "bfloat16"

    Returns:
        (x,)
            x: [batch_size, *input_shape]
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

    shape = (batch_size, *input_shape)

    if distribution == "uniform":
        x = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "normal":
        x = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        x = torch.distributions.Cauchy(0, 2).sample(shape).to(device=device, dtype=torch_dtype).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (x, dim)


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END