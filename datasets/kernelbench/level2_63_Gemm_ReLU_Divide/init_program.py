# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple

import torch
from torch import nn


def custom_kernel(data: Tuple[torch.Tensor, int, int, float]) -> torch.Tensor:
    x, in_features, out_features, divisor = data
    # Your optimized implementation

# Reference code in PyTorch
class Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        divisor: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        x = x / self.divisor
        return x


def ref_kernel(
    data: Tuple[torch.Tensor, int, int, float]
) -> torch.Tensor:
    x, in_features, out_features, divisor = data
    model = Model(
        in_features=in_features,
        out_features=out_features,
        divisor=divisor,
    ).to(x.device)
    return model(x)


def generate_input(
    batch_size: int,
    in_features: int,
    out_features: int,
    divisor: float,
    seed: int,
    distribution: str = "uniform",
    dtype: str = "float32",
) -> Tuple[torch.Tensor, int, int, float]:
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    shape = (batch_size, in_features)

    if distribution == "uniform":
        x = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "normal":
        x = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        x = torch.distributions.Cauchy(0, 2).sample(shape).to(device=device, dtype=torch_dtype).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (
        x,
        in_features,
        out_features,
        divisor,
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END