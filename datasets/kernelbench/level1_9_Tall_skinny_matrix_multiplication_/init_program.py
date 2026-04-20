# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple, Dict

import torch
from torch import nn

def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    A, B = data
    # Your optimized implementation

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        return torch.matmul(A, B)

def ref_kernel(data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    A, B = data
    model = Model().to(A.device)
    return model(A, B)

def generate_input(
    M: int,
    N: int,
    seed: int = 17717,
    distribution: str = "normal",
    dtype: str = "float32",
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    if distribution == "normal":
        A = torch.randn((M, N), device=device, dtype=torch_dtype, generator=gen).contiguous()
        B = torch.randn((N, M), device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "uniform":
        A = torch.rand((M, N), device=device, dtype=torch_dtype, generator=gen).contiguous()
        B = torch.rand((N, M), device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        A = torch.distributions.Cauchy(0, 2).sample((M, N)).to(device=device, dtype=torch_dtype).contiguous()
        B = torch.distributions.Cauchy(0, 2).sample((N, M)).to(device=device, dtype=torch_dtype).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (A, B)

check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END