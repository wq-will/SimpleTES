import torch
from typing import TypeVar

from utils import make_match_reference, DeterministicContext

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


def generate_input(bsz: int, n: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    x = torch.empty(bsz, n, device='cuda', dtype=torch.float32)
    x.uniform_(0, 1, generator=gen)
    output = torch.empty(bsz, n, device='cuda', dtype=torch.float32)
    return x, output


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        x, output = data
        return torch.cumsum(x, dim=1)


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
