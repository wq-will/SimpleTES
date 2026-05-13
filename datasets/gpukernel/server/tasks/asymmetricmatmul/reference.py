import torch
from typing import TypeVar

from utils import make_match_reference, DeterministicContext

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


def generate_input(m: int, k: int, n: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    a = torch.empty(m, k, device='cuda', dtype=torch.float32)
    a.uniform_(0, 1, generator=gen)
    b = torch.empty(k, n, device='cuda', dtype=torch.float32)
    b.uniform_(0, 1, generator=gen)
    c = torch.empty(m, n, device='cuda', dtype=torch.float32)
    return a, b, c


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        a, b, c = data
        return a @ b


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
