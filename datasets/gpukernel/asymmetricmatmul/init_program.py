# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl
from typing import TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    c[...] = a @ b
    # return c
# EVOLVE-BLOCK-END
