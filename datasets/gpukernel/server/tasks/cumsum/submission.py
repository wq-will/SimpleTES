import torch
from typing import TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


def custom_kernel(data: input_t) -> output_t:
    x, output = data
    output[...] = torch.cumsum(x, dim=1)
    return output
