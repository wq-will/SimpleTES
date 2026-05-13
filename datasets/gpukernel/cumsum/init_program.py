# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl
from typing import TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


def custom_kernel(data: input_t) -> output_t:
    """
    Inclusive cumulative sum (scan) along dim=1.
    Args:
        data: Tuple of (input tensor, output buffer of same shape)
    Returns:
        output tensor containing the inclusive cumsum along dim=1
    """
    x, output = data
    output[...] = torch.cumsum(x, dim=1)
    #return output
# EVOLVE-BLOCK-END
