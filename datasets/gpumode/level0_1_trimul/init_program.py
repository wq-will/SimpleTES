# EVOLVE-BLOCK-START
from utils import make_match_reference, DisableCuDNNTF32
from typing import TypedDict, TypeVar, Tuple, Dict

import torch
from torch import nn, einsum
import math

def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    input_tensor, mask, weights, config = data
    # Your optimized implementation



class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)
        
        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)



def ref_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> Tuple[torch.Tensor, Dict]:
    # Use deterministic kernels and disable TF32 for accuracy
    with DisableCuDNNTF32():
        input_tensor, mask, weights, config = data
        trimul = TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"]).to(input_tensor.device)

        # Fill in the given weights of the model
        trimul.norm.weight = nn.Parameter(weights['norm.weight'])
        trimul.norm.bias = nn.Parameter(weights['norm.bias'])
        trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'])
        trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'])
        trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'])
        trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'])
        trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'])
        trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'])
        trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'])
        trimul.to_out.weight = nn.Parameter(weights['to_out.weight'])

        output = trimul(input_tensor, mask)

        return output

# Input generation for the reference code
def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]:

    # Really dumb but for now _ isn't parsing correctly.
    batch_size = bs
    seq_len = seqlen
    hidden_dim = hiddendim
    no_mask = nomask

    config = {
        "hidden_dim": hidden_dim,
        "dim": dim,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    weights = {}

    # Generate input tensor based on distribution
    if distribution == "cauchy":
        # Heavier tail distribution
        input_tensor = torch.distributions.Cauchy(0, 2).sample(
            (batch_size, seq_len, seq_len, dim)
        ).to(device='cuda', dtype=torch.float32)
    else:  # normal distribution
        input_tensor = torch.randn(
            (batch_size, seq_len, seq_len, dim),
            device='cuda',
            dtype=torch.float32,
            generator=gen
        ).contiguous()

    if no_mask:
        mask = torch.ones(batch_size, seq_len, seq_len, device=input_tensor.device)
    else:
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), device=input_tensor.device, generator=gen)

    # Initialize model weights based on distribution
    weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["left_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["out_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["to_out_norm.weight"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    weights["to_out.weight"] = torch.randn(dim, hidden_dim, device="cuda", dtype=torch.float32) / math.sqrt(dim)
    weights["to_out_norm.bias"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

    return (input_tensor, mask, weights, config)

check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
# EVOLVE-BLOCK-END