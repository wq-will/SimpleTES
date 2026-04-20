"""GPUMode TriMul spec text for prompting.

Derived from the official GPUMode TriMul task description; augmented with explicit
I/O + entrypoint requirements so the LLM knows exactly what to generate.
"""

from __future__ import annotations

from pathlib import Path


def _read_vendored_trimul_reference_submission_py() -> str:
    """
    Load the vendored reference `submission.py` so prompts can include a concrete baseline.
    This is best-effort; tasks should still function if the file is missing.
    """
    try:
        p = Path(__file__).resolve().parent / "submission.py"
        return p.read_text()
    except Exception:
        return ""


_TRIMUL_REFERENCE_SUBMISSION_PY = _read_vendored_trimul_reference_submission_py().strip()
_TRIMUL_REFERENCE_BLOCK = (
    ("\nReference code (baseline `submission.py`):\n\n" + _TRIMUL_REFERENCE_SUBMISSION_PY + "\n")
    if _TRIMUL_REFERENCE_SUBMISSION_PY
    else ""
)


TRIMUL_SPEC_TEXT_CUDA = """GPUMode TriMul (CUDA submission)

Task:
- Optimize the *forward* pass of the outgoing Triangle Multiplicative Update (TriMul).
- You are allowed to use *mixed precision* computations, but make sure your final output is in float32.
- You do not have to implement everything in CUDA, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your improved implementation.

Data interface:
- IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:
  <header_file name="kernel.h">
  </header_file>

  <cuda_file name="kernel.cu">
  ... content ...
  </cuda_file>

  <cpp_file name="main.cpp">
  - implement `torch::Tensor run(torch::Tensor input, torch::Tensor mask, py::dict weights, py::dict config)`
  in main.cpp and expose it via PYBIND11_MODULE.
  </cpp_file>

Inputs:
- input_tensor: torch.Tensor, shape [B, S, S, D], dtype float32 (typically CUDA)
- mask: torch.Tensor, shape [B, S, S], dtype float32 or bool (0/1)
- weights: Dict[str, torch.Tensor] (float32, on the same device as input)
  Required keys:
  - norm.weight, norm.bias
  - left_proj.weight, right_proj.weight
  - left_gate.weight, right_gate.weight
  - out_gate.weight
  - to_out_norm.weight, to_out_norm.bias
  - to_out.weight
- config: Dict with keys:
  - dim: int (D)
  - hidden_dim: int

Output:
- torch.Tensor, shape [B, S, S, D], dtype float32

Correctness:
- Must match reference within typical tolerances: rtol=2e-2, atol=2e-2.
""" + _TRIMUL_REFERENCE_BLOCK


TRIMUL_SPEC_TEXT_TRITON = """GPUMode TriMul (Triton submission)
Task:
- Optimize the *forward* pass of the outgoing Triangle Multiplicative Update (TriMul).
- Primary goal: Analyze the shapes of the test cases and find the best hybrid Triton + PyTorch approach to achive high performance.
  - **Fuse** operators to minimize large intermediate tensors, global-memory traffic, and kernel launch overhead.
  - Use a **mixed precision** strategy where appropriate (e.g. fp16 for math and storage, fp32 for accumulations). Only the returned tensor **must be float32**.
  - You do **not** need to re-implement everything in Triton. You may choose to have some of the operations done in pytorch -- to leverage cuBLAS etc.
    - You need to experiment with **many different** hybrid Triton + PyTorch approaches to find the best one. 
    - Analyze on the shapes of the test cases to find the best hybrid approach. 
- Include a short comment at the top summarizing your new implementation.
- For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step.

Data interface:
- Python/Triton: custom_kernel(data) where:
  data = (input_tensor, mask, weights, config)
- Include the code inside '```' and '```' blocks.

Inputs:
- input_tensor: torch.Tensor, shape [B, S, S, D], dtype float32
- mask: torch.Tensor, shape [B, S, S], dtype float32 or bool (0/1)
- weights: Dict[str, torch.Tensor] (float32, on the same device as input)
  Required keys:
  - norm.weight, norm.bias
  - left_proj.weight, right_proj.weight
  - left_gate.weight, right_gate.weight
  - out_gate.weight
  - to_out_norm.weight, to_out_norm.bias
  - to_out.weight
- config: Dict with keys:
  - dim: int (D)
  - hidden_dim: int

Output:
- torch.Tensor, shape [B, S, S, D], dtype float32

Correctness:
- Must match reference within typical tolerances: rtol=2e-2, atol=2e-2.

**Problem Constraints:**
- B ∈ {1,2}, S ∈ {128,256,512,1024}, hidden_dim (H) = 128, dim (D) ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

Test Cases for runtime (optimize runtime for these):
- {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
- {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
- {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
- {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
- {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
- {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
- {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}

Remarks:
- This problem is tricky because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N.
- The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!).
- It is a true test of “fusions” that torch.compile() doesn't do that well.

""" + _TRIMUL_REFERENCE_BLOCK