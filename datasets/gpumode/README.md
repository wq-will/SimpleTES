# GPUMode Kernels

GPU-kernel tasks adapted from [GPUMode](https://github.com/gpu-mode). Evolves a CUDA / Triton kernel; the evaluator measures latency against the reference PyTorch implementation.

| Task | Op | What to evolve |
|------|-----|----------------|
| **gpumode/level0_1_trimul** | TriMul (triangular matrix multiplication) | `custom_kernel(...)` in `init_program.py` — either a CUDA kernel (`.cuda` reference) or a `torch` implementation (`.torch` reference) |

Output must match the reference element-wise within `EVAL_TOLERANCE = 2e-2` (absolute and relative). Score is `1 / geomean(latency)` over the 7 `EVAL_CONFIGS` (varying batch / dim / seqlen / dtype) declared in `eval_config.py`.

## Architecture

GPU tasks use a separate compiler-evaluation server instead of running candidates in the main evaluation subprocess. `evaluator.py` is a thin HTTP client that posts candidates to one of the servers in `server_info.yaml` (default `localhost:8000`).

### Launching the server

```bash
python datasets/gpumode/server_ksearch/launch_server.py \
  --port 8000 \
  --num-gpus 8 \
  --backend cuda            # cuda | torch | triton
```

`--triton-compiler-gpus` / `--triton-eval-gpus` split compile vs eval workloads across cards. See `--help`.

## Requirements

- NVIDIA GPU(s) with `nvcc` on PATH
- Python + `torch`

## Running a task

Once the server is up:

```bash
python main.py \
  --init-program datasets/gpumode/level0_1_trimul/init_program.py \
  --evaluator    datasets/gpumode/level0_1_trimul/evaluator.py \
  --instruction  datasets/gpumode/level0_1_trimul/level0_1_trimul.cuda \
  --model        <your-model>
```

Kernel tasks pass the reference implementation (`.cuda` / `.torch` / `.triton`) as the instruction. The LLM sees what semantics to match, not a natural-language description.
