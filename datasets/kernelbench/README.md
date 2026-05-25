# KernelBench

Triton-kernel tasks adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench). Evolves a Triton kernel; the evaluator measures latency against the reference PyTorch implementation.

| Task | Level / # | Op | Reference |
|------|-----------|-----|-----------|
| **kernelbench/level1_89_cumsum** | L1 #89 | Batched prefix sum | `level1_89_cumsum.triton` |
| **kernelbench/level1_9_Tall_skinny_matrix_multiplication_** | L1 #9 | Tall-skinny matmul | `level1_9_..._matrix_multiplication_.triton` |

Output must match the reference element-wise within `EVAL_TOLERANCE = 1e-2` (absolute and relative). Score is `1 / geomean(latency)` over each subtask's `eval_config.py` configurations. Lower-precision dtypes (FP16, BF16) are allowed if the output stays in tolerance.

## Architecture

GPU tasks use a separate compiler-evaluation server instead of running candidates in the main evaluation subprocess. `evaluator.py` is a thin HTTP client that posts candidates to one of the servers in `server_info.yaml` (default `localhost:8000`).

### Launching the server

```bash
python datasets/kernelbench/server_dev/launch_server.py \
  --port 8000 \
  --num-gpus 8 \
  --backend triton
```

See `--help` (compile-vs-eval GPU split, worker count, precision, timeout).

## Requirements

- NVIDIA GPU(s) with a Triton-compatible CUDA toolchain
- Python + `torch` + `triton`

## Running a task

Once the server is up:

```bash
python main.py \
  --init-program datasets/kernelbench/level1_89_cumsum/init_program.py \
  --evaluator    datasets/kernelbench/level1_89_cumsum/evaluator.py \
  --instruction  datasets/kernelbench/level1_89_cumsum/level1_89_cumsum.triton \
  --model        <your-model>
```

Kernel tasks pass the reference Triton kernel as the instruction. The LLM sees what semantics to match, not a natural-language description.
