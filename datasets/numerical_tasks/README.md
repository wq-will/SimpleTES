# Numerical Optimization Tasks

Shared environment for numerical optimization tasks adapted from [AlgoTune](https://github.com/oripress/AlgoTune).

| Task | Problem | Reference Solver | What to evolve |
|------|---------|-----------------|----------------|
| **lasso_path** | Gaussian Lasso regularization path — solve `min (1/2n) ||y - Xw||² + λ||w||₁` over a decreasing sequence of λ | sklearn `lasso_path` | C++ source (`CPP_CODE`) compiled to a binary that reads the problem and writes a coefficient matrix |

The evolved C++ binary beats sklearn on wall-clock time while matching its solution to high precision. The evaluator handles compilation, subprocess invocation, and timing.

## Setup

```bash
cd datasets/numerical_tasks
bash setup.sh           # creates venv/ and downloads Eigen into lasso_path/eigen/
```

Or via the top-level launcher:

```bash
python scripts/prepare_task.py --task numerical_tasks
```

This installs `requirements.txt` into `datasets/numerical_tasks/venv/` and downloads [Eigen 3.4.0](https://eigen.tuxfamily.org/) into `lasso_path/eigen/` (both are gitignored). `g++` must be on PATH.

## Running

SimpleTES auto-detects `datasets/numerical_tasks/venv/` when the evaluator is under `datasets/numerical_tasks/<subtask>/`:

```bash
python main.py \
  --init-program datasets/numerical_tasks/lasso_path/init_program.py \
  --evaluator    datasets/numerical_tasks/lasso_path/evaluator.py \
  --instruction  datasets/numerical_tasks/lasso_path/lasso_path.txt \
  --model <your-model>
```

The scoring function is `1 / geo_mean(solve_time_ms)` across a suite of synthetic problem sizes and generators; correctness is validated against sklearn with a tolerance on the `max_gap` metric. See `lasso_path/lasso_path.txt` for the exact wire format, problem sizes, and scoring criteria shown to the LLM.
