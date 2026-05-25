# Scaling-Law Discovery

Symbolic regression for LLM scaling laws on [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench). Evolves a parameterised functional form + fitter; the evaluator fits the train split and scores held-out points.

| Subtask | What the law models | `combined_score` | TIMEOUT |
|---------|---------------------|------------------|---------|
| **scaling_law/domain_mixture_scaling_law** | How multi-domain loss depends on domain mixture proportions across model sizes | $R^2$ on the held-out fit | 600 s |
| **scaling_law/easy_question_scaling_law** | Easy-question accuracy vs compute (FLOPs) — the U-shaped / double-descent regime | $R^2$ on the held-out fit | 600 s |
| **scaling_law/lr_bsz_scaling_law** | Training loss as a function of learning rate, batch size, data size, and model parameters | $R^2$ on the held-out fit | 600 s |
| **scaling_law/parallel_scaling_law** | LM loss vs model parameters under parallel scaling (the `parallel_size` axis) | $R^2$ on the held-out fit | 600 s |

All four subtasks have a hard cap of 35 parameters. Exceeding the cap, or failing to load / fit / converge, records `combined_score = -1e6`.

The evaluator also reports per-dim `nmse`, `nmae`, `r2` for analysis. Only `r2` (averaged across output dimensions) drives the score.

## Setup

The benchmark is fetched from HuggingFace Hub on first use:

```bash
python scripts/prepare_task.py --task scaling_law
# or, directly:
cd datasets/scaling_law && python prepare_dataset.py
```

Pre-caches each subtask's split into `.sldbench_cached/` (gitignored). HF cache respects `HF_HOME` / `HF_DATASETS_CACHE`.

## Requirements

`numpy`, `scipy`, `datasets` (HuggingFace), `scikit-learn`. See `datasets/scaling_law/requirements.txt`.

## Running

```bash
python main.py \
  --init-program datasets/scaling_law/parallel_scaling_law/init_program.py \
  --evaluator    datasets/scaling_law/parallel_scaling_law/evaluator.py \
  --instruction  datasets/scaling_law/parallel_scaling_law/parallel_scaling_law.txt \
  --model <your-model>
```
