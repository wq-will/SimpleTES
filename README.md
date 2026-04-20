<img src="picture/will-typography-c.png" alt="WILL" height="20">

<div align="center">

# SimpleTES

</div>

**Evaluation-driven Scaling for Scientific Discovery.** SimpleTES is the reference instantiation of the TES framework: a training-free search loop that discovers state-of-the-art solutions across 21 scientific problems in six domains using open-source models like `gpt-oss-120b`.

---

## 1. Overview

If test-time compute scales generation, what are the right dimensions along which to scale the **evaluation-driven loop itself**? TES decomposes a total evaluation budget $N$ along three orthogonal axes:

| Axis | Operation | Why it matters |
|---|---|---|
| **$C$** — parallel exploration | Run $C$ independent trajectories concurrently | Breadth is where genuinely new directions enter |
| **$L$** — feedback-driven refinement | Revise each trajectory over $L$ committed steps, shaped by the evaluator | Concrete signals let partial progress compound |
| **$K$** — local selection | Score $K$ candidates per step; commit only the best | Selection at the right granularity keeps compute focused |

The axes interact: $N \approx C \cdot K \cdot L$. Allocating well — not scaling any single axis — is what produces SOTA. On 21 tasks spanning mathematics, AtCoder heuristic contests, neural scaling laws, data-science optimization, GPU kernels, and quantum compilation, SimpleTES on `gpt-oss-20b` / `gpt-oss-120b` beats frontier baselines (Claude Opus, GPT-5) and tuned domain pipelines.

## 2. How it runs

Each trajectory maintains a history $S$ of evaluated nodes $(y, r, m)$ — solution, scalar reward, evaluator metadata. At every step a **context construction** $\Phi$ picks a subset of $S$ and formats the proposal $x = \Phi(S)$; the generator returns $K$ candidates; each is evaluated in an isolated subprocess; the best commits to $S$. The scheduler dispatches generation and evaluation asynchronously with trajectory-level backpressure. Default selector: **RPUCG**, a DAG-aware PUCT using $U_i = \max(r_i,\, \gamma \max_{j \in \mathrm{Ch}(i)} U_j)$ with percentile-rank priors.

---

## 3. Install

Requires Python ≥ 3.11.

```bash
uv sync                      # or: pip install -e .
uv sync --extra vllm         # optional: vLLM token-forcing backend
export GEMINI_API_KEY=...    # or OPENAI_API_KEY / ANTHROPIC_API_KEY / ...
```

## 4. Quickstart

A 50-step run on circle packing with $N = 20$:

```bash
uv run python main.py \
  --init-program datasets/circle_packing/circle_packing_20/init_program.py \
  --evaluator    datasets/circle_packing/circle_packing_20/evaluator.py \
  --instruction  datasets/circle_packing/circle_packing_20/circle_packing_20.txt \
  --max-generations 50 \
  --model gemini/gemini-2.0-flash
```

Checkpoints land under `checkpoints/<date>/instance-<id>/`; resume with `--resume <instance_dir>`.

Tasks needing external data or a task-local venv (`ahc`, `numerical_tasks`, `scaling_law`, `open_problems_bio`, `znaa`) need one extra step:

```bash
python scripts/prepare_task.py --list      # see status
python scripts/prepare_task.py --task ahc  # download one family
```

For an interactive launcher that browses the benchmark tree and walks through the knobs, run `uv run python main_wizard.py` (`--dry-run` to print the command only; `--save-profile <name>` / `--load-profile <name>` to persist and replay configs).

## 5. Tuning the $(C, K, L, \Phi)$ budget

Core knobs map directly to the scaling axes:

| Flag | Symbol | Typical | Notes |
|---|---|---|---|
| `--num-chains` | $C$ | 4–32 | Parallel trajectories, each with its own $S$ |
| `--k-candidates` | $K$ | 1–4 | Candidates per step; best commits to $S$ |
| `--max-generations` | $N$ | 100–4096 | Total eval budget; $L = \lfloor N / (CK) \rfloor$ |
| `--restart-every-n` | — | 7 | Per-trajectory restart period |
| `--selector` | $\Phi$ | `rpucg` | Context construction; RPUCG is the paper default |

Concurrency is independent of the axes: set `--gen-concurrency` by LLM rate limit, `--eval-concurrency` by CPU/GPU. `--gen-concurrency = --num-chains` recovers a one-at-a-time-per-trajectory baseline.

### Selectors ($\Phi$)

All selectors operate on the trajectory's own $S$ and differ only in which historical nodes they condition on.

| Selector | How it scores $S$ |
|---|---|
| `balance` | Ratio sampling across exploitation / exploration / elite / random |
| `puct` | UCB with 1-hop value propagation |
| **`rpucg`** | **Paper default.** DAG-aware PUCT with percentile-rank priors and 1-hop neighborhood exclusion |
| `llm_puct`, `llm_rpucg` | Algorithmic shortlist + LLM rerank |
| `llm_elite` | LLM curates a bounded elite pool (ADD / REPLACE / REJECT) |

LLM-driven selectors take a separate selector model:

```bash
--selector llm_elite \
--llm-policy-model openai/gpt-oss-120b \
--llm-policy-api-base http://your-endpoint/v1 \
--llm-policy-pool-size 15
```

For every other flag, run `python main.py --help` for the full `EngineConfig` reference.

---

## 6. Datasets

### `datasets/` — runnable tasks (13 families, 42 subtasks)

Every task lives at `datasets/<family>/<subtask>/` as three files: `init_program.{py,rs,cpp}` (seed), `evaluator.py` returning `{"combined_score": float, ...}`, and `<subtask>.txt` (LLM instruction). The families line up with the six discovery domains of §1:

| Family | What it is |
|---|---|
| `qubit_routing/` | Swap-reduction policy for superconducting chips (Rust) |
| `znaa/` | Circuit compilation to zoned neutral-atom architectures |
| `gpumode/`, `kernelbench/` | CUDA / GPU kernel optimization (TriMul, batched cumsum, asymmetric matmul) |
| `numerical_tasks/` | Lasso-path solver (C++ evolved) |
| `ahc/` | AtCoder Heuristic Contest problems (C++, Docker eval) |
| `erdos/` | Erdős minimum overlap problem |
| `autocorrelation/` | First / second / third autocorrelation inequalities |
| `sums_diffs/` | Sum-and-difference set constructions |
| `circle_packing/` | Unit-square circle packing, $N = 18 \ldots 32$ |
| `hadamard_maximal_det/` | Maximal determinant of $\pm 1$ matrices |
| `scaling_law/` | Neural scaling-law symbolic regression |
| `open_problems_bio/` | Single-cell RNA-seq denoising (OpenProblems benchmark) |

### `best_results/` — 21 paper-featured SOTA programs

The exact programs behind the paper's numbers, organized by the six domains of §1:

| Domain | Tasks |
|---|---|
| `quantum_circuit_compilation/` | Qubit routing on superconducting chips; compilation for zoned neutral-atom architectures |
| `gpu_kernel_optimization/` | TriMul, batched cumsum, asymmetric matmul (CUDA kernels) |
| `algorithm_engineering/` | LASSO regularization path; AHC039 purse-seine fishing; AHC058 apple production planning |
| `mathematics_extremal_analysis/` | Erdős minimum overlap; first / second / third autocorrelation inequalities |
| `combinatorial_construction/` | Sum-difference problem; circle packing in a unit square ($N=26, 32$); Hadamard maximum determinant order 29 |
| `data_science/` | Four scaling-law splits (parallel, domain mixture, lr-and-batch-size, U-shaped); single-cell RNA-seq denoising |

Each task ships the evolved program (`*_best.py`, `*_best.cpp`, or `*_best.rs`) and, when applicable, a concrete artifact (`*_best_construction.json`). See [`best_results/README.md`](best_results/README.md) for per-task notes.

### Add your own task

Drop `init_program.py`, `evaluator.py` (with `evaluate(filepath) -> {"combined_score": float, ...}`), and a `<name>.txt` instruction into a new directory. If the evaluator has extra dependencies, ship a task-local `pyproject.toml` + `uv.lock` + `venv/` under `datasets/<family>/` — SimpleTES auto-detects it, or override with `--eval-venv`. Pre-locked envs already exist for `circle_packing/`, `autocorrelation/`, and `numerical_tasks/`.

---

## Troubleshooting

- **`"No checkpoints found"` when resuming** — point `--resume` at the instance directory, not its parent.
- **Scheduler hangs on `waiting for 30 seconds…`** — a gen worker is blocked on an LLM call. Lower `--timeout`, raise `--retry`, or check the endpoint.
- **Evaluation timeouts** — raise `--eval-timeout` (default 600s); infinite loops are caught by the hard kill.
- **Rate limits** — reduce `--gen-concurrency` or set `--retry 3`.
- **Selector mismatch on resume** — pass the same `--selector` used at save time, or accept the new selector starting fresh on top of preserved nodes.

## License

SimpleTES is released under [GNU AGPL-3.0-or-later](LICENSE), Copyright (C) 2026 WILL.

- **Research & local use**: unrestricted — running SimpleTES, publishing papers, and using programs *discovered* by SimpleTES are all free (the discovered programs themselves are not bound by AGPL).
- **Derivative frameworks / forks**: must be released under AGPL-3.0-or-later with source.
- **Network services**: modifications exposed as a hosted service must offer source to users (§13).
- **Library integration**: combined works are AGPL; reach out if this conflicts with your licensing.

---

<p align="center">
  <img src="picture/will-symbol-c.png" alt="WILL" height="55">
</p>
