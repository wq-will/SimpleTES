# SimpleTES Tasks

Task packages are organized by implementation family and aligned with [`best_results/`](../best_results). The launcher auto-discovers anything at `datasets/<family>/<subtask>/init_program.{py,cpp,rs,...}`.

- [Catalogue](#catalogue) — what's here
- [Designing a task](#designing-a-task) — the file contract + worked example

---

## Catalogue

<table>
<thead>
<tr>
  <th align="left">Task area</th>
  <th align="left">Family</th>
  <th align="center">Subtasks</th>
  <th align="center">Language</th>
  <th align="center">Setup</th>
</tr>
</thead>
<tbody>

<tr><td rowspan="2"><b>🪐 Quantum circuit compilation</b></td>
    <td><a href="qubit_routing"><code>qubit_routing</code></a></td>
    <td align="center">1</td><td align="center">Rust</td>
    <td align="center"><sub>Rust toolchain</sub></td></tr>
<tr><td><a href="znaa"><code>znaa</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>family venv</sub></td></tr>

<tr><td rowspan="1"><b>🛰️ Astrodynamics</b></td>
    <td><a href="astrodynamics"><code>astrodynamics</code></a></td>
    <td align="center">5</td><td align="center">Python</td>
    <td align="center"><sub>family venv + NAIF kernels</sub></td></tr>

<tr><td rowspan="1"><b>⚡ GPU kernel optimization</b></td>
    <td><a href="gpukernel"><code>gpukernel</code></a></td>
    <td align="center">1</td><td align="center">CUDA / Triton</td>
    <td align="center"><sub>GPU + server</sub></td></tr>

<tr><td rowspan="2"><b>🧮 Algorithm engineering</b></td>
    <td><a href="ahc"><code>ahc</code></a></td>
    <td align="center">2</td><td align="center">C++ in Docker</td>
    <td align="center"><sub>Docker</sub></td></tr>
<tr><td><a href="numerical_tasks"><code>numerical_tasks</code></a></td>
    <td align="center">1</td><td align="center">C++ via Python</td>
    <td align="center"><sub><code>g++</code> + Eigen</sub></td></tr>

<tr><td rowspan="2"><b>📐 Mathematics — extremal analysis</b></td>
    <td><a href="erdos"><code>erdos</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>none</sub></td></tr>
<tr><td><a href="autocorrelation"><code>autocorrelation</code></a></td>
    <td align="center">3</td><td align="center">Python</td>
    <td align="center"><sub>none</sub></td></tr>

<tr><td rowspan="3"><b>🧩 Combinatorial construction</b></td>
    <td><a href="circle_packing"><code>circle_packing</code></a></td>
    <td align="center">2</td><td align="center">Python</td>
    <td align="center"><sub>none</sub></td></tr>
<tr><td><a href="hadamard_maximal_det"><code>hadamard_maximal_det</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>none</sub></td></tr>
<tr><td><a href="sums_diffs"><code>sums_diffs</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>none</sub></td></tr>

<tr><td rowspan="3"><b>🧬 Data science</b></td>
    <td><a href="scaling_law"><code>scaling_law</code></a></td>
    <td align="center">4</td><td align="center">Python</td>
    <td align="center"><sub>HuggingFace cache</sub></td></tr>
<tr><td><a href="zapbench"><code>zapbench</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>GPU + task venv + dataset</sub></td></tr>
<tr><td><a href="open_problems_bio"><code>open_problems_bio</code></a></td>
    <td align="center">1</td><td align="center">Python</td>
    <td align="center"><sub>bundled venv + dataset</sub></td></tr>

</tbody>
</table>

First-time pick: any `Setup: none` row. For setup-heavy families:

```bash
uv run python scripts/prepare_task.py --list
uv run python scripts/prepare_task.py --check
uv run python scripts/prepare_task.py --task scaling_law
```

Each family has its own `README.md` with task-specific assumptions and a run command.

---

## Designing a Task

One task = one directory with three files: the instruction states the constraints, the seed implements the entry function, the evaluator checks constraints and scores.

### Layout

```text
datasets/<family>/<task>/
├── init_program.{py|cpp|rs|...}   required — seed, with EVOLVE-BLOCK markers
├── evaluator.py                   required — scores a candidate
└── <task>.txt                     required — problem statement for the LLM

datasets/<family>/
├── requirements.txt               optional — packages allowed in the evolved code
├── pyproject.toml + venv/         optional — family-local Python env (auto-detected)
├── data_manifest.json             optional — data this family needs
└── README.md                      optional — family-level notes
```

### 1. Seed program

The region between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` is the part that gets evolved. Everything else is the fixed harness — imports, the entry function, anything the evaluator depends on.

```python
# EVOLVE-BLOCK-START
import numpy as np

def construct_circles():
    ...                       # ← the LLM edits this
# EVOLVE-BLOCK-END


def run_code():               # fixed entry point the evaluator calls
    return construct_circles()
```

Requirements: markers paired; entry function in the fixed region; the seed runs and earns a finite, non-zero score. C++ / Rust seeds use the same markers via the host language's comment syntax.

### 2. Evaluator

Runs the candidate in an isolated subprocess, checks constraints, recomputes the score. The framework reads only `combined_score` from the returned dict.

Six structural parts. Only parts 3 and 4 are task-specific; copy an existing evaluator and edit those two.

| # | Part | Per-task? |
|---|------|-----------|
| 1 | Configuration: `TIMEOUT_SECONDS`, concurrency, memory fraction. | shared |
| 2 | Exception classes: `EvaluatorTimeoutError`, `MemoryLimitExceededError`. | shared |
| 3 | `validate_solution(sol)` — every hard constraint with tolerances. | **per task** |
| 4 | `compute_score(sol)` — recompute the score from the solution. | **per task** |
| 5 | `run_with_timeout(path, …)` — subprocess + timeout + memory cap + BLAS thread cap. | shared |
| 6 | `evaluate(path)` — entry; must never raise. | shared |

```python
def evaluate(path: str) -> dict:
    # success
    return {"combined_score": 12.34, "validity": 1.0, "eval_time": 8.7}
    # failure
    return {"combined_score": 0.0, "validity": 0.0, "error": "Timeout: ..."}
```

Design rules:

- Higher-is-better score. Minimising `q`? Use `1 / (eps + q)` (autocorrelation, erdos) or a reference ratio (hadamard).
- Score must discriminate. A 0/1 verdict degenerates into random search; aggregate over cases (ahc, kernelbench) or normalise (kissing-number-style).
- Hack-proof: recompute the score from the solution; never trust a self-reported value.
- `evaluate()` never raises. Wrap in `try/except` and return `combined_score: 0` on failure.

### 3. Instruction file — `<task>.txt`

Shown verbatim to the model. Cover: the problem, every constraint, the objective (max/min + how `combined_score` is computed), resource limits (especially per-evaluation timeout in seconds), any reshaping / discretisation, and the program interface.

Always include the sentence

> *Do this by evolving the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.*

so the code extractor knows the evolved region.

### 4. Optional files

| File | Purpose |
|------|---------|
| `requirements.txt` | Not installed by SimpleTES; package names go into the prompt so the LLM does not hallucinate dependencies. |
| `venv/` (or `<family>/venv/`) | Task-local Python env, auto-detected. Override with `--eval-venv <path>`. |
| `<family>/pyproject.toml` + `uv.lock` | Reproducible family-level env. |
| `data_manifest.json` | Files the task needs; `scripts/prepare_task.py --task <family>` runs declared commands, `--check` verifies. |

Minimal `data_manifest.json`:

```json
{
  "prepare_commands": [
    {"command": ["bash", "setup.sh"], "cwd": ".", "description": "Build local deps"}
  ],
  "required_files": ["my_deps/built_artifact"]
}
```

Don't commit large data; declare it.

---

## Worked Example

End-to-end square-root task.

```text
datasets/my_demo/square_root/
├── init_program.py
├── evaluator.py
└── square_root.txt
```

`init_program.py`:

```python
# EVOLVE-BLOCK-START
def sqrt(x: float) -> float:
    return x / 2          # bad baseline; the LLM will fix it
# EVOLVE-BLOCK-END

def run_code():
    return sqrt
```

`evaluator.py`:

```python
import importlib.util, math, sys, time

TIMEOUT_SECONDS = 30

def evaluate(filepath: str) -> dict:
    try:
        t0 = time.time()
        spec = importlib.util.spec_from_file_location("cand", filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cand"] = mod
        spec.loader.exec_module(mod)
        sqrt = mod.run_code()

        targets = [0.25, 1.0, 2.0, 9.0, 1024.0, 1e-6]
        errors = [abs(sqrt(x) - math.sqrt(x)) for x in targets]
        return {
            "combined_score": -sum(errors),   # higher = better → negate error
            "validity": 1.0,
            "eval_time": time.time() - t0,
            "max_error": max(errors),
        }
    except Exception as e:
        return {"combined_score": 0.0, "validity": 0.0, "eval_time": 0.0, "error": str(e)}
```

`square_root.txt`:

```text
Improve sqrt(x) so it approximates math.sqrt as closely as possible on the
listed targets. You may use the math module but not math.sqrt.

Do this by evolving the code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.
The time limit for each program evaluation is 30 seconds.
```

Run:

```bash
uv run python main.py \
  --init-program datasets/my_demo/square_root/init_program.py \
  --evaluator    datasets/my_demo/square_root/evaluator.py \
  --instruction  datasets/my_demo/square_root/square_root.txt \
  --model        gemini/gemini-2.0-flash \
  --max-generations 30
```

---

## Pre-flight Checklist

- [ ] `<task>.txt`, `run_code()`, and `validate_solution` agree on the interface. Timeout values in `.txt`, `TIMEOUT_SECONDS`, and `--eval-timeout` all match.
- [ ] EVOLVE-BLOCK markers paired; entry function in the fixed region; seed earns a finite, non-zero score locally.
- [ ] Evaluator has subprocess isolation, hard timeout, memory cap. `validate_solution` covers every hard constraint with tolerances. Score is recomputed from the solution. `evaluate()` never raises.
- [ ] `combined_score` is higher-is-better and discriminates (not 0/1).
- [ ] `main.py ... --max-generations 5` completes locally. If the family has setup steps, `scripts/prepare_task.py --check` is clean.
- [ ] Family `README.md` covers problem, scoring, and environment.
