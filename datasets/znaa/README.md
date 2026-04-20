# Zoned Neutral-Atom Architecture Compilation

Task family for quantum-circuit compilation to a zoned neutral-atom architecture. The evolved program schedules gates, analyses qubit reuse, places qubits across zones, and routes moves between them — the four components `scheduler_class`, `reuse_analyzer_class`, `placer_class`, `router_class` returned by `run_code()` in `init_program.py`.

| Task | Problem | What to evolve |
|------|---------|----------------|
| **znaa/znaa** | Compile a benchmark quantum circuit to a valid, low-stage ZNAA execution plan | A `Placer` (and optionally `Scheduler` / `ReuseAnalyzer` / `Router`) class conforming to the `AbstractPlacer` interface in `utils_ae.py` |

The evaluator runs the candidate against a suite of 36 benchmark circuits shipped under `znaa/benchmark/` (bv, qft, adder, ising, cat, ghz, knn, etc.), validating correctness against trusted baselines in `.evaluator_baselines_zac.json`.

## Setup

```bash
cd datasets/znaa
bash setup.sh       # creates venv/ and installs requirements.txt
```

Or via the launcher:

```bash
python scripts/prepare_task.py --task znaa
```

The venv lives at `datasets/znaa/venv/` (gitignored). SimpleTES auto-detects it when the evaluator is under `datasets/znaa/znaa/`.

## Running

```bash
python main.py \
  --init-program datasets/znaa/znaa/init_program.py \
  --evaluator    datasets/znaa/znaa/evaluator.py \
  --instruction  datasets/znaa/znaa/znaa.txt \
  --model <your-model>
```

Score aggregates per-circuit metrics (fidelity, stage count, timing) into a single `combined_score`; see `znaa/evaluator.py` for the weighting.
