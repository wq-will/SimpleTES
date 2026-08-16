# Astrodynamics: Impulsive Gravity-Assist Trajectory Design

This family contains the five historical mission tasks reported in the
SimpleTES paper.  Each mission is an independently discoverable SimpleTES
subtask; evaluator physics and support code are shared at the family root.
The released instances and evolvable seed regions match the paper experiments;
only repository layout and import wiring have been adapted for SimpleTES.

| Subtask | Mission instance | Launch C3 (km²/s²) | Archived best cost (km/s) |
|---|---|---:|---:|
| `mariner_10` | Mariner 10 | 36 | 0.326993397642 |
| `voyager_2` | Voyager 2 | 36 | 3.430213614701 |
| `galileo` | Galileo | 20 | 0.795107617370 |
| `cassini` | Cassini | 20 | 0.820129158750 |
| `rosetta` | Rosetta | 30.25 | 1.552967656327 |

## Layout

```text
datasets/astrodynamics/
├── evaluator_core.py, problem_config.py, recorder.py
├── tools_wrapper.py, trajectory_schema.py
├── TrajectoryToolKit/              # attributed source subset; no kernels
├── data/ephemerides/               # downloaded locally; gitignored
└── <mission>/
    ├── init_program.py
    ├── evaluator.py                # binds one explicit instance.json
    ├── instance.json
    └── <mission>.txt
```

Candidate programs run from temporary directories.  The task-level evaluator
therefore passes its explicit instance path to the shared evaluator, which
forwards it through `SIMPLETES_ASTRODYNAMICS_INSTANCE` during candidate import.
No global mission-selection environment variable is required.

## Setup

From the repository root:

```bash
uv sync --project datasets/astrodynamics
uv run python scripts/prepare_task.py --task astrodynamics
```

The second command downloads `de430.bsp` and `naif0012.tls` directly from the
NASA/JPL NAIF archive, verifies their recorded SHA-256 digests, and stores them
under `datasets/astrodynamics/data/ephemerides/`.  The kernels are not committed.

Example launcher invocation:

```bash
python main.py \
  --init-program datasets/astrodynamics/mariner_10/init_program.py \
  --evaluator datasets/astrodynamics/mariner_10/evaluator.py \
  --instruction datasets/astrodynamics/mariner_10/mariner_10.txt \
  --eval-venv datasets/astrodynamics/.venv \
  --model <your-model>
```

The explicit `--eval-venv` is required because `uv sync --project` creates
`datasets/astrodynamics/.venv`, while SimpleTES auto-detection looks for a
family environment named `venv`.

## Licensing and provenance

The SimpleTES task and evaluator integration follow the repository's
AGPL-3.0-or-later license.  The vendored poliastro-derived subset remains
under the upstream MIT license; see `THIRD_PARTY_NOTICES.md`,
`TrajectoryToolKit/LICENSE`, and `TrajectoryToolKit/SOURCE_MAP.md`.
