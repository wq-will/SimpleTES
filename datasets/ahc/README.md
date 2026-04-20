# AtCoder Heuristic Contest Tasks

Two competitive programming optimization problems from AtCoder Heuristic Contests,
adapted for SimpleTES. The LLM evolves C++ programs that are compiled and evaluated
inside Docker against 150 official test cases.

## Tasks

### AHC039 — Purse Seine Fishing (Geometry)

Given 5000 mackerels and 5000 sardines on a 2D plane, construct an axis-aligned
rectilinear polygon that maximizes `mackerels_inside - sardines_inside`.
Constraints: ≤1000 vertices, perimeter ≤400,000, no self-intersection.

- **Initial program**: ALE-Agent 5th-place solution (~3700 avg per case)
- **Target**: 5000 avg per case
- **Score**: `combined_score` = average score per test case (total_score / 150)

### AHC058 — Apple Incremental Game (Scheduling)

Manage a hierarchy of machines over 500 turns to maximize apple production.
Level 0 machines produce apples; higher-level machines produce lower-level machines.
Each turn: strengthen one machine or do nothing.

- **Initial program**: Empty (starts from scratch)
- **Target**: 6,500,000 avg per case
- **Score**: `combined_score` = average score per test case (total_score / 150)

## Setup

### 1. Download cache (test inputs + tester binaries)

```bash
cd datasets/ahc
bash get_cache.sh
```

### 2. Pull Docker image

```bash
docker pull yimjk/ale-bench:cpp20-202301
```

### 3. Verify

```bash
# Check cache
ls cache/public_inputs_150/ahc039_inputs/ | head -3
ls cache/tester_binaries/

# Make tester binaries executable
chmod +x cache/tester_binaries/*

# Check Docker
docker run --rm yimjk/ale-bench:cpp20-202301 g++-12 --version

# Test evaluator (from repo root)
export AHC_CACHE_DIR=$(pwd)/datasets/ahc/cache
export AHC_CASE_WORKERS=4
python3 -c "
from datasets.ahc.ahc039.evaluator import evaluate
print(evaluate('datasets/ahc/ahc039/init_program.py'))
"
# Expected: combined_score ~3700 (average score per case)
```

## Running

Every run script must set these environment variables at the top:

```bash
export AHC_CACHE_DIR=$(pwd)/datasets/ahc/cache
export AHC_CASE_WORKERS=24
export AHC_DOCKER_TIMEOUT=180

# Limit BLAS threads to prevent CPU contention
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

### AHC039

```bash
python main.py \
  --init-program datasets/ahc/ahc039/init_program.py \
  --evaluator datasets/ahc/ahc039/evaluator.py \
  --instruction datasets/ahc/ahc039/ahc039.txt \
  --model openai/gpt-oss-120b \
  --max-generations 25600 \
  --k-candidates 8 \
  --num-chains 8 \
  --num-inspirations 3 \
  --temperature 0.7 \
  --max-tokens 32768 \
  --backpressure-multiplier 4 \
  --inspiration-policy balance \
  --gen-concurrency 8 \
  --eval-concurrency 8 \
  --timeout 300 \
  --eval-timeout 180 \   # needs to be this
  --log-interval 1024
```

### AHC058

```bash
python main.py \
  --init-program datasets/ahc/ahc058/init_program.py \
  --evaluator datasets/ahc/ahc058/evaluator.py \
  --instruction datasets/ahc/ahc058/ahc058.txt \
  --model openai/gpt-oss-120b \
  --max-generations 25600 \
  --k-candidates 8 \
  --num-chains 8 \
  --num-inspirations 3 \
  --temperature 0.7 \
  --max-tokens 32768 \
  --backpressure-multiplier 4 \
  --inspiration-policy balance \
  --gen-concurrency 8 \
  --eval-concurrency 8 \
  --timeout 300 \
  --eval-timeout 180 \  # needs to be this
  --log-interval 1024
```

## Tuning

### CPU allocation

Each eval spawns one Docker container running `AHC_CASE_WORKERS` parallel test cases.
With `--eval-concurrency N`, you have N containers × workers processes total.

Target ~2 cores per process: `eval_concurrency × AHC_CASE_WORKERS ≤ total_CPUs / 2`

| CPUs | eval_concurrency | AHC_CASE_WORKERS | Cores/proc |
|------|-----------------|------------------|------------|
| 384  | 8               | 24               | 2.0        |
| 128  | 4               | 16               | 2.0        |
| 64   | 4               | 8                | 2.0        |

### Eval timing

- AHC039: ~15s per eval (SA runs for ~2s per case × 150 cases / 24 workers)
- AHC058: ~1-2s per eval (greedy/beam search completes quickly)

## Extracting final submission

Use the provided extraction script:

```bash
python datasets/ahc/extract_code.py ahc039 path/to/best_program.py
# → writes submission_ahc039.cpp

python datasets/ahc/extract_code.py ahc058 path/to/best_program.py
# → writes submission_ahc058.cpp
```

Submit the `.cpp` file to AtCoder using language **C++23 (GCC 15.2.0)**.

Before submitting, verify the program uses a single global timer that stays under 1.9s
wall time. Programs with multiple timers may pass local eval but TLE on AtCoder.

## Evaluation details

| Aspect | Details |
|--------|---------|
| Docker image | yimjk/ale-bench:cpp20-202301 |
| Compiler | g++-12 -std=gnu++20 -O2 -mtune=native -march=native + boost/gmp/eigen3/ac-library |
| Execution | timeout 4.5s wall, prlimit --cpu=3s, /usr/bin/time for profiling, sync after run |
| Score parsing | `re.match(r"Score = (\d+)")` from tester stderr |
| Test cases | 150 cached inputs per problem (downloaded via get_cache.sh) |
| Score | Average score per case = total_score / 150 |

## File layout

```
datasets/ahc/
├── README.md              # This file
├── get_cache.sh           # Downloads test inputs + tester binaries
├── docker_runner.py       # Runs inside Docker (compile + run + judge)
├── extract_code.py        # Extracts submission-ready C++ from best program
├── cache/                 # Downloaded by get_cache.sh
│   ├── public_inputs_150/
│   │   ├── ahc039_inputs/ # 150 test inputs
│   │   └── ahc058_inputs/ # 150 test inputs
│   └── tester_binaries/
│       ├── ahc039_tester  # Official scorer
│       └── ahc058_tester  # Official scorer
├── ahc039/
│   ├── init_program.py    # Initial C++ (evolve block)
│   ├── evaluator.py       # Host-side evaluator (calls Docker)
│   └── ahc039.txt         # Problem description for LLM
└── ahc058/
    ├── init_program.py    # Empty init (evolve block)
    ├── evaluator.py       # Host-side evaluator (calls Docker)
    └── ahc058.txt         # Problem description for LLM
```