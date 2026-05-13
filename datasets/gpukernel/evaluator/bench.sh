#!/usr/bin/env bash
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK="trimul"                # task name under server/tasks/, or absolute path
NUM_ITERS=3

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)     TASK="$2";      shift 2 ;;
        --iters)    NUM_ITERS="$2"; shift 2 ;;
        -h|--help)
            cat >&2 <<'HELP'
Usage: bench.sh [OPTIONS]

Benchmark .py kernel files in this directory against any task in server/tasks/.

Options:
  --task NAME|PATH        Task name under server/tasks/, or absolute path.
                          Default: trimul
                          Available tasks (in server/tasks/):
                            trimul
                            gemm
                            sort
                            histogram
                            conv2d
                            prefixsum
                            cumsum
                            shortfatmatmul
                            mladecode
                            mxfp4quant
                            /abs/path/to/custom/task_dir
  --iters N               Number of benchmark iterations per kernel. Default: 3

Adding a custom task:
  Create a directory with a task.yml and pass its path via --task.
  Minimum task.yml fields:
    files:      list of {name, source} entries; "@SUBMISSION@" for the kernel
    config:     {main: "eval.py"}
    benchmarks: list of parameter dicts
HELP
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_KERNEL_DIR="$(cd "$SCRIPT_DIR/../server/tasks" && pwd)"

# ── Resolve task directory ────────────────────────────────────────────────────
if [[ "$TASK" == /* ]] || [[ "$TASK" == ./* ]] || [[ "$TASK" == ../* ]]; then
    TASK_DIR="$(cd "$TASK" 2>/dev/null && pwd)" \
        || { echo "Error: task dir not found: $TASK" >&2; exit 1; }
else
    TASK_DIR="$GPU_KERNEL_DIR/$TASK"
fi

if [ ! -f "$TASK_DIR/task.yml" ]; then
    echo "Error: no task.yml found in $TASK_DIR" >&2
    echo "" >&2
    echo "Available tasks:" >&2
    find "$GPU_KERNEL_DIR" -maxdepth 1 -name "task.yml" \
        | sed "s|$GPU_KERNEL_DIR/||;s|/task.yml||" \
        | sort | sed 's/^/  /' >&2
    find "$GPU_KERNEL_DIR" -maxdepth 2 -mindepth 2 -name "task.yml" \
        | sed "s|$GPU_KERNEL_DIR/||;s|/task.yml||" \
        | sort | sed 's/^/  /' >&2
    exit 1
fi

# ── Generate benchfile from task.yml ─────────────────────────────────────────
# Converts YAML benchmark entries → the "key: value; ..." format used by eval.py
BENCHFILE=$(mktemp /tmp/gpukernel_bench_XXXXXX.txt)

read -r TASK_NAME NUM_BENCHMARKS < <(python3 - "$TASK_DIR/task.yml" "$BENCHFILE" <<'PYEOF'
import sys, yaml

task_yml_path, benchfile_path = sys.argv[1], sys.argv[2]
with open(task_yml_path) as f:
    cfg = yaml.safe_load(f)

benchmarks = cfg.get("benchmarks", [])
lines = []
for bm in benchmarks:
    parts = []
    for k, v in bm.items():
        # Booleans → Python-style True/False; strings unquoted; ints as-is
        parts.append(f"{k}: {v}")
    lines.append("; ".join(parts))

with open(benchfile_path, "w") as bf:
    bf.write("\n".join(lines) + "\n")

import os
task_name = cfg.get("name") or os.path.basename(task_yml_path.rstrip("/").rsplit("/task.yml", 1)[0])
print(task_name, len(benchmarks))
PYEOF
)

echo "Task:       $TASK_NAME"
echo "Directory:  $TASK_DIR"
echo "Benchmarks: $NUM_BENCHMARKS  |  Iterations: $NUM_ITERS"
echo ""

# ── Discover .py kernel files ─────────────────────────────────────────────────
KERNELS=()
KERNEL_FILES=()
for pyfile in "$SCRIPT_DIR"/*.py; do
    [ -f "$pyfile" ] || continue
    name="$(basename "$pyfile" .py)"
    KERNELS+=("$name")
    KERNEL_FILES+=("$pyfile")
done

if [ ${#KERNELS[@]} -eq 0 ]; then
    echo "No .py kernel files found in $SCRIPT_DIR" >&2
    exit 1
fi

echo "Found ${#KERNELS[@]} kernels: ${KERNELS[*]}"
echo ""

RESULTS_DIR=$(mktemp -d /tmp/gpukernel_results_XXXXXX)
CACHE_DIR="/tmp/triton_cache"

# ── Run each kernel ───────────────────────────────────────────────────────────
for i in "${!KERNELS[@]}"; do
    name="${KERNELS[$i]}"
    kfile="${KERNEL_FILES[$i]}"

    echo "============================================================"
    echo "  Kernel: $name  ($NUM_ITERS run(s))"
    echo "============================================================"

    # Clean triton cache before each kernel for fair autotuning
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    export TRITON_CACHE_DIR="$CACHE_DIR"

    for iter in $(seq 1 $NUM_ITERS); do
        outfile="$RESULTS_DIR/${name}_run${iter}.txt"

        # Build a per-run sandbox by copying all files listed in task.yml,
        # resolving paths relative to $TASK_DIR.
        # This handles tasks whose eval.py/utils.py live in a parent dir
        # (e.g. pmpp_v2/) as well as tasks that are self-contained (bioml/trimul).
        SANDBOX=$(python3 - "$TASK_DIR" "$kfile" <<'PYEOF'
import sys, yaml, shutil, tempfile, os

task_dir   = sys.argv[1]
kernel_src = sys.argv[2]

with open(os.path.join(task_dir, "task.yml")) as f:
    cfg = yaml.safe_load(f)

sandbox = tempfile.mkdtemp(prefix="/tmp/gpukernel_sandbox_")
for entry in cfg.get("files", []):
    dest = os.path.join(sandbox, entry["name"])
    source = entry["source"]
    if source == "@SUBMISSION@":
        shutil.copy2(kernel_src, dest)
    else:
        src_path = os.path.normpath(os.path.join(task_dir, source))
        shutil.copy2(src_path, dest)

print(sandbox)
PYEOF
)

        EVAL_MAIN=$(python3 -c "
import yaml, sys
with open('$TASK_DIR/task.yml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('config', {}).get('main', 'eval.py'))
")

        (
            cd "$SANDBOX"
            POPCORN_FD=3 python3 "$EVAL_MAIN" leaderboard "$BENCHFILE" \
                3>"$outfile" 2>&1 || true
        )

        rm -rf "$SANDBOX"
        echo "  Run $iter done"
    done
    echo ""
done

# ── Parse and display results ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Parsing results..."
echo "============================================================"

KERNEL_LIST=$(IFS=,; echo "${KERNELS[*]}")

python3 - "$RESULTS_DIR" "$NUM_ITERS" "$KERNEL_LIST" <<'PYEOF'
import sys, os, math

results_dir = sys.argv[1]
num_iters   = int(sys.argv[2])
kernels     = sys.argv[3].split(",")

def parse_file(path):
    d = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    d[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return d

# Discover benchmark count and spec labels from the first available result file
num_bm, bm_labels = 0, []
for k in kernels:
    d = parse_file(os.path.join(results_dir, f"{k}_run1.txt"))
    cnt = int(d.get("benchmark-count", 0))
    if cnt:
        num_bm = cnt
        bm_labels = [d.get(f"benchmark.{i}.spec", f"benchmark {i}") for i in range(cnt)]
        break

if not num_bm:
    print("No benchmark results found – check kernel output above for errors.")
    sys.exit(1)

MAX_LABEL = 56
bm_labels_short = [lb[:MAX_LABEL] for lb in bm_labels]

# Collect per-run stats
all_means, all_geos = {}, {}
for kname in kernels:
    all_means[kname], all_geos[kname] = [], []
    for r in range(1, num_iters + 1):
        d = parse_file(os.path.join(results_dir, f"{kname}_run{r}.txt"))
        means = []
        for bi in range(num_bm):
            m = d.get(f"benchmark.{bi}.mean")
            means.append(float(m) / 1e6 if m is not None else None)  # ns -> ms
        all_means[kname].append(means)
        valid = [v for v in means if v is not None]
        all_geos[kname].append(
            math.exp(sum(math.log(v) for v in valid) / len(valid)) if valid else None
        )

col_w   = max(max(len(k) for k in kernels) + 4, 26)
lbl_w   = max(len(lb) for lb in bm_labels_short)
total_w = lbl_w + 2 + col_w * len(kernels)

def sep(c="="): return c * total_w

# Per-benchmark table
print()
print(sep())
print(f"{'PER-BENCHMARK RESULTS (mean +/- std across ' + str(num_iters) + ' run(s), ms)':^{total_w}s}")
print(sep())
hdr = f"{'Benchmark':>{lbl_w}s}"
for k in kernels:
    hdr += f"  {k:^{col_w}s}"
print(hdr)
print(sep("-"))

for bi in range(num_bm):
    row = f"{bm_labels_short[bi]:>{lbl_w}s}"
    for k in kernels:
        vals = [all_means[k][r][bi] for r in range(num_iters) if all_means[k][r][bi] is not None]
        if vals:
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((v-m)**2 for v in vals) / (len(vals)-1)) if len(vals) > 1 else 0.0
            row += f"  {f'{m:8.3f} +/- {s:<5.3f} ms':^{col_w}s}"
        else:
            row += f"  {'ERR':^{col_w}s}"
    print(row)

# Geo mean summary table
print()
print(sep())
print(f"{'GEOMETRIC MEAN ACROSS ' + str(num_iters) + ' RUN(S)':^{total_w}s}")
print(sep())
hdr2 = f"{'':>15s}"
for k in kernels:
    hdr2 += f"  {k:^{col_w}s}"
print(hdr2)
geo_w = 15 + 2 + col_w * len(kernels)
print("-" * geo_w)

for r in range(num_iters):
    row = f"{'Run ' + str(r+1):>15s}"
    for k in kernels:
        g = all_geos[k][r]
        row += f"  {f'{g:8.3f} ms':^{col_w}s}" if g is not None else f"  {'ERR':^{col_w}s}"
    print(row)

print("-" * geo_w)
row_avg = f"{'Avg Geo Mean':>15s}"
row_std = f"{'Std Geo Mean':>15s}"
for k in kernels:
    geos = [g for g in all_geos[k] if g is not None]
    if geos:
        avg = sum(geos) / len(geos)
        std = math.sqrt(sum((g-avg)**2 for g in geos) / (len(geos)-1)) if len(geos) > 1 else 0.0
        row_avg += f"  {f'{avg:8.3f} ms':^{col_w}s}"
        row_std += f"  {f'{std:8.4f} ms':^{col_w}s}"
    else:
        row_avg += f"  {'N/A':^{col_w}s}"
        row_std += f"  {'N/A':^{col_w}s}"
print(row_avg)
print(row_std)
print()
PYEOF

rm -rf "$RESULTS_DIR"
rm -f "$BENCHFILE"
echo "Done."
