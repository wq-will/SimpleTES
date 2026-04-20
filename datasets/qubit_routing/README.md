# Qubit Routing on Superconducting Quantum Computers

Task family for swap-reduction in quantum-circuit routing. The evolved program is a **Rust** routing policy that minimises added SWAP gates when mapping logical qubits onto a physical coupling graph.

| Task | Problem | What to evolve |
|------|---------|----------------|
| **qubit_routing/swap_reduction** | Given a qubit coupling graph and a circuit of two-qubit gates, output a low-SWAP execution order | Rust source (`init_program.rs`) implementing the SWAP-scoring routine consumed by `router_cli` |

The evaluator extracts the `EVOLVE-BLOCK` from the candidate `.rs` file, drops it into `rust/router_core/src/candidate.rs`, compiles `router_cli` via `cargo build --release`, and runs it against a suite of benchmark circuits (SABRE, QASMBench) under `swap_reduction/python/benchmarks/circuits/`.

## Requirements

- **Rust toolchain** on PATH (`cargo`, `rustc`) — no Python venv needed; the evaluator only uses stdlib
- The Rust crates (`rust/router_core`, `rust/router_cli`) are checked in; first run compiles them

## Running

The task is Rust-native, so `main_wizard.py` will list it (the wizard globs `init_program.{py,rs,cpp}`). You can also invoke `main.py` directly:

```bash
python main.py \
  --init-program datasets/qubit_routing/swap_reduction/init_program.rs \
  --evaluator    datasets/qubit_routing/swap_reduction/evaluator.py \
  --instruction  datasets/qubit_routing/swap_reduction/swap_reduction.txt \
  --model <your-model>
```

Score is `1 / mean(swap_overhead_ratio)` across the benchmark circuits; the evaluator reports per-circuit SWAP counts and validation results in the metrics dict.
