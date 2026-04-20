# Best-known solutions

The highest-scoring program SimpleTES discovered on each task in the paper — 21 tasks organized by the six domains of §3 of the paper. One subdirectory per task, each containing the evolved program (`*_best.py`, `*_best.cpp`, or `*_best.rs`) and, where applicable, the concrete evaluated artifact (`*_best_construction.json`).

## Quantum Circuit Compilation

| Task | What it is |
|---|---|
| `quantum_circuit_compilation/qubit_routing_on_superconducting_quantum_computer/` | Routing policy for two-qubit gates on a superconducting chip, minimizing added SWAPs (Rust) |
| `quantum_circuit_compilation/compilation_for_zoned_neutral_atom_quantum_architectures/` | Gate scheduling for a zoned neutral-atom quantum architecture, minimizing total stage count |

## GPU Kernel Optimization

| Task | What it is |
|---|---|
| `gpu_kernel_optimization/trimul/` | CUDA kernel for triangular matrix multiplication (H100, ms) |
| `gpu_kernel_optimization/batched_cumsum/` | CUDA kernel for batched prefix-sum (H100, ms) |
| `gpu_kernel_optimization/asymmetric_matrix_multiplication/` | CUDA kernel for asymmetric matmul |

## Algorithm Engineering

| Task | What it is |
|---|---|
| `algorithm_engineering/lasso_regularization_path/` | LASSO solver along a full regularization path (ms) |
| `algorithm_engineering/ahc039_purse_seine_fishing/` | AtCoder Heuristic Contest 039 (Purse Seine Fishing), C++ |
| `algorithm_engineering/ahc058_apple_production_planning/` | AtCoder Heuristic Contest 058 (Apple Production Planning), C++ |

## Mathematics Extremal Analysis

| Task | What it is |
|---|---|
| `mathematics_extremal_analysis/erdos_minimum_overlap/` | Erdős minimum overlap problem — constructions minimizing the overlap statistic |
| `mathematics_extremal_analysis/first_autocorrelation_inequality/` | First autocorrelation inequality — functions minimizing the self-convolution ratio |
| `mathematics_extremal_analysis/second_autocorrelation_inequality/` | Second autocorrelation inequality |
| `mathematics_extremal_analysis/third_autocorrelation_inequality/` | Third autocorrelation inequality |

## Combinatorial Construction

| Task | What it is |
|---|---|
| `combinatorial_construction/sum_difference_problem/` | Sum-difference set constructions maximizing $\|A+A\| / \|A-A\|$ |
| `combinatorial_construction/circle_packing_in_a_unit_square_n26/` | 26 non-overlapping circles packed in a unit square, maximizing the minimum radius |
| `combinatorial_construction/circle_packing_in_a_unit_square_n32/` | Same task at $N = 32$ |
| `combinatorial_construction/hadamard_maximum_determinant_order_29/` | $\pm 1$ matrix of order 29 maximizing $\|\det\|$ |

## Data Science

| Task | What it is |
|---|---|
| `data_science/parallel_scaling_law/` | Symbolic scaling-law extrapolation on the `parallel` split |
| `data_science/domain_mixture_scaling_law/` | Scaling law on the `domain_mixture` split |
| `data_science/learning_rate_and_batch_size_scaling_law/` | Scaling law on the `lr & bsz` split |
| `data_science/easy_question_u_shaped_scaling_law/` | Scaling law on the `u_shape` split |
| `data_science/single_cell_rna_seq_denoising/` | Single-cell RNA-seq denoising policy, evaluated with the OpenProblems benchmark |

## File conventions

| File | What it is |
|---|---|
| `<task>_best.py` | Evolved Python program (most tasks) |
| `<task>_best.cpp` | Evolved C++ program (AHC tasks) |
| `<task>_best.rs` | Evolved Rust program (qubit routing) |
| `<task>_best_construction.json` | Concrete construction the program was evaluated on, stored as tagged JSON (numpy arrays round-trip via `simpletes.construction.decode_construction`) |

Older artifacts tag the JSON with `__simpleevolve_type__`; the current encoder writes `__simpletes_type__`. The decoder accepts both.
