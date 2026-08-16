# Best-known solutions

Result artifacts are organized by the five scientific domains used in the paper. Each benchmark problem has one subdirectory for its evolved program (usually `*_best.py`, `*_best.cpp`, or `*_best.rs`) and, where applicable, its concrete evaluated artifact (`*_best_construction.json`). The ZAPBench releases retain the filename `best_program.py`.

## Quantum Compilation

| Task | What it is |
|---|---|
| `quantum_compilation/qubit_routing_on_superconducting_quantum_computer/` | Routing policy for two-qubit gates on a superconducting chip, minimizing added SWAPs (Rust) |
| `quantum_compilation/compilation_for_zoned_neutral_atom_quantum_architectures/` | Compilation policy for a zoned neutral-atom architecture; the paper reports geometric-mean execution time |

## Astrodynamics

| Task | What it is |
|---|---|
| [`astrodynamics/mariner_10/`](astrodynamics/mariner_10/) | Gravity-assist trajectory design for Mariner 10 |
| [`astrodynamics/voyager_2/`](astrodynamics/voyager_2/) | Gravity-assist trajectory design for Voyager 2 |
| [`astrodynamics/galileo/`](astrodynamics/galileo/) | Gravity-assist trajectory design for Galileo |
| [`astrodynamics/cassini/`](astrodynamics/cassini/) | Gravity-assist trajectory design for Cassini |
| [`astrodynamics/rosetta/`](astrodynamics/rosetta/) | Gravity-assist trajectory design for Rosetta |

## Scientific Algorithms

| Task | What it is |
|---|---|
| `scientific_algorithms/lasso_regularization_path/` | LASSO solver along a full regularization path (ms) |
| [`scientific_algorithms/zapbench_forecasting_h1/`](scientific_algorithms/zapbench_forecasting_h1/) | Whole-brain activity forecasting at horizon 1 |
| [`scientific_algorithms/zapbench_forecasting_h4/`](scientific_algorithms/zapbench_forecasting_h4/) | Whole-brain activity forecasting at horizon 4 |
| [`scientific_algorithms/zapbench_forecasting_h8/`](scientific_algorithms/zapbench_forecasting_h8/) | Whole-brain activity forecasting at horizon 8 |
| [`scientific_algorithms/zapbench_forecasting_h16/`](scientific_algorithms/zapbench_forecasting_h16/) | Whole-brain activity forecasting at horizon 16 |
| [`scientific_algorithms/zapbench_forecasting_h32/`](scientific_algorithms/zapbench_forecasting_h32/) | Whole-brain activity forecasting at horizon 32 |
| `scientific_algorithms/single_cell_rna_seq_denoising/` | Single-cell RNA-seq denoising policy, evaluated with the OpenProblems benchmark |

Additional released artifacts: [`scientific_algorithms/ahc039_purse_seine_fishing/`](scientific_algorithms/ahc039_purse_seine_fishing/) and [`scientific_algorithms/ahc058_apple_production_planning/`](scientific_algorithms/ahc058_apple_production_planning/).

## AI Foundations

| Task | What it is |
|---|---|
| `ai_foundations/trimul/` | Triton kernel for triangular matrix multiplication (headline result on H100, ms) |
| `ai_foundations/asymmetric_matrix_multiplication/` | Triton kernel for asymmetric matmul (H200, ms) |
| `ai_foundations/batched_cumsum/` | Triton kernel for batched prefix-sum (H200, ms) |
| `ai_foundations/parallel_scaling_law/` | Symbolic scaling-law extrapolation on the `parallel` split |
| `ai_foundations/domain_mixture_scaling_law/` | Scaling law on the `domain_mixture` split |
| `ai_foundations/learning_rate_and_batch_size_scaling_law/` | Scaling law on the `lr & bsz` split |
| `ai_foundations/easy_question_u_shaped_scaling_law/` | Scaling law on the `u_shape` split |

## Mathematics Discovery

| Task | What it is |
|---|---|
| `mathematics_discovery/erdos_minimum_overlap/` | Erdős minimum overlap problem — constructions minimizing the overlap statistic |
| `mathematics_discovery/second_autocorrelation_inequality/` | Second autocorrelation inequality |
| `mathematics_discovery/third_autocorrelation_inequality/` | Third autocorrelation inequality |
| `mathematics_discovery/sum_difference_problem/` | Sum-difference set constructions maximizing $|A+A| / |A-A|$ |
| `mathematics_discovery/circle_packing_in_a_unit_square_n26/` | 26 non-overlapping circles packed in a unit square, maximizing the sum of radii |
| `mathematics_discovery/circle_packing_in_a_unit_square_n32/` | Same task at $N = 32$ |
| `mathematics_discovery/hadamard_maximum_determinant_order_29/` | $\pm 1$ matrix of order 29 maximizing $|\det|$ |

Additional released artifact: [`mathematics_discovery/first_autocorrelation_inequality/`](mathematics_discovery/first_autocorrelation_inequality/).

## File conventions

| File | What it is |
|---|---|
| `<task>_best.py` | Evolved Python program (most tasks) |
| `best_program.py` | Evolved Python program (ZAPBench result artifacts) |
| `<task>_best.cpp` | Evolved C++ program (AHC tasks) |
| `<task>_best.rs` | Evolved Rust program (qubit routing) |
| `<task>_best_construction.json` | Concrete construction the program was evaluated on, stored as tagged JSON (numpy arrays round-trip via `simpletes.construction.decode_construction`) |

Older artifacts tag the JSON with `__simpleevolve_type__`; the current encoder writes `__simpletes_type__`. The decoder accepts both.
