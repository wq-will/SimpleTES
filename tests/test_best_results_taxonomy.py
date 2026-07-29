from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BEST_RESULTS = REPO_ROOT / "best_results"
PAPER_RESULTS = {
    "quantum_compilation": (
        "qubit_routing_on_superconducting_quantum_computer",
        "compilation_for_zoned_neutral_atom_quantum_architectures",
    ),
    "astrodynamics": ("mariner_10", "voyager_2", "galileo", "cassini", "rosetta"),
    "scientific_algorithms": (
        "lasso_regularization_path",
        "zapbench_forecasting_h1",
        "zapbench_forecasting_h4",
        "zapbench_forecasting_h8",
        "zapbench_forecasting_h16",
        "zapbench_forecasting_h32",
        "single_cell_rna_seq_denoising",
    ),
    "ai_foundations": (
        "trimul",
        "asymmetric_matrix_multiplication",
        "batched_cumsum",
        "parallel_scaling_law",
        "domain_mixture_scaling_law",
        "learning_rate_and_batch_size_scaling_law",
        "easy_question_u_shaped_scaling_law",
    ),
    "mathematics_discovery": (
        "erdos_minimum_overlap",
        "second_autocorrelation_inequality",
        "third_autocorrelation_inequality",
        "sum_difference_problem",
        "circle_packing_in_a_unit_square_n26",
        "circle_packing_in_a_unit_square_n32",
        "hadamard_maximum_determinant_order_29",
    ),
}
EXTRA_RESULTS = (
    "scientific_algorithms/ahc039_purse_seine_fishing",
    "scientific_algorithms/ahc058_apple_production_planning",
    "mathematics_discovery/first_autocorrelation_inequality",
)


def test_best_results_uses_the_five_paper_domains() -> None:
    domains = {path.name for path in BEST_RESULTS.iterdir() if path.is_dir()}

    assert domains == set(PAPER_RESULTS)


def test_paper_taxonomy_covers_exactly_28_problems() -> None:
    assert sum(map(len, PAPER_RESULTS.values())) == 28


@pytest.mark.parametrize(
    ("domain", "task"),
    tuple(
        (domain, task)
        for domain, tasks in PAPER_RESULTS.items()
        for task in tasks
    ),
)
def test_each_paper_problem_has_a_result_location(domain: str, task: str) -> None:
    assert (BEST_RESULTS / domain / task).is_dir()


@pytest.mark.parametrize("relative_path", EXTRA_RESULTS)
def test_additional_released_results_remain_available(relative_path: str) -> None:
    assert (BEST_RESULTS / relative_path).is_dir()


@pytest.mark.parametrize(
    "task_path",
    (
        "astrodynamics/gravity_assist_trajectory_design",
        "zapbench/whole_brain_forecasting",
    ),
)
def test_new_dataset_packages_have_import_locations(task_path: str) -> None:
    assert (REPO_ROOT / "datasets" / task_path).is_dir()
