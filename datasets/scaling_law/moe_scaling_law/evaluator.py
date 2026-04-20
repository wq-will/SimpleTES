# -*- coding: utf-8 -*-
"""
Evaluator for MoE Scaling Law Discovery.

Evaluates scaling law functions that model the relationship between
MoE architecture parameters and validation loss.
"""
import argparse
import concurrent.futures
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import datasets


# --- Configuration ---
HUB_REPO_ID = "pkuHaowei/sldbench"
TASK_NAME = "moe_scaling_law"
FEATURE_NAMES = ["num_experts", "dense_parameter_count"]
TARGET_NAME = "loss_validation"
TIMEOUT_SECONDS = 600
LOCAL_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
MAX_PARAM_COUNT = 6


# --- Core Functions ---

def get_failure_result(error_msg: str = "Evaluation failed or timed out.") -> Dict[str, Any]:
    """Returns a standardized dictionary for failure cases."""
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "r2": -1e6,
        "combined_score": -1e6,
        "error": error_msg,
    }


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds: int = 600):
    """Runs a function with a specified timeout."""
    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as e:
            print(f"Function {func.__name__} timed out or failed: {e}", file=sys.stderr)
            raise


def load_data(train: bool = True) -> Dict[Any, tuple]:
    """Load data from local disk (preferred) or Hugging Face Hub (fallback)."""
    split = 'train' if train else 'test'

    if LOCAL_DATASET_DIR.exists():
        try:
            dataset_dict = datasets.load_from_disk(str(LOCAL_DATASET_DIR))
        except Exception as e:
            raise IOError(f"Failed to load local dataset at '{LOCAL_DATASET_DIR}'. Reason: {e}")

        if split not in dataset_dict:
            raise IOError(f"Split '{split}' not found in local dataset at '{LOCAL_DATASET_DIR}'.")

        dataset = dataset_dict[split]
    else:
        try:
            dataset = datasets.load_dataset(HUB_REPO_ID, name=TASK_NAME, split=split)
        except Exception as e:
            raise IOError(f"Failed to load dataset '{TASK_NAME}' with split '{split}' from '{HUB_REPO_ID}'. Reason: {e}")

    target_names = [TARGET_NAME] if not isinstance(TARGET_NAME, list) else TARGET_NAME

    processed_data = {}
    unique_groups = sorted(list(set(dataset['group'])))

    for group_key in unique_groups:
        group_data = dataset.filter(lambda example: example['group'] == group_key)

        X_list = [np.array(group_data[fname]) for fname in FEATURE_NAMES]
        X = np.stack(X_list, axis=1)

        y_list = [np.array(group_data[tname]) for tname in target_names]
        y_stacked = np.stack(y_list, axis=1)
        y = y_stacked.squeeze(axis=1) if y_stacked.shape[1] == 1 else y_stacked

        processed_data[group_key] = (X, y)

    return processed_data


def calculate_final_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> Dict[str, Any]:
    """Calculates evaluation metrics."""
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions or true values to float arrays.")

    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")

    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    if true.shape != pred.shape:
        return get_failure_result(f"Shape mismatch: true values {true.shape} vs. predictions {pred.shape}.")
    if true.size == 0:
        return get_failure_result("Cannot evaluate on empty data.")

    test_mse_per_dim = np.mean((true - pred) ** 2, axis=0)
    test_mae_per_dim = np.mean(np.abs(true - pred), axis=0)

    variance_per_dim = np.var(true, axis=0)
    mean_abs_dev_per_dim = np.mean(np.abs(true - np.mean(true, axis=0)), axis=0)

    nmse_per_dim = np.divide(test_mse_per_dim, variance_per_dim,
                             out=np.full_like(test_mse_per_dim, np.inf),
                             where=variance_per_dim > 1e-9)
    nmae_per_dim = np.divide(test_mae_per_dim, mean_abs_dev_per_dim,
                             out=np.full_like(test_mae_per_dim, np.inf),
                             where=mean_abs_dev_per_dim > 1e-9)

    r2_per_dim = 1.0 - nmse_per_dim
    
    nmse = np.mean(nmse_per_dim)
    nmae = np.mean(nmae_per_dim)
    r2 = 1.0 - nmse

    results = {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": float(r2),
    }

    if true.shape[1] > 1:
        results["nmse_per_dim"] = nmse_per_dim.tolist()
        results["nmae_per_dim"] = nmae_per_dim.tolist()
        results["r2_per_dim"] = r2_per_dim.tolist()

    return results


def _import_program(program_path: str):
    """Imports a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from path: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_parameter_count(params: Any) -> int:
    """Counts the total number of scalar parameter values returned by fit_scaling_law."""
    return int(np.asarray(params).size)


def evaluate(program_path: str, use_test_data: bool = False) -> Dict[str, Any]:
    """
    Evaluate a scaling law program.
    
    By default (use_test_data=False): fits on training data, tests on training data.
    When use_test_data=True: fits on training data, tests on test data.
    
    Args:
        program_path: Path to the user's Python script with scaling law functions.
        use_test_data: If True, evaluate on test data; otherwise evaluate on training data.
    
    Returns:
        A dictionary containing the evaluation results.
    """
    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law
        scaling_law_func = program.scaling_law_func

        # --- FIT on training data ---
        train_data = load_data(train=True)
        if not train_data:
            return get_failure_result("No training data found.")

        fitted_params_map = {}
        for key, (X_train, y_train) in train_data.items():
            params = run_with_timeout(fit_scaling_law, args=(X_train, y_train), timeout_seconds=TIMEOUT_SECONDS)
            param_count = get_parameter_count(params)
            if MAX_PARAM_COUNT is not None and param_count > MAX_PARAM_COUNT:
                return get_failure_result(
                    f"Parameter count exceeded for group '{key}': got {param_count}, "
                    f"maximum allowed is {MAX_PARAM_COUNT}."
                )
            fitted_params_map[key] = params

        # --- EVALUATE on train or test data ---
        eval_data = load_data(train=not use_test_data)
        if not eval_data:
            return get_failure_result("No evaluation data found.")

        all_predictions, all_true_values = [], []
        for key, (X_eval, y_eval) in eval_data.items():
            if key not in fitted_params_map:
                print(f"Warning: No params for eval group '{key}'. Skipping.", file=sys.stderr)
                continue

            params = fitted_params_map[key]
            predictions = run_with_timeout(scaling_law_func, args=(X_eval, params), timeout_seconds=TIMEOUT_SECONDS)
            all_predictions.append(np.asarray(predictions))
            all_true_values.append(np.asarray(y_eval))

        if not all_predictions:
            return get_failure_result("No predictions were generated.")

        final_predictions = np.concatenate(all_predictions)
        final_true_values = np.concatenate(all_true_values)

        return calculate_final_metrics(final_predictions, final_true_values)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(str(e))


# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluator for MoE Scaling Law Discovery.")
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Program: {args.program_path} ---")
    print(f"Task: {TASK_NAME}")
    
    # When run from command line: fit on train, evaluate on test
    final_results = evaluate(args.program_path, use_test_data=True)

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ EVALUATION FAILED ⛔ ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ Final Test Results ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R²):        {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")
    
    if "nmse_per_dim" in final_results:
        print("\n  --- Per-Dimension Metrics ---")
        nmse_vals = final_results["nmse_per_dim"]
        nmae_vals = final_results["nmae_per_dim"]
        r2_vals = final_results["r2_per_dim"]
        for i, (nmse_d, nmae_d, r2_d) in enumerate(zip(nmse_vals, nmae_vals, r2_vals)):
            print(f"    Dim {i+1}: NMSE={nmse_d:.4f}, NMAE={nmae_d:.4f}, R²={r2_d:.4f}")

    print("--------------------------")
