# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Initial program with a simple linear form that can be evolved
"""
import numpy as np

def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    params = np.asarray(params, dtype=float)

    if params.ndim == 1:
        params = params[None, :]

    bias = params[:, 0]
    weights = params[:, 1:]
    pred = bias[None, :] + X @ weights.T
    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float)

    design = np.column_stack([np.ones(X.shape[0]), X])

    if y.ndim == 1:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coeffs

    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs.T
# EVOLVE-BLOCK-END
