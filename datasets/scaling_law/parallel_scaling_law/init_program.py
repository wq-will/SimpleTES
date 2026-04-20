# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Initial program with a simple log-linear form that can be evolved
"""
import numpy as np

def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    params = np.asarray(params, dtype=float)

    log_num_params = np.log(np.maximum(X[:, 0], 1.0))
    log_parallel_size = np.log(np.maximum(X[:, 1], 1.0))

    if params.ndim == 1:
        params = params[None, :]

    pred = (
        params[None, :, 0]
        + log_num_params[:, None] * params[None, :, 1]
        + log_parallel_size[:, None] * params[None, :, 2]
    )
    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float)

    design = np.column_stack([
        np.ones(X.shape[0]),
        np.log(np.maximum(X[:, 0], 1.0)),
        np.log(np.maximum(X[:, 1], 1.0)),
    ])

    if y.ndim == 1:
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        return params

    params = []
    for i in range(y.shape[1]):
        coeffs, *_ = np.linalg.lstsq(design, y[:, i], rcond=None)
        params.append(coeffs)
    return np.asarray(params)
# EVOLVE-BLOCK-END
