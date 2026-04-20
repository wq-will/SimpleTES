# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Initial program with a simple power law form that can be evolved
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):

    X = np.atleast_2d(np.asarray(data_points))           # (N, F)
    N, F = X.shape
    params = np.asarray(params)

    if params.ndim == 1:
        params = params[None, :]                         # (1, P)
    T, P = params.shape

    coeffs    = params[:, :F]                            # (T, F)
    exponents = params[:, F:2*F]                         # (T, F)
    bias      = params[:, -1]                            # (T,)

    pred = (coeffs[None, :, :] * (X[:, None, :] ** exponents[None, :, :])).sum(axis=2) + bias[None, :]

    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_scaling_law(data_points, loss_values):

    X = np.atleast_2d(np.asarray(data_points))           # (N, F)
    y = np.asarray(loss_values)
    N, F = X.shape
    P = 2 * F + 1

    if y.ndim == 1:
        y2d = y[:, None]
    else:
        y2d = y
    T = y2d.shape[1]

    init = np.ones((T, P))

    def objective(flat_params):
        params = flat_params.reshape(T, P)
        pred = scaling_law_func(X, params)               # (N, T)
        mse = np.mean((pred - y2d) ** 2)
        return mse

    result = minimize(objective, init.ravel(), method='BFGS')
    params_opt = result.x.reshape(T, P) if result.success else init

    return params_opt[0] if T == 1 else params_opt
# EVOLVE-BLOCK-END
