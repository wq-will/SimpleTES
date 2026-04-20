# EVOLVE-BLOCK-START
import numpy as np
import graphtools
import scprep


def magic_denoise(X, knn=5, t=3, n_pca=100, solver="approximate", decay=1, knn_max=None, random_state=None, n_jobs=1, verbose=False):
    if knn_max is None:
        knn_max = knn * 3
    X_work = scprep.utils.toarray(X).astype(np.float64)
    X_work = np.sqrt(X_work)
    X_work, libsize = scprep.normalize.library_size_normalize(X_work, rescale=1, return_library_size=True)
    graph = graphtools.Graph(
        X_work,
        n_pca=n_pca if X_work.shape[1] > n_pca else None,
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        thresh=1e-4,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    diff_op = graph.diff_op
    if solver == "approximate":
        data = graph.data_nu
    else:
        data = scprep.utils.to_array_or_spmatrix(graph.data)
    if verbose:
        print(f"    [magic_denoise] data shape: {data.shape}, sum: {data.sum():.6f}")
        print(f"    [magic_denoise] diff_op sum: {diff_op.sum():.6f}")
    data_imputed = scprep.utils.toarray(data)
    if t > 0 and diff_op.shape[1] < data_imputed.shape[1]:
        diff_op_t = np.linalg.matrix_power(scprep.utils.toarray(diff_op), t)
        data_imputed = diff_op_t.dot(data_imputed)
        if verbose:
            print(f"    [magic_denoise] used matrix_power path")
    else:
        for _ in range(t):
            data_imputed = diff_op.dot(data_imputed)
        if verbose:
            print(f"    [magic_denoise] used iteration path")
    if verbose:
        print(f"    [magic_denoise] after diffusion sum: {data_imputed.sum():.6f}")
    if solver == "approximate":
        data_imputed = graph.inverse_transform(data_imputed, columns=None)
        if verbose:
            print(f"    [magic_denoise] after inverse_transform sum: {data_imputed.sum():.6f}")
    data_imputed = np.square(data_imputed)
    data_imputed = scprep.utils.matrix_vector_elementwise_multiply(data_imputed, libsize, axis=0)
    return data_imputed
# EVOLVE-BLOCK-END