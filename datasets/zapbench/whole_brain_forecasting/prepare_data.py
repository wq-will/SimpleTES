"""Prepare cached ZAPBench training and validation arrays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _import_zapbench():
    try:
        import numpy as np  # noqa: F401
        import tensorstore as ts  # noqa: F401
        from zapbench import constants
        from zapbench import data_utils
    except ImportError as e:
        sys.stderr.write(
            "ERROR: ZAPBench import failed. Activate the venv created by "
            "datasets/zapbench/setup.sh first.\n"
            f"Underlying error: {e}\n"
        )
        sys.exit(2)
    return constants, data_utils


def _open_trace_array(constants, data_utils):
    """Open the canonical traces matrix from GCS as a tensorstore Future."""
    import tensorstore as ts

    spec = data_utils.get_spec(constants.TIMESERIES_NAME)
    return ts.open(spec).result()


def _make_train_array(traces, constants, data_utils, context_len: int):
    """Build the per-condition concatenated training array.

    Returns (X_train: np.ndarray, condition_lengths: list[int]).
    """
    import numpy as np

    pieces = []
    lengths = []
    for cond in constants.CONDITIONS_TRAIN:
        inc, exc = data_utils.get_condition_bounds(cond)
        inc_t, exc_t = data_utils.adjust_condition_bounds_for_split(
            "train", inc, exc, num_timesteps_context=context_len
        )
        arr = traces[inc_t:exc_t, :].read().result()
        pieces.append(np.asarray(arr, dtype=np.float32))
        lengths.append(int(exc_t - inc_t))
        print(
            f"  cond {cond} ({constants.CONDITION_NAMES[cond]}): "
            f"train slice [{inc_t}, {exc_t}) = {exc_t - inc_t} timesteps"
        )
    X_train = np.concatenate(pieces, axis=0)
    return X_train, lengths


def _make_val_windows(traces, constants, data_utils, context_len: int, horizon: int):
    """Build stride-1 (4-step input, 32-step target) val windows from every
    training condition's validation slice.

    Returns (predict_inputs, targets) each shape (n_windows, ..., 71721) float32.
    """
    import numpy as np

    inputs = []
    targets = []
    for cond in constants.CONDITIONS_TRAIN:
        inc, exc = data_utils.get_condition_bounds(cond)
        inc_v, exc_v = data_utils.adjust_condition_bounds_for_split(
            "val", inc, exc, num_timesteps_context=context_len
        )
        n_windows = data_utils.get_num_windows(inc_v, exc_v, context_len)
        if n_windows <= 0:
            print(f"  cond {cond}: skipping (val slice too short for any window)")
            continue
        # Load the full val slice including the trailing horizon window.
        arr = traces[inc_v : exc_v, :].read().result()
        arr = np.asarray(arr, dtype=np.float32)
        # Stride-1 windows: starts in [0, n_windows)
        ctx = np.stack([arr[s : s + context_len] for s in range(n_windows)], axis=0)
        tgt = np.stack(
            [arr[s + context_len : s + context_len + horizon] for s in range(n_windows)],
            axis=0,
        )
        inputs.append(ctx)
        targets.append(tgt)
        print(f"  cond {cond}: val windows = {n_windows}")
    predict_inputs = np.concatenate(inputs, axis=0)
    target_array = np.concatenate(targets, axis=0)
    return predict_inputs, target_array


def _mean_baseline_mae(predict_inputs, target_array):
    """MAE of predicting all future steps = per-window per-neuron context mean."""
    import numpy as np

    # Mean over the 4-step context axis, then repeat across horizon.
    mean_pred = predict_inputs.mean(axis=1, keepdims=True)  # (n_windows, 1, F)
    mean_pred = np.broadcast_to(mean_pred, target_array.shape)  # (n, 32, F)
    return float(np.mean(np.abs(mean_pred.astype(np.float64) - target_array.astype(np.float64))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--context-len",
        type=int,
        default=4,
        help="Number of input timesteps (ERA paper uses 4; ZAPBench allows up to 256).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction window length (default: ZAPBench's PREDICTION_WINDOW_LENGTH=32).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "zapbench_cache",
        help="Where to write the cached .npy + json files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing cached arrays.",
    )
    args = parser.parse_args()

    import numpy as np

    constants, data_utils = _import_zapbench()
    horizon = args.horizon if args.horizon is not None else constants.PREDICTION_WINDOW_LENGTH

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "X_train": out_dir / "X_train.npy",
        "X_train_condition_lengths": out_dir / "X_train_condition_lengths.npy",
        "val_predict_inputs": out_dir / "val_predict_inputs.npy",
        "val_targets": out_dir / "val_targets.npy",
        "mean_baseline": out_dir / "mean_baseline_mae_val.json",
    }
    if not args.force and all(p.exists() for p in required.values()):
        print(f"All cached files present in {out_dir}. Use --force to regenerate.")
        return

    print("Opening ZAPBench trace matrix from GCS...")
    traces = _open_trace_array(constants, data_utils)
    print(
        f"  shape={tuple(traces.shape)}, dtype={traces.dtype}, "
        f"context_len={args.context_len}, horizon={horizon}"
    )

    print("Building concatenated training array (CONDITIONS_TRAIN)...")
    X_train, lengths = _make_train_array(traces, constants, data_utils, args.context_len)
    print(f"  X_train shape={X_train.shape}, dtype={X_train.dtype}")

    print("Building val prediction windows...")
    predict_inputs, val_targets = _make_val_windows(
        traces, constants, data_utils, args.context_len, horizon
    )
    print(
        f"  predict_inputs={predict_inputs.shape}, val_targets={val_targets.shape}"
    )

    print("Computing mean-baseline MAE on val targets...")
    mae_baseline = _mean_baseline_mae(predict_inputs, val_targets)
    print(f"  mean-baseline MAE = {mae_baseline:.6f}")

    np.save(required["X_train"], X_train)
    np.save(required["X_train_condition_lengths"], np.asarray(lengths, dtype=np.int32))
    np.save(required["val_predict_inputs"], predict_inputs)
    np.save(required["val_targets"], val_targets)
    with open(required["mean_baseline"], "w") as f:
        json.dump(
            {
                "mae": mae_baseline,
                "context_len": args.context_len,
                "horizon": horizon,
                "n_val_windows": int(predict_inputs.shape[0]),
                "n_neurons": int(predict_inputs.shape[-1]),
            },
            f,
            indent=2,
        )

    print(f"Wrote {len(required)} files to {out_dir}")


if __name__ == "__main__":
    main()
