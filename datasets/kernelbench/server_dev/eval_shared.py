"""
Shared evaluation logic for all backends.
"""

import importlib.util
import inspect
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from server_utils import (
    clear_l2_cache,
    get_error_name,
    get_tolerance_for_precision,
    read_file,
    set_seed,
    tprint,
)


DATASETS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SEED_TRIALS = 3
DEFAULT_SEED_STEP = 7


def _exec_baseline_with_local_imports(baseline_path: str, source: str) -> Dict[str, Any]:
    baseline_dir = os.path.dirname(os.path.abspath(baseline_path))
    baseline_context: Dict[str, Any] = {}

    with _temporary_sys_path([baseline_dir]):
        compile(source, baseline_path, "exec")
        exec(source, baseline_context)

    return baseline_context


@contextmanager
def _temporary_sys_path(paths: List[str]):
    inserted_paths: List[str] = []

    for path in paths:
        if path and path not in sys.path:
            sys.path.insert(0, path)
            inserted_paths.append(path)

    try:
        yield
    finally:
        for path in reversed(inserted_paths):
            if sys.path and sys.path[0] == path:
                sys.path.pop(0)
            elif path in sys.path:
                sys.path.remove(path)


def _call_with_supported_kwargs(fn, setting: Dict[str, Any]):
    """Call fn with only supported kwargs from setting."""
    if not callable(fn):
        raise TypeError(f"Expected a callable, got {type(fn).__name__}")

    setting = setting or {}

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return fn()

    params = sig.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if accepts_var_kwargs:
        return fn(**setting) if setting else fn()

    filtered_kwargs = {
        k: v
        for k, v in setting.items()
        if k in params
        and params[k].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    if filtered_kwargs:
        return fn(**filtered_kwargs)

    return fn()

def _load_eval_configs(
    eval_config_path: str,
    tprint_sub,
) -> tuple[List[Dict[str, Any]], float | None, int, int]:
    eval_tolerance = None
    seed_trials = DEFAULT_SEED_TRIALS
    seed_step = DEFAULT_SEED_STEP

    if os.path.exists(eval_config_path):
        try:
            cfg_spec = importlib.util.spec_from_file_location("eval_config", eval_config_path)
            cfg_module = importlib.util.module_from_spec(cfg_spec)
            cfg_spec.loader.exec_module(cfg_module)

            eval_configs = getattr(cfg_module, "EVAL_CONFIGS", None)
            if not isinstance(eval_configs, list):
                eval_configs = [{}]

            raw_eval_tolerance = getattr(cfg_module, "EVAL_TOLERANCE", None)
            if raw_eval_tolerance is not None:
                try:
                    eval_tolerance = float(raw_eval_tolerance)
                except Exception:
                    tprint_sub(
                        f"[CPU] Invalid EVAL_TOLERANCE={raw_eval_tolerance!r} in eval_config.py, using precision default"
                    )
                    eval_tolerance = None

            configured_trials = getattr(cfg_module, "EVAL_SEED_TRIALS", DEFAULT_SEED_TRIALS)
            try:
                seed_trials = max(1, int(configured_trials))
            except Exception:
                seed_trials = DEFAULT_SEED_TRIALS

            configured_seed_step = getattr(cfg_module, "EVAL_SEED_STEP", DEFAULT_SEED_STEP)
            try:
                seed_step = max(1, int(configured_seed_step))
            except Exception:
                seed_step = DEFAULT_SEED_STEP

            tprint_sub(
                f"[CPU] Loaded {len(eval_configs)} setting(s) from eval_config.py "
                f"(seed_trials={seed_trials}, seed_step={seed_step})"
            )
        except Exception as e:
            tprint_sub(f"[CPU] Failed to load eval_config.py ({e}), using default")
            eval_configs = [{}]
    else:
        eval_configs = [{}]

    return eval_configs, eval_tolerance, seed_trials, seed_step


def _seed_sequence(base_seed: int, num_trials: int, seed_step: int) -> List[int]:
    return [int(base_seed) + i * int(seed_step) for i in range(num_trials)]


def _resolve_setting(setting: Dict[str, Any], seed_num: int) -> tuple[Dict[str, Any], int]:
    resolved = dict(setting or {})
    try:
        base_seed = int(resolved.get("seed", seed_num))
    except Exception as e:
        raise ValueError(f"Invalid seed in EVAL_CONFIGS entry: {resolved.get('seed')}") from e

    resolved["seed"] = base_seed
    return resolved, base_seed




BASELINE_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "gpu_kernel_eval", "baseline_cache")

_RUNTIME_GUARD_OBJECTS: List[Tuple[str, Callable[[], Any]]] = [
    ("time.perf_counter", lambda: time.perf_counter),
    ("torch.cuda.synchronize", lambda: torch.cuda.synchronize),
    ("torch.cuda.Event", lambda: torch.cuda.Event),
    ("torch.cuda.empty_cache", lambda: torch.cuda.empty_cache),
    ("torch.manual_seed", lambda: torch.manual_seed),
    ("torch.cuda.manual_seed", lambda: torch.cuda.manual_seed),
    ("torch.randn", lambda: torch.randn),
    ("torch.randint", lambda: torch.randint),
    ("torch.load", lambda: torch.load),
    ("torch.nn", lambda: torch.nn),
]


def _sanitize_cache_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))
    return token[:120] if token else "unknown"


def _collect_runtime_guard_snapshot() -> Dict[str, int]:
    snapshot: Dict[str, int] = {}
    for name, getter in _RUNTIME_GUARD_OBJECTS:
        try:
            snapshot[name] = id(getter())
        except Exception:
            snapshot[name] = -1
    return snapshot


def _detect_runtime_patch(runtime_snapshot: Dict[str, int]) -> List[Dict[str, Any]]:
    patched: List[Dict[str, Any]] = []
    for name, getter in _RUNTIME_GUARD_OBJECTS:
        old_id = runtime_snapshot.get(name, -1)
        try:
            new_id = id(getter())
        except Exception:
            new_id = -1

        if new_id != old_id:
            patched.append({"object": name, "before": old_id, "after": new_id})

    return patched


def _minimal_failure_result(
    *,
    error: Any,
    error_name: str,
    compiled: bool,
) -> Dict[str, Any]:
    return {
        "success": False,
        "compiled": bool(compiled),
        "correctness": False,
        "combined_score": 0.0,
        "error": str(error),
        "error_name": str(error_name),
    }


def _runtime_patch_failure_result(
    *,
    patches: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    profiling: Dict[str, Any],
    start_time: float,
    trusted_perf_counter,
) -> Dict[str, Any]:
    return _minimal_failure_result(
        error=f"Runtime patch detected: {patches}",
        error_name="RuntimePatchDetected",
        compiled=False,
    )


def _baseline_cache_file_path(
    *,
    task: str,
    level_id: int,
    task_id: int,
    precision: str,
    setting_name: str,
    setting_idx: int,
    seed: int,
) -> str:
    case_dir = _sanitize_cache_token(f"{task}_level{level_id}_{task_id}")
    precision_token = _sanitize_cache_token(precision)
    setting_token = _sanitize_cache_token(setting_name)

    cache_dir = os.path.join(
        BASELINE_CACHE_ROOT,
        case_dir,
        precision_token,
    )
    os.makedirs(cache_dir, exist_ok=True)

    filename = f"{setting_token}__idx{setting_idx}__seed{int(seed)}.pt"
    return os.path.join(cache_dir, filename)



def _atomic_torch_save(obj: Any, target_path: str, trusted_torch_save) -> None:
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    trusted_torch_save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def _build_eval_plan(
    eval_configs: List[Dict[str, Any]],
    seed_num: int,
    seed_trials: int,
    seed_step: int,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    used_names: set[str] = set()

    for idx, setting in enumerate(eval_configs):
        raw_setting = setting or {}
        raw_name = str(raw_setting.get("name", f"setting_{idx}"))
        setting_name = raw_name if raw_name not in used_names else f"{raw_name}_{idx}"
        used_names.add(setting_name)

        resolved_setting, base_seed = _resolve_setting(raw_setting, seed_num)
        correctness_seeds = _seed_sequence(base_seed, seed_trials, seed_step)
        perf_seeds = [seed + seed_trials * seed_step for seed in correctness_seeds]

        setting_summary = {k: v for k, v in resolved_setting.items() if k != "name"}
        setting_summary["correctness_seeds"] = correctness_seeds
        setting_summary["perf_seeds"] = perf_seeds

        plan.append(
            {
                "idx": idx,
                "setting_name": setting_name,
                "resolved_setting": resolved_setting,
                "correctness_seeds": correctness_seeds,
                "perf_seeds": perf_seeds,
                "setting_summary": setting_summary,
            }
        )

    return plan


def _prepare_baseline_cache(
    *,
    eval_plan: List[Dict[str, Any]],
    generate_input,
    ref_kernel,
    task: str,
    level_id: int,
    task_id: int,
    precision: str,
    device: torch.device,
    trusted_cuda_synchronize,
    trusted_torch_save,
    trusted_set_seed,
    trusted_perf_counter,
    profiling: Dict[str, Any],
    metadata: Dict[str, Any],
    tprint_sub,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    cache_entries: Dict[Tuple[str, int], Dict[str, Any]] = {}
    hits = 0
    misses = 0

    t0 = trusted_perf_counter()

    for entry in eval_plan:
        setting_name = entry["setting_name"]
        setting_summary = entry["setting_summary"]
        setting_idx = entry["idx"]

        resolved_setting = entry["resolved_setting"]
        correctness_seeds = entry["correctness_seeds"]

        for trial_seed in correctness_seeds:
            cache_path = _baseline_cache_file_path(
                task=task,
                level_id=level_id,
                task_id=task_id,
                precision=precision,
                setting_name=setting_name,
                setting_idx=setting_idx,
                seed=trial_seed,
            )

            if not os.path.exists(cache_path):
                misses += 1
                trial_setting = dict(resolved_setting)
                trial_setting["seed"] = trial_seed

                trusted_set_seed(trial_seed)
                data = _call_with_supported_kwargs(generate_input, trial_setting)

                with torch.no_grad():
                    trusted_set_seed(trial_seed)
                    expected = ref_kernel(data)
                    trusted_cuda_synchronize(device=device)

                _atomic_torch_save(expected, cache_path, trusted_torch_save=trusted_torch_save)
                tprint_sub(
                    f"[CPU] Baseline cache MISS -> generated {setting_name} seed={trial_seed}"
                )
            else:
                hits += 1

            cache_entries[(setting_name, trial_seed)] = {
                "path": cache_path,
                "setting": setting_summary,
            }

    profiling["baseline_cache_prepare_ms"] = (trusted_perf_counter() - t0) * 1000
    metadata["baseline_cache"] = {
        "root": BASELINE_CACHE_ROOT,
        "hits": hits,
        "misses": misses,
        "entries": len(cache_entries),
    }

    tprint_sub(
        f"[CPU] Baseline cache ready: hits={hits}, misses={misses}, "
        f"entries={len(cache_entries)}, time={profiling['baseline_cache_prepare_ms']:.0f}ms"
    )

    return cache_entries


def _compare_outputs_with_tolerance(
    received: Any,
    expected: Any,
    tolerance: float,
    path: str = "output",
) -> tuple[bool, str]:
    if isinstance(received, torch.Tensor) and isinstance(expected, torch.Tensor):
        if received.shape != expected.shape:
            return False, f"{path}: shape mismatch {tuple(received.shape)} != {tuple(expected.shape)}"

        if received.is_floating_point() or expected.is_floating_point():
            if not torch.allclose(received, expected, rtol=tolerance, atol=tolerance, equal_nan=True):
                diff = (received - expected).abs()
                max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
                denom = expected.abs().clamp_min(1e-12)
                max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
                return (
                    False,
                    f"{path}: tensor mismatch (max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, tol={tolerance})",
                )
            return True, ""

        if not torch.equal(received, expected):
            return False, f"{path}: non-floating tensor mismatch"
        return True, ""

    if isinstance(received, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(received) != len(expected):
            return False, f"{path}: length mismatch {len(received)} != {len(expected)}"
        for i, (rv, ev) in enumerate(zip(received, expected)):
            ok, reason = _compare_outputs_with_tolerance(rv, ev, tolerance, f"{path}[{i}]")
            if not ok:
                return False, reason
        return True, ""

    if isinstance(received, dict) and isinstance(expected, dict):
        if set(received.keys()) != set(expected.keys()):
            return False, f"{path}: dict keys mismatch"
        for key in sorted(received.keys(), key=lambda x: str(x)):
            ok, reason = _compare_outputs_with_tolerance(
                received[key],
                expected[key],
                tolerance,
                f"{path}.{key}",
            )
            if not ok:
                return False, reason
        return True, ""

    if isinstance(received, (int, float, np.number)) and isinstance(expected, (int, float, np.number)):
        rv = float(received)
        ev = float(expected)
        abs_diff = abs(rv - ev)
        rel_base = max(abs(ev), 1e-12)
        rel_diff = abs_diff / rel_base
        if abs_diff > tolerance and rel_diff > tolerance:
            return False, f"{path}: scalar mismatch (abs={abs_diff:.3e}, rel={rel_diff:.3e}, tol={tolerance})"
        return True, ""

    if received != expected:
        return False, f"{path}: value mismatch"

    return True, ""


def evaluate_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
) -> Dict:
    """Evaluate from Python file using unified multi-setting flow."""
    return _evaluate_generic(
        cache_path=cache_path,
        level_id=level_id,
        task_id=task_id,
        task=task,
        seed_num=seed_num,
        precision=precision,
        request_id=request_id,
        gpu_id=gpu_id,
    )


def evaluate_correctness_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
) -> Dict:
    """Triton staged mode: correctness-only evaluation."""
    return _evaluate_generic(
        cache_path=cache_path,
        level_id=level_id,
        task_id=task_id,
        task=task,
        seed_num=seed_num,
        precision=precision,
        request_id=request_id,
        gpu_id=gpu_id,
        stage="correctness",
    )


def evaluate_performance_from_python_file(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
) -> Dict:
    """Triton staged mode: performance-only evaluation."""
    return _evaluate_generic(
        cache_path=cache_path,
        level_id=level_id,
        task_id=task_id,
        task=task,
        seed_num=seed_num,
        precision=precision,
        request_id=request_id,
        gpu_id=gpu_id,
        stage="performance",
    )


def _load_cached_expected_output(
    cache_entry: Dict[str, Any],
    trusted_torch_load,
    device: torch.device,
):
    cache_path = cache_entry["path"]
    try:
        return trusted_torch_load(cache_path, map_location=device, weights_only=False)
    except TypeError:
        return trusted_torch_load(cache_path, map_location=device)


def _evaluate_custom_kernel(
    custom_module,
    baseline_context: Dict[str, Any],
    eval_plan: List[Dict[str, Any]],
    baseline_cache_entries: Dict[Tuple[str, int], Dict[str, Any]],
    eval_tolerance: float | None,
    stage: str,
    precision_dtype: torch.dtype,
    gpu_id: int,
    start_time: float,
    profiling: Dict[str, Any],
    metadata: Dict[str, Any],
    runtime_snapshot: Dict[str, int],
    trusted_perf_counter,
    trusted_cuda_synchronize,
    trusted_cuda_empty_cache,
    trusted_set_seed,
    trusted_torch_load,
    tprint_sub,
) -> Dict:
    custom_kernel = getattr(custom_module, "custom_kernel", None)
    if not callable(custom_kernel):
        raise AttributeError("Cannot find callable custom_kernel in custom code")

    generate_input = baseline_context.get("generate_input")
    if not callable(generate_input):
        raise AttributeError("Cannot find callable generate_input in baseline init_program.py")

    tolerance = float(eval_tolerance) if eval_tolerance is not None else get_tolerance_for_precision(precision_dtype)
    device = torch.device("cuda:0")

    def _check_runtime_patch_or_fail():
        patches = _detect_runtime_patch(runtime_snapshot)
        if patches:
            tprint_sub(f"[SECURITY] Runtime patch detected: {patches}")
            return _runtime_patch_failure_result(
                patches=patches,
                metadata=metadata,
                profiling=profiling,
                start_time=start_time,
                trusted_perf_counter=trusted_perf_counter,
            )
        return None

    patch_failure = _check_runtime_patch_or_fail()
    if patch_failure is not None:
        return patch_failure

    if stage in {"full", "correctness"}:
        tprint_sub(f"[GPU] Starting correctness tests for {len(eval_plan)} setting(s)")

        for entry in eval_plan:
            setting_name = entry["setting_name"]
            resolved_setting = entry["resolved_setting"]
            correctness_seeds = entry["correctness_seeds"]

            for trial_idx, trial_seed in enumerate(correctness_seeds):
                patch_failure = _check_runtime_patch_or_fail()
                if patch_failure is not None:
                    return patch_failure

                trial_setting = dict(resolved_setting)
                trial_setting["seed"] = trial_seed

                try:
                    trusted_set_seed(trial_seed)
                    data = _call_with_supported_kwargs(generate_input, trial_setting)

                    with torch.no_grad():
                        trusted_set_seed(trial_seed)
                        output_new = custom_kernel(data)
                        trusted_cuda_synchronize(device=device)

                    cache_entry = baseline_cache_entries[(setting_name, trial_seed)]
                    expected = _load_cached_expected_output(
                        cache_entry,
                        trusted_torch_load=trusted_torch_load,
                        device=device,
                    )

                    good, reason = _compare_outputs_with_tolerance(
                        output_new,
                        expected,
                        tolerance=tolerance,
                    )

                    if not good:
                        return _minimal_failure_result(
                            error=f"{setting_name} trial#{trial_idx} seed={trial_seed}: {reason}",
                            error_name="CorrectnessFailed",
                            compiled=True,
                        )

                except Exception as e:
                    return _minimal_failure_result(
                        error=f"{setting_name} trial#{trial_idx} seed={trial_seed}: {e}",
                        error_name=get_error_name(e),
                        compiled=True,
                    )
                finally:
                    try:
                        trusted_cuda_empty_cache()
                    except Exception:
                        pass

    if stage == "correctness":
        return {
            "success": True,
            "compiled": True,
            "correctness": True,
            "combined_score": 0.0,
        }

    num_perf_trials = 10
    num_warmup = 3
    latency_ms_list: List[float] = []
    perf_errors: List[str] = []

    tprint_sub(f"[GPU] Starting performance tests for {len(eval_plan)} setting(s)")

    for entry in eval_plan:
        setting_name = entry["setting_name"]
        resolved_setting = entry["resolved_setting"]
        perf_seeds = entry["perf_seeds"]

        seed_median_times = []

        for trial_seed in perf_seeds:
            patch_failure = _check_runtime_patch_or_fail()
            if patch_failure is not None:
                return patch_failure

            step = int(metadata.get("seed_step", DEFAULT_SEED_STEP))
            warmup_seed = int(trial_seed) + step

            try:
                with torch.no_grad():
                    for warmup_idx in range(num_warmup):
                        warmup_iter_seed = warmup_seed + warmup_idx * step
                        warmup_setting = dict(resolved_setting)
                        warmup_setting["seed"] = warmup_iter_seed

                        trusted_set_seed(warmup_iter_seed)
                        warmup_data = _call_with_supported_kwargs(generate_input, warmup_setting)

                        trusted_set_seed(warmup_iter_seed)
                        custom_kernel(warmup_data)
                        trusted_cuda_synchronize(device=device)

                    trusted_cuda_empty_cache()
                    times = []

                    for perf_idx in range(num_perf_trials):
                        perf_iter_seed = int(trial_seed) + (num_warmup + perf_idx + 1) * step
                        perf_setting = dict(resolved_setting)
                        perf_setting["seed"] = perf_iter_seed

                        trusted_set_seed(perf_iter_seed)
                        perf_data = _call_with_supported_kwargs(generate_input, perf_setting)

                        trusted_cuda_synchronize(device=device)
                        clear_l2_cache(device=device)

                        trusted_set_seed(perf_iter_seed)
                        t_start = trusted_perf_counter()
                        custom_kernel(perf_data)
                        trusted_cuda_synchronize(device=device)
                        t_end = trusted_perf_counter()
                        times.append((t_end - t_start) * 1000.0)

                seed_median_times.append(float(np.median(times)))

            except Exception as e:
                perf_errors.append(f"{setting_name} seed={trial_seed}: {e}")
                break
            finally:
                try:
                    trusted_cuda_empty_cache()
                except Exception:
                    pass

        if not seed_median_times:
            perf_errors.append(f"{setting_name}: No successful performance trial")
            continue

        latency_ms_list.append(float(np.median(seed_median_times)))

    if not latency_ms_list:
        return _minimal_failure_result(
            error=perf_errors[0] if perf_errors else "No successful performance trial",
            error_name="NoPerformanceTrial",
            compiled=True,
        )

    arr = np.array(latency_ms_list, dtype=np.float64)
    geomean_time = float(np.exp(np.mean(np.log(arr))))
    combined_score = 1.0 / geomean_time if geomean_time > 0 else 0.0

    return {
        "success": True,
        "compiled": True,
        "correctness": True,
        "latency_ms_list": latency_ms_list,
        "latency_geomean_ms": geomean_time,
        "combined_score": combined_score,
    }


def _evaluate_generic(
    cache_path: str,
    level_id: int,
    task_id: int,
    task: str,
    seed_num: int,
    precision: str,
    request_id: str,
    gpu_id: int,
    stage: str = "full",
) -> Dict:
    if stage not in {"full", "correctness", "performance"}:
        raise ValueError(f"Unknown stage: {stage}")

    trusted_perf_counter = time.perf_counter
    trusted_cuda_synchronize = torch.cuda.synchronize
    trusted_cuda_empty_cache = torch.cuda.empty_cache
    trusted_torch_load = torch.load
    trusted_torch_save = torch.save
    trusted_set_seed = set_seed

    start_time = trusted_perf_counter()
    profiling: Dict[str, Any] = {}

    def tprint_sub(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tprint(f"[{timestamp}] [{request_id}] [GPU{gpu_id}]", *args, **kwargs)

    metadata: Dict[str, Any] = {}

    precision_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    precision_dtype = precision_map.get(precision, torch.float32)

    try:
        gpu_kernel_path = os.path.join(DATASETS_PATH, task)
        baseline_dir = next(
            d
            for d in os.listdir(gpu_kernel_path)
            if d.startswith(f"level{level_id}_{task_id}_")
        )
        task_dir = os.path.join(gpu_kernel_path, baseline_dir)
        baseline_path = os.path.join(task_dir, "init_program.py")

        t0 = trusted_perf_counter()
        original_src = read_file(baseline_path)
        if not original_src:
            raise FileNotFoundError(f"Failed to read baseline file: {baseline_path}")

        baseline_context = _exec_baseline_with_local_imports(baseline_path, original_src)

        generate_input = baseline_context.get("generate_input")
        ref_kernel = baseline_context.get("ref_kernel")
        if not callable(generate_input):
            raise AttributeError("Cannot find callable generate_input in baseline init_program.py")
        if not callable(ref_kernel):
            raise AttributeError("Cannot find callable ref_kernel in baseline init_program.py")

        eval_config_path = os.path.join(os.path.dirname(baseline_path), "eval_config.py")
        eval_configs, eval_tolerance, seed_trials, seed_step = _load_eval_configs(eval_config_path, tprint_sub)
        eval_plan = _build_eval_plan(
            eval_configs=eval_configs,
            seed_num=seed_num,
            seed_trials=seed_trials,
            seed_step=seed_step,
        )

        baseline_load_ms = (trusted_perf_counter() - t0) * 1000
        profiling["baseline_model_load_ms"] = baseline_load_ms
        tprint_sub(f"[CPU] Baseline module loaded in {baseline_load_ms:.0f}ms")

        baseline_cache_entries = _prepare_baseline_cache(
            eval_plan=eval_plan,
            generate_input=generate_input,
            ref_kernel=ref_kernel,
            task=task,
            level_id=level_id,
            task_id=task_id,
            precision=precision,
            device=torch.device("cuda:0"),
            trusted_cuda_synchronize=trusted_cuda_synchronize,
            trusted_torch_save=trusted_torch_save,
            trusted_set_seed=trusted_set_seed,
            trusted_perf_counter=trusted_perf_counter,
            profiling=profiling,
            metadata=metadata,
            tprint_sub=tprint_sub,
        )

        runtime_snapshot = _collect_runtime_guard_snapshot()

        t0 = trusted_perf_counter()
        with _temporary_sys_path([task_dir, os.path.dirname(cache_path)]):
            spec = importlib.util.spec_from_file_location("custom_module", cache_path)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

        profiling["custom_model_load_ms"] = (trusted_perf_counter() - t0) * 1000
        tprint_sub(f"[CPU] Custom module loaded in {profiling['custom_model_load_ms']:.0f}ms")

        patches_after_import = _detect_runtime_patch(runtime_snapshot)
        if patches_after_import:
            tprint_sub(f"[SECURITY] Runtime patch detected right after custom import: {patches_after_import}")
            return _runtime_patch_failure_result(
                patches=patches_after_import,
                metadata=metadata,
                profiling=profiling,
                start_time=start_time,
                trusted_perf_counter=trusted_perf_counter,
            )

        profiling["cpu_stage_total_ms"] = (
            profiling.get("baseline_model_load_ms", 0.0)
            + profiling.get("baseline_cache_prepare_ms", 0.0)
            + profiling.get("custom_model_load_ms", 0.0)
        )

        return _evaluate_custom_kernel(
            custom_module=custom_module,
            baseline_context=baseline_context,
            eval_plan=eval_plan,
            baseline_cache_entries=baseline_cache_entries,
            eval_tolerance=eval_tolerance,
            stage=stage,
            precision_dtype=precision_dtype,
            gpu_id=gpu_id,
            start_time=start_time,
            profiling=profiling,
            metadata=metadata,
            runtime_snapshot=runtime_snapshot,
            trusted_perf_counter=trusted_perf_counter,
            trusted_cuda_synchronize=trusted_cuda_synchronize,
            trusted_cuda_empty_cache=trusted_cuda_empty_cache,
            trusted_set_seed=trusted_set_seed,
            trusted_torch_load=trusted_torch_load,
            tprint_sub=tprint_sub,
        )

    except Exception as e:
        import traceback

        tprint_sub(f"FAILED: {e}")
        traceback.print_exc()

        profiling["total_ms"] = (trusted_perf_counter() - start_time) * 1000
        metadata["profiling"] = profiling
        metadata["error"] = str(e)
        metadata["error_name"] = type(e).__name__

        return _minimal_failure_result(
            error=metadata["error"],
            error_name=metadata["error_name"],
            compiled=False,
        )

    finally:
        try:
            trusted_cuda_empty_cache()
        except Exception:
            pass
