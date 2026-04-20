# evaluator.py
# Evaluate one candidate file on benchmark circuits.
# Merge per-circuit metrics into one prefixed output dict.
from __future__ import annotations

import importlib.util
import collections
from contextlib import contextmanager
import inspect
import json
import math
import numpy as np
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable, Optional

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_EVAL_ROOT = Path(__file__).resolve().parent
_LIB_CIRCUITS = _EVAL_ROOT / "benchmark"
_UTILS_AE_PATH = _EVAL_ROOT / "utils_ae.py"
_EVAL_CANDIDATE_MODULE: str = "_ae_evaluated_candidate"
_trusted_hw_baseline_cache: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None

_COMPONENT_NAMES = ("scheduler", "reuseanalyzer", "placer", "router")
_TIME_SEGMENT_NAMES = ("2q", "1q", "move", "transfer")
_FIDELITY_SEGMENT_NAMES = ("2q", "1q", "move", "transfer", "deco")
_FAILURE_REASON_BY_ERROR = {
    "StageFrontierMismatch": "Plan frontier validation failed.",
    "PlacementValidationError": "Grouped placement validation failed.",
    "ReuseValidationError": "Reuse validation failed.",
    "RoutePlacementMismatch": "Router execution placement mismatch.",
    "FinalStorageZoneError": "Final qubit placement must be in storage zone.",
    "GateExecutionMismatch": "Gate execution mismatch.",
    "HardwareConfigTampered": "Invalid hardware configuration.",
    "MachineTimingModelTampered": "Invalid machine timing model.",
}

MIN_QUBIT: int = 0
MAX_QUBIT: int = 300

_EVAL_CIRCUIT_NQ = re.compile(r"_n(\d+)$", re.IGNORECASE)
_BAD_DROP_COORD_RE = re.compile(r"bad\s+drop\s+\{([A-Z])\(([-+]?\d+),([-+]?\d+)\)\}", re.IGNORECASE)

# Total wall-clock budget for one evaluate() run. None or <= 0 disables it.
TOTAL_RUNTIME_TIMEOUT_SEC: float | None = 1500.0

# Per-circuit wall-clock limit. None or <= 0 disables it.
COMPILATION_TIMEOUT_SEC: float | None = None

KEEP_PER_CIRCUIT_INFO: bool = False
KEEP_COMPONENT_TOP3_CIRCUITS: bool = False
RANKED_CIRCUIT_INFO_COUNT: int = 3

SNAPSHOT_PLACEMENT_AS_ZONE_MATRICES: bool = False

FAILURE_CIRCUIT_SCORE: float = -1.0
SCORE_EPS: float = 1e-12
IMPROVEMENT_METRIC: str = "log_ratio"
IMPROVEMENT_EQUAL_TOL: float = 1e-9
NO_IMPROVEMENT_PENALTY: float = 0
BEST_UPDATE_ALPHA: float = 0.8
_BEST_STORE_DIR: Path = _EVAL_ROOT / "evaluator_best"
_BEST_STORE_PATH: Path = _BEST_STORE_DIR / "records.json"
_BEST_PROGRAM_INDEX_PATH: Path = _BEST_STORE_DIR / "program_index.json"
_BEST_PROGRAMS_DIR: Path = _BEST_STORE_DIR / "programs"
_BEST_PLANS_DIR: Path = _BEST_STORE_DIR / "plans"
_BEST_STORE_LOCK_PATH: Path = _BEST_STORE_DIR / ".lock"
_BASELINE_STORE_PATH: Path = _EVAL_ROOT / ".evaluator_baselines.json"
_CIRCUIT_BASELINES: dict[str, dict[str, Any]] = {}

STAGE_ANALYSIS_WINDOW_SIZE: int = 3
STAGE_ANALYSIS_MIN_COMBINED_SCORE: float | None =  -0.2
STAGE_ANALYSIS_MIN_BEST_CIRCUIT_SCORE: float | None = 0.2
STAGE_ANALYSIS_MAX_WORST_CIRCUIT_SCORE: float | None = -0.2
# Primary threshold for worst-case snapshot selection. When set, only circuits
# with score < this threshold are eligible; when None, fallback to
# STAGE_ANALYSIS_MAX_WORST_CIRCUIT_SCORE.
STAGE_ANALYSIS_MAX_SNAPSHOT_CIRCUIT_SCORE: float | None = -0.2
# Threshold for adjacent snapshot window averages.
# When None, fall back to STAGE_ANALYSIS_MAX_WORST_CIRCUIT_SCORE.
STAGE_ANALYSIS_FEEDBACK_INCLUDE_BEST_GATE: bool = False
# Weighted-random snapshot selection temperatures.
# Smaller T1 prefers smaller circuits more strongly.
# Smaller T2 prefers worse adjacent-stage windows more strongly.
STAGE_ANALYSIS_SNAPSHOT_CIRCUIT_QUBIT_T1: float = 20.0
STAGE_ANALYSIS_SNAPSHOT_WINDOW_SCORE_T2: float = 0.2
STAGE_ANALYSIS_PROBABILITY: float = 0.25

# ------------------------------------------------------------------------ #


class HardwareBaselineGuard:
    """Cache and validate the trusted hardware signature."""

    @staticmethod
    def shape_signature(shape: Any) -> tuple[Any, ...] | None:
        if shape is None:
            return None
        try:
            return (
                tuple(int(x) for x in getattr(shape, "storage")),
                tuple(int(x) for x in getattr(shape, "entangling")),
                tuple(int(x) for x in getattr(shape, "readout")),
            )
        except Exception:
            return None

    @staticmethod
    def params_signature(params: Any) -> tuple[Any, ...] | None:
        if params is None:
            return None
        try:
            return (
                float(getattr(params, "time_1q")),
                float(getattr(params, "time_2q")),
                float(getattr(params, "time_readout")),
                float(getattr(params, "time_transfer")),
                float(getattr(params, "fidelity_1q")),
                float(getattr(params, "fidelity_2q")),
                float(getattr(params, "fidelity_readout")),
                float(getattr(params, "fidelity_transfer")),
                float(getattr(params, "fidelity_execution")),
                float(getattr(params, "coherence_time_storage")),
                float(getattr(params, "coherence_time_else")),
                float(getattr(params, "aod_speed")),
                tuple(float(x) for x in getattr(params, "distance_storage")),
                tuple(float(x) for x in getattr(params, "distance_entangle")),
                tuple(float(x) for x in getattr(params, "distance_readout")),
                float(getattr(params, "distance_interzone")),
                float(getattr(params, "rydberg_radius")),
                float(getattr(params, "delta")),
            )
        except Exception:
            return None

    @classmethod
    def trusted_baseline(cls) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
        global _trusted_hw_baseline_cache
        if _trusted_hw_baseline_cache is not None:
            return _trusted_hw_baseline_cache
        if not _UTILS_AE_PATH.is_file():
            return None

        spec = importlib.util.spec_from_file_location("_evaluator_utils_ae_pristine", _UTILS_AE_PATH)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            return None

        config_fn = getattr(mod, "config_test", None)
        if not callable(config_fn):
            return None
        try:
            cfg = config_fn()
        except Exception:
            return None

        shape_sig = cls.shape_signature(cfg)
        params_sig = cls.params_signature(cfg)
        if shape_sig is None or params_sig is None:
            return None
        _trusted_hw_baseline_cache = (shape_sig, params_sig)
        return _trusted_hw_baseline_cache


class CircuitCatalog:
    """Circuit discovery and filename-derived metadata."""

    @staticmethod
    def resolve_existing_path(circuit_path: str | Path) -> Path:
        path = Path(circuit_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Circuit file not found: {circuit_path!r} (resolved {path!r})")
        return path

    @staticmethod
    def qubit_count_from_path(path: Path) -> int | None:
        match = _EVAL_CIRCUIT_NQ.search(path.stem)
        return int(match.group(1)) if match else None

    @classmethod
    def discover(cls, min_qubits: int, max_qubits: int) -> list[str]:
        if not _LIB_CIRCUITS.is_dir():
            return []
        lo, hi = int(min_qubits), int(max_qubits)
        if lo > hi:
            return []

        found: list[tuple[float, Path]] = []
        for path in sorted(_LIB_CIRCUITS.rglob("*.znaa")):
            nq = cls.qubit_count_from_path(path)
            if nq is not None and lo <= nq <= hi:
                resolved_path = path.resolve()
                found.append((cls.circuit_weight(resolved_path), resolved_path))
        found.sort(key=lambda item: (item[0], str(item[1])))
        return [str(path) for _weight, path in found]

    @staticmethod
    def resolve_file(circuit_id: str | Path) -> Path:
        path = Path(circuit_id)
        if path.is_file():
            return path.resolve()
        return (_LIB_CIRCUITS / path).resolve()

    @staticmethod
    def build_run_code_arg(circuit_id: str | Path) -> str:
        path = Path(circuit_id)
        if path.is_absolute():
            return str(path.resolve())
        return path.as_posix().replace("\\", "/")

    @staticmethod
    def circuit_stem(circuit_id: str | Path) -> str:
        return Path(circuit_id).stem

    @staticmethod
    def baseline_key(circuit_id: str | Path) -> str:
        return Path(circuit_id).stem

    @classmethod
    def qubit_count_from_circuit_id(cls, circuit_id: str | Path) -> int | None:
        path = Path(circuit_id)
        nq = cls.qubit_count_from_path(path)
        if nq is not None:
            return nq
        if path.is_file():
            return cls._qubit_count_from_lines(path.resolve())
        return None

    @staticmethod
    def _qubit_count_from_lines(full_path: Path) -> int | None:
        if not full_path.is_file():
            return None
        qmax = -1
        try:
            with full_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    for token in parts[2:]:
                        try:
                            qmax = max(qmax, int(token))
                        except ValueError:
                            continue
        except OSError:
            return None
        return (qmax + 1) if qmax >= 0 else None

    @classmethod
    def circuit_weight(cls, full_path: Path) -> float:
        nq = cls.qubit_count_from_circuit_id(full_path)
        if nq is None or nq <= 0:
            return 0.0

        lq = math.log(float(nq))
        if not math.isfinite(lq):
            return 0.0
        weight = lq
        return float(weight) if math.isfinite(weight) else 0.0


class CandidateContract:
    """Candidate entrypoint validation and evaluator-owned pipeline execution."""

    _REQUIRED_PLAN_KEYS = {
        "stage",
        "reuse_info",
        "placements",
        "gates",
        "operations",
    }
    _METHOD_NAME_BY_COMPONENT = {
        "scheduler_class": "schedule",
        "reuse_analyzer_class": "analyze",
        "placer_class": "place",
        "router_class": "route",
    }

    @classmethod
    def assert_plan_schema(cls, plan: Any) -> None:
        if not isinstance(plan, list) or not plan:
            raise RuntimeError("run_code contract violation: plan must be a non-empty list.")
        for block_index, block in enumerate(plan):
            if not isinstance(block, dict):
                raise RuntimeError(f"run_code contract violation: plan[{block_index}] must be a dict.")
            missing = cls._REQUIRED_PLAN_KEYS.difference(block.keys())
            if missing:
                raise RuntimeError(
                    f"run_code contract violation: plan[{block_index}] missing keys {sorted(missing)}."
                )

    @classmethod
    def validate_component_spec(cls, spec: Any) -> dict[str, Any]:
        """Validate the strict component-spec returned by ``run_code()``."""
        if not isinstance(spec, dict):
            return {
                "ok": False,
                "error": "TypeError",
                "message": "run_code must return a dict component spec.",
                "combined_score": float(FAILURE_CIRCUIT_SCORE),
            }

        missing = sorted(cls._METHOD_NAME_BY_COMPONENT.keys() - spec.keys())
        if missing:
            return {
                "ok": False,
                "error": "KeyError",
                "message": f"run_code component spec missing keys {missing}.",
                "combined_score": float(FAILURE_CIRCUIT_SCORE),
            }

        validated: dict[str, Any] = {}
        for key, method_name in cls._METHOD_NAME_BY_COMPONENT.items():
            component_class = spec[key]
            if not inspect.isclass(component_class):
                return {
                    "ok": False,
                    "error": "TypeError",
                    "message": f"run_code component spec field '{key}' must be a class.",
                    "combined_score": float(FAILURE_CIRCUIT_SCORE),
                }
            entry = getattr(component_class, method_name, None)
            if not callable(entry):
                return {
                    "ok": False,
                    "error": "TypeError",
                    "message": (
                        f"run_code component spec field '{key}' must be a class with a callable '{method_name}' method."
                    ),
                    "combined_score": float(FAILURE_CIRCUIT_SCORE),
                }
            validated[key] = component_class
        return validated

    @staticmethod
    def prepare_import_extras() -> str | None:
        """Load bundled ``utils_ae`` into ``sys.modules`` for candidate imports."""
        sys.modules.pop(_EVAL_CANDIDATE_MODULE, None)
        sys.modules.pop("utils_ae", None)
        if not _UTILS_AE_PATH.is_file():
            return f"Bundled utils_ae.py not found: {_UTILS_AE_PATH}"
        spec_u = importlib.util.spec_from_file_location("utils_ae", _UTILS_AE_PATH)
        if spec_u is None or spec_u.loader is None:
            return f"Could not create import spec for {_UTILS_AE_PATH}"
        mod_u = importlib.util.module_from_spec(spec_u)
        sys.modules["utils_ae"] = mod_u
        try:
            spec_u.loader.exec_module(mod_u)
        except Exception as exc:
            sys.modules.pop("utils_ae", None)
            return f"Failed to load utils_ae: {type(exc).__name__}: {exc}"
        if getattr(mod_u, "ZNAAMachine", None) is None:
            return "Failed to load utils_ae: missing ZNAAMachine symbol."
        return None

    @staticmethod
    def execute_pipeline(component_spec: dict[str, Any], circuit_id: str, mod: Any) -> dict[str, Any]:
        """Build evaluator-owned solver state and compile one circuit."""
        import utils_ae as _ua

        config_factory = _ua.config_test
        scheduler = component_spec["scheduler_class"](config_factory=config_factory)
        reuse_analyzer = component_spec["reuse_analyzer_class"](config_factory=config_factory)
        placer = component_spec["placer_class"](config_factory=config_factory)
        router = component_spec["router_class"](config_factory=config_factory)
        solver = _ua.Solver(
            config_factory=config_factory,
            scheduler=scheduler,
            reuse_analyzer=reuse_analyzer,
            placer=placer,
            router=router,
        )
        mod.solver = solver
        solver.last_solve_intermediates = None
        path = CircuitCatalog.resolve_existing_path(circuit_id)
        loaded = _ua.ZNAACircuit.from_file(path)
        if type(loaded) is not _ua.ZNAACircuit:
            raise RuntimeError("evaluator pipeline violation: circuit must be ZNAACircuit.")
        plan = solver.solve(_ua.ZNAACircuit.from_file(path))
        CandidateContract.assert_plan_schema(plan)
        machine = _ua.ZNAAMachine(hardware_config=config_factory)
        return {"plan": plan, "machine": machine, "circuit": loaded}


class BaselineStore:
    """Persistent baseline cache for per-circuit scoring."""

    @staticmethod
    def _coerce_stage_details(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, list):
            return None
        stage_details: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                stage_details.append(dict(item))
        return stage_details if stage_details else []

    @staticmethod
    def load() -> dict[str, dict[str, Any]]:
        """Load baseline cache and cast numeric fields to float."""
        path = _BASELINE_STORE_PATH
        if not path.is_file():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        out: dict[str, dict[str, Any]] = {}
        for path_key, rec in data.items():
            if not isinstance(rec, dict) or "time_us" not in rec:
                continue
            item: dict[str, Any] = {"time_us": float(rec["time_us"])}
            if "f_all" in rec and rec["f_all"] is not None:
                item["f_all"] = float(rec["f_all"])
            if "f_non12" in rec and rec["f_non12"] is not None:
                item["f_non12"] = float(rec["f_non12"])
            stage_details = BaselineStore._coerce_stage_details(rec.get("two_q_stage_details"))
            if stage_details is not None:
                item["two_q_stage_details"] = stage_details
            out[CircuitCatalog.baseline_key(path_key)] = item
        return out

    @staticmethod
    def save(baselines: dict[str, dict[str, Any]]) -> None:
        """Persist baseline cache to disk without failing evaluation."""
        path = _BASELINE_STORE_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(baselines, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass


class BestPerformanceStore:
    """Persistent best-per-circuit records and associated candidate programs."""

    @staticmethod
    def _nonempty_text_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _best_store_relative_path(path: Path) -> str:
        try:
            return path.relative_to(_BEST_STORE_DIR).as_posix()
        except ValueError:
            return path.as_posix()

    @classmethod
    def _json_safe_payload(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {
                (key if isinstance(key, str) else str(key)): cls._json_safe_payload(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe_payload(item) for item in value]

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return cls._json_safe_payload(tolist())
            except Exception:
                pass

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return cls._json_safe_payload(item_method())
            except Exception:
                pass

        return str(value)

    @classmethod
    def resolve_program_path(cls, raw_path: Any) -> Path | None:
        text = cls._nonempty_text_or_none(raw_path)
        if text is None:
            return None

        candidate = Path(text)
        attempts: list[Path] = []
        if candidate.is_absolute():
            attempts.append(candidate)
        else:
            attempts.append(_BEST_STORE_DIR / candidate)
            attempts.append(_EVAL_ROOT / candidate)
            attempts.append(Path.cwd() / candidate)

        seen: set[str] = set()
        for attempt in attempts:
            normalized = str(attempt.resolve(strict=False))
            if normalized in seen:
                continue
            seen.add(normalized)
            if attempt.is_file():
                return attempt

        if candidate.is_absolute():
            return candidate
        return _BEST_STORE_DIR / candidate

    @classmethod
    def canonical_program_path(cls, raw_path: Any) -> str | None:
        resolved = cls.resolve_program_path(raw_path)
        if resolved is not None:
            return cls._best_store_relative_path(resolved)
        text = cls._nonempty_text_or_none(raw_path)
        if text is None:
            return None
        return Path(text).as_posix()

    @classmethod
    def program_path_exists(cls, raw_path: Any) -> bool:
        resolved = cls.resolve_program_path(raw_path)
        return bool(resolved is not None and resolved.is_file())

    @classmethod
    def preferred_program_path(cls, record: Any, index_entry: Any) -> str | None:
        candidates: list[str] = []
        if isinstance(record, dict):
            best_program_path = cls._nonempty_text_or_none(record.get("best_program_path"))
            if best_program_path is not None:
                candidates.append(best_program_path)
        if isinstance(index_entry, dict):
            for key in ("best_program_path", "program_file"):
                best_program_path = cls._nonempty_text_or_none(index_entry.get(key))
                if best_program_path is not None:
                    candidates.append(best_program_path)

        program_uuid = None
        if isinstance(record, dict):
            program_uuid = cls._nonempty_text_or_none(record.get("program_uuid"))
        if program_uuid is None and isinstance(index_entry, dict):
            program_uuid = cls._nonempty_text_or_none(index_entry.get("program_uuid"))

        first_candidate: str | None = None
        first_existing: str | None = None
        for raw_path in candidates:
            canonical = cls.canonical_program_path(raw_path)
            if canonical is None:
                continue
            if first_candidate is None:
                first_candidate = canonical
            if not cls.program_path_exists(canonical):
                continue
            if program_uuid is not None and Path(canonical).stem == program_uuid:
                return canonical
            if first_existing is None:
                first_existing = canonical

        return first_existing or first_candidate

    @classmethod
    def referenced_program_paths(
        cls,
        records: dict[str, dict[str, Any]],
        program_index: dict[str, dict[str, Any]],
    ) -> set[str]:
        referenced: set[str] = set()
        for stem, record in records.items():
            path_text = cls.preferred_program_path(record, program_index.get(stem))
            canonical = cls.canonical_program_path(path_text)
            if canonical is not None:
                referenced.add(canonical)
        return referenced

    @classmethod
    def delete_unreferenced_program_files(cls, referenced_paths: set[str]) -> None:
        if not _BEST_PROGRAMS_DIR.is_dir():
            return
        for child in _BEST_PROGRAMS_DIR.iterdir():
            if not child.is_file():
                continue
            child_key = cls._best_store_relative_path(child)
            if child_key in referenced_paths:
                continue
            try:
                child.unlink()
            except OSError:
                continue

    @staticmethod
    @contextmanager
    def locked() -> Any:
        _BEST_STORE_DIR.mkdir(parents=True, exist_ok=True)
        with _BEST_STORE_LOCK_PATH.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)

            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _load_json_dict(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def load_records(cls) -> dict[str, dict[str, Any]]:
        records = cls._load_json_dict(_BEST_STORE_PATH)
        out: dict[str, dict[str, Any]] = {}
        for circuit_key, record in records.items():
            if isinstance(record, dict):
                out[CircuitCatalog.baseline_key(circuit_key)] = dict(record)
        return out

    @classmethod
    def load_program_index(cls) -> dict[str, dict[str, Any]]:
        index = cls._load_json_dict(_BEST_PROGRAM_INDEX_PATH)
        out: dict[str, dict[str, Any]] = {}
        for circuit_key, record in index.items():
            if isinstance(record, dict):
                out[CircuitCatalog.baseline_key(circuit_key)] = dict(record)
        return out

    @staticmethod
    def _save_json(path: Path, payload: Any) -> bool:
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            return True
        except (OSError, TypeError, ValueError):
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False

    @classmethod
    def save_records(cls, records: dict[str, dict[str, Any]]) -> bool:
        return cls._save_json(_BEST_STORE_PATH, records)

    @classmethod
    def save_program_index(cls, index: dict[str, dict[str, Any]]) -> bool:
        return cls._save_json(_BEST_PROGRAM_INDEX_PATH, index)

    @classmethod
    def save_candidate_program(cls, candidate: Path) -> tuple[str, str] | None:
        try:
            _BEST_PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
            program_uuid = uuid.uuid4().hex
            destination = _BEST_PROGRAMS_DIR / f"{program_uuid}{candidate.suffix or '.txt'}"
            shutil.copy2(candidate, destination)
        except OSError:
            return None

        relative_path = cls._best_store_relative_path(destination)
        return program_uuid, relative_path

    @classmethod
    def save_circuit_plan(cls, circuit_stem: str, plan: Any) -> bool:
        if not isinstance(plan, list):
            return False
        destination = _BEST_PLANS_DIR / f"{circuit_stem}_best_plan.json"
        return cls._save_json(destination, cls._json_safe_payload(plan))


class StageValidation:
    """Stage-level reuse and placement checks."""

    @staticmethod
    def stage_type(stage: Any) -> str | None:
        if stage is None:
            return None
        stage_type = stage.get("stage_type") if isinstance(stage, dict) else getattr(stage, "stage_type", None)
        if stage_type is None:
            return None
        normalized = str(stage_type).lower()
        if normalized == "rydberg":
            normalized = "2q"
        return normalized if normalized in ("1q", "2q") else None

    @staticmethod
    def gates(stage: Any) -> list[Any]:
        if isinstance(stage, dict):
            return list(stage.get("gates") or [])
        gates = getattr(stage, "gates", None)
        return list(gates) if gates else []

    @staticmethod
    def gate_type(gate: Any) -> str:
        if isinstance(gate, dict):
            return str(gate.get("gate_type", "")).lower()
        return str(getattr(gate, "gate_type", "")).lower()

    @staticmethod
    def gate_qubits(gate: Any) -> list[int]:
        qubits = gate.get("qubits") if isinstance(gate, dict) else getattr(gate, "qubits", None)
        return [int(x) for x in (qubits or [])]

    @classmethod
    def two_q_pairs_and_qubits(cls, stage: Any) -> tuple[set[frozenset[int]], set[int]]:
        pairs: set[frozenset[int]] = set()
        qubits: set[int] = set()
        for gate in cls.gates(stage):
            if cls.gate_type(gate) != "2q":
                continue
            gate_qubits = cls.gate_qubits(gate)
            qubits.update(gate_qubits)
            if len(gate_qubits) == 2:
                pairs.add(frozenset(gate_qubits))
        return pairs, qubits

    @classmethod
    def next_two_q_stage_index(cls, stages: list[Any], si: int) -> int | None:
        for j in range(si + 1, len(stages)):
            if cls.stage_type(stages[j]) == "2q":
                return j
        return None

    @staticmethod
    def placement_triple(
        placement: list[Any], q: int, label: str, errors: list[str]
    ) -> tuple[int, int, int] | None:
        if q < 0 or q >= len(placement):
            errors.append(f"{label}: qubit index {q} out of range (len={len(placement)})")
            return None
        cell = placement[q]
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            errors.append(f"{label}: qubit {q} placement {cell!r} is not a length-3 (zone,x,y) tuple")
            return None
        try:
            return (int(cell[0]), int(cell[1]), int(cell[2]))
        except (TypeError, ValueError):
            errors.append(f"{label}: qubit {q} placement {cell!r} has non-integral zone/x/y")
            return None

    @classmethod
    def check_positions_unique(cls, placement: list[Any], label: str, errors: list[str]) -> None:
        seen: dict[tuple[int, int, int], int] = {}
        for q, _cell in enumerate(placement):
            triple = cls.placement_triple(placement, q, label, errors)
            if triple is None:
                continue
            if triple in seen:
                errors.append(
                    f"{label}: duplicate physical position {triple} for logical qubits {seen[triple]} and {q} "
                    f"(zone 0=storage, 1=entangling)"
                )
                continue
            seen[triple] = q

    @classmethod
    def check_transition_moved_qubits(
        cls,
        previous_placement: list[Any],
        current_placement: list[Any],
        previous_label: str,
        current_label: str,
        errors: list[str],
    ) -> None:
        previous_positions: dict[tuple[int, int, int], int] = {}
        for q in range(len(previous_placement)):
            triple = cls.placement_triple(previous_placement, q, previous_label, errors)
            if triple is None or triple in previous_positions:
                continue
            previous_positions[triple] = q

        for q in range(min(len(previous_placement), len(current_placement))):
            previous_triple = cls.placement_triple(previous_placement, q, previous_label, errors)
            current_triple = cls.placement_triple(current_placement, q, current_label, errors)
            if previous_triple is None or current_triple is None or current_triple == previous_triple:
                continue
            previous_owner = previous_positions.get(current_triple)
            if previous_owner is None or previous_owner == q:
                continue
            errors.append(
                f"{current_label}: moved qubit {q} cannot move to {current_triple}; "
                f"{previous_label} had logical qubit {previous_owner} there"
            )

    @staticmethod
    def adjacent_entangling_columns(xa: int, ya: int, xb: int, yb: int) -> bool:
        return xa == xb and ya // 2 == yb // 2 and abs(ya - yb) == 1

    @classmethod
    def validate_placements(
        cls,
        stages: list[Any],
        reuse_info: list[list[int]],
        initial_placement: list[Any],
        placements_by_2q_stage: list[list[list[Any]]],
        two_q_stage_indices: list[int] | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if not isinstance(initial_placement, list):
            errors.append("initial_placement must be a list")
            return errors
        if not isinstance(placements_by_2q_stage, list):
            errors.append("placements_by_2q_stage must be a list")
            return errors

        if two_q_stage_indices is None:
            two_q_stage_indices = [
                i for i, stage in enumerate(stages) if cls.stage_type(stage) == "2q" and cls.gates(stage)
            ]
        if len(placements_by_2q_stage) != len(two_q_stage_indices):
            errors.append(
                "placements_by_2q_stage length mismatch: "
                f"expected {len(two_q_stage_indices)} from 2q stages, got {len(placements_by_2q_stage)}"
            )
            return errors

        n_q = len(initial_placement)
        cls.check_positions_unique(initial_placement, "initial_placement", errors)
        for qi in range(n_q):
            triple = cls.placement_triple(initial_placement, qi, "initial_placement", errors)
            if triple is not None and triple[0] != 0:
                errors.append(f"initial_placement: qubit {qi} must be in storage zone (z=0), got {triple}")

        prev_group_last = initial_placement
        prev_two_q_si: int | None = None
        for gi, si in enumerate(two_q_stage_indices):
            stage = stages[si]
            group = placements_by_2q_stage[gi]
            label = f"placements_by_2q_stage[{gi}](stage={gi})"
            if not isinstance(group, list) or not group:
                errors.append(f"{label}: must contain at least one placement")
                continue

            for pi, placement in enumerate(group):
                if len(placement) != n_q:
                    errors.append(f"{label}[{pi}]: expected {n_q} qubits, got {len(placement)}")
                cls.check_positions_unique(placement, f"{label}[{pi}]", errors)

            first = group[0]
            previous_label = "initial_placement" if gi == 0 else f"placements_by_2q_stage[{gi - 1}][-1]"
            cls.check_transition_moved_qubits(prev_group_last, first, previous_label, f"{label}[0]", errors)
            involved: set[int] = set()
            gate_pairs: list[tuple[int, int]] = []
            for gate in cls.gates(stage):
                if cls.gate_type(gate) != "2q":
                    continue
                gate_qubits = cls.gate_qubits(gate)
                if len(gate_qubits) != 2:
                    errors.append(f"{label}: invalid 2q gate qubits={gate_qubits!r}")
                    continue
                a, b = gate_qubits
                involved.update((a, b))
                gate_pairs.append((a, b))

            for q in range(n_q):
                triple = cls.placement_triple(first, q, f"{label}[0]", errors)
                if triple is None:
                    continue
                if q in involved and triple[0] != 1:
                    errors.append(f"{label}[0]: 2q-involved qubit {q} must be in entangling, got {triple}")
                if q not in involved and triple[0] != 0:
                    errors.append(f"{label}[0]: non-involved qubit {q} must be in storage, got {triple}")
            for a, b in gate_pairs:
                ta = cls.placement_triple(first, a, f"{label}[0]", errors)
                tb = cls.placement_triple(first, b, f"{label}[0]", errors)
                if ta is None or tb is None:
                    continue
                if ta[0] == 1 and tb[0] == 1 and not cls.adjacent_entangling_columns(ta[1], ta[2], tb[1], tb[2]):
                    errors.append(f"{label}[0]: pair ({a},{b}) not adjacent in entangling: {ta}, {tb}")

            prev_reuse: set[int] = set()
            if prev_two_q_si is not None and prev_two_q_si < len(reuse_info):
                prev_reuse = {int(x) for x in reuse_info[prev_two_q_si]}
            for q in sorted(prev_reuse):
                if q < 0 or q >= n_q:
                    errors.append(f"{label}[0]: previous reuse has invalid qubit {q}")
                    continue
                cls.placement_triple(prev_group_last, q, f"prev_group_last(stage={gi - 1})", errors)
                cls.placement_triple(first, q, f"{label}[0]", errors)

            cur_reuse: set[int] = set()
            if si < len(reuse_info):
                cur_reuse = {int(x) for x in reuse_info[si]}
            for pi in range(1, len(group)):
                placement = group[pi]
                cls.check_transition_moved_qubits(
                    group[pi - 1],
                    placement,
                    f"{label}[{pi - 1}]",
                    f"{label}[{pi}]",
                    errors,
                )
                for q in sorted(cur_reuse):
                    if q < 0 or q >= n_q:
                        errors.append(f"{label}[{pi}]: current reuse has invalid qubit {q}")
                        continue
                    cls.placement_triple(first, q, f"{label}[0]", errors)
                    cls.placement_triple(placement, q, f"{label}[{pi}]", errors)

            prev_group_last = group[-1]
            prev_two_q_si = si

        return errors

    @classmethod
    def validate_reuse_info(cls, stages: list[Any], reuse_info: list[list[int]]) -> list[str]:
        errors: list[str] = []
        for si, stage in enumerate(stages):
            if cls.stage_type(stage) != "2q":
                continue
            row = reuse_info[si] if si < len(reuse_info) else []
            if not row:
                continue
            reuse_set = {int(x) for x in row}
            next_index = cls.next_two_q_stage_index(stages, si)
            if next_index is None:
                errors.append(
                    f"2q stage index {si}: reuse_info={sorted(reuse_set)} is non-empty but there is no following 2q stage"
                )
                continue

            cur_pairs, cur_qubits = cls.two_q_pairs_and_qubits(stage)
            next_pairs, next_qubits = cls.two_q_pairs_and_qubits(stages[next_index])
            for q in sorted(reuse_set):
                if q not in cur_qubits:
                    errors.append(f"2q@{si}: reuse qubit {q} does not appear in any 2q gate of this stage")
                if q not in next_qubits:
                    errors.append(
                        f"2q@{si}: reuse qubit {q} does not appear in any 2q gate of the next 2q stage (index {next_index})"
                    )
            for pair in cur_pairs:
                a, b = tuple(pair)
                if a in reuse_set and b in reuse_set and pair not in next_pairs:
                    errors.append(
                        f"2q@{si}: qubits {a},{b} are both in reuse but the next 2q stage (index {next_index}) "
                        f"has no 2q gate on that pair (next stage pairs: {sorted(map(sorted, next_pairs))})"
                    )
            for pair in next_pairs:
                a, b = tuple(pair)
                if a in reuse_set and b in reuse_set and pair not in cur_pairs:
                    errors.append(
                        f"2q@{si} → next 2q@{next_index}: pair {sorted(pair)} appears in the next stage and both "
                        f"qubits are in reuse, but that pair is not a 2q gate in the current stage"
                    )
        return errors

    @classmethod
    def apply_reuse_info_validation(
        cls,
        core: dict[str, Any],
        mod: Any,
        reuse_validation_stats: dict[str, bool] | None = None,
    ) -> None:
        sol = getattr(mod, "solver", None)
        li = getattr(sol, "last_solve_intermediates", None) if sol is not None else None
        if not isinstance(li, dict) or "stages" not in li or "reuse_info" not in li:
            core["reuse_info_validation_skipped"] = True
            core["reuse_info_validation_ok"] = None
            core["reuse_info_validation_errors"] = None
            return
        stages = li["stages"]
        reuse_info = li["reuse_info"]
        if not isinstance(stages, list) or not isinstance(reuse_info, list):
            core["reuse_info_validation_skipped"] = True
            core["reuse_info_validation_ok"] = None
            core["reuse_info_validation_errors"] = None
            return

        errs = cls.validate_reuse_info(stages, reuse_info)
        core["reuse_info_validation_skipped"] = False
        core["reuse_info_validation_ok"] = len(errs) == 0
        core["reuse_info_validation_errors"] = errs
        if reuse_validation_stats is not None and errs:
            reuse_validation_stats["failed"] = True

    @classmethod
    def apply_placement_validation(
        cls,
        core: dict[str, Any],
        mod: Any,
        placement_validation_stats: dict[str, bool] | None = None,
    ) -> None:
        sol = getattr(mod, "solver", None)
        li = getattr(sol, "last_solve_intermediates", None) if sol is not None else None
        if not isinstance(li, dict):
            core["placement_validation_skipped"] = True
            core["placement_validation_ok"] = None
            core["placement_validation_errors"] = None
            return

        need = ("stages", "reuse_info", "initial_placement", "placements_by_2q_stage")
        if not all(k in li for k in need):
            core["placement_validation_skipped"] = True
            core["placement_validation_ok"] = None
            core["placement_validation_errors"] = None
            return

        stages = li["stages"]
        reuse_info = li["reuse_info"]
        initial_placement = li["initial_placement"]
        placements_by_2q_stage = li["placements_by_2q_stage"]
        two_q_stage_indices = li.get("two_q_stage_indices")
        if (
            not isinstance(stages, list)
            or not isinstance(reuse_info, list)
            or not isinstance(initial_placement, list)
            or not isinstance(placements_by_2q_stage, list)
        ):
            core["placement_validation_skipped"] = True
            core["placement_validation_ok"] = None
            core["placement_validation_errors"] = None
            return

        errs = cls.validate_placements(
            stages=stages,
            reuse_info=reuse_info,
            initial_placement=initial_placement,
            placements_by_2q_stage=placements_by_2q_stage,
            two_q_stage_indices=two_q_stage_indices if isinstance(two_q_stage_indices, list) else None,
        )
        core["placement_validation_skipped"] = False
        core["placement_validation_ok"] = len(errs) == 0
        core["placement_validation_errors"] = errs
        if placement_validation_stats is not None and errs:
            placement_validation_stats["failed"] = True

    @staticmethod
    def apply_auxiliary_failures(core: dict[str, Any]) -> None:
        if not core.get("ok"):
            return

        reuse_errors = core.get("reuse_info_validation_errors")
        placement_errors = core.get("placement_validation_errors")
        parts: list[str] = []
        if isinstance(reuse_errors, list) and reuse_errors:
            parts.extend(str(x) for x in reuse_errors)
        if isinstance(placement_errors, list) and placement_errors:
            parts.extend(str(x) for x in placement_errors)
        if not parts:
            return

        core["ok"] = False
        if isinstance(reuse_errors, list) and reuse_errors and isinstance(placement_errors, list) and placement_errors:
            core["error"] = "ValidationError"
            core["error_category"] = "placer_or_2q_related"
        elif isinstance(reuse_errors, list) and reuse_errors:
            core["error"] = "ReuseValidationError"
            core["error_category"] = "reuse_related"
        else:
            core["error"] = "PlacementValidationError"
            core["error_category"] = "placer_or_2q_related"
        core["message"] = "; ".join(parts)


class SolverDiagnostics:
    """Shared solver diagnostics and failure payload helpers."""

    @staticmethod
    def build_failure_core(message: str, error: str = "Error") -> dict[str, Any]:
        return {
            "ok": False,
            "error": error,
            "message": message,
            "time_us": None,
            "total_fidelity": None,
            "fidelity_components": {},
            "neg_log_components": {},
        }

    @classmethod
    def attach_intermediates(cls, core: dict[str, Any], mod: Any) -> None:
        sol = getattr(mod, "solver", None)
        intermediates = getattr(sol, "last_solve_intermediates", None) if sol is not None else None
        if not isinstance(intermediates, dict) or not intermediates:
            return

        if "machine_error_context" in intermediates:
            core["machine_error_context"] = SnapshotFormatter.strip_route_inputs(
                cls._sanitize_value(intermediates["machine_error_context"])
            )
            return

        for key in (
            "stages",
            "reuse_info",
            "initial_placement",
            "placements_by_2q_stage",
            "two_q_stage_indices",
            "routes",
            "router_checks",
            "operations",
        ):
            if key not in intermediates:
                continue
            value = intermediates[key]
            if key in ("stages", "routes", "operations"):
                core[key] = cls._sanitize_value(value)
            else:
                core[key] = value

    @staticmethod
    def extract_component_times(mod: Any) -> dict[str, float] | None:
        sol = getattr(mod, "solver", None)
        intermediates = getattr(sol, "last_solve_intermediates", None) if sol is not None else None
        if not isinstance(intermediates, dict):
            return None
        component_times = intermediates.get("component_times")
        if not isinstance(component_times, dict):
            return None

        out: dict[str, float] = {}
        for name in _COMPONENT_NAMES:
            value = component_times.get(name)
            if value is None:
                return None
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                return None
            out[name] = numeric_value
        return out

    @staticmethod
    def result_to_core_dict(result: Any) -> dict[str, Any]:
        """Normalize a solver result into the evaluator's plain-dict core format."""
        if isinstance(result, dict) and "plan" in result and "machine" in result:
            return PlanExecutionValidator.validate(result)
        if result is None:
            return SolverDiagnostics.build_failure_core("run_code returned None", "run_code_returned_none")

        summary_builder = getattr(result, "to_summary_dict", None)
        if callable(summary_builder):
            core = summary_builder()
            if not isinstance(core, dict):
                core = {"raw_summary": core}
        else:
            core = {
                "time_us": getattr(result, "time", None),
                "total_fidelity": getattr(result, "total_fidelity", None),
                "split_fidelity_linear": getattr(result, "split_fidelity_linear", None),
                "split_fidelity": getattr(result, "split_fidelity", None),
            }

        out = dict(core)
        out.pop("per_qubit", None)
        out["ok"] = True
        return out

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        type_name = type(value).__name__
        if type_name == "ZNAAStage":
            try:
                stage_type = getattr(value, "stage_type", None)
                gates = getattr(value, "gates", None) or []
                return {"stage_type": stage_type, "gates": [str(gate) for gate in gates]}
            except Exception:
                return repr(value)
        if type_name.startswith("Operation_"):
            try:
                return str(value)
            except Exception:
                return repr(value)
        if isinstance(value, list):
            if value and all(isinstance(cell, (list, tuple)) and len(cell) == 3 for cell in value):
                return SnapshotFormatter.placement_to_labeled_string(value)
            return [cls._sanitize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._sanitize_value(item) for key, item in value.items()}
        return value


class SnapshotFormatter:
    """Snapshot formatting for failure diagnostics."""

    @staticmethod
    def strip_route_inputs(obj: Any) -> Any:
        if isinstance(obj, list):
            return [SnapshotFormatter.strip_route_inputs(item) for item in obj]
        if isinstance(obj, dict):
            return {
                key: SnapshotFormatter.strip_route_inputs(value)
                for key, value in obj.items()
                if key != "placement_before_route"
            }
        return obj

    @staticmethod
    def stage_kind(block: dict[str, Any]) -> str:
        stage = str(block.get("stage") or "")
        if "@" in stage:
            return stage.split("@", 1)[0]
        return stage

    @staticmethod
    def zone_name(zone: int) -> str:
        if zone == 0:
            return "storage"
        if zone == 1:
            return "entangling"
        if zone == 2:
            return "readout"
        return str(zone)

    @staticmethod
    def normalize_zone_name(zone: Any) -> str | None:
        if zone is None:
            return None
        text = str(zone).strip().lower()
        if text in ("0", "s", "storage"):
            return "storage"
        if text in ("1", "e", "entangling"):
            return "entangling"
        if text in ("2", "r", "readout"):
            return "readout"
        if text in ("a", "aod"):
            return "aod"
        return text or None

    @classmethod
    def zone_index(cls, zone: Any) -> int | None:
        normalized = cls.normalize_zone_name(zone)
        if normalized == "storage":
            return 0
        if normalized == "entangling":
            return 1
        if normalized == "readout":
            return 2
        return None

    @staticmethod
    def placement_to_triples_array(placement: Any) -> list[list[int]]:
        rows: list[list[int]] = []
        for cell in list(placement or []):
            if isinstance(cell, (list, tuple)) and len(cell) == 3:
                try:
                    rows.append([int(cell[0]), int(cell[1]), int(cell[2])])
                except (TypeError, ValueError):
                    rows.append([-1, -1, -1])
                continue
            rows.append([-1, -1, -1])
        return rows

    @staticmethod
    def zone_shapes_from_machine(machine: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
        hardware_shape = getattr(machine, "hardware_shape", None) or getattr(machine, "hardware_config", None)
        if hardware_shape is None:
            return None
        try:
            storage = getattr(hardware_shape, "storage", None)
            entangling = getattr(hardware_shape, "entangling", None)
            if storage is None or entangling is None:
                return None
            sr, sc = int(storage[0]), int(storage[1])
            er, ec = int(entangling[0]), int(entangling[1])
            if sr <= 0 or sc <= 0 or er <= 0 or ec <= 0:
                return None
            return (sr, sc), (er, ec)
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def placement_to_zone_matrices(
        placement: Any,
        storage_shape: tuple[int, int],
        entangling_shape: tuple[int, int],
    ) -> dict[str, list[list[int]]]:
        sr, sc = storage_shape
        er, ec = entangling_shape
        content_storage = -np.ones((sr, sc), dtype=int)
        content_entangling = -np.ones((er, ec), dtype=int)
        for q, cell in enumerate(list(placement or [])):
            if not isinstance(cell, (list, tuple)) or len(cell) != 3:
                continue
            try:
                z, x, y = int(cell[0]), int(cell[1]), int(cell[2])
            except (TypeError, ValueError):
                continue
            if z == 0 and 0 <= x < sr and 0 <= y < sc:
                content_storage[x, y] = int(q)
            if z == 1 and 0 <= x < er and 0 <= y < ec:
                content_entangling[x, y] = int(q)
        return {
            "content_storage": content_storage.tolist(),
            "content_entangling": content_entangling.tolist(),
        }

    @staticmethod
    def placement_to_labeled_string(placement: Any) -> str:
        parts: list[str] = []
        for q, cell in enumerate(list(placement or [])):
            label = f"q{q}"
            if isinstance(cell, (list, tuple)) and len(cell) == 3:
                try:
                    z = int(cell[0])
                    x = int(cell[1])
                    y = int(cell[2])
                    parts.append(f"{label}: [{z}, {x}, {y}]")
                    continue
                except (TypeError, ValueError):
                    pass
            parts.append(f"{label}: [-1, -1, -1]")
        return ", ".join(parts) if parts else "[]"

    @classmethod
    def format_one_placement(cls, placement: Any, machine: Any) -> Any:
        return cls.placement_to_labeled_string(placement)

    @classmethod
    def format_placements_group(cls, placements: Any, machine: Any) -> list[Any]:
        return [cls.format_one_placement(placement, machine) for placement in list(placements or [])]

    @staticmethod
    def _shape_of_placement(placement: Any) -> str:
        if isinstance(placement, np.ndarray):
            return f"ndarray{tuple(int(x) for x in placement.shape)}"
        if isinstance(placement, (list, tuple)):
            n_rows = len(placement)
            row_lengths: list[int] = []
            non_sequence_rows = 0
            for row in placement:
                if isinstance(row, (list, tuple, np.ndarray)):
                    try:
                        row_lengths.append(len(row))
                    except TypeError:
                        non_sequence_rows += 1
                else:
                    non_sequence_rows += 1
            if non_sequence_rows > 0:
                return f"list(len={n_rows}, non_sequence_rows={non_sequence_rows})"
            unique_lengths = sorted(set(int(x) for x in row_lengths))
            if len(unique_lengths) == 1:
                return f"list({n_rows},{unique_lengths[0]})"
            return f"list(len={n_rows}, ragged_row_lens={unique_lengths})"
        return type(placement).__name__

    @classmethod
    def placement_shape_mismatch_info(
        cls,
        placement: Any,
        expected_n_qubits: int | None,
    ) -> dict[str, Any] | None:
        if expected_n_qubits is None:
            return None
        actual_shape = cls._shape_of_placement(placement)
        if not isinstance(placement, (list, tuple)):
            return {
                "shape_mismatch": True,
                "expected_shape": [int(expected_n_qubits), 3],
                "actual_shape": actual_shape,
            }
        if len(placement) != int(expected_n_qubits):
            return {
                "shape_mismatch": True,
                "expected_shape": [int(expected_n_qubits), 3],
                "actual_shape": actual_shape,
            }
        for row in placement:
            if not isinstance(row, (list, tuple, np.ndarray)):
                return {
                    "shape_mismatch": True,
                    "expected_shape": [int(expected_n_qubits), 3],
                    "actual_shape": actual_shape,
                }
            try:
                if len(row) != 3:
                    return {
                        "shape_mismatch": True,
                        "expected_shape": [int(expected_n_qubits), 3],
                        "actual_shape": actual_shape,
                    }
            except TypeError:
                return {
                    "shape_mismatch": True,
                    "expected_shape": [int(expected_n_qubits), 3],
                    "actual_shape": actual_shape,
                }
        return None


class PlanExecutionValidator:
    """Validate and execute the evaluator-owned plan payload."""

    def __init__(self, plan: list[dict[str, Any]], machine: Any, circuit: Any) -> None:
        self.plan = plan
        self.machine = machine
        self.circuit = circuit
        self.expected_snapshot_n_qubits = self._read_expected_snapshot_qubits()
        self.two_qubit_block_indices = [
            index for index, block in enumerate(self.plan) if SnapshotFormatter.stage_kind(block) == "2q"
        ]
    @staticmethod
    def _aligned_stage_index(stage_index: int | None) -> int:
        try:
            numeric_index = int(stage_index) if stage_index is not None else 0
        except (TypeError, ValueError):
            return 0
        return max(numeric_index - 1, 0)

    @classmethod
    def validate(cls, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict) or {"plan", "machine", "circuit"} - result.keys():
            return SolverDiagnostics.build_failure_core(
                "run_code must return {'plan': ..., 'machine': ..., 'circuit': ...}",
                "InvalidRunCodeReturn",
            )

        plan = result.get("plan")
        machine = result.get("machine")
        circuit = result.get("circuit")
        if not isinstance(plan, list):
            return SolverDiagnostics.build_failure_core("run_code return field 'plan' must be a list", "InvalidPlan")
        if machine is None or not hasattr(machine, "append_operation"):
            return SolverDiagnostics.build_failure_core(
                "run_code return field 'machine' must be a ZNAAMachine-like object",
                "InvalidMachine",
            )
        if circuit is None or not hasattr(circuit, "remove_frontier_gate") or not hasattr(circuit, "is_empty"):
            return SolverDiagnostics.build_failure_core(
                "run_code return field 'circuit' must be a ZNAACircuit-like object",
                "InvalidCircuit",
            )

        validator = cls(plan, machine, circuit)
        for step in (
            validator._validate_machine_contract,
            validator._validate_frontier_order,
            validator._validate_two_qubit_blocks,
            validator._execute_plan,
            validator._validate_final_storage_zone,
        ):
            failure = step()
            if failure is not None:
                return failure
        return validator._build_success_core()

    def _read_expected_snapshot_qubits(self) -> int | None:
        circuit_qubits = getattr(self.circuit, "qubits", None)
        if not isinstance(circuit_qubits, (list, tuple, set)):
            return None
        try:
            return int(len({int(qubit) for qubit in circuit_qubits}))
        except (TypeError, ValueError):
            return None

    def _validate_machine_contract(self) -> dict[str, Any] | None:
        trusted = HardwareBaselineGuard.trusted_baseline()
        machine_shape_sig = HardwareBaselineGuard.shape_signature(getattr(self.machine, "hardware_shape", None))
        machine_params_sig = HardwareBaselineGuard.params_signature(getattr(self.machine, "hardware_parameters", None))
        if trusted is not None:
            trusted_shape_sig, trusted_params_sig = trusted
            if machine_shape_sig != trusted_shape_sig or machine_params_sig != trusted_params_sig:
                return SolverDiagnostics.build_failure_core(
                    "run_code returned machine with modified hardware shape/parameters baseline.",
                    "HardwareConfigTampered",
                )
        if bool(getattr(self.machine, "skip_open_close_chain_time", False)):
            return SolverDiagnostics.build_failure_core(
                "run_code returned machine with tampered timing model (skip_open_close_chain_time=True).",
                "MachineTimingModelTampered",
            )
        return None

    def _build_stage_package(self, block: dict[str, Any]) -> dict[str, Any]:
        placements_group = list(block.get("placements_group") or [])
        stage_summary = StageAnalysis._format_two_q_gate_pairs(block.get("two_q_gates") or [])
        for placement_index, placement in enumerate(placements_group):
            mismatch = SnapshotFormatter.placement_shape_mismatch_info(
                placement,
                self.expected_snapshot_n_qubits,
            )
            if mismatch is not None:
                return {
                    "stage": stage_summary or block.get("stage"),
                    "two_q_stage_index": block.get("two_q_stage_index"),
                    "shape_mismatch": True,
                    "placement_index": int(placement_index),
                    "error_message": "snapshot placement shape mismatch; expected (n,3) with n=circuit qubit count",
                    "expected_shape": mismatch["expected_shape"],
                    "actual_shape": mismatch["actual_shape"],
                }
        return {
            "stage": stage_summary or block.get("stage"),
            "two_q_stage_index": block.get("two_q_stage_index"),
            "reuse_info": list(block.get("reuse_info") or []),
            "two_q_stage_definition": stage_summary,
            "placements": SnapshotFormatter.format_placements_group(placements_group, self.machine),
        }

    def _build_two_qubit_snapshot(self, block_index: int) -> dict[str, Any]:
        current_index = block_index if block_index in self.two_qubit_block_indices else None
        if current_index is None and block_index > 0:
            for candidate_index in reversed(self.two_qubit_block_indices):
                if candidate_index <= block_index:
                    current_index = candidate_index
                    break
        if current_index is None:
            return {"error_category": "placer_or_2q_related", "two_q_stages": []}

        previous_index: int | None = None
        for candidate_index in reversed(self.two_qubit_block_indices):
            if candidate_index < current_index:
                previous_index = candidate_index
                break

        items: list[dict[str, Any]] = []
        if previous_index is not None:
            items.append(self._build_stage_package(self.plan[previous_index]))
        items.append(self._build_stage_package(self.plan[current_index]))
        return {"error_category": "placer_or_2q_related", "two_q_stages": items}

    @staticmethod
    def _route_op_span(route_check: Any) -> tuple[int, int] | None:
        if not isinstance(route_check, dict):
            return None
        span = route_check.get("op_span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return None
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end < start:
            return None
        return (start, end)

    @classmethod
    def _route_index_for_operation_index(
        cls,
        router_checks: list[dict[str, Any]],
        operation_index: int,
    ) -> int | None:
        fallback_index: int | None = None
        for route_index, route_check in enumerate(router_checks):
            span = cls._route_op_span(route_check)
            if span is None:
                continue
            start, end = span
            if start <= operation_index < end:
                return route_index
            if end <= operation_index:
                fallback_index = route_index
        if fallback_index is not None:
            return fallback_index
        return 0 if router_checks else None

    @staticmethod
    def _runtime_numeric(value: Any) -> Any:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value if isinstance(value, (str, int, bool)) else repr(value)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value

    @staticmethod
    def _serialize_runtime_position(qubit_index: int, position: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"qubit": int(qubit_index)}
        if position is None:
            item["missing"] = True
            return item

        zone_name = SnapshotFormatter.normalize_zone_name(getattr(position, "zone", None))
        x_value = getattr(position, "x", None)
        y_value = getattr(position, "y", None)
        item["zone"] = zone_name if zone_name is not None else str(getattr(position, "zone", ""))
        if x_value is None:
            item["x"] = repr(x_value)
        else:
            try:
                item["x"] = int(x_value)
            except (TypeError, ValueError):
                item["x"] = repr(x_value)
        if y_value is None:
            item["y"] = repr(y_value)
        else:
            try:
                item["y"] = int(y_value)
            except (TypeError, ValueError):
                item["y"] = repr(y_value)

        zone_index = SnapshotFormatter.zone_index(zone_name)
        if zone_index is not None and isinstance(item.get("x"), int) and isinstance(item.get("y"), int):
            item["placement"] = [int(zone_index), int(item["x"]), int(item["y"])]
        return item

    def _runtime_machine_positions(self) -> list[dict[str, Any]] | None:
        qubit_to_position = getattr(self.machine, "qubit_to_position", None)
        if not isinstance(qubit_to_position, dict):
            return None

        if self.expected_snapshot_n_qubits is not None:
            qubit_ids: list[int] = list(range(int(self.expected_snapshot_n_qubits)))
        else:
            qubit_id_set: set[int] = set()
            for key in qubit_to_position.keys():
                try:
                    qubit_id_set.add(int(key))
                except (TypeError, ValueError):
                    continue
            current_qubits = getattr(self.machine, "current_qubits", None)
            if isinstance(current_qubits, (set, list, tuple)):
                for key in current_qubits:
                    try:
                        qubit_id_set.add(int(key))
                    except (TypeError, ValueError):
                        continue
            qubit_ids = sorted(qubit_id_set)

        return [
            self._serialize_runtime_position(qubit_index, qubit_to_position.get(int(qubit_index)))
            for qubit_index in qubit_ids
        ]

    def _runtime_machine_aod_state(self) -> dict[str, Any] | None:
        zone_name = SnapshotFormatter.normalize_zone_name(getattr(self.machine, "aod_current_place", None))
        rows_raw = getattr(self.machine, "aod_rows", None)
        cols_raw = getattr(self.machine, "aod_columns", None)
        content_raw = getattr(self.machine, "content_aod", None)

        rows = [self._runtime_numeric(value) for value in list(rows_raw or [])]
        columns = [self._runtime_numeric(value) for value in list(cols_raw or [])]
        content: list[list[Any]] = []
        for row in list(content_raw or []):
            if isinstance(row, (list, tuple, np.ndarray)):
                content.append([self._runtime_numeric(value) for value in list(row)])
            else:
                content.append([repr(row)])

        if zone_name is None and not rows and not columns and not content:
            return None

        state: dict[str, Any] = {
            "zone": zone_name,
            "rows": rows,
            "columns": columns,
            "content": content,
        }
        try:
            time_value = float(getattr(self.machine, "time", 0.0))
        except (TypeError, ValueError):
            time_value = None
        if time_value is not None and math.isfinite(time_value):
            state["time_us"] = time_value
        return state

    def _runtime_operation_history_tail(self, limit: int = 3) -> list[dict[str, Any]] | None:
        history = getattr(self.machine, "operation_history", None)
        if not isinstance(history, list) or not history:
            return None
        return SolverDiagnostics._sanitize_value(history[-max(int(limit), 0) :])

    @staticmethod
    def _machine_zone_grid(machine: Any, zone_name: str | None) -> Any:
        if zone_name == "storage":
            return getattr(machine, "content_storage", None)
        if zone_name == "entangling":
            return getattr(machine, "content_entangling", None)
        if zone_name == "readout":
            return getattr(machine, "content_readout", None)
        return None

    def _runtime_bad_drop_context(self, message: str) -> dict[str, Any] | None:
        match = _BAD_DROP_COORD_RE.search(str(message))
        if match is None:
            return None

        zone_name = SnapshotFormatter.normalize_zone_name(match.group(1))
        if zone_name is None:
            return None

        x_value = int(match.group(2))
        y_value = int(match.group(3))
        detail: dict[str, Any] = {
            "zone": zone_name,
            "x": x_value,
            "y": y_value,
        }
        zone_index = SnapshotFormatter.zone_index(zone_name)
        if zone_index is not None:
            detail["placement"] = [int(zone_index), x_value, y_value]

        grid = self._machine_zone_grid(self.machine, zone_name)
        if grid is None:
            return detail

        shape: tuple[int, int] | None = None
        try:
            shape = (int(grid.shape[0]), int(grid.shape[1]))
        except Exception:
            try:
                n_rows = len(grid)
                n_cols = len(grid[0]) if n_rows > 0 else 0
                shape = (int(n_rows), int(n_cols))
            except Exception:
                shape = None

        if shape is None:
            return detail

        n_rows, n_cols = shape
        detail["zone_shape"] = [n_rows, n_cols]
        is_legal = 0 <= x_value < n_rows and 0 <= y_value < n_cols
        detail["is_legal"] = is_legal
        if not is_legal:
            return detail

        raw_occupant: Any = None
        try:
            raw_occupant = grid[x_value][y_value]
        except Exception:
            try:
                raw_occupant = grid[x_value, y_value]
            except Exception:
                raw_occupant = None
        try:
            occupant = int(raw_occupant) if raw_occupant is not None else None
        except (TypeError, ValueError):
            occupant = None
        if occupant is not None and occupant >= 0:
            detail["occupied_by_qubit"] = occupant
        return detail

    def _runtime_machine_snapshot(self) -> dict[str, Any] | None:
        snapshot: dict[str, Any] = {}
        positions = self._runtime_machine_positions()
        if positions is not None:
            snapshot["qubit_positions"] = positions

        aod_state = self._runtime_machine_aod_state()
        if aod_state is not None:
            snapshot["aod_state"] = aod_state

        history_tail = self._runtime_operation_history_tail(limit=3)
        if history_tail is not None:
            snapshot["operation_history_tail"] = history_tail
        return snapshot or None

    def _build_router_snapshot(
        self,
        block: dict[str, Any],
        route_index: int | None,
        message: str,
        *,
        operation_index: int | None = None,
        operation: Any = None,
    ) -> dict[str, Any]:
        stage_summary = StageAnalysis._format_two_q_gate_pairs(block.get("two_q_gates") or [])
        checks = list(block.get("router_checks") or [])
        selected_route_index = route_index
        if checks and selected_route_index is not None and 0 <= selected_route_index < len(checks):
            route_check = checks[selected_route_index]
        elif checks:
            route_check = checks[-1]
            selected_route_index = len(checks) - 1
        else:
            route_check = {
                "from_placement": None,
                "to_placement": None,
                "operations": [],
                "placement_edge_in_group": None,
            }
            selected_route_index = None

        from_placement = route_check.get("from_placement")
        to_placement = route_check.get("to_placement")
        route_span = self._route_op_span(route_check)
        runtime_error: dict[str, Any] = {}
        if operation_index is not None:
            runtime_error["operation_index"] = int(operation_index)
            if route_span is not None and route_span[0] <= int(operation_index):
                runtime_error["operation_index_in_route"] = int(operation_index) - int(route_span[0])
        if operation is not None:
            runtime_error["operation"] = str(operation)
            operation_type = getattr(operation, "type_name", None)
            if operation_type is not None:
                runtime_error["operation_type"] = str(operation_type)
        bad_drop = self._runtime_bad_drop_context(message)
        if bad_drop is not None:
            runtime_error["bad_drop"] = bad_drop

        runtime_machine_state = self._runtime_machine_snapshot()
        snapshot: dict[str, Any] = {
            "error_category": "router_related",
            "stage": stage_summary or block.get("stage"),
            "two_q_stage_index": block.get("two_q_stage_index"),
            "route_index": selected_route_index,
            "placement_edge_in_group": route_check.get("placement_edge_in_group"),
            "route_op_span": list(route_span) if route_span is not None else None,
            "operations": list(route_check.get("operations") or []),
            "error_message": message,
        }
        if runtime_error:
            snapshot["runtime_error"] = runtime_error
        if runtime_machine_state is not None:
            snapshot["runtime_machine_state"] = runtime_machine_state

        from_mismatch = SnapshotFormatter.placement_shape_mismatch_info(
            from_placement,
            self.expected_snapshot_n_qubits,
        )
        to_mismatch = SnapshotFormatter.placement_shape_mismatch_info(
            to_placement,
            self.expected_snapshot_n_qubits,
        )
        if from_mismatch is not None or to_mismatch is not None:
            details: dict[str, Any] = {}
            if from_mismatch is not None:
                details["from_placement"] = from_mismatch
            if to_mismatch is not None:
                details["to_placement"] = to_mismatch
            snapshot["shape_mismatch"] = True
            snapshot["error_message"] = "snapshot placement shape mismatch; expected (n,3) with n=circuit qubit count"
            snapshot["shape_mismatch_details"] = details
            return snapshot

        snapshot["from_placement"] = SnapshotFormatter.format_one_placement(from_placement, self.machine)
        snapshot["to_placement"] = SnapshotFormatter.format_one_placement(to_placement, self.machine)
        return snapshot

    def _build_reuse_snapshot(self, block: dict[str, Any], message: str) -> dict[str, Any]:
        stage_summary = StageAnalysis._format_two_q_gate_pairs(block.get("two_q_gates") or [])
        return {
            "error_category": "reuse_related",
            "stage": stage_summary or block.get("stage"),
            "two_q_stage_index": block.get("two_q_stage_index"),
            "reuse_info": list(block.get("reuse_info") or []),
            "two_q_stage_definition": stage_summary,
            "error_message": message,
        }

    def _build_failure(
        self,
        stage_index: int,
        details: list[str],
        error: str,
        *,
        category: str,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        aligned_stage_index = self._aligned_stage_index(stage_index)
        adjusted_details: list[str] = []
        for detail in details:
            if isinstance(detail, str):
                adjusted_details.append(
                    detail.replace(f"stage[{stage_index}]", f"stage[{aligned_stage_index}]")
                )
            else:
                adjusted_details.append(detail)
        core = SolverDiagnostics.build_failure_core("; ".join(adjusted_details), error)
        core["details"] = adjusted_details
        core["failure_stage_index"] = aligned_stage_index
        core["failure_reason"] = _FAILURE_REASON_BY_ERROR.get(error, f"Plan validation/execution failed ({error}).")
        core["error_category"] = category
        core["machine_error_context"] = SnapshotFormatter.strip_route_inputs(snapshot)
        return core

    def _validate_frontier_order(self) -> dict[str, Any] | None:
        for block_index, block in enumerate(self.plan):
            for gate_index, gate in enumerate(list(block.get("gates") or [])):
                try:
                    self.circuit.remove_frontier_gate(gate)
                except Exception as exc:
                    return self._build_failure(
                        block_index,
                        [f"stage frontier mismatch at stage[{block_index}] gate[{gate_index}]: {type(exc).__name__}: {exc}"],
                        "StageFrontierMismatch",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )
        if bool(self.circuit.is_empty()):
            return None

        remaining_gates = getattr(self.circuit, "gates", None)
        remaining_count = len(remaining_gates) if isinstance(remaining_gates, list) else None
        last_stage_index = max(len(self.plan) - 1, 0)
        return self._build_failure(
            last_stage_index,
            [f"stage frontier mismatch: circuit still has remaining gates after all stages, remaining={remaining_count}"],
            "StageFrontierMismatch",
            category="placer_or_2q_related",
            snapshot=self._build_two_qubit_snapshot(last_stage_index),
        )

    def _validate_two_qubit_blocks(self) -> dict[str, Any] | None:
        previous_two_qubit_block_index: int | None = None
        for two_qubit_position, block_index in enumerate(self.two_qubit_block_indices):
            block = self.plan[block_index]
            placements_group = block.get("placements_group")
            if not isinstance(placements_group, list) or not placements_group:
                return self._build_failure(
                    block_index,
                    [f"stage[{block_index}] missing placements_group"],
                    "PlacementValidationError",
                    category="placer_or_2q_related",
                    snapshot=self._build_two_qubit_snapshot(block_index),
                )

            placement_before_route = block.get("placement_before_route")
            if not isinstance(placement_before_route, list):
                return self._build_failure(
                    block_index,
                    [f"stage[{block_index}] missing placement_before_route"],
                    "PlacementValidationError",
                    category="placer_or_2q_related",
                    snapshot=self._build_two_qubit_snapshot(block_index),
                )

            first_placement = placements_group[0]
            placement_errors: list[str] = []
            StageValidation.check_positions_unique(
                placement_before_route,
                f"stage[{block_index}] placement_before_route",
                placement_errors,
            )
            for placement_index, placement in enumerate(placements_group):
                if not isinstance(placement, list):
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] placements_group[{placement_index}] must be a list"],
                        "PlacementValidationError",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )
                StageValidation.check_positions_unique(
                    placement,
                    f"stage[{block_index}] placements_group[{placement_index}]",
                    placement_errors,
                )
            StageValidation.check_transition_moved_qubits(
                placement_before_route,
                first_placement,
                f"stage[{block_index}] placement_before_route",
                f"stage[{block_index}] placements_group[0]",
                placement_errors,
            )
            for placement_index in range(1, len(placements_group)):
                StageValidation.check_transition_moved_qubits(
                    placements_group[placement_index - 1],
                    placements_group[placement_index],
                    f"stage[{block_index}] placements_group[{placement_index - 1}]",
                    f"stage[{block_index}] placements_group[{placement_index}]",
                    placement_errors,
                )
            if placement_errors:
                return self._build_failure(
                    block_index,
                    placement_errors,
                    "PlacementValidationError",
                    category="placer_or_2q_related",
                    snapshot=self._build_two_qubit_snapshot(block_index),
                )

            pairs: list[tuple[int, int]] = []
            involved_qubits: set[int] = set()
            for gate in list(block.get("two_q_gates") or []):
                if str(gate.get("gate_type", "")).lower() != "2q":
                    continue
                qubits = [int(qubit) for qubit in (gate.get("qubits") or [])]
                if len(qubits) != 2:
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] invalid two_q_gates entry: {gate!r}"],
                        "PlacementValidationError",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )
                left_qubit, right_qubit = qubits
                pairs.append((left_qubit, right_qubit))
                involved_qubits.update((left_qubit, right_qubit))

            for qubit_index in range(len(first_placement)):
                cell = first_placement[qubit_index]
                if not isinstance(cell, (list, tuple)) or len(cell) != 3:
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] first placement invalid at q{qubit_index}: {cell!r}"],
                        "PlacementValidationError",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )
                zone, row, col = int(cell[0]), int(cell[1]), int(cell[2])
                if qubit_index in involved_qubits and zone != 1:
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] first placement involved q{qubit_index} must be entangling, got {(zone, row, col)}"],
                        "PlacementValidationError",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )
                if qubit_index not in involved_qubits and zone != 0:
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] first placement non-involved q{qubit_index} must be storage, got {(zone, row, col)}"],
                        "PlacementValidationError",
                        category="placer_or_2q_related",
                        snapshot=self._build_two_qubit_snapshot(block_index),
                    )

            for left_qubit, right_qubit in pairs:
                left_cell = first_placement[left_qubit]
                right_cell = first_placement[right_qubit]
                if int(left_cell[0]) != 1 or int(right_cell[0]) != 1:
                    continue
                if StageValidation.adjacent_entangling_columns(
                    int(left_cell[1]),
                    int(left_cell[2]),
                    int(right_cell[1]),
                    int(right_cell[2]),
                ):
                    continue
                return self._build_failure(
                    block_index,
                    [f"stage[{block_index}] first placement pair ({left_qubit},{right_qubit}) not adjacent: {left_cell}, {right_cell}"],
                    "PlacementValidationError",
                    category="placer_or_2q_related",
                    snapshot=self._build_two_qubit_snapshot(block_index),
                )

            if previous_two_qubit_block_index is not None:
                previous_block = self.plan[previous_two_qubit_block_index]
                previous_last_placement = list(previous_block.get("placements_group") or [[]])[-1]
                previous_reuse = {int(qubit) for qubit in (previous_block.get("reuse_info") or [])}
                for qubit in sorted(previous_reuse):
                    if qubit < 0 or qubit >= len(first_placement) or qubit >= len(previous_last_placement):
                        return self._build_failure(
                            block_index,
                            [f"stage[{block_index}] previous reuse has invalid qubit {qubit}"],
                            "PlacementValidationError",
                            category="placer_or_2q_related",
                            snapshot=self._build_two_qubit_snapshot(block_index),
                        )

            current_reuse = {int(qubit) for qubit in (block.get("reuse_info") or [])}
            for placement_index in range(1, len(placements_group)):
                placement = placements_group[placement_index]
                for qubit in sorted(current_reuse):
                    if qubit < 0 or qubit >= len(first_placement) or qubit >= len(placement):
                        return self._build_failure(
                            block_index,
                            [f"stage[{block_index}] current reuse has invalid qubit {qubit} at placement[{placement_index}]"],
                            "PlacementValidationError",
                            category="placer_or_2q_related",
                            snapshot=self._build_two_qubit_snapshot(block_index),
                        )

            if current_reuse:
                if two_qubit_position + 1 >= len(self.two_qubit_block_indices):
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] has reuse={sorted(current_reuse)} but no next 2q stage"],
                        "ReuseValidationError",
                        category="reuse_related",
                        snapshot=self._build_reuse_snapshot(block, "no next 2q stage"),
                    )
                next_block = self.plan[self.two_qubit_block_indices[two_qubit_position + 1]]
                current_involved = {qubit for left_qubit, right_qubit in pairs for qubit in (left_qubit, right_qubit)}
                next_involved: set[int] = set()
                for gate in list(next_block.get("two_q_gates") or []):
                    if str(gate.get("gate_type", "")).lower() != "2q":
                        continue
                    qubits = [int(qubit) for qubit in (gate.get("qubits") or [])]
                    if len(qubits) == 2:
                        next_involved.update(qubits)
                for qubit in sorted(current_reuse):
                    if qubit in current_involved and qubit in next_involved:
                        continue
                    return self._build_failure(
                        block_index,
                        [f"stage[{block_index}] reuse q{qubit} not in both current/next 2q stage"],
                        "ReuseValidationError",
                        category="reuse_related",
                        snapshot=self._build_reuse_snapshot(block, f"reuse q{qubit} mismatch with next 2q stage"),
                    )

            previous_two_qubit_block_index = block_index
        return None

    def _execute_plan(self) -> dict[str, Any] | None:
        for block_index, block in enumerate(self.plan):
            failure = self._execute_stage(block_index, block)
            if failure is not None:
                return failure
        return None

    def _execute_stage(self, block_index: int, block: dict[str, Any]) -> dict[str, Any] | None:
        operations = list(block.get("operations") or [])
        machine_circuit = getattr(self.machine, "znaa_format_circuit", None)
        gates_before = len(getattr(machine_circuit, "gates", []))
        router_checks = list(block.get("router_checks") or [])
        route_checkpoints: dict[int, int] = {}
        for route_index, route_check in enumerate(router_checks):
            span = route_check.get("op_span")
            if not isinstance(span, list) or len(span) != 2:
                continue
            try:
                route_checkpoints[int(span[1])] = route_index
            except (TypeError, ValueError):
                continue

        try:
            for operation_index, operation in enumerate(operations):
                self.machine.append_operation(operation)
                route_index = route_checkpoints.get(operation_index + 1)
                if route_index is not None:
                    failure = self._validate_route_checkpoint(block_index, block, router_checks, route_index)
                    if failure is not None:
                        return failure
        except Exception as exc:
            operation_name = getattr(operation, "type_name", "") if "operation" in locals() else ""
            category = "router_related" if operation_name in ("open", "move", "close") else "placer_or_2q_related"
            failing_operation_index = operation_index if "operation_index" in locals() else None
            failing_route_index = (
                self._route_index_for_operation_index(router_checks, int(failing_operation_index))
                if category == "router_related" and failing_operation_index is not None
                else None
            )
            snapshot = (
                self._build_router_snapshot(
                    block,
                    failing_route_index,
                    f"{type(exc).__name__}: {exc}",
                    operation_index=failing_operation_index,
                    operation=operation if "operation" in locals() else None,
                )
                if category == "router_related"
                else self._build_two_qubit_snapshot(block_index)
            )
            return self._build_failure(
                block_index,
                [f"operation execution failed at stage[{block_index}]: {type(exc).__name__}: {exc}"],
                type(exc).__name__,
                category=category,
                snapshot=snapshot,
            )

        return self._validate_stage_gate_execution(block_index, block, gates_before)

    def _validate_route_checkpoint(
        self,
        block_index: int,
        block: dict[str, Any],
        router_checks: list[dict[str, Any]],
        route_index: int,
    ) -> dict[str, Any] | None:
        expected_placement = router_checks[route_index].get("to_placement")
        if not isinstance(expected_placement, list):
            return None

        for qubit_index, cell in enumerate(expected_placement):
            if not isinstance(cell, (list, tuple)) or len(cell) != 3:
                return self._build_failure(
                    block_index,
                    [f"router expected placement malformed at stage[{block_index}] route[{route_index}] q{qubit_index}: {cell!r}"],
                    "RoutePlacementMismatch",
                    category="router_related",
                    snapshot=self._build_router_snapshot(block, route_index, "malformed expected placement"),
                )
            expected_zone, expected_x, expected_y = int(cell[0]), int(cell[1]), int(cell[2])
            position = getattr(self.machine, "qubit_to_position", {}).get(qubit_index)
            if position is None:
                return self._build_failure(
                    block_index,
                    [f"after router stage[{block_index}] route[{route_index}] missing q{qubit_index} position"],
                    "RoutePlacementMismatch",
                    category="router_related",
                    snapshot=self._build_router_snapshot(block, route_index, f"missing q{qubit_index}"),
                )
            actual_zone = str(getattr(position, "zone", ""))
            actual_x = int(getattr(position, "x", -10**9))
            actual_y = int(getattr(position, "y", -10**9))
            if actual_zone == SnapshotFormatter.zone_name(expected_zone) and actual_x == expected_x and actual_y == expected_y:
                continue
            return self._build_failure(
                block_index,
                [
                    f"after router stage[{block_index}] route[{route_index}] q{qubit_index} mismatch: "
                    f"expected ({SnapshotFormatter.zone_name(expected_zone)},{expected_x},{expected_y}) got ({actual_zone},{actual_x},{actual_y})"
                ],
                "RoutePlacementMismatch",
                category="router_related",
                snapshot=self._build_router_snapshot(block, route_index, f"q{qubit_index} mismatch"),
            )
        return None

    def _validate_stage_gate_execution(
        self,
        block_index: int,
        block: dict[str, Any],
        gates_before: int,
    ) -> dict[str, Any] | None:
        expected_gates = list(block.get("gates") or [])
        machine_circuit = getattr(self.machine, "znaa_format_circuit", None)
        all_gates = list(getattr(machine_circuit, "gates", []))
        actual_new_gates = all_gates[gates_before:]
        if len(actual_new_gates) != len(expected_gates):
            return self._build_failure(
                block_index,
                [f"gate count mismatch at stage[{block_index}]: expected {len(expected_gates)} got {len(actual_new_gates)}"],
                "GateExecutionMismatch",
                category="placer_or_2q_related",
                snapshot=self._build_two_qubit_snapshot(block_index),
            )

        expected_two_qubit = collections.Counter(
            self._expected_gate_signature(gate)
            for gate in expected_gates
            if str(gate.get("gate_type", "")).lower() == "2q"
        )
        actual_two_qubit = collections.Counter(
            self._actual_gate_signature(gate)
            for gate in actual_new_gates
            if str(getattr(gate, "gate_type", "")).lower() == "2q"
        )
        expected_other_gates = [
            self._expected_gate_signature(gate)
            for gate in expected_gates
            if str(gate.get("gate_type", "")).lower() != "2q"
        ]
        actual_other_gates = [
            self._actual_gate_signature(gate)
            for gate in actual_new_gates
            if str(getattr(gate, "gate_type", "")).lower() != "2q"
        ]
        if expected_two_qubit == actual_two_qubit and expected_other_gates == actual_other_gates:
            return None
        return self._build_failure(
            block_index,
            [
                f"gate mismatch stage[{block_index}]: 2q_multiset_equal={expected_two_qubit == actual_two_qubit}, "
                f"non2q_sequence_equal={expected_other_gates == actual_other_gates}"
            ],
            "GateExecutionMismatch",
            category="placer_or_2q_related",
            snapshot=self._build_two_qubit_snapshot(block_index),
        )

    @staticmethod
    def _expected_gate_signature(gate: dict[str, Any]) -> tuple[str, str, tuple[int, ...]]:
        gate_type = str(gate.get("gate_type", "")).lower()
        gate_name = str(gate.get("gate_name", "")).upper()
        qubits_raw = [int(qubit) for qubit in (gate.get("qubits") or [])]
        qubits = tuple(sorted(qubits_raw)) if gate_type == "2q" else tuple(qubits_raw)
        return (gate_type, gate_name, qubits)

    @staticmethod
    def _actual_gate_signature(gate: Any) -> tuple[str, str, tuple[int, ...]]:
        gate_type = str(getattr(gate, "gate_type", "")).lower()
        gate_name = str(getattr(gate, "gate_name", "")).upper()
        qubits_raw = [int(qubit) for qubit in (getattr(gate, "qubits", []) or [])]
        qubits = tuple(sorted(qubits_raw)) if gate_type == "2q" else tuple(qubits_raw)
        return (gate_type, gate_name, qubits)

    def _validate_final_storage_zone(self) -> dict[str, Any] | None:
        last_stage_index = max(len(self.plan) - 1, 0)
        try:
            qubit_to_position = getattr(self.machine, "qubit_to_position", None) or {}
            current_qubits = getattr(self.machine, "current_qubits", None) or set()
            if self.expected_snapshot_n_qubits is not None:
                qubit_ids = list(range(int(self.expected_snapshot_n_qubits)))
            else:
                qubit_ids = sorted(set(qubit_to_position.keys()) | set(current_qubits))

            offenders: list[str] = []
            for qubit in qubit_ids:
                position = qubit_to_position.get(int(qubit))
                if position is None:
                    offenders.append(f"q{qubit}: missing position")
                    continue
                zone = getattr(position, "zone", None)
                if zone == "storage":
                    continue
                x = getattr(position, "x", None)
                y = getattr(position, "y", None)
                offenders.append(f"q{qubit}: zone={zone} pos=({x},{y})")

            if not offenders:
                return None

            details = offenders[:10]
            suffix = f"; ... (total {len(offenders)} offenders)" if len(offenders) > 10 else ""
            return self._build_failure(
                last_stage_index,
                [f"final storage-zone violation: {', '.join(details)}{suffix}"],
                "FinalStorageZoneError",
                category="router_related",
                snapshot=self._build_two_qubit_snapshot(last_stage_index),
            )
        except Exception as exc:
            return self._build_failure(
                last_stage_index,
                [f"final storage-zone check failed with exception: {type(exc).__name__}: {exc}"],
                "FinalStorageZoneError",
                category="router_related",
                snapshot=self._build_two_qubit_snapshot(last_stage_index),
            )

    def _build_success_core(self) -> dict[str, Any]:
        final_result = self.machine.run()
        core = SolverDiagnostics.result_to_core_dict(final_result)
        time_segments = self._extract_machine_time_segments(self.machine)
        if time_segments is not None:
            core["time_segment"] = time_segments
        fidelity_segments = self._extract_machine_fidelity_neglog_segments(self.machine)
        if fidelity_segments is not None:
            core["fidelity_neglog_segment"] = fidelity_segments
        core["plan_execution_checked"] = True
        return core

    @staticmethod
    def _extract_machine_time_segments(machine: Any) -> dict[str, float] | None:
        """Derive per-operation time segments from machine operation history."""
        hist = getattr(machine, "operation_history", None)
        if not isinstance(hist, list):
            return None
        seg = {"1q": 0.0, "2q": 0.0, "move": 0.0, "transfer": 0.0}
        prev_t = 0.0
        for rec in hist:
            if not isinstance(rec, dict):
                continue
            try:
                t_after = float(rec.get("time_after", prev_t))
            except (TypeError, ValueError):
                t_after = prev_t
            if not math.isfinite(t_after):
                t_after = prev_t
            dt = max(0.0, t_after - prev_t)
            prev_t = t_after
            op_type = str(rec.get("op_type", "")).lower()
            if op_type == "1qgate":
                seg["1q"] += dt
            elif op_type == "2qgate":
                seg["2q"] += dt
            elif op_type == "move":
                seg["move"] += dt
            elif op_type in ("open", "close"):
                seg["transfer"] += dt
        if not all(math.isfinite(value) and value >= 0.0 for value in seg.values()):
            return None
        return seg

    @staticmethod
    def _extract_machine_fidelity_neglog_segments(machine: Any) -> dict[str, float] | None:
        """Derive fidelity negative-log segments from machine counters and hardware constants."""
        hp = getattr(machine, "hardware_parameters", None)
        if hp is None:
            return None

        def _neglog_prob(value: Any) -> float:
            try:
                prob = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not (math.isfinite(prob) and 0.0 < prob <= 1.0):
                return 0.0
            return -math.log(prob)

        try:
            n1 = int(getattr(machine, "total_num1q", 0))
            n2 = int(getattr(machine, "total_num2q", 0))
            nt = int(getattr(machine, "total_numtrans", 0))
        except (TypeError, ValueError):
            return None
        if n1 < 0 or n2 < 0 or nt < 0:
            return None

        seg = {
            "1q": float(n1) * _neglog_prob(getattr(hp, "fidelity_1q", None)),
            "2q": float(n2) * _neglog_prob(getattr(hp, "fidelity_2q", None)),
            "move": 0.0,
            "transfer": float(nt) * _neglog_prob(getattr(hp, "fidelity_transfer", None)),
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in seg.values()):
            return None
        return seg


class StageAnalysis:
    """Build per-2Q-stage details, compare them to baseline, and select snapshots."""

    @staticmethod
    def _to_finite_float(value: Any) -> float | None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _to_nonnegative_float(value: Any) -> float | None:
        numeric_value = StageAnalysis._to_finite_float(value)
        if numeric_value is None or numeric_value < 0.0:
            return None
        return numeric_value

    @staticmethod
    def _sanitize_segment_dict(segment: Any) -> dict[str, float] | None:
        if not isinstance(segment, dict):
            return None
        out: dict[str, float] = {}
        for key, value in segment.items():
            numeric_value = StageAnalysis._to_nonnegative_float(value)
            if numeric_value is None:
                continue
            out[str(key)] = numeric_value
        return out if out else None

    @staticmethod
    def _normalize_placement(placement: Any) -> str:
        return SnapshotFormatter.placement_to_labeled_string(placement)

    @classmethod
    def _normalize_placements_chain(cls, start_placement: Any, placements_group: Any) -> list[str]:
        placements: list[str] = []
        if start_placement is not None:
            placements.append(cls._normalize_placement(start_placement))
        for placement in list(placements_group or []):
            placements.append(cls._normalize_placement(placement))
        return placements

    @staticmethod
    def _build_map_operations(utils_module: Any, placement: Any) -> list[Any]:
        map_operations: list[Any] = []
        for qubit_id, cell in enumerate(list(placement or [])):
            if not isinstance(cell, (list, tuple)) or len(cell) != 3:
                raise RuntimeError(f"Invalid stage placement cell: q{qubit_id} -> {cell!r}")
            zone, row, col = int(cell[0]), int(cell[1]), int(cell[2])
            if zone == 0:
                zone_name = "storage"
            elif zone == 1:
                zone_name = "entangling"
            elif zone == 2:
                zone_name = "readout"
            else:
                raise RuntimeError(f"Unsupported stage placement zone: {zone!r}")
            map_operations.append(
                utils_module.Operation_Map(
                    qubit_id=int(qubit_id),
                    coord=(row, col),
                    zone=zone_name,
                )
            )
        return map_operations

    @staticmethod
    def _format_float_text(value: float) -> str:
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @classmethod
    def _format_duration_text(cls, duration_us: float | None) -> str:
        if duration_us is None:
            return "unknown time"
        return f"{cls._format_float_text(duration_us)} us"

    @staticmethod
    def _extract_two_q_pairs(gates: Any) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for gate in list(gates or []):
            if isinstance(gate, dict):
                gate_type = str(gate.get("gate_type", "")).lower()
                qubits = gate.get("qubits") or []
            else:
                gate_type = str(getattr(gate, "gate_type", "")).lower()
                qubits = getattr(gate, "qubits", []) or []
            if gate_type and gate_type != "2q":
                continue
            try:
                parsed_qubits = [int(qubit) for qubit in qubits]
            except (TypeError, ValueError):
                continue
            if len(parsed_qubits) != 2:
                continue
            left_qubit, right_qubit = sorted(parsed_qubits)
            pairs.append((left_qubit, right_qubit))
        return pairs

    @classmethod
    def _format_two_q_gate_pairs(cls, gates: Any) -> str:
        pairs = cls._extract_two_q_pairs(gates)
        return ", ".join(f"(q{left_qubit}, q{right_qubit})" for left_qubit, right_qubit in pairs)

    @classmethod
    def _operation_durations_from_history(cls, operation_history: Any, skip_count: int) -> list[float | None]:
        if not isinstance(operation_history, list):
            return []
        durations: list[float | None] = []
        previous_time = 0.0
        for index, item in enumerate(operation_history):
            current_time = None
            if isinstance(item, dict):
                current_time = cls._to_nonnegative_float(item.get("time_after"))
            if current_time is not None:
                duration = max(0.0, current_time - previous_time)
                previous_time = current_time
            else:
                duration = None
            if index >= skip_count:
                durations.append(duration)
        return durations

    @classmethod
    def _annotate_operations_with_durations(cls, operations: Any, durations: list[float | None]) -> list[str]:
        annotated: list[str] = []
        for index, operation in enumerate(list(operations or [])):
            duration = durations[index] if index < len(durations) else None
            annotated.append(f"{str(operation)} ({cls._format_duration_text(duration)})")
        return annotated

    @staticmethod
    def _strip_stage_identifiers(stage_item: Any) -> dict[str, Any] | None:
        if not isinstance(stage_item, dict):
            return None
        sanitized = dict(stage_item)
        for key in ("stage_index", "plan_block_index", "stage_label", "two_q_stage_index"):
            sanitized.pop(key, None)
        return sanitized

    @staticmethod
    def _group_annotated_route_operations(
        annotated_operations: list[str],
        route_op_counts: Any,
    ) -> list[list[str]]:
        grouped: list[list[str]] = []
        cursor = 0
        for raw_count in list(route_op_counts or []):
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            grouped.append(list(annotated_operations[cursor : cursor + count]))
            cursor += count
        return grouped

    @classmethod
    def _replay_block_metrics(
        cls,
        block: dict[str, Any],
    ) -> tuple[
        float | None,
        float | None,
        dict[str, float] | None,
        dict[str, float] | None,
        list[float | None],
    ]:
        try:
            import utils_ae as _ua
        except Exception:
            return None, None, None, None, []

        start_placement = block.get("placement_before_route")
        if not isinstance(start_placement, list):
            placements_group = block.get("placements_group")
            if isinstance(placements_group, list) and placements_group:
                start_placement = placements_group[0]
            else:
                start_placement = block.get("placements")
        if not isinstance(start_placement, list):
            return None, None, None, None, []

        try:
            machine = _ua.ZNAAMachine(hardware_config=_ua.config_test)
            map_operations = cls._build_map_operations(_ua, start_placement)
            for operation in map_operations:
                machine.append_operation(operation)
            for operation in list(block.get("operations") or []):
                machine.append_operation(operation)
            stage_result = machine.run()
        except Exception:
            return None, None, None, None, []

        stage_core = SolverDiagnostics.result_to_core_dict(stage_result)
        time_value = cls._to_nonnegative_float(stage_core.get("time_us"))
        fidelity_neglog = cls._to_nonnegative_float(CircuitScoring.total_fidelity_neglog(stage_core))
        time_segment = cls._sanitize_segment_dict(PlanExecutionValidator._extract_machine_time_segments(machine))
        fidelity_segment = cls._sanitize_segment_dict(
            PlanExecutionValidator._extract_machine_fidelity_neglog_segments(machine)
        )
        deco_value = cls._to_nonnegative_float(CircuitScoring.deco_neglog(stage_core))
        if deco_value is not None:
            if fidelity_segment is None:
                fidelity_segment = {}
            fidelity_segment["deco"] = deco_value
        operation_durations = cls._operation_durations_from_history(
            getattr(machine, "operation_history", None),
            skip_count=len(map_operations),
        )
        return time_value, fidelity_neglog, time_segment, fidelity_segment, operation_durations

    @classmethod
    def build_two_q_stage_details(cls, plan: Any) -> list[dict[str, Any]]:
        if not isinstance(plan, list):
            return []

        out: list[dict[str, Any]] = []
        stage_index = 0
        for block_index, block in enumerate(plan):
            if not isinstance(block, dict) or SnapshotFormatter.stage_kind(block) != "2q":
                continue

            start_placement = block.get("placement_before_route")
            placements_group = list(block.get("placements_group") or [])
            time_value, fidelity_neglog, time_segment, fidelity_segment, operation_durations = cls._replay_block_metrics(block)
            annotated_operations = cls._annotate_operations_with_durations(
                block.get("operations") or [],
                operation_durations,
            )
            stage_detail: dict[str, Any] = {
                "stage_index": int(stage_index),
                "plan_block_index": int(block_index),
                "stage_label": block.get("stage"),
                "two_q_stage_index": block.get("two_q_stage_index"),
                "gates": cls._format_two_q_gate_pairs(block.get("two_q_gates") or []),
                "placements": cls._normalize_placements_chain(start_placement, placements_group),
                "routing_operations": cls._group_annotated_route_operations(
                    annotated_operations,
                    block.get("route_op_counts"),
                ),
                "operations": annotated_operations,
                "time_us": time_value,
                "fidelity_neglog": fidelity_neglog,
            }
            if time_segment is not None:
                stage_detail["time_segment"] = time_segment
            if fidelity_segment is not None:
                stage_detail["fidelity_segment"] = fidelity_segment
            out.append(stage_detail)
            stage_index += 1
        return out

    @classmethod
    def serialize_for_baseline(cls, stage_details: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in list(stage_details or []):
            if not isinstance(item, dict):
                continue
            serialized: dict[str, Any] = {
                "time_us": item.get("time_us"),
                "fidelity_neglog": item.get("fidelity_neglog"),
            }
            out.append(serialized)
        return out

    @classmethod
    def compare_against_baseline(
        cls,
        current_stage_details: Any,
        baseline_stage_details: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(current_stage_details, list) or not isinstance(baseline_stage_details, list):
            return None

        compared_count = min(len(current_stage_details), len(baseline_stage_details))
        stages: list[dict[str, Any]] = []
        for index in range(compared_count):
            current_stage = current_stage_details[index]
            baseline_stage = baseline_stage_details[index]
            if not isinstance(current_stage, dict) or not isinstance(baseline_stage, dict):
                continue

            current_time = cls._to_nonnegative_float(current_stage.get("time_us"))
            baseline_time = cls._to_nonnegative_float(baseline_stage.get("time_us"))
            current_fidelity = cls._to_nonnegative_float(current_stage.get("fidelity_neglog"))
            baseline_fidelity = cls._to_nonnegative_float(baseline_stage.get("fidelity_neglog"))

            time_improvement = (
                CircuitScoring.compute_improvement(baseline_time, current_time)
                if baseline_time is not None and baseline_time > 0.0 and current_time is not None
                else None
            )
            fidelity_improvement = (
                CircuitScoring.compute_improvement(baseline_fidelity, current_fidelity)
                if baseline_fidelity is not None and baseline_fidelity > 0.0 and current_fidelity is not None
                else None
            )
            combined_improvement = (
                0.5 * (float(time_improvement) + float(fidelity_improvement))
                if time_improvement is not None and fidelity_improvement is not None
                else None
            )

            stage_item: dict[str, Any] = {
                "stage_index": current_stage.get("stage_index", index),
                "stage_label": current_stage.get("stage_label"),
                "two_q_stage_index": current_stage.get("two_q_stage_index"),
                "baseline_time_us": baseline_time,
                "time_us": current_time,
                "time_improvement": time_improvement,
                "baseline_fidelity_neglog": baseline_fidelity,
                "fidelity_neglog": current_fidelity,
                "fidelity_improvement": fidelity_improvement,
                "combined_improvement": combined_improvement,
                "gates": str(current_stage.get("gates") or ""),
                "placements": list(current_stage.get("placements") or []),
                "operations": list(current_stage.get("operations") or []),
            }
            if isinstance(current_stage.get("time_segment"), dict):
                stage_item["time_segment"] = dict(current_stage["time_segment"])
            if isinstance(current_stage.get("fidelity_segment"), dict):
                stage_item["fidelity_segment"] = dict(current_stage["fidelity_segment"])
            stages.append(stage_item)

        return {
            "stage_count_current": len(current_stage_details),
            "stage_count_baseline": len(baseline_stage_details),
            "stage_count_compared": len(stages),
            "stage_count_mismatch": len(current_stage_details) != len(baseline_stage_details),
            "stages": stages,
        }

    @staticmethod
    def _window_size() -> int:
        try:
            size = int(STAGE_ANALYSIS_WINDOW_SIZE)
        except (TypeError, ValueError):
            return 0
        return max(size, 0)

    @staticmethod
    def _threshold_min_ok(value: float | None, threshold: float | None) -> bool:
        if threshold is None:
            return True
        if value is None:
            return False
        return float(value) > float(threshold)

    @staticmethod
    def _threshold_max_ok(value: float | None, threshold: float | None) -> bool:
        if threshold is None:
            return True
        if value is None:
            return False
        return float(value) < float(threshold)

    @staticmethod
    def _positive_temperature(value: Any) -> float | None:
        numeric_value = StageAnalysis._to_finite_float(value)
        if numeric_value is None or numeric_value <= 0.0:
            return None
        return numeric_value

    @staticmethod
    def _stable_weights_from_log_weights(log_weights: list[float | None]) -> list[float]:
        finite_log_weights = [float(value) for value in log_weights if value is not None and math.isfinite(value)]
        if not finite_log_weights:
            return []

        max_log_weight = max(finite_log_weights)
        weights: list[float] = []
        for value in log_weights:
            if value is None or not math.isfinite(value):
                weights.append(0.0)
                continue
            weights.append(math.exp(float(value) - max_log_weight))
        return weights

    @staticmethod
    def _weighted_random_index(weights: list[float]) -> int | None:
        total_weight = 0.0
        for weight in weights:
            if math.isfinite(weight) and weight > 0.0:
                total_weight += float(weight)
        if total_weight <= 0.0:
            return None

        target = random.random() * total_weight
        cumulative = 0.0
        fallback_index: int | None = None
        for index, weight in enumerate(weights):
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            cumulative += float(weight)
            fallback_index = index
            if target <= cumulative:
                return index
        return fallback_index

    @classmethod
    def _window_score(cls, stage_item: Any) -> float | None:
        if not isinstance(stage_item, dict):
            return None
        stage_score = cls._to_finite_float(stage_item.get("combined_improvement"))
        if stage_score is not None:
            return float(stage_score)

        time_improvement = cls._to_finite_float(stage_item.get("time_improvement"))
        fidelity_improvement = cls._to_finite_float(stage_item.get("fidelity_improvement"))
        if time_improvement is None or fidelity_improvement is None:
            return None
        return 0.5 * (float(time_improvement) + float(fidelity_improvement))

    @classmethod
    def _snapshot_windows_for_circuit(
        cls,
        stem: str,
        analysis: dict[str, Any],
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        stages = analysis.get("stages")
        if not isinstance(stages, list) or not stages:
            return []

        window_size = cls._window_size()
        if window_size <= 0 or len(stages) < window_size:
            return []

        circuit_windows: list[dict[str, Any]] = []
        for start in range(0, len(stages) - window_size + 1):
            window = stages[start : start + window_size]
            stage_window_scores: list[float] = []
            for item in window:
                stage_score = cls._window_score(item)
                if stage_score is None:
                    stage_window_scores = []
                    break
                stage_window_scores.append(float(stage_score))
            if len(stage_window_scores) != window_size:
                continue

            window_score = float(sum(stage_window_scores) / float(window_size))
            if threshold is not None and not (window_score < float(threshold)):
                continue

            sanitized_stages: list[dict[str, Any]] = []
            for item in window:
                sanitized = cls._strip_stage_identifiers(item)
                if sanitized is None:
                    sanitized_stages = []
                    break
                sanitized_stages.append(sanitized)
            if len(sanitized_stages) != window_size:
                continue

            circuit_windows.append(
                {
                    "circuit_stem": stem,
                    "circuit_stage": f"{stem} stage {start + 1} ~ {start + window_size}",
                    "window_combined_improvement": window_score,
                    "circuit_score": analysis.get("circuit_score"),
                    "circuit_path": analysis.get("circuit_path"),
                    "circuit_qubit_count": analysis.get("circuit_qubit_count"),
                    "stages": sanitized_stages,
                }
            )
        return circuit_windows

    @staticmethod
    def snapshot_uses_best_reference() -> bool:
        return bool(STAGE_ANALYSIS_FEEDBACK_INCLUDE_BEST_GATE)

    @classmethod
    def snapshot_circuit_score_threshold(cls) -> float | None:
        if STAGE_ANALYSIS_MAX_SNAPSHOT_CIRCUIT_SCORE is not None:
            return cls._to_finite_float(STAGE_ANALYSIS_MAX_SNAPSHOT_CIRCUIT_SCORE)
        return cls._to_finite_float(STAGE_ANALYSIS_MAX_WORST_CIRCUIT_SCORE)

    @classmethod
    def should_emit_snapshot(
        cls,
        combined_score: float | None,
        best_circuit_score: float | None,
        worst_circuit_score: float | None,
    ) -> bool:
        if not cls._threshold_min_ok(combined_score, STAGE_ANALYSIS_MIN_COMBINED_SCORE):
            return False
        if not cls._threshold_min_ok(best_circuit_score, STAGE_ANALYSIS_MIN_BEST_CIRCUIT_SCORE):
            return False
        if not cls._threshold_max_ok(worst_circuit_score, cls.snapshot_circuit_score_threshold()):
            return False
        try:
            probability = float(STAGE_ANALYSIS_PROBABILITY)
        except (TypeError, ValueError):
            return False
        probability = min(max(probability, 0.0), 1.0)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return random.random() < probability

    @classmethod
    def select_smallest_qubit_snapshot(cls, stage_analysis_by_circuit: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        threshold = cls.snapshot_circuit_score_threshold()
        circuit_candidates: list[dict[str, Any]] = []

        for stem, analysis in stage_analysis_by_circuit.items():
            if not isinstance(analysis, dict):
                continue

            qubit_count_raw = analysis.get("circuit_qubit_count")
            if qubit_count_raw is None:
                qubit_count = None
            else:
                try:
                    qubit_count = int(qubit_count_raw)
                except (TypeError, ValueError):
                    qubit_count = None

            if qubit_count is None or qubit_count < 0:
                qubit_rank = float("inf")
            else:
                qubit_rank = float(qubit_count)

            circuit_windows = cls._snapshot_windows_for_circuit(stem, analysis, threshold)
            if not circuit_windows:
                continue

            best_window_score = min(
                float(window["window_combined_improvement"])
                for window in circuit_windows
            )
            circuit_candidates.append(
                {
                    "circuit_stem": stem,
                    "circuit_qubit_count": qubit_count,
                    "qubit_rank": qubit_rank,
                    "best_window_score": best_window_score,
                    "windows": circuit_windows,
                }
            )

        if not circuit_candidates:
            return None

        selected_circuit = min(
            circuit_candidates,
            key=lambda item: (
                float(item["qubit_rank"]),
                float(item["best_window_score"]),
                str(item["circuit_stem"]),
            ),
        )

        circuit_temperature = cls._positive_temperature(STAGE_ANALYSIS_SNAPSHOT_CIRCUIT_QUBIT_T1)
        if circuit_temperature is not None:
            circuit_log_weights: list[float | None] = []
            for item in circuit_candidates:
                qubit_count = item.get("circuit_qubit_count")
                if qubit_count is None:
                    circuit_log_weights.append(None)
                    continue
                circuit_log_weights.append(-float(qubit_count) / circuit_temperature)
            circuit_weights = cls._stable_weights_from_log_weights(circuit_log_weights)
            circuit_index = cls._weighted_random_index(circuit_weights)
            if circuit_index is not None:
                selected_circuit = circuit_candidates[circuit_index]

        windows = list(selected_circuit.get("windows") or [])
        if not windows:
            return None

        selected_window = min(
            windows,
            key=lambda item: (
                float(item["window_combined_improvement"]),
                str(item["circuit_stage"]),
            ),
        )

        window_temperature = cls._positive_temperature(STAGE_ANALYSIS_SNAPSHOT_WINDOW_SCORE_T2)
        if window_temperature is not None:
            window_log_weights: list[float | None] = []
            for item in windows:
                window_score = cls._to_finite_float(item.get("window_combined_improvement"))
                if window_score is None:
                    window_log_weights.append(None)
                    continue
                window_log_weights.append(-float(window_score) / window_temperature)
            window_weights = cls._stable_weights_from_log_weights(window_log_weights)
            window_index = cls._weighted_random_index(window_weights)
            if window_index is not None:
                selected_window = windows[window_index]

        return dict(selected_window)

    @classmethod
    def min_window_score(cls, stage_analysis_by_circuit: dict[str, dict[str, Any]]) -> float | None:
        threshold = cls.snapshot_circuit_score_threshold()
        best_score: float | None = None
        for analysis in stage_analysis_by_circuit.values():
            if not isinstance(analysis, dict):
                continue
            stages = analysis.get("stages")
            if not isinstance(stages, list) or not stages:
                continue
            window_size = cls._window_size()
            if window_size <= 0 or len(stages) < window_size:
                continue

            for start in range(0, len(stages) - window_size + 1):
                window = stages[start : start + window_size]
                stage_window_scores: list[float] = []
                for item in window:
                    if not isinstance(item, dict):
                        stage_window_scores = []
                        break
                    stage_score = cls._to_finite_float(item.get("combined_improvement"))
                    if stage_score is None:
                        time_improvement = cls._to_finite_float(item.get("time_improvement"))
                        fidelity_improvement = cls._to_finite_float(item.get("fidelity_improvement"))
                        if time_improvement is None or fidelity_improvement is None:
                            stage_window_scores = []
                            break
                        stage_score = 0.5 * (float(time_improvement) + float(fidelity_improvement))
                    stage_window_scores.append(float(stage_score))
                if len(stage_window_scores) != window_size:
                    continue

                window_score = float(sum(stage_window_scores) / float(window_size))
                if threshold is not None and not (window_score < float(threshold)):
                    continue
                if best_score is None or window_score < best_score:
                    best_score = window_score
        return best_score


class CircuitScoring:
    """Per-circuit score calculation and segment aggregation."""

    @staticmethod
    def best_update_alpha() -> float:
        try:
            alpha_value = float(BEST_UPDATE_ALPHA)
        except (TypeError, ValueError):
            return 0.5
        return min(max(alpha_value, 0.0), 1.0)

    @classmethod
    def combine_best_update_improvements(
        cls,
        time_improvement: float | None,
        fidelity_improvement: float | None,
    ) -> float | None:
        if time_improvement is None or fidelity_improvement is None:
            return None
        alpha = cls.best_update_alpha()
        return alpha * float(time_improvement) + (1.0 - alpha) * float(fidelity_improvement)

    @staticmethod
    def compute_improvement(baseline_cost: float, current_cost: float) -> float:
        """Return an improvement score where larger means better."""
        metric = str(IMPROVEMENT_METRIC).strip().lower()
        if metric == "relative":
            return (baseline_cost - current_cost) / (current_cost + SCORE_EPS)
        if metric in ("log_ratio", "ln_ratio", "log"):
            baseline_value = max(float(baseline_cost), SCORE_EPS)
            current_value = max(float(current_cost), SCORE_EPS)
            return math.log(baseline_value / current_value)
        raise ValueError(f"Unknown IMPROVEMENT_METRIC: {IMPROVEMENT_METRIC!r}")

    @staticmethod
    def is_no_improvement(time_improvement: float, fidelity_improvement: float) -> bool:
        """Return true when both dimensions are effectively unchanged."""
        if str(IMPROVEMENT_METRIC).strip().lower() in ("log_ratio", "ln_ratio", "log"):
            tol = float(IMPROVEMENT_EQUAL_TOL)
            return abs(time_improvement) <= tol and abs(fidelity_improvement) <= tol
        return time_improvement == 0.0 and fidelity_improvement == 0.0

    @staticmethod
    def non_1q2q_neglog(core: dict[str, Any]) -> float | None:
        """Return the non-(1q,2q) fidelity contribution in negative-log space."""
        comp_neglog = core.get("neg_log_components")
        if isinstance(comp_neglog, dict):
            total = 0.0
            count = 0
            for key, value in comp_neglog.items():
                normalized_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
                if normalized_key in {"1q", "2q", "f1q", "f2q"}:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_value):
                    continue
                total += numeric_value
                count += 1
            return total if count > 0 else 0.0

        comp_lin = core.get("fidelity_components")
        if not isinstance(comp_lin, dict):
            return None

        total = 0.0
        count = 0
        for key, value in comp_lin.items():
            normalized_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
            if normalized_key in {"1q", "2q", "f1q", "f2q"}:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(numeric_value) and numeric_value > 0.0):
                continue
            total += -math.log(numeric_value)
            count += 1
        return total if count > 0 else 0.0

    @staticmethod
    def total_fidelity_neglog(core: dict[str, Any]) -> float | None:
        """Return the total fidelity contribution in negative-log space."""
        comp_neglog = core.get("neg_log_components")
        if isinstance(comp_neglog, dict):
            total = 0.0
            count = 0
            for value in comp_neglog.values():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(numeric_value) and numeric_value >= 0.0):
                    continue
                total += numeric_value
                count += 1
            return total if count > 0 else 0.0

        comp_lin = core.get("fidelity_components")
        if not isinstance(comp_lin, dict):
            return None

        total = 0.0
        count = 0
        for value in comp_lin.values():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(numeric_value) and 0.0 < numeric_value <= 1.0):
                continue
            total += -math.log(numeric_value)
            count += 1
        return total if count > 0 else 0.0

    @staticmethod
    def deco_neglog(core: dict[str, Any]) -> float | None:
        """Extract deco-related fidelity contribution in negative-log space."""
        comp_neglog = core.get("neg_log_components")
        if isinstance(comp_neglog, dict):
            total = 0.0
            found = False
            for key, value in comp_neglog.items():
                normalized_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
                if "deco" not in normalized_key:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(numeric_value) and numeric_value >= 0.0):
                    continue
                total += numeric_value
                found = True
            return total if found else None

        comp_lin = core.get("fidelity_components")
        if not isinstance(comp_lin, dict):
            return None

        total = 0.0
        found = False
        for key, value in comp_lin.items():
            normalized_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
            if "deco" not in normalized_key:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(numeric_value) and 0.0 < numeric_value <= 1.0):
                continue
            total += -math.log(numeric_value)
            found = True
        return total if found else None

    @classmethod
    def attach_circuit_score(
        cls,
        out: dict[str, Any],
        stem: str,
        core: dict[str, Any],
        full_path: Path,
        score_accum: list[tuple[float, float]],
        time_imp_accum: list[tuple[float, float]],
        fidelity_imp_accum: list[tuple[float, float]],
        stage_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        fail = float(FAILURE_CIRCUIT_SCORE)
        weight = CircuitCatalog.circuit_weight(full_path)
        out[f"{stem}_weight"] = weight
        baseline_key = CircuitCatalog.baseline_key(stem)
        meta: dict[str, Any] = {
            "stage_comparison": None,
        }

        if not core.get("ok"):
            out[f"{stem}_score"] = fail
            score_accum.append((fail, weight))
            return meta

        time_us = core.get("time_us")
        fidelity_now_all = cls.total_fidelity_neglog(core)
        if time_us is None or fidelity_now_all is None:
            out[f"{stem}_score"] = fail
            score_accum.append((fail, weight))
            return meta

        time_value = float(time_us)
        if not math.isfinite(time_value) or time_value <= 0.0:
            out[f"{stem}_score"] = fail
            score_accum.append((fail, weight))
            return meta

        if baseline_key not in _CIRCUIT_BASELINES:
            fidelity_now_non12 = cls.non_1q2q_neglog(core)
            baseline_entry: dict[str, Any] = {
                "time_us": time_value,
                "f_all": float(fidelity_now_all),
                "f_non12": float(fidelity_now_non12 if fidelity_now_non12 is not None else fidelity_now_all),
            }
            if stage_details:
                baseline_entry["two_q_stage_details"] = StageAnalysis.serialize_for_baseline(stage_details)
            _CIRCUIT_BASELINES[baseline_key] = baseline_entry
            BaselineStore.save(_CIRCUIT_BASELINES)
            out[f"{stem}_time_improvement"] = 0.0
            out[f"{stem}_fidelity_improvement"] = 0.0
            out[f"{stem}_score"] = 0.0
            score_accum.append((0.0, weight))
            time_imp_accum.append((0.0, weight))
            fidelity_imp_accum.append((0.0, weight))
            return meta

        baseline = _CIRCUIT_BASELINES[baseline_key]
        baseline_time = float(baseline.get("time_us", time_value))
        baseline_dirty = False
        baseline_had_stage_details = isinstance(baseline.get("two_q_stage_details"), list)
        if "f_all" not in baseline:
            baseline["f_all"] = float(fidelity_now_all)
            if "f_non12" not in baseline:
                fidelity_now_non12 = cls.non_1q2q_neglog(core)
                baseline["f_non12"] = float(fidelity_now_non12 if fidelity_now_non12 is not None else fidelity_now_all)
            baseline_dirty = True
        if stage_details and not baseline_had_stage_details:
            baseline["two_q_stage_details"] = StageAnalysis.serialize_for_baseline(stage_details)
            baseline_dirty = True
        if baseline_dirty:
            BaselineStore.save(_CIRCUIT_BASELINES)
        baseline_fidelity = float(baseline.get("f_all", fidelity_now_all))
        if not math.isfinite(baseline_time) or baseline_time <= 0.0:
            out[f"{stem}_score"] = fail
            score_accum.append((fail, weight))
            return meta

        time_improvement = cls.compute_improvement(baseline_time, time_value)
        fidelity_now_value = float(fidelity_now_all)
        if math.isfinite(baseline_fidelity) and baseline_fidelity > 0.0 and math.isfinite(fidelity_now_value) and fidelity_now_value >= 0.0:
            fidelity_improvement = cls.compute_improvement(baseline_fidelity, fidelity_now_value)
        else:
            fidelity_improvement = 0.0

        out[f"{stem}_time_improvement"] = float(time_improvement)
        out[f"{stem}_fidelity_improvement"] = float(fidelity_improvement)
        if cls.is_no_improvement(time_improvement, fidelity_improvement):
            score = -NO_IMPROVEMENT_PENALTY
        else:
            score = 0.5 * (time_improvement + fidelity_improvement)
        if not math.isfinite(score):
            out[f"{stem}_score"] = fail
            score_accum.append((fail, weight))
            return meta

        out[f"{stem}_score"] = float(score)
        score_accum.append((float(score), weight))
        time_imp_accum.append((float(time_improvement), weight))
        fidelity_imp_accum.append((float(fidelity_improvement), weight))
        if stage_details and baseline_had_stage_details:
            meta["stage_comparison"] = StageAnalysis.compare_against_baseline(
                stage_details,
                baseline.get("two_q_stage_details"),
            )
        return meta

    @classmethod
    def attach_circuit_segments(
        cls,
        out: dict[str, Any],
        stem: str,
        core: dict[str, Any],
        segment_accum_time: dict[str, list[tuple[float, float]]],
        segment_accum_fidelity: dict[str, list[tuple[float, float]]],
    ) -> None:
        try:
            weight = float(out.get(f"{stem}_weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0

        time_segment = core.get("time_segment")
        if isinstance(time_segment, dict):
            for key in _TIME_SEGMENT_NAMES:
                value = time_segment.get(key)
                if value is None:
                    continue
                try:
                    segment_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(segment_value) and segment_value >= 0.0):
                    continue
                segment_accum_time[key].append((segment_value, weight))

        fidelity_segment = core.get("fidelity_neglog_segment")
        if isinstance(fidelity_segment, dict):
            for key in _TIME_SEGMENT_NAMES:
                value = fidelity_segment.get(key)
                if value is None:
                    continue
                try:
                    segment_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(segment_value) and segment_value >= 0.0):
                    continue
                segment_accum_fidelity[key].append((segment_value, weight))

        deco_value = cls.deco_neglog(core)
        if deco_value is None:
            return
        try:
            deco_neglog = float(deco_value)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(deco_neglog) and deco_neglog >= 0.0):
            return
        segment_accum_fidelity["deco"].append((deco_neglog, weight))


class PublicOutputBuilder:
    """Helpers for the compact public evaluator output."""

    _HIDDEN_OUTPUT_KEYS = {
        "snapshot_reference_mode",
    }

    @staticmethod
    def prefix_keys(circuit_stem: str, core: dict[str, Any]) -> dict[str, Any]:
        return {f"{circuit_stem}_{key}": value for key, value in core.items() if key != "per_qubit"}

    @staticmethod
    def error_snapshot_key(circuit_stem: str) -> str:
        return f"{circuit_stem}_error_snapshot"

    @staticmethod
    def regression_snapshot_key(circuit_stem: str) -> str:
        return f"{circuit_stem}_regression_snapshot"

    @staticmethod
    def is_snapshot_output_key(key: Any) -> bool:
        return isinstance(key, str) and key.endswith("_snapshot")

    @classmethod
    def _should_hide_output_key(cls, key: Any) -> bool:
        if not isinstance(key, str):
            return False
        key_text = key.strip().lower()
        if not key_text:
            return False
        if key_text in cls._HIDDEN_OUTPUT_KEYS:
            return True
        return "best" in key_text

    @classmethod
    def _strip_hidden_output_fields(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            stripped: dict[str, Any] = {}
            for key, value in obj.items():
                key_text = str(key) if key is not None else ""
                if cls._should_hide_output_key(key_text):
                    continue
                stripped[key] = cls._strip_hidden_output_fields(value)
            return stripped
        if isinstance(obj, list):
            return [cls._strip_hidden_output_fields(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(cls._strip_hidden_output_fields(item) for item in obj)
        return obj

    @staticmethod
    def _prefixed_field(out: dict[str, Any], stem: str, field: str) -> Any:
        return out.get(f"{stem}_{field}")

    @classmethod
    def build_circuit_info(cls, out: dict[str, Any], stem: str) -> dict[str, Any] | None:
        score = cls._prefixed_field(out, stem, "score")
        if score is None:
            return None

        ok = bool(cls._prefixed_field(out, stem, "ok"))
        circuit_info: dict[str, Any] = {
            "circuit": stem,
            "success": ok,
            "score": score,
            "compilation_time": cls._prefixed_field(out, stem, "compilation_time"),
            "time_improvement": cls._prefixed_field(out, stem, "time_improvement"),
            "fidelity_improvement": cls._prefixed_field(out, stem, "fidelity_improvement"),
            "avg_2q_gates_per_2q_stage": cls._prefixed_field(out, stem, "avg_2q_gates_per_2q_stage"),
        }
        if ok:
            return circuit_info

        error = cls._prefixed_field(out, stem, "error")
        message = cls._prefixed_field(out, stem, "message")
        category = cls._prefixed_field(out, stem, "error_category")
        error_text = str(error).strip() if error is not None else ""
        message_text = str(message).strip() if message is not None else ""
        if error_text and message_text:
            circuit_info["error_message"] = f"{error_text}: {message_text}"
        elif error_text:
            circuit_info["error_message"] = error_text
        elif message_text:
            circuit_info["error_message"] = message_text
        else:
            circuit_info["error_message"] = "UnknownError"
        if category is not None:
            circuit_info["error_category"] = str(category)
        return circuit_info

    @classmethod
    def ranked_success_circuit_infos(
        cls,
        out: dict[str, Any],
        circuit_stems: list[str],
        *,
        reverse: bool,
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, str]] = []
        for stem in circuit_stems:
            if not bool(cls._prefixed_field(out, stem, "ok")):
                continue
            raw_score = cls._prefixed_field(out, stem, "score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            ranked.append((score, stem))

        if reverse:
            ranked.sort(key=lambda item: (-item[0], item[1]))
        else:
            ranked.sort(key=lambda item: (item[0], item[1]))

        out_items: list[dict[str, Any]] = []
        for _score, stem in ranked[:RANKED_CIRCUIT_INFO_COUNT]:
            circuit_info = cls.build_circuit_info(out, stem)
            if circuit_info is not None:
                out_items.append(circuit_info)
        return out_items

    @staticmethod
    def prune(out: dict[str, Any], circuit_stems: list[str]) -> dict[str, Any]:
        pruned: dict[str, Any] = {}
        if "ok" in out:
            pruned["ok"] = out["ok"]
        for key in (
            "combined_score",
            "combined_time_improvement",
            "combined_fidelity_improvement",
            "combined_time_segment",
            "combined_fidelity_segment",
            "success_over_total",
        ):
            if key in out:
                pruned[key] = out[key]
        for key, value in out.items():
            if PublicOutputBuilder.is_snapshot_output_key(key):
                pruned[key] = value
        for key in ("component_total_times", "component_time_total_sum"):
            if key in out:
                pruned[key] = out[key]
        if KEEP_COMPONENT_TOP3_CIRCUITS and "component_top3_circuits" in out:
            pruned["component_top3_circuits"] = out["component_top3_circuits"]
        for key in (
            "stopped_on_timeout",
            "stopped_on_total_runtime_timeout",
            "compilation_timeout_sec",
            "total_runtime_timeout_sec",
            "stopped_at_circuit_stem",
        ):
            if key in out:
                pruned[key] = out[key]

        if not circuit_stems:
            for key in ("error", "message", "traceback"):
                if key in out:
                    pruned[key] = out[key]
            return PublicOutputBuilder._strip_hidden_output_fields(pruned)

        error_type_counts: dict[str, int] = {}
        for stem in circuit_stems:
            if bool(out.get(f"{stem}_ok")):
                continue
            error = out.get(f"{stem}_error")
            error_type = str(error).strip() if error is not None else "UnknownError"
            if not error_type:
                error_type = "UnknownError"
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        pruned["error_type_counts"] = error_type_counts

        failed_items: list[tuple[float, str]] = []
        for stem in circuit_stems:
            if bool(out.get(f"{stem}_ok")):
                continue
            try:
                raw_weight = out.get(f"{stem}_weight")
                if raw_weight is None:
                    raise TypeError
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = float("inf")
            failed_items.append((weight, stem))
        failed_items.sort(key=lambda item: item[0])

        for _weight, stem in failed_items[:3]:
            tb = out.get(f"{stem}_traceback")
            if isinstance(tb, str) and tb.strip():
                pruned[f"{stem}_error_traceback"] = tb
                continue
            error = out.get(f"{stem}_error")
            message = out.get(f"{stem}_message")
            stage_idx = out.get(f"{stem}_failure_stage_index")
            pruned[f"{stem}_error_traceback"] = (
                "PseudoTraceback (no Python exception traceback captured)\n"
                f"Error: {error}\n"
                f"Message: {message}\n"
                f"FailureStageIndex: {stage_idx}"
            )

        if failed_items:
            _, stem = failed_items[0]
            snapshot = out.get(f"{stem}_machine_error_context")
            if snapshot is not None:
                pruned[PublicOutputBuilder.error_snapshot_key(stem)] = snapshot

        pruned["highest_score_circuits"] = PublicOutputBuilder.ranked_success_circuit_infos(
            out,
            circuit_stems,
            reverse=True,
        )
        pruned["lowest_score_circuits"] = PublicOutputBuilder.ranked_success_circuit_infos(
            out,
            circuit_stems,
            reverse=False,
        )

        if KEEP_PER_CIRCUIT_INFO:
            for stem in circuit_stems:
                circuit_info = PublicOutputBuilder.build_circuit_info(out, stem)
                if circuit_info is not None:
                    pruned[stem] = circuit_info
        return PublicOutputBuilder._strip_hidden_output_fields(pruned)

    @staticmethod
    def format_number(value: float) -> str:
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4g}"

    @classmethod
    def stringify_numbers(
        cls,
        obj: Any,
        parent_key: str | None = None,
        *,
        in_snapshot: bool = False,
    ) -> Any:
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for key, value in obj.items():
                key_text = str(key) if key is not None else ""
                next_in_snapshot = in_snapshot or PublicOutputBuilder.is_snapshot_output_key(key_text)
                out[key] = cls.stringify_numbers(value, parent_key=key_text, in_snapshot=next_in_snapshot)
            return out
        if isinstance(obj, list):
            return [cls.stringify_numbers(item, parent_key=parent_key, in_snapshot=in_snapshot) for item in obj]
        if isinstance(obj, tuple):
            return tuple(cls.stringify_numbers(item, parent_key=parent_key, in_snapshot=in_snapshot) for item in obj)
        if isinstance(obj, (int, float)):
            if parent_key == "combined_score" or in_snapshot:
                return obj
            return cls.format_number(float(obj))
        return obj


class _PipelineAttempt:
    """Outcome of one candidate pipeline attempt."""

    def __init__(
        self,
        status: str,
        elapsed: float,
        raw: Any = None,
        error_type: str | None = None,
        message: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        self.status = status
        self.elapsed = float(elapsed)
        self.raw = raw
        self.error_type = error_type
        self.message = message
        self.traceback_text = traceback_text


class EvaluationSession:
    """One evaluator run, including candidate import, circuit execution, and aggregation."""

    def __init__(self, filepath: str | Path) -> None:
        self.candidate = Path(filepath).resolve()
        self.out: dict[str, Any] = {"candidate": str(self.candidate)}
        self.paths: list[str] = []
        self.run_started_at: float | None = None
        self.total_timeout_sec: float | None = None
        self.total_timeout_deadline: float | None = None
        self.timeout_sec: float | None = None
        self.circuit_ok_flags: list[bool] = []
        self.score_accum: list[tuple[float, float]] = []
        self.time_imp_accum: list[tuple[float, float]] = []
        self.fidelity_imp_accum: list[tuple[float, float]] = []
        self.segment_accum_time: dict[str, list[tuple[float, float]]] = {name: [] for name in _TIME_SEGMENT_NAMES}
        self.segment_accum_fidelity: dict[str, list[tuple[float, float]]] = {name: [] for name in _FIDELITY_SEGMENT_NAMES}
        self.reuse_validation_stats: dict[str, bool] = {"failed": False}
        self.placement_validation_stats: dict[str, bool] = {"failed": False}
        self.component_time_records: dict[str, list[tuple[str, float]]] = {name: [] for name in _COMPONENT_NAMES}
        self.stage_plan_by_circuit: dict[str, list[dict[str, Any]]] = {}
        self.stage_details_cache: dict[str, list[dict[str, Any]]] = {}
        self.stage_analysis_by_circuit: dict[str, dict[str, Any]] = {}

    def run(self) -> dict[str, Any]:
        global _CIRCUIT_BASELINES
        self.run_started_at = time.perf_counter()
        self.total_timeout_sec = self._resolve_total_runtime_timeout_sec()
        if self.total_timeout_sec is not None:
            self.total_timeout_deadline = self.run_started_at + self.total_timeout_sec
        _CIRCUIT_BASELINES = BaselineStore.load()

        if not self.candidate.is_file():
            return self._build_fatal_output("FileNotFoundError", f"No such file: {self.candidate}")

        prep_err = CandidateContract.prepare_import_extras()
        if prep_err is not None:
            return self._build_fatal_output("ImportError", prep_err)

        mod = self._import_candidate_module()
        if isinstance(mod, dict):
            return mod

        component_spec = self._load_component_spec(mod)
        if isinstance(component_spec, dict) and not component_spec.get("ok", True):
            self.out.update(component_spec)
            return PublicOutputBuilder.prune(self.out, [])

        self.paths = CircuitCatalog.discover(MIN_QUBIT, MAX_QUBIT)
        if not self.paths:
            return self._build_fatal_output(
                "ValueError",
                "Ensure lib/znaa-format circuit contains .znaa files matching *_n{{n}}.znaa with "
                f"MIN_QUBIT <= n <= MAX_QUBIT (currently {MIN_QUBIT}..{MAX_QUBIT}).",
            )

        remaining_total = self._remaining_total_runtime_sec()
        if remaining_total is not None and remaining_total <= 0.0:
            self._record_timeout_skips(
                0,
                error_type="TimeOutError",
                message=self._total_timeout_message(),
            )
            self._finalize_aggregates()
            self._maybe_attach_stage_analysis_snapshot()
            stems = [CircuitCatalog.circuit_stem(CircuitCatalog.build_run_code_arg(path)) for path in self.paths]
            pruned = PublicOutputBuilder.prune(self.out, stems)
            return PublicOutputBuilder.stringify_numbers(pruned)

        self.timeout_sec = self._resolve_timeout_sec()
        self._evaluate_circuits(mod, component_spec)
        self._finalize_aggregates()
        self._update_best_records()
        self._maybe_attach_stage_analysis_snapshot()
        stems = [
            CircuitCatalog.circuit_stem(CircuitCatalog.build_run_code_arg(path))
            for path in self.paths
        ]
        pruned = PublicOutputBuilder.prune(self.out, stems)
        return PublicOutputBuilder.stringify_numbers(pruned)

    def _build_fatal_output(
        self,
        error: str,
        message: str,
        *,
        traceback_text: str | None = None,
    ) -> dict[str, Any]:
        self.out["ok"] = False
        self.out["error"] = error
        self.out["message"] = message
        if traceback_text:
            self.out["traceback"] = traceback_text
        self.out["combined_score"] = float(FAILURE_CIRCUIT_SCORE)
        return PublicOutputBuilder.prune(self.out, [])

    def _import_candidate_module(self) -> Any:
        spec = importlib.util.spec_from_file_location(_EVAL_CANDIDATE_MODULE, self.candidate)
        if spec is None or spec.loader is None:
            return self._build_fatal_output("ImportError", f"Could not create import spec for {self.candidate}")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[_EVAL_CANDIDATE_MODULE] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            sys.modules.pop(_EVAL_CANDIDATE_MODULE, None)
            return self._build_fatal_output(
                type(exc).__name__,
                str(exc),
                traceback_text=traceback.format_exc(),
            )
        return mod

    def _load_component_spec(self, mod: Any) -> dict[str, Any]:
        run_fn: Optional[Callable[..., Any]] = getattr(mod, "run_code", None)
        if run_fn is None or not callable(run_fn):
            return {
                "ok": False,
                "error": "AttributeError",
                "message": "Candidate must define callable run_code() with no arguments.",
                "combined_score": float(FAILURE_CIRCUIT_SCORE),
            }
        return CandidateContract.validate_component_spec(run_fn())

    @staticmethod
    def _resolve_timeout_sec() -> float | None:
        timeout_sec = COMPILATION_TIMEOUT_SEC
        if timeout_sec is None or float(timeout_sec) <= 0:
            return None
        return float(timeout_sec)

    @staticmethod
    def _resolve_total_runtime_timeout_sec() -> float | None:
        timeout_sec = TOTAL_RUNTIME_TIMEOUT_SEC
        if timeout_sec is None or float(timeout_sec) <= 0:
            return None
        return float(timeout_sec)

    def _remaining_total_runtime_sec(self) -> float | None:
        if self.total_timeout_deadline is None:
            return None
        return self.total_timeout_deadline - time.perf_counter()

    def _total_timeout_message(self) -> str:
        return (
            f"evaluation exceeded TOTAL_RUNTIME_TIMEOUT_SEC={self.total_timeout_sec}s "
            "before completing all circuits."
        )

    def _evaluate_circuits(self, mod: Any, component_spec: dict[str, Any]) -> None:
        for path_index, raw_path in enumerate(self.paths):
            remaining_total = self._remaining_total_runtime_sec()
            if remaining_total is not None and remaining_total <= 0.0:
                self._record_timeout_skips(
                    path_index,
                    error_type="TimeOutError",
                    message=self._total_timeout_message(),
                )
                break

            stem, full_path, core, attempt, stage_details = self._evaluate_one_circuit(
                raw_path,
                mod,
                component_spec,
                remaining_total,
            )
            self._record_component_times(stem, SolverDiagnostics.extract_component_times(mod))
            self._record_circuit_output(stem, core, full_path, stage_details)

            if attempt is None or attempt.status != "timeout":
                continue
            timeout_error = attempt.error_type or "TimeoutError"
            timeout_message = attempt.message or ""
            if timeout_error == "TimeOutError":
                self.out["stopped_on_total_runtime_timeout"] = True
                self.out["total_runtime_timeout_sec"] = self.total_timeout_sec
                skip_error_type = "TimeOutError"
                skip_message = self._total_timeout_message()
            else:
                self.out["compilation_timeout_sec"] = self.timeout_sec
                skip_error_type = "SkippedError"
                skip_message = "Not run: evaluation stopped after a compilation timeout on an earlier circuit."
            self.out["stopped_on_timeout"] = True
            self.out["stopped_at_circuit_stem"] = stem
            self._record_timeout_skips(path_index + 1, error_type=skip_error_type, message=skip_message)
            break

    def _evaluate_one_circuit(
        self,
        raw_path: str,
        mod: Any,
        component_spec: dict[str, Any],
        remaining_total_sec: float | None,
    ) -> tuple[str, Path, dict[str, Any], _PipelineAttempt | None, list[dict[str, Any]] | None]:
        circuit_id = CircuitCatalog.build_run_code_arg(raw_path)
        full_path = CircuitCatalog.resolve_file(circuit_id)
        stem = CircuitCatalog.circuit_stem(circuit_id)
        if not full_path.is_file():
            core = SolverDiagnostics.build_failure_core(f"Circuit path does not exist: {full_path}", "FileNotFoundError")
            core["compilation_time"] = 0.0
            core["path"] = str(full_path)
            StageValidation.apply_auxiliary_failures(core)
            return stem, full_path, core, None, None

        attempt = self._run_candidate_pipeline_attempt(component_spec, circuit_id, mod, remaining_total_sec)
        stage_plan: list[dict[str, Any]] | None = None
        if attempt.status == "ok":
            if isinstance(attempt.raw, dict) and isinstance(attempt.raw.get("plan"), list):
                stage_plan = attempt.raw.get("plan")
            core = self._build_success_core(attempt.raw, attempt.elapsed, full_path, mod)
        else:
            core = self._build_failure_core_from_attempt(attempt, full_path, mod)
        return stem, full_path, core, attempt, stage_plan

    def _run_candidate_pipeline_attempt(
        self,
        component_spec: dict[str, Any],
        circuit_id: str,
        mod: Any,
        remaining_total_sec: float | None,
    ) -> _PipelineAttempt:
        started = time.perf_counter()
        if self.timeout_sec is None and remaining_total_sec is None:
            try:
                raw = CandidateContract.execute_pipeline(component_spec, circuit_id, mod)
            except Exception as exc:
                return _PipelineAttempt(
                    status="error",
                    elapsed=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            return _PipelineAttempt(status="ok", elapsed=time.perf_counter() - started, raw=raw)

        effective_timeout = self.timeout_sec
        timeout_error = "TimeoutError"
        timeout_message = (
            f"candidate pipeline exceeded COMPILATION_TIMEOUT_SEC={self.timeout_sec}s "
            "(worker may still be running)."
        )
        if remaining_total_sec is not None and remaining_total_sec > 0.0:
            if effective_timeout is None or remaining_total_sec <= effective_timeout:
                effective_timeout = remaining_total_sec
                timeout_error = "TimeOutError"
                timeout_message = (
                    f"candidate pipeline exceeded TOTAL_RUNTIME_TIMEOUT_SEC={self.total_timeout_sec}s "
                    "(worker may still be running)."
                )

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(CandidateContract.execute_pipeline, component_spec, circuit_id, mod)
        try:
            if effective_timeout is None:
                raw = future.result()
            else:
                raw = future.result(timeout=effective_timeout)
        except FuturesTimeoutError:
            return _PipelineAttempt(
                status="timeout",
                elapsed=time.perf_counter() - started,
                error_type=timeout_error,
                message=timeout_message,
            )
        except Exception as exc:
            return _PipelineAttempt(
                status="error",
                elapsed=time.perf_counter() - started,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return _PipelineAttempt(status="ok", elapsed=time.perf_counter() - started, raw=raw)

    def _build_failure_core_from_attempt(
        self,
        attempt: _PipelineAttempt,
        full_path: Path,
        mod: Any,
    ) -> dict[str, Any]:
        core = SolverDiagnostics.build_failure_core(attempt.message or "", attempt.error_type or "Error")
        core["compilation_time"] = attempt.elapsed
        core["path"] = str(full_path)
        if attempt.traceback_text:
            core["traceback"] = attempt.traceback_text
        SolverDiagnostics.attach_intermediates(core, mod)
        StageValidation.apply_reuse_info_validation(core, mod, self.reuse_validation_stats)
        StageValidation.apply_placement_validation(core, mod, self.placement_validation_stats)
        StageValidation.apply_auxiliary_failures(core)
        return core

    def _build_success_core(
        self,
        raw: Any,
        elapsed: float,
        full_path: Path,
        mod: Any,
    ) -> dict[str, Any]:
        core = SolverDiagnostics.result_to_core_dict(raw)
        core["path"] = str(full_path)
        core["compilation_time"] = elapsed
        if not core.get("ok") and "machine_error_context" not in core:
            SolverDiagnostics.attach_intermediates(core, mod)
        if not core.get("plan_execution_checked"):
            StageValidation.apply_reuse_info_validation(core, mod, self.reuse_validation_stats)
            StageValidation.apply_placement_validation(core, mod, self.placement_validation_stats)
        StageValidation.apply_auxiliary_failures(core)
        return core

    def _record_component_times(self, stem: str, component_times: dict[str, float] | None) -> None:
        if component_times is None:
            return
        for name in _COMPONENT_NAMES:
            self.component_time_records[name].append((stem, component_times[name]))

    def _record_circuit_output(
        self,
        stem: str,
        core: dict[str, Any],
        full_path: Path,
        stage_plan: list[dict[str, Any]] | None,
    ) -> None:
        stage_details: list[dict[str, Any]] | None = None
        if bool(core.get("ok")) and isinstance(stage_plan, list):
            self.stage_plan_by_circuit[stem] = stage_plan
            avg_2q_gates = self._average_2q_gates_per_2q_stage(stage_plan)
            if avg_2q_gates is not None:
                self.out[f"{stem}_avg_2q_gates_per_2q_stage"] = avg_2q_gates
            baseline_key = CircuitCatalog.baseline_key(full_path)
            baseline_entry = _CIRCUIT_BASELINES.get(baseline_key)
            baseline_stage_details = baseline_entry.get("two_q_stage_details") if isinstance(baseline_entry, dict) else None
            if not isinstance(baseline_stage_details, list):
                stage_details = StageAnalysis.build_two_q_stage_details(stage_plan)
                self.stage_details_cache[stem] = stage_details
        self.out.update(PublicOutputBuilder.prefix_keys(stem, core))
        score_meta = CircuitScoring.attach_circuit_score(
            self.out,
            stem,
            core,
            full_path,
            self.score_accum,
            self.time_imp_accum,
            self.fidelity_imp_accum,
            stage_details=stage_details,
        )
        CircuitScoring.attach_circuit_segments(
            self.out,
            stem,
            core,
            self.segment_accum_time,
            self.segment_accum_fidelity,
        )
        stage_comparison = score_meta.get("stage_comparison") if isinstance(score_meta, dict) else None
        if isinstance(stage_comparison, dict):
            self.stage_analysis_by_circuit[stem] = self._annotate_stage_comparison(
                stage_comparison,
                circuit_score=self.out.get(f"{stem}_score"),
                circuit_time_improvement=self.out.get(f"{stem}_time_improvement"),
                circuit_fidelity_improvement=self.out.get(f"{stem}_fidelity_improvement"),
                circuit_path=str(full_path),
                circuit_qubit_count=CircuitCatalog.qubit_count_from_circuit_id(full_path),
            )
        self.circuit_ok_flags.append(bool(core.get("ok")))

    def _ensure_stage_analysis_by_circuit(self) -> None:
        for raw_path in self.paths:
            stem = CircuitCatalog.circuit_stem(CircuitCatalog.build_run_code_arg(raw_path))
            if stem in self.stage_analysis_by_circuit:
                continue
            stage_plan = self.stage_plan_by_circuit.get(stem)
            if not isinstance(stage_plan, list):
                continue
            baseline_entry = _CIRCUIT_BASELINES.get(CircuitCatalog.baseline_key(stem))
            baseline_stage_details = baseline_entry.get("two_q_stage_details") if isinstance(baseline_entry, dict) else None
            if not isinstance(baseline_stage_details, list):
                continue
            stage_details = self.stage_details_cache.get(stem)
            if not isinstance(stage_details, list):
                stage_details = StageAnalysis.build_two_q_stage_details(stage_plan)
                self.stage_details_cache[stem] = stage_details
            stage_comparison = StageAnalysis.compare_against_baseline(stage_details, baseline_stage_details)
            if not isinstance(stage_comparison, dict):
                continue
            self.stage_analysis_by_circuit[stem] = self._annotate_stage_comparison(
                stage_comparison,
                circuit_score=self.out.get(f"{stem}_score"),
                circuit_time_improvement=self.out.get(f"{stem}_time_improvement"),
                circuit_fidelity_improvement=self.out.get(f"{stem}_fidelity_improvement"),
                circuit_path=str(CircuitCatalog.resolve_file(CircuitCatalog.build_run_code_arg(raw_path))),
                circuit_qubit_count=CircuitCatalog.qubit_count_from_circuit_id(raw_path),
            )

    def _stage_details_for_stem(self, stem: str) -> list[dict[str, Any]] | None:
        stage_details = self.stage_details_cache.get(stem)
        if isinstance(stage_details, list):
            return stage_details
        stage_plan = self.stage_plan_by_circuit.get(stem)
        if not isinstance(stage_plan, list):
            return None
        stage_details = StageAnalysis.build_two_q_stage_details(stage_plan)
        self.stage_details_cache[stem] = stage_details
        return stage_details

    @staticmethod
    def _average_2q_gates_per_2q_stage(stage_plan: Any) -> float | None:
        if not isinstance(stage_plan, list):
            return None

        total_two_q_gates = 0
        total_two_q_stages = 0
        for block in stage_plan:
            if not isinstance(block, dict) or SnapshotFormatter.stage_kind(block) != "2q":
                continue
            total_two_q_stages += 1
            total_two_q_gates += len(StageAnalysis._extract_two_q_pairs(block.get("two_q_gates") or []))

        if total_two_q_stages == 0:
            return 0.0
        return float(total_two_q_gates) / float(total_two_q_stages)

    @staticmethod
    def _annotate_stage_comparison(
        stage_comparison: dict[str, Any],
        *,
        circuit_score: Any,
        circuit_time_improvement: Any,
        circuit_fidelity_improvement: Any,
        circuit_path: Any,
        circuit_qubit_count: Any,
    ) -> dict[str, Any]:
        stage_comparison["circuit_score"] = circuit_score
        stage_comparison["circuit_time_improvement"] = circuit_time_improvement
        stage_comparison["circuit_fidelity_improvement"] = circuit_fidelity_improvement
        stage_comparison["circuit_path"] = circuit_path
        stage_comparison["circuit_qubit_count"] = circuit_qubit_count
        return stage_comparison

    @staticmethod
    def _finite_float_or_none(value: Any) -> float | None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _nonempty_text_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    def _build_best_record_for_circuit(self, stem: str, full_path: Path) -> dict[str, Any] | None:
        if not bool(self.out.get(f"{stem}_ok")):
            return None

        time_us = self._finite_float_or_none(self.out.get(f"{stem}_time_us"))
        if time_us is None or time_us <= 0.0:
            return None

        core_like = {
            "neg_log_components": self.out.get(f"{stem}_neg_log_components"),
            "fidelity_components": self.out.get(f"{stem}_fidelity_components"),
        }
        total_fidelity_neglog = CircuitScoring.total_fidelity_neglog(core_like)
        if total_fidelity_neglog is None:
            return None
        non_1q2q_neglog = CircuitScoring.non_1q2q_neglog(core_like)
        stage_details = self._stage_details_for_stem(stem) or []

        record: dict[str, Any] = {
            "path": str(full_path),
            "time_us": float(time_us),
            "f_all": float(total_fidelity_neglog),
            "f_non12": float(non_1q2q_neglog if non_1q2q_neglog is not None else total_fidelity_neglog),
            "two_q_stage_details": StageAnalysis.serialize_for_baseline(stage_details),
        }

        time_improvement = self._finite_float_or_none(self.out.get(f"{stem}_time_improvement"))
        fidelity_improvement = self._finite_float_or_none(self.out.get(f"{stem}_fidelity_improvement"))
        best_update_score = CircuitScoring.combine_best_update_improvements(
            time_improvement,
            fidelity_improvement,
        )
        if time_improvement is not None:
            record["time_improvement"] = time_improvement
        if fidelity_improvement is not None:
            record["fidelity_improvement"] = fidelity_improvement
        if best_update_score is not None and math.isfinite(best_update_score):
            record["best_update_score"] = float(best_update_score)
        evaluator_score = self._finite_float_or_none(self.out.get(f"{stem}_score"))
        if evaluator_score is not None:
            record["evaluator_score"] = evaluator_score
        record["best_update_alpha"] = CircuitScoring.best_update_alpha()

        compilation_time = self._finite_float_or_none(self.out.get(f"{stem}_compilation_time"))
        if compilation_time is not None:
            record["compilation_time"] = compilation_time

        time_segment = StageAnalysis._sanitize_segment_dict(self.out.get(f"{stem}_time_segment"))
        if time_segment is not None:
            record["time_segment"] = time_segment

        fidelity_segment = StageAnalysis._sanitize_segment_dict(self.out.get(f"{stem}_fidelity_neglog_segment"))
        if fidelity_segment is not None:
            record["fidelity_neglog_segment"] = fidelity_segment

        return record

    def _best_record_is_complete(self, record: Any, best_program_path: str | None = None) -> bool:
        if not isinstance(record, dict):
            return False
        time_us = self._finite_float_or_none(record.get("time_us"))
        fidelity_neglog = self._finite_float_or_none(record.get("f_all"))
        stage_details = record.get("two_q_stage_details")
        if best_program_path is None:
            best_program_path = self._nonempty_text_or_none(record.get("best_program_path"))
        if time_us is None or time_us <= 0.0:
            return False
        if fidelity_neglog is None or fidelity_neglog < 0.0:
            return False
        if not isinstance(stage_details, list):
            return False
        if best_program_path is None or not BestPerformanceStore.program_path_exists(best_program_path):
            return False
        return True

    def _baseline_record_for_stem(self, stem: str) -> dict[str, Any] | None:
        baseline = _CIRCUIT_BASELINES.get(CircuitCatalog.baseline_key(stem))
        return baseline if isinstance(baseline, dict) else None

    def _record_best_program_path(self, stem: str, record: Any, program_index: dict[str, dict[str, Any]]) -> str | None:
        index_entry = program_index.get(stem)
        return BestPerformanceStore.preferred_program_path(record, index_entry)

    def _build_program_index_entry(
        self,
        stem: str,
        record: Any,
        existing_entry: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None

        best_program_path = BestPerformanceStore.preferred_program_path(record, existing_entry)
        if best_program_path is None:
            return None

        entry: dict[str, Any] = {
            "best_program_path": best_program_path,
            "best_update_alpha": CircuitScoring.best_update_alpha(),
        }

        if isinstance(existing_entry, dict):
            for key, value in existing_entry.items():
                if key not in entry:
                    entry[key] = value

        for key in ("candidate_path", "updated_at", "program_uuid"):
            value = self._nonempty_text_or_none(record.get(key))
            if value is None and isinstance(existing_entry, dict):
                value = self._nonempty_text_or_none(existing_entry.get(key))
            if value is not None:
                entry[key] = value

        metrics = self._best_metrics_against_baseline(stem, record)
        if metrics is not None:
            entry.update(metrics)
        else:
            for key in ("time_improvement", "fidelity_improvement", "best_update_score"):
                value = self._finite_float_or_none(record.get(key))
                if value is None and isinstance(existing_entry, dict):
                    value = self._finite_float_or_none(existing_entry.get(key))
                if value is not None:
                    entry[key] = value

        time_us = self._finite_float_or_none(record.get("time_us"))
        if time_us is None and isinstance(existing_entry, dict):
            time_us = self._finite_float_or_none(existing_entry.get("time_us"))
        if time_us is not None:
            entry["time_us"] = time_us

        fidelity_neglog = self._finite_float_or_none(record.get("f_all"))
        if fidelity_neglog is None and isinstance(existing_entry, dict):
            fidelity_neglog = self._finite_float_or_none(existing_entry.get("f_all"))
        if fidelity_neglog is not None:
            entry["f_all"] = fidelity_neglog

        return entry

    def _rebuild_program_index(
        self,
        records: dict[str, dict[str, Any]],
        existing_index: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        rebuilt: dict[str, dict[str, Any]] = {}
        for stem, record in records.items():
            entry = self._build_program_index_entry(stem, record, existing_index.get(stem))
            if entry is not None:
                rebuilt[stem] = entry
        return rebuilt, rebuilt != existing_index

    def _best_metrics_against_baseline(
        self,
        stem: str,
        record: dict[str, Any],
    ) -> dict[str, float] | None:
        baseline = self._baseline_record_for_stem(stem)
        if baseline is None:
            return None

        baseline_time = self._finite_float_or_none(baseline.get("time_us"))
        baseline_fidelity = self._finite_float_or_none(baseline.get("f_all"))
        current_time = self._finite_float_or_none(record.get("time_us"))
        current_fidelity = self._finite_float_or_none(record.get("f_all"))
        if baseline_time is None or baseline_time <= 0.0 or current_time is None or current_time <= 0.0:
            return None
        if baseline_fidelity is None or baseline_fidelity < 0.0:
            return None
        if current_fidelity is None or current_fidelity < 0.0:
            return None

        time_improvement = CircuitScoring.compute_improvement(baseline_time, current_time)
        fidelity_improvement = CircuitScoring.compute_improvement(baseline_fidelity, current_fidelity)
        best_update_score = CircuitScoring.combine_best_update_improvements(
            time_improvement,
            fidelity_improvement,
        )
        if best_update_score is None or not math.isfinite(best_update_score):
            return None

        return {
            "time_improvement": float(time_improvement),
            "fidelity_improvement": float(fidelity_improvement),
            "best_update_score": float(best_update_score),
        }

    def _normalize_existing_best_record(
        self,
        stem: str,
        record: Any,
        program_index: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, bool]:
        if not isinstance(record, dict):
            return None, False

        normalized = dict(record)
        changed = False

        best_program_path = self._record_best_program_path(stem, normalized, program_index)
        if best_program_path is not None and normalized.get("best_program_path") != best_program_path:
            normalized["best_program_path"] = best_program_path
            changed = True

        index_entry = program_index.get(stem)
        if isinstance(index_entry, dict):
            for record_key, index_key in (
                ("program_uuid", "program_uuid"),
                ("candidate_path", "candidate_path"),
                ("updated_at", "updated_at"),
            ):
                if normalized.get(record_key) is None and index_entry.get(index_key) is not None:
                    normalized[record_key] = index_entry.get(index_key)
                    changed = True

        metrics = self._best_metrics_against_baseline(stem, normalized)
        if metrics is not None:
            for key, value in metrics.items():
                current_value = self._finite_float_or_none(normalized.get(key))
                if current_value is None or abs(current_value - value) > SCORE_EPS:
                    normalized[key] = value
                    changed = True
            alpha_value = CircuitScoring.best_update_alpha()
            current_alpha = self._finite_float_or_none(normalized.get("best_update_alpha"))
            if current_alpha is None or abs(current_alpha - alpha_value) > SCORE_EPS:
                normalized["best_update_alpha"] = alpha_value
                changed = True

        return normalized, changed

    def _update_best_records(self) -> None:
        if not self.paths or not self.candidate.is_file():
            return

        current_candidates: dict[str, tuple[dict[str, Any], dict[str, float]]] = {}

        for raw_path in self.paths:
            circuit_id = CircuitCatalog.build_run_code_arg(raw_path)
            full_path = CircuitCatalog.resolve_file(circuit_id)
            stem = CircuitCatalog.circuit_stem(circuit_id)
            current_record = self._build_best_record_for_circuit(stem, full_path)
            if current_record is None:
                continue
            current_metrics = self._best_metrics_against_baseline(stem, current_record)
            if current_metrics is None:
                continue
            current_record.update(current_metrics)
            current_candidates[stem] = (current_record, current_metrics)

        with BestPerformanceStore.locked():
            best_records = BestPerformanceStore.load_records()
            loaded_program_index = BestPerformanceStore.load_program_index()
            records_dirty = False

            for stem in sorted(set(best_records).union(loaded_program_index)):
                normalized_previous_record, previous_changed = self._normalize_existing_best_record(
                    stem,
                    best_records.get(stem),
                    loaded_program_index,
                )
                if previous_changed and normalized_previous_record is not None:
                    best_records[stem] = normalized_previous_record
                    records_dirty = True

            saved_program: tuple[str, str] | None = None
            updated_at: str | None = None

            for stem, (current_record, current_metrics) in current_candidates.items():
                previous_record = best_records.get(stem)
                previous_program_path = self._record_best_program_path(stem, previous_record, loaded_program_index)

                update_reason: str | None = None
                if not self._best_record_is_complete(previous_record, previous_program_path):
                    update_reason = "missing_or_incomplete_best_record"
                else:
                    assert isinstance(previous_record, dict)
                    previous_metrics = self._best_metrics_against_baseline(stem, previous_record)
                    if previous_metrics is None:
                        update_reason = "missing_best_baseline_metrics"
                    elif current_metrics["best_update_score"] > previous_metrics["best_update_score"] + float(IMPROVEMENT_EQUAL_TOL):
                        update_reason = "better_than_previous_best"

                if update_reason is None:
                    continue

                if saved_program is None:
                    saved_program = BestPerformanceStore.save_candidate_program(self.candidate)
                    if saved_program is None:
                        return
                    updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                assert updated_at is not None
                program_uuid, best_program_path = saved_program
                BestPerformanceStore.save_circuit_plan(stem, self.stage_plan_by_circuit.get(stem))

                record = dict(current_record)
                record["candidate_path"] = str(self.candidate)
                record["updated_at"] = updated_at
                record["best_program_path"] = best_program_path
                record["program_uuid"] = program_uuid
                record["best_update_alpha"] = CircuitScoring.best_update_alpha()
                record["best_update_reason"] = update_reason
                best_records[stem] = record
                records_dirty = True

            program_index, index_dirty = self._rebuild_program_index(best_records, loaded_program_index)
            if not records_dirty and not index_dirty:
                return

            if records_dirty and not BestPerformanceStore.save_records(best_records):
                return
            if (records_dirty or index_dirty) and not BestPerformanceStore.save_program_index(program_index):
                return

            referenced_paths = BestPerformanceStore.referenced_program_paths(best_records, program_index)
            BestPerformanceStore.delete_unreferenced_program_files(referenced_paths)

    def _best_snapshot_reference_stem(self) -> str | None:
        best_stem: str | None = None
        best_score: float | None = None
        best_qubit_count: int | None = None
        for stem, analysis in self.stage_analysis_by_circuit.items():
            if not isinstance(analysis, dict):
                continue
            score = StageAnalysis._to_finite_float(analysis.get("circuit_score"))
            if score is None:
                continue
            qubit_count_raw = analysis.get("circuit_qubit_count")
            try:
                qubit_count = int(qubit_count_raw) if qubit_count_raw is not None else None
            except (TypeError, ValueError):
                qubit_count = None

            if best_stem is None:
                best_stem = stem
                best_score = score
                best_qubit_count = qubit_count
                continue

            assert best_score is not None
            current_rank = (score, -float(qubit_count) if qubit_count is not None else float("-inf"))
            best_rank = (best_score, -float(best_qubit_count) if best_qubit_count is not None else float("-inf"))
            if current_rank > best_rank:
                best_stem = stem
                best_score = score
                best_qubit_count = qubit_count
        return best_stem

    def _build_stage_analysis_against_reference(self, reference_stage_details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not isinstance(reference_stage_details, list):
            return {}

        out: dict[str, dict[str, Any]] = {}
        for stem, baseline_analysis in self.stage_analysis_by_circuit.items():
            if not isinstance(baseline_analysis, dict):
                continue
            current_stage_details = self._stage_details_for_stem(stem)
            if not isinstance(current_stage_details, list):
                continue

            stage_comparison = StageAnalysis.compare_against_baseline(current_stage_details, reference_stage_details)
            if not isinstance(stage_comparison, dict):
                continue
            out[stem] = self._annotate_stage_comparison(
                stage_comparison,
                circuit_score=baseline_analysis.get("circuit_score"),
                circuit_time_improvement=baseline_analysis.get("circuit_time_improvement"),
                circuit_fidelity_improvement=baseline_analysis.get("circuit_fidelity_improvement"),
                circuit_path=baseline_analysis.get("circuit_path"),
                circuit_qubit_count=baseline_analysis.get("circuit_qubit_count"),
            )
        return out

    def _record_timeout_skips(
        self,
        start_index: int,
        *,
        error_type: str,
        message: str,
    ) -> None:
        self.out["stopped_on_timeout"] = True
        if error_type == "TimeOutError":
            self.out["stopped_on_total_runtime_timeout"] = True
            self.out["total_runtime_timeout_sec"] = self.total_timeout_sec
        else:
            self.out["compilation_timeout_sec"] = self.timeout_sec

        if start_index < len(self.paths):
            first_path = self.paths[start_index]
            if not self.out.get("stopped_at_circuit_stem"):
                self.out["stopped_at_circuit_stem"] = CircuitCatalog.circuit_stem(first_path)

        for raw_path in self.paths[start_index:]:
            circuit_id = CircuitCatalog.build_run_code_arg(raw_path)
            full_path = CircuitCatalog.resolve_file(circuit_id)
            stem = CircuitCatalog.circuit_stem(circuit_id)
            skipped = SolverDiagnostics.build_failure_core(
                message,
                error_type,
            )
            skipped["compilation_time"] = None
            skipped["path"] = str(full_path)
            StageValidation.apply_auxiliary_failures(skipped)
            self._record_circuit_output(stem, skipped, full_path, None)

    def _finalize_aggregates(self) -> None:
        success_count = sum(1 for value in self.circuit_ok_flags if value)
        total_count = len(self.circuit_ok_flags)
        self.out["ok"] = bool(total_count > 0 and success_count == total_count)
        self.out["success_over_total"] = f"{success_count}/{total_count}"
        self.out["combined_score"] = self._weighted_average(self.score_accum, default=float(FAILURE_CIRCUIT_SCORE))
        self.out["combined_time_improvement"] = self._weighted_average(self.time_imp_accum, default=0.0)
        self.out["combined_fidelity_improvement"] = self._weighted_average(self.fidelity_imp_accum, default=0.0)

        self.out["combined_time_segment"] = {
            name: self._weighted_average(self.segment_accum_time[name], default=0.0)
            for name in _TIME_SEGMENT_NAMES
        }
        self.out["combined_fidelity_segment"] = {
            name: self._weighted_average(self.segment_accum_fidelity[name], default=0.0)
            for name in _FIDELITY_SEGMENT_NAMES
        }

        component_total_times = {
            name: float(sum(value for _stem, value in self.component_time_records[name]))
            for name in _COMPONENT_NAMES
        }
        self.out["component_total_times"] = component_total_times
        self.out["component_time_total_sum"] = float(sum(component_total_times.values()))
        if KEEP_COMPONENT_TOP3_CIRCUITS:
            self.out["component_top3_circuits"] = {
                name: [
                    {"circuit": stem, "time": float(value)}
                    for stem, value in sorted(
                        self.component_time_records[name],
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                ]
                for name in _COMPONENT_NAMES
            }

    def _numeric_circuit_scores(self) -> list[float]:
        scores: list[float] = []
        for raw_path in self.paths:
            stem = CircuitCatalog.circuit_stem(CircuitCatalog.build_run_code_arg(raw_path))
            value = self.out.get(f"{stem}_score")
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                scores.append(numeric_value)
        return scores

    def _all_circuits_snapshot_eligible(self) -> bool:
        if not self.paths:
            return False
        if not bool(self.out.get("ok")):
            return False
        for raw_path in self.paths:
            stem = CircuitCatalog.circuit_stem(CircuitCatalog.build_run_code_arg(raw_path))
            if not bool(self.out.get(f"{stem}_ok")):
                return False
            if self.out.get(f"{stem}_error") not in (None, ""):
                return False
            if self.out.get(f"{stem}_plan_execution_checked") is not True:
                return False
            if self.out.get(f"{stem}_reuse_info_validation_ok") is False:
                return False
            if self.out.get(f"{stem}_placement_validation_ok") is False:
                return False
        return True

    def _maybe_attach_stage_analysis_snapshot(self) -> None:
        if not self._all_circuits_snapshot_eligible():
            return

        circuit_scores = self._numeric_circuit_scores()
        raw_combined_score = self.out.get("combined_score")
        if raw_combined_score is None:
            combined_score = None
        else:
            try:
                combined_score = float(raw_combined_score)
            except (TypeError, ValueError):
                combined_score = None
        best_circuit_score = max(circuit_scores) if circuit_scores else None

        self._ensure_stage_analysis_by_circuit()
        if not self.stage_analysis_by_circuit:
            return

        reference_mode = "best" if StageAnalysis.snapshot_uses_best_reference() else "baseline"
        analysis_by_circuit = self.stage_analysis_by_circuit
        if reference_mode == "best":
            best_stem = self._best_snapshot_reference_stem()
            if best_stem is None:
                return
            reference_stage_details = self._stage_details_for_stem(best_stem)
            if not isinstance(reference_stage_details, list):
                return
            analysis_by_circuit = self._build_stage_analysis_against_reference(reference_stage_details)
            if not analysis_by_circuit:
                return

        worst_window_score = StageAnalysis.min_window_score(analysis_by_circuit)
        if not StageAnalysis.should_emit_snapshot(combined_score, best_circuit_score, worst_window_score):
            return

        snapshot = StageAnalysis.select_smallest_qubit_snapshot(analysis_by_circuit)
        if snapshot is None:
            return

        snapshot_stem = snapshot.get("circuit_stem")
        if not isinstance(snapshot_stem, str) or not snapshot_stem:
            circuit_path = snapshot.get("circuit_path")
            if circuit_path is not None:
                try:
                    snapshot_stem = CircuitCatalog.circuit_stem(circuit_path)
                except Exception:
                    snapshot_stem = None
        if not isinstance(snapshot_stem, str) or not snapshot_stem:
            return

        self.out[PublicOutputBuilder.regression_snapshot_key(snapshot_stem)] = {
            "circuit_stage": snapshot.get("circuit_stage"),
            "best_circuit_score": best_circuit_score,
            "worst_circuit_score": worst_window_score,
            "worst_window_score": worst_window_score,
            "selected_circuit_score": snapshot.get("circuit_score"),
            "selected_circuit_qubit_count": snapshot.get("circuit_qubit_count"),
            "snapshot_circuit_score_threshold": StageAnalysis.snapshot_circuit_score_threshold(),
            "snapshot_window_score_threshold": StageAnalysis.snapshot_circuit_score_threshold(),
            "feedback_include_best_gate": STAGE_ANALYSIS_FEEDBACK_INCLUDE_BEST_GATE,
            "snapshot_reference_mode": reference_mode,
            "selected_window_score": snapshot.get("window_combined_improvement"),
            "window_combined_improvement": snapshot.get("window_combined_improvement"),
            "stages": snapshot.get("stages"),
        }

    @staticmethod
    def _weighted_average(values: list[tuple[float, float]], *, default: float) -> float:
        if not values:
            return default
        denominator = sum(weight for _value, weight in values)
        if denominator > 0.0:
            return float(sum(value * weight for value, weight in values) / denominator)
        return float(sum(value for value, _weight in values) / len(values)) if values else default


def evaluate(filepath: str | Path) -> dict[str, Any]:
    """Evaluate one candidate file on benchmark circuits and return merged public metrics."""
    return EvaluationSession(filepath).run()


def main() -> None:
    """CLI: evaluate bundled ``alphaevolve.py`` on auto-discovered circuits and print merged dict."""
    import pprint

    alpha_path = _EVAL_ROOT / "init_program.py"
    merged = evaluate(alpha_path)
    pprint.pprint(merged, width=120, sort_dicts=False)


if __name__ == "__main__":
    main()
