"""
SimpleTES evaluator for Rust qubit-routing policy mutation.

The candidate program may be either:
- a full `candidate.rs` file, or
- only the EVOLVE-BLOCK body wrapped by EVOLVE-BLOCK markers.

Each evaluation acquires a per-slot isolated Rust workspace under /var/tmp,
reconstructs `candidate.rs` when needed, compiles `router_cli`, runs the
benchmark suite, and returns a score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RUST_DIR = ROOT / "rust"
CANDIDATE_SCAFFOLD_PATH = RUST_DIR / "router_core" / "src" / "candidate.rs"
DEFAULT_SUITE_PATH = ROOT / "python" / "benchmarks" / "sabre_suite.json"
TARGET_DIR = RUST_DIR / ".cargo_target"
DEFAULT_RESULTS_DIR = REPO_ROOT / "checkpoints" / "qubit_routing_results" / "results"

DEFAULT_TOTAL_BUDGET_SECONDS = 1200.0
DEFAULT_TIMEOUT_HEADROOM_SECONDS = 60.0
DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS = 10.0
RESULTS_DIR = Path(
    os.environ.get("QUBIT_ROUTING_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
).expanduser()
SLOT_ROOT = Path(
    os.environ.get("QUBIT_ROUTING_SLOT_ROOT", "/tmp/simpletes-qubit-routing-slots")
)
SLOT_COUNT = max(
    1,
    int(os.environ.get("QUBIT_ROUTING_SLOT_COUNT") or "4"),
)
DEFAULT_SLOT_NAMESPACE = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:16]
SLOT_NAMESPACE = os.environ.get("QUBIT_ROUTING_SLOT_NAMESPACE", DEFAULT_SLOT_NAMESPACE)
SLOT_PREWARM_TARGET = os.environ.get("QUBIT_ROUTING_SLOT_PREWARM_TARGET", "1") != "0"

MAX_EQUIVALENCE_BRANCH_POINTS = int(
    os.environ.get("QUBIT_ROUTING_EQUIVALENCE_BRANCH_POINTS", "48")
)

FAST_LAYOUT_TRIALS = 4
FAST_ROUTING_TRIALS = 4
FULL_LAYOUT_TRIALS = 20
FULL_ROUTING_TRIALS = 20
PROMPT_SUMMARY_TOP_K = 3

EVOLVE_BLOCK_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<start>[ \t]*(?://\s*)?EVOLVE-BLOCK-START[ \t]*\n)"
    r"(?P<body>.*?)(?P<end>[ \t]*(?://\s*)?EVOLVE-BLOCK-END[ \t]*(?:\n|$))"
    r"(?P<suffix>.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class SuiteCaseSpec:
    case_id: str
    qasm3_path: Path
    topology_path: Path
    original_cnot_added: int
    weight: float


@dataclass(frozen=True)
class TopologySpec:
    num_qubits: int
    edges: set[tuple[int, int]]


@dataclass(frozen=True)
class CircuitInstructionView:
    op: Any
    name: str
    qargs: tuple[int, ...]
    cargs: tuple[int, ...]
    position: int


@dataclass(frozen=True)
class OriginalDagNode:
    index: int
    op: Any
    qargs: tuple[int, ...]
    cargs: tuple[int, ...]
    successors: tuple[int, ...]


@dataclass
class ReplayState:
    pending_predecessors: list[int]
    front_layer: set[int]
    logical_to_physical: dict[int, int]
    physical_to_logical: dict[int, int]
    executed_nodes: int

    def clone(self) -> ReplayState:
        return ReplayState(
            pending_predecessors=list(self.pending_predecessors),
            front_layer=set(self.front_layer),
            logical_to_physical=dict(self.logical_to_physical),
            physical_to_logical=dict(self.physical_to_logical),
            executed_nodes=self.executed_nodes,
        )


@dataclass
class SlotHandle:
    slot_id: int
    slot_dir: Path
    rust_dir: Path
    target_dir: Path
    lock_path: Path
    lock_fd: int


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _resolve_total_budget_seconds() -> float:
    headroom = _env_float(
        "QUBIT_ROUTING_TIMEOUT_HEADROOM_SECONDS",
        DEFAULT_TIMEOUT_HEADROOM_SECONDS,
    )
    if not math.isfinite(headroom):
        headroom = DEFAULT_TIMEOUT_HEADROOM_SECONDS
    headroom = max(0.0, headroom)

    parent_raw = os.environ.get("QUBIT_ROUTING_PARENT_EVAL_TIMEOUT_SECONDS")
    if parent_raw is not None:
        try:
            parent_timeout = float(parent_raw)
            if math.isfinite(parent_timeout):
                return max(1.0, parent_timeout - headroom)
        except ValueError:
            pass

    fallback_budget = _env_float(
        "QUBIT_ROUTING_TOTAL_BUDGET_SECONDS",
        DEFAULT_TOTAL_BUDGET_SECONDS,
    )
    if not math.isfinite(fallback_budget):
        fallback_budget = DEFAULT_TOTAL_BUDGET_SECONDS
    return max(1.0, fallback_budget)


def _resolve_slot_lock_timeout_seconds(total_budget_seconds: float) -> float:
    value = _env_float("QUBIT_ROUTING_SLOT_LOCK_TIMEOUT_SECONDS", total_budget_seconds)
    if not math.isfinite(value):
        return total_budget_seconds
    return max(1.0, value)


def _resolve_results_dir(raw_path: str | None) -> Path:
    results_dir = RESULTS_DIR if raw_path is None else Path(raw_path).expanduser()
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()
    else:
        results_dir = results_dir.resolve()
    return results_dir


@lru_cache(maxsize=1)
def _resolve_python_probe_timeout_seconds() -> float:
    value = _env_float(
        "QUBIT_ROUTING_PYTHON_PROBE_TIMEOUT_SECONDS",
        DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS,
    )
    if not math.isfinite(value):
        value = DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS
    return max(1.0, value)


def _resolve_trial_counts(
    trial_mode: str,
    layout_trials: int | None,
    routing_trials: int | None,
) -> tuple[str, int, int]:
    mode = str(trial_mode or "fast").strip().lower()
    if mode not in {"fast", "full"}:
        raise RuntimeError(
            f"invalid trial_mode={trial_mode!r}; expected 'fast' or 'full'"
        )

    if mode == "full":
        default_layout_trials = FULL_LAYOUT_TRIALS
        default_routing_trials = FULL_ROUTING_TRIALS
    else:
        default_layout_trials = FAST_LAYOUT_TRIALS
        default_routing_trials = FAST_ROUTING_TRIALS

    resolved_layout_trials = (
        int(layout_trials) if layout_trials is not None else default_layout_trials
    )
    resolved_routing_trials = (
        int(routing_trials) if routing_trials is not None else default_routing_trials
    )

    if resolved_layout_trials < 1:
        raise RuntimeError(
            f"layout_trials must be >= 1, got {resolved_layout_trials}"
        )
    if resolved_routing_trials < 1:
        raise RuntimeError(
            f"routing_trials must be >= 1, got {resolved_routing_trials}"
        )

    return mode, resolved_layout_trials, resolved_routing_trials


def acquire_slot(
    start_time: float,
    total_budget_seconds: float,
    slot_lock_timeout_seconds: float,
) -> tuple[SlotHandle, float]:
    namespace_root = SLOT_ROOT / SLOT_NAMESPACE
    namespace_root.mkdir(parents=True, exist_ok=True)
    slot_wait_start = time.time()

    while True:
        for slot_id in range(SLOT_COUNT):
            slot_dir = namespace_root / f"slot-{slot_id}"
            lock_path = slot_dir / "lock"
            slot_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue

            return (
                SlotHandle(
                    slot_id=slot_id,
                    slot_dir=slot_dir,
                    rust_dir=slot_dir / "rust",
                    target_dir=slot_dir / "target",
                    lock_path=lock_path,
                    lock_fd=fd,
                ),
                time.time() - slot_wait_start,
            )

        elapsed = time.time() - slot_wait_start
        budget_elapsed = time.time() - start_time
        if elapsed >= slot_lock_timeout_seconds or budget_elapsed >= total_budget_seconds:
            raise TimeoutError(
                "Timed out waiting for an evaluator slot "
                f"(slots={SLOT_COUNT}, timeout={slot_lock_timeout_seconds:.1f}s)"
            )
        time.sleep(0.1)


def release_slot(slot: SlotHandle | None) -> None:
    if slot is None:
        return
    try:
        fcntl.flock(slot.lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(slot.lock_fd)


def clean_slot_workspaces() -> None:
    """Delete all slot workspaces for the active namespace after acquiring locks."""
    namespace_root = SLOT_ROOT / SLOT_NAMESPACE
    if not namespace_root.exists():
        return

    locked_fds: list[int] = []
    try:
        for slot_id in range(SLOT_COUNT):
            slot_dir = namespace_root / f"slot-{slot_id}"
            lock_path = slot_dir / "lock"
            slot_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise RuntimeError(
                    f"Cannot clean slot workspaces: slot-{slot_id} is currently in use."
                ) from exc
            locked_fds.append(fd)

        shutil.rmtree(namespace_root, ignore_errors=True)
    finally:
        for fd in locked_fds:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def prepare_slot_workspace(slot: SlotHandle) -> None:
    slot.rust_dir.mkdir(parents=True, exist_ok=True)
    canonical_items = [
        "Cargo.toml",
        "Cargo.lock",
        "router_core",
        "router_cli",
        "vendor",
    ]

    # Initialize slot source once and reuse it across evaluations.
    # Re-sync only if required files are missing.
    ready_marker = slot.slot_dir / ".workspace_ready"
    needs_sync = not ready_marker.exists()
    if not needs_sync:
        for rel in canonical_items:
            if not (slot.rust_dir / rel).exists():
                needs_sync = True
                break

    if needs_sync:
        for rel in canonical_items:
            _copy_path(RUST_DIR / rel, slot.rust_dir / rel)
        ready_marker.write_text("1\n", encoding="utf-8")

    if slot.target_dir.exists():
        return
    if SLOT_PREWARM_TARGET and TARGET_DIR.exists():
        shutil.copytree(TARGET_DIR, slot.target_dir)
    else:
        slot.target_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _candidate_scaffold_source() -> str:
    scaffold = CANDIDATE_SCAFFOLD_PATH.read_text(encoding="utf-8")
    if EVOLVE_BLOCK_RE.search(scaffold) is None:
        raise RuntimeError(
            f"candidate scaffold missing EVOLVE-BLOCK markers: {CANDIDATE_SCAFFOLD_PATH}"
        )
    return scaffold


def _extract_evolve_block_body(text: str) -> str | None:
    block = EVOLVE_BLOCK_RE.search(text)
    if block is None:
        return None

    return block.group("body").strip() + "\n"


def _reconstruct_candidate_from_body(body: str) -> str:
    scaffold = _candidate_scaffold_source()
    block = EVOLVE_BLOCK_RE.search(scaffold)
    if block is None:
        raise RuntimeError(
            f"candidate scaffold missing EVOLVE-BLOCK markers: {CANDIDATE_SCAFFOLD_PATH}"
        )

    body_clean = body.strip("\n")
    if not body_clean:
        raise RuntimeError("candidate EVOLVE-BLOCK body is empty")

    return (
        f"{block.group('prefix')}{block.group('start')}"
        f"{body_clean}\n"
        f"{block.group('end')}{block.group('suffix')}"
    )


def _extract_candidate_code(program_path: str) -> str:
    text = Path(program_path).read_text(encoding="utf-8")

    body = _extract_evolve_block_body(text)
    if body is not None:
        return _reconstruct_candidate_from_body(body)

    rust_block = re.search(r"```rust\s*(.*?)```", text, re.DOTALL)
    if rust_block:
        fenced = rust_block.group(1).strip() + "\n"
        fenced_body = _extract_evolve_block_body(fenced)
        if fenced_body is not None:
            return _reconstruct_candidate_from_body(fenced_body)
        return fenced

    return text.strip() + "\n"


def _inject_swap_selection_context_compat_shim(candidate_code: str) -> str:
    """Backfill optional APIs expected by newer route engines.

    Older checkpoint programs may not define
    `SwapSelectionContext::new_with_precomputed_extended_set(...)`.
    Newer route implementations can call that constructor to pass cached
    extended-set data. When absent, inject a fallback constructor that
    ignores the precomputed payload and delegates to the legacy `new(...)`.
    """

    if not re.search(r"\bstruct\s+SwapSelectionContext\b", candidate_code):
        return candidate_code
    if re.search(
        r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:crate\s+)?fn\s+new_with_precomputed_extended_set\s*\(",
        candidate_code,
    ):
        return candidate_code
    if not re.search(
        r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:crate\s+)?fn\s+new\s*\(",
        candidate_code,
    ):
        return candidate_code

    shim = """
impl<'a> SwapSelectionContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_precomputed_extended_set(
        topology: TopologyView<'a>,
        layout: LayoutView<'a>,
        circuit: CircuitDagView<'a>,
        remaining: RemainingDagView<'a>,
        front_layer: FrontLayerView<'a>,
        _precomputed_extended_set_logical_pairs: &'a [[usize; 2]],
        swaps_since_progress: usize,
        last_applied_swap: Option<(usize, usize)>,
    ) -> Self {
        Self::new(
            topology,
            layout,
            circuit,
            remaining,
            front_layer,
            swaps_since_progress,
            last_applied_swap,
        )
    }
}
"""
    return candidate_code.rstrip() + "\n\n" + shim.lstrip()


def _remaining_budget(start: float, total_budget_seconds: float) -> float:
    return max(1.0, total_budget_seconds - (time.time() - start))


def _select_pyo3_python(env: Dict[str, str]) -> str | None:
    explicit = env.get("PYO3_PYTHON")
    if explicit and Path(explicit).exists():
        return explicit

    current = Path(sys.executable)
    if current.exists():
        return str(current)

    venv = env.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    uv_env = env.get("UV_PROJECT_ENVIRONMENT")
    if uv_env:
        candidate = Path(uv_env) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    return None


def _run(cmd, cwd: Path, timeout_seconds: float, env: Dict[str, str] | None = None) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        def _coerce_output(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        partial_stdout = _coerce_output(exc.stdout)
        partial_stderr = _coerce_output(exc.stderr)

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        try:
            tail_stdout, tail_stderr = proc.communicate(timeout=2.0)
        except Exception:
            tail_stdout, tail_stderr = "", ""
        else:
            tail_stdout = _coerce_output(tail_stdout)
            tail_stderr = _coerce_output(tail_stderr)

        raise subprocess.TimeoutExpired(
            cmd,
            timeout_seconds,
            output=(partial_stdout + tail_stdout),
            stderr=(partial_stderr + tail_stderr),
        ) from exc

    if proc.returncode != 0:
        stderr = (stderr or "").strip()
        stdout = (stdout or "").strip()
        msg = "\n".join(part for part in [stderr, stdout] if part)
        raise RuntimeError(msg[:8000] if msg else f"Command failed: {' '.join(cmd)}")


def _site_packages_for_python(python_exe: str) -> list[str]:
    probe = (
        "import json, site; "
        "paths = list(site.getsitepackages()); "
        "usersite = site.getusersitepackages(); "
        "paths.append(usersite); "
        "print(json.dumps([p for p in paths if p]))"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            text=True,
            capture_output=True,
            timeout=_resolve_python_probe_timeout_seconds(),
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [str(Path(p)) for p in payload if isinstance(p, str)]


def _stdlib_paths_for_python(python_exe: str) -> list[str]:
    probe = (
        "import json, sysconfig; "
        "paths = []; "
        "for key in ('stdlib', 'platstdlib'): "
        "    p = sysconfig.get_path(key); "
        "    paths.append(p if p else ''); "
        "print(json.dumps([p for p in paths if p]))"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            text=True,
            capture_output=True,
            timeout=_resolve_python_probe_timeout_seconds(),
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [str(Path(p)) for p in payload if isinstance(p, str)]


def _python_home_for_python(python_exe: str) -> str | None:
    probe = "import sys; print(getattr(sys, 'base_prefix', sys.prefix))"
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            text=True,
            capture_output=True,
            timeout=_resolve_python_probe_timeout_seconds(),
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        return None
    return str(path)


def _resolve_suite_relative_path(suite_dir: Path, raw_path: str, field_name: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = suite_dir / path
    if not path.exists():
        raise RuntimeError(f"suite {field_name} path does not exist: {path}")
    return path


def _load_suite_metadata(suite_path: Path) -> Dict[str, SuiteCaseSpec]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("suite.cases must be a list")

    suite_dir = suite_path.parent
    by_case_id: Dict[str, SuiteCaseSpec] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("each suite case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("each suite case must include a non-empty string 'id'")
        if case_id in by_case_id:
            raise RuntimeError(f"duplicate suite case id: {case_id}")

        raw_qasm3_path = case.get("qasm3_path")
        if not isinstance(raw_qasm3_path, str) or not raw_qasm3_path:
            raise RuntimeError(
                f"suite case '{case_id}' is missing required string field 'qasm3_path'"
            )
        raw_topology_path = case.get("topology_path")
        if not isinstance(raw_topology_path, str) or not raw_topology_path:
            raise RuntimeError(
                f"suite case '{case_id}' is missing required string field 'topology_path'"
            )

        raw_original = case.get("original_cnot_added")
        if raw_original is None:
            raise RuntimeError(
                f"suite case '{case_id}' is missing required field 'original_cnot_added'"
            )
        try:
            original_cnot_added = int(raw_original)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"suite case '{case_id}' has invalid original_cnot_added={raw_original!r}"
            ) from exc
        if original_cnot_added < 0:
            raise RuntimeError(
                f"suite case '{case_id}' has negative original_cnot_added={original_cnot_added}"
            )

        raw_weight = case.get("weight", 1.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"suite case '{case_id}' has invalid weight={raw_weight!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise RuntimeError(
                f"suite case '{case_id}' has non-finite or negative weight={weight}"
            )

        by_case_id[case_id] = SuiteCaseSpec(
            case_id=case_id,
            qasm3_path=_resolve_suite_relative_path(suite_dir, raw_qasm3_path, "qasm3_path"),
            topology_path=_resolve_suite_relative_path(
                suite_dir, raw_topology_path, "topology_path"
            ),
            original_cnot_added=original_cnot_added,
            weight=weight,
        )

    return by_case_id


def _load_original_cnot_added(suite_metadata: Dict[str, SuiteCaseSpec]) -> Dict[str, int]:
    return {
        case_id: case_spec.original_cnot_added
        for case_id, case_spec in suite_metadata.items()
    }


def _load_case_weights(suite_metadata: Dict[str, SuiteCaseSpec]) -> Dict[str, float]:
    return {
        case_id: float(case_spec.weight)
        for case_id, case_spec in suite_metadata.items()
    }


@lru_cache(maxsize=1)
def _qiskit_api() -> Dict[str, Any]:
    try:
        from qiskit import qasm3
        from qiskit.converters import circuit_to_dag
        from qiskit.dagcircuit import DAGOpNode
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "Qiskit is required for routing equivalence checks. "
            "Ensure qiskit and qiskit_qasm3_import are installed."
        ) from exc
    return {
        "qasm3": qasm3,
        "circuit_to_dag": circuit_to_dag,
        "DAGOpNode": DAGOpNode,
    }


def _instruction_parts(inst: Any) -> tuple[Any, list[Any], list[Any]]:
    if hasattr(inst, "operation") and hasattr(inst, "qubits") and hasattr(inst, "clbits"):
        return inst.operation, list(inst.qubits), list(inst.clbits)
    if isinstance(inst, (tuple, list)) and len(inst) == 3:
        op, qargs, cargs = inst
        return op, list(qargs), list(cargs)
    raise RuntimeError(f"unexpected instruction representation: {type(inst)!r}")


def _find_bit_index(owner: Any, bit: Any, bit_kind: str) -> int:
    find_bit = getattr(owner, "find_bit", None)
    if callable(find_bit):
        try:
            located = find_bit(bit)
            located_index = getattr(located, "index", None)
            if isinstance(located_index, int):
                return located_index
        except Exception:
            # Fall through to compatibility fallbacks below.
            pass

    for attr in ("index", "_index"):
        fallback = getattr(bit, attr, None)
        if isinstance(fallback, int):
            return fallback
    raise RuntimeError(f"could not resolve {bit_kind} bit index for {bit!r}")


def _extract_instruction_views(circuit: Any) -> list[CircuitInstructionView]:
    views: list[CircuitInstructionView] = []

    for position, inst in enumerate(circuit.data):
        op, qbits, cbits = _instruction_parts(inst)
        qargs = tuple(_find_bit_index(circuit, bit, "quantum") for bit in qbits)
        cargs = tuple(_find_bit_index(circuit, bit, "classical") for bit in cbits)
        name = str(getattr(op, "name", "") or "").lower()
        views.append(
            CircuitInstructionView(
                op=op,
                name=name,
                qargs=qargs,
                cargs=cargs,
                position=position,
            )
        )

    return views


def _dag_node_id(node: Any) -> int:
    raw = getattr(node, "_node_id", None)
    if isinstance(raw, int):
        return raw

    raw = getattr(node, "node_id", None)
    if callable(raw):
        maybe = raw()
        if isinstance(maybe, int):
            return maybe
    elif isinstance(raw, int):
        return raw

    raise RuntimeError(f"unable to determine DAG node id for {type(node)!r}")


def _build_original_nodes(original_circuit: Any) -> list[OriginalDagNode]:
    api = _qiskit_api()
    DAGOpNode = api["DAGOpNode"]
    circuit_to_dag = api["circuit_to_dag"]
    dag = circuit_to_dag(original_circuit)

    if any(getattr(node.op, "blocks", ()) for node in dag.op_nodes()):
        raise RuntimeError(
            "control-flow operations are not yet supported by this equivalence checker"
        )

    topo_nodes = list(dag.topological_op_nodes())
    id_to_index = {_dag_node_id(node): i for i, node in enumerate(topo_nodes)}
    op_successors: list[list[int]] = [[] for _ in topo_nodes]

    for node in topo_nodes:
        node_index = id_to_index[_dag_node_id(node)]
        seen_successors: set[int] = set()
        for successor in dag.successors(node):
            if isinstance(successor, DAGOpNode):
                successor_index = id_to_index[_dag_node_id(successor)]
                if successor_index not in seen_successors:
                    op_successors[node_index].append(successor_index)
                    seen_successors.add(successor_index)

    nodes: list[OriginalDagNode] = []
    for node in topo_nodes:
        node_index = id_to_index[_dag_node_id(node)]
        qargs = tuple(_find_bit_index(dag, bit, "quantum") for bit in node.qargs)
        cargs = tuple(_find_bit_index(dag, bit, "classical") for bit in node.cargs)
        nodes.append(
            OriginalDagNode(
                index=node_index,
                op=node.op,
                qargs=qargs,
                cargs=cargs,
                successors=tuple(op_successors[node_index]),
            )
        )

    return nodes


def _param_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, (int, float, complex)) and isinstance(rhs, (int, float, complex)):
        lhs_complex = complex(lhs)
        rhs_complex = complex(rhs)

        def _component_equal(a: float, b: float) -> bool:
            # Keep validator consistent with router fallback sanitization in Rust:
            # sub-1e-4 literals may be rewritten to 0.0 before DAG import.
            if abs(a) < 1.0e-4 and abs(b) < 1.0e-4:
                return True
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

        return _component_equal(lhs_complex.real, rhs_complex.real) and _component_equal(
            lhs_complex.imag, rhs_complex.imag
        )
    return str(lhs) == str(rhs)


def _ops_match(original_op: Any, routed_op: Any) -> bool:
    soft_compare = getattr(original_op, "soft_compare", None)
    if callable(soft_compare):
        try:
            if bool(soft_compare(routed_op)):
                return True
        except Exception:
            # Fall back to explicit matching below.
            pass

    if getattr(original_op, "name", None) != getattr(routed_op, "name", None):
        return False
    if getattr(original_op, "num_qubits", None) != getattr(routed_op, "num_qubits", None):
        return False
    if getattr(original_op, "num_clbits", None) != getattr(routed_op, "num_clbits", None):
        return False
    if repr(getattr(original_op, "condition", None)) != repr(
        getattr(routed_op, "condition", None)
    ):
        return False

    original_params = list(getattr(original_op, "params", ()))
    routed_params = list(getattr(routed_op, "params", ()))
    if len(original_params) != len(routed_params):
        return False
    return all(_param_equal(lhs, rhs) for lhs, rhs in zip(original_params, routed_params))


def _match_assignments(
    logical_qargs: tuple[int, ...],
    physical_qargs: tuple[int, ...],
    logical_to_physical: dict[int, int],
    physical_to_logical: dict[int, int],
) -> tuple[int, int] | None:
    if len(logical_qargs) != len(physical_qargs):
        return None
    if len(set(physical_qargs)) != len(physical_qargs):
        return None

    known_matches = 0
    new_assignments = 0
    for logical, physical in zip(logical_qargs, physical_qargs):
        known_physical = logical_to_physical.get(logical)
        if known_physical is not None:
            if known_physical != physical:
                return None
            known_matches += 1
        else:
            new_assignments += 1

        known_logical = physical_to_logical.get(physical)
        if known_logical is not None and known_logical != logical:
            return None

    return known_matches, new_assignments


def _apply_node_match(
    state: ReplayState,
    node: OriginalDagNode,
    physical_qargs: tuple[int, ...],
) -> None:
    for logical, physical in zip(node.qargs, physical_qargs):
        if logical not in state.logical_to_physical:
            state.logical_to_physical[logical] = physical
        if physical not in state.physical_to_logical:
            state.physical_to_logical[physical] = logical

    state.front_layer.remove(node.index)
    state.executed_nodes += 1
    for successor in node.successors:
        state.pending_predecessors[successor] -= 1
        if state.pending_predecessors[successor] == 0:
            state.front_layer.add(successor)


def _apply_inserted_swap(state: ReplayState, a: int, b: int) -> None:
    logical_a = state.physical_to_logical.get(a)
    logical_b = state.physical_to_logical.get(b)

    if logical_a is not None:
        state.logical_to_physical[logical_a] = b
    if logical_b is not None:
        state.logical_to_physical[logical_b] = a

    if logical_a is not None:
        state.physical_to_logical[b] = logical_a
    else:
        state.physical_to_logical.pop(b, None)

    if logical_b is not None:
        state.physical_to_logical[a] = logical_b
    else:
        state.physical_to_logical.pop(a, None)


def _replay_front_layer_equivalence(
    original_nodes: list[OriginalDagNode],
    routed_instructions: list[CircuitInstructionView],
    topology_edges: set[tuple[int, int]],
    initial_logical_to_physical: dict[int, int],
    initial_physical_to_logical: dict[int, int],
) -> str | None:
    state = ReplayState(
        pending_predecessors=[0 for _ in original_nodes],
        front_layer=set(),
        logical_to_physical=dict(initial_logical_to_physical),
        physical_to_logical=dict(initial_physical_to_logical),
        executed_nodes=0,
    )
    for node in original_nodes:
        for successor in node.successors:
            state.pending_predecessors[successor] += 1
    for node in original_nodes:
        if state.pending_predecessors[node.index] == 0:
            state.front_layer.add(node.index)

    def recurse(
        op_index: int,
        cur_state: ReplayState,
        remaining_branch_points: int,
    ) -> str | None:
        i = op_index
        state_local = cur_state
        while i < len(routed_instructions):
            inst = routed_instructions[i]
            candidate_actions: list[tuple[tuple[int, int, int, int], str, int | None]] = []

            for node_index in state_local.front_layer:
                node = original_nodes[node_index]
                if not _ops_match(node.op, inst.op):
                    continue
                if node.cargs != inst.cargs:
                    continue
                assignment = _match_assignments(
                    node.qargs,
                    inst.qargs,
                    state_local.logical_to_physical,
                    state_local.physical_to_logical,
                )
                if assignment is None:
                    continue
                known_matches, new_assignments = assignment
                # Prefer consuming an already-constrained, low-ambiguity front-layer op first.
                rank = (0, -known_matches, new_assignments, node_index)
                candidate_actions.append((rank, "consume", node_index))

            if inst.name == "swap" and len(inst.qargs) == 2:
                a, b = inst.qargs
                edge = (a, b) if a <= b else (b, a)
                if edge in topology_edges:
                    # Try consuming original DAG node(s) first if available; if not, allow inserted SWAP.
                    candidate_actions.append(((1, 0, 0, 0), "insert_swap", None))

            if not candidate_actions:
                front_preview = sorted(state_local.front_layer)[:8]
                return (
                    f"equivalence replay failed at routed op #{inst.position} '{inst.name}' "
                    f"on qargs={list(inst.qargs)} cargs={list(inst.cargs)}; "
                    f"front_layer_size={len(state_local.front_layer)} front_preview={front_preview}"
                )

            candidate_actions.sort(key=lambda item: item[0])

            if len(candidate_actions) == 1:
                _, action, maybe_node_index = candidate_actions[0]
                if action == "consume":
                    _apply_node_match(
                        state_local,
                        original_nodes[int(maybe_node_index)],
                        inst.qargs,
                    )
                else:
                    _apply_inserted_swap(state_local, inst.qargs[0], inst.qargs[1])
                i += 1
                continue

            if remaining_branch_points <= 0:
                _, action, maybe_node_index = candidate_actions[0]
                if action == "consume":
                    _apply_node_match(
                        state_local,
                        original_nodes[int(maybe_node_index)],
                        inst.qargs,
                    )
                else:
                    _apply_inserted_swap(state_local, inst.qargs[0], inst.qargs[1])
                i += 1
                continue

            for _, action, maybe_node_index in candidate_actions:
                branched_state = state_local.clone()
                if action == "consume":
                    _apply_node_match(
                        branched_state,
                        original_nodes[int(maybe_node_index)],
                        inst.qargs,
                    )
                else:
                    _apply_inserted_swap(branched_state, inst.qargs[0], inst.qargs[1])
                err = recurse(i + 1, branched_state, remaining_branch_points - 1)
                if err is None:
                    return None
            return (
                f"equivalence replay ambiguity could not be resolved at routed op #{inst.position} "
                f"('{inst.name}') with {len(candidate_actions)} candidate actions"
            )

        if state_local.executed_nodes != len(original_nodes):
            remaining = len(original_nodes) - state_local.executed_nodes
            front_preview = sorted(state_local.front_layer)[:8]
            return (
                f"routed circuit ended with {remaining} original DAG node(s) not executed; "
                f"front_layer_size={len(state_local.front_layer)} front_preview={front_preview}"
            )

        return None

    return recurse(0, state, MAX_EQUIVALENCE_BRANCH_POINTS)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_topology_spec(topology_path: Path) -> TopologySpec:
    payload = json.loads(topology_path.read_text(encoding="utf-8"))
    num_qubits = payload.get("num_qubits")
    if not _is_plain_int(num_qubits):
        raise RuntimeError(f"topology.num_qubits must be an integer in {topology_path}")
    num_qubits = int(num_qubits)
    if num_qubits < 0:
        raise RuntimeError(f"topology.num_qubits must be non-negative in {topology_path}")

    edges = payload.get("edges")
    if not isinstance(edges, list):
        raise RuntimeError(f"topology.edges must be a list in {topology_path}")

    normalized: set[tuple[int, int]] = set()
    for edge in edges:
        if (
            not isinstance(edge, (list, tuple))
            or len(edge) != 2
            or not _is_plain_int(edge[0])
            or not _is_plain_int(edge[1])
        ):
            raise RuntimeError(f"invalid topology edge entry {edge!r} in {topology_path}")
        a, b = int(edge[0]), int(edge[1])
        if a < 0 or b < 0 or a >= num_qubits or b >= num_qubits:
            raise RuntimeError(
                f"topology edge {edge!r} uses out-of-range qubit index in {topology_path}"
            )
        normalized.add((a, b) if a <= b else (b, a))
    return TopologySpec(num_qubits=num_qubits, edges=normalized)


def _num_qubits(circuit: Any) -> int:
    raw = getattr(circuit, "num_qubits", None)
    if callable(raw):
        raw = raw()
    if not _is_plain_int(raw):
        raise RuntimeError(f"unable to resolve num_qubits for circuit type {type(circuit)!r}")
    return int(raw)


def _validate_initial_mapping(
    case: Dict[str, Any],
    num_logical_qubits: int,
    num_physical_qubits: int,
) -> tuple[dict[int, int], dict[int, int], str | None]:
    raw_mapping = case.get("initial_mapping")
    if not isinstance(raw_mapping, list):
        return {}, {}, "missing initial_mapping in router result"
    if len(raw_mapping) != num_logical_qubits:
        return (
            {},
            {},
            "invalid initial_mapping length: "
            f"expected {num_logical_qubits}, got {len(raw_mapping)}",
        )

    logical_to_physical: dict[int, int] = {}
    physical_to_logical: dict[int, int] = {}
    for logical, raw_physical in enumerate(raw_mapping):
        if not _is_plain_int(raw_physical):
            return (
                {},
                {},
                f"invalid initial_mapping[{logical}]={raw_physical!r}: must be an integer",
            )
        physical = int(raw_physical)
        if physical < 0 or physical >= num_physical_qubits:
            return (
                {},
                {},
                f"invalid initial_mapping[{logical}]={physical}: out of range for "
                f"{num_physical_qubits} physical qubits",
            )
        if physical in physical_to_logical:
            prior_logical = physical_to_logical[physical]
            return (
                {},
                {},
                f"invalid initial_mapping: physical qubit {physical} assigned to "
                f"both logical qubits {prior_logical} and {logical}",
            )
        logical_to_physical[logical] = physical
        physical_to_logical[physical] = logical
    return logical_to_physical, physical_to_logical, None


def _validate_case_routing(case: Dict[str, Any], case_spec: SuiteCaseSpec) -> str | None:
    output_circuit = case.get("output_circuit")
    if not isinstance(output_circuit, str) or not output_circuit.strip():
        return "missing or empty output_circuit in router result"

    api = _qiskit_api()
    qasm3 = api["qasm3"]
    try:
        original_circuit = qasm3.loads(case_spec.qasm3_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"failed to parse original qasm3 '{case_spec.qasm3_path}': {exc}"
    try:
        routed_circuit = qasm3.loads(output_circuit)
    except Exception as exc:
        return f"failed to parse routed output_circuit as qasm3: {exc}"

    topology = _load_topology_spec(case_spec.topology_path)
    original_num_logical_qubits = _num_qubits(original_circuit)
    initial_logical_to_physical, initial_physical_to_logical, mapping_error = (
        _validate_initial_mapping(case, original_num_logical_qubits, topology.num_qubits)
    )
    if mapping_error is not None:
        return mapping_error

    routed_instructions = _extract_instruction_views(routed_circuit)
    for inst in routed_instructions:
        for qarg in inst.qargs:
            if qarg < 0 or qarg >= topology.num_qubits:
                return (
                    f"topology violation at routed op #{inst.position} '{inst.name}': "
                    f"qubit index {qarg} is out of range for {topology.num_qubits} qubits"
                )
        if inst.name == "barrier":
            continue
        if len(inst.qargs) != 2:
            continue
        a, b = inst.qargs
        edge = (a, b) if a <= b else (b, a)
        if edge not in topology.edges:
            return (
                f"topology violation at routed op #{inst.position} '{inst.name}': "
                f"two-qubit gate on non-edge ({a}, {b})"
            )

    original_nodes = _build_original_nodes(original_circuit)
    return _replay_front_layer_equivalence(
        original_nodes,
        routed_instructions,
        topology.edges,
        initial_logical_to_physical,
        initial_physical_to_logical,
    )


def _apply_equivalence_validation(
    results: Dict[str, Any],
    suite_metadata: Dict[str, SuiteCaseSpec],
) -> None:
    cases = results.get("cases")
    if not isinstance(cases, list):
        return

    for case in cases:
        if not isinstance(case, dict):
            continue
        if not case.get("ok"):
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str):
            continue
        case_spec = suite_metadata.get(case_id)
        if case_spec is None:
            continue

        try:
            validation_error = _validate_case_routing(case, case_spec)
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"

        case["validation_error"] = validation_error
        if validation_error is not None:
            case["ok"] = False
            previous_error = case.get("error")
            if isinstance(previous_error, str) and previous_error:
                case["error"] = f"{previous_error}; validation_failed: {validation_error}"
            else:
                case["error"] = f"validation_failed: {validation_error}"


def _score_results(
    results: Dict[str, Any],
    eval_time: float,
    original_cnot_added_by_case_id: Dict[str, int],
    case_weight_by_case_id: Dict[str, float],
) -> Dict[str, Any]:
    cases = results.get("cases", [])
    if not isinstance(cases, list):
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(eval_time),
            "error": "results.cases must be a list",
        }

    total_cases = len(cases)
    ok_cases = [c for c in cases if c.get("ok")]
    failed_cases = [c for c in cases if not c.get("ok")]

    total_swaps = int(sum(c.get("swap_count", 0) for c in ok_cases))
    total_depth = int(sum(c.get("depth", 0) for c in ok_cases))
    total_runtime_ms = int(sum(c.get("time_ms", 0) for c in ok_cases))

    if total_cases == 0:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(eval_time),
            "error": "suite has zero cases",
        }

    total_original_cnot_added = 0
    total_added_cnot_by_candidate = 0
    weighted_total_original_cnot_added = 0.0
    weighted_total_added_cnot_by_candidate = 0.0
    seen_case_ids: set[str] = set()
    unknown_result_case_ids: list[str] = []

    for case in cases:
        if not isinstance(case, dict):
            return {
                "combined_score": 0.0,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "error": "each results case must be an object",
            }

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            return {
                "combined_score": 0.0,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "error": "each results case must include a non-empty string 'id'",
            }
        seen_case_ids.add(case_id)

        original_cnot_added = original_cnot_added_by_case_id.get(case_id)
        case_weight = case_weight_by_case_id.get(case_id)
        if original_cnot_added is None or case_weight is None:
            unknown_result_case_ids.append(case_id)
            continue

        total_original_cnot_added += original_cnot_added
        if case.get("ok"):
            added_cnot_by_candidate = _additional_cnot_added_from_swap_count(
                case.get("swap_count", 0)
            )
        else:
            # Do not give failed cases credit.
            added_cnot_by_candidate = original_cnot_added
        total_added_cnot_by_candidate += added_cnot_by_candidate
        weighted_total_original_cnot_added += float(case_weight) * float(original_cnot_added)
        weighted_total_added_cnot_by_candidate += float(case_weight) * float(added_cnot_by_candidate)

    missing_result_case_ids = sorted(set(original_cnot_added_by_case_id) - seen_case_ids)
    if unknown_result_case_ids or missing_result_case_ids:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(eval_time),
            "error": (
                "suite/results case id mismatch: "
                f"unknown_result_case_ids={sorted(set(unknown_result_case_ids))}, "
                f"missing_result_case_ids={missing_result_case_ids}"
            ),
            "total_cases": int(total_cases),
            "ok_cases": int(len(ok_cases)),
            "failed_cases": int(len(failed_cases)),
        }

    combined = total_original_cnot_added - total_added_cnot_by_candidate
    weighted_combined = weighted_total_original_cnot_added - weighted_total_added_cnot_by_candidate
    total_weight = float(sum(case_weight_by_case_id.values()))

    return {
        "combined_score": float(weighted_combined),
        "validity": float(1.0 if len(failed_cases) == 0 else 0.5 if ok_cases else 0.0),
        "eval_time": float(eval_time),
        "total_cases": int(total_cases),
        "ok_cases": int(len(ok_cases)),
        "failed_cases": int(len(failed_cases)),
        "total_swaps": int(total_swaps),
        "total_depth": int(total_depth),
        "total_runtime_ms": int(total_runtime_ms),
        "total_original_cnot_added": int(total_original_cnot_added),
        "total_added_cnot_by_candidate": int(total_added_cnot_by_candidate),
        "total_cnot_delta": int(combined),
        "total_weight": float(total_weight),
        "weighted_total_original_cnot_added": float(weighted_total_original_cnot_added),
        "weighted_total_added_cnot_by_candidate": float(weighted_total_added_cnot_by_candidate),
        "weighted_total_cnot_delta": float(weighted_combined),
    }


def _additional_cnot_added_from_swap_count(raw_swap_count: Any) -> int:
    """Convert swap_count into equivalent additional CNOT gates."""
    try:
        swap_count = max(0, int(raw_swap_count))
    except (TypeError, ValueError):
        swap_count = 0
    # Each inserted SWAP is equivalent to 3 added CNOT gates.
    return 3 * swap_count


def _coerce_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_rust_results_for_storage(
    results: Dict[str, Any],
    suite_metadata: Dict[str, SuiteCaseSpec],
) -> Dict[str, Any]:
    """Keep compact per-case metrics."""
    compact_cases = []
    for case in results.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        case_spec = suite_metadata.get(case_id) if isinstance(case_id, str) else None
        ok_raw = case.get("ok")
        ok = ok_raw if isinstance(ok_raw, bool) else None
        original_cnot_added = (
            int(case_spec.original_cnot_added) if case_spec is not None else None
        )
        if ok is True:
            additional_cnot_added: int | None = _additional_cnot_added_from_swap_count(
                case.get("swap_count", 0)
            )
        else:
            additional_cnot_added = original_cnot_added
        raw_error = case.get("error")
        compact_cases.append(
            {
                "id": case_id if isinstance(case_id, str) else None,
                "ok": ok,
                "topology": case_spec.topology_path.stem if case_spec is not None else None,
                "original_cnot_added": original_cnot_added,
                "weight": (
                    float(case_spec.weight) if case_spec is not None else None
                ),
                "additional_cnot_added": additional_cnot_added,
                "depth": case.get("depth"),
                "twoq_count": case.get("twoq_count"),
                "time_ms": case.get("time_ms"),
                "error": str(raw_error) if raw_error is not None else None,
            }
        )
    return {"cases": compact_cases}


def _compact_rust_results(
    results: Dict[str, Any],
    suite_metadata: Dict[str, SuiteCaseSpec],
) -> Dict[str, Any]:
    """Keep only prompt-friendly topology aggregates plus top wins/losses."""
    topology_summary: dict[str, dict[str, Any]] = {}
    ranked_cases: list[dict[str, Any]] = []

    for case in results.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        case_spec = suite_metadata.get(case_id) if isinstance(case_id, str) else None
        ok_raw = case.get("ok")
        ok = ok_raw if isinstance(ok_raw, bool) else None
        original_cnot_added = (
            int(case_spec.original_cnot_added) if case_spec is not None else None
        )
        if ok is True:
            additional_cnot_added: int | None = _additional_cnot_added_from_swap_count(
                case.get("swap_count", 0)
            )
        else:
            # Mirror evaluator scoring behavior: failed or unknown-OK cases get no credit.
            additional_cnot_added = original_cnot_added
        raw_error = case.get("error")
        topology = case_spec.topology_path.stem if case_spec is not None else None
        weight = float(case_spec.weight) if case_spec is not None else None
        cnot_delta = None
        if original_cnot_added is not None and additional_cnot_added is not None:
            cnot_delta = original_cnot_added - additional_cnot_added
        weighted_delta = None
        if cnot_delta is not None and weight is not None:
            weighted_delta = float(cnot_delta) * weight

        topology_key = topology if isinstance(topology, str) and topology else "unknown"
        agg = topology_summary.setdefault(
            topology_key,
            {
                "topology": topology_key,
                "case_count": 0,
                "ok_case_count": 0,
                "failed_case_count": 0,
                "weighted_total_cnot_delta": 0.0,
                "total_cnot_delta": 0,
                "total_time_ms": 0,
            },
        )
        agg["case_count"] += 1
        if ok is True:
            agg["ok_case_count"] += 1
        elif ok is False:
            agg["failed_case_count"] += 1
        if weighted_delta is not None:
            agg["weighted_total_cnot_delta"] += weighted_delta
        if cnot_delta is not None:
            agg["total_cnot_delta"] += cnot_delta
        time_ms = _coerce_float_or_none(case.get("time_ms"))
        if time_ms is not None:
            agg["total_time_ms"] += int(round(time_ms))

        ranked_cases.append(
            {
                "id": case_id if isinstance(case_id, str) else None,
                "topology": topology,
                "ok": ok,
                "weighted_cnot_delta": (
                    round(weighted_delta, 3) if weighted_delta is not None else None
                ),
                "cnot_delta": cnot_delta,
                "time_ms": int(round(time_ms)) if time_ms is not None else None,
                "error": str(raw_error) if raw_error is not None else None,
            }
        )

    wins = [
        case
        for case in ranked_cases
        if isinstance(case.get("weighted_cnot_delta"), float)
        and case["weighted_cnot_delta"] > 0
    ]
    wins.sort(key=lambda case: case["weighted_cnot_delta"], reverse=True)

    losses = [
        case
        for case in ranked_cases
        if isinstance(case.get("weighted_cnot_delta"), float)
        and case["weighted_cnot_delta"] < 0
    ]
    losses.sort(key=lambda case: case["weighted_cnot_delta"])

    return {
        "topology_summary": [
            {
                **topology_summary[name],
                "weighted_total_cnot_delta": round(
                    topology_summary[name]["weighted_total_cnot_delta"], 3
                ),
            }
            for name in sorted(topology_summary)
        ],
        "top_wins": wins[:PROMPT_SUMMARY_TOP_K],
        "top_losses": losses[:PROMPT_SUMMARY_TOP_K],
    }


def _write_results_with_metrics(
    out_path: Path,
    scored_metrics: Dict[str, Any],
) -> None:
    """Persist the evaluator payload exactly as requested by the caller."""
    payload: Dict[str, Any] = (
        dict(scored_metrics) if isinstance(scored_metrics, dict) else {}
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate(
    program_path: str,
    save_results: bool = False,
    results_dir: str | None = None,
    run_equivalence_validation: bool = False,
    clean_slot_workspaces_before_run: bool = False,
    trial_mode: str = "fast",
    layout_trials: int | None = None,
    routing_trials: int | None = None,
) -> Dict[str, Any]:
    start = time.time()
    effective_total_budget_seconds = _resolve_total_budget_seconds()
    slot_lock_timeout_seconds = _resolve_slot_lock_timeout_seconds(effective_total_budget_seconds)
    out_path: Path | None = None
    should_cleanup_out_path = False
    slot: SlotHandle | None = None
    slot_wait_time = 0.0
    build_time = 0.0
    router_time = 0.0
    timeout_stage: str = "slot"
    suite_path = Path(os.environ.get("QUBIT_ROUTING_SUITE_PATH", str(DEFAULT_SUITE_PATH)))
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    resolved_trial_mode, resolved_layout_trials, resolved_routing_trials = _resolve_trial_counts(
        trial_mode,
        layout_trials,
        routing_trials,
    )

    try:
        if clean_slot_workspaces_before_run:
            clean_slot_workspaces()

        slot, slot_wait_time = acquire_slot(
            start,
            effective_total_budget_seconds,
            slot_lock_timeout_seconds,
        )
        prepare_slot_workspace(slot)
        slot_candidate_rs = slot.rust_dir / "router_core" / "src" / "candidate.rs"
        slot_router_bin = slot.target_dir / "release" / "router_cli"

        suite_metadata = _load_suite_metadata(suite_path)
        original_cnot_added_by_case_id = _load_original_cnot_added(suite_metadata)
        case_weight_by_case_id = _load_case_weights(suite_metadata)

        candidate_code = _extract_candidate_code(program_path)
        candidate_code = _inject_swap_selection_context_compat_shim(candidate_code)
        slot_candidate_rs.write_text(candidate_code, encoding="utf-8")

        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(slot.target_dir)
        env["QUBIT_ROUTING_LAYOUT_TRIALS"] = str(resolved_layout_trials)
        env["QUBIT_ROUTING_ROUTING_TRIALS"] = str(resolved_routing_trials)
        pyo3_python = _select_pyo3_python(env)
        if pyo3_python is None:
            raise RuntimeError(
                "No active supported environment found. Activate conda env or set "
                "UV_PROJECT_ENVIRONMENT (or explicitly set PYO3_PYTHON)."
            )
        env["PYO3_PYTHON"] = pyo3_python
        env["ROUTER_PYTHON"] = pyo3_python
        python_home = _python_home_for_python(pyo3_python)
        if python_home:
            # Critical for embedded Python under uv CPython builds.
            env["PYTHONHOME"] = python_home

        runtime_paths: list[str] = []
        for path in _stdlib_paths_for_python(pyo3_python) + _site_packages_for_python(pyo3_python):
            if path and path not in runtime_paths:
                runtime_paths.append(path)

        if runtime_paths:
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                os.pathsep.join(runtime_paths + [existing_pythonpath])
                if existing_pythonpath
                else os.pathsep.join(runtime_paths)
            )

        build_timeout = _remaining_budget(start, effective_total_budget_seconds)
        build_start = time.time()
        timeout_stage = "build"
        try:
            _run(
                ["cargo", "build", "--release", "-p", "router_cli"],
                cwd=slot.rust_dir,
                timeout_seconds=build_timeout,
                env=env,
            )
        finally:
            build_time = time.time() - build_start

        if save_results:
            resolved_results_dir = _resolve_results_dir(results_dir)
            resolved_results_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = (
                resolved_results_dir
                / f"router_results_{stamp}_{int(time.time() * 1000)}.json"
            )
        else:
            fd, tmp_path = tempfile.mkstemp(prefix="router_results_", suffix=".json")
            os.close(fd)
            out_path = Path(tmp_path)
            should_cleanup_out_path = True

        run_timeout = _remaining_budget(start, effective_total_budget_seconds)
        router_start = time.time()
        timeout_stage = "router"
        try:
            _run(
                [
                    str(slot_router_bin),
                    "--suite",
                    str(suite_path),
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                timeout_seconds=run_timeout,
                env=env,
            )
        finally:
            router_time = time.time() - router_start

        results = json.loads(out_path.read_text(encoding="utf-8"))
        if run_equivalence_validation:
            _apply_equivalence_validation(results, suite_metadata)

        eval_time = time.time() - start
        scored = _score_results(
            results,
            eval_time,
            original_cnot_added_by_case_id,
            case_weight_by_case_id,
        )
        scored["results_path"] = str(out_path) if save_results else None
        scored["slot_id"] = slot.slot_id
        scored["slot_wait_time"] = float(slot_wait_time)
        scored["build_time"] = float(build_time)
        scored["router_time"] = float(router_time)
        scored["trial_mode"] = resolved_trial_mode
        scored["layout_trials"] = int(resolved_layout_trials)
        scored["routing_trials"] = int(resolved_routing_trials)
        # Keep only a prompt-friendly summary in the returned metrics.
        scored["rust_results"] = _compact_rust_results(results, suite_metadata)
        if save_results:
            saved_scored = dict(scored)
            saved_scored["rust_results"] = _compact_rust_results_for_storage(
                results,
                suite_metadata,
            )
            _write_results_with_metrics(out_path, saved_scored)
        return scored

    except subprocess.TimeoutExpired as exc:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(time.time() - start),
            "error": f"timeout[{timeout_stage}]: {exc}",
            "results_path": str(out_path) if save_results and out_path is not None else None,
            "slot_id": slot.slot_id if slot is not None else None,
            "slot_wait_time": float(slot_wait_time),
            "build_time": float(build_time),
            "router_time": float(router_time),
            "trial_mode": resolved_trial_mode,
            "layout_trials": int(resolved_layout_trials),
            "routing_trials": int(resolved_routing_trials),
        }
    except TimeoutError as exc:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(time.time() - start),
            "error": f"slot_timeout: {exc}",
            "results_path": str(out_path) if save_results and out_path is not None else None,
            "slot_id": None,
            "slot_wait_time": float(slot_wait_time),
            "build_time": float(build_time),
            "router_time": float(router_time),
            "trial_mode": resolved_trial_mode,
            "layout_trials": int(resolved_layout_trials),
            "routing_trials": int(resolved_routing_trials),
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "eval_time": float(time.time() - start),
            "error": f"{type(exc).__name__}: {exc}",
            "results_path": str(out_path) if save_results and out_path is not None else None,
            "slot_id": slot.slot_id if slot is not None else None,
            "slot_wait_time": float(slot_wait_time),
            "build_time": float(build_time),
            "router_time": float(router_time),
            "trial_mode": resolved_trial_mode,
            "layout_trials": int(resolved_layout_trials),
            "routing_trials": int(resolved_routing_trials),
        }
    finally:
        if should_cleanup_out_path and out_path is not None:
            out_path.unlink(missing_ok=True)
        release_slot(slot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "program_path",
        nargs="?",
        default=str(ROOT / "init_program.rs"),
        help="Path to candidate Rust code (default: init_program.rs).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help=(
            "Persist router_results_<timestamp>_<ms>.json to --results-dir "
            "(or QUBIT_ROUTING_RESULTS_DIR if set)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Directory for saved router_results_*.json files when --save-results is set. "
            "Default: checkpoints/qubit_routing_results/results."
        ),
    )
    parser.add_argument(
        "--equivalence-check",
        action="store_true",
        help=(
            "Run optional topology/front-layer equivalence validation on each successful case. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--clean",
        dest="clean_slot_workspaces",
        action="store_true",
        help=(
            "Delete all evaluator slot workspaces in the active slot namespace before running. "
            "Useful when slot source trees become stale after refactors."
        ),
    )
    parser.add_argument(
        "--trial-mode",
        choices=["fast", "full"],
        default="fast",
        help=(
            "Routing trial profile. 'fast' uses 4x4 trials (default). "
            "'full' uses 20x20 trials for LightSABRE-style comparison."
        ),
    )
    parser.add_argument(
        "--layout-trials",
        type=int,
        default=None,
        help=(
            "Override layout trial count. If omitted, uses profile default "
            "(fast=4, full=20)."
        ),
    )
    parser.add_argument(
        "--routing-trials",
        type=int,
        default=None,
        help=(
            "Override routing trial count. If omitted, uses profile default "
            "(fast=4, full=20)."
        ),
    )
    args = parser.parse_args()

    result = evaluate(
        args.program_path,
        save_results=args.save_results,
        results_dir=args.results_dir,
        run_equivalence_validation=args.equivalence_check,
        clean_slot_workspaces_before_run=args.clean_slot_workspaces,
        trial_mode=args.trial_mode,
        layout_trials=args.layout_trials,
        routing_trials=args.routing_trials,
    )
    print(json.dumps(result, indent=2))
