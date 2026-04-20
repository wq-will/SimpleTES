# utils_ae.py — ZNAA machine model, circuit I/O, and run_code helpers (split from init_program).
# Evaluator loads this file into ``sys.modules['utils_ae']`` before executing the candidate so
# ``import utils_ae`` works when the candidate is the only Python source in the working directory.
from __future__ import annotations

import bisect
import copy
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import numpy as np

# Sentinel for empty SLM / AOD grid cells (qubit ids may be 0).
_EMPTY_QUBIT_CELL = -1


class Solver:
    """
    Compiler orchestrator that converts a logical circuit into staged machine operations.

    Pipeline:
    1) ``scheduler.schedule``: build logical stages.
    2) ``reuse_analyzer.analyze``: compute stage-aligned reuse hints.
    3) ``placer.place``: return placements grouped by 2Q stage
       (``list[list[list[(z,x,y)]]]``), while exposing one initial placement.
    4) pairwise ``router.route`` on adjacent placement pairs.
    5) stitch route ops + gate ops into executable ``plan`` blocks.

    Output contract (critical for evaluator):
    - candidate ``run_code()`` only supplies component classes; evaluator calls ``solve`` and builds ``plan`` / ``machine``,
    - each non-initial block is anchored at one 2Q stage and may contain:
      route-to-first-placement, 2Q gate, interleaved 1Q gates, and extra intra-group routes,
    - ``last_solve_intermediates`` includes grouped placement/router diagnostics.
    """

    @staticmethod
    def _prev_2q_stage_index(stages: list[ZNAAStage], si: int) -> int | None:
        # Walk backward to find the nearest non-empty 2Q stage.
        for j in range(si - 1, -1, -1):
            if stages[j].stage_type == "2q" and stages[j].gates:
                return j
        return None

    @staticmethod
    def _reuse_for_plan_block(
        stages: list[ZNAAStage], si: int, reuse_info: list[list[int]]
    ) -> list[int]:
        """Per-stage display: 2Q row = outgoing reuse; 1Q row reuses previous 2Q row."""
        st = stages[si]
        if st.stage_type == "1q":
            pj = Solver._prev_2q_stage_index(stages, si)
            if pj is None:
                return []
            row = reuse_info[pj] if pj < len(reuse_info) else []
            return list(row)
        row = reuse_info[si] if si < len(reuse_info) else []
        return list(row)

    @staticmethod
    def _serialize_placement_snap(snap: list[tuple]) -> list[list[int]]:
        # Keep placements JSON-friendly while preserving (zone,row,col) ordering.
        return [list(p) for p in snap]

    @staticmethod
    def _stage_gates_for_plan(stage: ZNAAStage) -> list[dict[str, Any]]:
        # Canonical gate payload for evaluator-side gate/frontier checks.
        return [
            {"gate_type": g.gate_type, "gate_name": g.gate_name, "qubits": list(g.qubits)}
            for g in stage.gates
        ]

    @staticmethod
    def _serialize_solve_block_for_output(block: dict[str, Any]) -> dict[str, Any]:
        """JSON-friendly copy of one solve block (``operations`` as strings)."""
        d: dict[str, Any] = {
            "stage": block["stage"],
            "reuse_info": list(block.get("reuse_info", [])),
            "placements": block["placements"],
            "operations": [str(op) for op in block["operations"]],
        }
        if "gates" in block:
            d["gates"] = list(block["gates"])
        if "placement_before_route" in block:
            d["placement_before_route"] = block["placement_before_route"]
        if "route_edge_index" in block:
            d["route_edge_index"] = block["route_edge_index"]
        return d

    def __init__(
        self,
        name: str = "Solver",
        version: str = "0.0",
        *,
        config_factory: Callable[[], ZNAAConfig],
        scheduler: AbstractScheduler = None,
        reuse_analyzer: AbstractReuseAnalyzer = None,
        placer: AbstractPlacer = None,
        router: AbstractRouter = None,
    ):
        self.name = name
        self.version = version
        self.config_factory = config_factory
        self.config = config_factory()
        self.scheduler = scheduler
        self.reuse_analyzer = reuse_analyzer
        self.placer = placer
        self.router = router
        self.last_solve_intermediates: dict[str, Any] | None = None

    def _route_pairwise(
        self,
        src: list[tuple[int, int, int]],
        dst: list[tuple[int, int, int]],
    ) -> list[ZNAAOperation]:
        # Input format: src/dst are full placements for all logical qubits.
        # Output format: a single transition segment operations list.
        routed = self.router.route([src, dst])
        if not isinstance(routed, list) or len(routed) != 1 or not isinstance(routed[0], list):
            raise RuntimeError(
                "Solver.solve router pairwise contract violation: expected route([src,dst]) -> [ops]."
            )
        return list(routed[0])

    @staticmethod
    def _build_1q_stage_ops(stage: ZNAAStage) -> list[ZNAAOperation]:
        # Input format: one 1Q stage; output format: ordered parallel Operation_1QGate layers.
        gate_ops: list[ZNAAOperation] = []
        per_qubit: dict[int, int] = {}
        layers: list[list[ZNAAGate]] = []
        for gate in stage.gates:
            if not gate.qubits:
                continue
            q_log = gate.qubits[0]
            k = per_qubit.get(q_log, 0)
            per_qubit[q_log] = k + 1
            while len(layers) <= k:
                layers.append([])
            layers[k].append(gate)
        for layer in layers:
            if not layer:
                continue
            qubit_ids = [g.qubits[0] for g in layer]
            gate_types = [g.gate_name for g in layer]
            op = Operation_1QGate(gate_types[0], qubit_ids)
            op.gate_types = gate_types
            gate_ops.append(op)
        return gate_ops


    def solve(self, circuit: ZNAACircuit) -> list[dict[str, Any]]:
        """
        Build a full execution plan from a logical circuit.

        Returns:
        - ``plan``: list[dict], starting with one ``initial_placement`` block, then one block
          per 2Q stage group.

        Side output:
        - ``self.last_solve_intermediates`` stores grouped placements and pairwise routes.
        """
        self.last_solve_intermediates = {}

        qubits_present = set(circuit.qubits)
        n_q = max(qubits_present) + 1 if qubits_present else 0
        t0 = time.perf_counter()
        # Stage scheduling is the logical dependency backbone for all later steps.
        stages = self.scheduler.schedule(circuit)
        t_scheduler = time.perf_counter() - t0
        t1 = time.perf_counter()
        # Reuse hints are indexed exactly by stage position.
        reuse_info = self.reuse_analyzer.analyze(stages)
        t_reuse_analyzer = time.perf_counter() - t1
        t2 = time.perf_counter()
        # Placer returns grouped placements for each 2Q stage.
        placements_by_2q_stage = self.placer.place(stages, reuse_info)
        initial_placement = getattr(self.placer, "initial_placement", None)
        t_placer = time.perf_counter() - t2
        # Router is called pairwise while building plan; no global route call here.
        routes: list[list[ZNAAOperation]] = []
        router_checks: list[dict[str, Any]] = []
        t3 = time.perf_counter()
        two_q_stage_indices = [
            i for i, st in enumerate(stages) if st.stage_type == "2q" and st.gates
        ]
        self._validate_pipeline_outputs(
            stages=stages,
            reuse_info=reuse_info,
            initial_placement=initial_placement,
            placements_by_2q_stage=placements_by_2q_stage,
            two_q_stage_indices=two_q_stage_indices,
        )

        zone_chr = {0: "s", 1: "e"}
        plan: list[dict[str, Any]] = []
        first_1q_stage_index = next(
            (
                i
                for i, st in enumerate(stages)
                if st.stage_type == "1q" and st.gates
            ),
            None,
        )
        initial_1q_ops: list[ZNAAOperation] = []
        initial_1q_gates: list[dict[str, Any]] = []
        if first_1q_stage_index is not None:
            first_1q_stage = stages[first_1q_stage_index]
            initial_1q_ops = self._build_1q_stage_ops(first_1q_stage)
            initial_1q_gates = self._stage_gates_for_plan(first_1q_stage)
        map_ops: list[ZNAAOperation] = []
        for i in range(n_q):
            z, x, y = initial_placement[i]
            zs = zone_chr.get(z, "s")
            map_ops.append(Operation_Map(qubit_id=i, coord=(x, y), zone=zs))
        map_ops.extend(initial_1q_ops)
        plan.append(
            {
                "stage": "initial_placement",
                "reuse_info": [],
                "placements": self._serialize_placement_snap(initial_placement),
                "gates": initial_1q_gates,
                "operations": map_ops,
            }
        )

        current_placement = list(initial_placement)
        for gi, si in enumerate(two_q_stage_indices):
            stage = stages[si]
            group = placements_by_2q_stage[gi]
            merged: list[ZNAAOperation] = []
            block_gates: list[dict[str, Any]] = []
            block_route_checks: list[dict[str, Any]] = []
            prefix_ops: list[ZNAAOperation] = []
            prefix_gates: list[dict[str, Any]] = []

            # Keep logical frontier order: 1Q stages before the first 2Q stage
            # must be emitted before the first 2Q block content.
            if gi == 0:
                for sj in range(0, si):
                    st_j = stages[sj]
                    if st_j.stage_type != "1q" or not st_j.gates:
                        continue
                    if first_1q_stage_index is not None and sj == first_1q_stage_index:
                        continue
                    prefix_ops.extend(self._build_1q_stage_ops(st_j))
                    prefix_gates.extend(self._stage_gates_for_plan(st_j))

            # Step i: route from current placement to the first placement in this 2Q group.
            first_target = group[0]
            first_start = len(merged)
            first_ops = self._route_pairwise(current_placement, first_target)
            merged.extend(first_ops)
            first_end = len(merged)
            routes.append(first_ops)
            first_route = {
                "two_q_stage_index": gi,
                "group_index": gi,
                "placement_edge_in_group": 0,
                "from_placement": self._serialize_placement_snap(current_placement),
                "to_placement": self._serialize_placement_snap(first_target),
                "operations": first_ops,
                "op_span": [first_start, first_end],
            }
            router_checks.append(first_route)
            block_route_checks.append(first_route)
            current_placement = first_target

            # Keep route ops contiguous at the beginning of merged ops so evaluator
            # can slice each route segment by route_op_counts deterministically.
            if prefix_ops:
                merged.extend(prefix_ops)
                block_gates.extend(prefix_gates)

            # Step ii: execute this 2Q stage.
            gate_name = stage.gates[0].gate_name if stage.gates else "CZ"
            merged.append(Operation_2QGate(gate_name))
            block_gates.extend(self._stage_gates_for_plan(stage))

            # Step iii: execute all 1Q stages until next 2Q stage.
            next_two_q_si = (
                two_q_stage_indices[gi + 1] if gi + 1 < len(two_q_stage_indices) else len(stages)
            )
            for sj in range(si + 1, next_two_q_si):
                st_j = stages[sj]
                if st_j.stage_type != "1q" or not st_j.gates:
                    continue
                merged.extend(self._build_1q_stage_ops(st_j))
                block_gates.extend(self._stage_gates_for_plan(st_j))

            # Step iv: follow the remaining placements in this 2Q group.
            for pi in range(1, len(group)):
                target = group[pi]
                seg_start = len(merged)
                seg_ops = self._route_pairwise(current_placement, target)
                merged.extend(seg_ops)
                seg_end = len(merged)
                routes.append(seg_ops)
                route_item = {
                    "two_q_stage_index": gi,
                    "group_index": gi,
                    "placement_edge_in_group": pi,
                    "from_placement": self._serialize_placement_snap(current_placement),
                    "to_placement": self._serialize_placement_snap(target),
                    "operations": seg_ops,
                    "op_span": [seg_start, seg_end],
                }
                router_checks.append(route_item)
                block_route_checks.append(route_item)
                current_placement = target

            blk: dict[str, Any] = {
                "stage": f"{stage.stage_type}@{gi}",
                "reuse_info": self._reuse_for_plan_block(stages, si, reuse_info),
                "placements": self._serialize_placement_snap(current_placement),
                "gates": block_gates,
                "two_q_gates": self._stage_gates_for_plan(stage),
                "operations": merged,
                "placement_before_route": self._serialize_placement_snap(first_route["from_placement"]),
                "route_edge_index": gi,
                "placements_group": [self._serialize_placement_snap(p) for p in group],
                "router_checks": [
                    {
                        "placement_edge_in_group": int(item["placement_edge_in_group"]),
                        "from_placement": item["from_placement"],
                        "to_placement": item["to_placement"],
                        "operations": [str(op) for op in item["operations"]],
                        "op_span": list(item.get("op_span") or []),
                    }
                    for item in block_route_checks
                ],
                "route_op_counts": [len(item["operations"]) for item in block_route_checks],
                "route_targets": [item["to_placement"] for item in block_route_checks],
                "two_q_stage_index": gi,
            }
            plan.append(blk)
        t_router = time.perf_counter() - t3
        # Keep rich intermediates for failure diagnostics and auxiliary checks.
        self.last_solve_intermediates = {
            "stages": stages,
            "reuse_info": [list(row) for row in reuse_info],
            "initial_placement": list(initial_placement),
            "placements_by_2q_stage": placements_by_2q_stage,
            "two_q_stage_indices": list(two_q_stage_indices),
            "routes": routes,
            "router_checks": [
                {
                    "two_q_stage_index": int(item["two_q_stage_index"]),
                    "group_index": int(item["group_index"]),
                    "placement_edge_in_group": int(item["placement_edge_in_group"]),
                    "from_placement": item["from_placement"],
                    "to_placement": item["to_placement"],
                    "operations": [str(op) for op in item["operations"]],
                }
                for item in router_checks
            ],
            "component_times": {
                "scheduler": float(t_scheduler),
                "reuseanalyzer": float(t_reuse_analyzer),
                "placer": float(t_placer),
                "router": float(t_router),
            },
        }
        return plan

    @staticmethod
    def _validate_placement_snapshot(
        snap: list[tuple[Any, Any, Any]], expected_n_q: int, stage_index: int
    ) -> None:
        # Snapshot contract:
        # - full-width row for all logical qubits
        # - tuple format (zone, row, col)
        # - zone in {0,1}, coordinates integer-valued
        if len(snap) != expected_n_q:
            raise RuntimeError(
                "Solver.solve placement width mismatch: "
                f"stage={stage_index}, len(placement)={len(snap)}, expected={expected_n_q}"
            )
        for qi, cell in enumerate(snap):
            if not isinstance(cell, tuple) or len(cell) != 3:
                raise RuntimeError(
                    "Solver.solve invalid placement tuple: "
                    f"stage={stage_index}, qubit={qi}, value={cell!r}"
                )
            z, x, y = cell
            if int(z) not in (0, 1):
                raise RuntimeError(
                    "Solver.solve invalid placement zone: "
                    f"stage={stage_index}, qubit={qi}, zone={z!r}"
                )
            if int(x) != x or int(y) != y:
                raise RuntimeError(
                    "Solver.solve non-integer placement coordinate: "
                    f"stage={stage_index}, qubit={qi}, coord=({x!r},{y!r})"
                )

    @staticmethod
    def _validate_pipeline_outputs(
        stages: list["ZNAAStage"],
        reuse_info: list[list[int]],
        initial_placement: list[tuple[Any, Any, Any]],
        placements_by_2q_stage: list[list[list[tuple[Any, Any, Any]]]],
        two_q_stage_indices: list[int],
    ) -> None:
        # Pipeline-level contract used by downstream evaluator logic.
        n_stages = len(stages)
        if len(reuse_info) != n_stages:
            raise RuntimeError(
                f"Solver.solve length mismatch: len(reuse_info)={len(reuse_info)} != len(stages)={n_stages}"
            )
        if not isinstance(initial_placement, list):
            raise RuntimeError(
                "Solver.solve invalid initial placement: placer must expose list[(z,x,y)]."
            )
        if len(placements_by_2q_stage) != len(two_q_stage_indices):
            raise RuntimeError(
                "Solver.solve length mismatch: "
                f"len(placements_by_2q_stage)={len(placements_by_2q_stage)} "
                f"!= number_of_2q_stages={len(two_q_stage_indices)}"
            )
        expected_n_q = len(initial_placement)
        for qi, cell in enumerate(initial_placement):
            z = int(cell[0])
            if z != 0:
                raise RuntimeError(
                    "Solver.solve invalid initial placement zone: "
                    f"qubit={qi}, zone={z}, expected=0(storage)"
                )
        Solver._validate_placement_snapshot(initial_placement, expected_n_q, 0)
        for gi, group in enumerate(placements_by_2q_stage):
            if not isinstance(group, list) or not group:
                raise RuntimeError(
                    f"Solver.solve invalid grouped placement: 2q-group={gi} must contain at least one placement."
                )
            for pi, snap in enumerate(group):
                Solver._validate_placement_snapshot(
                    snap,
                    expected_n_q,
                    1 + gi * 1000 + pi,
                )


class AbstractScheduler:
    """
    Interface contract for stage schedulers.

    Expected input:
    - ``ZNAACircuit`` with ordered logical gates.
    Expected output:
    - ``list[ZNAAStage]`` where each stage is type ``1q`` or ``2q``.
    """
    def __init__(
        self,
        name: str = "Abstract Scheduler",
        version: str = "0.0",
        *,
        config_factory: Callable[[], ZNAAConfig],
    ):
        self.name = name
        self.version = version
        self.config_factory = config_factory
        self.config = config_factory()

    @staticmethod
    def schedule(self, circuit: ZNAACircuit) -> list[ZNAAStage]:
        pass


class AbstractReuseAnalyzer:
    """
    Interface contract for reuse analyzers.

    Expected input:
    - scheduler output ``list[ZNAAStage]``.
    Expected output:
    - stage-aligned ``list[list[int]]`` describing reusable logical qubits.
    """
    def __init__(
        self,
        name: str = "Abstract Reuse Analyzer",
        version: str = "0.0",
        *,
        config_factory: Callable[[], ZNAAConfig],
    ):
        self.name = name
        self.version = version
        self.config_factory = config_factory
        self.config = config_factory()

    @staticmethod
    def analyze(self, stage: list[ZNAAStage]) -> list[list[int]]:
        pass


class AbstractPlacer:
    """
    Interface contract for placers.

    Expected input:
    - ``stages`` and stage-aligned ``reuse_info``.
    Expected output:
    - grouped placements for each non-empty 2Q stage:
            ``list[list[list[(zone,row,col)]]]``.
    - implementation should also expose ``initial_placement`` as one full
      initial placement list.
        - ``calc_physical_coordinate(...)`` converts one logical placement tuple
            into physical ``(x,y)``.
    """
    def __init__(
        self,
        name: str = "Abstract Placer",
        version: str = "0.0",
        *,
        config_factory: Callable[[], ZNAAConfig],
    ):
        self.name = name
        self.version = version
        self.config_factory = config_factory
        self.config = config_factory()

    def calc_physical_coordinate(
        self,
        zone: int | str,
        row: int,
        col: int,
    ) -> tuple[float, float]:
        """
        Convert one logical placement ``(zone,row,col)`` into physical ``(x,y)``.

        Supported zone labels:
        - ``0`` / ``storage``
        - ``1`` / ``entangling``
        """
        logical_row = int(row)
        logical_col = int(col)
        zone_key = zone.lower() if isinstance(zone, str) else int(zone)
        geometry = self.config.build_slm_geometry()
        dict_slm = geometry["dict_slm"]

        if zone_key in (0, "0", "s", "storage"):
            slm_id = int(geometry["storage_zone"][0])
            mapped_row = int(dict_slm[slm_id]["n_r"]) - 1 - logical_row
            mapped_col = logical_col
        elif zone_key in (1, "1", "e", "entangling"):
            ent_ids = list(geometry["entangle_slm_ids_flat"])
            slm_id = int(ent_ids[logical_col % len(ent_ids)])
            mapped_row = int(dict_slm[slm_id]["n_r"]) - 1 - logical_row
            mapped_col = logical_col // len(ent_ids)
        else:
            raise ValueError(f"Unsupported placement zone: {zone!r}")

        slm = dict_slm[slm_id]
        if not (0 <= mapped_row < int(slm["n_r"])):
            raise ValueError(f"Placement row out of range: {(zone, row, col)!r}")
        if not (0 <= mapped_col < int(slm["n_c"])):
            raise ValueError(f"Placement col out of range: {(zone, row, col)!r}")

        origin_x, origin_y = slm["location"]
        step_x, step_y = slm["site_seperation"]
        return (
            float(origin_x) + mapped_row * float(step_x),
            float(origin_y) + mapped_col * float(step_y),
        )

    @staticmethod
    def place(
        self,
        stages: list[ZNAAStage],
        reuse_info: list[list[int]],
    ) -> list[list[list[tuple]]]:
        pass


class AbstractRouter:
    """
    Interface contract for routers.

    Expected input:
    - a placement list where each item is one full qubit placement.
    - solver may call this interface with only two placements (adjacent pair).
    Expected output:
    - one route segment per adjacent placement pair.
    - ``calc_operations_time(...)`` replays one operation list on a fresh
      machine and returns the final elapsed time.
    """
    def __init__(
        self,
        name: str = "Abstract Router",
        version: str = "0.0",
        *,
        config_factory: Callable[[], ZNAAConfig],
    ):
        self.name = name
        self.version = version
        self.config_factory = config_factory
        self.config = config_factory()

    def calc_operations_time(
        self,
        operations_list: list[ZNAAOperation],
        initial_placement: Optional[list[tuple[int | str, int, int]]] = None,
    ) -> float:
        """
        Replay one operation list on a fresh machine and return the final time.

        If ``initial_placement`` is provided, the placement index is treated as the
        logical qubit id and one ``Operation_Map`` is emitted before replay.
        """
        machine = ZNAAMachine(
            hardware_config=self.config_factory,
        )
        if initial_placement is not None:
            for qubit_id, (zone, row, col) in enumerate(initial_placement):
                zone_key = zone.lower() if isinstance(zone, str) else int(zone)
                if zone_key in (0, "0", "s", "storage"):
                    zone_name = "storage"
                elif zone_key in (1, "1", "e", "entangling"):
                    zone_name = "entangling"
                else:
                    raise ValueError(f"Unsupported placement zone: {zone!r}")
                machine.append_operation(
                    Operation_Map(
                        qubit_id=int(qubit_id),
                        coord=(int(row), int(col)),
                        zone=zone_name,
                    )
                )
        for operation in operations_list:
            machine.append_operation(operation)
        return float(machine.time)

    @staticmethod
    def route(
        self,
        placements: list[list[tuple]]
    ) -> list[list[ZNAAOperation]]:
        pass


class ZNAACoord:
    """Integer lattice site (x,y) inside one logical zone."""

    def __init__(self, zone: str, x: int, y: int):
        self.x = x
        self.y = y
        z = zone.lower()
        if z in ("s", "storage"):
            self.zone = "storage"
        elif z in ("e", "entangling"):
            self.zone = "entangling"
        elif z in ("r", "readout"):
            self.zone = "readout"
        elif z in ("a", "aod"):
            self.zone = "aod"
        else:
            raise ZNAAMachineError(f"Invalid zone: {zone}.")


    def __str__(self):
        return "{" + f"{self.zone[0].upper()}({self.x},{self.y})" + "}"

    def __repr__(self):
        return self.__str__()

    def row(self):
        return ZNAARow(self.zone, self.x)

    def column(self):
        return ZNAAColumn(self.zone, self.y)

    def __eq__(self, other):
        if not isinstance(other, ZNAACoord):
            return False
        return self.zone == other.zone and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(("coord", self.zone, self.x, self.y))


class ZNAARow:
    """Continuous row index inside a zone (used by AOD Open / Move)."""

    def __init__(self, zone: str, r: float):
        self.r = float(r)
        z = zone.lower()
        if z in ("s", "storage"):
            self.zone = "storage"
        elif z in ("e", "entangling"):
            self.zone = "entangling"
        elif z in ("r", "readout"):
            self.zone = "readout"
        else:
            raise ZNAAMachineError(f"Invalid zone: {zone}.")

    def __str__(self):
        return f"{self.zone[0].upper()}(r{self.r})"

    def __eq__(self, other):
        if not isinstance(other, ZNAARow):
            return False
        return self.zone == other.zone and self.r == other.r

    def __hash__(self):
        return hash(("row", self.zone, self.r))


class ZNAAColumn:
    """Continuous column index inside a zone (used by AOD Open / Move)."""

    def __init__(self, zone: str, c: float):
        self.c = float(c)
        z = zone.lower()
        if z in ("s", "storage"):
            self.zone = "storage"
        elif z in ("e", "entangling"):
            self.zone = "entangling"
        elif z in ("r", "readout"):
            self.zone = "readout"
        else:
            raise ZNAAMachineError(f"Invalid zone: {zone}.")

    def __str__(self):
        return f"{self.zone[0].upper()}(c{self.c})"

    def __eq__(self, other):
        if not isinstance(other, ZNAAColumn):
            return False
        return self.zone == other.zone and self.c == other.c

    def __hash__(self):
        return hash(("column", self.zone, self.c))


class ZNAAGate:
    """One logical instruction: gate_type in {1q,2q}, name is uppercase for QASM-style logs."""

    def __init__(self, gate_type: str, gate_name: str, qubits: list[int]):
        if gate_type not in ("1q", "2q"):
            raise ValueError("Invalid gate type. Must be '1q' or '2q'.")
        self.gate_type = gate_type
        self.gate_name = str(gate_name).upper()
        self.qubits = list(qubits)
        self.uuid = uuid.uuid1()

    def __str__(self):
        qs = ", ".join("q" + str(x) for x in self.qubits)
        return f"{self.gate_name}({qs})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, ZNAAGate):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


class ZNAACircuit:
    """Mutable ordered gate list; used by schedulers that peel frontier gates (see get_frontier_gates)."""

    def __init__(self, gates: Optional[list[ZNAAGate]] = None):
        self.gates: list[ZNAAGate] = []
        self.qubits: set[int] = set()
        if gates:
            self.extend_gates(gates)

    def append_gate(self, gate: ZNAAGate):
        self.gates.append(gate)
        for q in gate.qubits:
            self.qubits.add(q)
        self.n_q = self.qubit_count()

    def extend_gates(self, gates: list[ZNAAGate]):
        for g in gates:
            self.append_gate(g)

    def is_empty(self) -> bool:
        return len(self.gates) == 0

    def qubit_count(self) -> int:
        return len(self.qubits)

    def __str__(self) -> str:
        # Avoid printing full circuits; keep it compact.
        return f"[ZNAACircuit] gates={len(self.gates)}, qubits={sorted(self.qubits)}"

    def __repr__(self) -> str:
        return f"ZNAACircuit(gates={len(self.gates)}, qubits={len(self.qubits)})"

    def remove_gate(self, gate: ZNAAGate):
        self.gates.remove(gate)
        for q in gate.qubits:
            if not any(q in g.qubits for g in self.gates):
                self.qubits.discard(q)
        self.n_q = self.qubit_count()

    def remove_frontier_gate(self, gate: Any):
        """
        Remove one "frontier" gate without computing a frontier layer.

        Rule (per your spec):
        - Traverse ``self.gates`` in order.
        - If we encounter a gate that shares at least one qubit with the target gate, and it is not
          the "same" gate (same ``gate_type`` and same involved qubits), raise ``ZNAAMachineError``.
        - If we encounter the "same" gate, remove it and break.

        Equality:
        - 1Q gates: ``gate.qubits`` must contain exactly 1 qubit; equality uses only (gate_type, qubit_id).
        - 2Q gates: ``gate.qubits`` must contain exactly 2 qubits; equality uses only (gate_type, pair),
          order-independent.
        """

        def _gate_type_and_qubits(g: Any) -> tuple[str, tuple[int, ...]]:
            # Accept both dict payload and object payload to match evaluator helpers.
            if isinstance(g, dict):
                gt = str(g.get("gate_type", "")).lower()
                qs = tuple(int(x) for x in (g.get("qubits") or []))
            else:
                gt = str(getattr(g, "gate_type", "")).lower()
                qs = tuple(int(x) for x in (getattr(g, "qubits", None) or []))
            if gt == "rydberg":
                gt = "2q"
            return gt, qs

        def _gate_qubit_set(gt: str, qs: tuple[int, ...]) -> set[int]:
            # Enforce arity assumptions used by frontier removal checks.
            if gt == "1q":
                if len(qs) != 1:
                    raise ZNAAMachineError(f"Invalid 1Q gate: expected 1 qubit, got {qs!r}.")
                return {qs[0]}
            if gt == "2q":
                if len(qs) != 2:
                    raise ZNAAMachineError(f"Invalid 2Q gate: expected 2 qubits, got {qs!r}.")
                return {qs[0], qs[1]}
            # Fallback: allow but still compute a set.
            return set(qs)

        def _same_gate(a: Any, b: Any) -> bool:
            # Frontier matching ignores gate names and uses type + logical qubits only.
            gta, qsa = _gate_type_and_qubits(a)
            gtb, qsb = _gate_type_and_qubits(b)
            if gta != gtb:
                return False
            if gta == "1q":
                return len(qsa) == 1 and len(qsb) == 1 and qsa[0] == qsb[0]
            if gta == "2q":
                return set(qsa) == set(qsb)
            return qsa == qsb

        target_gt, target_qs = _gate_type_and_qubits(gate)
        target_qset = _gate_qubit_set(target_gt, target_qs)

        for fg in list(self.gates):
            if _same_gate(fg, gate):
                self.remove_gate(fg)
                return
            fg_gt, fg_qs = _gate_type_and_qubits(fg)
            fg_qset = _gate_qubit_set(fg_gt, fg_qs)
            if not target_qset.isdisjoint(fg_qset) and fg_gt != target_gt:
                # Conflict only when a different gate *type* (1q vs 2q) still touches
                # at least one of the target qubits. Gates of the same type are allowed
                # to coexist in the remaining list.
                raise ZNAAMachineError(
                    "Gate is not removable as frontier gate: "
                    f"target=(gate_type={target_gt}, qubits={sorted(target_qset)}), "
                    f"conflicting_gate=(gate_type={fg_gt}, qubits={sorted(fg_qset)})"
                )

        # Target gate not found in the remaining circuit.
        raise ZNAAMachineError(
            "Gate not found in circuit for removal: "
            f"target=(gate_type={target_gt}, qubits={sorted(target_qset)})"
        )

    @staticmethod
    def from_file(file_path: str | Path) -> "ZNAACircuit":
        """Line format: ``<1q|2q> <NAME> <q0> [q1 ...]``; skip empty and ``#`` lines."""
        gates: list[ZNAAGate] = []
        path = Path(file_path).resolve()
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                gtype, gname = parts[0], parts[1]
                qubits = list(map(int, parts[2:]))
                gates.append(ZNAAGate(gtype, gname, qubits))
        return ZNAACircuit(gates)


class ZNAAStage:
    """A batch of gates scheduled together; stage_type is '1q' or '2q' (alias 'rydberg' → '2q')."""

    def __init__(self, stage_type: str, gates: Optional[list[ZNAAGate]] = None):
        st = stage_type.lower()
        if st == "rydberg":
            st = "2q"
        if st not in ("1q", "2q"):
            raise TypeError("Invalid stage type. Must be '1q', '2q', or 'rydberg'.")
        self.stage_type = st
        self.gates: list[ZNAAGate] = []
        if gates:
            self.extend_gates(gates)

    def append_gate(self, gate: ZNAAGate):
        if gate.gate_type != self.stage_type:
            raise TypeError(f"Gate {gate} type {gate.gate_type} != stage {self.stage_type}")
        self.gates.append(gate)

    def extend_gates(self, gates: list[ZNAAGate]):
        for g in gates:
            self.append_gate(g)

    def is_empty(self) -> bool:
        return len(self.gates) == 0

    def __str__(self) -> str:
        # Keep it compact: stages may contain many gates.
        n = len(self.gates)
        if n == 0:
            return f"[Stage {self.stage_type}] gates=0"
        # Show a small prefix for readability.
        shown = self.gates[:6]
        suffix = "..." if n > 6 else ""
        gates_str = ", ".join(str(g) for g in shown) + suffix
        return f"[Stage {self.stage_type}] gates={n}: {gates_str}"

    def __repr__(self) -> str:
        return f"ZNAAStage(stage_type={self.stage_type!r}, gates={len(self.gates)})"


_NAME_FOR_FID = {
    "1q": "f_1Q",
    "2q": "f_2Q",
    "transfer": "f_Trans",
    "decoherence1": "f_Deco-v1",
    "decoherence2": "f_Deco-v2",
    "readout": "f_Read",
    "execution": "f_Exec",
}


class ZNAAResult:
    """Post-run metrics: one row per physical qubit ID plus global time / fidelity breakdown."""

    def __init__(self, fidelity_list: list[str]):
        self.result_qubits: list[dict[str, Any]] = []
        self.fidelity_keys = list(fidelity_list)
        self.fidelity_labels = [_NAME_FOR_FID.get(x, x) for x in fidelity_list]
        self.time: float = 0.0
        self.total_fidelity: float = 1.0
        self.split_fidelity: list[float] = []
        self.split_fidelity_linear: list[float] = []  # multiplicative components before -log

    def append(self, qubit: dict):
        self.result_qubits.append(qubit)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "time_us": self.time,
            "total_fidelity": self.total_fidelity,
            "fidelity_components": dict(zip(self.fidelity_labels, self.split_fidelity_linear)),
        }

    def __str__(self) -> str:
        return f"[ZNAAResult] time_us={self.time}, total_fidelity={self.total_fidelity}"

    def __repr__(self) -> str:
        return f"ZNAAResult(time_us={self.time!r}, total_fidelity={self.total_fidelity!r})"


class ZNAAConfig:
    """Unified hardware config: logical shape + physical/timing parameters."""

    def __init__(
        self,
        shape_storage: tuple[int, int],
        shape_entangling: tuple[int, int],
        shape_readout: tuple[int, int],
        *,
        time_1q: float,
        time_2q: float,
        time_readout: float,
        time_transfer: float,
        fidelity_1q: float,
        fidelity_2q: float,
        fidelity_readout: float,
        fidelity_transfer: float,
        fidelity_execution: float,
        coherence_time_storage: float,
        coherence_time_else: float,
        aod_accelerate: float,
        distance_storage: tuple[float, float],
        distance_entangle: tuple[float, float],
        distance_readout: tuple[float, float],
        distance_interzone: float,
        rydberg_radius: float,
        delta: float,
        name: str = "",
    ):
        self.storage, self.entangling, self.readout = shape_storage, shape_entangling, shape_readout
        self.time_1q, self.time_2q, self.time_readout, self.time_transfer = time_1q, time_2q, time_readout, time_transfer
        self.fidelity_1q, self.fidelity_2q, self.fidelity_readout = fidelity_1q, fidelity_2q, fidelity_readout
        self.fidelity_transfer, self.fidelity_execution = fidelity_transfer, fidelity_execution
        self.coherence_time_storage, self.coherence_time_else, self.aod_accelerate = coherence_time_storage, coherence_time_else, aod_accelerate
        self.distance_storage, self.distance_entangle = distance_storage, distance_entangle
        self.distance_readout, self.distance_interzone, self.rydberg_radius, self.delta, self.name = distance_readout, distance_interzone, rydberg_radius, float(delta), name
        self._slm_geometry_cache = None
        self._zone_layout_cache = None

    def max_qubits(self):
        return self.storage[0] * self.storage[1]

    def storage_sep(self) -> tuple[float, float]:
        return float(self.distance_storage[0]), float(self.distance_storage[1])

    def entangle_sep(self) -> tuple[float, float]:
        return float(self.distance_entangle[0]), float(self.distance_entangle[1])

    def interzone_sep(self) -> float:
        return float(self.distance_interzone)

    # Shared SLM rectilinear grid: VM coordinates and mapped machine coordinates use the same (slm_id, row, col) frame.
    @staticmethod
    def _grid_xy(sep: tuple[float, float], origin: tuple[float, float], row: int, col: int) -> tuple[float, float]:
        return sep[0] * row + origin[0], sep[1] * col + origin[1]

    def _xy_from_slm(self, slm_map: dict, idx: int, r: int, c: int) -> tuple[float, float]:
        slm = slm_map[idx]
        return self._grid_xy(
            (float(slm["site_seperation"][0]), float(slm["site_seperation"][1])),
            (float(slm["location"][0]), float(slm["location"][1])),
            r,
            c,
        )

    def _dis_from_slm(self, slm_map: dict, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        return math.dist(
            self._xy_from_slm(slm_map, a[0], a[1], a[2]),
            self._xy_from_slm(slm_map, b[0], b[1], b[2]),
        )

    # Input format: storage/entangling shapes are (row_count, col_count) tuples.
    def build_slm_geometry(self) -> dict:
        if self._slm_geometry_cache is not None:
            return self._slm_geometry_cache
        sr, sc = self.storage
        er, ec = self.entangling
        sep_s, sep_e = self.storage_sep(), self.entangle_sep()
        di, ec2 = self.interzone_sep(), max(1, ec // 2)
        storage_zone = [0]
        dict_slm = {
            0: {"idx": 0, "site_seperation": [sep_s[0], sep_s[1]], "n_r": max(1, sr), "n_c": max(1, sc), "location": [sep_e[0] * (er - 1) + di, -sep_s[1] * (sc - 1)/2], "entanglement_id": -1},
            1: {"idx": 1, "site_seperation": [sep_e[0], sep_e[1]], "n_r": max(1, er), "n_c": ec2, "location": [0.0,  - sep_e[1] * (ec/2 - 1)/2 - self.rydberg_radius/2], "entanglement_id": 0},
            2: {"idx": 2, "site_seperation": [sep_e[0], sep_e[1]], "n_r": max(1, er), "n_c": ec2, "location": [0.0,  - sep_e[1] * (ec/2 - 1)/2 + self.rydberg_radius/2], "entanglement_id": 0},
        }
        entanglement_zone, entangle_slm_ids_flat = [[1, 2]], [1, 2]
        entanglement_site_row_space, entanglement_site_col_space = [], {}
        # Axis convention in this codebase:
        # - row aligns with x (vertical)
        # - col aligns with y (horizontal)
        x_site = sorted((dict_slm[row[0]]["location"][0], i) for i, row in enumerate(entanglement_zone))
        for i in range(len(x_site) - 1):
            slm = dict_slm[entanglement_zone[x_site[i][1]][0]]
            edge = (x_site[i + 1][0] + x_site[i][0] + slm["site_seperation"][0] * (slm["n_r"] - 1)) / 2.0
            entanglement_site_row_space.append((edge, slm["idx"]))
        entanglement_site_row_space.append((math.inf, dict_slm[entanglement_zone[x_site[-1][1]][0]]["idx"]))
        for row in entanglement_zone:
            idx, slm = row[0], dict_slm[row[0]]
            y0 = slm["location"][1] + slm["site_seperation"][1] / 2.0
            entanglement_site_col_space[idx] = [y0 + c * slm["site_seperation"][1] for c in range(slm["n_c"] - 1)] + [math.inf]
        storage_to_rydberg, storage_to_rydberg_dis, rydberg_to_storage = {}, {}, {}
        for row in entanglement_zone:
            slm = dict_slm[row[0]]
            rydberg_to_storage[row[0]] = [[-1 for _ in range(slm["n_c"])] for _ in range(2)]
        for idx in storage_zone:
            slm = dict_slm[idx]
            storage_to_rydberg[idx] = [[0 for _ in range(slm["n_c"])] for _ in range(slm["n_r"])]
            storage_to_rydberg_dis[idx] = [[0 for _ in range(slm["n_c"])] for _ in range(slm["n_r"])]
            x, y = slm["location"]
            ns = entanglement_site_row_space[-1][1]
            ns2, ylim = -1, entanglement_site_row_space[-1][0]
            half = dict_slm[entanglement_site_row_space[-1][1]]["n_r"] // 2
            row = 0 if abs(x - dict_slm[ns]["location"][0]) < abs(x - (dict_slm[ns]["location"][0] + (dict_slm[ns]["n_r"] - 1) * slm["site_seperation"][0])) else dict_slm[ns]["n_r"] - 1
            inc = False
            for i in range(len(entanglement_site_row_space) - 1):
                if x < entanglement_site_row_space[i][0]:
                    ns = entanglement_site_row_space[i][1]
                    ns2 = entanglement_site_row_space[i + 1][1]
                    ylim = entanglement_site_row_space[i][0]
                    row = dict_slm[ns]["n_r"] - 1
                    inc = True
                    break
            init_y, y_lim, col = y, entanglement_site_col_space[ns][-1], dict_slm[ns]["n_c"] - 1
            for i, lim in enumerate(entanglement_site_col_space[ns]):
                if y < lim:
                    y_lim, col = lim, i
                    break
            for r in range(slm["n_r"]):
                y, yl, c = init_y, y_lim, col
                for cc in range(slm["n_c"]):
                    site = (ns, row, c)
                    storage_to_rydberg[idx][r][cc] = site
                    storage_to_rydberg_dis[idx][r][cc] = self._dis_from_slm(dict_slm, (idx, r, cc), site)
                    ridx = 0 if row < half else 1
                    prev = rydberg_to_storage[ns][ridx][c]
                    if prev == -1 or storage_to_rydberg_dis[prev[0]][prev[1]][prev[2]] > storage_to_rydberg_dis[idx][r][cc]:
                        rydberg_to_storage[ns][ridx][c] = (idx, r, cc)
                    y += slm["site_seperation"][1]
                    if y > yl and c + 1 < dict_slm[ns]["n_c"]:
                        c += 1
                        yl = entanglement_site_col_space[ns][c]
                x += slm["site_seperation"][0]
                if inc and x > ylim and ns2 > -1:
                    inc, ns, row = False, ns2, 0
        for rows in rydberg_to_storage.values():
            first, last = [-1, -1], [-1, -1]
            for i, line in enumerate(rows):
                for j, site in enumerate(line):
                    if site != -1:
                        first[i] = j if first[i] == -1 else first[i]
                        last[i] = j
            for i in range(2):
                if first[i] != -1:
                    for j in range(first[i] - 1, -1, -1):
                        rows[i][j] = rows[i][first[i]]
                if last[i] != -1:
                    for j in range(last[i] + 1, len(rows[i])):
                        rows[i][j] = rows[i][last[i]]
            if first[0] == -1 and last[0] == -1:
                rows[0] = rows[1]
            elif first[1] == -1 and last[1] == -1:
                rows[1] = rows[0]
        self._slm_geometry_cache = {
            "storage_zone": storage_zone,
            "entanglement_zone": entanglement_zone,
            "dict_slm": dict_slm,
            "entangle_slm_ids_flat": entangle_slm_ids_flat,
            "storage_to_rydberg": storage_to_rydberg,
            "storage_to_rydberg_dis": storage_to_rydberg_dis,
            "rydberg_to_storage": rydberg_to_storage,
        }
        return self._slm_geometry_cache

    # Input format: loc is (slm_id, row, col) in the VM or machine-mapped SLM frame.
    def slm_site_xy(self, loc: tuple[int, int, int]) -> tuple[float, float]:
        g = self.build_slm_geometry()
        return self._xy_from_slm(g["dict_slm"], loc[0], loc[1], loc[2])

    def storage_nearest_rydberg_dis(self, idx: int, r: int, c: int) -> float:
        g = self.build_slm_geometry()
        return g["storage_to_rydberg_dis"][idx][r][c]

    # Input format: (slm_id, row, col) on an entangling SLM; output is the paired storage SLM cell.
    def nearest_storage_site(self, idx: int, r: int, c: int) -> tuple[int, int, int]:
        g = self.build_slm_geometry()
        dict_slm, ent_zone = g["dict_slm"], g["entanglement_zone"]
        anchor = dict_slm[ent_zone[dict_slm[idx]["entanglement_id"]][0]]
        return g["rydberg_to_storage"][anchor["idx"]][0 if r < anchor["n_r"] // 2 else 1][c]

    # Input format: two storage SLM cells; output is a short list of entangling sites used for gate placement heuristics.
    def nearest_ent_sites(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        g = self.build_slm_geometry()
        s1 = g["storage_to_rydberg"][a[0]][a[1]][a[2]]
        s2 = g["storage_to_rydberg"][b[0]][b[1]][b[2]]
        if s1 == s2:
            return [s1]
        if s1[0] == s2[0]:
            return [(s1[0], s1[1], (s1[2] + s2[2]) // 2)]
        return [s1, s2]

    def nearest_ent_distance(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        pa, pb, best = self.slm_site_xy(a), self.slm_site_xy(b), math.inf
        for site in self.nearest_ent_sites(a, b):
            ps = self.slm_site_xy(site)
            if a[0] == b[0] and a[1] == b[1]:
                best = min(best, max(math.dist(pa, ps), math.dist(pb, ps)))
            else:
                best = min(best, math.dist(pa, ps) + math.dist(pb, ps))
        return best

    # Compact zone layout derived from dict_slm so sep/loc stay aligned with build_slm_geometry().
    def build_zone_layout(self) -> dict:
        if self._zone_layout_cache is not None:
            return self._zone_layout_cache
        g = self.build_slm_geometry()
        ds = g["dict_slm"]
        self._zone_layout_cache = {
            "storage": 0,
            "ent": [1, 2],
            "slm": {
                k: {
                    "n_r": int(ds[k]["n_r"]),
                    "sep": (float(ds[k]["site_seperation"][0]), float(ds[k]["site_seperation"][1])),
                    "loc": (float(ds[k]["location"][0]), float(ds[k]["location"][1])),
                }
                for k in (0, 1, 2)
            },
        }
        return self._zone_layout_cache

    def __str__(self) -> str:
        return (
            "[ZNAAConfig] "
            f"storage={self.storage}, entangling={self.entangling}, "
            f"time_2q={self.time_2q}, fidelity_2q={self.fidelity_2q}, "
            f"rydberg_radius={self.rydberg_radius}, delta={self.delta}"
        )

    def __repr__(self) -> str:
        return (
            f"ZNAAConfig(storage={self.storage!r}, entangling={self.entangling!r}, readout={self.readout!r}, "
            f"time_1q={self.time_1q!r}, time_2q={self.time_2q!r}, distance_storage={self.distance_storage!r}, "
            f"distance_entangle={self.distance_entangle!r}, distance_interzone={self.distance_interzone!r})"
        )


class ZNAAOperation:
    """
    Base class for all machine-level operations.

    Notes:
    - Every operation carries a UUID so instances are identity-distinct even if fields match.
    - Solver / evaluator may keep references to operation objects while also serializing them via ``str(...)``.
    - Equality/hash are UUID-based by design (not structural).
    """

    def __init__(self):
        self.uuid = uuid.uuid4()

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if not isinstance(other, ZNAAOperation):
            return False
        return self.uuid == other.uuid

    def __ne__(self, other):
        return not self == other


class Operation_Map(ZNAAOperation):
    """SLM placement; requires explicit 0-based ``qubit_id`` matching circuit / ``ZNAAGate.qubits`` indices."""
    type_name = "map"
    
    def __init__(self, qubit_id: int, coord: tuple, zone: str = "s"):
        super().__init__()
        self.qubit_id = qubit_id
        self.coord = coord
        self.zone = zone
        if qubit_id < 0 :
            raise ZNAAMachineError(f"Invalid qubit id: {qubit_id}.")

    def __str__(self):
        return f"[Map] q{self.qubit_id} -> {self.zone}{self.coord}"


class Operation_1QGate(ZNAAOperation):
    """Parallel 1Q on 0-based qubit ids; optional ``gate_types`` list same length as ``qubit_ids``."""
    type_name = "1qg"
    
    def __init__(self, gate_type: str, qubit_ids: list[int]):
        super().__init__()
        self.gate_type = gate_type
        self.qubit_ids = qubit_ids

    def __str__(self):
        return f"[1QGate] {self.gate_type} ({', '.join(f'q{q}' for q in self.qubit_ids)})"


class Operation_2QGate(ZNAAOperation):
    """Global 2Q pulse; interacting pairs inferred from layout (SLM pairs + AOD within rydberg_radius)."""
    type_name = "2qg"
    
    def __init__(self, gate_type: str):
        super().__init__()
        self.gate_type = gate_type

    def __str__(self):
        return f"[2QGate] {self.gate_type}"


class Operation_Open(ZNAAOperation):
    """
    Add AOD row/column lines and optionally pick a subset of qubits.

    Input format:
    - ``rows``/``columns`` are axis values in the chosen open zone.
    - ``pick_ids`` limits which occupied coordinates are loaded onto AOD.
    """
    type_name = "open"
    
    def __init__(self, rows: list[float] = None, columns: list[float] = None, pick_ids: set[int] = None, open_zone: str | None = None):
        super().__init__()
        self.rows = rows or []
        self.columns = columns or []
        self.pick_ids = pick_ids
        self.open_zone = open_zone

    def __str__(self):
        sel = f" pick_ids={self.pick_ids}" if self.pick_ids is not None else ""
        oz = f" zone={self.open_zone}" if self.open_zone else ""
        return f"[Open] rows={self.rows}; columns={self.columns}{oz}{sel}"


class Operation_Close(ZNAAOperation):
    """
    Remove selected AOD lines and drop carried qubits back to SLM.

    Input format:
    - ``rows``/``columns`` refer to currently open AOD lines.
    - target SLM coordinates must be legal and empty.
    """
    type_name = "close"
    
    def __init__(self, rows: list[float] = None, columns: list[float] = None, close_zone: str | None = None):
        super().__init__()
        self.rows = rows or []
        self.columns = columns or []
        self.close_zone = close_zone

    def __str__(self):
        cz = f" zone={self.close_zone}" if self.close_zone else ""
        return f"[Close] rows={self.rows}; columns={self.columns}{cz}"


class Operation_Move(ZNAAOperation):
    """
    Move active AOD lines to target coordinates in one zone.

    Input format:
    - ``target_rows`` and ``target_columns`` are ascending lists.
    - lengths must match currently open AOD row/column counts.
    """
    type_name = "move"
    
    def __init__(self, target_zone: str, target_rows: list[float], target_columns: list[float]):
        super().__init__()
        self.target_zone = target_zone
        self.target_rows = target_rows
        self.target_columns = target_columns
        for i in range(len(target_rows) - 1):
            if target_rows[i] > target_rows[i+1]:
                raise ZNAAMachineError(f"Invalid target rows: {target_rows}.")
        for i in range(len(target_columns) - 1):
            if target_columns[i] > target_columns[i+1]:
                raise ZNAAMachineError(f"Invalid target columns: {target_columns}.")

    def __str__(self):
        return f"[Move] zone={self.target_zone} rows={self.target_rows}; cols={self.target_columns}"


class ZNAAMachineError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ZNAAMachine:
    """
    Ground-truth executor for the ZNAA operation language.

    Conceptual model:
    - SLM zones hold persistent qubit occupancy on integer lattice sites:
      ``storage``, ``entangling``, and ``readout``.
    - AOD is a transient transport mechanism represented as sorted row/column line sets plus
      a matrix ``content_aod`` of currently carried qubits.
    - ``append_operation(...)`` mutates occupancy/positions/counters/time in-place and records
      step-level history for debugging.

    Key invariants enforced by the machine:
    - Coordinates must be legal integer indices for their target zone.
    - AOD rows/columns remain sorted; ``Move`` targets must preserve ascending order and size.
    - AOD line spacing must satisfy ``hardware_parameters.delta`` in absolute geometry.
    - ``Close`` can only drop onto empty legal SLM cells.
    - ``Operation_2QGate`` is global: interacting pairs are inferred geometrically, then validated
      so each candidate qubit participates in exactly one pair.
    """

    def __init__(
        self,
        hardware_config: Callable[[], ZNAAConfig],
        qreg_name: str = "q",
        creg_name: str = "c",
        fidelity_list=None,
        skip_open_close_chain_time: bool = False,
        selective_transfer: bool = False
    ):
        self.hardware_config = hardware_config()
        self.hardware_shape = self.hardware_config
        self.hardware_parameters = self.hardware_config
        self.content_entangling = np.full(self.hardware_shape.entangling, _EMPTY_QUBIT_CELL, dtype=int)
        self.content_storage = np.full(self.hardware_shape.storage, _EMPTY_QUBIT_CELL, dtype=int)
        self.content_readout = np.full(self.hardware_shape.readout, _EMPTY_QUBIT_CELL, dtype=int)
        self.aod_rows: list[float] = []
        self.aod_columns: list[float] = []
        self.content_aod: list[list[int]] = []
        self.aod_current_place = None
        self.current_qubits: set[int] = set()
        self.qubit_to_num1q: dict[int, int] = {}
        self.qubit_to_num2q: dict[int, int] = {}
        self.qubit_to_numtrans: dict[int, int] = {}
        self.qubit_to_timeS: dict[int, float] = {}
        self.qubit_to_timeE: dict[int, float] = {}
        self.qubit_to_ismeasured: dict[int, bool] = {}
        self.qubit_to_position: dict[int, ZNAACoord] = {}
        self.total_num1q = 0
        self.total_num2q = 0
        self.total_numtrans = 0
        self.total_numread = 0
        self.precalc_absolute()
        self.qreg_name = qreg_name
        self.creg_name = creg_name
        self.znaa_format_circuit = ZNAACircuit()
        fidelity_list = fidelity_list or ["1q", "2q", "transfer", "readout", "decoherence1"]
        for fidelity_type in fidelity_list:
            if fidelity_type not in ("1q", "2q", "transfer", "readout", "decoherence1", "decoherence2"):
                raise ZNAAMachineError(f"Invalid fidelity type: {fidelity_type}.")
        self.fidelity_list = fidelity_list
        self.operations: list[ZNAAOperation] = []
        self.time = 0.0
        self.operation_history: list[dict] = []
        self.skip_open_close_chain_time = skip_open_close_chain_time
        self.selective_transfer = selective_transfer

    def _get_relevant_prev_op_type_for_open_close(self) -> Optional[str]:
        if not self.operation_history:
            return None
        for i in range(len(self.operation_history) - 1, -1, -1):
            t = self.operation_history[i].get("op_type", "")
            if t in ("1qgate", "2qgate"):
                continue
            if t in ("move", "open", "close"):
                return t
            return None
        return None

    def _coord_values_are_integer(self, coord: ZNAACoord) -> bool:
        def ok(v):
            return isinstance(v, int) or (isinstance(v, float) and v.is_integer())

        return ok(coord.x) and ok(coord.y)

    def is_legal_coord(self, coord: ZNAACoord) -> bool:
        if not self._coord_values_are_integer(coord):
            return False
        if coord.zone == "storage":
            if not (0 <= coord.x < self.hardware_shape.storage[0] and 0 <= coord.y < self.hardware_shape.storage[1]):
                return False
        elif coord.zone == "entangling":
            if not (0 <= coord.x < self.hardware_shape.entangling[0] and 0 <= coord.y < self.hardware_shape.entangling[1]):
                return False
        elif coord.zone == "readout":
            if not (0 <= coord.x < self.hardware_shape.readout[0] and 0 <= coord.y < self.hardware_shape.readout[1]):
                return False
        elif coord.zone == "aod":
            if not (0 <= coord.x < len(self.aod_rows) and 0 <= coord.y < len(self.aod_columns)):
                return False
        return True

    def get_id_by_coord(self, coord: ZNAACoord) -> int:
        if not self.is_legal_coord(coord):
            raise ZNAAMachineError(f"Invalid coordinate: {coord}.")
        if coord.zone == "storage":
            return int(self.content_storage[int(coord.x)][int(coord.y)])
        if coord.zone == "entangling":
            return int(self.content_entangling[int(coord.x)][int(coord.y)])
        if coord.zone == "readout":
            return int(self.content_readout[int(coord.x)][int(coord.y)])
        if coord.zone == "aod":
            return int(self.content_aod[int(coord.x)][int(coord.y)])
        return _EMPTY_QUBIT_CELL

    def is_occupied_coord(self, coord: ZNAACoord) -> bool:
        return self.get_id_by_coord(coord) != _EMPTY_QUBIT_CELL

    def set_id(self, coord: ZNAACoord, qubit_id: int):
        if not self.is_legal_coord(coord):
            raise ZNAAMachineError(f"Invalid coordinate: {coord}.")
        if coord.zone == "storage":
            self.content_storage[int(coord.x)][int(coord.y)] = qubit_id
        elif coord.zone == "entangling":
            self.content_entangling[int(coord.x)][int(coord.y)] = qubit_id
        elif coord.zone == "readout":
            self.content_readout[int(coord.x)][int(coord.y)] = qubit_id
        elif coord.zone == "aod":
            self.content_aod[int(coord.x)][int(coord.y)] = qubit_id

    def _is_integer_coord(self, r_or_c: float) -> bool:
        return isinstance(r_or_c, int) or (isinstance(r_or_c, float) and r_or_c.is_integer())

    def _short_zone(self, zone: str) -> str:
        return {"storage": "s", "entangling": "e", "aod": "a", "readout": "r"}.get(zone, zone[0] if zone else "?")

    def _aod_row(self, i: int) -> ZNAARow:
        if self.aod_current_place is None:
            raise ZNAAMachineError("AOD has no current place.")
        return ZNAARow(self.aod_current_place, self.aod_rows[i])

    def _aod_col(self, j: int) -> ZNAAColumn:
        if self.aod_current_place is None:
            raise ZNAAMachineError("AOD has no current place.")
        return ZNAAColumn(self.aod_current_place, self.aod_columns[j])

    def _validate_aod_spacing(self, zone: str, rows: list[float], cols: list[float]):
        delta = float(getattr(self.hardware_parameters, "delta", 0.0))
        if delta <= 0:
            return
        srows = sorted(float(r) for r in rows)
        for i in range(len(srows) - 1):
            d = abs(
                self.calc_absolute(ZNAARow(zone, srows[i + 1]))
                - self.calc_absolute(ZNAARow(zone, srows[i]))
            )
            if d < delta:
                raise ZNAAMachineError(
                    f"AOD row spacing too small: |r{i+1}-r{i}| absolute distance {d:.6f} < delta {delta:.6f}"
                )
        scols = sorted(float(c) for c in cols)
        for i in range(len(scols) - 1):
            d = abs(
                self.calc_absolute(ZNAAColumn(zone, scols[i + 1]))
                - self.calc_absolute(ZNAAColumn(zone, scols[i]))
            )
            if d < delta:
                raise ZNAAMachineError(
                    f"AOD column spacing too small: |c{i+1}-c{i}| absolute distance {d:.6f} < delta {delta:.6f}"
                )

    def _add_elapsed_time(self, delta: float):
        for q in self.current_qubits:
            pos = self.qubit_to_position.get(q)
            if pos is None:
                continue
            if pos.zone == "storage":
                self.qubit_to_timeS[q] = self.qubit_to_timeS.get(q, 0) + delta
            else:
                self.qubit_to_timeE[q] = self.qubit_to_timeE.get(q, 0) + delta

    def _flush_qubit_segment(self, qubit_id: int):
        t = self.time - self.qubit_to_timeS[qubit_id] - self.qubit_to_timeE[qubit_id]
        if self.qubit_to_position[qubit_id].zone == "storage":
            self.qubit_to_timeS[qubit_id] += t
        else:
            self.qubit_to_timeE[qubit_id] += t

    def append_operation(self, operation: ZNAAOperation):
        """
        Apply one machine operation and update all side effects.

        Per-op responsibilities include:
        - occupancy/position transitions (SLM <-> AOD),
        - timing accumulation and coherence segment bookkeeping,
        - logical gate tracing into ``znaa_format_circuit``,
        - per-qubit/global counters used by ``run()`` fidelity aggregation,
        - rich ``operation_history`` snapshots for diagnostics.

        This method is intentionally strict: invalid physical transitions fail fast via
        ``ZNAAMachineError`` so planner bugs surface early.
        """
        self.operations.append(operation)
        step = len(self.operation_history) + 1
        if isinstance(operation, Operation_Map):
            # MAP initializes qubit existence, occupancy, counters, and timeline state.
            if operation.qubit_id < 0:
                raise ZNAAMachineError(f"Invalid qubit id {operation.qubit_id}")
            if operation.qubit_id in self.current_qubits:
                raise ZNAAMachineError(f"Qubit [{operation.qubit_id}] already exists.")
            coord = ZNAACoord(
                operation.zone,
                int(operation.coord[0]),
                int(operation.coord[1]),
            )
            if self.is_occupied_coord(coord):
                raise ZNAAMachineError(f"Coordinate {coord} occupied.")
            self.set_id(coord, operation.qubit_id)
            self.current_qubits.add(operation.qubit_id)
            self.qubit_to_num1q[operation.qubit_id] = 0
            self.qubit_to_num2q[operation.qubit_id] = 0
            self.qubit_to_numtrans[operation.qubit_id] = 0
            self.qubit_to_timeS[operation.qubit_id] = 0.0
            self.qubit_to_timeE[operation.qubit_id] = 0.0
            self.qubit_to_ismeasured[operation.qubit_id] = False
            self.qubit_to_position[operation.qubit_id] = coord
            self.operation_history.append(
                {
                    "step": step,
                    "op_type": "map",
                    "operation": str(operation),
                    "consequence": f"{self.qreg_name}[{operation.qubit_id}] -> {self._short_zone(coord.zone)}({int(coord.x)},{int(coord.y)})",
                    "time_after": self.time,
                }
            )
        elif isinstance(operation, Operation_1QGate):
            # 1Q operation updates counters and appends logical trace gates.
            qubit_ids = operation.qubit_ids
            gate_types = getattr(operation, "gate_types", [operation.gate_type] * len(qubit_ids))
            if len(gate_types) != len(qubit_ids):
                raise ZNAAMachineError("Operation_1QGate gate_types length mismatch.")
            for q in qubit_ids:
                if q not in self.current_qubits:
                    raise ZNAAMachineError(f"Qubit [{q}] does not exist.")
            for q, gtype in zip(qubit_ids, gate_types):
                self.qubit_to_num1q[q] += 1
                self.total_num1q += 1
                self.znaa_format_circuit.append_gate(ZNAAGate("1q", gtype, [q]))
            t1q = float(getattr(self.hardware_parameters, "time_1q", 0.0))
            if t1q > 0:
                self.time += t1q * len(qubit_ids)
                self._add_elapsed_time(t1q)
            self.operation_history.append(
                {
                    "step": step,
                    "op_type": "1qgate",
                    "operation": str(operation),
                    "qubits": list(qubit_ids),
                    "time_after": self.time,
                }
            )
        elif isinstance(operation, Operation_Open):
            # OPEN may load qubits from SLM into AOD at selected rows/columns.
            oz = getattr(operation, "open_zone", None)
            open_zone = oz if oz else (self.aod_current_place or "storage")
            rows_to_open = [ZNAARow(open_zone, float(r)) for r in (operation.rows or [])]
            columns_to_open = [ZNAAColumn(open_zone, float(c)) for c in (operation.columns or [])]
            open_zone_norm = ZNAARow(open_zone, 0.0).zone
            rows_after_open = list(self.aod_rows) + [float(r.r) for r in rows_to_open]
            cols_after_open = list(self.aod_columns) + [float(c.c) for c in columns_to_open]
            self._validate_aod_spacing(open_zone_norm, rows_after_open, cols_after_open)
            sel = operation.pick_ids
            open_picked = []
            for row in rows_to_open:
                zone = row.zone
                r_val = float(row.r)
                if self.aod_current_place is not None and zone != self.aod_current_place:
                    raise ZNAAMachineError("Open: mixed AOD zones.")
                if r_val in self.aod_rows:
                    raise ZNAAMachineError(f"Open: row r={r_val} already open.")
                r_int = int(row.r)
                picked = []
                pos_ins = bisect.bisect_left(self.aod_rows, r_val)
                for j, c_val in enumerate(self.aod_columns):
                    if not self._is_integer_coord(c_val):
                        continue
                    coord_slm = ZNAACoord(zone, r_int, int(c_val))
                    if not self.is_legal_coord(coord_slm):
                        continue
                    q = self.get_id_by_coord(coord_slm)
                    if q != _EMPTY_QUBIT_CELL and ((not self.selective_transfer) or sel is None or q in sel):
                        self._flush_qubit_segment(q)
                        picked.append((q, pos_ins, j))
                self.aod_rows.insert(pos_ins, r_val)
                self.content_aod.insert(
                    pos_ins,
                    [_EMPTY_QUBIT_CELL] * len(self.aod_columns) if self.aod_columns else [],
                )
                for (q, ai, aj) in picked:
                    coord_slm = ZNAACoord(zone, r_int, int(self.aod_columns[aj]))
                    self.set_id(coord_slm, _EMPTY_QUBIT_CELL)
                    self.content_aod[ai][aj] = q
                    self.qubit_to_position[q] = ZNAACoord("aod", ai, aj)
                    self.qubit_to_numtrans[q] += 1
                    open_picked.append((q, zone, r_int, int(self.aod_columns[aj]), ai, aj))
            for col in columns_to_open:
                if not self._is_integer_coord(col.c):
                    raise ZNAAMachineError(f"Open: column must be integer grid, got {col.c}.")
                zone = col.zone
                c_val = float(col.c)
                if self.aod_current_place is not None and zone != self.aod_current_place:
                    raise ZNAAMachineError("Open: mixed AOD zones.")
                if c_val in self.aod_columns:
                    raise ZNAAMachineError(f"Open: column c={c_val} already open.")
                c_int = int(col.c)
                picked = []
                col_ins = bisect.bisect_left(self.aod_columns, c_val)
                for i, r_val in enumerate(self.aod_rows):
                    if not self._is_integer_coord(r_val):
                        continue
                    coord_slm = ZNAACoord(zone, int(r_val), c_int)
                    if not self.is_legal_coord(coord_slm):
                        continue
                    q = self.get_id_by_coord(coord_slm)
                    if q != _EMPTY_QUBIT_CELL and ((not self.selective_transfer) or sel is None or q in sel):
                        self._flush_qubit_segment(q)
                        picked.append((q, i, col_ins))
                self.aod_columns.insert(col_ins, c_val)
                for r in self.content_aod:
                    r.insert(col_ins, _EMPTY_QUBIT_CELL)
                for (q, ai, aj) in picked:
                    coord_slm = ZNAACoord(zone, int(self.aod_rows[ai]), c_int)
                    self.set_id(coord_slm, _EMPTY_QUBIT_CELL)
                    self.content_aod[ai][aj] = q
                    self.qubit_to_position[q] = ZNAACoord("aod", ai, aj)
                    self.qubit_to_numtrans[q] += 1
                    open_picked.append((q, zone, int(self.aod_rows[ai]), c_int, ai, aj))
            if (rows_to_open or columns_to_open) and self.aod_current_place is None:
                first = (rows_to_open or columns_to_open)[0]
                self.aod_current_place = first.zone
            if rows_to_open or columns_to_open:
                # OPEN contributes transfer latency unless chain-time optimization applies.
                if self.skip_open_close_chain_time and self.operation_history:
                    prev = self._get_relevant_prev_op_type_for_open_close()
                    t_open = self.hardware_parameters.time_transfer if (prev == "move" or prev is None) else 0.0
                else:
                    t_open = self.hardware_parameters.time_transfer
                self.time += t_open
                self.total_numtrans += len(open_picked)
                if t_open > 0:
                    self._add_elapsed_time(t_open)
                pick_lines = [f"q{q}: {self._short_zone(zn)}({fr},{fc})->a({ai},{aj})" for (q, zn, fr, fc, ai, aj) in open_picked]
                consequence = f"{operation}\n  " + ("\n  ".join(pick_lines) if pick_lines else "")
                self.operation_history.append(
                    {
                        "step": step,
                        "op_type": "open",
                        "operation": str(operation),
                        "consequence": consequence,
                        # Keep the exact picked logical ids for evaluator-side router checks.
                        "picked_qubits": sorted({int(q) for (q, _zn, _fr, _fc, _ai, _aj) in open_picked}),
                        "time_after": self.time,
                        "aod_content": copy.deepcopy(self.content_aod),
                    }
                )
        elif isinstance(operation, Operation_Close):
            # CLOSE validates all drops first, then flushes and materializes drops.
            if self.aod_current_place is None:
                raise ZNAAMachineError("Close with no active AOD.")
            zone_drop = self.aod_current_place
            row_targets = [ZNAARow(zone_drop, float(r)) for r in (operation.rows or [])]
            col_targets = [ZNAAColumn(zone_drop, float(c)) for c in (operation.columns or [])]
            to_flush: set[int] = set()
            close_dropped = []
            for target in row_targets:
                try:
                    idx = self.aod_rows.index(target.r)
                except ValueError as e:
                    raise ZNAAMachineError(f"No AOD row r={target.r}.") from e
                for j in range(len(self.aod_columns)):
                    if self.content_aod[idx][j] == _EMPTY_QUBIT_CELL:
                        continue
                    to_flush.add(self.content_aod[idx][j])
                    r_val, c_val = self.aod_rows[idx], self.aod_columns[j]
                    if not self._is_integer_coord(r_val) or not self._is_integer_coord(c_val):
                        raise ZNAAMachineError("Close: non-integer drop.")
                    drop_coord = ZNAACoord(zone_drop, int(r_val), int(c_val))
                    if not self.is_legal_coord(drop_coord) or self.get_id_by_coord(drop_coord) != _EMPTY_QUBIT_CELL:
                        raise ZNAAMachineError(f"Close: bad drop {drop_coord}.")
            for target in col_targets:
                try:
                    idx = self.aod_columns.index(target.c)
                except ValueError as e:
                    raise ZNAAMachineError(f"No AOD column c={target.c}.") from e
                for i in range(len(self.aod_rows)):
                    if self.content_aod[i][idx] == _EMPTY_QUBIT_CELL:
                        continue
                    to_flush.add(self.content_aod[i][idx])
                    r_val, c_val = self.aod_rows[i], self.aod_columns[idx]
                    if not self._is_integer_coord(r_val) or not self._is_integer_coord(c_val):
                        raise ZNAAMachineError("Close: non-integer drop.")
                    drop_coord = ZNAACoord(zone_drop, int(r_val), int(c_val))
                    if not self.is_legal_coord(drop_coord) or self.get_id_by_coord(drop_coord) != _EMPTY_QUBIT_CELL:
                        raise ZNAAMachineError(f"Close: bad drop {drop_coord}.")
            for q in to_flush:
                self._flush_qubit_segment(q)
            if row_targets or col_targets:
                if self.skip_open_close_chain_time and self.operation_history:
                    prev = self._get_relevant_prev_op_type_for_open_close()
                    t_close = self.hardware_parameters.time_transfer if (prev == "move" or prev is None) else 0.0
                else:
                    t_close = self.hardware_parameters.time_transfer
                self.time += t_close
                self.total_numtrans += len(close_dropped)
                if t_close > 0:
                    self._add_elapsed_time(t_close)
            row_indices = []
            for target in row_targets:
                try:
                    row_indices.append(self.aod_rows.index(target.r))
                except ValueError:
                    pass
            for idx in sorted(row_indices, reverse=True):
                r_int = int(self.aod_rows[idx])
                zone_d = zone_drop
                for j in range(len(self.aod_columns)):
                    q = self.content_aod[idx][j]
                    if q != _EMPTY_QUBIT_CELL:
                        drop_coord = ZNAACoord(zone_d, r_int, int(self.aod_columns[j]))
                        self.set_id(drop_coord, q)
                        self.qubit_to_position[q] = drop_coord
                        self.qubit_to_numtrans[q] += 1
                        close_dropped.append((q, zone_d, r_int, int(self.aod_columns[j]), idx, j))
                self.aod_rows.pop(idx)
                self.content_aod.pop(idx)
            col_indices = []
            for target in col_targets:
                try:
                    col_indices.append(self.aod_columns.index(target.c))
                except ValueError:
                    pass
            for idx in sorted(col_indices, reverse=True):
                c_int = int(self.aod_columns[idx])
                zone_d = zone_drop
                for i in range(len(self.aod_rows)):
                    q = self.content_aod[i][idx]
                    if q != _EMPTY_QUBIT_CELL:
                        drop_coord = ZNAACoord(zone_d, int(self.aod_rows[i]), c_int)
                        self.set_id(drop_coord, q)
                        self.qubit_to_position[q] = drop_coord
                        self.qubit_to_numtrans[q] += 1
                        close_dropped.append((q, zone_d, int(self.aod_rows[i]), c_int, i, idx))
                self.aod_columns.pop(idx)
                for r in self.content_aod:
                    r.pop(idx)
            if not self.aod_rows and not self.aod_columns:
                self.aod_current_place = None
                self.content_aod = []
            if row_targets or col_targets:
                drop_lines = [f"q{q}: a({ai},{aj})->{self._short_zone(zn)}({r},{c})" for (q, zn, r, c, ai, aj) in close_dropped]
                consequence = f"{operation}\n  " + ("\n  ".join(drop_lines) if drop_lines else "")
                self.operation_history.append(
                    {"step": step, "op_type": "close", "operation": str(operation), "consequence": consequence, "time_after": self.time, "aod_content": copy.deepcopy(self.content_aod)}
                )
        elif isinstance(operation, Operation_Move):
            # MOVE keeps AOD occupancy matrix shape unchanged and only updates line geometry.
            target_rows = [float(x) for x in operation.target_rows]
            target_columns = [float(x) for x in operation.target_columns]
            target_zone = ZNAARow(operation.target_zone or "storage", 0.0).zone
            if len(target_rows) != len(self.aod_rows) or len(target_columns) != len(self.aod_columns):
                raise ZNAAMachineError("Operation_Move size mismatch.")

            def _asc(lst):
                return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)) if len(lst) > 1 else True

            if not _asc(target_rows) or not _asc(target_columns):
                raise ZNAAMachineError("Operation_Move targets must be ascending.")
            self._validate_aod_spacing(target_zone, target_rows, target_columns)
            old_zone = self._short_zone(self.aod_current_place) if self.aod_current_place else "?"
            old_rows, old_columns = list(self.aod_rows), list(self.aod_columns)
            t = self.calc_time_of_aod_to(target_rows, target_columns, target_zone)
            self.time += t
            self._add_elapsed_time(t)
            self.aod_rows = list(target_rows)
            self.aod_columns = list(target_columns)
            self.aod_current_place = target_zone
            consequence = f"{operation}\n  rows: {old_zone}{old_rows} -> {self._short_zone(target_zone)}{target_rows}\n  cols: {old_zone}{old_columns} -> {self._short_zone(target_zone)}{target_columns}"
            self.operation_history.append(
                {"step": step, "op_type": "move", "operation": str(operation), "consequence": consequence, "time_after": self.time, "aod_content": copy.deepcopy(self.content_aod)}
            )
        elif isinstance(operation, Operation_2QGate):
            # 2Q is a global pulse: derive candidate qubits by geometry, then enforce
            # one-and-only-one pairing for every candidate qubit.
            candidate_positions: dict[int, tuple[float, float]] = {}
            for i in range(self.hardware_shape.entangling[0]):
                for j in range(self.hardware_shape.entangling[1]):
                    qid = int(self.content_entangling[i, j])
                    if qid == _EMPTY_QUBIT_CELL:
                        continue
                    p = self.calc_absolute(ZNAACoord("entangling", i, j))
                    candidate_positions[qid] = (float(p[0]), float(p[1]))
            if self.aod_current_place == "entangling":
                for i in range(len(self.aod_rows)):
                    for j in range(len(self.aod_columns)):
                        qid = int(self.content_aod[i][j])
                        if qid == _EMPTY_QUBIT_CELL:
                            continue
                        p = (
                            self.calc_absolute(self._aod_row(i)),
                            self.calc_absolute(self._aod_col(j)),
                        )
                        candidate_positions[qid] = (float(p[0]), float(p[1]))
            entangle_pairs: list[tuple[int, int]] = []
            candidate_ids = sorted(candidate_positions.keys())
            for i in range(len(candidate_ids)):
                q1 = candidate_ids[i]
                p1 = candidate_positions[q1]
                for j in range(i + 1, len(candidate_ids)):
                    q2 = candidate_ids[j]
                    p2 = candidate_positions[q2]
                    # Global 2Q pulse uses geometric proximity in absolute coordinates.
                    # For canonical entangling-neighbor pairs in this architecture, this
                    # corresponds to same-row / same-y//2 partner placement.
                    if self.euclidean_distance(p1, p2) <= self.hardware_parameters.rydberg_radius:
                        entangle_pairs.append((q1, q2))
            participation_count: dict[int, int] = {q: 0 for q in candidate_ids}
            for q1, q2 in entangle_pairs:
                participation_count[q1] = participation_count.get(q1, 0) + 1
                participation_count[q2] = participation_count.get(q2, 0) + 1
            multi_partner = [q for q, cnt in participation_count.items() if cnt > 1]
            if multi_partner:
                raise ZNAAMachineError(
                    f"2QGate invalid: qubits participate in multiple pairs in one 2q op: {entangle_pairs}"
                )
            idle_candidates = [q for q, cnt in participation_count.items() if cnt == 0]
            if idle_candidates:
                raise ZNAAMachineError(
                    f"2QGate invalid: candidate qubits not entangled in this 2q op: {sorted(idle_candidates)}"
                )
            for q1, q2 in entangle_pairs:
                self.qubit_to_num2q[q1] += 1
                self.qubit_to_num2q[q2] += 1
                self.znaa_format_circuit.append_gate(ZNAAGate("2q", operation.gate_type, [q1, q2]))
            self.total_num2q += len(entangle_pairs)
            t2q = float(getattr(self.hardware_parameters, "time_2q", 0.0))
            if t2q > 0:
                self.time += t2q
                self._add_elapsed_time(t2q)
            self.operation_history.append(
                {
                    "step": step,
                    "op_type": "2qgate",
                    "gate_type": getattr(operation, "gate_type", "cz"),
                    "operation": str(operation),
                    "time_after": self.time,
                    "entangle_pairs": list(entangle_pairs),
                }
            )
        else:
            raise ZNAAMachineError(f"Invalid operation: {operation}")

    def run(self) -> ZNAAResult:
        """Flush S/E dwell, then multiply fidelity components from ``fidelity_list``."""
        result = ZNAAResult(self.fidelity_list)
        spilt: list[float] = []
        deco_1 = deco_2 = 1.0
        for qubit in self.current_qubits:
            if self.qubit_to_position[qubit].zone == "storage":
                self.qubit_to_timeS[qubit] += self.time - self.qubit_to_timeS[qubit] - self.qubit_to_timeE[qubit]
            else:
                self.qubit_to_timeE[qubit] += self.time - self.qubit_to_timeS[qubit] - self.qubit_to_timeE[qubit]
            result.append(
                {
                    "qubit_id": qubit,
                    "num_1q": self.qubit_to_num1q[qubit],
                    "num_2q": self.qubit_to_num2q[qubit],
                    "num_transfer": self.qubit_to_numtrans[qubit],
                    "is_measured": self.qubit_to_ismeasured[qubit],
                    "time_coherence_S": self.qubit_to_timeS[qubit],
                    "time_coherence_E": self.qubit_to_timeE[qubit],
                }
            )
            deco_1 *= self.get_decoherence1(qubit)
            deco_2 *= self.get_decoherence2(qubit)
        for f_type in self.fidelity_list:
            if f_type == "1q":
                spilt.append(self.hardware_parameters.fidelity_1q**self.total_num1q)
            elif f_type == "2q":
                spilt.append(self.hardware_parameters.fidelity_2q**self.total_num2q)
            elif f_type == "transfer":
                spilt.append(self.hardware_parameters.fidelity_transfer**self.total_numtrans)
            elif f_type == "readout":
                spilt.append(self.hardware_parameters.fidelity_readout**self.total_numread)
            elif f_type == "decoherence1":
                spilt.append(deco_1)
            elif f_type == "decoherence2":
                spilt.append(deco_2)
        total_fidelity = math.prod(spilt) if spilt else 1.0
        result.time = self.time
        result.total_fidelity = total_fidelity
        result.split_fidelity_linear = list(spilt)
        result.split_fidelity = [-math.log(s) for s in spilt]
        return result

    def get_decoherence1(self, qubit_id: int) -> float:
        t = self.qubit_to_timeS[qubit_id] / self.hardware_parameters.coherence_time_storage
        t += self.qubit_to_timeE[qubit_id] / self.hardware_parameters.coherence_time_else
        return math.exp(-t)

    def get_decoherence2(self, qubit_id: int) -> float:
        t = self.qubit_to_timeE[qubit_id] / self.hardware_parameters.coherence_time_else
        return math.exp(-t)

    def calc_aod_time_single(self, coord1, coord2) -> float:
        if (isinstance(coord1, ZNAARow) and isinstance(coord2, ZNAARow)) or (
            isinstance(coord1, ZNAAColumn) and isinstance(coord2, ZNAAColumn)
        ):
            return math.sqrt(abs(self.calc_absolute(coord1) - self.calc_absolute(coord2)) / self.hardware_parameters.aod_accelerate)
        raise ZNAAMachineError("calc_aod_time_single: type mismatch.")

    def precalc_absolute(self):
        self.absolute_row_entangling = 0
        self.absolute_row_storage = (self.hardware_shape.entangling[0] - 1) * self.hardware_parameters.distance_entangle[0]
        self.absolute_row_storage += self.hardware_parameters.distance_interzone
        self.absolute_row_readout = self.absolute_row_storage + (self.hardware_shape.storage[0] - 1) * self.hardware_parameters.distance_storage[0]
        self.absolute_row_readout += self.hardware_parameters.distance_interzone
        n_ent_cols = int(self.hardware_shape.entangling[1])
        if n_ent_cols <= 1:
            self.totallength_col_entangling = 0.0
        else:
            self.totallength_col_entangling = self._entangling_col_unshifted(float(n_ent_cols - 1))
        self.totallength_col_storage = (self.hardware_shape.storage[1] - 1) * self.hardware_parameters.distance_storage[1]
        self.totallength_col_readout = (self.hardware_shape.readout[1] - 1) * self.hardware_parameters.distance_readout[1]

    def _entangling_col_unshifted(self, c: float) -> float:
        """Uncentered entangling-column x-position before global centering.

        Integer columns follow: 0, r, d, d+r, 2d, 2d+r, ... where
        r = rydberg_radius and d = distance_entangle[1].
        Non-integer c uses linear interpolation between neighboring integer columns.

        Adjacency note used by planner/evaluator:
        in this entangling layout, two sites are the intended nearest pair partners iff
        they are on the same row and in the same pair bucket (`same x` and `same y//2`,
        equivalent to |y_a - y_b| == 1 with both in that bucket).
        """
        d = float(self.hardware_parameters.distance_entangle[1])
        r = float(self.hardware_parameters.rydberg_radius)

        def at_int(k: int) -> float:
            return float((k // 2) * d + (k % 2) * r)

        c0 = int(math.floor(c))
        t = float(c - c0)
        if t == 0.0:
            return at_int(c0)
        return (1.0 - t) * at_int(c0) + t * at_int(c0 + 1)

    def calc_time_of_aod_to(self, target_row_positions: list, target_column_positions: list, target_zone: str) -> float:
        if len(target_row_positions) != len(self.aod_rows) or len(target_column_positions) != len(self.aod_columns):
            raise ZNAAMachineError("calc_time_of_aod_to size mismatch.")
        if not target_row_positions and not target_column_positions:
            return 0.0
        target_rows = [ZNAARow(target_zone, r) for r in target_row_positions]
        target_columns = [ZNAAColumn(target_zone, c) for c in target_column_positions]
        max_row_distance = max(
            abs(self.calc_absolute(self._aod_row(i)) - self.calc_absolute(target_rows[i]))
            for i in range(len(self.aod_rows))
        ) if target_rows else 0.0
        max_col_distance = max(
            abs(self.calc_absolute(self._aod_col(j)) - self.calc_absolute(target_columns[j]))
            for j in range(len(self.aod_columns))
        ) if target_columns else 0.0
        diagonal_distance = math.sqrt(max_row_distance ** 2 + max_col_distance ** 2)
        return math.sqrt(diagonal_distance / self.hardware_parameters.aod_accelerate)

    def calc_absolute(self, obj):
        if isinstance(obj, ZNAARow):
            if obj.zone == "entangling":
                return self.absolute_row_entangling + obj.r * self.hardware_parameters.distance_entangle[0]
            if obj.zone == "storage":
                return self.absolute_row_storage + obj.r * self.hardware_parameters.distance_storage[0]
            if obj.zone == "readout":
                return self.absolute_row_readout + obj.r * self.hardware_parameters.distance_readout[0]
        if isinstance(obj, ZNAAColumn):
            if obj.zone == "entangling":
                return self._entangling_col_unshifted(obj.c) - self.totallength_col_entangling / 2
            if obj.zone == "storage":
                return obj.c * self.hardware_parameters.distance_storage[1] - self.totallength_col_storage / 2
            if obj.zone == "readout":
                return obj.c * self.hardware_parameters.distance_readout[1] - self.totallength_col_readout / 2
        if isinstance(obj, ZNAACoord):
            return self.calc_absolute(obj.row()), self.calc_absolute(obj.column())
        raise ZNAAMachineError("calc_absolute: invalid object.")

    def euclidean_distance(self, p: tuple[float, float], q: tuple[float, float]) -> float:
        return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))


def config_test():
    # Absolute 2D position (x, y) for any SLM site uses integer indices (row=r, col=c) inside that zone.
    # x is the row-axis coordinate and y is the column-axis coordinate: a full site is (x(r), y(c)).
    #
    # Common precalculated scalars (er, ec = entangling shape rows/cols; sr, sc = storage shape rows/cols):
    #   d_er, d_ec = distance_entangle[0], distance_entangle[1]
    #   d_sr, d_sc = distance_storage[0], distance_storage[1]
    #   inter = distance_interzone, ryd = rydberg_radius
    #   Row offset for storage (below the entangling block, plus one inter-zone gap):
    #     row0_storage = (er - 1) * d_er + inter
    #   Entangling column span (used only to center columns; if ec <= 1 the span is 0):
    #     L_ec = f_ent_col(ec - 1)  where f_ent_col is defined below at integer k
    #   Storage column span (used to center columns):
    #     L_sc = (sc - 1) * d_sc
    #
    # Entangling zone, site (r, c) with 0 <= r < er and 0 <= c < ec:
    #   x_ent(r) = r * d_er
    #   y_ent(c) = f_ent_col(c) - L_ec / 2
    #   At integer column k, f_ent_col(k) = (k // 2) * d_ec + (k % 2) * ryd, i.e. 0, ryd, d_ec, d_ec+ryd, 2*d_ec, ...
    #   For non-integer c, f_ent_col(c) linearly interpolates between f_ent_col(floor(c)) and f_ent_col(floor(c)+1).
    #
    # Storage zone, site (r, c) with 0 <= r < sr and 0 <= c < sc:
    #   x_sto(r) = row0_storage + r * d_sr
    #   y_sto(c) = c * d_sc - L_sc / 2
    return ZNAAConfig(
        shape_storage=(100, 100),
        shape_entangling=(7, 40),
        shape_readout=(20, 20),
        time_1q=0.625, time_2q=0.360, time_readout=500.0, time_transfer=15.0,
        fidelity_1q=0.9997, fidelity_2q=0.995, fidelity_readout=0.998, fidelity_transfer=0.999, fidelity_execution=0.995,
        coherence_time_storage=1e8, coherence_time_else=1.5e6, aod_accelerate=0.00275,
        distance_storage=(3.0, 3.0), distance_entangle=(10.0, 12.0), distance_readout=(6.0, 6.0),
        distance_interzone=10.0, rydberg_radius=2.0, delta=2.0, name="config_testonly"
    )


def config_routing_aware_paper():
    """Hardware configuration used in the routing-aware placement paper's evaluation."""
    return ZNAAConfig(
        shape_storage=(100, 100),
        shape_entangling=(7, 40),
        shape_readout=(20, 20),
        time_1q=0.625,
        time_2q=0.360,
        time_readout=500.0,
        time_transfer=15.0,
        fidelity_1q=0.9997,
        fidelity_2q=0.995,
        fidelity_readout=0.998,
        fidelity_transfer=0.999,
        fidelity_execution=0.995,
        coherence_time_storage=1e8,
        coherence_time_else=1.5e6,
        aod_accelerate=0.00275,
        distance_storage=(3.0, 3.0),
        distance_entangle=(10.0, 12.0),
        distance_readout=(6.0, 6.0),
        distance_interzone=10.0,
        rydberg_radius=2.0,
        delta=2.0,
        name="full_compute_store_architecture",
    )

class Scheduler(AbstractScheduler):
    def _extract_program_from_circuit(self, circuit: ZNAACircuit):
        # Input format: raw gate stream with "1q"/"2q" tags.
        n_q = max(max(g.qubits) for g in circuit.gates) + 1
        g_q, dict_g_1q_parent = [], {-1: []}
        last_2q = [-1] * n_q
        for gate in circuit.gates:
            if gate.gate_type == "2q":
                q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
                idx = len(g_q)
                last_2q[q0] = idx
                last_2q[q1] = idx
                g_q.append([q0, q1] if q0 < q1 else [q1, q0])
                continue
            if gate.gate_type == "1q":
                q0 = int(gate.qubits[0])
                dict_g_1q_parent.setdefault(last_2q[q0], []).append((str(gate.gate_name).lower(), q0))
        return g_q, dict_g_1q_parent, n_q

    @staticmethod
    def _asap(g_q: list[list[int]], n_q: int) -> list[list[int]]:
        layers, q_time = [], [0] * n_q
        for gate_idx, (q0, q1) in enumerate(g_q):
            layer = max(q_time[q0], q_time[q1])
            if layer >= len(layers):
                layers.append([])
            layers[layer].append(gate_idx)
            q_time[q0] = layer + 1
            q_time[q1] = layer + 1
        return layers

    @staticmethod
    def _max_gate_num_from_shape(shape) -> int:
        er, ec = max(1, int(shape.entangling[0])), max(1, int(shape.entangling[1]))
        return max(1, er * max(1, ec // 2))

    @staticmethod
    def _split_layers_by_capacity(gate_scheduling_idx: list[list[int]], max_gate_num: int) -> list[list[int]]:
        out = []
        for gates in gate_scheduling_idx:
            if len(gates) < max_gate_num:
                out.append(gates)
                continue
            num_layer = (len(gates) + max_gate_num - 1) // max_gate_num
            gates_per_layer = (len(gates) + num_layer - 1) // num_layer
            for i in range(0, len(gates), gates_per_layer):
                out.append(gates[i:i + gates_per_layer])
        return out

    @staticmethod
    def _append_1q_stage(stages: list[ZNAAStage], one_q: list[tuple[str, int]]) -> None:
        if one_q:
            stages.append(ZNAAStage("1q", [ZNAAGate("1q", name, [q]) for name, q in one_q]))

    def schedule(self, circuit: ZNAACircuit) -> list[ZNAAStage]:
        # Input format: ZNAACircuit with ordered mixed gates.
        if not circuit.gates:
            return []
        g_q, one_q_parent, n_q = self._extract_program_from_circuit(circuit)
        stages: list[ZNAAStage] = []
        self._append_1q_stage(stages, one_q_parent.get(-1, []))
        if not g_q:
            return stages
        layers = self._split_layers_by_capacity(self._asap(g_q, n_q), self._max_gate_num_from_shape(self.config))
        for layer in layers:
            two_q = [ZNAAGate("2q", "CZ", list(g_q[i])) for i in layer]
            if two_q:
                stages.append(ZNAAStage("2q", two_q))
            self._append_1q_stage(stages, [g for i in layer for g in one_q_parent.get(i, [])])
        return stages

        

class ReuseAnalyzer(AbstractReuseAnalyzer):
    def analyze(self, stages: list[ZNAAStage]) -> list[list[int]]:
        # Input format: stage list; only non-empty 2Q stages participate.
        stages_2q, gate_scheduling = self._build_gate_scheduling(stages)
        if not stages_2q:
            return [[] for _ in stages]
        n_q = max(max(g.qubits[0], g.qubits[1]) for s in stages_2q for g in s.gates) + 1
        return self._expand_reuse_to_stage_order(stages, self._collect_reuse_qubit(gate_scheduling, n_q))

    @staticmethod
    def _build_gate_scheduling(stages: list[ZNAAStage]) -> tuple[list[ZNAAStage], list[list[list[int]]]]:
        stages_2q = [s for s in stages if s.stage_type == "2q" and s.gates]
        return stages_2q, [[[min(g.qubits[0], g.qubits[1]), max(g.qubits[0], g.qubits[1])] for g in s.gates] for s in stages_2q]

    def _collect_reuse_qubit(
        self, gate_scheduling: list[list[list[int]]], n_q: int
    ) -> list[set[int]]:
        if not gate_scheduling:
            return []
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import maximum_bipartite_matching
        except ImportError as e:
            raise ImportError(
                "ZACIndependentReuseAnalyzer requires scipy for bipartite matching."
            ) from e

        reuse_qubit: list[set[int]] = []
        qubit_is_used = [[-1 for _ in range(n_q)] for _ in range(len(gate_scheduling))]
        for gate_idx, gate in enumerate(gate_scheduling[0]):
            for q in gate:
                qubit_is_used[0][q] = gate_idx

        extra_reuse_qubit = 0
        for i in range(1, len(gate_scheduling)):
            reuse_qubit.append(set())
            matrix = [
                [0 for _ in range(len(gate_scheduling[i - 1]))]
                for _ in range(len(gate_scheduling[i]))
            ]
            for gate_idx, gate in enumerate(gate_scheduling[i]):
                if (
                    qubit_is_used[i - 1][gate[0]] != -1
                    and qubit_is_used[i - 1][gate[0]] == qubit_is_used[i - 1][gate[1]]
                ):
                    reuse_qubit[-1].add(gate[0])
                    reuse_qubit[-1].add(gate[1])
                else:
                    for q in gate:
                        if qubit_is_used[i - 1][q] > -1:
                            matrix[gate_idx][qubit_is_used[i - 1][q]] = 1
                            extra_reuse_qubit += 1
                for q in gate:
                    qubit_is_used[i][q] = gate_idx

            sparse_matrix = csr_matrix(matrix)
            matching = maximum_bipartite_matching(sparse_matrix, perm_type="column")
            for gate_idx, reuse_gate in enumerate(matching):
                if reuse_gate == -1:
                    continue
                extra_reuse_qubit -= 1
                gate = gate_scheduling[i][gate_idx]
                for q in gate:
                    if qubit_is_used[i - 1][q] == reuse_gate:
                        reuse_qubit[-1].add(q)

        assert extra_reuse_qubit >= 0
        reuse_qubit.append(set())
        return reuse_qubit

    @staticmethod
    def _expand_reuse_to_stage_order(stages: list[ZNAAStage], reuse_list: list[set[int]]) -> list[list[int]]:
        out, idx_2q = [], 0
        for stage in stages:
            if stage.stage_type == "1q" or not stage.gates:
                out.append([])
                continue
            out.append(sorted(reuse_list[idx_2q]) if idx_2q < len(reuse_list) else [])
            idx_2q += 1
        return out


class Router(AbstractRouter):
    @staticmethod
    def _compatible_2d(a, b):
        for i in (0, 2):
            s0, d0, s1, d1 = a[i], a[i + 1], b[i], b[i + 1]
            if (s0 == s1 and d0 != d1) or (d0 == d1 and s0 != s1) or (s0 < s1 and d0 >= d1) or (s0 > s1 and d0 <= d1):
                return False
        return True

    def _to_map(self, pl: list[tuple[int, int, int]], layout: dict):
        sid, ent, nrs = layout["storage"], layout["ent"], layout["slm"][layout["storage"]]["n_r"]
        out = []
        for zone, row, col in pl:
            if zone == 0:
                out.append((sid, nrs - 1 - row, col))
            else:
                slm = ent[col % 2]
                out.append((slm, layout["slm"][slm]["n_r"] - 1 - row, col // 2))
        return out

    def _get_offset(self, z):
        if z == 0:
            return (1/self.config.distance_storage[0], 1/self.config.distance_storage[1])
        return (1/self.config.distance_entangle[0], 1/(self.config.distance_entangle[1]-self.config.rydberg_radius))

    def _append_ops(self, ops: list[ZNAAOperation], src_pl: list[tuple[int, int, int]], dst_pl: list[tuple[int, int, int]], moved: list[int]):
        zone_name = {0: "storage", 1: "entangling"}
        grouped: dict[tuple[int, int], list[int]] = {}
        for q in moved:
            grouped.setdefault((int(src_pl[q][0]), int(dst_pl[q][0])), []).append(q)
        for (sz, dz), ids in grouped.items():
            offset = self._get_offset(sz)
            if not ids:
                continue

            # Build src->dst one-to-one maps for row/column lines.
            row_to_dst: dict[int, int] = {}
            col_to_dst: dict[int, int] = {}
            by_src_row: dict[int, list[int]] = {}
            for q in ids:
                sr = int(src_pl[q][1])
                sc = int(src_pl[q][2])
                dr = int(dst_pl[q][1])
                dc = int(dst_pl[q][2])
                if sr in row_to_dst and row_to_dst[sr] != dr:
                    raise RuntimeError(f"row mapping conflict for src row {sr}: {row_to_dst[sr]} vs {dr}")
                if sc in col_to_dst and col_to_dst[sc] != dc:
                    raise RuntimeError(f"col mapping conflict for src col {sc}: {col_to_dst[sc]} vs {dc}")
                row_to_dst[sr] = dr
                col_to_dst[sc] = dc
                by_src_row.setdefault(sr, []).append(int(q))

            # Deterministic classification/sort by current source placement.
            src_rows_sorted = sorted(by_src_row.keys())
            for sr in src_rows_sorted:
                by_src_row[sr].sort(key=lambda qid: int(src_pl[qid][2]))

            active_rows: list[int] = []
            active_cols: list[int] = []
            cur_row_pos: dict[int, float] = {}
            cur_col_pos: dict[int, float] = {}
            base_col_pos: dict[int, float] = {}

            for i, sr in enumerate(src_rows_sorted):
                batch_ids = by_src_row[sr]
                batch_cols = sorted({int(src_pl[q][2]) for q in batch_ids})

                # One pick batch: open one row each time (plus any new columns needed now).
                open_rows = [float(sr)]
                open_cols = [float(c) for c in batch_cols if c not in cur_col_pos]
                ops.append(
                    Operation_Open(
                        rows=open_rows,
                        columns=open_cols,
                        pick_ids={int(q) for q in batch_ids},
                        open_zone=zone_name[sz],
                    )
                )

                active_rows.append(sr)
                cur_row_pos[sr] = float(sr)
                for c in batch_cols:
                    if c not in cur_col_pos:
                        active_cols.append(c)
                        cur_col_pos[c] = float(c)
                        base_col_pos[c] = float(c)

                next_cols: set[int] = set()
                if i + 1 < len(src_rows_sorted):
                    nr = src_rows_sorted[i + 1]
                    next_cols = {int(src_pl[q][2]) for q in by_src_row[nr]}

                    # After each pick, move existing AOD:
                    # - newly opened row goes to current + 0.5
                    # - columns toggle between base and base+0.5 by next-step usage.
                    row_targets: list[float] = []
                    for rkey in active_rows:
                        cur = cur_row_pos[rkey]
                        tgt = cur + offset[0] if rkey == sr else cur
                        row_targets.append(tgt)
                        cur_row_pos[rkey] = tgt

                    col_targets: list[float] = []
                    for ckey in active_cols:
                        base = base_col_pos[ckey]
                        if sz == 0:
                            tgt = base if ckey in next_cols else base + offset[1]
                        else:
                            tgt = base if ckey in next_cols else base + offset[1] * (base % 2 - 0.5) * 2
                        col_targets.append(tgt)
                        cur_col_pos[ckey] = tgt

                    ops.append(
                        Operation_Move(
                            target_zone=zone_name[sz],
                            target_rows=sorted(row_targets),
                            target_columns=sorted(col_targets),
                        )
                    )

            # Final align to destination line coordinates, then drop.
            final_rows = [float(row_to_dst[r]) for r in active_rows]
            final_cols = [float(col_to_dst[c]) for c in active_cols]
            ops.append(
                Operation_Move(
                    target_zone=zone_name[dz],
                    target_rows=sorted(final_rows),
                    target_columns=sorted(final_cols),
                )
            )
            ops.append(Operation_Close(rows=sorted(final_rows), columns=sorted(final_cols), close_zone=zone_name[dz]))

    def route(self, placements: list[list[tuple[int, int, int]]]) -> list[list[ZNAAOperation]]:
        if len(placements) <= 1:
            return []
        layout, mapped = self.config.build_zone_layout(), []
        for pl in placements:
            mapped.append(self._to_map(pl, layout))
        out = [[] for _ in range(len(placements) - 1)]
        for i in range(len(mapped) - 1):
            src_m, dst_m, remain = mapped[i], mapped[i + 1], [q for q in range(len(mapped[i])) if mapped[i][q] != mapped[i + 1][q]]
            while remain:
                vec = []
                for q in remain:
                    qx, qy = self.config.slm_site_xy(src_m[q])
                    sx, sy = self.config.slm_site_xy(dst_m[q])
                    vec.append((qx, sx, qy, sy))
                pick = []
                for j, cur in enumerate(vec):
                    if all(self._compatible_2d(cur, vec[k]) for k in pick):
                        pick.append(j)
                moved = [remain[j] for j in pick]
                self._append_ops(out[i], placements[i], placements[i + 1], moved)
                used = set(moved)
                remain = [q for q in remain if q not in used]
        return out

