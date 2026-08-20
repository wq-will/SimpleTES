"""Trajectory node schema and structural validation.

Unified node format for all four types:
``{type, time, planet_id, r, v_before, v_after}``

planet_id = "0" for DSM and state-reference boundary nodes.
planet_id = "1"-"8" for planet-bound nodes (Mercury through Neptune).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

NODE_START = "start"
NODE_END = "end"
NODE_GA = "GA"
NODE_DSM = "DSM"
NODE_TYPES = (NODE_START, NODE_END, NODE_GA, NODE_DSM)

_VALID_PLANET_IDS = frozenset(str(i) for i in range(9))  # "0" through "8"

# All node types share the same vector and scalar fields.
_VEC_FIELDS = ("r", "v_before", "v_after")
_SCALAR_FIELDS = ("type", "time", "planet_id")


class SchemaError(ValueError):
    """Raised when a trajectory violates the structural contract."""


def _check_vec3(name: str, idx: int, value: Any) -> None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise SchemaError(
            f"node {idx}: field {name!r} must be shape (3,), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise SchemaError(
            f"node {idx}: field {name!r} contains non-finite values"
        )


def _check_node(idx: int, node: Any) -> str:
    if not isinstance(node, Mapping):
        raise SchemaError(
            f"node {idx}: expected dict, got {type(node).__name__}"
        )
    if "type" not in node:
        raise SchemaError(f"node {idx}: missing 'type'")
    ntype = node["type"]
    if ntype not in NODE_TYPES:
        raise SchemaError(
            f"node {idx}: unknown type {ntype!r}; valid={NODE_TYPES}"
        )

    # Scalar fields
    for key in _SCALAR_FIELDS:
        if key not in node:
            raise SchemaError(
                f"node {idx} ({ntype}): missing field {key!r}"
            )
    t = node["time"]
    if not (isinstance(t, (int, float)) and np.isfinite(t)):
        raise SchemaError(
            f"node {idx} ({ntype}): 'time' must be a finite scalar, got {t!r}"
        )

    # planet_id must be a string
    pid = node["planet_id"]
    if not isinstance(pid, str):
        raise SchemaError(
            f"node {idx} ({ntype}): 'planet_id' must be a string, got {type(pid).__name__}"
        )
    if pid not in _VALID_PLANET_IDS:
        raise SchemaError(
            f"node {idx} ({ntype}): invalid planet_id {pid!r}; must be one of {sorted(_VALID_PLANET_IDS)}"
        )

    # Vector fields
    for key in _VEC_FIELDS:
        if key not in node:
            raise SchemaError(
                f"node {idx} ({ntype}): missing field {key!r}"
            )
        _check_vec3(key, idx, node[key])

    return ntype


def _validate_boundary_spec_node_consistency(
    node: Mapping[str, Any],
    boundary_spec: Mapping[str, Any],
    node_label: str,
) -> None:
    """Validate that node fields are consistent with the boundary spec.

    Rules:
    - node planet_id MUST match the boundary planet_id
    - periapsis_maneuver: planet_id MUST != "0"
    - piecewise_linear + planet_id == "0": boundary MUST have state_r, state_v;
      time MUST be "exact"
    """
    btype = boundary_spec["type"]
    pid = str(node["planet_id"])
    expected_pid = str(boundary_spec["planet_id"])

    if pid != expected_pid:
        raise SchemaError(
            f"{node_label}: planet_id={pid!r} does not match "
            f"the configured boundary planet_id={expected_pid!r}"
        )

    if btype == "periapsis_maneuver":
        if pid == "0":
            raise SchemaError(
                f"{node_label}: periapsis_maneuver requires planet_id != '0'"
            )

    elif btype == "piecewise_linear":
        if pid == "0":
            ts = boundary_spec.get("time")
            if ts is None or ts.get("kind") != "exact":
                raise SchemaError(
                    f"{node_label}: piecewise_linear with planet_id='0' "
                    f"requires time.kind='exact'"
                )
            if "state_r" not in boundary_spec or "state_v" not in boundary_spec:
                raise SchemaError(
                    f"{node_label}: piecewise_linear with planet_id='0' "
                    f"requires state_r and state_v in boundary spec"
                )
        else:
            # planet_id != "0": time can be window or exact — either is fine
            pass


def validate_schema(
    trajectory: Any, instance: Mapping[str, Any]
) -> None:
    """Validate trajectory structure against *instance* config.

    Raises SchemaError on failure.
    """
    if not isinstance(trajectory, list):
        raise SchemaError(
            f"trajectory must be a list, got {type(trajectory).__name__}"
        )
    if len(trajectory) < 2:
        raise SchemaError(
            f"trajectory must contain at least start+end, got {len(trajectory)} nodes"
        )

    max_nodes = instance["max_nodes"]
    if len(trajectory) > max_nodes:
        raise SchemaError(
            f"trajectory has {len(trajectory)} nodes, exceeds max_nodes={max_nodes}"
        )

    # Validate each node
    types = [_check_node(i, node) for i, node in enumerate(trajectory)]

    # First / last must be start / end
    if types[0] != NODE_START:
        raise SchemaError(
            f"first node must be 'start', got {types[0]!r}"
        )
    if types[-1] != NODE_END:
        raise SchemaError(
            f"last node must be 'end', got {types[-1]!r}"
        )

    # Intermediate nodes must be GA or DSM
    for i, ntype in enumerate(types):
        if i == 0 or i == len(types) - 1:
            continue
        if ntype not in (NODE_GA, NODE_DSM):
            raise SchemaError(
                f"node {i}: intermediate nodes must be GA or DSM, got {ntype!r}"
            )

    # DSM must have planet_id "0"
    for i, ntype in enumerate(types):
        pid = str(trajectory[i]["planet_id"])
        if ntype == NODE_DSM and pid != "0":
            raise SchemaError(
                f"node {i} ({ntype}): planet_id must be '0', got {pid!r}"
            )
        if ntype == NODE_GA and pid == "0":
            raise SchemaError(
                f"node {i} ({ntype}): planet_id must not be '0'"
            )

    # Boundary logic: start/end node consistency with boundary specs
    _validate_boundary_spec_node_consistency(
        trajectory[0], instance["start"], "start node"
    )
    _validate_boundary_spec_node_consistency(
        trajectory[-1], instance["end"], "end node"
    )

    # GA planet_id must be in allowed_GA_planets
    allowed = set(instance.get("allowed_GA_planets", []))
    for i, ntype in enumerate(types):
        if ntype == NODE_GA:
            pid = str(trajectory[i]["planet_id"])
            if pid not in allowed:
                raise SchemaError(
                    f"node {i} (GA): planet_id={pid!r} not in "
                    f"allowed_GA_planets={sorted(allowed)}"
                )

    # Times strictly increasing
    times = [float(n["time"]) for n in trajectory]
    for i in range(1, len(times)):
        if not (times[i] > times[i - 1]):
            raise SchemaError(
                f"time not strictly increasing between node {i - 1} "
                f"(t={times[i - 1]}) and node {i} (t={times[i]})"
            )

    # Node count limits
    ga_count = sum(1 for t in types if t == NODE_GA)
    dsm_count = sum(1 for t in types if t == NODE_DSM)
    max_GA = instance.get("max_GA")
    if max_GA is not None and ga_count > max_GA:
        raise SchemaError(f"GA count {ga_count} exceeds max_GA={max_GA}")
    max_DSM = instance.get("max_DSM")
    if max_DSM is not None and dsm_count > max_DSM:
        raise SchemaError(f"DSM count {dsm_count} exceeds max_DSM={max_DSM}")
