"""Problem configuration for impulsive MGA mission design.

Provides physical constants for all eight planets and deterministic
single-instance problem construction.  Mission JSON files live beside their
task entry points rather than in one environment-selected instance pool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# km^3 / s^2
MU_SUN = 1.32712440018e11

MU: dict[str, float] = {
    "1": 2.2032e4,        # Mercury
    "2": 3.24859e5,       # Venus
    "3": 3.986004418e5,   # Earth
    "4": 4.282837e4,      # Mars
    "5": 1.26686534e8,    # Jupiter
    "6": 3.7931187e7,     # Saturn
    "7": 5.793939e6,      # Uranus
    "8": 6.836529e6,      # Neptune
}

# km
RADIUS: dict[str, float] = {
    "1": 2440.0,          # Mercury
    "2": 6051.8,          # Venus
    "3": 6378.137,        # Earth
    "4": 3396.2,          # Mars
    "5": 71492.0,         # Jupiter
    "6": 60268.0,         # Saturn
    "7": 25559.0,         # Uranus
    "8": 24764.0,         # Neptune
}

def load_instance(path: str | Path) -> dict[str, Any]:
    """Load a single instance JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_problem(instance: dict[str, Any]) -> dict[str, Any]:
    """Build the evaluator-facing problem dict for one mission instance."""
    return {
        "mu_sun": MU_SUN,
        "planet_mu": dict(MU),
        "planet_radius": dict(RADIUS),
        "id": instance["id"],
        "description": instance["description"],
        "mission_name": instance.get("mission_name", instance["id"]),
        "mission_class": instance.get("mission_class", instance.get("category", "")),
        "design_goal": instance.get("design_goal", ""),
        "reference_family": instance.get("reference_family", ""),
        "start": instance["start"],
        "end": instance["end"],
        "allowed_GA_planets": instance["allowed_GA_planets"],
        "flyby": instance.get("flyby", {}),
        "max_nodes": instance["max_nodes"],
        "max_GA": instance["max_GA"],
        "max_DSM": instance["max_DSM"],
        "target_dv": float(instance.get("target_dv", 1.0)),
        "timeout_seconds": float(instance.get("timeout_seconds", 60)),
    }


def load_problem_for_candidate(candidate_file: str | Path) -> dict[str, Any]:
    """Resolve the instance used while importing a candidate program.

    The evaluator sets ``SIMPLETES_ASTRODYNAMICS_INSTANCE`` because evolved
    candidates execute from temporary directories.  Direct execution of a
    checked-out seed falls back to the sibling ``instance.json``.
    """
    configured = os.environ.get("SIMPLETES_ASTRODYNAMICS_INSTANCE")
    if configured:
        instance_path = Path(configured).expanduser()
    else:
        candidate_path = Path(candidate_file).resolve()
        adjacent = candidate_path.with_name("instance.json")
        family_instance = Path(__file__).resolve().parent / candidate_path.parent.name / "instance.json"
        instance_path = adjacent if adjacent.is_file() else family_instance
    return build_problem(load_instance(instance_path))


def default_flyby_altitude(instance: dict[str, Any], planet_id: str) -> float:
    """Return min flyby altitude (km) for *planet_id*, default 200."""
    flyby = instance.get("flyby", {}).get("min_altitude_km", {})
    return float(flyby.get(str(planet_id), 200.0))
