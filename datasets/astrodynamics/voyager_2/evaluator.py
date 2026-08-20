"""Voyager 2 task entry point."""

import sys
from pathlib import Path

_FAMILY_DIR = Path(__file__).resolve().parent.parent
if str(_FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(_FAMILY_DIR))

from evaluator_core import evaluate_instance

_INSTANCE_PATH = Path(__file__).with_name("instance.json")


def evaluate(program_path: str) -> dict:
    return evaluate_instance(program_path, _INSTANCE_PATH)
