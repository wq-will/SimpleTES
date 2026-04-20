"""GPUMode task utilities vendored into k-search.

The top-level `gpu_mode/` folder in this repo is legacy and may be removed.
New code should depend on `k_search.tasks.gpu_mode.*` instead.
"""

from pathlib import Path

# Canonical location of the vendored TriMul task bundle (task.yml + eval/reference code).
DEFAULT_TRIMUL_TASK_DIR = Path(__file__).resolve().parent / "trimul"

__all__ = ["DEFAULT_TRIMUL_TASK_DIR"]


