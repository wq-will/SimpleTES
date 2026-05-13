"""GPUKernel task utilities vendored into k-search."""

from pathlib import Path

DEFAULT_TRIMUL_TASK_DIR = Path(__file__).resolve().parent / "trimul"
DEFAULT_ASYMMETRICMATMUL_TASK_DIR = Path(__file__).resolve().parent / "asymmetricmatmul"
DEFAULT_CUMSUM_TASK_DIR = Path(__file__).resolve().parent / "cumsum"

__all__ = [
    "DEFAULT_TRIMUL_TASK_DIR",
    "DEFAULT_ASYMMETRICMATMUL_TASK_DIR",
    "DEFAULT_CUMSUM_TASK_DIR",
]
