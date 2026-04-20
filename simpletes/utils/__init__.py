"""Small, dependency-light helpers shared across the package.

Submodules:
- ``text``     — clip / normalize / summarize; metrics → text; Harmony token
- ``log``      — visible_len / padding_for / format_log / TeeStream
- ``task_prep`` — task-directory data prep + requirements.txt parsing
- ``code_extract`` — EVOLVE-BLOCK parsing (added in a later commit)
"""
from simpletes.utils.log import format_log, install_tee_logger, padding_for
from simpletes.utils.text import (
    DEFAULT_METRICS_ERROR_MAX_CHARS,
    clip_text,
    extract_approach_insight,
    metrics_to_text,
    summarize_error,
    truncate_error_in_metrics,
)

__all__ = [
    "DEFAULT_METRICS_ERROR_MAX_CHARS",
    "clip_text",
    "extract_approach_insight",
    "format_log",
    "install_tee_logger",
    "metrics_to_text",
    "padding_for",
    "summarize_error",
    "truncate_error_in_metrics",
]
