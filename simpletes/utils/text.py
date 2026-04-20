"""
Small text helpers used across SimpleTES.

We keep these utilities dependency-free to avoid import cycles.
"""
from __future__ import annotations

import json
import re
from typing import Any


_WS_RE = re.compile(r"\s+")

DEFAULT_METRICS_ERROR_MAX_CHARS = 4000

HARMONY_MESSAGE_TOKEN = "<|message|>"


def metrics_to_text(metrics: dict[str, Any]) -> str:
    try:
        return json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return str(metrics)


def extract_approach_insight(text: str) -> str:
    """Extract Approach/Insight reflection body, stripping Harmony message prefix."""
    if not text:
        return ""
    if HARMONY_MESSAGE_TOKEN in text:
        text = text.rsplit(HARMONY_MESSAGE_TOKEN, 1)[-1].strip()
    else:
        text = text.strip()
    if not text.startswith("Approach"):
        return ""
    return text


def clip_text(text: str, max_chars: int, *, suffix: str = "...[truncated]") -> str:
    """Hard-truncate *text* to at most *max_chars* characters (prefix-preserving)."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if len(suffix) >= max_chars:
        return text[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def clip_text_middle(text: str, max_chars: int, *, marker: str = "\n...[truncated]...\n") -> str:
    """Hard-truncate while preserving both start and end (useful for tracebacks)."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if len(marker) >= max_chars:
        return text[:max_chars]

    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) into single spaces."""
    return _WS_RE.sub(" ", text).strip()


def summarize_error(text: str, max_chars: int) -> str:
    """Return a compact, single-line summary suitable for prompts/logs."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]
    last = lines[-1]
    if first.lower().startswith("traceback") and len(lines) > 1:
        summary = last
    else:
        summary = last if first == last else f"{first} | {last}"

    return clip_text(normalize_whitespace(summary), max_chars)


def truncate_error_in_metrics(metrics: Any, *, max_chars: int) -> Any:
    """In-place truncate metrics['error'] if present to avoid huge checkpoints/prompts."""
    if not isinstance(metrics, dict):
        return metrics
    if "error" not in metrics or metrics["error"] is None:
        return metrics

    # Always coerce to string; evaluator errors can contain newlines/tracebacks.
    err_text = str(metrics["error"])
    metrics["error"] = clip_text_middle(err_text, max_chars)
    return metrics
