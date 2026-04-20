"""
Shared construction helpers for prompt-time summaries and eval-time globals.

The shared artifact is stored as JSON with a small tagged schema so it can be
round-tripped safely across processes.
"""
from __future__ import annotations

import builtins
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from simpletes.utils.text import clip_text

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is available in normal runtime
    np = None  # type: ignore[assignment]


GLOBAL_NAME = "GLOBAL_BEST_CONSTRUCTION"
SHARED_CONSTRUCTION_ENV = "SIMPLETES_SHARED_CONSTRUCTION_PATH"
CAPTURE_CONSTRUCTION_ENV = "SIMPLETES_CAPTURE_CONSTRUCTION_PATH"
MAX_SNAPSHOT_BYTES_ENV = "SIMPLETES_SHARED_CONSTRUCTION_MAX_BYTES"

_TAG_KEY = "__simpletes_type__"
# Legacy tag from the simpleevolve era; decoder still honors it so that
# pre-rename JSON artifacts (e.g. those shipped in best_results/) round-trip.
# The encoder always writes the current tag, so new files never use this.
_LEGACY_TAG_KEY = "__simpleevolve_type__"
_TAG_NDARRAY = "ndarray"
_TAG_TUPLE = "tuple"
_DEFAULT_MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
_SUMMARY_MAX_CHARS = 1200
_PREVIEW_ITEMS = 6


def max_snapshot_bytes() -> int:
    raw = os.environ.get(MAX_SNAPSHOT_BYTES_ENV)
    if not raw:
        return _DEFAULT_MAX_SNAPSHOT_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SNAPSHOT_BYTES
    return max(1024, value)


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, str))


def _normalize_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite float in construction: {value!r}")
    return float(value)


def encode_construction(value: Any) -> Any:
    """Convert a construction artifact into a JSON-compatible payload."""
    if _is_primitive(value):
        return value

    if isinstance(value, float):
        return _normalize_float(value)

    if np is not None:
        if isinstance(value, np.generic):
            scalar = value.item()
            return encode_construction(scalar)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise ValueError("object-dtype numpy arrays are not supported")
            if value.dtype.kind == "c":
                raise ValueError("complex numpy arrays are not supported")
            if not np.all(np.isfinite(value)) and value.dtype.kind in {"f", "c"}:
                raise ValueError("non-finite values in numpy construction")
            return {
                _TAG_KEY: _TAG_NDARRAY,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }

    if isinstance(value, tuple):
        return {_TAG_KEY: _TAG_TUPLE, "items": [encode_construction(v) for v in value]}

    if isinstance(value, list):
        return [encode_construction(v) for v in value]

    if isinstance(value, dict):
        return {str(k): encode_construction(v) for k, v in value.items()}

    if hasattr(value, "tolist"):
        return encode_construction(value.tolist())

    raise ValueError(f"unsupported construction type: {type(value).__name__}")


def decode_construction(payload: Any) -> Any:
    """Restore a JSON payload to Python/numpy objects."""
    if isinstance(payload, list):
        return [decode_construction(v) for v in payload]

    if not isinstance(payload, dict):
        return payload

    tag = payload.get(_TAG_KEY, payload.get(_LEGACY_TAG_KEY))
    if tag == _TAG_NDARRAY:
        data = payload.get("data")
        if np is None:
            return data
        return np.asarray(data, dtype=payload.get("dtype"))
    if tag == _TAG_TUPLE:
        items = payload.get("items", [])
        return tuple(decode_construction(v) for v in items)

    return {k: decode_construction(v) for k, v in payload.items()}


def payload_to_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def write_payload(path: str | Path, payload: Any) -> int:
    """Write payload atomically. Returns encoded size in bytes."""
    text = payload_to_json(payload)
    directory = os.path.dirname(os.fspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".shared_ctx_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, os.fspath(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return len(text.encode("utf-8"))


def read_payload(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def capture_construction_if_requested(value: Any) -> bool:
    """Capture a validated construction artifact when the evaluator asks for it."""
    path = os.environ.get(CAPTURE_CONSTRUCTION_ENV)
    if not path:
        return False

    try:
        payload = encode_construction(value)
        text = payload_to_json(payload)
    except Exception:
        return False

    if len(text.encode("utf-8")) > max_snapshot_bytes():
        return False

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".capture_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False
    return True


def load_shared_construction_from_env() -> Any:
    path = os.environ.get(SHARED_CONSTRUCTION_ENV)
    if not path or not os.path.exists(path):
        return None
    try:
        return decode_construction(read_payload(path))
    except Exception:
        return None


def install_global_from_env() -> None:
    setattr(builtins, GLOBAL_NAME, load_shared_construction_from_env())


def summarize_construction_payload(payload: Any, *, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    try:
        value = decode_construction(payload)
    except Exception:
        return ""
    return summarize_construction(value, max_chars=max_chars)


def summarize_construction(value: Any, *, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    if value is None:
        return "None"

    if np is not None and isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        preview = flat[:_PREVIEW_ITEMS].tolist()
        summary = (
            f"numpy.ndarray(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"preview={preview})"
        )
        return clip_text(summary, max_chars)

    if isinstance(value, tuple):
        items = ", ".join(
            summarize_construction(v, max_chars=max(40, max_chars // 4))
            for v in value[:_PREVIEW_ITEMS]
        )
        suffix = ", ..." if len(value) > _PREVIEW_ITEMS else ""
        return clip_text(f"tuple(len={len(value)}, items=[{items}{suffix}])", max_chars)

    if isinstance(value, list):
        if value and all(isinstance(v, (bool, int, float, str)) or v is None for v in value[:_PREVIEW_ITEMS]):
            preview = value[:_PREVIEW_ITEMS]
            suffix = ", ..." if len(value) > _PREVIEW_ITEMS else ""
            return clip_text(f"list(len={len(value)}, preview={preview}{suffix})", max_chars)
        items = ", ".join(
            summarize_construction(v, max_chars=max(40, max_chars // 4))
            for v in value[:_PREVIEW_ITEMS]
        )
        suffix = ", ..." if len(value) > _PREVIEW_ITEMS else ""
        return clip_text(f"list(len={len(value)}, items=[{items}{suffix}])", max_chars)

    if isinstance(value, dict):
        keys = list(value.keys())
        preview_items = []
        for key in keys[:_PREVIEW_ITEMS]:
            item_summary = summarize_construction(value[key], max_chars=max(40, max_chars // 4))
            preview_items.append(f"{key}={item_summary}")
        suffix = ", ..." if len(keys) > _PREVIEW_ITEMS else ""
        return clip_text(
            f"dict(keys={keys[:_PREVIEW_ITEMS]}{suffix}, preview={{{', '.join(preview_items)}}})",
            max_chars,
        )

    return clip_text(repr(value), max_chars)
