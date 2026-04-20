"""Logging helpers for Rich-formatted output and file tee-ing.

This module centralizes small utilities used across the codebase:
- Rich markup-aware padding helpers for aligned console output
  (used by checkpointing + engine panels)
- A stdout tee stream that mirrors output to a plain-text log file
  (used by the engine to write `run.log`)
"""
from __future__ import annotations

import io
import os
import re
import sys
import threading
from datetime import datetime

_RICH_MARKUP_RE = re.compile(r"\[/?[^\]]+\]")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def visible_len(text: str) -> int:
    """Return visible length after stripping Rich markup tags."""
    return len(_RICH_MARKUP_RE.sub("", text))


def padding_for(text: str, width: int = 50) -> str:
    """Return padding spaces to reach a target visible width."""
    return " " * max(0, width - visible_len(text))


def format_log(icon: str = "", msg: str = "", status: str = "", *, prefix: str = "", width: int = 50) -> str:
    """Padded log line. Default prefix is the current time; callers can pass any prefix."""
    if not prefix:
        prefix = f"({datetime.now().strftime('%H:%M:%S')})"
    icon_part = f"{icon} " if icon else ""
    status_part = f" [{status}]" if status else ""
    return f"{prefix} {icon_part}{msg}{padding_for(msg, width=width)}{status_part}"


class TeeStream(io.TextIOBase):
    """A write-through stream that duplicates output to a log file.

    Wraps an original text stream (typically sys.__stdout__) and mirrors
    every write() call to a plain-text log file with ANSI codes stripped.
    All other attributes are proxied to the original stream so that
    libraries like Rich can query terminal capabilities normally.
    """

    def __init__(self, original: io.TextIOBase, log_path: str) -> None:
        self._original = original
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    # -- io.TextIOBase write interface --

    def write(self, s: str) -> int:
        # Write to original stream (console)
        result = self._original.write(s)
        # Write ANSI-stripped version to log file
        plain = _ANSI_RE.sub("", s)
        with self._lock:
            self._log_file.write(plain)
            self._log_file.flush()
        return result

    def flush(self) -> None:
        self._original.flush()
        with self._lock:
            self._log_file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._log_file.closed:
                self._log_file.close()

    # -- Proxy everything else to the original stream --

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:
        return getattr(self._original, "errors", None)

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def name(self) -> str:
        return getattr(self._original, "name", "<tee>")

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def install_tee_logger(log_path: str) -> TeeStream:
    """Install a tee logger that duplicates stdout to *log_path*.

    Patches three things so all output is captured:
      1. ``sys.stdout`` — regular ``print()`` calls
      2. ``sys.__stdout__`` — fallback original stdout
      3. The Rich ``Console._file`` inside ``rich_print`` (bound at import time
         in ``evaluator.py``) so Rich markup output is also tee'd.

    Returns the TeeStream instance (useful for cleanup / close).
    """
    tee = TeeStream(sys.__stdout__, log_path)
    sys.__stdout__ = tee  # type: ignore[assignment]
    sys.stdout = tee  # type: ignore[assignment]

    # Patch the already-bound Rich Console used by rich_print.
    # rich_print = Console(file=sys.__stdout__).print  — the Console captured
    # the old sys.__stdout__ object at import time.  We reach into it via
    # the bound method's __self__ and swap its _file to our tee stream.
    try:
        from simpletes.evaluator import rich_print as _rp
        console = getattr(_rp, "__self__", None)
        if console is not None:
            console._file = tee
    except Exception:
        pass  # non-critical; regular stdout tee still works

    return tee
