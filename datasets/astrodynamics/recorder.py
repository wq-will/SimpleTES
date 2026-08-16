"""Lightweight runtime record for evolved strategies.

The strategy calls set() and event() to accumulate free-form text.
The evaluator calls to_string() after run_code() and truncates the
result to 3K characters before forwarding to the next-generation LLM.

The first set/event call also captures the absolute wall-clock time
as a full ISO 8601 timestamp (with local timezone and ms precision,
e.g. [t0=2026-06-12T10:42:03.456+08:00]) so the LLM can correlate
multiple lineages. Subsequent entries are prefixed with a self-
relative monotonic timestamp to ms precision (e.g. [+1.234s]) so the
LLM can see when in the strategy runtime each event happened.

A 500-entry / 3000-char cap prevents runaway growth; once hit,
further set/event calls become silent no-ops.
"""
import time
from datetime import datetime
from typing import Any

MAX_ENTRIES = 500
MAX_CHARS = 3000


class Record:
    def __init__(self):
        self._parts = []
        self._n_entries = 0
        self._total_chars = 0
        self._t0_mono = time.monotonic()
        self._t0_wall = None
        self._capped = False

    def _stamp(self) -> str:
        dt = time.monotonic() - self._t0_mono
        return f"[+{dt:.3f}s]"

    def _iso_now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _would_exceed(self, fragment: str) -> bool:
        if self._n_entries + 1 > MAX_ENTRIES:
            return True
        added = len(fragment) + (2 if self._n_entries > 0 else 0)
        return self._total_chars + added > MAX_CHARS

    def set(self, key: str, value: Any) -> None:
        if self._t0_wall is None:
            self._t0_wall = self._iso_now()
            frag = f"[t0={self._t0_wall}] {key}={value}"
        else:
            frag = f"{self._stamp()} {key}={value}"
        if self._would_exceed(frag):
            self._capped = True
            return
        self._parts.append(frag)
        self._n_entries += 1
        self._total_chars += len(frag) + (2 if self._n_entries > 1 else 0)

    def event(self, msg: str) -> None:
        if self._t0_wall is None:
            self._t0_wall = self._iso_now()
            frag = f"[t0={self._t0_wall}] {msg}"
        else:
            frag = f"{self._stamp()} {msg}"
        if self._would_exceed(frag):
            self._capped = True
            return
        self._parts.append(frag)
        self._n_entries += 1
        self._total_chars += len(frag) + (2 if self._n_entries > 1 else 0)

    def to_string(self) -> str:
        return "; ".join(self._parts)
