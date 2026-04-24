"""Task-local slot workspace management for qubit-routing evaluators.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
_DEFAULT_SLOT_ROOT = "/tmp/simpletes-qubit-routing-slots"
_NAMESPACE_PREFIX = "se-qubit_routing-"


class QubitRoutingSlotWorkspace:
    """Manage a run-scoped qubit-routing evaluator slot namespace."""

    def __init__(
        self,
        *,
        evaluator_path: str,
        instance_id: str,
        eval_concurrency: int,
        eval_timeout: float,
        log_message: Callable[[str, str], str],
        printer: Callable[[Any], None],
    ) -> None:
        self._log_message = log_message
        self._printer = printer
        self.enabled = self._resolve_evaluator_path(evaluator_path).is_relative_to(ROOT)

        self._slot_root = Path(
            os.environ.get("QUBIT_ROUTING_SLOT_ROOT", _DEFAULT_SLOT_ROOT)
        ).expanduser()
        self._cleanup_dir: Path | None = None
        self._cleanup_done = False
        self._auto_cleanup = os.environ.get("QUBIT_ROUTING_SLOT_AUTO_CLEANUP", "1") != "0"
        self._startup_janitor = os.environ.get("QUBIT_ROUTING_SLOT_STARTUP_JANITOR", "1") != "0"
        self._stale_ttl_seconds = float(
            os.environ.get("QUBIT_ROUTING_SLOT_STALE_TTL_SECONDS", "86400")
        )
        self._namespace_lock_fd: int | None = None

        if not self.enabled:
            return

        if "QUBIT_ROUTING_SLOT_NAMESPACE" not in os.environ:
            run_slot_namespace = (
                f"{_NAMESPACE_PREFIX}{instance_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            os.environ["QUBIT_ROUTING_SLOT_NAMESPACE"] = run_slot_namespace
            self._cleanup_dir = self._slot_root / run_slot_namespace
            self._cleanup_stale_namespaces()
            self._acquire_namespace_lock()

        # Task-local hint consumed by the qubit-routing evaluator for slot fallback.
        os.environ.setdefault("QUBIT_ROUTING_SLOT_COUNT", str(eval_concurrency))
        # Total per-evaluation timeout from the SimpleTES evaluator worker.
        os.environ["QUBIT_ROUTING_PARENT_EVAL_TIMEOUT_SECONDS"] = str(eval_timeout)

    @staticmethod
    def _resolve_evaluator_path(evaluator_path: str) -> Path:
        path = Path(evaluator_path).expanduser()
        if not path.is_absolute():
            return (Path.cwd() / path).resolve()
        return path.resolve()

    def cleanup(self) -> None:
        """Clean the auto-created evaluator slot namespace, if any."""
        if not self.enabled or self._cleanup_done:
            return
        self._cleanup_done = True

        slot_dir = self._cleanup_dir
        if slot_dir is None:
            self._release_namespace_lock()
            return
        if not self._auto_cleanup:
            self._emit("📁", f"[dim]Keeping evaluator slot workspace: {slot_dir}[/dim]")
            self._release_namespace_lock()
            return

        try:
            shutil.rmtree(slot_dir, ignore_errors=True)
            self._emit("🧹", f"[dim]Cleaned evaluator slot workspace: {slot_dir}[/dim]")
        except Exception as exc:
            self._emit("⚠", f"[yellow]Failed to clean evaluator slot workspace: {exc}[/yellow]")
        finally:
            self._release_namespace_lock()

    def _cleanup_stale_namespaces(self) -> None:
        """Delete stale auto-created slot namespaces from previous hard-killed runs."""
        if not self._startup_janitor:
            return
        if self._stale_ttl_seconds <= 0:
            return
        try:
            if not self._slot_root.is_dir():
                return
            now = time.time()
            removed = 0
            for entry in self._slot_root.iterdir():
                if not entry.is_dir():
                    continue
                if not entry.name.startswith(_NAMESPACE_PREFIX):
                    continue
                if self._cleanup_dir is not None and entry == self._cleanup_dir:
                    continue
                try:
                    if now - entry.stat().st_mtime < self._stale_ttl_seconds:
                        continue
                except FileNotFoundError:
                    continue

                lock_fd: int | None = None
                try:
                    lock_path = entry / ".namespace.lock"
                    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
                except Exception:
                    continue
                finally:
                    if lock_fd is not None:
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        except OSError:
                            pass
                        os.close(lock_fd)

            if removed > 0:
                self._emit(
                    "🧹",
                    (
                        "[dim]Startup janitor removed "
                        f"{removed} stale evaluator slot namespace(s) older than "
                        f"{int(self._stale_ttl_seconds)}s[/dim]"
                    ),
                )
        except Exception as exc:
            self._emit("⚠", f"[yellow]Evaluator slot startup janitor failed: {exc}[/yellow]")

    def _acquire_namespace_lock(self) -> None:
        """Hold a namespace-level lock for the full run to mark it as active."""
        if self._cleanup_dir is None:
            return
        try:
            self._cleanup_dir.mkdir(parents=True, exist_ok=True)
            lock_path = self._cleanup_dir / ".namespace.lock"
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._namespace_lock_fd = fd
        except Exception as exc:
            self._emit("⚠", f"[yellow]Failed to acquire evaluator namespace lock: {exc}[/yellow]")

    def _release_namespace_lock(self) -> None:
        fd = self._namespace_lock_fd
        if fd is None:
            return
        self._namespace_lock_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _emit(self, icon: str, msg: str) -> None:
        self._printer(self._log_message(icon, msg))

