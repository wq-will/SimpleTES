"""Core scheduler + runtime + checkpoint for SimpleTES.

The engine package owns the asynchronous Scheduler from
Algorithm~\\ref{alg:async_method}: it dispatches generation and
evaluation work through queues, invokes policy callbacks on batch
completion, and persists intermediate state.

Submodules:
- ``core``       — SimpleTESEngine (the scheduler)
- ``runtime``    — RuntimeBase + LocalRuntime (in-process worker pools)
- ``checkpoint`` — CheckpointManager (save / resume)
"""
from simpletes.engine.core import SimpleTESEngine
from simpletes.engine.checkpoint import CheckpointManager
from simpletes.engine.runtime import LocalRuntime, RuntimeBase

__all__ = [
    "CheckpointManager",
    "LocalRuntime",
    "RuntimeBase",
    "SimpleTESEngine",
]
