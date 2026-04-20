"""
Runtime backends for SimpleTES.

LocalRuntime: in-process queues and workers.
"""
from __future__ import annotations

import asyncio
import uuid

from simpletes.evaluator import rich_print, TEMP_EVAL_DIR
from simpletes.generator import GenerationResult
from simpletes.node import Node, Status


class RuntimeBase:
    """Base class for execution runtimes."""

    def decorate_init_info(self, base_info: str) -> str:
        return base_info

    def queue_stats(self, engine) -> dict[str, int]:
        return {
            "gen_concurrency": engine.config.gen_concurrency,
            "k_candidates": engine.config.k_candidates,
            "gen_queue": engine.gen_queue.qsize(),
            "gen_inflight": engine._gen_inflight,
            "eval_queue": engine.eval_queue.qsize(),
            "pending": len(engine._pending_nodes),
            "eval_inflight": self.eval_active(engine),
        }

    def eval_active(self, engine) -> int:
        return engine._eval_inflight

    def eval_concurrency_for_backpressure(self, engine) -> int:
        return engine.config.eval_concurrency

    async def enqueue_generation_task(self, engine, task) -> bool:
        while not engine._stop_event.is_set():
            try:
                await asyncio.wait_for(engine.gen_queue.put(task), timeout=0.1)
                return True
            except TimeoutError:
                continue
        return False

    async def handle_generation_output(
        self,
        engine,
        task,
        result: GenerationResult,
        track_io: bool,
    ) -> None:
        """Handle a generated child program (local pending node + eval queue)."""
        node = Node(
            id=uuid.uuid4().hex,
            code=result.code,
            parent_ids=list(task.inspiration_ids),
            gen_id=task.gen_id,
            chain_idx=task.chain_idx,
            shared_construction_id=task.shared_construction_id,
            status=Status.EVAL_PENDING,
        )
        if track_io:
            node.llm_input = result.llm_input
            node.llm_output = result.llm_output
            node.token_usage = result.token_usage

        async with engine._db_lock:
            engine._pending_nodes[node.id] = node

        try:
            await engine.eval_queue.put(node.id)
        except asyncio.CancelledError:
            async with engine._db_lock:
                engine._pending_nodes.pop(node.id, None)
            raise

    async def run(self, engine) -> None:
        raise NotImplementedError("RuntimeBase.run must be implemented by subclasses")


class LocalRuntime(RuntimeBase):
    """Single-node, in-process runtime."""

    async def run(self, engine) -> None:
        # Initialize if empty
        async with engine._db_lock:
            is_empty = len(engine.db.nodes) == 0
        if is_empty:
            await engine._initialize_from_scratch()

        rich_print(engine._log("🏃", f"[cyan]Starting {engine.config.gen_concurrency} gen workers and {engine.config.eval_concurrency} eval workers...[/cyan]"))
        rich_print(engine._log("🏃", f"[cyan]Temp Eval Directory: {TEMP_EVAL_DIR}[/cyan]"))
        # Start workers
        engine._gen_workers = [
            asyncio.create_task(engine._gen_worker(i), name=f"gen_{i}")
            for i in range(engine.config.gen_concurrency)
        ]
        engine._eval_workers = [
            asyncio.create_task(engine._eval_worker(i), name=f"eval_{i}")
            for i in range(engine.config.eval_concurrency)
        ]
        rich_print(engine._log("🏃", f"[cyan]Starting scheduler loop...[/cyan]"))

        await engine._scheduler_loop()

        # Shutdown workers
        all_workers = engine._gen_workers + engine._eval_workers
        for w in all_workers:
            w.cancel()

        if all_workers:
            rich_print(engine._log("⚠", f"[yellow][dim]Stopping {len(all_workers)} workers...[/dim][/yellow]"))
            await asyncio.gather(*all_workers, return_exceptions=True)

        # Drain queues
        for q in (engine.gen_queue, engine.eval_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except asyncio.QueueEmpty:
                    break

        # Final checkpoint + summary
        await engine._finalize_run()
