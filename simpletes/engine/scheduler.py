"""
Scheduler mixin for SimpleTESEngine.

Implements Algorithm 2's ``Procedure Scheduler`` loop: pick a ready chain, ask
the policy for inspirations, build a prompt, dispatch a batch of k generation
jobs, and yield on a progress event until some batch completes.

Methods defined here are mixed into ``SimpleTESEngine`` and rely on its
state (``self.config``, ``self.db``, ``self.selector``,
``self.generator``, ``self.runtime``, ``self._db_lock``, ``self._counter_lock``,
``self._stop_event``, ``self._progress_event``, ``self.generation_attempts``,
and the static-prompt caches). They never run standalone.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from simpletes.evaluator import rich_print
from simpletes.generator import GenerationTask
from simpletes.policies import PendingFinalize

if TYPE_CHECKING:
    from simpletes.node import NodeDatabaseSnapshot


# Brief sleep when no inspirations available - balances CPU usage vs responsiveness
_SCHEDULER_IDLE_SLEEP_SEC = 0.02
# Double-check delay for TOCTOU race prevention in completion detection
_COMPLETION_CHECK_DELAY_SEC = 0.02


class SchedulerMixin:
    """Scheduler loop, inspiration selection, and batch dispatch."""

    def _get_static_prompt(
        self,
        db_snapshot: NodeDatabaseSnapshot,
        shared_ref: dict[str, str] | None,
    ) -> tuple[str, list[str], str | None] | None:
        """Return cached prompt/parents for init_program-only mode."""
        shared_id = shared_ref["snapshot_id"] if shared_ref is not None else None
        if (
            self._static_prompt is not None
            and self._static_inspiration_ids is not None
            and self._static_shared_construction_id == shared_id
        ):
            return self._static_prompt, list(self._static_inspiration_ids), shared_id

        if not db_snapshot.nodes:
            return None

        root = next((nd for nd in db_snapshot.nodes.values() if not nd.parent_ids), None)
        if root is None:
            root = db_snapshot.best() or next(iter(db_snapshot.nodes.values()))

        prompt = self.generator.build_prompt(
            [root],
            shared_construction_summary=shared_ref["summary"] if shared_ref is not None else None,
        )
        self._static_prompt = prompt
        self._static_inspiration_ids = [root.id]
        self._static_shared_construction_id = shared_id
        return prompt, [root.id], shared_id

    async def _select_inspirations_and_prompt(
        self,
    ) -> tuple[str, list[str], int, str | None] | None:
        """Select inspirations and build a prompt."""
        async with self._db_lock:
            stats = self._queue_stats_locked()
            db_snapshot = self._db_snapshot_locked()
            shared_refs = self._shared_construction_refs_locked()

        if self._static_prompt_mode:
            prepared = self._get_static_prompt(db_snapshot, shared_refs.get(0))
            if prepared is None:
                await asyncio.sleep(_SCHEDULER_IDLE_SLEEP_SEC)
                return None
            return prepared[0], prepared[1], 0, prepared[2]

        n_cap = (
            self.config.max_inspirations_cnt
            if self.config.max_inspirations_cnt is not None
            else self.config.num_inspirations
        )
        inspirations, failure_patterns, chain_idx = self.selector.select(
            db_snapshot,
            n_cap,
            stats=stats,
            eval_concurrency=self.runtime.eval_concurrency_for_backpressure(self),
            backpressure_multiplier=self.config.backpressure_multiplier,
        )

        if not inspirations:
            await asyncio.sleep(_SCHEDULER_IDLE_SLEEP_SEC)
            return None

        inspiration_ids = [n.id for n in inspirations]
        shared_ref = shared_refs.get(chain_idx)
        policy_context = self.selector.get_policy_context(chain_idx, db_snapshot)
        prompt = self.generator.build_prompt(
            inspirations,
            failure_patterns,
            policy_context=policy_context,
            shared_construction_summary=shared_ref["summary"] if shared_ref is not None else None,
        )
        return prompt, inspiration_ids, chain_idx, (
            shared_ref["snapshot_id"] if shared_ref is not None else None
        )

    def _prepare_generation_tasks(
        self,
        prompt: str,
        inspiration_ids: list[str],
        chain_idx: int,
        k: int,
        shared_construction_id: str | None,
    ) -> tuple[int, list[GenerationTask]]:
        """Prepare one logical generation batch as one-or-many queued tasks."""
        assert k > 0, "k_candidates must be > 0"
        gen_id = self.generator.next_gen_id()
        self.selector.register_batch(gen_id, chain_idx, inspiration_ids, k)

        if self.config.stream_k_candidates:
            tasks = [
                self.generator.create_task_with_gen_id(
                    prompt=prompt,
                    inspiration_ids=inspiration_ids,
                    k=1,
                    chain_idx=chain_idx,
                    gen_id=gen_id,
                    shared_construction_id=shared_construction_id,
                )
                for _ in range(k)
            ]
            assert len(tasks) == k, "stream dispatch task count mismatch"
            assert all(task.k == 1 for task in tasks), "stream dispatch requires k=1 tasks"
        else:
            tasks = [
                self.generator.create_task_with_gen_id(
                    prompt=prompt,
                    inspiration_ids=inspiration_ids,
                    k=k,
                    chain_idx=chain_idx,
                    gen_id=gen_id,
                    shared_construction_id=shared_construction_id,
                )
            ]
            assert len(tasks) == 1, "batched dispatch must create exactly one task"
            assert tasks[0].k == k, "batched dispatch task k mismatch"

        assert all(task.gen_id == gen_id for task in tasks), "task gen_id mismatch"
        submitted = sum(task.k for task in tasks)
        assert submitted == k, "prepared task attempts must equal logical batch size"
        return gen_id, tasks

    async def _enqueue_generation_task(self, task: GenerationTask) -> bool:
        """Enqueue one generation task."""
        return await self.runtime.enqueue_generation_task(self, task)

    async def _handle_pending_finalize(self, pending: PendingFinalize | None) -> None:
        """Handle policy batch finalization (reflection + hooks + commit to chain).

        Called when on_child_done or on_generation_failed returns a PendingFinalize.
        Policy handles reflection internally using llm_policy config.
        """
        if pending is None:
            return

        await self.selector.finalize_batch(pending, self.db)

    async def _schedule_generation(self) -> bool:
        """Schedule a single generation. Returns True if scheduled or should continue, False to stop.

        Uses DB snapshot for thread-safe inspiration selection without holding locks.
        """
        if self._stop_event.is_set():
            return False

        k = self.config.k_candidates
        assert k > 0, "k_candidates must be > 0"

        # Check budget under counter lock
        async with self._counter_lock:
            if self.generation_attempts >= self.config.max_generations:
                return False

        prepared = await self._select_inspirations_and_prompt()
        if prepared is None:
            return False

        prompt, inspiration_ids, chain_idx, shared_construction_id = prepared

        # Increment counter BEFORE queueing to prevent race condition:
        # If we queue first and get cancelled before incrementing,
        # the task is in queue but counter doesn't reflect it.
        async with self._counter_lock:
            self.generation_attempts += k

        gen_id, tasks = self._prepare_generation_tasks(
            prompt,
            inspiration_ids,
            chain_idx,
            k,
            shared_construction_id,
        )

        queued_attempts = 0
        for task in tasks:
            queued = await self._enqueue_generation_task(task)
            if not queued:
                break
            queued_attempts += task.k

        missing_attempts = k - queued_attempts
        assert 0 <= missing_attempts <= k, "missing attempt accounting out of range"

        # If some candidates were not queued (e.g. stop event), rollback and close
        # the logical batch for those missing candidates.
        if missing_attempts > 0:
            async with self._counter_lock:
                self.generation_attempts -= missing_attempts
            for _ in range(missing_attempts):
                completion = self.selector.on_generation_failed(gen_id)
                await self._handle_pending_finalize(completion)

        return True

    async def _is_run_complete(self) -> bool:
        """Check if all work is drained.

        Uses atomic snapshot of all relevant state to avoid TOCTOU race conditions.
        Double-checks after a brief yield to ensure no in-flight work.
        """
        # First check: atomic snapshot of all state
        async with self._counter_lock:
            gen_attempts = self.generation_attempts
            max_gens = self.config.max_generations

        if gen_attempts < max_gens:
            return False

        # Check queues and pending nodes atomically
        async with self._db_lock:
            stats = self._queue_stats_locked()
        pending_count = stats["pending"]
        gen_q_size = stats["gen_queue"]
        eval_q_size = stats["eval_queue"]

        if gen_q_size > 0 or eval_q_size > 0 or pending_count > 0:
            return False

        # Double-check after brief yield to catch any in-flight operations
        await asyncio.sleep(_COMPLETION_CHECK_DELAY_SEC)

        # Second atomic check
        async with self._db_lock:
            stats = self._queue_stats_locked()
        pending_count = stats["pending"]
        gen_q_size = stats["gen_queue"]
        eval_q_size = stats["eval_queue"]

        return gen_q_size == 0 and eval_q_size == 0 and pending_count == 0

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            if await self._schedule_generation():
                continue

            if await self._is_run_complete():
                rich_print(self._log("✓", "[bold green]Max generations reached. Stopping.[/bold green]"))
                self._stop_event.set()
                break

            self._progress_event.clear()
            try:
                await asyncio.wait_for(self._progress_event.wait(), timeout=30)
            except TimeoutError:
                async with self._db_lock:
                    stats = self._queue_stats_locked()
                gen_busy = stats.get("gen_inflight", 0)
                eval_busy = stats.get("eval_inflight", 0)
                gen_q = stats.get("gen_queue", 0)
                eval_q = stats.get("eval_queue", 0)
                rich_print(self._log(
                    "⏳",
                    f"[dim]Scheduler waiting… "
                    f"gen workers: {gen_busy} active / {gen_q} queued, "
                    f"eval workers: {eval_busy} active / {eval_q} queued[/dim]",
                ))
