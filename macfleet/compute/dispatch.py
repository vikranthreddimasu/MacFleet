"""Coordinator-side task dispatcher.

Distributes tasks to pool workers round-robin and collects results.
Runs on the coordinator node (rank 0).

    dispatcher = TaskDispatcher(transport, ["worker-1", "worker-2"])
    await dispatcher.start()
    results = await dispatcher.map(fn, items)
    await dispatcher.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Iterable

from macfleet.comm.protocol import MessageType
from macfleet.comm.transport import PeerTransport
from macfleet.compute.models import TaskFuture, TaskResult, TaskSpec

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """Distributes tasks to pool workers and collects results.

    Assigns tasks round-robin to connected workers. Each task gets a
    unique ID; the dispatcher matches incoming RESULT messages to
    their corresponding TaskFuture.

    The dispatcher owns worker connections during compute mode.
    Do not run gradient sync simultaneously with task dispatch on
    the same transport — they share TCP connections.
    """

    def __init__(self, transport: PeerTransport, worker_peer_ids: list[str]):
        self._transport = transport
        self._workers = list(worker_peer_ids)
        self._pending: dict[str, TaskFuture] = {}
        # task_id -> worker_id, so a disconnect only fails THAT worker's
        # outstanding tasks rather than every pending future.
        self._task_to_worker: dict[str, str] = {}
        self._next_worker = 0
        self._worker_listeners: dict[str, asyncio.Task] = {}
        self._running = False

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def start(self) -> None:
        """Start per-worker result listeners."""
        if not self._workers:
            return
        self._running = True
        for worker_id in self._workers:
            self._worker_listeners[worker_id] = asyncio.create_task(
                self._worker_listener(worker_id)
            )

    async def stop(self) -> None:
        """Stop all listeners and cancel pending futures."""
        self._running = False
        for task in self._worker_listeners.values():
            if not task.done():
                task.cancel()
        for task in self._worker_listeners.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker_listeners.clear()
        self._fail_pending("dispatcher stopped")

    def _fail_pending(self, reason: str) -> None:
        """Resolve every pending TaskFuture with a failure result.

        Without this, callers awaiting future.result() block until their
        own per-call timeout fires (or forever if no timeout was passed).
        """
        if not self._pending:
            return
        from macfleet.compute.models import TaskResult
        for task_id, future in list(self._pending.items()):
            if not future.done:
                future.set_result(
                    TaskResult(task_id=task_id, ok=False, error=reason)
                )
        self._pending.clear()
        self._task_to_worker.clear()

    def _fail_pending_for_worker(self, worker_id: str, reason: str) -> None:
        """Fail just the tasks routed to a dead worker; leave others alone."""
        from macfleet.compute.models import TaskResult
        dead_ids = [tid for tid, w in self._task_to_worker.items() if w == worker_id]
        for tid in dead_ids:
            future = self._pending.pop(tid, None)
            self._task_to_worker.pop(tid, None)
            if future is not None and not future.done:
                future.set_result(TaskResult(task_id=tid, ok=False, error=reason))

    async def submit(
        self,
        fn: Callable,
        *args: Any,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> TaskFuture:
        """Submit a single task to the next available worker.

        Returns a TaskFuture that resolves when the worker sends back
        the result.
        """
        if not self._workers:
            raise RuntimeError("No workers available")

        spec = TaskSpec.from_call(fn, args, kwargs, timeout=timeout)
        future = TaskFuture(task_id=spec.task_id)

        worker_id = self._workers[self._next_worker % len(self._workers)]
        self._next_worker += 1

        # Record the assignment BEFORE the send so that a send-side
        # failure can be cleaned up by callers if needed.
        self._pending[spec.task_id] = future
        self._task_to_worker[spec.task_id] = worker_id
        try:
            await self._transport.send(worker_id, spec.pack(), msg_type=MessageType.TASK)
        except (ConnectionError, OSError) as e:
            self._pending.pop(spec.task_id, None)
            self._task_to_worker.pop(spec.task_id, None)
            from macfleet.compute.models import TaskResult
            future.set_result(TaskResult(task_id=spec.task_id, ok=False, error=str(e)))
            return future
        logger.debug("Dispatched task %s to %s", spec.task_id[:8], worker_id)
        return future

    async def map(
        self,
        fn: Callable,
        iterable: Iterable,
        timeout: float = 300.0,
    ) -> list:
        """Apply fn to each item across workers, return results in order.

        Submits all tasks round-robin, then awaits all results.
        Results are returned in the same order as the input iterable.
        """
        futures = []
        for item in iterable:
            future = await self.submit(fn, item, timeout=timeout)
            futures.append(future)

        results = []
        for future in futures:
            results.append(await future.result(timeout=timeout))
        return results

    async def _worker_listener(self, worker_id: str) -> None:
        """Listen for RESULT messages from one worker.

        Blocks on recv_message() for this worker's connection.
        Routes results to the matching TaskFuture.
        """
        conn = self._transport.get_connection(worker_id)
        if conn is None:
            logger.error("No connection to worker %s", worker_id)
            return

        while self._running:
            try:
                msg = await conn.recv_message(
                    timeout=self._transport.config.recv_timeout_sec,
                )
                if msg.msg_type == MessageType.RESULT:
                    result = TaskResult.unpack(msg.payload)
                    future = self._pending.pop(result.task_id, None)
                    self._task_to_worker.pop(result.task_id, None)
                    if future:
                        future.set_result(result)
                        logger.debug(
                            "Result for task %s from %s (ok=%s)",
                            result.task_id[:8], worker_id, result.ok,
                        )
                    else:
                        logger.warning(
                            "Received result for unknown task %s",
                            result.task_id[:8],
                        )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            except (ConnectionError, OSError) as e:
                logger.warning("Lost connection to worker %s: %s", worker_id, e)
                self._fail_pending_for_worker(worker_id, f"worker {worker_id} disconnected: {e}")
                return
