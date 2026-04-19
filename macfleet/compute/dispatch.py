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
        self._pending[spec.task_id] = future

        worker_id = self._workers[self._next_worker % len(self._workers)]
        self._next_worker += 1

        await self._transport.send(worker_id, spec.pack(), msg_type=MessageType.TASK)
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
                return
