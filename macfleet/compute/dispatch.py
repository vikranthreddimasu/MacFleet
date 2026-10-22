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
from typing import Any, Callable, Iterable, Optional

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
        self._workers = list(dict.fromkeys(worker_peer_ids))
        self._pending: dict[str, TaskFuture] = {}
        self._deadlines: dict[str, asyncio.TimerHandle] = {}
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
        if self._running:
            raise RuntimeError("TaskDispatcher is already running")
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

    def _finish(self, result: TaskResult) -> None:
        future = self._pending.pop(result.task_id, None)
        self._task_to_worker.pop(result.task_id, None)
        timer = self._deadlines.pop(result.task_id, None)
        if timer is not None:
            timer.cancel()
        if future is not None and not future.done:
            future.set_result(result)

    def _fail_pending(self, reason: str) -> None:
        for task_id in list(self._pending):
            self._finish(TaskResult(task_id=task_id, ok=False, error=reason))

    def _fail_pending_for_worker(self, worker_id: str, reason: str) -> None:
        for task_id, assigned in list(self._task_to_worker.items()):
            if assigned == worker_id:
                self._finish(TaskResult(task_id=task_id, ok=False, error=reason))

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
        if not self._running:
            raise RuntimeError("TaskDispatcher must be started before submitting tasks")

        spec = TaskSpec.from_call(fn, args, kwargs, timeout=timeout)
        try:
            payload = spec.pack()
        except (TypeError, ValueError) as e:
            raise ValueError(f"TaskSpec could not be serialized: {e}") from e
        future = TaskFuture(task_id=spec.task_id)

        worker_id = self._pick_live_worker()
        if worker_id is None:
            future.set_result(TaskResult(
                task_id=spec.task_id, ok=False,
                error="no live workers (all known workers are disconnected)",
            ))
            return future

        # Record the assignment BEFORE the send so that a send-side
        # failure can be cleaned up by callers if needed.
        self._pending[spec.task_id] = future
        self._task_to_worker[spec.task_id] = worker_id
        timeout_result = TaskResult(task_id=spec.task_id, ok=False, error=(
            f"Task timed out: worker {worker_id} did not return {spec.task_id} within {spec.timeout_sec}s. "
            "Execution may still be running; check the worker before retrying."
        ))
        self._deadlines[spec.task_id] = asyncio.get_running_loop().call_later(
            spec.timeout_sec, self._finish, timeout_result,
        )
        try:
            await asyncio.wait_for(
                self._transport.send(worker_id, payload, msg_type=MessageType.TASK),
                timeout=spec.timeout_sec,
            )
        except asyncio.TimeoutError:
            self._finish(timeout_result)
            return future
        except (ConnectionError, OSError) as e:
            self._finish(TaskResult(task_id=spec.task_id, ok=False, error=str(e)))
            return future
        except BaseException:
            self._finish(TaskResult(task_id=spec.task_id, ok=False, error="submission cancelled or failed"))
            raise
        logger.debug("Dispatched task %s to %s", spec.task_id[:8], worker_id)
        return future

    def _pick_live_worker(self) -> Optional[str]:
        """Round-robin to the next worker with an active transport connection.

        Skips entries in self._workers that no longer have a live
        PeerConnection so a dead peer doesn't immediately strand the task.
        Returns None when all workers are dead.
        """
        n = len(self._workers)
        for _ in range(n):
            candidate = self._workers[self._next_worker % n]
            self._next_worker += 1
            if self._transport.get_connection(candidate) is not None:
                return candidate
        return None

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

        return list(await asyncio.gather(
            *(future.result(timeout=timeout) for future in futures)
        ))

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
                    timeout=None,  # Task deadlines own liveness; an idle worker is not a failed worker.
                )
                if msg.msg_type == MessageType.RESULT:
                    try:
                        result = TaskResult.unpack(msg.payload)
                    except ValueError as e:
                        logger.warning("Ignoring malformed RESULT from %s: %s", worker_id, e)
                        continue
                    if self._task_to_worker.get(result.task_id) != worker_id:
                        logger.warning("Ignoring unassigned task result %s from %s", result.task_id, worker_id)
                        continue
                    self._finish(result)
            except asyncio.TimeoutError:
                self._fail_pending_for_worker(worker_id, f"worker {worker_id} connection timed out")
                return
            except asyncio.CancelledError:
                return
            except (ConnectionError, OSError, EOFError, asyncio.IncompleteReadError, ValueError) as e:
                logger.warning("Lost connection to worker %s: %s", worker_id, e)
                self._fail_pending_for_worker(worker_id, f"worker {worker_id} disconnected: {e}")
                return
