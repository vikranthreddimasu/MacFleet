"""Worker-side task executor.

Receives tasks from the coordinator, executes them in a thread pool,
and sends results back.

    worker = TaskWorker(transport, "coordinator-id")
    await worker.start()
    # ... worker runs until stopped ...
    await worker.stop()

v2.2 PR 7 (Issue 20 + A2 + A8): no more cloudpickle. Tasks must be
registered via @macfleet.task before the worker starts — the wire
carries the task NAME; the worker looks the callable up locally.
Unknown names are rejected without executing anything.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from macfleet.comm.protocol import MessageType
from macfleet.comm.transport import PeerTransport
from macfleet.compute.authz import DEFAULT_TASK_POLICY, TaskAuthorizationError, TaskAuthorizationPolicy
from macfleet.compute.models import TaskResult, TaskSpec
from macfleet.security.audit import audit_event

logger = logging.getLogger(__name__)


class TaskWorker:
    """Receives tasks from coordinator, executes them, sends results.

    Only registered, authorized tasks enter the thread pool. Threads share
    worker state and cannot be forcibly stopped; a timeout reports failure
    but retains the execution slot until the underlying callable exits.
    """

    def __init__(
        self,
        transport: PeerTransport,
        coordinator_peer_id: str,
        max_workers: Optional[int] = None,
        policy: Optional[TaskAuthorizationPolicy] = None,
    ):
        if max_workers is not None and (
            not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer when provided")
        self._transport = transport
        self._coordinator = coordinator_peer_id
        self._max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self._policy = policy or DEFAULT_TASK_POLICY
        self._executor: Optional[ThreadPoolExecutor] = None
        self._listener_task: Optional[asyncio.Task] = None
        # Strong-ref every spawned _execute_and_reply task so Python's GC
        # can't collect them mid-run (asyncio create_task is fire-and-forget
        # with only a weak reference inside the event loop).
        self._inflight: set[asyncio.Task] = set()
        self._executions: set[asyncio.Future] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        """Start listening for tasks from the coordinator."""
        async with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("TaskWorker is already running")
            if any(not execution.done() for execution in self._executions):
                raise RuntimeError("Previous worker tasks are still executing; wait before restarting")
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._running = True
            self._listener_task = asyncio.create_task(self._listen_tasks(asyncio.Semaphore(self._max_workers)))
            logger.info(
                "TaskWorker started (max_workers=%d, coordinator=%s)",
                self._max_workers, self._coordinator,
            )

    async def stop(self) -> None:
        """Stop the worker and shut down the thread pool."""
        async with self._lifecycle_lock:
            self._running = False
            try:
                if self._listener_task is not None:
                    self._listener_task.cancel()
                    await asyncio.gather(self._listener_task, return_exceptions=True)
                # Drain results; each execution and send is bounded by timeout_sec.
                await asyncio.gather(*self._inflight, return_exceptions=True)
            finally:
                inflight = tuple(self._inflight)
                for task in inflight:
                    task.cancel()
                if self._executor:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor = None
                await asyncio.gather(*inflight, return_exceptions=True)

    async def _listen_tasks(self, slots: asyncio.Semaphore) -> None:
        """Listen for TASK messages from the coordinator."""
        conn = self._transport.get_connection(self._coordinator)
        if conn is None:
            logger.error("No connection to coordinator %s", self._coordinator)
            return

        while self._running:
            try:
                msg = await conn.recv_message(
                    timeout=None,  # Task deadlines own liveness; an idle worker is not a failed worker.
                )
                if msg.msg_type == MessageType.TASK:
                    try:
                        spec = TaskSpec.unpack(msg.payload)
                    except ValueError as e:
                        logger.warning("Ignoring malformed TASK from %s: %s", self._coordinator, e)
                        continue
                    await slots.acquire()
                    inflight = asyncio.create_task(self._execute_and_reply(spec, slots))
                    self._inflight.add(inflight)
                    inflight.add_done_callback(self._inflight.discard)
            except asyncio.TimeoutError:
                logger.error("Coordinator connection timed out; reconnect before accepting more tasks")
                return
            except asyncio.CancelledError:
                return
            except (ConnectionError, OSError, EOFError, asyncio.IncompleteReadError, ValueError) as e:
                logger.warning(
                    "Lost connection to coordinator %s: %s",
                    self._coordinator, e,
                )
                return

    async def _execute_and_reply(self, spec: TaskSpec, slots: asyncio.Semaphore) -> None:
        """Execute a task in the thread pool and send the result back."""
        loop = asyncio.get_running_loop()
        execution = None
        try:
            try:
                entry = spec.resolve()
                self._policy.authorize(spec, entry)
                args, kwargs = spec.validated_args(entry)
                execution = loop.run_in_executor(self._executor, partial(entry.fn, *args, **kwargs))
                self._executions.add(execution)
                execution.add_done_callback(self._executions.discard)
                value = await asyncio.wait_for(
                    asyncio.shield(execution),
                    timeout=spec.timeout_sec,
                )
                result = TaskResult.success(spec.task_id, value)
                audit_event(
                    "task.executed",
                    task_name=spec.task_name,
                    coordinator=self._coordinator,
                )
            except asyncio.TimeoutError:
                result = TaskResult(
                    task_id=spec.task_id,
                    ok=False,
                    error=(f"Task timed out after {spec.timeout_sec}s. "
                           "Execution may still be running; check the worker before retrying."),
                )
            except TaskAuthorizationError as e:
                audit_event(
                    "task.authorization_denied",
                    task_name=spec.task_name,
                    coordinator=self._coordinator,
                    reason=str(e),
                )
                result = TaskResult.failure(spec.task_id, e)
            except Exception as e:
                audit_event(
                    "task.failed",
                    task_name=spec.task_name,
                    coordinator=self._coordinator,
                    error_type=type(e).__name__,
                )
                result = TaskResult.failure(spec.task_id, e)

            try:
                payload = result.pack()
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(
                    "Task %s produced an unserializable result: %s",
                    spec.task_id[:8], e,
                )
                audit_event(
                    "task.result_serialization_failed",
                    task_name=spec.task_name,
                    coordinator=self._coordinator,
                    error_type=type(e).__name__,
                )
                result = TaskResult(
                    task_id=spec.task_id,
                    ok=False,
                    error=f"Task result could not be serialized: {e}",
                )
                payload = result.pack()

            try:
                await asyncio.wait_for(
                    self._transport.send(self._coordinator, payload, msg_type=MessageType.RESULT),
                    timeout=spec.timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.error("Result delivery for task %s timed out after %ss", spec.task_id[:8], spec.timeout_sec)
            except (ConnectionError, OSError, EOFError, asyncio.IncompleteReadError, ValueError) as e:
                logger.error(
                    "Failed to send result for task %s: %s",
                    spec.task_id[:8], e,
                )
        finally:
            if execution is None:
                slots.release()
            else:
                def release_slot(done: asyncio.Future) -> None:
                    slots.release()
                    if not done.cancelled():
                        done.exception()  # Observe failures that arrive after the caller's timeout.

                execution.add_done_callback(release_slot)
