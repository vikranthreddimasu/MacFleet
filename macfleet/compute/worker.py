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
from typing import Any, Optional

from macfleet.comm.protocol import MessageType
from macfleet.comm.transport import PeerTransport
from macfleet.compute.models import TaskNotRegisteredError, TaskResult, TaskSpec

logger = logging.getLogger(__name__)


def _execute_task(task_name: str, args: list, kwargs: dict) -> Any:
    """Execute a registered task in a worker thread.

    Looks up `task_name` in the process-local TaskRegistry — if the name
    isn't registered, raises TaskNotRegisteredError. The worker process
    MUST have imported the module that declares the task BEFORE accepting
    TASK messages.
    """
    from pydantic import BaseModel

    from macfleet.compute.registry import get_default_registry

    entry = get_default_registry().get(task_name)
    if entry is None:
        raise TaskNotRegisteredError(
            task_name, get_default_registry().names(),
        )

    # Apply schema validation if declared on the task
    resolved_args: list = list(args)
    resolved_kwargs: dict = dict(kwargs)
    if entry.schema is not None:
        if len(resolved_args) == 1 and isinstance(resolved_args[0], dict):
            resolved_args = [entry.schema(**resolved_args[0])]
        elif resolved_kwargs:
            resolved_args = [entry.schema(**resolved_kwargs)]
            resolved_kwargs = {}

    result = entry.fn(*resolved_args, **resolved_kwargs)
    # Return values are msgpack-native. Pydantic models get dumped for transport.
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return result


class TaskWorker:
    """Receives tasks from coordinator, executes them, sends results.

    v2.2 PR 7: switched from ProcessPoolExecutor to ThreadPoolExecutor.
    The old process-pool isolation was a defense against arbitrary pickled
    code crashing the worker process. Now that tasks are looked up by name
    in a pre-registered TaskRegistry, the attack surface is gone — if a
    registered task crashes, it's a bug in user code, not an RCE primitive.
    Threads let the worker access in-process state (e.g. models loaded once
    at startup) without re-forking for each call.
    """

    def __init__(
        self,
        transport: PeerTransport,
        coordinator_peer_id: str,
        max_workers: Optional[int] = None,
    ):
        self._transport = transport
        self._coordinator = coordinator_peer_id
        self._max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start listening for tasks from the coordinator."""
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_tasks())
        logger.info(
            "TaskWorker started (max_workers=%d, coordinator=%s)",
            self._max_workers, self._coordinator,
        )

    async def stop(self) -> None:
        """Stop the worker and shut down the thread pool."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def _listen_tasks(self) -> None:
        """Listen for TASK messages from the coordinator."""
        conn = self._transport.get_connection(self._coordinator)
        if conn is None:
            logger.error("No connection to coordinator %s", self._coordinator)
            return

        while self._running:
            try:
                msg = await conn.recv_message(
                    timeout=self._transport.config.recv_timeout_sec,
                )
                if msg.msg_type == MessageType.TASK:
                    spec = TaskSpec.unpack(msg.payload)
                    asyncio.create_task(self._execute_and_reply(spec))
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            except (ConnectionError, OSError) as e:
                logger.warning(
                    "Lost connection to coordinator %s: %s",
                    self._coordinator, e,
                )
                return

    async def _execute_and_reply(self, spec: TaskSpec) -> None:
        """Execute a task in the thread pool and send the result back."""
        loop = asyncio.get_event_loop()
        try:
            value = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    _execute_task,
                    spec.task_name,
                    spec.args,
                    spec.kwargs,
                ),
                timeout=spec.timeout_sec,
            )
            result = TaskResult.success(spec.task_id, value)
        except asyncio.TimeoutError:
            result = TaskResult(
                task_id=spec.task_id,
                ok=False,
                error=f"Task timed out after {spec.timeout_sec}s",
            )
        except Exception as e:
            result = TaskResult.failure(spec.task_id, e)

        try:
            await self._transport.send(
                self._coordinator,
                result.pack(),
                msg_type=MessageType.RESULT,
            )
        except (ConnectionError, OSError) as e:
            logger.error(
                "Failed to send result for task %s: %s",
                spec.task_id[:8], e,
            )
