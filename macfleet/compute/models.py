"""Data models for distributed task dispatch.

TaskSpec and TaskResult are the wire-format messages exchanged between
coordinator and workers. They use msgpack for the envelope and
cloudpickle for function/data serialization (same approach as Ray/Dask).

TaskFuture is the user-facing handle returned by Pool.submit().
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import cloudpickle
import msgpack


class RemoteTaskError(Exception):
    """Raised when a task fails on a remote worker.

    Wraps the remote traceback string so the coordinator can display
    what went wrong on the worker side.
    """

    def __init__(self, task_id: str, remote_traceback: str):
        self.task_id = task_id
        self.remote_traceback = remote_traceback
        super().__init__(
            f"Task {task_id} failed on remote worker:\n{remote_traceback}"
        )


@dataclass
class TaskSpec:
    """A task to be executed on a remote worker.

    Wire format (msgpack):
        { "task_id": str, "fn": bytes, "args": bytes,
          "kwargs": bytes, "timeout": float }
    """

    task_id: str
    fn_bytes: bytes
    args_bytes: bytes
    kwargs_bytes: bytes
    timeout_sec: float = 300.0

    @classmethod
    def from_call(
        cls,
        fn: Any,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        timeout: float = 300.0,
    ) -> TaskSpec:
        """Create a TaskSpec from a function call."""
        return cls(
            task_id=uuid.uuid4().hex,
            fn_bytes=cloudpickle.dumps(fn),
            args_bytes=cloudpickle.dumps(args),
            kwargs_bytes=cloudpickle.dumps(kwargs or {}),
            timeout_sec=timeout,
        )

    def pack(self) -> bytes:
        """Serialize to msgpack bytes for wire transport."""
        return msgpack.packb({
            "task_id": self.task_id,
            "fn": self.fn_bytes,
            "args": self.args_bytes,
            "kwargs": self.kwargs_bytes,
            "timeout": self.timeout_sec,
        })

    @classmethod
    def unpack(cls, data: bytes) -> TaskSpec:
        """Deserialize from msgpack bytes."""
        d = msgpack.unpackb(data, raw=False)
        return cls(
            task_id=d["task_id"],
            fn_bytes=d["fn"],
            args_bytes=d["args"],
            kwargs_bytes=d["kwargs"],
            timeout_sec=d.get("timeout", 300.0),
        )

    def load_fn(self) -> Any:
        """Deserialize the function."""
        return cloudpickle.loads(self.fn_bytes)

    def load_args(self) -> tuple:
        """Deserialize positional arguments."""
        return cloudpickle.loads(self.args_bytes)

    def load_kwargs(self) -> dict:
        """Deserialize keyword arguments."""
        return cloudpickle.loads(self.kwargs_bytes)


@dataclass
class TaskResult:
    """Result of executing a task on a worker.

    Wire format (msgpack):
        { "task_id": str, "ok": bool, "value": bytes, "error": str|None }
    """

    task_id: str
    ok: bool
    value_bytes: bytes = b""
    error: Optional[str] = None

    @classmethod
    def success(cls, task_id: str, value: Any) -> TaskResult:
        """Create a successful result."""
        return cls(
            task_id=task_id,
            ok=True,
            value_bytes=cloudpickle.dumps(value),
        )

    @classmethod
    def failure(cls, task_id: str, exc: BaseException) -> TaskResult:
        """Create a failure result from an exception."""
        return cls(
            task_id=task_id,
            ok=False,
            error=traceback.format_exc(),
        )

    def pack(self) -> bytes:
        """Serialize to msgpack bytes for wire transport."""
        return msgpack.packb({
            "task_id": self.task_id,
            "ok": self.ok,
            "value": self.value_bytes,
            "error": self.error,
        })

    @classmethod
    def unpack(cls, data: bytes) -> TaskResult:
        """Deserialize from msgpack bytes."""
        d = msgpack.unpackb(data, raw=False)
        return cls(
            task_id=d["task_id"],
            ok=d["ok"],
            value_bytes=d.get("value", b""),
            error=d.get("error"),
        )

    def unwrap(self) -> Any:
        """Return the deserialized value, or raise RemoteTaskError."""
        if self.ok:
            return cloudpickle.loads(self.value_bytes)
        raise RemoteTaskError(self.task_id, self.error or "Unknown error")


class TaskFuture:
    """Awaitable handle for a submitted task.

    Returned by TaskDispatcher.submit(). The result is set by the
    dispatcher's background listener when the worker sends a RESULT.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._event = asyncio.Event()
        self._result: Optional[TaskResult] = None

    @property
    def done(self) -> bool:
        """True if the result has been received."""
        return self._event.is_set()

    def set_result(self, result: TaskResult) -> None:
        """Called by the dispatcher when the result arrives."""
        self._result = result
        self._event.set()

    async def result(self, timeout: Optional[float] = None) -> Any:
        """Await and return the task result.

        Raises:
            RemoteTaskError: If the task failed on the worker.
            asyncio.TimeoutError: If timeout is exceeded.
        """
        await asyncio.wait_for(self._event.wait(), timeout=timeout)
        assert self._result is not None
        return self._result.unwrap()
