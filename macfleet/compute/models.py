"""Data models for distributed task dispatch.

TaskSpec and TaskResult are the wire-format messages exchanged between
coordinator and workers.

v2.2 PR 7 (Issue 20 + A2 + A8): rewritten to use named callables + msgpack
args instead of cloudpickled closures. This closes the RCE vector where a
rogue coordinator (or a network attacker past auth) could ship arbitrary
pickled code for the worker to execute. Now the wire carries only:
    - task_name: str (looked up in the worker's TaskRegistry)
    - args:      list of msgpack-native values
    - kwargs:    dict of msgpack-native values
    - timeout:   float

If the name isn't registered on the worker, the task is rejected outright.
Args and return values can be enriched with a Pydantic schema declared on
the decorator — the worker validates inputs against the schema before
calling the function.

TaskFuture is the user-facing handle returned by TaskDispatcher.submit().
"""

from __future__ import annotations

import asyncio
import logging
import math
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import msgpack

from macfleet.compute.registry import TaskEntry, get_default_registry

logger = logging.getLogger(__name__)

# Bounds on wire payload sizes — prevent OOM from malicious or corrupt messages.
MAX_ARGS_BYTES = 64 * 1024 * 1024  # 64 MB msgpack args
MAX_RESULT_BYTES = 256 * 1024 * 1024  # 256 MB msgpack result
MAX_ERROR_TEXT_CHARS = 64 * 1024  # 64 KiB remote traceback/error text


def _truncate_error_text(text: str) -> str:
    """Bound remote error text while preserving the start and final exception."""
    if len(text) <= MAX_ERROR_TEXT_CHARS:
        return text

    marker = f"\n\n...[remote error truncated from {len(text)} to {MAX_ERROR_TEXT_CHARS} chars]...\n\n"
    available = MAX_ERROR_TEXT_CHARS - len(marker)
    if available <= 0:
        return text[:MAX_ERROR_TEXT_CHARS]

    head_chars = available // 2
    tail_chars = available - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def _unpack_msgpack_dict(data: bytes, label: str) -> dict:
    """Decode msgpack and require a top-level mapping."""
    try:
        payload = msgpack.unpackb(data, raw=False)
    except msgpack.exceptions.UnpackException as e:
        raise ValueError(f"{label} payload is not valid msgpack") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload not a dict: {type(payload).__name__}")
    return payload


def _require_str(payload: dict, key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} field {key!r} must be a non-empty string")
    return value


def _require_bool(payload: dict, key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} field {key!r} must be a bool")
    return value


def _validate_optional_timeout(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be None or a positive finite number")
    return float(timeout)


class RemoteTaskError(Exception):
    """Raised when a task fails on a remote worker.

    Wraps the remote traceback string so the coordinator can display
    what went wrong on the worker side.
    """

    def __init__(self, task_id: str, remote_traceback: str):
        self.task_id = task_id
        self.remote_traceback = remote_traceback
        super().__init__(f"Task {task_id} failed on remote worker:\n{remote_traceback}")


class TaskNotRegisteredError(Exception):
    """Raised on the worker when a TASK message names an unregistered callable.

    This is NOT forwarded to the user as-is — it's converted into a
    TaskResult.failure() so the coordinator sees a structured error.
    """

    def __init__(self, task_name: str, known: list[str]):
        self.task_name = task_name
        self.known = known
        super().__init__(f"Task {task_name!r} not registered on this worker. Known tasks: {known}")


@dataclass
class TaskSpec:
    """A task to be executed on a remote worker.

    Wire format (msgpack):
        { "task_id": str, "name": str, "args": list, "kwargs": dict,
          "timeout": float }

    `args` and `kwargs` must be msgpack-native (int, float, str, bytes,
    bool, list, dict, None). Anything else must go through a Pydantic
    schema — the schema is declared on the `@task` decorator and the
    worker uses it to validate args before calling the function.
    """

    task_id: str
    task_name: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    timeout_sec: float = 300.0

    @classmethod
    def from_call(
        cls,
        fn: Any,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        timeout: float = 300.0,
    ) -> TaskSpec:
        """Build a TaskSpec from a registered callable.

        The callable MUST have been decorated with `@macfleet.task` — we
        read the registered name off `fn.task_name`. If it wasn't, raise
        with a helpful error rather than silently cloudpickling it.
        """
        task_name = getattr(fn, "task_name", None)
        if task_name is None:
            raise ValueError(f"Function {fn!r} is not registered. Decorate it with @macfleet.task first.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")

        # If the registered callable has a Pydantic schema, validate the
        # args now (fail fast on the coordinator) and serialize the model
        # dump to msgpack-native types.
        schema = getattr(fn, "schema", None)
        arg_list: list = list(args)
        kwarg_dict: dict = dict(kwargs or {})
        if schema is not None:
            if len(arg_list) == 1 and isinstance(arg_list[0], schema):
                # Common shape: @task(schema=X) def f(args: X): ...
                # User passed an X instance positionally.
                arg_list = [arg_list[0].model_dump(mode="json")]
            # Otherwise leave args/kwargs as-is; validate on the worker side.

        return cls(
            task_id=uuid.uuid4().hex,
            task_name=task_name,
            args=arg_list,
            kwargs=kwarg_dict,
            timeout_sec=timeout,
        )

    def pack(self) -> bytes:
        """Serialize to msgpack bytes for wire transport."""
        packed = cast(
            bytes,
            msgpack.packb(
                {
                    "task_id": self.task_id,
                    "name": self.task_name,
                    "args": self.args,
                    "kwargs": self.kwargs,
                    "timeout": self.timeout_sec,
                },
                use_bin_type=True,
            ),
        )
        if len(packed) > MAX_ARGS_BYTES:
            raise ValueError(f"TaskSpec size {len(packed)}B exceeds max {MAX_ARGS_BYTES}B")
        return packed

    @classmethod
    def unpack(cls, data: bytes) -> TaskSpec:
        """Deserialize from msgpack bytes.

        Rejects payloads over MAX_ARGS_BYTES to prevent a malicious
        coordinator from OOMing the worker with a huge args blob.
        """
        if len(data) > MAX_ARGS_BYTES:
            raise ValueError(f"TaskSpec size {len(data)}B exceeds max {MAX_ARGS_BYTES}B")
        d = _unpack_msgpack_dict(data, "TaskSpec")
        args = d["args"] if "args" in d else []
        kwargs = d["kwargs"] if "kwargs" in d else {}
        timeout = d.get("timeout", 300.0)
        if not isinstance(args, list):
            raise ValueError("TaskSpec field 'args' must be a list")
        if not isinstance(kwargs, dict):
            raise ValueError("TaskSpec field 'kwargs' must be a dict")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("TaskSpec field 'timeout' must be a number")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("TaskSpec field 'timeout' must be positive and finite")
        return cls(
            task_id=_require_str(d, "task_id", "TaskSpec"),
            task_name=_require_str(d, "name", "TaskSpec"),
            args=args,
            kwargs=kwargs,
            timeout_sec=float(timeout),
        )

    def resolve(self) -> TaskEntry:
        """Look up the registered TaskEntry for this spec's task_name.

        Raises TaskNotRegisteredError if the worker doesn't know this name.
        """
        reg = get_default_registry()
        entry = reg.get(self.task_name)
        if entry is None:
            raise TaskNotRegisteredError(self.task_name, reg.names())
        return entry

    def validated_args(self, entry: TaskEntry) -> tuple[list, dict]:
        """Apply the task's Pydantic schema (if any) to args before invocation.

        Called on the worker after name lookup. When a schema is declared
        AND the call shape is a single positional arg dict, we rebuild the
        Pydantic model and pass the instance through. Otherwise we pass
        args through as-is (simple typed args, no schema).
        """
        if entry.schema is None:
            return self.args, self.kwargs
        if len(self.args) == 1 and isinstance(self.args[0], dict):
            model = entry.schema(**self.args[0])
            return [model], self.kwargs
        # Schema declared but non-standard shape → validate as kwargs
        model = entry.schema(**self.kwargs)
        return [model], {}


@dataclass
class TaskResult:
    """Result of executing a task on a worker.

    Wire format (msgpack):
        { "task_id": str, "ok": bool, "value": Any, "error": str|None }

    `value` is a msgpack-native encoding of the return value. For
    schema'd tasks, this is `model.model_dump(mode="json")`.
    """

    task_id: str
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.error, str):
            self.error = _truncate_error_text(self.error)

    @classmethod
    def success(cls, task_id: str, value: Any) -> TaskResult:
        """Create a successful result.

        If `value` is a Pydantic model, dump it to a dict so msgpack
        can serialize it. Otherwise pass through (msgpack will reject
        anything non-native).
        """
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return cls(task_id=task_id, ok=True, value=value)

    @classmethod
    def failure(cls, task_id: str, exc: Optional[BaseException] = None) -> TaskResult:
        """Create a failure result from an exception or current traceback."""
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            tb = traceback.format_exc()
        return cls(task_id=task_id, ok=False, error=tb)

    def pack(self) -> bytes:
        """Serialize to msgpack bytes for wire transport."""
        packed = cast(
            bytes,
            msgpack.packb(
                {
                    "task_id": self.task_id,
                    "ok": self.ok,
                    "value": self.value,
                    "error": self.error,
                },
                use_bin_type=True,
            ),
        )
        if len(packed) > MAX_RESULT_BYTES:
            raise ValueError(f"TaskResult size {len(packed)}B exceeds max {MAX_RESULT_BYTES}B")
        return packed

    @classmethod
    def unpack(cls, data: bytes) -> TaskResult:
        """Deserialize from msgpack bytes."""
        if len(data) > MAX_RESULT_BYTES:
            raise ValueError(f"TaskResult size {len(data)}B exceeds max {MAX_RESULT_BYTES}B")
        d = _unpack_msgpack_dict(data, "TaskResult")
        error = d.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("TaskResult field 'error' must be a string or null")
        return cls(
            task_id=_require_str(d, "task_id", "TaskResult"),
            ok=_require_bool(d, "ok", "TaskResult"),
            value=d.get("value"),
            error=error,
        )

    def unwrap(self) -> Any:
        """Return the deserialized value, or raise RemoteTaskError."""
        if self.ok:
            return self.value
        raise RemoteTaskError(self.task_id, self.error or "Unknown error")


class TaskFuture:
    """Awaitable handle for a submitted task.

    Returned by TaskDispatcher.submit(). The result is set by the
    dispatcher's background listener when the worker sends a RESULT.
    """

    def __init__(self, task_id: str):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")
        self.task_id = task_id
        self._event = asyncio.Event()
        self._result: Optional[TaskResult] = None

    @property
    def done(self) -> bool:
        """True if the result has been received."""
        return self._event.is_set()

    def set_result(self, result: TaskResult) -> None:
        """Called by the dispatcher when the result arrives."""
        if result.task_id != self.task_id:
            raise ValueError(f"result task_id {result.task_id!r} does not match future {self.task_id!r}")
        if self.done:
            raise RuntimeError(f"result for task {self.task_id!r} was already set")
        self._result = result
        self._event.set()

    async def result(self, timeout: Optional[float] = None) -> Any:
        """Await and return the task result.

        Raises:
            RemoteTaskError: If the task failed on the worker.
            asyncio.TimeoutError: If timeout is exceeded.
            ValueError: If timeout is not None or a positive finite number.
        """
        wait_timeout = _validate_optional_timeout(timeout)
        await asyncio.wait_for(self._event.wait(), timeout=wait_timeout)
        assert self._result is not None
        return self._result.unwrap()
