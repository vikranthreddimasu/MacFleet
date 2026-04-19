"""Tests for compute data models: TaskSpec, TaskResult, TaskFuture, RemoteTaskError."""

import asyncio

import pytest

from macfleet.compute.models import (
    RemoteTaskError,
    TaskFuture,
    TaskResult,
    TaskSpec,
)

# --------------------------------------------------------------------------- #
# TaskSpec                                                                     #
# --------------------------------------------------------------------------- #


class TestTaskSpec:
    def test_from_call_basic(self):
        spec = TaskSpec.from_call(lambda x: x + 1, args=(42,))
        assert spec.task_id  # non-empty UUID hex
        assert len(spec.task_id) == 32
        assert spec.fn_bytes
        assert spec.args_bytes
        assert spec.timeout_sec == 300.0

    def test_from_call_with_kwargs(self):
        spec = TaskSpec.from_call(len, args=([1, 2, 3],), kwargs=None, timeout=60.0)
        assert spec.timeout_sec == 60.0

    def test_pack_unpack_roundtrip(self):
        spec = TaskSpec.from_call(lambda x: x ** 2, args=(7,))
        data = spec.pack()
        assert isinstance(data, bytes)

        restored = TaskSpec.unpack(data)
        assert restored.task_id == spec.task_id
        assert restored.fn_bytes == spec.fn_bytes
        assert restored.args_bytes == spec.args_bytes
        assert restored.kwargs_bytes == spec.kwargs_bytes
        assert restored.timeout_sec == spec.timeout_sec

    def test_load_fn_and_args(self):
        def square(x):
            return x * x

        spec = TaskSpec.from_call(square, args=(5,), kwargs={"extra": True})
        fn = spec.load_fn()
        args = spec.load_args()
        kwargs = spec.load_kwargs()

        assert fn(5) == 25
        assert args == (5,)
        assert kwargs == {"extra": True}

    def test_unique_task_ids(self):
        ids = {TaskSpec.from_call(len, args=([],)).task_id for _ in range(100)}
        assert len(ids) == 100

    def test_closure_serialization(self):
        """cloudpickle can serialize closures (stdlib pickle cannot)."""
        offset = 10
        spec = TaskSpec.from_call(lambda x: x + offset, args=(5,))
        fn = spec.load_fn()
        assert fn(5) == 15

    def test_empty_args(self):
        spec = TaskSpec.from_call(lambda: 42)
        assert spec.load_args() == ()
        assert spec.load_kwargs() == {}


# --------------------------------------------------------------------------- #
# TaskResult                                                                   #
# --------------------------------------------------------------------------- #


class TestTaskResult:
    def test_success_roundtrip(self):
        result = TaskResult.success("abc123", {"key": [1, 2, 3]})
        assert result.ok is True
        assert result.error is None
        assert result.unwrap() == {"key": [1, 2, 3]}

    def test_failure_roundtrip(self):
        try:
            raise ValueError("bad input")
        except ValueError:
            result = TaskResult.failure("abc123", None)

        assert result.ok is False
        assert "ValueError" in result.error
        assert "bad input" in result.error

    def test_pack_unpack_success(self):
        result = TaskResult.success("task-1", 42)
        data = result.pack()
        restored = TaskResult.unpack(data)

        assert restored.task_id == "task-1"
        assert restored.ok is True
        assert restored.unwrap() == 42

    def test_pack_unpack_failure(self):
        result = TaskResult(task_id="task-2", ok=False, error="Something broke")
        data = result.pack()
        restored = TaskResult.unpack(data)

        assert restored.task_id == "task-2"
        assert restored.ok is False
        assert restored.error == "Something broke"

    def test_unwrap_failure_raises(self):
        result = TaskResult(task_id="bad", ok=False, error="kaboom")
        with pytest.raises(RemoteTaskError) as exc_info:
            result.unwrap()
        assert "kaboom" in str(exc_info.value)
        assert exc_info.value.task_id == "bad"


# --------------------------------------------------------------------------- #
# TaskFuture                                                                   #
# --------------------------------------------------------------------------- #


class TestTaskFuture:
    @pytest.mark.asyncio
    async def test_done_flag(self):
        future = TaskFuture(task_id="f-1")
        assert not future.done

        future.set_result(TaskResult.success("f-1", "hello"))
        assert future.done

    @pytest.mark.asyncio
    async def test_result_resolves(self):
        future = TaskFuture(task_id="f-2")

        # Set result from another task
        async def set_later():
            await asyncio.sleep(0.05)
            future.set_result(TaskResult.success("f-2", 99))

        asyncio.create_task(set_later())
        value = await future.result(timeout=2.0)
        assert value == 99

    @pytest.mark.asyncio
    async def test_result_timeout(self):
        future = TaskFuture(task_id="f-3")
        with pytest.raises(asyncio.TimeoutError):
            await future.result(timeout=0.05)

    @pytest.mark.asyncio
    async def test_result_failure(self):
        future = TaskFuture(task_id="f-4")
        future.set_result(TaskResult(task_id="f-4", ok=False, error="oops"))
        with pytest.raises(RemoteTaskError):
            await future.result(timeout=1.0)


# --------------------------------------------------------------------------- #
# RemoteTaskError                                                              #
# --------------------------------------------------------------------------- #


class TestRemoteTaskError:
    def test_attributes(self):
        err = RemoteTaskError("task-99", "Traceback:\n  File ...\nValueError: bad")
        assert err.task_id == "task-99"
        assert "task-99" in str(err)
        assert "Traceback" in err.remote_traceback
