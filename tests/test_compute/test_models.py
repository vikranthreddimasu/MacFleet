"""Tests for compute data models: TaskSpec, TaskResult, TaskFuture, RemoteTaskError.

v2.2 PR 7: tasks are identified by name (registered via @macfleet.task) rather
than cloudpickle'd inline. These tests cover the new name-based dispatch,
Pydantic schema validation, and the msgpack wire format.
"""

import asyncio

import pytest
from pydantic import BaseModel

from macfleet import task
from macfleet.compute.models import (
    RemoteTaskError,
    TaskFuture,
    TaskNotRegisteredError,
    TaskResult,
    TaskSpec,
)
from macfleet.compute.registry import get_default_registry

# --------------------------------------------------------------------------- #
# Tasks registered at module level for reuse across tests                     #
# --------------------------------------------------------------------------- #


@task
def add_one(x: int) -> int:
    return x + 1


@task
def my_len(xs: list) -> int:
    return len(xs)


@task
def square(x: int, extra: bool = False) -> int:
    return x * x if not extra else (x * x) + 1


class TrainArgs(BaseModel):
    epochs: int
    lr: float


@task(schema=TrainArgs)
def fake_train(args: TrainArgs) -> dict:
    return {"epochs_done": args.epochs, "final_lr": args.lr}


# --------------------------------------------------------------------------- #
# TaskSpec: name-based dispatch                                               #
# --------------------------------------------------------------------------- #


class TestTaskSpec:
    def test_from_call_basic(self):
        spec = TaskSpec.from_call(add_one, args=(42,))
        assert spec.task_id
        assert len(spec.task_id) == 32
        assert spec.task_name == add_one.task_name
        assert spec.args == [42]
        assert spec.kwargs == {}
        assert spec.timeout_sec == 300.0

    def test_from_call_with_kwargs(self):
        spec = TaskSpec.from_call(my_len, args=([1, 2, 3],), kwargs=None, timeout=60.0)
        assert spec.timeout_sec == 60.0
        assert spec.args == [[1, 2, 3]]

    def test_pack_unpack_roundtrip(self):
        spec = TaskSpec.from_call(square, args=(7,))
        data = spec.pack()
        assert isinstance(data, bytes)

        restored = TaskSpec.unpack(data)
        assert restored.task_id == spec.task_id
        assert restored.task_name == spec.task_name
        assert restored.args == spec.args
        assert restored.kwargs == spec.kwargs
        assert restored.timeout_sec == spec.timeout_sec

    def test_unregistered_function_rejected(self):
        """A bare lambda has no task_name → from_call must refuse."""
        with pytest.raises(ValueError, match="not registered"):
            TaskSpec.from_call(lambda x: x + 1, args=(5,))

    def test_resolve_known_task(self):
        spec = TaskSpec.from_call(add_one, args=(1,))
        entry = spec.resolve()
        assert entry.name == add_one.task_name
        assert entry.fn is add_one

    def test_resolve_unknown_task_raises(self):
        spec = TaskSpec(
            task_id="abc", task_name="nonexistent.module.fn", args=[], kwargs={},
        )
        with pytest.raises(TaskNotRegisteredError) as excinfo:
            spec.resolve()
        assert "nonexistent.module.fn" in str(excinfo.value)

    def test_unique_task_ids(self):
        ids = {TaskSpec.from_call(my_len, args=([],)).task_id for _ in range(100)}
        assert len(ids) == 100

    def test_empty_args(self):
        @task
        def noop() -> int:
            return 42

        spec = TaskSpec.from_call(noop)
        assert spec.args == []
        assert spec.kwargs == {}

    def test_pack_size_bound_rejected(self):
        """A msgpack payload larger than MAX_ARGS_BYTES is rejected on unpack."""
        from macfleet.compute.models import MAX_ARGS_BYTES
        giant = b"\x00" * (MAX_ARGS_BYTES + 100)
        with pytest.raises(ValueError, match="exceeds max"):
            TaskSpec.unpack(giant)


# --------------------------------------------------------------------------- #
# Pydantic schema validation                                                  #
# --------------------------------------------------------------------------- #


class TestTaskSpecSchema:
    def test_schema_attached_to_decorator(self):
        assert fake_train.schema is TrainArgs
        assert fake_train.task_name.endswith("fake_train")

    def test_from_call_serializes_pydantic_instance(self):
        args = TrainArgs(epochs=3, lr=0.01)
        spec = TaskSpec.from_call(fake_train, args=(args,))
        # Wire carries the model dump, not the Pydantic instance itself
        assert spec.args == [{"epochs": 3, "lr": 0.01}]

    def test_validated_args_rehydrates_pydantic(self):
        args = TrainArgs(epochs=3, lr=0.01)
        spec = TaskSpec.from_call(fake_train, args=(args,))
        entry = spec.resolve()
        resolved_args, resolved_kwargs = spec.validated_args(entry)
        assert len(resolved_args) == 1
        assert isinstance(resolved_args[0], TrainArgs)
        assert resolved_args[0].epochs == 3
        assert resolved_args[0].lr == 0.01

    def test_schema_validation_rejects_bad_args(self):
        """If wire carries garbage that doesn't fit the schema, validation fails."""
        spec = TaskSpec(
            task_id="abc", task_name=fake_train.task_name,
            args=[{"epochs": "not an int", "lr": 0.01}], kwargs={},
        )
        entry = spec.resolve()
        # Pydantic 2 raises ValidationError (subclass of ValueError)
        with pytest.raises(Exception) as excinfo:
            spec.validated_args(entry)
        assert "epochs" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# TaskResult                                                                   #
# --------------------------------------------------------------------------- #


class TestTaskResult:
    def test_success_roundtrip(self):
        result = TaskResult.success("abc123", {"key": [1, 2, 3]})
        assert result.ok is True
        assert result.error is None
        assert result.unwrap() == {"key": [1, 2, 3]}

    def test_success_with_pydantic_model(self):
        args = TrainArgs(epochs=5, lr=0.001)
        result = TaskResult.success("abc123", args)
        # Pydantic model dumped for wire
        assert result.value == {"epochs": 5, "lr": 0.001}

    def test_failure_from_exception(self):
        try:
            raise ValueError("bad input")
        except ValueError as e:
            result = TaskResult.failure("abc123", e)

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


# --------------------------------------------------------------------------- #
# TaskRegistry (global default)                                                #
# --------------------------------------------------------------------------- #


class TestTaskRegistry:
    def test_decorator_bare_form_registers(self):
        @task
        def _ping() -> str:
            return "pong"

        assert _ping.task_name.endswith("_ping")
        entry = get_default_registry().get(_ping.task_name)
        assert entry is not None
        assert entry.fn is _ping
        assert entry.schema is None

    def test_decorator_with_name_and_schema(self):
        class _S(BaseModel):
            x: int

        @task(name="my.custom.fn", schema=_S)
        def _custom(s: _S) -> int:
            return s.x

        assert _custom.task_name == "my.custom.fn"
        entry = get_default_registry().get("my.custom.fn")
        assert entry is not None
        assert entry.schema is _S

    def test_missing_task_returns_none(self):
        assert get_default_registry().get("nonexistent") is None

    def test_names_sorted(self):
        names = get_default_registry().names()
        assert names == sorted(names)
