"""Tests for TaskDispatcher and TaskWorker over loopback TCP.

Integration tests that verify the full task dispatch pipeline:
Coordinator (TaskDispatcher) → PeerTransport → Worker (TaskWorker)

v2.2 PR 7: tasks registered via @macfleet.task are dispatched by name
instead of cloudpickled inline. Closure-capture tests are removed because
closures are no longer supported — they were the exact wire-RCE vector
Issue 20 fixed.
"""

import asyncio

import pytest

from macfleet import task
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.compute.dispatch import TaskDispatcher
from macfleet.compute.models import RemoteTaskError
from macfleet.compute.worker import TaskWorker

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


# --------------------------------------------------------------------------- #
# Tasks registered at module level (so the worker process can import them)    #
# --------------------------------------------------------------------------- #


@task
def times_two(x: int) -> int:
    return x * 2


@task
def pow_two(x: int) -> int:
    return x ** 2


@task
def times_ten(x: int) -> int:
    return x * 10


@task
def bad_fn(x: int) -> int:
    raise ValueError(f"bad value: {x}")


async def _setup_pair() -> tuple[PeerTransport, PeerTransport, int]:
    """Create a connected coordinator-worker transport pair over loopback."""
    coordinator = PeerTransport(local_id="coordinator", config=CONFIG)
    worker = PeerTransport(local_id="worker-0", config=CONFIG)

    await worker.start_server("127.0.0.1", 0)
    port = worker._server.sockets[0].getsockname()[1]
    await coordinator.connect("worker-0", "127.0.0.1", port)
    await asyncio.sleep(0.1)

    return coordinator, worker, port


async def _teardown(coordinator: PeerTransport, worker: PeerTransport) -> None:
    await coordinator.disconnect_all()
    await worker.disconnect_all()


class TestDispatcherWorkerIntegration:
    """Full pipeline: dispatcher → transport → worker → transport → dispatcher."""

    @pytest.mark.asyncio
    async def test_submit_simple_function(self):
        """Submit a registered function, get the result back."""
        coordinator, worker, _ = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coordinator, ["worker-0"])
            tw = TaskWorker(worker, "coordinator", max_workers=1)

            await dispatcher.start()
            await tw.start()

            future = await dispatcher.submit(times_two, 21, timeout=5.0)
            result = await future.result(timeout=5.0)
            assert result == 42

            await tw.stop()
            await dispatcher.stop()
        finally:
            await _teardown(coordinator, worker)

    @pytest.mark.asyncio
    async def test_map_returns_ordered_results(self):
        """map() returns results in the same order as input."""
        coordinator, worker, _ = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coordinator, ["worker-0"])
            tw = TaskWorker(worker, "coordinator", max_workers=2)

            await dispatcher.start()
            await tw.start()

            results = await dispatcher.map(
                pow_two,
                [1, 2, 3, 4, 5],
                timeout=5.0,
            )
            assert results == [1, 4, 9, 16, 25]

            await tw.stop()
            await dispatcher.stop()
        finally:
            await _teardown(coordinator, worker)

    @pytest.mark.asyncio
    async def test_worker_exception_becomes_remote_error(self):
        """An exception in the worker function surfaces as RemoteTaskError."""
        coordinator, worker, _ = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coordinator, ["worker-0"])
            tw = TaskWorker(worker, "coordinator", max_workers=1)

            await dispatcher.start()
            await tw.start()

            future = await dispatcher.submit(bad_fn, 99, timeout=5.0)
            with pytest.raises(RemoteTaskError) as exc_info:
                await future.result(timeout=5.0)

            assert "bad value: 99" in exc_info.value.remote_traceback

            await tw.stop()
            await dispatcher.stop()
        finally:
            await _teardown(coordinator, worker)

    @pytest.mark.asyncio
    async def test_unregistered_callable_rejected_before_dispatch(self):
        """A bare lambda has no task_name → dispatcher.submit() refuses."""
        coordinator, worker, _ = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coordinator, ["worker-0"])
            await dispatcher.start()

            with pytest.raises(ValueError, match="not registered"):
                await dispatcher.submit(lambda x: x + 1, 5, timeout=5.0)

            await dispatcher.stop()
        finally:
            await _teardown(coordinator, worker)

    @pytest.mark.asyncio
    async def test_map_empty_iterable(self):
        """map() with empty input returns empty list without touching network."""
        coordinator, worker, _ = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coordinator, ["worker-0"])
            await dispatcher.start()

            results = await dispatcher.map(times_two, [], timeout=1.0)
            assert results == []

            await dispatcher.stop()
        finally:
            await _teardown(coordinator, worker)

    @pytest.mark.asyncio
    async def test_multiple_tasks_round_robin(self):
        """With 2 workers, tasks alternate between them."""
        coord = PeerTransport(local_id="coord", config=CONFIG)
        w0 = PeerTransport(local_id="w-0", config=CONFIG)
        w1 = PeerTransport(local_id="w-1", config=CONFIG)

        await w0.start_server("127.0.0.1", 0)
        port0 = w0._server.sockets[0].getsockname()[1]
        await w1.start_server("127.0.0.1", 0)
        port1 = w1._server.sockets[0].getsockname()[1]

        await coord.connect("w-0", "127.0.0.1", port0)
        await coord.connect("w-1", "127.0.0.1", port1)
        await asyncio.sleep(0.1)

        try:
            dispatcher = TaskDispatcher(coord, ["w-0", "w-1"])
            tw0 = TaskWorker(w0, "coord", max_workers=1)
            tw1 = TaskWorker(w1, "coord", max_workers=1)

            await dispatcher.start()
            await tw0.start()
            await tw1.start()

            results = await dispatcher.map(
                times_ten,
                [1, 2, 3, 4],
                timeout=5.0,
            )
            assert results == [10, 20, 30, 40]

            await tw0.stop()
            await tw1.stop()
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
            await w0.disconnect_all()
            await w1.disconnect_all()


class TestDispatcherEdgeCases:
    def test_no_workers_raises(self):
        """submit() with no workers raises immediately."""
        transport = PeerTransport(local_id="solo", config=CONFIG)
        dispatcher = TaskDispatcher(transport, [])

        async def run():
            with pytest.raises(RuntimeError, match="No workers"):
                await dispatcher.submit(times_two, 42)

        asyncio.get_event_loop().run_until_complete(run())
