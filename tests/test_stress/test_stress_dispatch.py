"""Stress tests for TaskDispatcher under parallel load.

Production scenario: a coordinator dispatches N tasks across M workers
in parallel. Tasks complete out of order. Dispatcher must:
  - resolve every TaskFuture exactly once
  - never leak pending entries
  - handle worker disconnects without stranding other-worker tasks
"""

from __future__ import annotations

import asyncio

import pytest

from macfleet import task
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.compute.dispatch import TaskDispatcher
from macfleet.compute.worker import TaskWorker

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


@task
def _stress_double(x: int) -> int:
    return x * 2


@task
def _stress_slow(x: int) -> int:
    import time
    time.sleep(0.01)
    return x + 1000


@task
def _stress_passthrough(s: str) -> str:
    return s.upper()


async def _setup_pair() -> tuple[PeerTransport, PeerTransport]:
    coord = PeerTransport(local_id="coord", config=CONFIG)
    worker = PeerTransport(local_id="w-0", config=CONFIG)
    await worker.start_server("127.0.0.1", 0)
    port = worker._server.sockets[0].getsockname()[1]
    await coord.connect("w-0", "127.0.0.1", port)
    await asyncio.sleep(0.1)
    return coord, worker


async def _setup_pool(n_workers: int) -> tuple[
    PeerTransport, list[PeerTransport], list[str],
]:
    coord = PeerTransport(local_id="coord", config=CONFIG)
    workers: list[PeerTransport] = []
    worker_ids: list[str] = []
    for i in range(n_workers):
        wid = f"w-{i}"
        w = PeerTransport(local_id=wid, config=CONFIG)
        await w.start_server("127.0.0.1", 0)
        port = w._server.sockets[0].getsockname()[1]
        await coord.connect(wid, "127.0.0.1", port)
        workers.append(w)
        worker_ids.append(wid)
    await asyncio.sleep(0.1)
    return coord, workers, worker_ids


class TestDispatcherHighThroughput:
    @pytest.mark.asyncio
    async def test_500_tasks_one_worker(self):
        """Single worker, many tasks — verify ordering + no drops."""
        coord, worker = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coord, ["w-0"])
            tw = TaskWorker(worker, "coord", max_workers=4)
            await dispatcher.start()
            await tw.start()

            n = 500
            results = await dispatcher.map(
                _stress_double, list(range(n)), timeout=30.0,
            )
            assert results == [i * 2 for i in range(n)]
            assert dispatcher.pending_count == 0

            await tw.stop()
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
            await worker.disconnect_all()

    @pytest.mark.asyncio
    async def test_round_robin_balances_across_workers(self):
        n_workers = 4
        coord, workers, worker_ids = await _setup_pool(n_workers)
        try:
            dispatcher = TaskDispatcher(coord, worker_ids)
            tws = [
                TaskWorker(workers[i], "coord", max_workers=2)
                for i in range(n_workers)
            ]
            await dispatcher.start()
            for tw in tws:
                await tw.start()

            n = 200
            results = await dispatcher.map(
                _stress_double, list(range(n)), timeout=30.0,
            )
            assert results == [i * 2 for i in range(n)]

            for tw in tws:
                await tw.stop()
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
            for w in workers:
                await w.disconnect_all()


class TestDispatcherConcurrentSubmit:
    @pytest.mark.asyncio
    async def test_concurrent_submit_calls(self):
        """Many concurrent submit() calls — _pending mutations stay safe."""
        coord, worker = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coord, ["w-0"])
            tw = TaskWorker(worker, "coord", max_workers=4)
            await dispatcher.start()
            await tw.start()

            n = 100

            async def one(i: int) -> int:
                future = await dispatcher.submit(_stress_double, i, timeout=10.0)
                return await future.result(timeout=10.0)

            results = await asyncio.gather(*(one(i) for i in range(n)))
            assert sorted(results) == [i * 2 for i in range(n)]
            assert dispatcher.pending_count == 0

            await tw.stop()
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
            await worker.disconnect_all()


class TestDispatcherLeakDetection:
    @pytest.mark.asyncio
    async def test_pending_drains_after_completion(self):
        """After every task resolves, _pending must be empty."""
        coord, worker = await _setup_pair()
        try:
            dispatcher = TaskDispatcher(coord, ["w-0"])
            tw = TaskWorker(worker, "coord", max_workers=2)
            await dispatcher.start()
            await tw.start()

            for batch in range(5):
                _ = await dispatcher.map(
                    _stress_passthrough,
                    [f"item-{batch}-{i}" for i in range(20)],
                    timeout=10.0,
                )
                assert dispatcher.pending_count == 0
                # Internal task→worker map should also drain.
                assert len(dispatcher._task_to_worker) == 0

            await tw.stop()
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
            await worker.disconnect_all()

    @pytest.mark.asyncio
    async def test_stop_drains_remaining(self):
        """Dispatcher.stop() must resolve every still-pending future."""
        coord, worker = await _setup_pair()
        # Don't start the worker — submitted tasks will sit in _pending
        # because nothing replies.
        try:
            dispatcher = TaskDispatcher(coord, ["w-0"])
            await dispatcher.start()

            futures = []
            for i in range(20):
                future = await dispatcher.submit(_stress_double, i, timeout=30.0)
                futures.append(future)
            assert dispatcher.pending_count == 20

            await dispatcher.stop()

            # Every future resolved, no leak.
            assert dispatcher.pending_count == 0
            for f in futures:
                assert f.done
                # They resolve as failures because the dispatcher stopped.
                assert f._result is not None
                assert f._result.ok is False
        finally:
            await coord.disconnect_all()
            await worker.disconnect_all()


class TestDispatcherEmptyWorkerPool:
    @pytest.mark.asyncio
    async def test_no_workers_at_construction(self):
        coord = PeerTransport(local_id="coord", config=CONFIG)
        try:
            dispatcher = TaskDispatcher(coord, [])
            await dispatcher.start()  # no-op since no workers
            with pytest.raises(RuntimeError, match="No workers"):
                await dispatcher.submit(_stress_double, 1)
            await dispatcher.stop()
        finally:
            await coord.disconnect_all()
