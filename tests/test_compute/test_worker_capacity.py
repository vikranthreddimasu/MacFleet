"""A timed-out thread still occupies worker capacity until it really exits."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

from macfleet import task
from macfleet.comm.protocol import MessageType
from macfleet.compute.models import TaskResult, TaskSpec
from macfleet.compute.worker import TaskWorker


async def test_timeout_does_not_create_an_unbounded_execution_queue():
    incoming = asyncio.Queue()
    outgoing = asyncio.Queue()
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    @task(name="tests.worker_capacity.blocked")
    def blocked():
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=5)
        return 42

    async def receive(**kwargs):
        return await incoming.get()

    async def send(peer, payload, **kwargs):
        await outgoing.put(TaskResult.unpack(payload))

    transport = SimpleNamespace(get_connection=lambda peer: SimpleNamespace(recv_message=receive), send=send)
    worker = TaskWorker(transport, "coordinator", max_workers=1)
    await worker.start()
    try:
        incoming.put_nowait(SimpleNamespace(
            msg_type=MessageType.TASK, payload=TaskSpec.from_call(blocked, timeout=0.05).pack(),
        ))
        await asyncio.wait_for(started.wait(), 1)
        result = await asyncio.wait_for(outgoing.get(), 1)
        assert not result.ok
        for _ in range(20):
            incoming.put_nowait(SimpleNamespace(
                msg_type=MessageType.TASK, payload=TaskSpec.from_call(blocked, timeout=1).pack(),
            ))
        await asyncio.sleep(0.05)
        assert incoming.qsize() >= 19
        assert "may still be running" in result.error
        await asyncio.wait_for(worker.stop(), 0.5)
    finally:
        release.set()
        await worker.stop()


async def test_unauthorized_work_never_enters_executor():
    @task(name="tests.worker_capacity.local_only", remote=False)
    def local_only():
        raise AssertionError("must not run")

    incoming = asyncio.Queue()
    outgoing = asyncio.Queue()

    async def receive(**kwargs):
        return await incoming.get()

    async def send(peer, payload, **kwargs):
        await outgoing.put(TaskResult.unpack(payload))

    transport = SimpleNamespace(get_connection=lambda peer: SimpleNamespace(recv_message=receive), send=send)
    worker = TaskWorker(transport, "coordinator", max_workers=1)
    await worker.start()
    try:
        with patch.object(worker._executor, "submit", side_effect=AssertionError("unauthorized work was queued")):
            incoming.put_nowait(SimpleNamespace(
                msg_type=MessageType.TASK, payload=TaskSpec.from_call(local_only).pack(),
            ))
            result = await asyncio.wait_for(outgoing.get(), 1)
        assert not result.ok
        assert "not allowed to run remotely" in result.error
    finally:
        await worker.stop()


async def test_worker_shutdown_bounds_blocked_result_delivery():
    @task(name="tests.worker_capacity.completed")
    def completed():
        return 42

    incoming = asyncio.Queue()
    sending = asyncio.Event()
    send_cancelled = asyncio.Event()

    async def receive(**kwargs):
        return await incoming.get()

    async def blocked_send(*args, **kwargs):
        sending.set()
        try:
            await asyncio.Event().wait()
        finally:
            send_cancelled.set()

    transport = SimpleNamespace(get_connection=lambda peer: SimpleNamespace(recv_message=receive), send=blocked_send)
    worker = TaskWorker(transport, "coordinator", max_workers=1)
    await worker.start()
    try:
        incoming.put_nowait(SimpleNamespace(
            msg_type=MessageType.TASK, payload=TaskSpec.from_call(completed, timeout=0.02).pack(),
        ))
        await asyncio.wait_for(sending.wait(), 1)
        await asyncio.wait_for(worker.stop(), 0.5)
        assert send_cancelled.is_set()
        assert not worker._inflight
        assert worker._executor is None
    finally:
        await worker.stop()
