"""Successive training calls must not attach to a previous mesh's listener."""

import asyncio
import socket

import numpy as np
import pytest

from macfleet.comm.transport import PeerTransport
from macfleet.sdk.pool import _distributed_train_mlx, _distributed_train_torch
from macfleet.security.auth import SecurityConfig
from macfleet.training.mesh import Mesh, NodeSpec


@pytest.mark.parametrize("engine", ["torch", "mlx"])
async def test_next_training_waits_for_previous_mesh_listeners(monkeypatch, engine):
    if engine == "torch":
        torch = pytest.importorskip("torch")
        models = [torch.nn.Linear(4, 2) for _ in range(3)]
        runner = _distributed_train_torch
        options = {"device": "cpu", "loss_fn": torch.nn.CrossEntropyLoss()}
    else:
        nn = pytest.importorskip("mlx.nn")
        models = [nn.Linear(4, 2) for _ in range(3)]
        runner = _distributed_train_mlx

        def loss_fn(model, inputs, targets):
            return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")

        options = {"loss_fn": loss_fn}

    sockets = [socket.socket() for _ in range(3)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        nodes = [NodeSpec(f"node-{rank}", "127.0.0.1", sock.getsockname()[1])
                 for rank, sock in enumerate(sockets)]
    finally:
        for sock in sockets:
            sock.close()

    security = SecurityConfig(token="repeated-training-test-token")
    inputs = np.random.default_rng(42).normal(size=(12, 4)).astype(np.float32)
    targets = (inputs[:, 0] > 0).astype(np.int32)
    slow_closing = asyncio.Event()
    next_connect_attempted = asyncio.Event()
    original_close = Mesh.close
    original_connect = PeerTransport.connect
    delayed = False

    async def delayed_close(mesh):
        nonlocal delayed
        if mesh.rank == 2 and not delayed:
            delayed = True
            slow_closing.set()
            await next_connect_attempted.wait()
        await original_close(mesh)

    async def observed_connect(transport, peer_id, *args, **kwargs):
        try:
            return await original_connect(transport, peer_id, *args, **kwargs)
        finally:
            if transport.local_id == "node-0" and peer_id == "node-2" and slow_closing.is_set():
                # Old code connects to the lingering listener; fixed code gets
                # connection refused and retries after rank 2 opens its new mesh.
                next_connect_attempted.set()

    monkeypatch.setattr(Mesh, "close", delayed_close)
    monkeypatch.setattr(PeerTransport, "connect", observed_connect)

    async def train_rank(rank):
        reports = []
        for generation in range(2):
            if generation == 1 and rank == 0:
                await slow_closing.wait()
            reports.append(await runner(
                local_id=nodes[rank].node_id, nodes=nodes, security=security,
                model=models[rank], dataset=(inputs, targets), epochs=1,
                batch_size=6, lr=0.1, rendezvous_timeout_sec=3, **options,
            ))
        return reports

    tasks = [asyncio.create_task(train_rank(rank)) for rank in range(3)]
    try:
        runs = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        assert next_connect_attempted.is_set()
        for reports in zip(*runs, strict=True):
            assert len({report["params_sha256"] for report in reports}) == 1
            assert all(report["steps"] == 2 and not report["degraded"] for report in reports)
    finally:
        next_connect_attempted.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
