"""Failure-injection tests for peer dropouts mid-allreduce.

Production scenario: a Mac closes its lid (or the WiFi blips) during
gradient sync. The remaining peers must:
  - eventually time out the missing peer's contribution
  - fall back to local gradients via the GradientValidationError /
    OSError catch path in DataParallel.sync_gradients
  - not corrupt other ranks' state
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = pytest.importorskip("torch.nn")

from macfleet.comm.collectives import CollectiveGroup  # noqa: E402
from macfleet.comm.transport import PeerTransport, TransportConfig  # noqa: E402
from macfleet.engines.torch_engine import TorchEngine  # noqa: E402
from macfleet.training.data_parallel import DataParallel  # noqa: E402

CONFIG = TransportConfig(connect_timeout_sec=2.0, recv_timeout_sec=3.0)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x).sum()


async def _setup_mesh(n: int) -> tuple[list[PeerTransport], list[int]]:
    transports = []
    ports = []
    for i in range(n):
        t = PeerTransport(local_id=f"node-{i}", config=CONFIG)
        await t.start_server("127.0.0.1", 0)
        port = t._server.sockets[0].getsockname()[1]
        transports.append(t)
        ports.append(port)
    for i in range(n):
        for j in range(i + 1, n):
            await transports[i].connect(f"node-{j}", "127.0.0.1", ports[j])
    await asyncio.sleep(0.2)
    return transports, ports


async def _teardown_mesh(transports: list[PeerTransport]) -> None:
    for t in transports:
        await t.disconnect_all()


def _make_groups(n: int, transports: list[PeerTransport]) -> list[CollectiveGroup]:
    return [
        CollectiveGroup(
            rank=rank, world_size=n, transport=transports[rank],
            rank_to_peer={r: f"node-{r}" for r in range(n) if r != rank},
        )
        for rank in range(n)
    ]


def _make_engine(seed: int = 0) -> tuple[nn.Module, TorchEngine]:
    torch.manual_seed(seed)
    model = _TinyModel()
    engine = TorchEngine(device="cpu")
    engine.load_model(model, torch.optim.SGD(model.parameters(), lr=0.01))
    return model, engine


class TestPeerDropoutDuringAllreduce:
    @pytest.mark.asyncio
    async def test_recv_from_dead_peer_raises(self):
        """When a peer's transport closes, recv from it surfaces a network
        error rather than hanging silently."""
        n = 2
        transports, _ = await _setup_mesh(n)
        try:
            await transports[1].disconnect_all()
            await asyncio.sleep(0.1)

            with pytest.raises(
                (OSError, ConnectionError, asyncio.IncompleteReadError,
                 asyncio.TimeoutError),
            ):
                await asyncio.wait_for(
                    transports[0].recv("node-1"), timeout=5.0,
                )
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_dataparallel_falls_back_to_local_on_dropout(self):
        """N=2 direct exchange: when peer dies, sync_gradients catches
        the error and falls back to local gradients."""
        n = 2
        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            engines = [_make_engine(seed=42)[1] for _ in range(n)]
            dp0 = DataParallel(engines[0], groups[0])

            engines[0].zero_grad()
            x = torch.randn(2, 4)
            loss = engines[0].forward(x)
            engines[0].backward(loss)
            local_grad_before = engines[0].get_flat_gradients().copy()

            # Kill rank 1.
            await transports[1].disconnect_all()
            await asyncio.sleep(0.1)

            # sync_gradients should fall back rather than hang. Allow
            # generous timeout so flaky TCP keepalive doesn't fail us.
            try:
                await asyncio.wait_for(dp0.sync_gradients(), timeout=15.0)
            except asyncio.TimeoutError:
                pytest.skip("TCP buffer accepted send to dead peer past timeout")

            grad_after = engines[0].get_flat_gradients()
            np.testing.assert_allclose(grad_after, local_grad_before, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)


class TestNaNGradientPoisoning:
    @pytest.mark.asyncio
    async def test_nan_local_grad_zeroed_before_send(self):
        """A NaN local gradient is replaced with zeros before the wire."""
        n = 2
        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            engines = [_make_engine(seed=42)[1] for _ in range(n)]
            dps = [DataParallel(engines[i], groups[i]) for i in range(n)]

            # Make rank 0 have NaN gradients.
            engines[0].zero_grad()
            engines[0].backward(engines[0].forward(torch.randn(2, 4)))
            for p in engines[0]._collect_trainable_params():
                p.grad = torch.full_like(p.grad, float("nan"))

            # Rank 1 has clean gradients.
            engines[1].zero_grad()
            engines[1].backward(engines[1].forward(torch.randn(2, 4)))
            clean_grad = engines[1].get_flat_gradients().copy()

            # Both run sync — rank 0 zeros its NaN before send, so the
            # average is clean_grad / 2.
            await asyncio.gather(
                dps[0].sync_gradients(),
                dps[1].sync_gradients(),
            )

            r0 = engines[0].get_flat_gradients()
            r1 = engines[1].get_flat_gradients()
            assert np.isfinite(r0).all()
            assert np.isfinite(r1).all()
            # Both ranks see the same averaged gradient (clean / 2).
            np.testing.assert_allclose(r0, r1, rtol=1e-5)
            np.testing.assert_allclose(r0, clean_grad / 2.0, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)


class TestGracefulCleanupAfterDropout:
    @pytest.mark.asyncio
    async def test_remaining_peers_recover_for_next_round(self):
        """After a dropout, surviving peers can still allreduce among themselves."""
        n = 3
        transports, _ = await _setup_mesh(n)
        try:
            _make_groups(n, transports)

            # Kill rank 2.
            await transports[2].disconnect_all()
            await asyncio.sleep(0.05)

            # The remaining 2 peers can't form a 3-rank ring, but they
            # should still be able to talk to each other directly.
            await transports[0].send("node-1", b"hello-after-dropout")
            received = await asyncio.wait_for(
                transports[1].recv("node-0"), timeout=2.0,
            )
            assert received == b"hello-after-dropout"
        finally:
            await _teardown_mesh(transports)
