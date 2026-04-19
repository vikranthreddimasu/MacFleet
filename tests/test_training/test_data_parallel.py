"""Tests for data parallel gradient synchronization.

Tests the full pipeline: TorchEngine → DataParallel → CollectiveGroup
over loopback TCP connections. Verifies that gradients are correctly
averaged and models stay in sync across nodes.
"""

import asyncio

import numpy as np
import pytest
import torch
import torch.nn as nn

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.engines.torch_engine import TorchEngine
from macfleet.training.data_parallel import DataParallel

# --------------------------------------------------------------------------- #
# Test models                                                                 #
# --------------------------------------------------------------------------- #


class _SimpleModel(nn.Module):
    """Deterministic model for gradient sync tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)  # 8 params

    def forward(self, x):
        return self.fc(x).sum()


def _make_model_and_engine(seed: int = 0) -> tuple[nn.Module, TorchEngine]:
    """Create a model + engine pair with a fixed seed."""
    torch.manual_seed(seed)
    model = _SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)
    return model, engine


# --------------------------------------------------------------------------- #
# Loopback mesh helpers (same as test_collectives)                           #
# --------------------------------------------------------------------------- #

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


async def _setup_mesh(n: int) -> list[PeerTransport]:
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
    return transports


async def _teardown_mesh(transports: list[PeerTransport]) -> None:
    for t in transports:
        await t.disconnect_all()


def _make_groups(transports: list[PeerTransport]) -> list[CollectiveGroup]:
    n = len(transports)
    groups = []
    for rank in range(n):
        rank_to_peer = {r: f"node-{r}" for r in range(n) if r != rank}
        groups.append(
            CollectiveGroup(rank=rank, world_size=n, transport=transports[rank], rank_to_peer=rank_to_peer)
        )
    return groups


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestDataParallelSingleNode:
    """DataParallel with world_size=1 should be a no-op."""

    @pytest.mark.asyncio
    async def test_sync_noop(self):
        _, engine = _make_model_and_engine()
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=t, rank_to_peer={})
        dp = DataParallel(engine, group)

        x = torch.randn(3, 4)
        loss = engine.forward(x)
        engine.backward(loss)

        grads_before = engine.get_flat_gradients().copy()
        elapsed = await dp.sync_gradients()
        grads_after = engine.get_flat_gradients()

        assert elapsed == 0.0
        np.testing.assert_array_equal(grads_before, grads_after)


class TestDataParallelTwoNodes:
    """Two-node gradient synchronization over loopback."""

    @pytest.mark.asyncio
    async def test_gradient_averaging(self):
        """Two nodes with different inputs produce averaged gradients."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            # Both engines start with THE SAME model (seed=42)
            _, engine0 = _make_model_and_engine(seed=42)
            _, engine1 = _make_model_and_engine(seed=42)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            # Each node gets a DIFFERENT input → different gradients
            torch.manual_seed(0)
            x0 = torch.randn(3, 4)
            torch.manual_seed(1)
            x1 = torch.randn(3, 4)

            # Forward + backward
            loss0 = engine0.forward(x0)
            engine0.backward(loss0)
            loss1 = engine1.forward(x1)
            engine1.backward(loss1)

            grads0 = engine0.get_flat_gradients().copy()
            grads1 = engine1.get_flat_gradients().copy()

            # Gradients should be DIFFERENT before sync
            assert not np.allclose(grads0, grads1)

            # Sync gradients
            await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

            # After sync, gradients should be the AVERAGE
            synced0 = engine0.get_flat_gradients()
            synced1 = engine1.get_flat_gradients()

            expected = (grads0 + grads1) / 2.0
            np.testing.assert_allclose(synced0, expected, rtol=1e-5)
            np.testing.assert_allclose(synced1, expected, rtol=1e-5)

            # Both nodes should have IDENTICAL gradients
            np.testing.assert_allclose(synced0, synced1, rtol=1e-6)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_models_stay_in_sync(self):
        """After gradient sync + step, both models have identical params."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            _, engine0 = _make_model_and_engine(seed=42)
            _, engine1 = _make_model_and_engine(seed=42)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            # Different inputs
            x0 = torch.randn(3, 4)
            x1 = torch.randn(3, 4)

            # Full training step
            engine0.zero_grad()
            engine1.zero_grad()

            loss0 = engine0.forward(x0)
            engine0.backward(loss0)
            loss1 = engine1.forward(x1)
            engine1.backward(loss1)

            await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

            engine0.step()
            engine1.step()

            # Parameters should be identical
            params0 = engine0.get_flat_parameters()
            params1 = engine1.get_flat_parameters()
            np.testing.assert_allclose(params0, params1, rtol=1e-6)
        finally:
            await _teardown_mesh(transports)


class TestDataParallelThreeNodes:
    """Three-node ring allreduce gradient sync."""

    @pytest.mark.asyncio
    async def test_gradient_averaging_three_nodes(self):
        transports = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            engines = [_make_model_and_engine(seed=42)[1] for _ in range(3)]
            dps = [DataParallel(engines[i], groups[i]) for i in range(3)]

            # Different inputs per node
            inputs = [torch.randn(3, 4) for _ in range(3)]
            grads_before = []

            for i in range(3):
                engines[i].zero_grad()
                loss = engines[i].forward(inputs[i])
                engines[i].backward(loss)
                grads_before.append(engines[i].get_flat_gradients().copy())

            # Sync
            await asyncio.gather(*(dp.sync_gradients() for dp in dps))

            expected = sum(grads_before) / 3.0
            for i in range(3):
                synced = engines[i].get_flat_gradients()
                np.testing.assert_allclose(synced, expected, rtol=5e-4)
        finally:
            await _teardown_mesh(transports)


class TestDataParallelBroadcast:
    """Test parameter broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_from_coordinator(self):
        """After broadcast, all nodes have rank 0's parameters."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            # DIFFERENT initial weights
            _, engine0 = _make_model_and_engine(seed=0)
            _, engine1 = _make_model_and_engine(seed=99)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            params0_before = engine0.get_flat_parameters().copy()
            params1_before = engine1.get_flat_parameters().copy()
            assert not np.allclose(params0_before, params1_before)

            # Broadcast from rank 0
            await asyncio.gather(
                dp0.broadcast_parameters(src=0),
                dp1.broadcast_parameters(src=0),
            )

            params0_after = engine0.get_flat_parameters()
            params1_after = engine1.get_flat_parameters()

            # Both should match rank 0's original params
            np.testing.assert_allclose(params0_after, params0_before, rtol=1e-6)
            np.testing.assert_allclose(params1_after, params0_before, rtol=1e-6)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_setup_broadcasts(self):
        """DataParallel.setup() broadcasts params by default."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            _, engine0 = _make_model_and_engine(seed=0)
            _, engine1 = _make_model_and_engine(seed=99)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            await asyncio.gather(dp0.setup(), dp1.setup())

            params0 = engine0.get_flat_parameters()
            params1 = engine1.get_flat_parameters()
            np.testing.assert_allclose(params0, params1, rtol=1e-6)
        finally:
            await _teardown_mesh(transports)


class TestDataParallelMetrics:
    @pytest.mark.asyncio
    async def test_sync_time_tracking(self):
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            _, engine0 = _make_model_and_engine(seed=42)
            _, engine1 = _make_model_and_engine(seed=42)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            # Do a training step
            for eng in [engine0, engine1]:
                eng.zero_grad()
                loss = eng.forward(torch.randn(3, 4))
                eng.backward(loss)

            await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

            assert dp0.avg_sync_time_sec > 0
            assert dp0._step_count == 1
        finally:
            await _teardown_mesh(transports)


# --------------------------------------------------------------------------- #
# Edge case and safety tests                                                   #
# --------------------------------------------------------------------------- #


class TestModelConsistencyValidation:
    """Verify that setup() rejects mismatched model architectures."""

    @pytest.mark.asyncio
    async def test_matching_models_pass(self):
        """Two nodes with the same model architecture pass validation."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)
            _, engine0 = _make_model_and_engine(seed=0)
            _, engine1 = _make_model_and_engine(seed=99)  # different weights, same arch

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            # Should not raise
            await asyncio.gather(dp0.setup(), dp1.setup())
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_mismatched_models_raise(self):
        """Two nodes with different architectures raise RuntimeError."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            # Node 0: 4->2 (8 params)
            _, engine0 = _make_model_and_engine(seed=0)

            # Node 1: 4->8 (32 params) — different architecture
            torch.manual_seed(0)
            big_model = nn.Linear(4, 8, bias=False)
            optimizer = torch.optim.SGD(big_model.parameters(), lr=0.01)
            engine1 = TorchEngine(device="cpu")
            engine1.load_model(big_model, optimizer)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            with pytest.raises(RuntimeError, match="Model architecture mismatch"):
                await asyncio.gather(dp0.setup(), dp1.setup())
        finally:
            await _teardown_mesh(transports)


class TestNaNGradientGuard:
    """Verify that NaN gradients are caught before polluting allreduce."""

    @pytest.mark.asyncio
    async def test_nan_gradients_zeroed(self):
        """NaN local gradients are replaced with zeros before allreduce."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            _, engine0 = _make_model_and_engine(seed=42)
            _, engine1 = _make_model_and_engine(seed=42)

            dp0 = DataParallel(engine0, groups[0])
            dp1 = DataParallel(engine1, groups[1])

            # Normal forward/backward on both
            for eng in [engine0, engine1]:
                eng.zero_grad()
                loss = eng.forward(torch.randn(3, 4))
                eng.backward(loss)

            # Inject NaN into node 0's gradients
            for param in engine0._trainable_params:
                param.grad = torch.full_like(param.grad, float('nan'))

            # Should not raise — NaN is caught and zeroed
            await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

            # Node 0 should get valid gradients (node 1's / 2)
            synced = engine0.get_flat_gradients()
            assert np.isfinite(synced).all(), "Synced gradients should be finite"
        finally:
            await _teardown_mesh(transports)


class TestEmptyGradientGuard:
    """Verify sync_gradients handles models with no trainable params."""

    @pytest.mark.asyncio
    async def test_empty_gradients_skip(self):
        """Model with no trainable params → sync returns 0.0."""
        _, engine = _make_model_and_engine(seed=0)
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=2, transport=t, rank_to_peer={1: "peer"})
        dp = DataParallel(engine, group)

        # Clear all parameters to simulate no-grad model
        for param in engine._trainable_params:
            param.requires_grad_(False)
        engine._trainable_params = [p for p in engine._model.parameters() if p.requires_grad]

        elapsed = await dp.sync_gradients()
        assert elapsed == 0.0
