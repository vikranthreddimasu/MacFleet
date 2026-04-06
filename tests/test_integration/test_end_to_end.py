"""End-to-end integration tests for MacFleet v2.

Tests the complete distributed training pipeline over loopback:
    TorchEngine → DataParallel → CollectiveGroup → PeerTransport → TCP

These tests verify that:
1. Models converge when trained across multiple simulated nodes
2. Gradient sync produces correct averages
3. Models remain in sync across all nodes
4. The full training loop works end-to-end
5. The SDK Pool.train() works for single-node
"""

import asyncio

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.engines.torch_engine import TorchEngine
from macfleet.training.data_parallel import DataParallel
from macfleet.training.loop import TrainingConfig, training_loop
from macfleet.training.sampler import WeightedDistributedSampler

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


class _ClassifierModel(nn.Module):
    """Simple MLP for binary classification tests."""

    def __init__(self, input_dim: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x = batch
            return self.net(x).sum()

        logits = self.net(x)
        return self.loss_fn(logits, y)


def _make_classification_data(n: int = 200, seed: int = 42):
    """Linearly separable synthetic dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return TensorDataset(X, y)


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


async def _teardown_mesh(transports: list[PeerTransport]):
    for t in transports:
        await t.disconnect_all()


def _make_groups(transports: list[PeerTransport]) -> list[CollectiveGroup]:
    n = len(transports)
    return [
        CollectiveGroup(
            rank=i, world_size=n, transport=transports[i],
            rank_to_peer={r: f"node-{r}" for r in range(n) if r != i},
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# End-to-End: 2-node distributed training                                    #
# --------------------------------------------------------------------------- #


class TestDistributedTrainingTwoNodes:
    """Full distributed training pipeline over 2 loopback nodes."""

    @pytest.mark.asyncio
    async def test_two_node_training_convergence(self):
        """Train a classifier across 2 nodes and verify convergence."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            # Same model weights on both nodes (seed=42)
            torch.manual_seed(42)
            model0 = _ClassifierModel()
            torch.manual_seed(42)
            model1 = _ClassifierModel()

            opt0 = torch.optim.Adam(model0.parameters(), lr=0.01)
            opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)

            eng0 = TorchEngine(device="cpu")
            eng0.load_model(model0, opt0)
            eng1 = TorchEngine(device="cpu")
            eng1.load_model(model1, opt1)

            dp0 = DataParallel(eng0, groups[0])
            dp1 = DataParallel(eng1, groups[1])

            # Broadcast initial params
            await asyncio.gather(dp0.setup(), dp1.setup())

            # Same dataset, but each node sees different batches
            dataset = _make_classification_data(n=200, seed=42)
            sampler0 = WeightedDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=0)
            sampler1 = WeightedDistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=0)
            dl0 = DataLoader(dataset, batch_size=32, sampler=sampler0)
            dl1 = DataLoader(dataset, batch_size=32, sampler=sampler1)

            losses = []
            epochs = 5

            for epoch in range(epochs):
                sampler0.set_epoch(epoch)
                sampler1.set_epoch(epoch)
                epoch_loss = 0.0
                steps = 0

                for batch0, batch1 in zip(dl0, dl1):
                    # Forward + backward on each node
                    eng0.zero_grad()
                    eng1.zero_grad()

                    loss0 = model0(batch0)
                    eng0.backward(loss0)
                    loss1 = model1(batch1)
                    eng1.backward(loss1)

                    # Sync gradients
                    await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

                    # Step
                    eng0.step()
                    eng1.step()

                    epoch_loss += (loss0.item() + loss1.item()) / 2
                    steps += 1

                avg_loss = epoch_loss / max(steps, 1)
                losses.append(avg_loss)

            # Loss should decrease over training
            assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

            # Models should remain in sync
            params0 = eng0.get_flat_parameters()
            params1 = eng1.get_flat_parameters()
            np.testing.assert_allclose(params0, params1, rtol=1e-5)

        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_two_node_models_identical_after_training(self):
        """After N steps, both nodes have exactly the same model."""
        transports = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            torch.manual_seed(42)
            model0 = nn.Linear(4, 2, bias=False)
            torch.manual_seed(42)
            model1 = nn.Linear(4, 2, bias=False)

            eng0 = TorchEngine(device="cpu")
            eng0.load_model(model0, torch.optim.SGD(model0.parameters(), lr=0.1))
            eng1 = TorchEngine(device="cpu")
            eng1.load_model(model1, torch.optim.SGD(model1.parameters(), lr=0.1))

            dp0 = DataParallel(eng0, groups[0])
            dp1 = DataParallel(eng1, groups[1])
            await asyncio.gather(dp0.setup(), dp1.setup())

            # 10 steps with different data
            for step in range(10):
                x0 = torch.randn(8, 4)
                x1 = torch.randn(8, 4)

                eng0.zero_grad()
                eng1.zero_grad()

                loss0 = model0(x0).sum()
                eng0.backward(loss0)
                loss1 = model1(x1).sum()
                eng1.backward(loss1)

                await asyncio.gather(dp0.sync_gradients(), dp1.sync_gradients())

                eng0.step()
                eng1.step()

            # Parameters must be identical
            p0 = eng0.get_flat_parameters()
            p1 = eng1.get_flat_parameters()
            np.testing.assert_allclose(p0, p1, rtol=1e-5)

        finally:
            await _teardown_mesh(transports)


# --------------------------------------------------------------------------- #
# End-to-End: 3-node distributed training                                    #
# --------------------------------------------------------------------------- #


class TestDistributedTrainingThreeNodes:
    @pytest.mark.asyncio
    async def test_three_node_gradient_sync(self):
        """3 nodes produce correctly averaged gradients."""
        transports = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            engines = []
            for _ in range(3):
                torch.manual_seed(42)
                model = nn.Linear(4, 2, bias=False)
                eng = TorchEngine(device="cpu")
                eng.load_model(model, torch.optim.SGD(model.parameters(), lr=0.1))
                engines.append(eng)

            dps = [DataParallel(engines[i], groups[i]) for i in range(3)]
            await asyncio.gather(*(dp.setup() for dp in dps))

            # Each node processes different data
            inputs = [torch.randn(8, 4) for _ in range(3)]
            grads_before = []

            for i in range(3):
                engines[i].zero_grad()
                loss = engines[i].model(inputs[i]).sum()
                engines[i].backward(loss)
                grads_before.append(engines[i].get_flat_gradients().copy())

            await asyncio.gather(*(dp.sync_gradients() for dp in dps))

            # All nodes should have the same averaged gradients
            expected = sum(grads_before) / 3.0
            for i in range(3):
                synced = engines[i].get_flat_gradients()
                np.testing.assert_allclose(synced, expected, rtol=5e-4)

        finally:
            await _teardown_mesh(transports)


# --------------------------------------------------------------------------- #
# End-to-End: Training Loop                                                   #
# --------------------------------------------------------------------------- #


class TestTrainingLoop:
    @pytest.mark.asyncio
    async def test_single_node_training_loop(self):
        """The training_loop function works end-to-end on single node."""
        torch.manual_seed(42)
        dataset = _make_classification_data(n=100)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = _ClassifierModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        engine = TorchEngine(device="cpu")
        engine.load_model(model, optimizer)

        # Single-node collective (no-op allreduce)
        transport = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=transport, rank_to_peer={})
        dp = DataParallel(engine, group)

        config = TrainingConfig(epochs=3, log_every_n_steps=100)

        result = await training_loop(engine, dp, dataloader, config)

        assert result.epochs_completed == 3
        assert result.total_steps > 0
        assert result.final_loss > 0
        assert result.total_time_sec > 0

        # Loss should decrease
        epoch_losses = [er.avg_loss for er in result.epoch_results]
        assert epoch_losses[-1] < epoch_losses[0]


# --------------------------------------------------------------------------- #
# End-to-End: SDK Pool.train()                                                #
# --------------------------------------------------------------------------- #


class TestPoolSDK:
    def test_pool_train_single_node(self):
        """Pool.train() works for single-node training."""
        from macfleet.sdk.pool import Pool

        torch.manual_seed(42)
        # Plain model that returns logits (not loss) — works with external loss_fn
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
        dataset = _make_classification_data(n=100)

        with Pool(engine="torch") as pool:
            result = pool.train(
                model=model,
                dataset=dataset,
                epochs=3,
                batch_size=32,
                lr=0.01,
                loss_fn=nn.CrossEntropyLoss(),
            )

        assert result["epochs"] == 3
        assert result["loss"] > 0
        assert result["time_sec"] > 0
        # Loss should decrease
        assert result["loss_history"][-1] < result["loss_history"][0]

    def test_pool_train_tuple_dataset(self):
        """Pool.train() accepts (X, y) tuples."""
        from macfleet.sdk.pool import Pool

        torch.manual_seed(42)
        X = torch.randn(100, 4)
        y = (X[:, 0] > 0).long()

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

        with Pool() as pool:
            result = pool.train(
                model=model,
                dataset=(X, y),
                epochs=2,
                batch_size=32,
                loss_fn=nn.CrossEntropyLoss(),
            )

        assert result["epochs"] == 2

    def test_pool_requires_join(self):
        """Pool.train() raises if not joined."""
        from macfleet.sdk.pool import Pool

        pool = Pool()
        with pytest.raises(RuntimeError, match="Must join"):
            pool.train(model=nn.Linear(1, 1), dataset=[])


# --------------------------------------------------------------------------- #
# End-to-End: Training with weighted sampler                                  #
# --------------------------------------------------------------------------- #


class TestWeightedTraining:
    @pytest.mark.asyncio
    async def test_heterogeneous_batch_allocation(self):
        """Strong node (weight=0.7) gets more samples than weak (weight=0.3)."""
        dataset = _make_classification_data(n=100)

        sampler_strong = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=[0.7, 0.3], shuffle=False
        )
        sampler_weak = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=1, weights=[0.7, 0.3], shuffle=False
        )

        strong_samples = list(sampler_strong)
        weak_samples = list(sampler_weak)

        # Strong gets 70%, weak gets 30%
        assert len(strong_samples) == 70
        assert len(weak_samples) == 30
        # No overlap
        assert len(set(strong_samples) & set(weak_samples)) == 0
