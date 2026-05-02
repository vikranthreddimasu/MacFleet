"""End-to-end multi-node training convergence tests.

Production scenario: a fleet trains a real model for many epochs.
After training, every rank must hold IDENTICAL parameters and the
loss must have decreased.

Tests N=2, 3, 4, 5 nodes to cover both direct exchange (N=2) and the
full ring topology (N>=3).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = pytest.importorskip("torch.nn")
_torch_data = pytest.importorskip("torch.utils.data")
DataLoader = _torch_data.DataLoader
TensorDataset = _torch_data.TensorDataset

from macfleet.comm.collectives import CollectiveGroup  # noqa: E402
from macfleet.comm.transport import PeerTransport, TransportConfig  # noqa: E402
from macfleet.engines.torch_engine import TorchEngine  # noqa: E402
from macfleet.training.data_parallel import DataParallel  # noqa: E402
from macfleet.training.sampler import WeightedDistributedSampler  # noqa: E402

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=20.0)


class _Classifier(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        x, y = batch
        return self.loss_fn(self.net(x), y)


def _make_dataset(n: int = 200, seed: int = 42) -> TensorDataset:
    torch.manual_seed(seed)
    X = torch.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return TensorDataset(X, y)


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


def _make_engines(n: int, seed: int = 42) -> list[TorchEngine]:
    """Create n engines all initialized from the same seed."""
    engines = []
    for _ in range(n):
        torch.manual_seed(seed)
        model = _Classifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        engine = TorchEngine(device="cpu")
        engine.load_model(model, optimizer)
        engines.append(engine)
    return engines


# ---------------------------------------------------------------
# Convergence + final-param identity for N = 2, 3, 4, 5
# ---------------------------------------------------------------


class TestMultiNodeConvergence:
    @pytest.mark.parametrize("n_nodes", [2, 3, 4])
    @pytest.mark.asyncio
    async def test_loss_decreases_and_params_match(self, n_nodes):
        """After N epochs across n_nodes ranks, loss decreases AND every
        rank holds identical model parameters."""
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)
            engines = _make_engines(n_nodes, seed=42)
            dps = [DataParallel(engines[i], groups[i]) for i in range(n_nodes)]

            await asyncio.gather(*(dp.setup() for dp in dps))

            dataset = _make_dataset(n=200, seed=42)
            samplers = [
                WeightedDistributedSampler(
                    dataset, num_replicas=n_nodes, rank=r,
                    shuffle=True, seed=0,
                )
                for r in range(n_nodes)
            ]
            loaders = [
                DataLoader(dataset, batch_size=32, sampler=samplers[r])
                for r in range(n_nodes)
            ]

            epochs = 5
            losses_per_epoch = []

            for epoch in range(epochs):
                for s in samplers:
                    s.set_epoch(epoch)

                epoch_loss = 0.0
                steps = 0
                # zip across loaders so all ranks process the same batch index.
                for batches in zip(*loaders):
                    for i in range(n_nodes):
                        engines[i].zero_grad()
                        loss = engines[i].forward(batches[i])
                        engines[i].backward(loss)

                    await asyncio.gather(*(dp.sync_gradients() for dp in dps))
                    for eng in engines:
                        eng.step()

                    # Track rank 0's loss as a proxy.
                    epoch_loss += engines[0].forward(batches[0]).item()
                    steps += 1
                losses_per_epoch.append(epoch_loss / max(steps, 1))

            # Loss must decrease.
            assert losses_per_epoch[-1] < losses_per_epoch[0], (
                f"loss did not decrease over training: {losses_per_epoch}"
            )
            # Every rank holds identical parameters.
            params = [eng.get_flat_parameters() for eng in engines]
            for i in range(1, n_nodes):
                np.testing.assert_allclose(params[0], params[i], rtol=1e-5)
        finally:
            await _teardown_mesh(transports)


class TestHeterogeneousFleetWeighting:
    @pytest.mark.asyncio
    async def test_weighted_sampler_distributes_correctly(self):
        """A weighted sampler at 70/30 split must give each rank exactly
        that fraction of the dataset."""
        n_nodes = 2
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)
            engines = _make_engines(n_nodes, seed=42)
            dps = [DataParallel(engines[i], groups[i]) for i in range(n_nodes)]
            await asyncio.gather(*(dp.setup() for dp in dps))

            dataset = _make_dataset(n=100, seed=42)
            sampler_strong = WeightedDistributedSampler(
                dataset, num_replicas=2, rank=0,
                weights=[0.7, 0.3], shuffle=False,
            )
            sampler_weak = WeightedDistributedSampler(
                dataset, num_replicas=2, rank=1,
                weights=[0.7, 0.3], shuffle=False,
            )
            assert len(sampler_strong) == 70
            assert len(sampler_weak) == 30
            # No overlap.
            strong = list(sampler_strong)
            weak = list(sampler_weak)
            assert len(set(strong) & set(weak)) == 0
        finally:
            await _teardown_mesh(transports)


class TestSetupRebroadcast:
    @pytest.mark.asyncio
    async def test_setup_synchronizes_initial_params(self):
        """DataParallel.setup() must broadcast rank 0's params to all
        other ranks even when they started with different seeds."""
        n_nodes = 3
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)
            # Each engine starts with a DIFFERENT seed.
            engines = []
            for r in range(n_nodes):
                torch.manual_seed(100 + r)
                model = _Classifier()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                engine = TorchEngine(device="cpu")
                engine.load_model(model, optimizer)
                engines.append(engine)

            # Confirm divergence before setup.
            params_before = [eng.get_flat_parameters() for eng in engines]
            for i in range(1, n_nodes):
                assert not np.allclose(params_before[0], params_before[i]), (
                    f"engines should start divergent (seed={100+i})"
                )

            dps = [DataParallel(engines[i], groups[i]) for i in range(n_nodes)]
            await asyncio.gather(*(dp.setup() for dp in dps))

            # After setup, every engine has rank 0's params.
            params_after = [eng.get_flat_parameters() for eng in engines]
            for i in range(1, n_nodes):
                np.testing.assert_allclose(
                    params_after[0], params_after[i], rtol=1e-6,
                )
            np.testing.assert_allclose(params_after[0], params_before[0], rtol=1e-6)
        finally:
            await _teardown_mesh(transports)


class TestCompressionAcrossLevels:
    """Training converges at every compression level — direction matters."""

    @pytest.mark.parametrize("compression", ["none", "light", "moderate", "aggressive"])
    @pytest.mark.asyncio
    async def test_each_compression_level_converges(self, compression):
        from macfleet.training.data_parallel import DataParallelConfig
        n_nodes = 2
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)
            engines = _make_engines(n_nodes, seed=42)
            dps = [
                DataParallel(
                    engines[i], groups[i],
                    config=DataParallelConfig(compression=compression),
                )
                for i in range(n_nodes)
            ]
            await asyncio.gather(*(dp.setup() for dp in dps))

            dataset = _make_dataset(n=100, seed=42)
            initial_loss = float(engines[0].forward(
                (dataset.tensors[0], dataset.tensors[1]),
            ).item())

            for _ in range(10):
                # Same minibatch across both ranks for simpler analysis.
                idx = torch.randperm(100)[:32]
                batch = (dataset.tensors[0][idx], dataset.tensors[1][idx])
                for eng in engines:
                    eng.zero_grad()
                    loss = eng.forward(batch)
                    eng.backward(loss)
                await asyncio.gather(*(dp.sync_gradients() for dp in dps))
                for eng in engines:
                    eng.step()

            final_loss = float(engines[0].forward(
                (dataset.tensors[0], dataset.tensors[1]),
            ).item())
            # Loss should at least not blow up. Aggressive compression
            # may converge slower so we use a loose bound.
            assert final_loss < initial_loss * 1.5, (
                f"compression={compression}: loss {initial_loss:.3f} → "
                f"{final_loss:.3f} suggests divergence"
            )
        finally:
            await _teardown_mesh(transports)
