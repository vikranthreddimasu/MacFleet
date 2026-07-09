"""End-to-end tests for distributed Pool.train wiring.

Runs the module-level distributed runners (`_distributed_train_torch`,
`_distributed_train_mlx`) with N ranks as concurrent coroutines on one
event loop over loopback TCP — the same interleaving real SPMD nodes
produce, without needing mDNS or multiple processes.

Pass condition mirrors tools/two_mac_real_train.py: identical final
parameter hashes on every rank, decreasing loss, matching step counts.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from macfleet.sdk.pool import (  # noqa: E402
    Pool,
    _distributed_train_torch,
)
from macfleet.training.mesh import NodeSpec  # noqa: E402


def _free_ports(n: int) -> list[int]:
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _make_specs(n: int) -> list[NodeSpec]:
    ports = _free_ports(n)
    return [
        NodeSpec(node_id=f"node-{i}", ip_address="127.0.0.1", data_port=ports[i])
        for i in range(n)
    ]


def _make_model(seed: int) -> nn.Module:
    """Same architecture, different init per rank — dp.setup() must
    converge them via the rank-0 broadcast."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))


def _make_dataset(n: int = 512) -> tuple:
    torch.manual_seed(7)
    X = torch.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return (X, y)


class TestDistributedTorchTrain:
    @pytest.mark.asyncio
    async def test_two_rank_params_sync_and_convergence(self):
        specs = _make_specs(2)
        dataset = _make_dataset()
        epochs = 4

        async def _run(i: int) -> dict:
            return await _distributed_train_torch(
                local_id=f"node-{i}",
                nodes=specs,
                model=_make_model(seed=100 + i),  # divergent init
                dataset=dataset,
                epochs=epochs,
                batch_size=64,  # global → 32 per rank
                lr=0.01,
                loss_fn=nn.CrossEntropyLoss(),
                device="cpu",
                rendezvous_timeout_sec=15.0,
            )

        r0, r1 = await asyncio.gather(_run(0), _run(1))

        # Both ranks ended with byte-identical parameters.
        assert r0["params_sha256"] == r1["params_sha256"]
        # Rank/world metadata present and correct.
        assert {r0["rank"], r1["rank"]} == {0, 1}
        assert r0["world_size"] == r1["world_size"] == 2
        # Identical step counts (the no-deadlock invariant).
        assert r0["steps"] == r1["steps"] > 0
        # Training actually learned the (trivially separable) task.
        assert len(r0["loss_history"]) == epochs
        assert r0["loss_history"][-1] < r0["loss_history"][0]
        assert r1["loss_history"][-1] < r1["loss_history"][0]
        assert r0["avg_sync_time_sec"] > 0.0
        assert r0["unsynced_steps"] == 0
        assert r1["unsynced_steps"] == 0

    @pytest.mark.asyncio
    async def test_compression_path(self):
        """Distributed training with adaptive compression stays in sync."""
        specs = _make_specs(2)
        dataset = _make_dataset(256)

        async def _run(i: int) -> dict:
            return await _distributed_train_torch(
                local_id=f"node-{i}",
                nodes=specs,
                model=_make_model(seed=i),
                dataset=dataset,
                epochs=2,
                batch_size=64,
                lr=0.01,
                loss_fn=nn.CrossEntropyLoss(),
                device="cpu",
                compression="light",
                rendezvous_timeout_sec=15.0,
            )

        r0, r1 = await asyncio.gather(_run(0), _run(1))
        assert r0["params_sha256"] == r1["params_sha256"]
        assert r0["steps"] == r1["steps"] > 0
        assert r0["unsynced_steps"] == 0
        assert r1["unsynced_steps"] == 0


class TestPoolTrainRouting:
    def test_distributed_true_without_peers_raises(self):
        with Pool(open=True) as pool:
            with pytest.raises(RuntimeError, match="no live peers"):
                pool.train(
                    model=_make_model(0),
                    dataset=_make_dataset(128),
                    epochs=1,
                    distributed=True,
                )

    def test_default_stays_single_node_without_peers(self):
        """No peers → auto mode trains single-node and returns the
        legacy result shape (no rank/world keys)."""
        with Pool(open=True) as pool:
            result = pool.train(
                model=_make_model(0),
                dataset=_make_dataset(128),
                epochs=1,
                batch_size=32,
                loss_fn=nn.CrossEntropyLoss(),
                device="cpu",
            )
        assert "loss" in result and "steps" in result
        assert "rank" not in result

    def test_distributed_false_forces_single_node(self):
        with Pool(open=True) as pool:
            result = pool.train(
                model=_make_model(0),
                dataset=_make_dataset(128),
                epochs=1,
                batch_size=32,
                loss_fn=nn.CrossEntropyLoss(),
                device="cpu",
                distributed=False,
            )
        assert "rank" not in result


class TestDistributedMLXTrain:
    @pytest.mark.asyncio
    async def test_two_rank_mlx_params_sync(self):
        mx = pytest.importorskip("mlx.core")
        import mlx.nn as mlx_nn

        from macfleet.sdk.pool import _distributed_train_mlx

        specs = _make_specs(2)
        X, y = _make_dataset(256)
        dataset = (X.numpy(), y.numpy().astype("int32"))

        def _mlx_model(seed: int):
            class MLP(mlx_nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = mlx_nn.Linear(4, 16)
                    self.l2 = mlx_nn.Linear(16, 2)

                def __call__(self, x):
                    return self.l2(mx.maximum(self.l1(x), 0))

            mx.random.seed(seed)
            return MLP()

        def _loss_fn(model, inputs, targets):
            return mlx_nn.losses.cross_entropy(
                model(inputs), targets, reduction="mean"
            )

        async def _run(i: int) -> dict:
            return await _distributed_train_mlx(
                local_id=f"node-{i}",
                nodes=specs,
                model=_mlx_model(seed=i),
                dataset=dataset,
                epochs=2,
                batch_size=64,
                lr=0.01,
                loss_fn=_loss_fn,
                rendezvous_timeout_sec=15.0,
            )

        r0, r1 = await asyncio.gather(_run(0), _run(1))
        assert r0["params_sha256"] == r1["params_sha256"]
        assert r0["steps"] == r1["steps"] > 0
        assert r0["loss_history"][-1] < r0["loss_history"][0]
        assert r0["unsynced_steps"] == 0
        assert r1["unsynced_steps"] == 0
