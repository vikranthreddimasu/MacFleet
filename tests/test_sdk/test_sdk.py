"""Tests for MacFleet Python SDK."""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

import macfleet
from macfleet.sdk.decorators import distributed
from macfleet.sdk.pool import Pool

# ------------------------------------------------------------------ #
# Test models                                                        #
# ------------------------------------------------------------------ #


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ #
# Pool SDK                                                           #
# ------------------------------------------------------------------ #


class TestPoolSDK:
    def test_pool_context_manager(self):
        with Pool() as pool:
            assert pool._joined is True
        assert pool._joined is False

    def test_pool_train_torch(self):
        model = SimpleMLP()
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=100)

        with Pool(engine="torch") as pool:
            result = pool.train(
                model=model,
                dataset=(X, y),
                epochs=3,
                batch_size=32,
                loss_fn=nn.CrossEntropyLoss(),
            )

        assert "loss" in result
        assert "epochs" in result
        assert result["epochs"] == 3
        assert result["loss"] < 1.0  # should train somewhat

    def test_pool_train_requires_join(self):
        pool = Pool()
        with pytest.raises(RuntimeError, match="Must join"):
            pool.train(model=SimpleMLP(), dataset=(np.zeros((10, 4)), np.zeros(10)))

    def test_pool_train_invalid_engine(self):
        with Pool(engine="invalid") as pool:
            with pytest.raises(ValueError, match="not supported"):
                pool.train(model=SimpleMLP(), dataset=(np.zeros((10, 4)), np.zeros(10)))

    def test_pool_world_size(self):
        pool = Pool()
        assert pool.world_size == 1


# ------------------------------------------------------------------ #
# macfleet.train() convenience function                              #
# ------------------------------------------------------------------ #


class TestMacfleetTrain:
    def test_train_function(self):
        model = SimpleMLP()
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=50)

        result = macfleet.train(
            model=model,
            dataset=(X, y),
            epochs=2,
            batch_size=16,
            loss_fn=nn.CrossEntropyLoss(),
        )

        assert result["epochs"] == 2
        assert result["loss"] > 0


# ------------------------------------------------------------------ #
# @distributed decorator                                             #
# ------------------------------------------------------------------ #


class TestDistributed:
    def test_decorator_runs_function(self):
        called = [False]

        @distributed(engine="torch")
        def my_train():
            called[0] = True
            return 42

        result = my_train()
        assert called[0] is True
        assert result == 42

    def test_decorator_with_args(self):
        @distributed(engine="torch")
        def my_train(x, y=10):
            return x + y

        assert my_train(5) == 15
        assert my_train(5, y=20) == 25


# ------------------------------------------------------------------ #
# MLX SDK integration                                                #
# ------------------------------------------------------------------ #


mlx = pytest.importorskip("mlx.core", reason="MLX not installed")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")


class SimpleMLXModel(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = mlx_nn.Linear(4, 8)
        self.linear2 = mlx_nn.Linear(8, 2)

    def __call__(self, x):
        x = mlx_nn.relu(self.linear1(x))
        return self.linear2(x)


def mlx_loss_fn(model, x, y):
    logits = model(x)
    return mlx_nn.losses.cross_entropy(logits, y, reduction="mean")


class TestPoolMLX:
    def test_pool_train_mlx(self):
        model = SimpleMLXModel()
        mlx.eval(model.parameters())

        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.int32)

        with Pool(engine="mlx") as pool:
            result = pool.train(
                model=model,
                dataset=(X, y),
                epochs=3,
                batch_size=16,
                loss_fn=mlx_loss_fn,
            )

        assert result["epochs"] == 3
        assert result["loss"] > 0
        assert len(result["loss_history"]) == 3
