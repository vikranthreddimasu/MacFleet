"""Tests for MacFleet Python SDK."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import macfleet
from macfleet.sdk.decorators import distributed
from macfleet.sdk.pool import Pool
from macfleet.training.guards import TrainingConfigError

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
    @pytest.mark.parametrize("wrapped", [False, True])
    @pytest.mark.parametrize("custom_loss", [False, True])
    def test_input_only_dataset_uses_every_sample(self, wrapped, custom_loss):
        inputs = torch.ones(4, 1)
        dataset = torch.utils.data.TensorDataset(inputs) if wrapped else inputs
        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with Pool(open=True) as pool:
            result = pool.train(
                model, dataset, epochs=1, batch_size=4, optimizer=optimizer,
                loss_fn=(lambda output: output.mean()) if custom_loss else None,
                device="cpu",
            )
        assert result["steps"] == 1
        assert model.weight.item() == pytest.approx(0.9 if custom_loss else 0.6)

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

    def test_pool_train_rejects_invalid_training_options(self):
        model = SimpleMLP()
        dataset = (np.zeros((32, 4), dtype=np.float32), np.zeros(32, dtype=np.int64))

        cases = [
            {"epochs": 0},
            {"batch_size": 0},
            {"lr": 0},
            {"lr": float("nan")},
            {"compression": "typo"},
        ]

        with Pool(open=True) as pool:
            for kwargs in cases:
                with pytest.raises(TrainingConfigError):
                    pool.train(
                        model=model,
                        dataset=dataset,
                        loss_fn=nn.CrossEntropyLoss(),
                        **kwargs,
                    )

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
    def test_decorator_rejects_false_distributed_execution(self):
        called = []

        @distributed(engine="torch")
        def my_train():
            called.append(True)

        with pytest.raises(NotImplementedError, match="Pool"):
            my_train()
        assert called == []
