"""Tests for MLX training engine.

All tests are skipped if MLX is not installed (e.g., on Linux/CI).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if MLX is not available
mlx = pytest.importorskip("mlx.core", reason="MLX not installed")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")

from macfleet.engines.base import EngineType
from macfleet.engines.mlx_engine import MLXEngine, _flatten_params, _unflatten_params

# ------------------------------------------------------------------ #
# Test models                                                        #
# ------------------------------------------------------------------ #


class SimpleMLXModel(nn.Module):
    """Simple 2-layer MLP for testing."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)

    def __call__(self, x):
        x = nn.relu(self.linear1(x))
        return self.linear2(x)


def cross_entropy_loss(model, x, y):
    """Loss function for testing: model forward + cross entropy."""
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def mse_loss(model, x, y):
    """MSE loss for regression testing."""
    pred = model(x)
    return mlx.mean((pred - y) ** 2)


@pytest.fixture
def simple_model():
    """Create a simple MLX model."""
    model = SimpleMLXModel()
    mlx.eval(model.parameters())
    return model


@pytest.fixture
def engine_with_model(simple_model):
    """Create an MLXEngine loaded with a simple model."""
    engine = MLXEngine()
    optimizer = optim.Adam(learning_rate=0.01)
    engine.load_model(simple_model, optimizer, loss_fn=cross_entropy_loss)
    return engine


# ------------------------------------------------------------------ #
# Parameter flattening tests                                         #
# ------------------------------------------------------------------ #


class TestFlattenParams:
    def test_flatten_simple_model(self, simple_model):
        params = simple_model.parameters()
        flat = _flatten_params(params)
        # SimpleMLXModel has linear1.weight, linear1.bias, linear2.weight, linear2.bias
        names = [name for name, _ in flat]
        assert any("linear1" in n and "weight" in n for n in names)
        assert any("linear2" in n and "weight" in n for n in names)
        assert len(flat) >= 4  # at least 4 param arrays

    def test_roundtrip_unflatten(self, simple_model):
        params = simple_model.parameters()
        flat = _flatten_params(params)
        rebuilt = _unflatten_params(flat, params)

        # Rebuilt should have same structure
        assert isinstance(rebuilt, dict)
        flat_rebuilt = _flatten_params(rebuilt)
        assert len(flat_rebuilt) == len(flat)

        for (n1, a1), (n2, a2) in zip(flat, flat_rebuilt):
            assert n1 == n2
            np.testing.assert_array_equal(np.array(a1), np.array(a2))


# ------------------------------------------------------------------ #
# Engine lifecycle tests                                             #
# ------------------------------------------------------------------ #


class TestMLXEngineLifecycle:
    def test_init(self):
        engine = MLXEngine()
        assert engine.model is None
        assert engine.param_count() == 0

    def test_load_model(self, simple_model):
        engine = MLXEngine()
        optimizer = optim.Adam(learning_rate=0.01)
        engine.load_model(simple_model, optimizer, loss_fn=cross_entropy_loss)

        assert engine.model is simple_model
        assert engine.param_count() > 0
        assert len(engine._param_names) >= 4

    def test_capabilities(self):
        engine = MLXEngine()
        caps = engine.capabilities
        assert caps.engine_type == EngineType.MLX
        assert "float32" in caps.supported_dtypes
        assert "bfloat16" in caps.supported_dtypes

    def test_profile(self, engine_with_model):
        profile = engine_with_model.profile()
        assert profile.mlx_available is True
        assert "mlx" in profile.node_id

    def test_memory_usage(self, engine_with_model):
        mem = engine_with_model.memory_usage_gb()
        assert mem > 0.0
        assert mem < 1.0  # tiny model


# ------------------------------------------------------------------ #
# Forward / backward / step tests                                   #
# ------------------------------------------------------------------ #


class TestMLXEngineTraining:
    def test_forward(self, engine_with_model):
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        assert float(loss) > 0.0

    def test_backward_produces_gradients(self, engine_with_model):
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        engine_with_model.backward(loss)

        assert engine_with_model._grads is not None
        flat_grads = _flatten_params(engine_with_model._grads)
        assert len(flat_grads) > 0
        # At least one gradient should be non-zero
        assert any(np.any(np.array(g) != 0) for _, g in flat_grads)

    def test_step_updates_parameters(self, engine_with_model):
        # Record initial params
        initial_params = engine_with_model.get_flat_parameters().copy()

        # Forward + backward + step
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        engine_with_model.backward(loss)
        engine_with_model.step()

        # Params should have changed
        new_params = engine_with_model.get_flat_parameters()
        assert not np.allclose(initial_params, new_params)

    def test_zero_grad(self, engine_with_model):
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        engine_with_model.backward(loss)
        assert engine_with_model._grads is not None

        engine_with_model.zero_grad()
        assert engine_with_model._grads is None

    def test_training_reduces_loss(self, engine_with_model):
        """Multiple training steps should reduce loss."""
        x = mlx.random.normal((32, 4))
        y = mlx.array([0, 1] * 16)

        losses = []
        for _ in range(20):
            engine_with_model.zero_grad()
            loss = engine_with_model.forward((x, y))
            engine_with_model.backward(loss)
            engine_with_model.step()
            losses.append(float(loss))

        # Loss should decrease (first vs last few steps)
        assert np.mean(losses[-5:]) < np.mean(losses[:5])


# ------------------------------------------------------------------ #
# Flat gradient interface (DataParallel integration)                 #
# ------------------------------------------------------------------ #


class TestMLXFlatInterface:
    def test_get_flat_gradients(self, engine_with_model):
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        engine_with_model.backward(loss)

        flat = engine_with_model.get_flat_gradients()
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert flat.shape[0] == engine_with_model.param_count()

    def test_get_flat_gradients_no_backward(self, engine_with_model):
        flat = engine_with_model.get_flat_gradients()
        assert len(flat) == 0

    def test_apply_flat_gradients_roundtrip(self, engine_with_model):
        x = mlx.random.normal((8, 4))
        y = mlx.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss = engine_with_model.forward((x, y))
        engine_with_model.backward(loss)

        original = engine_with_model.get_flat_gradients().copy()

        # Simulate allreduce (multiply by 2)
        doubled = original * 2.0
        engine_with_model.apply_flat_gradients(doubled)

        after = engine_with_model.get_flat_gradients()
        np.testing.assert_allclose(after, doubled, rtol=1e-5)

    def test_get_flat_parameters(self, engine_with_model):
        flat = engine_with_model.get_flat_parameters()
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert flat.shape[0] == engine_with_model.param_count()

    def test_apply_flat_parameters_roundtrip(self, engine_with_model):
        original = engine_with_model.get_flat_parameters().copy()

        # Zero out and restore
        zeros = np.zeros_like(original)
        engine_with_model.apply_flat_parameters(zeros)
        assert np.allclose(engine_with_model.get_flat_parameters(), 0.0)

        engine_with_model.apply_flat_parameters(original)
        np.testing.assert_allclose(
            engine_with_model.get_flat_parameters(), original, rtol=1e-5
        )


# ------------------------------------------------------------------ #
# Dict-based gradient interface                                      #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# State serialization (checkpointing)                                #
# ------------------------------------------------------------------ #


class TestMLXCheckpointing:
    def test_state_dict_roundtrip(self, engine_with_model):
        original_params = engine_with_model.get_flat_parameters().copy()

        # Serialize
        state = engine_with_model.state_dict()
        assert isinstance(state, bytes)
        assert len(state) > 0

        # Modify params
        engine_with_model.apply_flat_parameters(np.zeros_like(original_params))
        assert not np.allclose(engine_with_model.get_flat_parameters(), original_params)

        # Restore
        engine_with_model.load_state_dict(state)
        np.testing.assert_allclose(
            engine_with_model.get_flat_parameters(), original_params, rtol=1e-5
        )

    def test_state_dict_cross_engine(self):
        """State dict from one engine instance loads into another."""
        model1 = SimpleMLXModel()
        mlx.eval(model1.parameters())
        engine1 = MLXEngine()
        engine1.load_model(model1, loss_fn=cross_entropy_loss)
        state = engine1.state_dict()
        params1 = engine1.get_flat_parameters().copy()

        model2 = SimpleMLXModel()
        mlx.eval(model2.parameters())
        engine2 = MLXEngine()
        engine2.load_model(model2, loss_fn=cross_entropy_loss)

        engine2.load_state_dict(state)
        np.testing.assert_allclose(
            engine2.get_flat_parameters(), params1, rtol=1e-5
        )
