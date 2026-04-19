"""Tests for the PyTorch training engine."""

import numpy as np
import torch
import torch.nn as nn

from macfleet.engines.base import EngineType
from macfleet.engines.torch_engine import TorchEngine


class _TinyModel(nn.Module):
    """Minimal model for testing (10 + 5 = 15 params)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 5, bias=False)  # 10 params

    def forward(self, x):
        return self.linear(x).sum()


class _TwoLayerModel(nn.Module):
    """Two-layer model for gradient bucketing tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8, bias=False)  # 32 params
        self.layer2 = nn.Linear(8, 2, bias=False)  # 16 params

    def forward(self, x):
        return self.layer2(self.layer1(x)).sum()


class TestTorchEngineBasic:
    def test_init_cpu(self):
        engine = TorchEngine(device="cpu")
        assert engine.device == torch.device("cpu")

    def test_load_model(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)
        assert engine.param_count() == 10

    def test_forward_backward(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        # Gradients should be populated
        assert model.linear.weight.grad is not None

    def test_step(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine.load_model(model, optimizer)

        params_before = model.linear.weight.data.clone()

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        engine.step()

        # Parameters should have changed
        assert not torch.equal(model.linear.weight.data, params_before)

    def test_zero_grad(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        assert model.linear.weight.grad is not None

        engine.zero_grad()
        assert model.linear.weight.grad is None or model.linear.weight.grad.abs().sum() == 0


class TestTorchEngineGradients:
    def test_get_flat_gradients(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        flat = engine.get_flat_gradients()
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert len(flat) == 10  # 5x2 weight matrix

    def test_flat_gradients_roundtrip(self):
        """get_flat_gradients → apply_flat_gradients preserves values."""
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        # Extract, modify (simulate allreduce), and apply
        flat = engine.get_flat_gradients()
        modified = flat * 2.0  # simulate averaged gradients
        engine.apply_flat_gradients(modified)

        # Verify the gradients match the modified values
        result = engine.get_flat_gradients()
        np.testing.assert_allclose(result, modified, rtol=1e-6)

    def test_flat_gradients_two_layers(self):
        """Flat gradients include all layers in parameter order."""
        engine = TorchEngine(device="cpu")
        model = _TwoLayerModel()
        engine.load_model(model)

        x = torch.randn(3, 4)
        loss = engine.forward(x)
        engine.backward(loss)

        flat = engine.get_flat_gradients()
        assert len(flat) == 32 + 16  # layer1 (4x8) + layer2 (8x2)

    def test_no_model_returns_empty(self):
        engine = TorchEngine(device="cpu")
        assert engine.param_count() == 0
        assert engine.memory_usage_gb() == 0.0


class TestTorchEngineParameters:
    def test_get_flat_parameters(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        flat = engine.get_flat_parameters()
        assert isinstance(flat, np.ndarray)
        assert len(flat) == 10

    def test_flat_parameters_roundtrip(self):
        """Broadcast simulation: get → apply preserves parameters."""
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        original = engine.get_flat_parameters().copy()

        # Simulate receiving broadcast parameters
        engine.apply_flat_parameters(original)

        result = engine.get_flat_parameters()
        np.testing.assert_allclose(result, original, rtol=1e-6)


class TestTorchEngineState:
    def test_state_dict_roundtrip(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)

        # Modify model
        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        engine.step()

        # Checkpoint
        params_after_step = engine.get_flat_parameters().copy()
        state_bytes = engine.state_dict()

        # Reset model
        engine.load_model(_TinyModel(), torch.optim.SGD(model.parameters(), lr=0.01))

        # Restore
        engine.load_state_dict(state_bytes)
        restored_params = engine.get_flat_parameters()
        np.testing.assert_allclose(restored_params, params_after_step, rtol=1e-6)

    def test_capabilities(self):
        engine = TorchEngine(device="cpu")
        caps = engine.capabilities
        assert caps.engine_type == EngineType.TORCH
        assert "float32" in caps.supported_dtypes

    def test_memory_usage(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)
        assert engine.memory_usage_gb() > 0

    def test_estimated_memory(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)
        assert engine.estimated_model_memory_gb() > 0
