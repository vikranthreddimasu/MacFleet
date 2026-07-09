"""Tests for torch-dependent tensor serialization helpers.

Torch is required here because this module exercises the v1 compat
torch-tensor serialization helpers in `macfleet/engines/serialization.py`
(scheduled for deletion in v2.3 per TODOS.md). Skip the entire module when
torch is not installed so framework-agnostic CI matrices collect cleanly.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch-dependent wire serialization tests")

from macfleet.comm.protocol import (  # noqa: E402
    MessageType,
)
from macfleet.engines.serialization import (  # noqa: E402
    bytes_to_tensor,
    deserialize_compressed_gradient,
    serialize_compressed_gradient,
    tensor_to_bytes,
)


class TestTensorSerialization:
    def test_float32_roundtrip(self):
        tensor = torch.randn(100, 50)
        data = tensor_to_bytes(tensor, MessageType.GRADIENT)
        result, msg_type = bytes_to_tensor(data)
        assert msg_type == MessageType.GRADIENT
        assert torch.allclose(tensor, result)

    def test_float16_roundtrip(self):
        tensor = torch.randn(100, dtype=torch.float16)
        data = tensor_to_bytes(tensor, MessageType.TENSOR)
        result, msg_type = bytes_to_tensor(data)
        assert result.dtype == torch.float16
        assert torch.allclose(tensor, result)

    def test_int32_roundtrip(self):
        tensor = torch.randint(0, 1000, (200,), dtype=torch.int32)
        data = tensor_to_bytes(tensor)
        result, _ = bytes_to_tensor(data)
        assert torch.equal(tensor, result)

    def test_various_shapes(self):
        for shape in [(10,), (5, 5), (2, 3, 4), (1,)]:
            tensor = torch.randn(*shape)
            data = tensor_to_bytes(tensor)
            result, _ = bytes_to_tensor(data)
            assert result.shape == tensor.shape
            assert torch.allclose(tensor, result)

    def test_bfloat16_converts_to_float16(self):
        tensor = torch.randn(100, dtype=torch.bfloat16)
        data = tensor_to_bytes(tensor)
        result, _ = bytes_to_tensor(data)
        assert result.dtype == torch.float16

    def test_large_tensor(self):
        tensor = torch.randn(1000, 1000)  # ~4MB
        data = tensor_to_bytes(tensor)
        result, _ = bytes_to_tensor(data)
        assert torch.allclose(tensor, result)


class TestCompressedGradientSerialization:
    def test_roundtrip(self):
        indices = torch.tensor([0, 5, 10, 99], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        original_numel = 1000
        original_dtype = torch.float32

        data = serialize_compressed_gradient(indices, values, original_numel, original_dtype)
        r_indices, r_values, r_numel, r_dtype = deserialize_compressed_gradient(data)

        assert torch.equal(indices, r_indices)
        assert torch.equal(values, r_values)
        assert r_numel == original_numel
        assert r_dtype == original_dtype

    def test_large_compressed(self):
        k = 1000
        indices = torch.randint(0, 100000, (k,), dtype=torch.int32)
        values = torch.randn(k, dtype=torch.float16)
        data = serialize_compressed_gradient(indices, values, 100000, torch.float32)
        r_indices, r_values, r_numel, _ = deserialize_compressed_gradient(data)
        assert len(r_indices) == k
        assert r_numel == 100000
