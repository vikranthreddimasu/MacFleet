"""Tests for tensor transport layer."""

import asyncio

import pytest
import torch

from macfleet.comm.transport import TensorTransport
from macfleet.utils.tensor_utils import (
    MessageType,
    bytes_to_tensor,
    deserialize_compressed_gradient,
    serialize_compressed_gradient,
    tensor_to_bytes,
)


class TestTensorSerialization:
    """Tests for tensor serialization utilities."""

    def test_basic_roundtrip(self):
        """Test basic tensor serialization round-trip."""
        tensor = torch.randn(64, 128)
        data = tensor_to_bytes(tensor)
        result, msg_type = bytes_to_tensor(data)

        assert torch.allclose(tensor, result)
        assert msg_type == MessageType.TENSOR_GRADIENT

    def test_different_dtypes(self):
        """Test serialization with different data types."""
        dtypes = [torch.float32, torch.float16, torch.float64, torch.int32, torch.int64]

        for dtype in dtypes:
            if dtype in [torch.float32, torch.float16, torch.float64]:
                tensor = torch.randn(100).to(dtype)
            else:
                tensor = torch.randint(0, 100, (100,), dtype=dtype)

            data = tensor_to_bytes(tensor)
            result, _ = bytes_to_tensor(data)

            assert torch.equal(tensor, result), f"Failed for dtype {dtype}"

    def test_different_shapes(self):
        """Test serialization with different tensor shapes."""
        shapes = [(1000,), (32, 32), (10, 10, 10), (2, 3, 4, 5)]

        for shape in shapes:
            tensor = torch.randn(shape)
            data = tensor_to_bytes(tensor)
            result, _ = bytes_to_tensor(data)

            assert torch.allclose(tensor, result), f"Failed for shape {shape}"
            assert result.shape == shape

    def test_message_types(self):
        """Test different message types are preserved."""
        tensor = torch.randn(100)

        for msg_type in MessageType:
            data = tensor_to_bytes(tensor, msg_type)
            _, received_type = bytes_to_tensor(data)
            assert received_type == msg_type

    def test_compressed_gradient_roundtrip(self):
        """Test compressed gradient serialization."""
        indices = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        original_numel = 100
        original_dtype = torch.float32

        data = serialize_compressed_gradient(indices, values, original_numel, original_dtype)
        result_indices, result_values, result_numel, result_dtype = (
            deserialize_compressed_gradient(data)
        )

        assert torch.equal(indices, result_indices)
        assert torch.allclose(values, result_values)
        assert result_numel == original_numel
        assert result_dtype == original_dtype


class TestTensorTransport:
    """Tests for async tensor transport."""

    @pytest.mark.asyncio
    async def test_basic_send_receive(self):
        """Test basic tensor send/receive."""
        port = 50200

        server = TensorTransport("127.0.0.1", port)
        received = []

        async def on_receive(tensor, msg_type, addr):
            received.append((tensor.clone(), msg_type))

        await server.start_server(on_receive)

        # Connect and send
        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", port)

        test_tensor = torch.randn(64, 128)
        await client.send_tensor(test_tensor, conn_key, MessageType.TENSOR_GRADIENT)

        # Wait for server to receive
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert torch.allclose(test_tensor, received[0][0])
        assert received[0][1] == MessageType.TENSOR_GRADIENT

        await client.disconnect(conn_key)
        await server.stop_server()

    @pytest.mark.asyncio
    async def test_multiple_tensors(self):
        """Test sending multiple tensors."""
        port = 50201

        server = TensorTransport("127.0.0.1", port)
        received = []

        async def on_receive(tensor, msg_type, addr):
            received.append(tensor.clone())

        await server.start_server(on_receive)

        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", port)

        for i in range(5):
            tensor = torch.randn(32, 32) * (i + 1)
            await client.send_tensor(tensor, conn_key)

        await asyncio.sleep(0.3)

        assert len(received) == 5

        await client.disconnect(conn_key)
        await server.stop_server()

    @pytest.mark.asyncio
    async def test_large_tensor(self):
        """Test sending large tensors."""
        port = 50202

        server = TensorTransport("127.0.0.1", port)
        received = []

        async def on_receive(tensor, msg_type, addr):
            received.append(tensor.clone())

        await server.start_server(on_receive)

        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", port)

        # ~10MB tensor
        large_tensor = torch.randn(2560 * 1024)
        await client.send_tensor(large_tensor, conn_key)

        await asyncio.sleep(0.5)

        assert len(received) == 1
        assert torch.allclose(large_tensor, received[0])

        await client.disconnect(conn_key)
        await server.stop_server()
