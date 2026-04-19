"""Tests for the wire protocol.

Torch is required here because this module exercises the v1 compat
torch-tensor serialization helpers in `macfleet/engines/serialization.py`
(scheduled for deletion in v2.3 per TODOS.md). Skip the entire module when
torch is not installed so framework-agnostic CI matrices collect cleanly.
"""

import asyncio
import struct

import pytest

torch = pytest.importorskip("torch", reason="torch-dependent wire serialization tests")

from macfleet.comm.protocol import (  # noqa: E402
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)
from macfleet.engines.serialization import (  # noqa: E402
    bytes_to_tensor,
    deserialize_compressed_gradient,
    serialize_compressed_gradient,
    tensor_to_bytes,
)


class TestWireMessage:
    def test_header_size(self):
        assert HEADER_SIZE == 24

    def test_pack_unpack_roundtrip(self):
        msg = WireMessage(
            stream_id=1,
            msg_type=MessageType.TENSOR,
            flags=MessageFlags.NONE,
            sequence=42,
            payload=b"hello world",
        )
        packed = msg.pack()
        unpacked = WireMessage.unpack(packed)

        assert unpacked.stream_id == 1
        assert unpacked.msg_type == MessageType.TENSOR
        assert unpacked.flags == MessageFlags.NONE
        assert unpacked.sequence == 42
        assert unpacked.payload == b"hello world"

    def test_crc32_verification(self):
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.HEARTBEAT,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=b"test payload",
        )
        packed = msg.pack()
        unpacked = WireMessage.unpack(packed)
        assert unpacked.checksum != 0

    def test_crc32_corruption_detected(self):
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.CONTROL,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=b"important data",
        )
        packed = bytearray(msg.pack())
        # Corrupt a payload byte
        packed[-1] ^= 0xFF
        import pytest
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            WireMessage.unpack(bytes(packed))

    def test_flags(self):
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.GRADIENT,
            flags=MessageFlags.COMPRESSED | MessageFlags.CHUNKED,
            sequence=0,
            payload=b"data",
        )
        packed = msg.pack()
        unpacked = WireMessage.unpack(packed)
        assert MessageFlags.COMPRESSED in unpacked.flags
        assert MessageFlags.CHUNKED in unpacked.flags
        assert MessageFlags.LAST_CHUNK not in unpacked.flags


class TestMaxPayloadSize:
    """Verify that read_from_stream rejects oversized payloads (OOM protection)."""

    async def test_oversized_payload_rejected(self):
        """A header claiming payload_size > MAX_PAYLOAD_SIZE must raise ValueError."""
        fake_header = struct.pack(
            HEADER_FORMAT,
            0,                          # stream_id
            MessageType.TENSOR,         # msg_type
            MessageFlags.NONE,          # flags
            MAX_PAYLOAD_SIZE + 1,       # payload_size — over limit
            0,                          # sequence
            0,                          # checksum (irrelevant, rejected before read)
            0,                          # reserved
        )
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()

        with pytest.raises(ValueError, match="exceeds maximum"):
            await WireMessage.read_from_stream(reader)

    async def test_max_boundary_accepted(self):
        """payload_size == MAX_PAYLOAD_SIZE should not be rejected by the size check."""
        # We only test the size check passes — we don't actually send 256MB.
        # Feed a header with payload_size=MAX_PAYLOAD_SIZE but EOF immediately
        # so readexactly raises IncompleteReadError (not ValueError).
        fake_header = struct.pack(
            HEADER_FORMAT,
            0, MessageType.TENSOR, MessageFlags.NONE,
            MAX_PAYLOAD_SIZE, 0, 0, 0,
        )
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()

        with pytest.raises(asyncio.IncompleteReadError):
            await WireMessage.read_from_stream(reader)


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
