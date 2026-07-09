"""Framework-agnostic tests for the MacFleet wire protocol."""

import asyncio
import struct

import pytest

import macfleet.comm.protocol as protocol
from macfleet.comm.protocol import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
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
        packed[-1] ^= 0xFF
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            WireMessage.unpack(bytes(packed))

    def test_truncated_payload_rejected(self):
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.CONTROL,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=b"important data",
        )
        with pytest.raises(ValueError, match="length mismatch"):
            WireMessage.unpack(msg.pack()[:-1])

    def test_trailing_bytes_rejected(self):
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.CONTROL,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=b"important data",
        )
        with pytest.raises(ValueError, match="length mismatch"):
            WireMessage.unpack(msg.pack() + b"extra")

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

    def test_reserved_header_field_must_be_zero(self):
        payload = b"data"
        packed = struct.pack(
            HEADER_FORMAT,
            0,
            MessageType.TENSOR,
            MessageFlags.NONE,
            len(payload),
            0,
            0,
            1,
        ) + payload

        with pytest.raises(ValueError, match="Reserved wire header field"):
            WireMessage.unpack(packed)

    def test_pack_rejects_payload_larger_than_wire_limit(self, monkeypatch):
        monkeypatch.setattr(protocol, "MAX_PAYLOAD_SIZE", 4)
        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.TENSOR,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=b"oversized",
        )

        with pytest.raises(ValueError, match="exceeds maximum"):
            msg.pack()


class TestMaxPayloadSize:
    """Verify that read_from_stream rejects oversized payloads."""

    async def test_oversized_payload_rejected(self):
        fake_header = struct.pack(
            HEADER_FORMAT,
            0,
            MessageType.TENSOR,
            MessageFlags.NONE,
            MAX_PAYLOAD_SIZE + 1,
            0,
            0,
            0,
        )
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()

        with pytest.raises(ValueError, match="exceeds maximum"):
            await WireMessage.read_from_stream(reader)

    async def test_max_boundary_accepted(self):
        fake_header = struct.pack(
            HEADER_FORMAT,
            0,
            MessageType.TENSOR,
            MessageFlags.NONE,
            MAX_PAYLOAD_SIZE,
            0,
            0,
            0,
        )
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()

        with pytest.raises(asyncio.IncompleteReadError):
            await WireMessage.read_from_stream(reader)

    async def test_unknown_type_rejected_before_payload_read(self):
        """Invalid headers should fail before a peer can stream a huge body."""
        fake_header = struct.pack(
            HEADER_FORMAT,
            0,
            999,
            MessageFlags.NONE,
            MAX_PAYLOAD_SIZE,
            0,
            0,
            0,
        )
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()

        with pytest.raises(ValueError, match="Unknown wire message type"):
            await WireMessage.read_from_stream(reader)
