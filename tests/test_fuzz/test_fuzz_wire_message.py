"""Hypothesis-based fuzz tests for WireMessage pack/unpack.

Targets two failure modes:
  1. Roundtrip property: pack(msg).unpack() == msg for any valid input.
  2. Robustness: arbitrary attacker-supplied bytes never panic the parser
     in unexpected ways. Allowed exits are ValueError or struct.error.

The CRC32 + size-cap defenses are critical — a malicious peer with a
verified token can still send junk through the transport, and the parser
must reject without crashing the agent.
"""

from __future__ import annotations

import struct
import zlib

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from macfleet.comm.protocol import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)

# -----------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------

valid_msg_types = st.sampled_from(list(MessageType))
valid_flag_combinations = st.builds(
    lambda *bits: MessageFlags(sum(bits)),
    st.sampled_from([0, MessageFlags.COMPRESSED.value]),
    st.sampled_from([0, MessageFlags.CHUNKED.value]),
    st.sampled_from([0, MessageFlags.LAST_CHUNK.value]),
    st.sampled_from([0, MessageFlags.HANDSHAKE_V2.value]),
)
small_payloads = st.binary(min_size=0, max_size=4096)
medium_payloads = st.binary(min_size=0, max_size=64 * 1024)


# -----------------------------------------------------------------
# Roundtrip property
# -----------------------------------------------------------------


class TestWireMessageRoundtrip:
    @given(
        stream_id=st.integers(min_value=0, max_value=2**32 - 1),
        msg_type=valid_msg_types,
        flags=valid_flag_combinations,
        sequence=st.integers(min_value=0, max_value=2**32 - 1),
        payload=small_payloads,
    )
    @settings(max_examples=300, deadline=None)
    def test_pack_unpack_roundtrip(self, stream_id, msg_type, flags, sequence, payload):
        msg = WireMessage(
            stream_id=stream_id, msg_type=msg_type, flags=flags,
            sequence=sequence, payload=payload,
        )
        packed = msg.pack()
        unpacked = WireMessage.unpack(packed)
        assert unpacked.stream_id == stream_id
        assert unpacked.msg_type == msg_type
        assert unpacked.flags == flags
        assert unpacked.sequence == sequence
        assert unpacked.payload == payload

    @given(payload=medium_payloads)
    @settings(max_examples=50, deadline=None)
    def test_medium_payload_roundtrip(self, payload):
        msg = WireMessage(
            stream_id=1, msg_type=MessageType.GRADIENT,
            flags=MessageFlags.NONE, sequence=0, payload=payload,
        )
        unpacked = WireMessage.unpack(msg.pack())
        assert unpacked.payload == payload


# -----------------------------------------------------------------
# Adversarial: bit-flip detection
# -----------------------------------------------------------------


class TestWireMessageCorruption:
    @given(
        payload=st.binary(min_size=1, max_size=2048),
        flip_byte=st.integers(min_value=0, max_value=2047),
    )
    @settings(
        max_examples=200, deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_single_bit_flip_caught_or_safe(self, payload, flip_byte):
        msg = WireMessage(
            stream_id=0, msg_type=MessageType.TENSOR,
            flags=MessageFlags.NONE, sequence=0, payload=payload,
        )
        packed = bytearray(msg.pack())
        # Flip a byte somewhere in the payload region (header bit-flips can
        # be valid because flags/seq are arbitrary 32-bit ints).
        flip_byte = HEADER_SIZE + (flip_byte % len(payload))
        packed[flip_byte] ^= 0xFF
        # Either CRC32 catches it, or the unpack succeeds with different
        # bytes — never a Python crash like AttributeError or IndexError.
        try:
            result = WireMessage.unpack(bytes(packed))
            assert result.payload != payload, (
                "byte flip in payload should change the payload"
            )
        except ValueError as e:
            assert "CRC32" in str(e) or "exceeds maximum" in str(e)


# -----------------------------------------------------------------
# Adversarial: oversized payload sizes
# -----------------------------------------------------------------


class TestWireMessageMaxSize:
    @given(claimed_size=st.integers(min_value=MAX_PAYLOAD_SIZE + 1, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def test_oversize_header_rejected_in_unpack(self, claimed_size):
        # Header claims a giant payload; the actual buffer is too small.
        # unpack should reject via "exceeds maximum" rather than allocate
        # or hang trying to slice past the buffer.
        fake_header = struct.pack(
            HEADER_FORMAT,
            0, MessageType.TENSOR, MessageFlags.NONE,
            claimed_size, 0, 0, 0,
        )
        with pytest.raises(ValueError, match="exceeds maximum"):
            WireMessage.unpack(fake_header + b"\x00")

    def test_at_max_payload_is_acceptable(self):
        # Don't actually allocate 256 MB — just check the size cap accepts
        # the boundary value when paired with that many bytes (we feed a
        # short buffer, which then surfaces as a CRC mismatch rather than
        # a size rejection).
        fake_header = struct.pack(
            HEADER_FORMAT,
            0, MessageType.TENSOR, MessageFlags.NONE,
            MAX_PAYLOAD_SIZE, 0, 0, 0,
        )
        # Short buffer → CRC32 mismatch error, not size rejection.
        with pytest.raises(ValueError, match="CRC32"):
            WireMessage.unpack(fake_header + b"\x00" * 32)


# -----------------------------------------------------------------
# Adversarial: arbitrary bytes can't panic the parser
# -----------------------------------------------------------------


class TestWireMessageRandomBytes:
    @given(data=st.binary(min_size=HEADER_SIZE, max_size=4096))
    @settings(max_examples=400, deadline=None)
    def test_random_bytes_safe(self, data):
        # Arbitrary attacker-controlled bytes. Allowed: ValueError (size
        # cap, CRC, MessageType validation), struct.error (header parse).
        try:
            WireMessage.unpack(data)
        except (ValueError, struct.error):
            pass

    @given(data=st.binary(min_size=0, max_size=HEADER_SIZE - 1))
    @settings(max_examples=50, deadline=None)
    def test_short_buffer_raises_struct_error(self, data):
        # Buffer shorter than the header — struct.unpack rejects it.
        with pytest.raises((struct.error, ValueError)):
            WireMessage.unpack(data)


# -----------------------------------------------------------------
# CRC32 boundary: random valid CRC checksums always parse
# -----------------------------------------------------------------


class TestWireMessageCRC:
    @given(payload=st.binary(min_size=0, max_size=2048))
    @settings(max_examples=100, deadline=None)
    def test_valid_crc_parses(self, payload):
        msg = WireMessage(
            stream_id=42, msg_type=MessageType.RESULT,
            flags=MessageFlags.COMPRESSED, sequence=99, payload=payload,
        )
        unpacked = WireMessage.unpack(msg.pack())
        # The packed checksum must match what zlib would compute now.
        assert unpacked.checksum == (zlib.crc32(payload) & 0xFFFFFFFF)
