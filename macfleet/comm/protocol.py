"""Binary wire protocol for tensor transport.

Extended from MacFleet v1's 16-byte header to 24 bytes with:
- Stream multiplexing (stream_id)
- CRC32 checksums (critical for WiFi reliability)
- Chunking flags for large tensors
- Sequence numbers for ordering

Header (24 bytes):
  stream_id:    uint32  (multiplexing: control=0, tensor=1..N)
  msg_type:     uint16  (CONTROL=1, TENSOR=2, HEARTBEAT=3, GRADIENT=4, COMPRESSED=5)
  flags:        uint16  (bit 0: compressed, bit 1: chunked, bit 2: last_chunk)
  payload_size: uint32  (bytes)
  sequence:     uint32  (ordering within stream)
  checksum:     uint32  (CRC32 of payload)
  reserved:     uint32  (future use)
"""

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum, IntFlag


class MessageType(IntEnum):
    """Message types for the wire protocol."""
    CONTROL = 0x01
    TENSOR = 0x02
    HEARTBEAT = 0x03
    GRADIENT = 0x04
    COMPRESSED_GRADIENT = 0x05
    BARRIER = 0x06
    STATE = 0x07
    TASK = 0x08
    RESULT = 0x09


class MessageFlags(IntFlag):
    """Bit flags for message metadata."""
    NONE = 0x00
    COMPRESSED = 0x01
    CHUNKED = 0x02
    LAST_CHUNK = 0x04
    # v2.2 PR 4: the handshake payload uses the structured v2 format
    # (carries a signed HW profile). Without this flag the server falls
    # back to v2.1 handshake parsing (bare `node_id + challenge`).
    HANDSHAKE_V2 = 0x08


# 24-byte header: stream_id(I) msg_type(H) flags(H) payload_size(I) sequence(I) checksum(I) reserved(I)
HEADER_FORMAT = "!IHHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes

# SECURITY: Maximum payload size to prevent OOM from malicious headers.
# 256 MB is larger than any realistic gradient tensor (100M float32 = 400 MB,
# but compressed gradients are much smaller). Set conservatively high.
MAX_PAYLOAD_SIZE = 256 * 1024 * 1024  # 256 MB


@dataclass
class WireMessage:
    """A message on the wire."""
    stream_id: int
    msg_type: MessageType
    flags: MessageFlags
    sequence: int
    payload: bytes
    checksum: int = 0

    def pack(self) -> bytes:
        """Serialize to bytes (header + payload)."""
        checksum = zlib.crc32(self.payload) & 0xFFFFFFFF
        header = struct.pack(
            HEADER_FORMAT,
            self.stream_id,
            self.msg_type,
            self.flags,
            len(self.payload),
            self.sequence,
            checksum,
            0,  # reserved
        )
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> "WireMessage":
        """Deserialize from bytes."""
        header = data[:HEADER_SIZE]
        stream_id, msg_type, flags, payload_size, sequence, checksum, _ = struct.unpack(
            HEADER_FORMAT, header
        )
        payload = data[HEADER_SIZE : HEADER_SIZE + payload_size]

        # Verify checksum
        actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_checksum != checksum:
            raise ValueError(
                f"CRC32 mismatch: expected {checksum:#x}, got {actual_checksum:#x}"
            )

        return cls(
            stream_id=stream_id,
            msg_type=MessageType(msg_type),
            flags=MessageFlags(flags),
            sequence=sequence,
            payload=payload,
            checksum=checksum,
        )

    @classmethod
    async def read_from_stream(cls, reader) -> "WireMessage":
        """Read a single message from an asyncio StreamReader."""
        header_data = await reader.readexactly(HEADER_SIZE)
        stream_id, msg_type, flags, payload_size, sequence, checksum, _ = struct.unpack(
            HEADER_FORMAT, header_data
        )
        if payload_size > MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"Payload size {payload_size} exceeds maximum {MAX_PAYLOAD_SIZE} "
                f"— possible OOM attack or corrupt header"
            )
        payload = await reader.readexactly(payload_size)

        actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_checksum != checksum:
            raise ValueError(
                f"CRC32 mismatch: expected {checksum:#x}, got {actual_checksum:#x}"
            )

        return cls(
            stream_id=stream_id,
            msg_type=MessageType(msg_type),
            flags=MessageFlags(flags),
            sequence=sequence,
            payload=payload,
            checksum=checksum,
        )


