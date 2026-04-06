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
from typing import Optional

import numpy as np
import torch


class MessageType(IntEnum):
    """Message types for the wire protocol."""
    CONTROL = 0x01
    TENSOR = 0x02
    HEARTBEAT = 0x03
    GRADIENT = 0x04
    COMPRESSED_GRADIENT = 0x05
    BARRIER = 0x06
    STATE = 0x07


class MessageFlags(IntFlag):
    """Bit flags for message metadata."""
    NONE = 0x00
    COMPRESSED = 0x01
    CHUNKED = 0x02
    LAST_CHUNK = 0x04


# 24-byte header: stream_id(I) msg_type(H) flags(H) payload_size(I) sequence(I) checksum(I) reserved(I)
HEADER_FORMAT = "!IHHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes


# Dtype mappings (framework-agnostic codes)
DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.float64: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bfloat16: 8,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
}


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


def tensor_to_bytes(
    tensor: torch.Tensor,
    msg_type: MessageType = MessageType.GRADIENT,
) -> bytes:
    """Serialize a tensor to raw bytes (payload only, no wire header).

    Ported from v1's tensor_utils.py with identical format:
    - 16 bytes: msg_type(4B) dtype(4B) n_dims(4B) payload_size(4B)
    - Shape: n_dims * 4 bytes
    - Payload: raw tensor bytes
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    dtype = tensor.dtype
    shape = tensor.shape
    n_dims = len(shape)

    # bfloat16 → float16 for transfer (numpy limitation)
    if dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
        dtype = torch.float16

    np_dtype = TORCH_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    np_array = tensor.numpy()
    payload = np_array.tobytes()

    # Inner header (v1 compatible)
    dtype_code = DTYPE_TO_CODE[dtype]
    inner_header = struct.pack("!IIII", msg_type, dtype_code, n_dims, len(payload))
    shape_bytes = struct.pack(f"!{n_dims}I", *shape)

    return inner_header + shape_bytes + payload


def bytes_to_tensor(
    data: bytes,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, MessageType]:
    """Deserialize bytes to a tensor.

    Returns:
        Tuple of (tensor, message_type).
    """
    inner_header_size = 16
    msg_type, dtype_code, n_dims, payload_size = struct.unpack(
        "!IIII", data[:inner_header_size]
    )
    msg_type = MessageType(msg_type)

    shape_start = inner_header_size
    shape_end = shape_start + n_dims * 4
    shape = struct.unpack(f"!{n_dims}I", data[shape_start:shape_end])

    payload_start = shape_end
    payload = data[payload_start : payload_start + payload_size]

    dtype = CODE_TO_DTYPE[dtype_code]
    np_dtype = TORCH_TO_NUMPY[dtype]
    np_array = np.frombuffer(payload, dtype=np_dtype).reshape(shape)

    tensor = torch.from_numpy(np_array.copy())
    if device is not None and device != "cpu":
        tensor = tensor.to(device)

    return tensor, msg_type


def serialize_compressed_gradient(
    indices: torch.Tensor,
    values: torch.Tensor,
    original_numel: int,
    original_dtype: torch.dtype,
) -> bytes:
    """Serialize a compressed gradient (Top-K sparse + FP16)."""
    indices = indices.cpu().to(torch.int32).contiguous()
    values = values.cpu().to(torch.float16).contiguous()

    topk_count = indices.numel()
    indices_bytes = indices.numpy().tobytes()
    values_bytes = values.numpy().tobytes()

    payload_size = len(indices_bytes) + len(values_bytes)

    header = struct.pack(
        "!IIII",
        MessageType.COMPRESSED_GRADIENT,
        DTYPE_TO_CODE.get(original_dtype, 0),
        0,
        payload_size,
    )
    orig_dtype_code = DTYPE_TO_CODE.get(original_dtype, 0)
    metadata = struct.pack("!III", original_numel, topk_count, orig_dtype_code)

    return header + metadata + indices_bytes + values_bytes


def deserialize_compressed_gradient(
    data: bytes,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
    """Deserialize a compressed gradient."""
    inner_header_size = 16
    msg_type, _, _, payload_size = struct.unpack("!IIII", data[:inner_header_size])

    if msg_type != MessageType.COMPRESSED_GRADIENT:
        raise ValueError(f"Expected COMPRESSED_GRADIENT, got {msg_type}")

    metadata_start = inner_header_size
    metadata_end = metadata_start + 12
    original_numel, topk_count, orig_dtype_code = struct.unpack(
        "!III", data[metadata_start:metadata_end]
    )
    original_dtype = CODE_TO_DTYPE.get(orig_dtype_code, torch.float32)

    indices_start = metadata_end
    indices_size = topk_count * 4
    indices = torch.from_numpy(
        np.frombuffer(data[indices_start : indices_start + indices_size], dtype=np.int32).copy()
    )

    values_start = indices_start + indices_size
    values_size = topk_count * 2
    values = torch.from_numpy(
        np.frombuffer(data[values_start : values_start + values_size], dtype=np.float16).copy()
    )

    if device is not None and device != "cpu":
        indices = indices.to(device)
        values = values.to(device)

    return indices, values, original_numel, original_dtype
