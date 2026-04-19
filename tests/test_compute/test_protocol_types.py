"""Tests for TASK/RESULT message types in the wire protocol."""


import pytest

from macfleet.comm.protocol import (
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)
from macfleet.compute.models import TaskResult, TaskSpec


class TestMessageTypes:
    def test_task_type_value(self):
        assert MessageType.TASK == 0x08

    def test_result_type_value(self):
        assert MessageType.RESULT == 0x09

    def test_task_message_pack_unpack(self):
        """TASK WireMessage survives pack/unpack with CRC32."""
        spec = TaskSpec.from_call(lambda x: x, args=(1,))
        payload = spec.pack()

        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.TASK,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=payload,
        )
        raw = msg.pack()
        assert len(raw) == HEADER_SIZE + len(payload)

        restored = WireMessage.unpack(raw)
        assert restored.msg_type == MessageType.TASK
        assert restored.payload == payload

    def test_result_message_pack_unpack(self):
        """RESULT WireMessage survives pack/unpack with CRC32."""
        result = TaskResult.success("task-abc", {"answer": 42})
        payload = result.pack()

        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.RESULT,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=payload,
        )
        raw = msg.pack()
        restored = WireMessage.unpack(raw)

        assert restored.msg_type == MessageType.RESULT
        r = TaskResult.unpack(restored.payload)
        assert r.task_id == "task-abc"
        assert r.unwrap() == {"answer": 42}

    def test_crc32_detects_corruption(self):
        """Corrupted TASK payload is caught by CRC32."""
        spec = TaskSpec.from_call(lambda: None)
        payload = spec.pack()

        msg = WireMessage(
            stream_id=0,
            msg_type=MessageType.TASK,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=payload,
        )
        raw = bytearray(msg.pack())

        # Corrupt one byte in the payload
        raw[-1] ^= 0xFF

        with pytest.raises(ValueError, match="CRC32 mismatch"):
            WireMessage.unpack(bytes(raw))

    def test_task_payload_within_max_size(self):
        """A typical task payload is well under MAX_PAYLOAD_SIZE."""
        spec = TaskSpec.from_call(lambda x: x * 2, args=(list(range(1000)),))
        payload = spec.pack()
        assert len(payload) < MAX_PAYLOAD_SIZE
