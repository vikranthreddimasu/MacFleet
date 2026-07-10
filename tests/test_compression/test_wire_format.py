"""Tests for the CompressedArray wire format (v2.3 sparse-on-wire).

Framework-agnostic (numpy only). Covers exact roundtrips for every
packable kind, bitwise-identical dequantization across the wire (the
parameter-identity requirement), and fail-closed rejection of malformed
or malicious payloads.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from macfleet.compression.adaptive import (
    AdaptiveCompressionConfig,
    AdaptiveCompressor,
    CompressedArray,
    CompressionLevel,
    decompress_to_dense,
    pack_compressed,
    unpack_compressed,
)
from macfleet.pool.network import LinkType
from macfleet.security.auth import GradientValidationError


def _compress(arr: np.ndarray, level: CompressionLevel) -> CompressedArray:
    comp = AdaptiveCompressor(
        config=AdaptiveCompressionConfig(fixed_level=level, min_compress_size=1),
    )
    out = comp.compress(arr)
    assert isinstance(out, CompressedArray)
    return out


class TestRoundtrip:
    def test_topk_fp16_roundtrip_is_bitwise_exact(self):
        """decompress_to_dense(unpack(pack(ca))) must equal the LOCAL
        decompress(ca) bit-for-bit — that identity is what keeps the two
        ranks' averaged gradients (and therefore parameters) identical."""
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(8192).astype(np.float32)
        comp = AdaptiveCompressor(
            config=AdaptiveCompressionConfig(
                fixed_level=CompressionLevel.AGGRESSIVE, min_compress_size=1,
            ),
        )
        ca = comp.compress(arr)
        assert isinstance(ca, CompressedArray)

        local_dense = comp.decompress(ca)
        wire_dense = decompress_to_dense(unpack_compressed(pack_compressed(ca)))
        np.testing.assert_array_equal(local_dense, wire_dense)

    def test_topk_fp32_roundtrip(self):
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(4096).astype(np.float32)
        comp = AdaptiveCompressor(
            config=AdaptiveCompressionConfig(
                fixed_level=CompressionLevel.MODERATE,
                min_compress_size=1,
                use_fp16=False,
            ),
        )
        ca = comp.compress(arr)
        assert isinstance(ca, CompressedArray)
        assert not ca.is_fp16

        wire = unpack_compressed(pack_compressed(ca))
        np.testing.assert_array_equal(
            decompress_to_dense(wire), comp.decompress(ca),
        )

    def test_dense_fp16_roundtrip(self):
        rng = np.random.default_rng(2)
        arr = rng.standard_normal(2048).astype(np.float32)
        ca = _compress(arr, CompressionLevel.LIGHT)
        assert ca.topk_indices is None and ca.is_fp16

        wire = unpack_compressed(pack_compressed(ca))
        comp = AdaptiveCompressor(
            config=AdaptiveCompressionConfig(
                fixed_level=CompressionLevel.LIGHT, min_compress_size=1,
            ),
        )
        np.testing.assert_array_equal(
            decompress_to_dense(wire), comp.decompress(ca),
        )

    def test_wire_is_actually_smaller(self):
        """The point of the exercise: aggressive payload << dense bytes."""
        rng = np.random.default_rng(3)
        arr = rng.standard_normal(100_000).astype(np.float32)
        ca = _compress(arr, CompressionLevel.AGGRESSIVE)
        payload = pack_compressed(ca)
        # TopK 1% int32+fp16 ≈ 1.5% of dense — allow generous slack.
        assert len(payload) < arr.nbytes * 0.05

    def test_dense_fp32_passthrough_not_packable(self):
        """compress() returns plain ndarrays for NONE/warmup — those use
        the dense allreduce path, never pack_compressed."""
        ca = CompressedArray(
            data=b"", original_shape=(4,), original_size=16,
            compressed_size=16, level=CompressionLevel.NONE,
        )
        with pytest.raises(ValueError):
            pack_compressed(ca)


class TestFailClosed:
    def _valid_payload(self) -> bytes:
        rng = np.random.default_rng(4)
        arr = rng.standard_normal(4096).astype(np.float32)
        return pack_compressed(_compress(arr, CompressionLevel.AGGRESSIVE))

    def test_bad_magic(self):
        payload = bytearray(self._valid_payload())
        payload[:4] = b"XXXX"
        with pytest.raises(GradientValidationError, match="magic"):
            unpack_compressed(bytes(payload))

    def test_unknown_kind(self):
        payload = bytearray(self._valid_payload())
        payload[4] = 99
        with pytest.raises(GradientValidationError, match="kind"):
            unpack_compressed(bytes(payload))

    def test_truncated_header(self):
        with pytest.raises(GradientValidationError, match="too short"):
            unpack_compressed(b"MFC1")

    def test_truncated_body(self):
        payload = self._valid_payload()
        with pytest.raises(GradientValidationError):
            unpack_compressed(payload[: len(payload) // 2])

    def test_trailing_garbage_rejected(self):
        payload = self._valid_payload()
        with pytest.raises(GradientValidationError):
            unpack_compressed(payload + b"\x00" * 8)

    def test_allocation_bomb_numel_rejected(self):
        """Header claiming a huge numel must be rejected BEFORE allocation."""
        # magic, kind=topk_fp16, ndims=1, scale, numel=2^33, k=1
        header = struct.pack(
            "!4sBBdII", b"MFC1", 2, 1, 1.0, 0xFFFFFFFF, 1,
        )
        body = header + struct.pack("!I", 0xFFFFFFFF) + b"\x00" * 6
        with pytest.raises(GradientValidationError):
            unpack_compressed(body)

    def test_k_exceeding_numel_rejected(self):
        header = struct.pack("!4sBBdII", b"MFC1", 2, 1, 1.0, 100, 200)
        with pytest.raises(GradientValidationError):
            unpack_compressed(header + struct.pack("!I", 100))

    def test_out_of_range_indices_rejected(self):
        """Sparse indices >= numel would scatter out of bounds."""
        k, numel = 4, 100
        header = struct.pack("!4sBBdII", b"MFC1", 2, 1, 1.0, numel, k)
        shape = struct.pack("!I", numel)
        indices = np.array([0, 1, 2, numel + 5], dtype=np.int32).tobytes()
        values = np.ones(k, dtype=np.float16).tobytes()
        with pytest.raises(GradientValidationError, match="out of range"):
            unpack_compressed(header + shape + indices + values)

    def test_negative_scale_rejected(self):
        payload = bytearray(self._valid_payload())
        # scale is the 8 bytes at offset 6 (after magic+kind+ndims)
        payload[6:14] = struct.pack("!d", -1.0)
        with pytest.raises(GradientValidationError, match="scale"):
            unpack_compressed(bytes(payload))

    def test_shape_numel_mismatch_rejected(self):
        k, numel = 2, 100
        header = struct.pack("!4sBBdII", b"MFC1", 2, 1, 1.0, numel, k)
        shape = struct.pack("!I", 50)  # claims 50, header says 100
        indices = np.array([0, 1], dtype=np.int32).tobytes()
        values = np.ones(k, dtype=np.float16).tobytes()
        with pytest.raises(GradientValidationError, match="shape"):
            unpack_compressed(header + shape + indices + values)

    def test_random_garbage_never_crashes(self):
        """Arbitrary bytes must produce GradientValidationError, not
        IndexError/MemoryError/segfault-adjacent behavior."""
        rng = np.random.default_rng(5)
        for size in (0, 1, 21, 22, 64, 500):
            blob = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
            with pytest.raises((GradientValidationError, ValueError)):
                unpack_compressed(blob)


class TestLinkDefaults:
    def test_wifi_default_is_aggressive(self):
        comp = AdaptiveCompressor(link_type=LinkType.WIFI)
        assert comp.level == CompressionLevel.AGGRESSIVE
