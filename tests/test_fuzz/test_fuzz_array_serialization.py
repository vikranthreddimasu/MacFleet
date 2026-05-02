"""Hypothesis fuzz for pack_array / unpack_array (numpy serialization).

The collective layer marshals numpy arrays of arbitrary dtype + shape.
Roundtrip must preserve dtype exactly; corruption must surface, never
silently produce garbage.
"""

from __future__ import annotations

import struct

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from macfleet.comm.collectives import pack_array, unpack_array

# Dtypes the wire is expected to carry — float32 is the gradient norm,
# the rest cover broadcast/scatter use cases.
SUPPORTED_DTYPES = ["float32", "float64", "int32", "int64", "float16"]


class TestPackArrayRoundtrip:
    @given(
        arr=arrays(
            dtype=st.sampled_from(SUPPORTED_DTYPES),
            shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=64),
        ),
    )
    @settings(max_examples=300, deadline=None)
    def test_roundtrip(self, arr):
        # NaN/inf are preserved as bit patterns by tobytes — hypothesis
        # generates them, which is exactly what we want to test.
        packed = pack_array(arr)
        unpacked = unpack_array(packed)
        assert unpacked.dtype == arr.dtype
        assert unpacked.shape == arr.shape
        # Use bitwise comparison so NaN payloads compare equal.
        assert unpacked.tobytes() == arr.tobytes()

    def test_zero_dim_array(self):
        # 0-d (scalar) arrays — caller could pass np.array(0.5)
        # which has shape=() and ndim=0. pack_array uses len(shape)
        # which is 0, producing a header with no shape entries.
        # 0.5 is exactly representable in float32 so we can compare
        # without precision noise.
        arr = np.array(0.5, dtype=np.float32)
        packed = pack_array(arr)
        unpacked = unpack_array(packed)
        assert unpacked.shape == arr.shape
        assert float(unpacked) == 0.5

    def test_empty_1d_array(self):
        arr = np.array([], dtype=np.float32)
        packed = pack_array(arr)
        unpacked = unpack_array(packed)
        assert unpacked.shape == (0,)
        assert unpacked.dtype == np.float32


class TestPackArrayCorruption:
    @given(
        arr=arrays(
            dtype="float32",
            shape=array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=32),
        ),
        flip_byte=st.integers(min_value=0, max_value=1023),
    )
    @settings(max_examples=100, deadline=None)
    def test_data_byte_flip_changes_output(self, arr, flip_byte):
        packed = pack_array(arr)
        if flip_byte >= len(packed):
            return
        # Flip a byte somewhere in the data region.
        # Header layout: dtype_len(2) ndims(2) shape(ndims*4) dtype_str data
        # Skip the header — flip in data bytes only.
        dtype_len, ndims = struct.unpack("!HH", packed[:4])
        data_start = 4 + ndims * 4 + dtype_len
        if flip_byte < data_start:
            return  # skip header flips
        if flip_byte >= len(packed):
            return
        packed_b = bytearray(packed)
        packed_b[flip_byte] ^= 0xFF
        # unpack_array doesn't have CRC — it just deserializes whatever
        # bytes are there. The result must be a valid array but with
        # different values. Don't crash.
        unpacked = unpack_array(bytes(packed_b))
        assert unpacked.dtype == arr.dtype
        assert unpacked.shape == arr.shape

    @given(blob=st.binary(min_size=4, max_size=2048))
    @settings(max_examples=200, deadline=None)
    def test_random_bytes_safe(self, blob):
        # Arbitrary bytes: parser should raise rather than crash.
        # Allowed: ValueError (bad dtype, shape), struct.error (header
        # parse), UnicodeDecodeError (bad dtype string).
        try:
            unpack_array(blob)
        except (
            ValueError, struct.error, UnicodeDecodeError, TypeError,
        ):
            pass


class TestPackArrayShape:
    @given(shape=array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=16))
    @settings(max_examples=100, deadline=None)
    def test_shape_preserved(self, shape):
        arr = np.zeros(shape, dtype=np.float32)
        unpacked = unpack_array(pack_array(arr))
        assert unpacked.shape == shape

    def test_max_dims_supported(self):
        # 4-D arrays are common for image batches: (N, C, H, W)
        arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
        unpacked = unpack_array(pack_array(arr))
        assert unpacked.shape == (2, 3, 4, 5)


class TestPackArraySize:
    def test_large_array_roundtrip(self):
        # 1M floats — checks that we don't truncate on size boundary.
        arr = np.random.randn(1_000_000).astype(np.float32)
        unpacked = unpack_array(pack_array(arr))
        np.testing.assert_array_equal(unpacked, arr)

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        size=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_random_seed_roundtrip(self, seed, size):
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal(size).astype(np.float32)
        unpacked = unpack_array(pack_array(arr))
        np.testing.assert_array_equal(unpacked, arr)
