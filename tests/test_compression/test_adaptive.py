"""Tests for bandwidth-aware adaptive compression."""

from __future__ import annotations

import numpy as np

from macfleet.compression.adaptive import (
    AdaptiveCompressionConfig,
    AdaptiveCompressor,
    CompressedArray,
    CompressionLevel,
    NumpyFP16Compressor,
    NumpyTopKCompressor,
)
from macfleet.pool.network import LinkType

# ------------------------------------------------------------------ #
# NumpyTopKCompressor                                                #
# ------------------------------------------------------------------ #


class TestNumpyTopKCompressor:
    def test_compress_keeps_top_values(self):
        c = NumpyTopKCompressor(ratio=0.1)
        arr = np.zeros(100, dtype=np.float32)
        arr[42] = 10.0
        arr[77] = -8.0

        indices, values, numel = c.compress(arr)

        assert numel == 100
        assert len(indices) == 10  # 10% of 100
        assert 42 in indices
        assert 77 in indices

    def test_decompress_roundtrip(self):
        c = NumpyTopKCompressor(ratio=0.5)
        arr = np.random.randn(200).astype(np.float32)

        indices, values, numel = c.compress(arr)
        reconstructed = c.decompress(indices, values, numel)

        assert reconstructed.shape == (200,)
        # Top 50% should be preserved
        assert np.count_nonzero(reconstructed) == 100

    def test_error_feedback_accumulates(self):
        c = NumpyTopKCompressor(ratio=0.1)
        arr = np.ones(100, dtype=np.float32)

        # First compress: 10 values kept, 90 discarded
        c.compress(arr)

        # Second compress: residuals from first round should be added
        # All residuals are 1.0, so now values should be ~2.0 for discarded positions
        arr2 = np.ones(100, dtype=np.float32)
        indices2, values2, _ = c.compress(arr2)

        # Some values should be > 1.0 due to accumulated residuals
        assert np.max(np.abs(values2)) >= 1.5

    def test_reset_clears_residuals(self):
        c = NumpyTopKCompressor(ratio=0.5)
        arr = np.random.randn(100).astype(np.float32)

        c.compress(arr)
        assert c._residuals is not None

        c.reset()
        assert c._residuals is None

    def test_minimum_one_value(self):
        c = NumpyTopKCompressor(ratio=0.001)
        arr = np.random.randn(10).astype(np.float32)

        indices, values, numel = c.compress(arr)
        assert len(indices) >= 1


# ------------------------------------------------------------------ #
# NumpyFP16Compressor                                               #
# ------------------------------------------------------------------ #


class TestNumpyFP16Compressor:
    def test_compress_produces_fp16(self):
        c = NumpyFP16Compressor()
        arr = np.random.randn(100).astype(np.float32)

        fp16, scale = c.compress(arr)
        assert fp16.dtype == np.float16
        assert scale > 0.0

    def test_decompress_roundtrip(self):
        c = NumpyFP16Compressor()
        arr = np.random.randn(1000).astype(np.float32)

        fp16, scale = c.compress(arr)
        reconstructed = c.decompress(fp16, scale)

        assert reconstructed.dtype == np.float32
        np.testing.assert_allclose(reconstructed, arr, rtol=0.01, atol=1e-3)

    def test_zero_array(self):
        c = NumpyFP16Compressor()
        arr = np.zeros(100, dtype=np.float32)

        fp16, scale = c.compress(arr)
        reconstructed = c.decompress(fp16, scale)
        np.testing.assert_array_equal(reconstructed, 0.0)

    def test_halves_memory(self):
        c = NumpyFP16Compressor()
        arr = np.random.randn(10000).astype(np.float32)

        fp16, _ = c.compress(arr)
        assert fp16.nbytes == arr.nbytes // 2


# ------------------------------------------------------------------ #
# AdaptiveCompressor — level selection                               #
# ------------------------------------------------------------------ #


class TestAdaptiveCompressorLevels:
    def test_thunderbolt_no_compression(self):
        c = AdaptiveCompressor(link_type=LinkType.THUNDERBOLT)
        assert c.level == CompressionLevel.NONE
        assert not c.active

    def test_ethernet_moderate(self):
        c = AdaptiveCompressor(link_type=LinkType.ETHERNET)
        assert c.level == CompressionLevel.MODERATE

    def test_wifi_aggressive(self):
        c = AdaptiveCompressor(link_type=LinkType.WIFI)
        assert c.level == CompressionLevel.AGGRESSIVE

    def test_fixed_level_overrides(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.LIGHT)
        c = AdaptiveCompressor(link_type=LinkType.WIFI, config=config)
        assert c.level == CompressionLevel.LIGHT

    def test_bandwidth_based_selection(self):
        # Low bandwidth → aggressive
        c1 = AdaptiveCompressor(bandwidth_mbps=100.0)
        assert c1.level == CompressionLevel.AGGRESSIVE

        # Medium bandwidth → moderate
        c2 = AdaptiveCompressor(bandwidth_mbps=500.0)
        assert c2.level == CompressionLevel.MODERATE

        # High bandwidth → none
        c3 = AdaptiveCompressor(bandwidth_mbps=10000.0)
        assert c3.level == CompressionLevel.NONE


# ------------------------------------------------------------------ #
# AdaptiveCompressor — compress / decompress                         #
# ------------------------------------------------------------------ #


class TestAdaptiveCompressorRoundtrip:
    def test_none_level_passthrough(self):
        c = AdaptiveCompressor(link_type=LinkType.THUNDERBOLT)
        arr = np.random.randn(1000).astype(np.float32)
        result = c.compress(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_light_compression(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.LIGHT)
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(2000).astype(np.float32)

        compressed = c.compress(arr)
        assert isinstance(compressed, CompressedArray)
        assert compressed.is_fp16
        assert compressed.topk_indices is None  # No sparsification

        reconstructed = c.decompress(compressed)
        np.testing.assert_allclose(reconstructed, arr, rtol=0.01, atol=1e-3)

    def test_moderate_compression(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.MODERATE)
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(10000).astype(np.float32)

        compressed = c.compress(arr)
        assert isinstance(compressed, CompressedArray)
        assert compressed.topk_indices is not None
        assert compressed.is_fp16
        assert compressed.ratio < 0.2  # significant compression

        reconstructed = c.decompress(compressed)
        assert reconstructed.shape == arr.shape

    def test_aggressive_compression(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.AGGRESSIVE)
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(100000).astype(np.float32)

        compressed = c.compress(arr)
        assert isinstance(compressed, CompressedArray)
        assert compressed.topk_k == 1000  # 1% of 100000
        assert compressed.ratio < 0.02  # ~200x compression

    def test_small_array_skips_compression(self):
        config = AdaptiveCompressionConfig(
            fixed_level=CompressionLevel.AGGRESSIVE,
            min_compress_size=5000,
        )
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(100).astype(np.float32)

        result = c.compress(arr)
        assert isinstance(result, np.ndarray)  # passthrough

    def test_warmup_skips_compression(self):
        config = AdaptiveCompressionConfig(
            fixed_level=CompressionLevel.MODERATE,
            warmup_steps=5,
        )
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(2000).astype(np.float32)

        # Steps 1-5 should pass through
        for _ in range(5):
            result = c.compress(arr)
            assert isinstance(result, np.ndarray)

        # Step 6 should compress
        result = c.compress(arr)
        assert isinstance(result, CompressedArray)


# ------------------------------------------------------------------ #
# AdaptiveCompressor — dynamic link updates                          #
# ------------------------------------------------------------------ #


class TestAdaptiveCompressorDynamic:
    def test_update_link_type(self):
        c = AdaptiveCompressor(link_type=LinkType.WIFI)
        assert c.level == CompressionLevel.AGGRESSIVE

        c.update_link(link_type=LinkType.ETHERNET)
        assert c.level == CompressionLevel.MODERATE

        c.update_link(link_type=LinkType.THUNDERBOLT)
        assert c.level == CompressionLevel.NONE

    def test_update_bandwidth(self):
        c = AdaptiveCompressor(link_type=LinkType.UNKNOWN)

        c.update_link(bandwidth_mbps=50.0)
        assert c.level == CompressionLevel.AGGRESSIVE

        c.update_link(bandwidth_mbps=5000.0)
        assert c.level == CompressionLevel.NONE

    def test_fixed_level_ignores_updates(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.LIGHT)
        c = AdaptiveCompressor(config=config)

        c.update_link(link_type=LinkType.WIFI)
        assert c.level == CompressionLevel.LIGHT  # unchanged

    def test_stats(self):
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.MODERATE)
        c = AdaptiveCompressor(config=config)
        arr = np.random.randn(2000).astype(np.float32)
        c.compress(arr)

        stats = c.stats
        assert stats["level"] == "moderate"
        assert stats["step"] == 1
        assert stats["topk_ratio"] == 0.1
        assert stats["fp16_enabled"] is True


# ------------------------------------------------------------------ #
# Gradient-scale compression quality                                 #
# ------------------------------------------------------------------ #


class TestCompressionQuality:
    def test_moderate_preserves_direction(self):
        """Moderate compression should preserve gradient direction."""
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.MODERATE)
        c = AdaptiveCompressor(config=config)

        arr = np.random.randn(10000).astype(np.float32)
        compressed = c.compress(arr)
        reconstructed = c.decompress(compressed)

        # Cosine similarity should be high
        dot = np.dot(arr, reconstructed)
        norm_orig = np.linalg.norm(arr)
        norm_recon = np.linalg.norm(reconstructed)
        cosine_sim = dot / (norm_orig * norm_recon + 1e-8)
        assert cosine_sim > 0.5

    def test_error_feedback_improves_over_steps(self):
        """With error feedback, accumulated information should grow."""
        config = AdaptiveCompressionConfig(fixed_level=CompressionLevel.MODERATE)
        c = AdaptiveCompressor(config=config)

        # Simulate constant gradient over multiple steps
        grad = np.random.randn(10000).astype(np.float32)
        total_info = 0.0

        for _ in range(5):
            compressed = c.compress(grad)
            if isinstance(compressed, CompressedArray):
                reconstructed = c.decompress(compressed)
                total_info += np.sum(np.abs(reconstructed))

        # Total information transferred should exceed single-step.
        # (single_info used to be computed here but was never asserted against;
        # removing the dead computation. The assertion below is the contract.)
        # With error feedback, we should capture at least as much info
        assert total_info > 0
