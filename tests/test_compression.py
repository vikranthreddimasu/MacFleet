"""Tests for gradient compression."""

import torch

from macfleet.compression.pipeline import (
    CompressionPipeline,
    create_pipeline,
)
from macfleet.compression.quantize import FP16Quantizer, dequantize_fp16, quantize_fp16
from macfleet.compression.topk import TopKCompressor, topk_compress, topk_decompress


class TestTopKCompression:
    """Tests for Top-K sparsification."""

    def test_basic_compression(self):
        """Test basic Top-K compression."""
        compressor = TopKCompressor(ratio=0.1)
        tensor = torch.randn(1000)

        indices, values, numel, dtype = compressor.compress(tensor, "test")

        assert len(indices) == 100  # 10% of 1000
        assert len(values) == 100
        assert numel == 1000
        assert dtype == torch.float32

    def test_decompression(self):
        """Test Top-K decompression."""
        compressor = TopKCompressor(ratio=0.1)
        tensor = torch.randn(1000)

        indices, values, numel, dtype = compressor.compress(tensor, "test")
        result = compressor.decompress(indices, values, numel, dtype)

        assert result.shape == tensor.shape

        # Top values should be preserved
        top_indices = tensor.abs().topk(100).indices
        for idx in top_indices[:50]:  # Check top 50
            assert result[idx] != 0 or tensor[idx].abs() < 1e-6

    def test_error_feedback(self):
        """Test error feedback accumulation."""
        compressor = TopKCompressor(ratio=0.1)

        # First compression
        tensor1 = torch.ones(1000) * 0.1  # Small values
        tensor1[0] = 10.0  # One large value

        indices1, values1, _, _ = compressor.compress(tensor1, "param1")

        # Residual should exist
        residual = compressor.get_residual("param1")
        assert residual is not None
        assert residual.sum().abs() > 0

        # Second compression - accumulated residuals should contribute
        tensor2 = torch.ones(1000) * 0.1
        indices2, values2, _, _ = compressor.compress(tensor2, "param1")

        # Some previously small values should now be selected
        assert len(set(indices2.tolist())) > 0

    def test_different_ratios(self):
        """Test different compression ratios."""
        tensor = torch.randn(10000)

        for ratio in [0.01, 0.05, 0.1, 0.5]:
            compressor = TopKCompressor(ratio=ratio)
            indices, values, numel, _ = compressor.compress(tensor)

            expected_count = int(numel * ratio)
            assert len(indices) == expected_count

    def test_stateless_functions(self):
        """Test stateless compression functions."""
        tensor = torch.randn(1000)

        indices, values, numel, dtype = topk_compress(tensor, ratio=0.1)
        result = topk_decompress(indices, values, numel, dtype)

        assert len(indices) == 100
        assert result.shape == tensor.shape


class TestFP16Quantization:
    """Tests for FP16 quantization."""

    def test_basic_quantization(self):
        """Test basic FP16 quantization."""
        quantizer = FP16Quantizer()
        tensor = torch.randn(1000, dtype=torch.float32)

        quantized, scale = quantizer.quantize(tensor)

        assert quantized.dtype == torch.float16
        assert len(quantized) == len(tensor)

    def test_dequantization(self):
        """Test FP16 dequantization round-trip."""
        tensor = torch.randn(1000, dtype=torch.float32)

        quantized, scale = quantize_fp16(tensor)
        result = dequantize_fp16(quantized, scale)

        # Should be close within FP16 precision
        max_error = (tensor - result).abs().max().item()
        assert max_error < 0.01  # Within 1% error

    def test_extreme_values(self):
        """Test quantization with extreme values."""
        quantizer = FP16Quantizer()

        # Large values
        tensor = torch.randn(1000) * 10000
        quantized, scale = quantizer.quantize(tensor)
        result = quantizer.dequantize(quantized, scale)

        # Should not have inf or nan
        assert not torch.isinf(quantized).any()
        assert not torch.isnan(quantized).any()
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_compression_ratio(self):
        """Test that FP16 gives 2x compression."""
        quantizer = FP16Quantizer()
        assert quantizer.compression_ratio == 0.5


class TestCompressionPipeline:
    """Tests for composable compression pipeline."""

    def test_empty_pipeline(self):
        """Test pipeline with no stages."""
        pipeline = CompressionPipeline([])
        tensor = torch.randn(1000)

        compressed = pipeline.compress(tensor)
        result = pipeline.decompress(compressed)

        assert torch.allclose(tensor, result)

    def test_topk_only(self):
        """Test pipeline with only Top-K."""
        pipeline = create_pipeline("topk", 0.1)
        tensor = torch.randn(1000)

        compressed = pipeline.compress(tensor)

        assert compressed.is_sparse
        assert len(compressed.indices) == 100

    def test_fp16_only(self):
        """Test pipeline with only FP16."""
        pipeline = create_pipeline("fp16")
        tensor = torch.randn(1000)

        compressed = pipeline.compress(tensor)
        result = pipeline.decompress(compressed)

        assert not compressed.is_sparse
        assert compressed.dense_data.dtype == torch.float16
        assert result.shape == tensor.shape

    def test_topk_fp16_combined(self):
        """Test combined TopK + FP16 pipeline."""
        pipeline = create_pipeline("topk_fp16", 0.1)
        tensor = torch.randn(10000)

        compressed = pipeline.compress(tensor)

        assert compressed.is_sparse
        assert len(compressed.indices) == 1000  # 10%
        assert compressed.values.dtype == torch.float16

        result = pipeline.decompress(compressed)
        assert result.shape == tensor.shape

    def test_compression_ratio(self):
        """Test overall compression ratio."""
        pipeline = create_pipeline("topk_fp16", 0.1)
        tensor = torch.randn(10000)

        compressed = pipeline.compress(tensor)

        # Original: 10000 * 4 = 40000 bytes
        # Compressed: 1000 indices * 4 + 1000 values * 2 = 6000 bytes
        # Ratio should be ~0.15
        assert compressed.compression_ratio < 0.2
        assert compressed.compression_ratio > 0.1

    def test_reset_residuals(self):
        """Test resetting pipeline residuals."""
        pipeline = create_pipeline("topk", 0.1)

        # Compress with residual accumulation
        tensor = torch.randn(1000)
        pipeline.compress(tensor, "test")
        pipeline.compress(tensor, "test")

        # Reset
        pipeline.reset()

        # Residuals should be cleared
        if hasattr(pipeline.stages[0], "_compressor"):
            residual = pipeline.stages[0]._compressor.get_residual("test")
            assert residual is None
