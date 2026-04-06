"""Tests for gradient compression. Ported from MacFleet v1."""

import pytest
import torch

from macfleet.compression.topk import TopKCompressor, topk_compress, topk_decompress
from macfleet.compression.quantize import FP16Quantizer, Int8Quantizer, quantize_fp16, dequantize_fp16
from macfleet.compression.pipeline import (
    CompressionPipeline,
    TopKStage,
    FP16Stage,
    create_pipeline,
)


class TestTopKCompression:
    def test_basic_compression(self):
        compressor = TopKCompressor(ratio=0.1)
        tensor = torch.randn(1000)
        indices, values, numel, dtype = compressor.compress(tensor, "test")
        assert len(indices) == 100
        assert len(values) == 100
        assert numel == 1000
        assert dtype == torch.float32

    def test_decompression(self):
        compressor = TopKCompressor(ratio=0.1)
        tensor = torch.randn(1000)
        indices, values, numel, dtype = compressor.compress(tensor, "test")
        result = compressor.decompress(indices, values, numel, dtype)
        assert result.shape == tensor.shape

    def test_error_feedback(self):
        compressor = TopKCompressor(ratio=0.1)
        tensor1 = torch.ones(1000) * 0.1
        tensor1[0] = 10.0
        compressor.compress(tensor1, "param1")
        residual = compressor.get_residual("param1")
        assert residual is not None
        assert residual.sum().abs() > 0

        tensor2 = torch.ones(1000) * 0.1
        indices2, values2, _, _ = compressor.compress(tensor2, "param1")
        assert len(set(indices2.tolist())) > 0

    def test_different_ratios(self):
        tensor = torch.randn(10000)
        for ratio in [0.01, 0.05, 0.1, 0.5]:
            compressor = TopKCompressor(ratio=ratio)
            indices, values, numel, _ = compressor.compress(tensor)
            expected_count = int(numel * ratio)
            assert len(indices) == expected_count

    def test_stateless_functions(self):
        tensor = torch.randn(1000)
        indices, values, numel, dtype = topk_compress(tensor, ratio=0.1)
        result = topk_decompress(indices, values, numel, dtype)
        assert len(indices) == 100
        assert result.shape == tensor.shape

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            TopKCompressor(ratio=0.0)
        with pytest.raises(ValueError):
            TopKCompressor(ratio=1.5)

    def test_reset_residuals(self):
        compressor = TopKCompressor(ratio=0.1)
        tensor = torch.randn(1000)
        compressor.compress(tensor, "test")
        assert compressor.get_residual("test") is not None
        compressor.reset_residuals()
        assert compressor.get_residual("test") is None


class TestFP16Quantization:
    def test_basic_quantization(self):
        quantizer = FP16Quantizer()
        tensor = torch.randn(1000, dtype=torch.float32)
        quantized, scale = quantizer.quantize(tensor)
        assert quantized.dtype == torch.float16
        assert len(quantized) == len(tensor)

    def test_dequantization(self):
        tensor = torch.randn(1000, dtype=torch.float32)
        quantized, scale = quantize_fp16(tensor)
        result = dequantize_fp16(quantized, scale)
        max_error = (tensor - result).abs().max().item()
        assert max_error < 0.01

    def test_extreme_values(self):
        quantizer = FP16Quantizer()
        tensor = torch.randn(1000) * 10000
        quantized, scale = quantizer.quantize(tensor)
        assert not torch.isinf(quantized).any()
        assert not torch.isnan(quantized).any()

    def test_compression_ratio(self):
        assert FP16Quantizer().compression_ratio == 0.5


class TestInt8Quantization:
    def test_basic_quantization(self):
        quantizer = Int8Quantizer()
        tensor = torch.randn(1000, dtype=torch.float32)
        quantized, scale = quantizer.quantize(tensor)
        assert quantized.dtype == torch.int8
        assert len(quantized) == len(tensor)

    def test_roundtrip(self):
        quantizer = Int8Quantizer()
        tensor = torch.randn(1000, dtype=torch.float32)
        quantized, scale = quantizer.quantize(tensor)
        result = quantizer.dequantize(quantized, scale)
        max_error = (tensor - result).abs().max().item()
        assert max_error < 0.1  # INT8 has more error than FP16

    def test_compression_ratio(self):
        assert Int8Quantizer().compression_ratio == 0.25


class TestCompressionPipeline:
    def test_empty_pipeline(self):
        pipeline = CompressionPipeline([])
        tensor = torch.randn(1000)
        compressed = pipeline.compress(tensor)
        result = pipeline.decompress(compressed)
        assert torch.allclose(tensor, result)

    def test_topk_only(self):
        pipeline = create_pipeline("topk", 0.1)
        tensor = torch.randn(1000)
        compressed = pipeline.compress(tensor)
        assert compressed.is_sparse
        assert len(compressed.indices) == 100

    def test_fp16_only(self):
        pipeline = create_pipeline("fp16")
        tensor = torch.randn(1000)
        compressed = pipeline.compress(tensor)
        result = pipeline.decompress(compressed)
        assert not compressed.is_sparse
        assert compressed.dense_data.dtype == torch.float16
        assert result.shape == tensor.shape

    def test_topk_fp16_combined(self):
        pipeline = create_pipeline("topk_fp16", 0.1)
        tensor = torch.randn(10000)
        compressed = pipeline.compress(tensor)
        assert compressed.is_sparse
        assert len(compressed.indices) == 1000
        assert compressed.values.dtype == torch.float16
        result = pipeline.decompress(compressed)
        assert result.shape == tensor.shape

    def test_compression_ratio(self):
        pipeline = create_pipeline("topk_fp16", 0.1)
        tensor = torch.randn(10000)
        compressed = pipeline.compress(tensor)
        assert compressed.compression_ratio < 0.2
        assert compressed.compression_ratio > 0.1

    def test_reset_residuals(self):
        pipeline = create_pipeline("topk", 0.1)
        tensor = torch.randn(1000)
        pipeline.compress(tensor, "test")
        pipeline.compress(tensor, "test")
        pipeline.reset()
        if hasattr(pipeline.stages[0], "_compressor"):
            residual = pipeline.stages[0]._compressor.get_residual("test")
            assert residual is None

    def test_pipeline_bool(self):
        assert not CompressionPipeline([])
        assert CompressionPipeline([FP16Stage()])

    def test_theoretical_ratio(self):
        pipeline = create_pipeline("topk_fp16", 0.1)
        assert abs(pipeline.theoretical_ratio - 0.05) < 0.001

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_pipeline("invalid")

    def test_create_none(self):
        pipeline = create_pipeline("none")
        assert not pipeline  # falsy
