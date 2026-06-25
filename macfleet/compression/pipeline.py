"""Composable compression pipeline for MacFleet.

Allows chaining multiple compression stages:
  CompressPipeline([TopKCompressor(0.1), FP16Quantizer()])

This gives ~20x compression (Top-10% + FP16).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from macfleet.compression.quantize import FP16Quantizer
from macfleet.compression.topk import TopKCompressor


@dataclass
class CompressedGradient:
    """Container for a compressed gradient with all metadata.

    Stores the compressed data and information needed for decompression.
    """
    # For sparse compression (Top-K)
    indices: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None

    # For dense compression (quantization only)
    dense_data: Optional[torch.Tensor] = None

    # Metadata
    original_numel: int = 0
    original_dtype: torch.dtype = torch.float32
    original_shape: Optional[tuple] = None
    scale: float = 1.0

    # Compression info
    is_sparse: bool = False
    compression_stages: tuple = ()

    def to_bytes_estimate(self) -> int:
        """Estimate serialized size in bytes."""
        if self.is_sparse:
            # indices (int32) + values (fp16 or fp32)
            idx_bytes = self.indices.numel() * 4 if self.indices is not None else 0
            if self.values is not None:
                val_bytes = self.values.numel() * self.values.element_size()
            else:
                val_bytes = 0
            return idx_bytes + val_bytes
        else:
            if self.dense_data is not None:
                return self.dense_data.numel() * self.dense_data.element_size()
            return 0

    @property
    def compression_ratio(self) -> float:
        """Compute actual compression ratio."""
        original_bytes = self.original_numel * 4  # Assume FP32 original
        compressed_bytes = self.to_bytes_estimate()
        if compressed_bytes == 0:
            return 1.0
        return compressed_bytes / original_bytes


class Compressor(ABC):
    """Base class for compression stages."""

    @abstractmethod
    def compress(
        self,
        data: Union[torch.Tensor, CompressedGradient],
        name: Optional[str] = None,
    ) -> CompressedGradient:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        """Decompress to tensor."""
        pass


class TopKStage(Compressor):
    """Top-K compression stage."""

    def __init__(self, ratio: float = 0.1, device: str = "cpu"):
        self._compressor = TopKCompressor(ratio=ratio, device=device)
        self.ratio = ratio

    def compress(
        self,
        data: Union[torch.Tensor, CompressedGradient],
        name: Optional[str] = None,
    ) -> CompressedGradient:
        """Apply Top-K compression."""
        if isinstance(data, CompressedGradient):
            # Already compressed - decompress first
            tensor = self.decompress(data)
        else:
            tensor = data

        original_shape = tensor.shape
        indices, values, numel, dtype = self._compressor.compress(tensor, name)

        return CompressedGradient(
            indices=indices,
            values=values,
            original_numel=numel,
            original_dtype=dtype,
            original_shape=original_shape,
            is_sparse=True,
            compression_stages=("topk",),
        )

    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        """Decompress Top-K."""
        tensor = self._compressor.decompress(
            compressed.indices,
            compressed.values,
            compressed.original_numel,
            compressed.original_dtype,
            compressed.original_shape,
        )
        return tensor

    def reset(self) -> None:
        """Reset residuals."""
        self._compressor.reset_residuals()


class FP16Stage(Compressor):
    """FP16 quantization stage."""

    def __init__(self):
        self._quantizer = FP16Quantizer()

    def compress(
        self,
        data: Union[torch.Tensor, CompressedGradient],
        name: Optional[str] = None,
    ) -> CompressedGradient:
        """Apply FP16 quantization."""
        if isinstance(data, CompressedGradient):
            if data.is_sparse:
                # Quantize sparse values
                quantized_values, scale = self._quantizer.quantize(data.values)
                return CompressedGradient(
                    indices=data.indices,
                    values=quantized_values,
                    original_numel=data.original_numel,
                    original_dtype=data.original_dtype,
                    original_shape=data.original_shape,
                    scale=scale,
                    is_sparse=True,
                    compression_stages=data.compression_stages + ("fp16",),
                )
            else:
                # Quantize dense data
                quantized, scale = self._quantizer.quantize(data.dense_data)
                return CompressedGradient(
                    dense_data=quantized,
                    original_numel=data.original_numel,
                    original_dtype=data.original_dtype,
                    original_shape=data.original_shape,
                    scale=scale,
                    is_sparse=False,
                    compression_stages=data.compression_stages + ("fp16",),
                )
        else:
            # Compress raw tensor
            original_shape = data.shape
            quantized, scale = self._quantizer.quantize(data.flatten())
            return CompressedGradient(
                dense_data=quantized,
                original_numel=data.numel(),
                original_dtype=data.dtype,
                original_shape=original_shape,
                scale=scale,
                is_sparse=False,
                compression_stages=("fp16",),
            )

    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        """Dequantize FP16."""
        if compressed.is_sparse:
            # Dequantize values
            values = self._quantizer.dequantize(
                compressed.values,
                compressed.scale,
                compressed.original_dtype,
            )
            # Need to reconstruct dense tensor
            dense = torch.zeros(
                compressed.original_numel,
                dtype=compressed.original_dtype,
            )
            dense.scatter_(0, compressed.indices.long(), values)
            if compressed.original_shape:
                dense = dense.view(compressed.original_shape)
            return dense
        else:
            tensor = self._quantizer.dequantize(
                compressed.dense_data,
                compressed.scale,
                compressed.original_dtype,
            )
            if compressed.original_shape:
                tensor = tensor.view(compressed.original_shape)
            return tensor


class NoOpStage(Compressor):
    """No-op stage for testing or when compression is disabled."""

    def compress(
        self,
        data: Union[torch.Tensor, CompressedGradient],
        name: Optional[str] = None,
    ) -> CompressedGradient:
        if isinstance(data, CompressedGradient):
            return data

        return CompressedGradient(
            dense_data=data.flatten(),
            original_numel=data.numel(),
            original_dtype=data.dtype,
            original_shape=data.shape,
            is_sparse=False,
            compression_stages=("none",),
        )

    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        if compressed.is_sparse:
            dense = torch.zeros(compressed.original_numel, dtype=compressed.original_dtype)
            dense.scatter_(
                0,
                compressed.indices.long(),
                compressed.values.to(compressed.original_dtype),
            )
        else:
            dense = compressed.dense_data.to(compressed.original_dtype)

        if compressed.original_shape:
            return dense.view(compressed.original_shape)
        return dense


class CompressionPipeline:
    """Composable pipeline of compression stages.

    Example:
        pipeline = CompressionPipeline([
            TopKStage(ratio=0.1),
            FP16Stage(),
        ])

        compressed = pipeline.compress(gradient)
        decompressed = pipeline.decompress(compressed)
    """

    def __init__(self, stages: Optional[list[Compressor]] = None):
        """Initialize the pipeline.

        Args:
            stages: List of compression stages to apply in order.
        """
        self.stages = stages or []

    def __bool__(self) -> bool:
        """Pipeline is falsy when it has no compression stages."""
        return bool(self.stages)

    def add_stage(self, stage: Compressor) -> "CompressionPipeline":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def compress(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
    ) -> CompressedGradient:
        """Compress tensor through all stages.

        Args:
            tensor: Input tensor to compress.
            name: Optional name for residual tracking.

        Returns:
            Compressed gradient container.
        """
        if not self.stages:
            # No compression
            return NoOpStage().compress(tensor, name)

        data: Union[torch.Tensor, CompressedGradient] = tensor
        for stage in self.stages:
            data = stage.compress(data, name)

        return data

    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        """Decompress through stages (reverse order not needed with metadata).

        Args:
            compressed: Compressed gradient to decompress.

        Returns:
            Decompressed tensor.
        """
        if not self.stages:
            return NoOpStage().decompress(compressed)

        # Use last stage to decompress (it has all the info)
        return self.stages[-1].decompress(compressed)

    def reset(self) -> None:
        """Reset all stages (clear residuals etc)."""
        for stage in self.stages:
            if hasattr(stage, "reset"):
                stage.reset()

    @property
    def theoretical_ratio(self) -> float:
        """Theoretical compression ratio."""
        ratio = 1.0
        for stage in self.stages:
            if isinstance(stage, TopKStage):
                ratio *= stage.ratio
            elif isinstance(stage, FP16Stage):
                ratio *= 0.5
        return ratio


def create_pipeline(compression_type: str, topk_ratio: float = 0.1) -> CompressionPipeline:
    """Create a compression pipeline from a type string.

    Args:
        compression_type: One of "none", "topk", "fp16", "topk_fp16".
        topk_ratio: Ratio for Top-K (if used).

    Returns:
        Configured CompressionPipeline.
    """
    stages = []

    if compression_type == "none":
        pass
    elif compression_type == "topk":
        stages.append(TopKStage(ratio=topk_ratio))
    elif compression_type == "fp16":
        stages.append(FP16Stage())
    elif compression_type == "topk_fp16":
        stages.append(TopKStage(ratio=topk_ratio))
        stages.append(FP16Stage())
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    return CompressionPipeline(stages)
