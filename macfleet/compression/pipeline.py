"""Composable compression pipeline for MacFleet.

Ported from v1. Allows chaining multiple compression stages:
    CompressionPipeline([TopKStage(0.1), FP16Stage()])
Gives ~20x compression (Top-10% + FP16).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from macfleet.compression.quantize import FP16Quantizer
from macfleet.compression.topk import TopKCompressor


@dataclass
class CompressedGradient:
    """Container for a compressed gradient with all metadata."""

    # Sparse compression (Top-K)
    indices: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None

    # Dense compression (quantization only)
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
            idx_bytes = self.indices.numel() * 4 if self.indices is not None else 0
            val_bytes = (
                self.values.numel() * self.values.element_size()
                if self.values is not None
                else 0
            )
            return idx_bytes + val_bytes
        else:
            return (
                self.dense_data.numel() * self.dense_data.element_size()
                if self.dense_data is not None
                else 0
            )

    @property
    def compression_ratio(self) -> float:
        """Actual compression ratio (compressed / original)."""
        original_bytes = self.original_numel * 4
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
        ...

    @abstractmethod
    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        ...


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
        if isinstance(data, CompressedGradient):
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
        return self._compressor.decompress(
            compressed.indices,
            compressed.values,
            compressed.original_numel,
            compressed.original_dtype,
            compressed.original_shape,
        )

    def reset(self) -> None:
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
        if isinstance(data, CompressedGradient):
            if data.is_sparse:
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
        if compressed.is_sparse:
            values = self._quantizer.dequantize(
                compressed.values, compressed.scale, compressed.original_dtype
            )
            dense = torch.zeros(compressed.original_numel, dtype=compressed.original_dtype)
            dense.scatter_(0, compressed.indices.long(), values)
            if compressed.original_shape:
                dense = dense.view(compressed.original_shape)
            return dense
        else:
            tensor = self._quantizer.dequantize(
                compressed.dense_data, compressed.scale, compressed.original_dtype
            )
            if compressed.original_shape:
                tensor = tensor.view(compressed.original_shape)
            return tensor


class NoOpStage(Compressor):
    """No-op stage for when compression is disabled."""

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
                0, compressed.indices.long(), compressed.values.to(compressed.original_dtype)
            )
        else:
            dense = compressed.dense_data.to(compressed.original_dtype)
        if compressed.original_shape:
            return dense.view(compressed.original_shape)
        return dense


class CompressionPipeline:
    """Composable pipeline of compression stages.

    Example:
        pipeline = CompressionPipeline([TopKStage(ratio=0.1), FP16Stage()])
        compressed = pipeline.compress(gradient)
        decompressed = pipeline.decompress(compressed)
    """

    def __init__(self, stages: Optional[list[Compressor]] = None):
        self.stages = stages or []

    def __bool__(self) -> bool:
        return bool(self.stages)

    def add_stage(self, stage: Compressor) -> "CompressionPipeline":
        self.stages.append(stage)
        return self

    def compress(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
    ) -> CompressedGradient:
        if not self.stages:
            return NoOpStage().compress(tensor, name)

        data: Union[torch.Tensor, CompressedGradient] = tensor
        for stage in self.stages:
            data = stage.compress(data, name)
        return data

    def decompress(self, compressed: CompressedGradient) -> torch.Tensor:
        if not self.stages:
            return NoOpStage().decompress(compressed)
        return self.stages[-1].decompress(compressed)

    def reset(self) -> None:
        for stage in self.stages:
            if hasattr(stage, "reset"):
                stage.reset()

    @property
    def theoretical_ratio(self) -> float:
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
    """
    stages: list[Compressor] = []

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
