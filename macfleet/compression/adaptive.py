"""Bandwidth-aware adaptive compression for gradient arrays.

Works at the numpy level (framework-agnostic), adjusting compression
based on network link quality. Integrated into DataParallel's
sync_gradients flow.

Compression levels by link type:
    Thunderbolt (>10 Gbps): OFF or FP16 only
    Ethernet (~1 Gbps):     TopK 10% + FP16
    WiFi (~100 Mbps):       TopK 1% + FP16 (aggressive)

The compressor tracks error feedback (residuals) for TopK to maintain
convergence despite lossy compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from macfleet.pool.network import LinkType


class CompressionLevel(Enum):
    """Compression presets mapped to network conditions."""
    NONE = "none"           # No compression (Thunderbolt)
    LIGHT = "light"         # FP16 only (~2x)
    MODERATE = "moderate"   # TopK 10% + FP16 (~20x)
    AGGRESSIVE = "aggressive"  # TopK 1% + FP16 (~200x)


@dataclass
class CompressedArray:
    """Compressed numpy array with metadata for decompression."""
    data: bytes             # Serialized compressed data
    original_shape: tuple   # Original array shape
    original_size: int      # Original byte count
    compressed_size: int    # Compressed byte count
    level: CompressionLevel
    # TopK metadata (if sparse)
    topk_k: int = 0
    topk_indices: Optional[np.ndarray] = None
    topk_values: Optional[np.ndarray] = None
    # FP16 metadata
    is_fp16: bool = False
    scale: float = 1.0

    @property
    def ratio(self) -> float:
        """Compression ratio (smaller is better)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


# ------------------------------------------------------------------ #
# Numpy-native TopK compressor with error feedback                   #
# ------------------------------------------------------------------ #


class NumpyTopKCompressor:
    """TopK sparsification on numpy arrays with residual error feedback.

    Keeps track of accumulated errors from discarded gradient values
    and adds them back in the next compression round, preserving
    convergence properties.
    """

    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio
        self._residuals: Optional[np.ndarray] = None

    def compress(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Compress array to top-K values.

        Args:
            array: 1D float32 numpy array.

        Returns:
            (indices, values, original_numel) tuple.
        """
        flat = array.flatten().astype(np.float32)

        # Add residuals from previous round
        if self._residuals is not None and self._residuals.shape == flat.shape:
            flat = flat + self._residuals

        k = max(1, int(flat.size * self.ratio))
        abs_vals = np.abs(flat)
        top_indices = np.argpartition(abs_vals, -k)[-k:]
        top_indices = top_indices[np.argsort(abs_vals[top_indices])[::-1]]
        top_values = flat[top_indices].astype(np.float32)

        # Store residuals (what we discarded)
        residual = flat.copy()
        residual[top_indices] = 0.0
        self._residuals = residual

        return top_indices.astype(np.int32), top_values, flat.size

    def decompress(
        self, indices: np.ndarray, values: np.ndarray, numel: int
    ) -> np.ndarray:
        """Reconstruct array from sparse representation."""
        result = np.zeros(numel, dtype=np.float32)
        result[indices] = values
        return result

    def reset(self) -> None:
        """Reset residual accumulation."""
        self._residuals = None


class NumpyFP16Compressor:
    """FP16 quantization for numpy arrays.

    Scales values to FP16 range, quantizes, and stores the scale
    factor for dequantization.
    """

    def compress(self, array: np.ndarray) -> tuple[np.ndarray, float]:
        """Quantize to FP16.

        Returns:
            (fp16_array, scale_factor) tuple.
        """
        flat = array.flatten().astype(np.float32)
        max_val = np.abs(flat).max()
        if max_val == 0:
            return flat.astype(np.float16), 1.0

        # Scale to FP16 range to minimize precision loss
        scale = max_val / np.finfo(np.float16).max * 1.1
        scaled = (flat / scale).astype(np.float16)
        return scaled, float(scale)

    def decompress(self, fp16_array: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize from FP16."""
        return fp16_array.astype(np.float32) * scale


# ------------------------------------------------------------------ #
# Adaptive compression pipeline                                      #
# ------------------------------------------------------------------ #


# Map link types to default compression levels
LINK_COMPRESSION_MAP: dict[LinkType, CompressionLevel] = {
    LinkType.THUNDERBOLT: CompressionLevel.NONE,
    LinkType.LOOPBACK: CompressionLevel.NONE,
    LinkType.ETHERNET: CompressionLevel.MODERATE,
    LinkType.WIFI: CompressionLevel.AGGRESSIVE,
    LinkType.UNKNOWN: CompressionLevel.MODERATE,
}

# TopK ratios per compression level
TOPK_RATIOS: dict[CompressionLevel, float] = {
    CompressionLevel.NONE: 1.0,
    CompressionLevel.LIGHT: 1.0,     # FP16 only, no sparsification
    CompressionLevel.MODERATE: 0.1,  # Keep 10%
    CompressionLevel.AGGRESSIVE: 0.01,  # Keep 1%
}


@dataclass
class AdaptiveCompressionConfig:
    """Configuration for adaptive compression."""
    # Override auto-detection with a fixed level
    fixed_level: Optional[CompressionLevel] = None
    # Enable FP16 quantization (applied after TopK if both active)
    use_fp16: bool = True
    # Minimum array size to compress (tiny arrays aren't worth it)
    min_compress_size: int = 1024
    # Warmup: disable compression for first N steps
    warmup_steps: int = 0
    # Bandwidth threshold (Mbps) for auto-selection when link type unknown
    bw_threshold_aggressive: float = 200.0   # below this → aggressive
    bw_threshold_moderate: float = 2000.0    # below this → moderate


class AdaptiveCompressor:
    """Bandwidth-aware gradient compression for numpy arrays.

    Automatically selects compression level based on network link type
    or measured bandwidth. Integrates TopK sparsification with error
    feedback and optional FP16 quantization.

    Usage:
        compressor = AdaptiveCompressor(link_type=LinkType.WIFI)
        compressed = compressor.compress(gradient_array)
        decompressed = compressor.decompress(compressed)

    Or with auto-detection:
        compressor = AdaptiveCompressor.for_link(link_type)
    """

    def __init__(
        self,
        link_type: LinkType = LinkType.UNKNOWN,
        config: Optional[AdaptiveCompressionConfig] = None,
        bandwidth_mbps: Optional[float] = None,
    ):
        self.config = config or AdaptiveCompressionConfig()
        self._step = 0

        # Determine compression level
        if self.config.fixed_level is not None:
            self._level = self.config.fixed_level
        elif bandwidth_mbps is not None:
            self._level = self._level_from_bandwidth(bandwidth_mbps)
        else:
            self._level = LINK_COMPRESSION_MAP.get(link_type, CompressionLevel.MODERATE)

        # Initialize sub-compressors
        topk_ratio = TOPK_RATIOS[self._level]
        self._topk = NumpyTopKCompressor(ratio=topk_ratio) if topk_ratio < 1.0 else None
        self._fp16 = NumpyFP16Compressor() if self.config.use_fp16 and self._level != CompressionLevel.NONE else None

    @property
    def level(self) -> CompressionLevel:
        return self._level

    @property
    def active(self) -> bool:
        """Whether compression is actually active (past warmup, level != NONE)."""
        if self._step <= self.config.warmup_steps:
            return False
        return self._level != CompressionLevel.NONE

    def _level_from_bandwidth(self, bw_mbps: float) -> CompressionLevel:
        """Select compression level from measured bandwidth."""
        if bw_mbps <= 0:
            return CompressionLevel.AGGRESSIVE
        if bw_mbps < self.config.bw_threshold_aggressive:
            return CompressionLevel.AGGRESSIVE
        if bw_mbps < self.config.bw_threshold_moderate:
            return CompressionLevel.MODERATE
        return CompressionLevel.NONE

    def compress(self, array: np.ndarray) -> np.ndarray | CompressedArray:
        """Compress a gradient array based on current level.

        During warmup or with NONE level, returns the array unchanged.
        Otherwise returns a CompressedArray with metadata.

        Args:
            array: 1D or nD float32 numpy array.

        Returns:
            Original array if no compression, or CompressedArray.
        """
        self._step += 1

        if not self.active:
            return array

        original_bytes = array.nbytes
        flat = array.flatten().astype(np.float32)

        if flat.size < self.config.min_compress_size:
            return array

        # Stage 1: TopK sparsification
        if self._topk is not None:
            indices, values, numel = self._topk.compress(flat)

            # Stage 2: FP16 on values
            if self._fp16 is not None:
                fp16_values, scale = self._fp16.compress(values)
                return CompressedArray(
                    data=b"",  # Not used for in-memory path
                    original_shape=array.shape,
                    original_size=original_bytes,
                    compressed_size=indices.nbytes + fp16_values.nbytes,
                    level=self._level,
                    topk_k=len(indices),
                    topk_indices=indices,
                    topk_values=fp16_values,
                    is_fp16=True,
                    scale=scale,
                )

            return CompressedArray(
                data=b"",
                original_shape=array.shape,
                original_size=original_bytes,
                compressed_size=indices.nbytes + values.nbytes,
                level=self._level,
                topk_k=len(indices),
                topk_indices=indices,
                topk_values=values,
                is_fp16=False,
            )

        # FP16 only (LIGHT level)
        if self._fp16 is not None:
            fp16_data, scale = self._fp16.compress(flat)
            return CompressedArray(
                data=b"",
                original_shape=array.shape,
                original_size=original_bytes,
                compressed_size=fp16_data.nbytes,
                level=self._level,
                topk_values=fp16_data,
                is_fp16=True,
                scale=scale,
            )

        return array

    def decompress(self, data: np.ndarray | CompressedArray) -> np.ndarray:
        """Decompress back to original numpy array.

        Args:
            data: Original array or CompressedArray from compress().

        Returns:
            Reconstructed float32 numpy array.
        """
        if isinstance(data, np.ndarray):
            return data

        ca = data

        # FP16 dequantize values first
        if ca.is_fp16 and ca.topk_values is not None:
            values = (
                self._fp16.decompress(ca.topk_values, ca.scale)
                if self._fp16
                else ca.topk_values.astype(np.float32)
            )
        elif ca.topk_values is not None:
            values = ca.topk_values.astype(np.float32)
        else:
            return np.zeros(ca.original_shape, dtype=np.float32)

        # TopK reconstruction
        if ca.topk_indices is not None:
            numel = 1
            for s in ca.original_shape:
                numel *= s
            result = np.zeros(numel, dtype=np.float32)
            result[ca.topk_indices] = values
            return result.reshape(ca.original_shape)

        # Dense FP16 only
        return values.reshape(ca.original_shape)

    def reset(self) -> None:
        """Reset residual accumulation (call between training runs)."""
        if self._topk:
            self._topk.reset()
        self._step = 0

    def update_link(
        self,
        link_type: Optional[LinkType] = None,
        bandwidth_mbps: Optional[float] = None,
    ) -> None:
        """Update compression level based on changed network conditions.

        Called periodically by the scheduler when link quality changes.
        """
        if self.config.fixed_level is not None:
            return

        if bandwidth_mbps is not None:
            new_level = self._level_from_bandwidth(bandwidth_mbps)
        elif link_type is not None:
            new_level = LINK_COMPRESSION_MAP.get(link_type, self._level)
        else:
            return

        if new_level != self._level:
            self._level = new_level
            ratio = TOPK_RATIOS[new_level]
            if ratio < 1.0:
                self._topk = NumpyTopKCompressor(ratio=ratio)
            else:
                self._topk = None
            if new_level == CompressionLevel.NONE:
                self._fp16 = None
            elif self.config.use_fp16:
                self._fp16 = NumpyFP16Compressor()

    @property
    def stats(self) -> dict:
        """Compression statistics."""
        return {
            "level": self._level.value,
            "step": self._step,
            "warmup_remaining": max(0, self.config.warmup_steps - self._step),
            "topk_ratio": self._topk.ratio if self._topk else 1.0,
            "fp16_enabled": self._fp16 is not None,
        }
