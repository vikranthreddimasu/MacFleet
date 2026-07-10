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

Wire format (v2.3): `pack_compressed` / `unpack_compressed` serialize a
CompressedArray for the 2-node sparse-on-wire exchange in DataParallel —
the payload that actually crosses the network shrinks by the compression
ratio instead of being decompressed back to dense first. Unpacking is
fail-closed: every length and count is validated before any allocation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from macfleet.pool.network import LinkType
from macfleet.security.auth import GradientValidationError, validate_gradient_metadata


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
# Wire format for CompressedArray (v2.3 sparse-on-wire)               #
# ------------------------------------------------------------------ #

# Header: magic(4s) kind(B) ndims(B) scale(d) numel(I) k(I)
# then shape (ndims * I), then indices (int32 * k, sparse kinds only),
# then values (fp16 or fp32). `scale` travels as float64 so both ranks
# dequantize with the bitwise-same factor — required for the parameter
# identity guarantee (each rank averages decompress(own)+decompress(remote)).
_WIRE_MAGIC = b"MFC1"
_WIRE_HEADER = "!4sBBdII"
_WIRE_HEADER_SIZE = struct.calcsize(_WIRE_HEADER)

_KIND_TOPK_FP32 = 1
_KIND_TOPK_FP16 = 2
_KIND_DENSE_FP16 = 3

_MAX_WIRE_NDIMS = 8


def pack_compressed(ca: CompressedArray) -> bytes:
    """Serialize a CompressedArray for the wire.

    Raises ValueError for shapes compress() never produces (dense fp32
    passthroughs are plain ndarrays and use the dense allreduce path).
    """
    if ca.topk_indices is not None:
        if ca.topk_values is None:
            raise ValueError("sparse CompressedArray missing values")
        kind = _KIND_TOPK_FP16 if ca.is_fp16 else _KIND_TOPK_FP32
        indices = np.ascontiguousarray(ca.topk_indices, dtype=np.int32)
        values = np.ascontiguousarray(
            ca.topk_values, dtype=np.float16 if ca.is_fp16 else np.float32,
        )
        k = indices.size
        body = indices.tobytes() + values.tobytes()
    elif ca.is_fp16 and ca.topk_values is not None:
        kind = _KIND_DENSE_FP16
        values = np.ascontiguousarray(ca.topk_values, dtype=np.float16)
        k = 0
        body = values.tobytes()
    else:
        raise ValueError(f"CompressedArray shape not packable: {ca!r}")

    shape = ca.original_shape
    if len(shape) > _MAX_WIRE_NDIMS:
        raise ValueError(f"too many dims for wire: {len(shape)}")
    numel = 1
    for s in shape:
        numel *= int(s)

    header = struct.pack(
        _WIRE_HEADER, _WIRE_MAGIC, kind, len(shape), float(ca.scale), numel, k,
    )
    shape_bytes = struct.pack(f"!{len(shape)}I", *shape)
    return header + shape_bytes + body


def unpack_compressed(data: bytes) -> CompressedArray:
    """Deserialize wire bytes back to a CompressedArray.

    SECURITY: fail-closed. Every count is validated against
    GRADIENT_MAX_NUMEL-style bounds BEFORE any allocation, and the body
    length must match the header exactly — a malicious peer can't
    trigger an allocation bomb or feed trailing garbage.

    Raises GradientValidationError on any structural problem.
    """
    if len(data) < _WIRE_HEADER_SIZE:
        raise GradientValidationError(
            f"compressed payload too short: {len(data)}B"
        )
    magic, kind, ndims, scale, numel, k = struct.unpack(
        _WIRE_HEADER, data[:_WIRE_HEADER_SIZE],
    )
    if magic != _WIRE_MAGIC:
        raise GradientValidationError(
            f"compressed payload bad magic: {magic!r} (peer compression "
            f"settings must match this node's)"
        )
    if kind not in (_KIND_TOPK_FP32, _KIND_TOPK_FP16, _KIND_DENSE_FP16):
        raise GradientValidationError(f"compressed payload unknown kind: {kind}")
    if not (0 < ndims <= _MAX_WIRE_NDIMS):
        raise GradientValidationError(f"compressed payload bad ndims: {ndims}")
    # Bounds-check numel and k before allocating anything.
    validate_gradient_metadata(numel, k)
    if not np.isfinite(scale) or scale <= 0:
        raise GradientValidationError(f"compressed payload bad scale: {scale}")

    offset = _WIRE_HEADER_SIZE
    shape_size = ndims * 4
    if len(data) < offset + shape_size:
        raise GradientValidationError("compressed payload truncated at shape")
    shape = struct.unpack(f"!{ndims}I", data[offset : offset + shape_size])
    offset += shape_size
    prod = 1
    for s in shape:
        prod *= s
    if prod != numel:
        raise GradientValidationError(
            f"compressed payload shape {shape} != numel {numel}"
        )

    if kind in (_KIND_TOPK_FP32, _KIND_TOPK_FP16):
        if k <= 0:
            raise GradientValidationError("sparse payload with k=0")
        value_itemsize = 2 if kind == _KIND_TOPK_FP16 else 4
        expected = k * 4 + k * value_itemsize
        body = data[offset:]
        if len(body) != expected:
            raise GradientValidationError(
                f"sparse payload body {len(body)}B != expected {expected}B"
            )
        indices = np.frombuffer(body[: k * 4], dtype=np.int32).copy()
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= numel
        ):
            raise GradientValidationError("sparse payload indices out of range")
        values = np.frombuffer(
            body[k * 4 :],
            dtype=np.float16 if kind == _KIND_TOPK_FP16 else np.float32,
        ).copy()
        return CompressedArray(
            data=b"",
            original_shape=tuple(shape),
            original_size=numel * 4,
            compressed_size=len(data),
            level=CompressionLevel.MODERATE,
            topk_k=k,
            topk_indices=indices,
            topk_values=values,
            is_fp16=(kind == _KIND_TOPK_FP16),
            scale=scale,
        )

    # Dense FP16
    expected = numel * 2
    body = data[offset:]
    if len(body) != expected:
        raise GradientValidationError(
            f"dense-fp16 payload body {len(body)}B != expected {expected}B"
        )
    values = np.frombuffer(body, dtype=np.float16).copy()
    return CompressedArray(
        data=b"",
        original_shape=tuple(shape),
        original_size=numel * 4,
        compressed_size=len(data),
        level=CompressionLevel.LIGHT,
        topk_values=values,
        is_fp16=True,
        scale=scale,
    )


def decompress_to_dense(ca: CompressedArray) -> np.ndarray:
    """Reconstruct the dense float32 array from a CompressedArray.

    Stateless module-level twin of AdaptiveCompressor.decompress — used
    for payloads received from a peer, where no compressor instance (or
    its error-feedback state) is involved.
    """
    if ca.topk_values is None:
        return np.zeros(ca.original_shape, dtype=np.float32)

    if ca.is_fp16:
        values = ca.topk_values.astype(np.float32) * ca.scale
    else:
        values = ca.topk_values.astype(np.float32)

    if ca.topk_indices is not None:
        numel = 1
        for s in ca.original_shape:
            numel *= s
        result = np.zeros(numel, dtype=np.float32)
        result[ca.topk_indices] = values
        return result.reshape(ca.original_shape)

    return values.reshape(ca.original_shape)


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
        import logging
        logger = logging.getLogger(__name__)
        flat = array.flatten().astype(np.float32)

        # Add residuals from previous round
        if self._residuals is not None and self._residuals.shape == flat.shape:
            flat = flat + self._residuals
        elif self._residuals is not None:
            logger.warning(
                "TopK residual shape changed (%s → %s); discarding residuals. "
                "Convergence guarantees from error feedback are broken until "
                "the gradient shape stabilizes.",
                self._residuals.shape, flat.shape,
            )
            self._residuals = None

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
                else ca.topk_values.astype(np.float32) * ca.scale
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
