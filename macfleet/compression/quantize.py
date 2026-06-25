"""Quantization utilities for gradient compression.

Implements FP16 and other quantization methods for reducing
communication bandwidth during distributed training.
"""

from typing import Optional

import torch


class FP16Quantizer:
    """FP16 quantization for gradient compression.

    Casts FP32 tensors to FP16 before transmission,
    providing 2x bandwidth reduction with minimal accuracy loss.

    Note: This class is NOT thread-safe. The internal scale state
    must only be accessed from a single thread/coroutine (the training loop).
    """

    def __init__(self, scale_factor: Optional[float] = None):
        """Initialize the quantizer.

        Args:
            scale_factor: Optional scaling factor to prevent overflow.
                         If None, uses dynamic scaling based on max value.
        """
        self.scale_factor = scale_factor
        self._dynamic_scale: Optional[float] = None

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Quantize tensor to FP16.

        Args:
            tensor: Input tensor (any dtype).

        Returns:
            Tuple of (quantized_tensor, scale_factor).
        """
        # Convert to float32 first for consistent handling
        t = tensor.float()

        # Compute scale factor if dynamic
        if self.scale_factor is not None:
            scale = self.scale_factor
        else:
            # Dynamic scaling to prevent FP16 overflow
            max_val = t.abs().max().item()
            # FP16 max is ~65504, leave headroom
            if max_val > 0:
                scale = min(1.0, 60000.0 / max_val)
            else:
                scale = 1.0
            self._dynamic_scale = scale

        # Scale and convert to FP16
        quantized = (t * scale).half()

        return quantized, scale

    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: float,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize FP16 tensor back to original precision.

        Args:
            tensor: FP16 tensor.
            scale: Scale factor used during quantization.
            target_dtype: Target dtype for output.

        Returns:
            Dequantized tensor.
        """
        return (tensor.float() / scale).to(target_dtype)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (2x for FP32 -> FP16)."""
        return 0.5


class DynamicFP16Quantizer(FP16Quantizer):
    """FP16 quantizer with per-tensor dynamic scaling."""

    def __init__(self):
        super().__init__(scale_factor=None)


class Int8Quantizer:
    """INT8 quantization for aggressive compression.

    Provides 4x compression from FP32 but with more accuracy loss.
    Uses symmetric quantization with scaling.
    """

    def __init__(self):
        self._scale: Optional[float] = None

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Quantize tensor to INT8.

        Args:
            tensor: Input tensor.

        Returns:
            Tuple of (quantized_tensor, scale_factor).
        """
        t = tensor.float()

        # Compute scale for symmetric quantization
        max_val = t.abs().max().item()
        if max_val > 0:
            scale = 127.0 / max_val
        else:
            scale = 1.0

        self._scale = scale

        # Quantize
        quantized = (t * scale).round().clamp(-128, 127).to(torch.int8)

        return quantized, scale

    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: float,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize INT8 tensor."""
        return (tensor.float() / scale).to(target_dtype)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (4x for FP32 -> INT8)."""
        return 0.25


def quantize_fp16(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Convenience function for FP16 quantization.

    Args:
        tensor: Input tensor.

    Returns:
        Tuple of (fp16_tensor, scale).
    """
    quantizer = FP16Quantizer()
    return quantizer.quantize(tensor)


def dequantize_fp16(
    tensor: torch.Tensor,
    scale: float,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience function for FP16 dequantization.

    Args:
        tensor: FP16 tensor.
        scale: Scale factor.
        target_dtype: Target dtype.

    Returns:
        Dequantized tensor.
    """
    quantizer = FP16Quantizer()
    return quantizer.dequantize(tensor, scale, target_dtype)


class QuantizedTensor:
    """Container for quantized tensor with metadata.

    Stores the quantized data along with information needed
    for dequantization.
    """

    def __init__(
        self,
        data: torch.Tensor,
        scale: float,
        original_dtype: torch.dtype,
        original_shape: tuple,
        quantization_type: str = "fp16",
    ):
        """Initialize quantized tensor container.

        Args:
            data: Quantized tensor data.
            scale: Scale factor for dequantization.
            original_dtype: Original tensor dtype.
            original_shape: Original tensor shape.
            quantization_type: Type of quantization ("fp16", "int8").
        """
        self.data = data
        self.scale = scale
        self.original_dtype = original_dtype
        self.original_shape = original_shape
        self.quantization_type = quantization_type

    def dequantize(self) -> torch.Tensor:
        """Dequantize and restore original tensor.

        Returns:
            Dequantized tensor with original dtype and shape.
        """
        if self.quantization_type == "fp16":
            result = (self.data.float() / self.scale).to(self.original_dtype)
        elif self.quantization_type == "int8":
            result = (self.data.float() / self.scale).to(self.original_dtype)
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")

        return result.view(self.original_shape)

    @property
    def numel(self) -> int:
        """Number of elements."""
        return self.data.numel()

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.data.numel() * self.data.element_size()


def auto_quantize(
    tensor: torch.Tensor,
    method: str = "fp16",
) -> QuantizedTensor:
    """Automatically quantize a tensor.

    Args:
        tensor: Input tensor.
        method: Quantization method ("fp16", "int8").

    Returns:
        QuantizedTensor container.
    """
    original_dtype = tensor.dtype
    original_shape = tensor.shape

    if method == "fp16":
        quantizer = FP16Quantizer()
    elif method == "int8":
        quantizer = Int8Quantizer()
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    data, scale = quantizer.quantize(tensor.flatten())

    return QuantizedTensor(
        data=data,
        scale=scale,
        original_dtype=original_dtype,
        original_shape=original_shape,
        quantization_type=method,
    )
