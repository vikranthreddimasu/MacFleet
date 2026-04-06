"""Quantization utilities for gradient compression.

Ported from MacFleet v1. FP16 and INT8 quantization for reducing
communication bandwidth during distributed training.
"""

from typing import Optional

import torch


class FP16Quantizer:
    """FP16 quantization for gradient compression.

    Casts FP32 tensors to FP16 before transmission,
    providing 2x bandwidth reduction with minimal accuracy loss.
    """

    def __init__(self, scale_factor: Optional[float] = None):
        self.scale_factor = scale_factor
        self._dynamic_scale: Optional[float] = None

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Quantize tensor to FP16.

        Returns:
            Tuple of (quantized_tensor, scale_factor).
        """
        t = tensor.float()

        if self.scale_factor is not None:
            scale = self.scale_factor
        else:
            max_val = t.abs().max().item()
            if max_val > 0:
                scale = min(1.0, 60000.0 / max_val)
            else:
                scale = 1.0
            self._dynamic_scale = scale

        quantized = (t * scale).half()
        return quantized, scale

    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: float,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize FP16 tensor back to original precision."""
        return (tensor.float() / scale).to(target_dtype)

    @property
    def compression_ratio(self) -> float:
        return 0.5


class Int8Quantizer:
    """INT8 quantization for aggressive compression (4x from FP32)."""

    def __init__(self):
        self._scale: Optional[float] = None

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Quantize tensor to INT8 using symmetric quantization."""
        t = tensor.float()
        max_val = t.abs().max().item()
        scale = 127.0 / max_val if max_val > 0 else 1.0
        self._scale = scale
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
        return 0.25


def quantize_fp16(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Convenience function for FP16 quantization."""
    return FP16Quantizer().quantize(tensor)


def dequantize_fp16(
    tensor: torch.Tensor,
    scale: float,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience function for FP16 dequantization."""
    return FP16Quantizer().dequantize(tensor, scale, target_dtype)
