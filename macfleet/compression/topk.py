"""Top-K gradient sparsification with error feedback.

Ported from MacFleet v1. Algorithm:
1. Accumulate gradient with residual from previous step
2. Select top-K values by magnitude
3. Store residual (values not sent) for next step
4. Error feedback preserves convergence guarantees
"""

from typing import Optional

import torch


class TopKCompressor:
    """Top-K gradient compressor with error feedback.

    Keeps only the top K% of gradient values by magnitude,
    accumulating the rest as residuals for future steps.
    Provides ~10x compression while preserving convergence.
    """

    def __init__(self, ratio: float = 0.1, device: str = "cpu"):
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.ratio = ratio
        self.device = device
        self._residuals: dict[str, torch.Tensor] = {}

    def compress(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
        """Compress a gradient tensor using Top-K sparsification.

        Returns:
            Tuple of (indices, values, original_numel, original_dtype).
        """
        original_dtype = tensor.dtype
        original_numel = tensor.numel()

        flat = tensor.flatten().float()

        # Add residual from previous step (error feedback)
        residual_key = name or str(id(tensor))
        if residual_key in self._residuals:
            residual = self._residuals[residual_key]
            if residual.numel() == flat.numel():
                flat = flat + residual.to(flat.device)

        k = max(1, int(original_numel * self.ratio))

        abs_values = flat.abs()
        _, indices = torch.topk(abs_values, k, sorted=False)
        values = flat[indices]

        # Store residual: values NOT selected
        new_residual = flat.clone()
        new_residual[indices] = 0
        self._residuals[residual_key] = new_residual.to(self.device)

        return indices.to(torch.int32), values, original_numel, original_dtype

    def decompress(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        original_numel: int,
        original_dtype: torch.dtype,
        original_shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Decompress a sparse gradient back to dense."""
        dense = torch.zeros(original_numel, dtype=torch.float32, device=values.device)
        dense.scatter_(0, indices.long(), values.float())
        dense = dense.to(original_dtype)
        if original_shape is not None:
            dense = dense.view(original_shape)
        return dense

    def reset_residuals(self) -> None:
        """Clear all stored residuals."""
        self._residuals.clear()

    def get_residual(self, name: str) -> Optional[torch.Tensor]:
        """Get residual for a specific parameter."""
        return self._residuals.get(name)

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio (compressed/original)."""
        return 1.5 * self.ratio


def topk_compress(
    tensor: torch.Tensor,
    ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
    """Stateless Top-K compression (no error feedback)."""
    original_dtype = tensor.dtype
    original_numel = tensor.numel()
    flat = tensor.flatten().float()
    k = max(1, int(original_numel * ratio))
    abs_values = flat.abs()
    _, indices = torch.topk(abs_values, k, sorted=False)
    values = flat[indices]
    return indices.to(torch.int32), values, original_numel, original_dtype


def topk_decompress(
    indices: torch.Tensor,
    values: torch.Tensor,
    original_numel: int,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """Stateless Top-K decompression."""
    dense = torch.zeros(original_numel, dtype=torch.float32, device=values.device)
    dense.scatter_(0, indices.long(), values.float())
    return dense.to(original_dtype)
