"""Top-K gradient sparsification with error feedback.

Implements the algorithm from Section 10.1 of DESIGN.md:
1. Accumulate gradient with residual from previous step
2. Select top-K values by magnitude
3. Store residual (values not sent) for next step
4. This preserves convergence guarantees via error feedback
"""

from typing import Optional

import torch


class TopKCompressor:
    """Top-K gradient compressor with error feedback.

    Keeps only the top K% of gradient values by magnitude,
    accumulating the rest as residuals for future steps.
    This provides ~10x compression while preserving convergence.

    Note: This class is NOT thread-safe. The internal residual state
    must only be accessed from a single thread/coroutine (the training loop).
    """

    def __init__(
        self,
        ratio: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize the compressor.

        Args:
            ratio: Fraction of values to keep (0.1 = top 10%).
            device: Device for residual storage.
        """
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

        Args:
            tensor: Gradient tensor to compress.
            name: Optional name for tracking residuals per-parameter.

        Returns:
            Tuple of (indices, values, original_numel, original_dtype).
        """
        original_dtype = tensor.dtype
        original_numel = tensor.numel()

        # Flatten for processing
        flat = tensor.flatten().float()

        # Add residual from previous step
        residual_key = name or str(id(tensor))
        if residual_key in self._residuals:
            residual = self._residuals[residual_key]
            if residual.numel() == flat.numel():
                flat = flat + residual.to(flat.device)

        # Compute number of elements to keep
        k = max(1, int(original_numel * self.ratio))

        # Get top-k by magnitude
        abs_values = flat.abs()
        _, indices = torch.topk(abs_values, k, sorted=False)
        values = flat[indices]

        # Update residual: keep values that weren't selected
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
        """Decompress a sparse gradient back to dense.

        Args:
            indices: Indices of non-zero values.
            values: Non-zero values.
            original_numel: Original number of elements.
            original_dtype: Original dtype.
            original_shape: Optional shape to reshape to.

        Returns:
            Dense gradient tensor.
        """
        # Create zero tensor
        dense = torch.zeros(original_numel, dtype=torch.float32, device=values.device)

        # Scatter values to indices
        dense.scatter_(0, indices.long(), values.float())

        # Convert to original dtype
        dense = dense.to(original_dtype)

        # Reshape if shape provided
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
        """Theoretical compression ratio."""
        # Compressed format: k indices (int32) + k values (float16 or float32)
        # Original: N values (float32)
        # Ratio = (k * 4 + k * 2) / (N * 4) with FP16 values
        # Or (k * 4 + k * 4) / (N * 4) with FP32 values
        # Simplified: approximately 1.5 * ratio for FP16
        return 1.5 * self.ratio


def topk_compress(
    tensor: torch.Tensor,
    ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
    """Stateless Top-K compression (no error feedback).

    For one-shot compression without residual tracking.

    Args:
        tensor: Tensor to compress.
        ratio: Fraction to keep.

    Returns:
        Tuple of (indices, values, numel, dtype).
    """
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
    """Stateless Top-K decompression.

    Args:
        indices: Sparse indices.
        values: Sparse values.
        original_numel: Original tensor size.
        original_dtype: Original dtype.

    Returns:
        Dense tensor.
    """
    dense = torch.zeros(original_numel, dtype=torch.float32, device=values.device)
    dense.scatter_(0, indices.long(), values.float())
    return dense.to(original_dtype)


class RandomKCompressor:
    """Random-K compressor for comparison/debugging.

    Randomly selects K values instead of top-K.
    Useful for ablation studies.
    """

    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio

    def compress(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
        """Compress using random selection."""
        original_dtype = tensor.dtype
        original_numel = tensor.numel()

        flat = tensor.flatten().float()
        k = max(1, int(original_numel * self.ratio))

        # Random indices
        indices = torch.randperm(original_numel)[:k]
        values = flat[indices]

        return indices.to(torch.int32), values, original_numel, original_dtype

    def decompress(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        original_numel: int,
        original_dtype: torch.dtype,
        original_shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Decompress random-K."""
        dense = torch.zeros(original_numel, dtype=torch.float32, device=values.device)
        dense.scatter_(0, indices.long(), values.float())
        result = dense.to(original_dtype)
        if original_shape:
            result = result.view(original_shape)
        return result
