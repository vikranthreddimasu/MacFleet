"""Shared test fixtures for MacFleet v2.

Torch fixtures are only registered when torch is installed. Framework-agnostic
CI jobs install only the core deps (no torch, no mlx) and must be able to
collect tests from the directories they don't ignore; if conftest fails to
import, pytest aborts the entire run before `--ignore` can take effect.
"""

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    @pytest.fixture
    def small_tensor():
        """Small tensor for quick tests."""
        return torch.randn(1000)

    @pytest.fixture
    def medium_tensor():
        """Medium tensor for compression tests."""
        return torch.randn(10000)

    @pytest.fixture
    def large_tensor():
        """Large tensor simulating real gradients (~10MB)."""
        return torch.randn(2500000)

    @pytest.fixture
    def gradient_like_tensor():
        """Tensor with gradient-like distribution (mostly small, few large)."""
        t = torch.randn(10000) * 0.01  # Mostly small
        t[:100] = torch.randn(100) * 10.0  # Few large
        return t
