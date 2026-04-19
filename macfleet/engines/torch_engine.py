"""PyTorch + MPS training engine for MacFleet v2.

Implements the Engine protocol using PyTorch with Metal Performance Shaders
(MPS) acceleration on Apple Silicon. Falls back to CPU if MPS is unavailable.

Gradients flow as numpy arrays through the framework-agnostic DataParallel
layer, while model computation stays in PyTorch/MPS for maximum performance.
"""

from __future__ import annotations

import io
import socket
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from macfleet.engines.base import (
    EngineCapabilities,
    EngineType,
    HardwareProfile,
)


def _detect_best_device(preference: str = "auto") -> torch.device:
    """Select the best available device."""
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class TorchEngine:
    """PyTorch training engine for Apple Silicon.

    Wraps a PyTorch model + optimizer and exposes the Engine protocol
    interface. Gradients are exported as numpy arrays so the
    communication layer stays framework-agnostic.

    Usage:
        engine = TorchEngine()
        engine.load_model(model, optimizer)
        loss = engine.forward(batch)
        engine.backward(loss)
        grads = engine.get_flat_gradients()   # numpy array
        engine.apply_flat_gradients(averaged) # from allreduce
        engine.step()
    """

    def __init__(self, device: str = "auto"):
        self._device = _detect_best_device(device)
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._trainable_params: list[nn.Parameter] = []

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    # ------------------------------------------------------------------ #
    # Engine protocol: core training methods                             #
    # ------------------------------------------------------------------ #

    def load_model(self, model: Any, optimizer: Any | None = None) -> None:
        """Load model and optimizer onto the target device."""
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._trainable_params = [p for p in self._model.parameters() if p.requires_grad]

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run forward pass. Accepts dict or positional args."""
        if isinstance(batch, dict):
            # Move tensors to device
            device_batch = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            return self._model(**device_batch)
        return self._model(batch)

    def backward(self, loss: Any) -> None:
        """Run backward pass."""
        loss.backward()

    def step(self) -> None:
        """Optimizer step."""
        if self._optimizer:
            self._optimizer.step()

    def zero_grad(self) -> None:
        """Zero all gradients."""
        if self._model:
            self._model.zero_grad()

    # ------------------------------------------------------------------ #
    # Flat gradient interface (used by DataParallel)                     #
    # ------------------------------------------------------------------ #

    def get_flat_gradients(self) -> np.ndarray:
        """Flatten all gradients into a single numpy array for allreduce.

        Returns:
            1D float32 numpy array of all gradients concatenated.
        """
        grads = []
        for param in self._trainable_params:
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().float().numpy().flatten())
            else:
                grads.append(np.zeros(param.numel(), dtype=np.float32))
        if not grads:
            return np.array([], dtype=np.float32)
        return np.concatenate(grads)

    def apply_flat_gradients(self, flat_grads: np.ndarray) -> None:
        """Write averaged flat gradients back to model parameters."""
        offset = 0
        for param in self._trainable_params:
            numel = param.numel()
            grad_data = flat_grads[offset : offset + numel].reshape(param.shape)
            param.grad = torch.from_numpy(grad_data.copy()).to(self._device)
            offset += numel

    def get_flat_parameters(self) -> np.ndarray:
        """Flatten all parameters into a single numpy array (for broadcast)."""
        params = []
        for param in self._trainable_params:
            params.append(param.detach().cpu().float().numpy().flatten())
        if not params:
            return np.array([], dtype=np.float32)
        return np.concatenate(params)

    def apply_flat_parameters(self, flat_params: np.ndarray) -> None:
        """Write flat parameters back to model (after broadcast)."""
        offset = 0
        for param in self._trainable_params:
            numel = param.numel()
            data = flat_params[offset : offset + numel].reshape(param.shape)
            param.data.copy_(torch.from_numpy(data.copy()).to(self._device))
            offset += numel

    # ------------------------------------------------------------------ #
    # Engine protocol: state serialization                               #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> bytes:
        """Serialize model + optimizer state for checkpointing."""
        buffer = io.BytesIO()
        state = {"model": self._model.state_dict()}
        if self._optimizer:
            state["optimizer"] = self._optimizer.state_dict()
        torch.save(state, buffer)
        return buffer.getvalue()

    def load_state_dict(self, data: bytes) -> None:
        """Load model + optimizer from checkpoint bytes."""
        buffer = io.BytesIO(data)
        state = torch.load(buffer, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state["model"])
        if self._optimizer and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])

    # ------------------------------------------------------------------ #
    # Engine protocol: introspection                                     #
    # ------------------------------------------------------------------ #

    def profile(self) -> HardwareProfile:
        """Profile local hardware."""
        return HardwareProfile(
            hostname=socket.gethostname(),
            node_id=f"{socket.gethostname()}-torch",
            gpu_cores=0,
            ram_gb=0.0,
            memory_bandwidth_gbps=0.0,
            has_ane=True,
            chip_name="Unknown",
            mps_available=torch.backends.mps.is_available(),
        )

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self._trainable_params) if self._model else 0

    def memory_usage_gb(self) -> float:
        """Current memory usage in GB (params + gradients)."""
        if not self._model:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self._model.parameters())
        for p in self._model.parameters():
            if p.grad is not None:
                total += p.grad.numel() * p.grad.element_size()
        return total / (1024**3)

    def estimated_model_memory_gb(self) -> float:
        """Estimated total memory needed (params + grads + optimizer + activations)."""
        if not self._model:
            return 0.0
        param_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())
        return param_bytes * 4.5 / (1024**3)  # ~4.5x for Adam optimizer

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_type=EngineType.TORCH,
            supports_mps=torch.backends.mps.is_available(),
            supports_gradient_checkpointing=True,
            supported_dtypes=["float32", "float16"],
        )
