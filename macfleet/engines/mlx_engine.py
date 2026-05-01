"""Apple MLX training engine for MacFleet v2.

Implements the Engine protocol using Apple's MLX framework, which is
optimized for Apple Silicon with unified memory (no CPU<->GPU copies).

Key differences from PyTorch:
- Functional gradients: nn.value_and_grad() instead of loss.backward()
- Lazy evaluation: mx.eval() materializes computations
- Unified memory: arrays live in shared CPU/GPU memory

Gradients flow as numpy arrays through the framework-agnostic DataParallel
layer, while model computation stays in MLX for maximum performance.
"""

from __future__ import annotations

import io
import socket
from typing import Any, Callable, Optional

import numpy as np

from macfleet.engines.base import (
    EngineCapabilities,
    EngineType,
    HardwareProfile,
)

# MLX is optional — only imported when MLXEngine is instantiated
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim  # noqa: F401 — availability probe for callers

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _flatten_params(tree: dict) -> list[tuple[str, Any]]:
    """Flatten a nested parameter dict into (name, array) pairs.

    MLX models store parameters as nested dicts:
        {"layers": [{"weight": mx.array, "bias": mx.array}, ...]}
    This flattens to:
        [("layers.0.weight", mx.array), ("layers.0.bias", mx.array), ...]
    """
    result = []

    def _recurse(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in sorted(obj.items()):
                _recurse(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _recurse(f"{prefix}.{i}" if prefix else str(i), v)
        elif MLX_AVAILABLE and isinstance(obj, mx.array):
            result.append((prefix, obj))

    _recurse("", tree)
    return result


def _unflatten_params(flat: list[tuple[str, Any]], template: dict) -> dict:
    """Reconstruct nested parameter dict from flat (name, array) pairs.

    Uses the template structure to rebuild the nesting. The dict
    iteration order matches `_flatten_params` (sorted by key) so the
    template's structural order doesn't matter.
    """
    lookup = dict(flat)

    def _rebuild(prefix: str, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _rebuild(f"{prefix}.{k}" if prefix else k, v)
                for k, v in sorted(obj.items())
            }
        elif isinstance(obj, (list, tuple)):
            rebuilt = [
                _rebuild(f"{prefix}.{i}" if prefix else str(i), v)
                for i, v in enumerate(obj)
            ]
            return type(obj)(rebuilt)
        elif MLX_AVAILABLE and isinstance(obj, mx.array):
            key = prefix
            if key in lookup:
                return lookup[key]
            return obj
        return obj

    return _rebuild("", template)


class MLXEngine:
    """MLX training engine for Apple Silicon.

    Wraps an MLX model + optimizer and exposes the Engine protocol
    interface. Gradients are exported as numpy arrays so the
    communication layer stays framework-agnostic.

    MLX uses functional-style training:
    - nn.value_and_grad() computes loss + gradients in one call
    - optimizer.update() applies parameter updates
    - mx.eval() materializes lazy computations

    Usage:
        engine = MLXEngine()
        engine.load_model(model, optimizer, loss_fn=nn.losses.cross_entropy)
        loss = engine.forward(batch)
        engine.backward(loss)
        grads = engine.get_flat_gradients()   # numpy array
        engine.apply_flat_gradients(averaged) # from allreduce
        engine.step()
    """

    def __init__(self) -> None:
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not installed. Install with: pip install mlx"
            )
        self._model: Any = None
        self._optimizer: Any = None
        self._loss_fn: Optional[Callable] = None
        self._loss_and_grad_fn: Optional[Callable] = None
        self._grads: Optional[dict] = None
        self._last_batch: Optional[dict] = None
        self._last_loss: Optional[Any] = None
        self._param_names: list[str] = []  # ordered parameter names

    @property
    def model(self) -> Any:
        return self._model

    # ------------------------------------------------------------------ #
    # Engine protocol: core training methods                             #
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model: Any,
        optimizer: Any | None = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """Load MLX model and optimizer.

        Args:
            model: mlx.nn.Module instance.
            optimizer: mlx.optimizers optimizer (e.g., optim.Adam).
            loss_fn: Loss function taking (model, inputs, targets) -> mx.array.
                     If None, model must have a __call__ that returns loss.
        """
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        # Cache parameter names for consistent ordering across allreduce
        self._param_names = [name for name, _ in _flatten_params(model.parameters())]

        # Build value_and_grad function
        if loss_fn is not None:
            self._loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        else:
            # Assume model's __call__ returns loss when called with inputs
            def _model_loss(model: Any, *args: Any, **kwargs: Any) -> Any:
                return model(*args, **kwargs)

            self._loss_and_grad_fn = nn.value_and_grad(model, _model_loss)

        mx.eval(model.parameters())

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run forward pass. Stores batch for backward().

        In MLX, forward and backward are computed together via
        value_and_grad, but we split them to match the Engine protocol.
        Forward stores the batch; backward computes gradients.
        """
        self._last_batch = batch

        # Run forward-only for loss value (no grad computation)
        if self._loss_fn is not None:
            if isinstance(batch, dict):
                loss = self._loss_fn(self._model, **batch)
            elif isinstance(batch, (tuple, list)):
                loss = self._loss_fn(self._model, *batch)
            else:
                loss = self._loss_fn(self._model, batch)
        else:
            if isinstance(batch, dict):
                loss = self._model(**batch)
            elif isinstance(batch, (tuple, list)):
                loss = self._model(*batch)
            else:
                loss = self._model(batch)

        mx.eval(loss)
        self._last_loss = loss
        return loss

    def backward(self, loss: Any) -> None:
        """Compute gradients using MLX's value_and_grad.

        Uses the batch stored by the last forward() call.
        MLX computes forward+backward together, so this re-evaluates
        the forward pass to get gradients.
        """
        batch = self._last_batch
        if batch is None:
            raise RuntimeError("backward() called without prior forward()")

        if isinstance(batch, dict):
            _, self._grads = self._loss_and_grad_fn(self._model, **batch)
        elif isinstance(batch, (tuple, list)):
            _, self._grads = self._loss_and_grad_fn(self._model, *batch)
        else:
            _, self._grads = self._loss_and_grad_fn(self._model, batch)

        mx.eval(self._grads)

    def step(self) -> None:
        """Apply optimizer update using computed gradients."""
        if self._optimizer is None:
            raise RuntimeError("No optimizer loaded")
        if self._grads is None:
            raise RuntimeError("step() called without backward()")

        self._optimizer.update(self._model, self._grads)
        mx.eval(self._model.parameters(), self._optimizer.state)

    def zero_grad(self) -> None:
        """Clear stored gradients.

        In MLX, gradients are returned fresh each time from value_and_grad,
        so this just clears our stored copy.
        """
        self._grads = None

    # ------------------------------------------------------------------ #
    # Flat gradient interface (used by DataParallel)                     #
    # ------------------------------------------------------------------ #

    def get_flat_gradients(self) -> np.ndarray:
        """Flatten all gradients into a single numpy array for allreduce.

        Returns:
            1D float32 numpy array of all gradients concatenated.
        """
        if self._grads is None:
            return np.array([], dtype=np.float32)

        flat_grads = _flatten_params(self._grads)
        arrays = []
        for _, grad_array in flat_grads:
            arrays.append(np.array(grad_array, dtype=np.float32).flatten())

        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays)

    def apply_flat_gradients(self, flat_grads: np.ndarray) -> None:
        """Write averaged flat gradients back to stored gradient dict.

        Reconstructs the gradient tree structure from the flat array
        so optimizer.update() works correctly.
        """
        if self._grads is None:
            return

        flat_params = _flatten_params(self._grads)
        offset = 0
        new_flat = []

        for name, grad_array in flat_params:
            numel = grad_array.size
            data = flat_grads[offset:offset + numel].reshape(grad_array.shape)
            new_flat.append((name, mx.array(data)))
            offset += numel

        self._grads = _unflatten_params(new_flat, self._grads)

    def get_flat_parameters(self) -> np.ndarray:
        """Flatten all parameters into a single numpy array (for broadcast)."""
        flat_params = _flatten_params(self._model.parameters())
        arrays = []
        for _, param_array in flat_params:
            arrays.append(np.array(param_array, dtype=np.float32).flatten())

        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays)

    def apply_flat_parameters(self, flat_params: np.ndarray) -> None:
        """Write flat parameters back to model (after broadcast)."""
        params = self._model.parameters()
        flat = _flatten_params(params)
        offset = 0
        new_flat = []

        for name, param_array in flat:
            numel = param_array.size
            data = flat_params[offset:offset + numel].reshape(param_array.shape)
            new_flat.append((name, mx.array(data)))
            offset += numel

        new_params = _unflatten_params(new_flat, params)
        self._model.update(new_params)
        mx.eval(self._model.parameters())

    # ------------------------------------------------------------------ #
    # Engine protocol: state serialization                               #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> bytes:
        """Serialize model state for checkpointing.

        Uses numpy serialization for cross-framework compatibility.
        """
        params = self._model.parameters()
        flat = _flatten_params(params)

        state = {}
        for name, arr in flat:
            state[name] = np.array(arr)

        buffer = io.BytesIO()
        np.savez(buffer, **state)
        return buffer.getvalue()

    def load_state_dict(self, data: bytes) -> None:
        """Load model state from checkpoint bytes."""
        buffer = io.BytesIO(data)
        state = np.load(buffer, allow_pickle=False)

        params = self._model.parameters()
        flat = _flatten_params(params)
        new_flat = []

        for name, arr in flat:
            if name in state:
                new_flat.append((name, mx.array(state[name])))
            else:
                new_flat.append((name, arr))

        new_params = _unflatten_params(new_flat, params)
        self._model.update(new_params)
        mx.eval(self._model.parameters())

    # ------------------------------------------------------------------ #
    # Engine protocol: introspection                                     #
    # ------------------------------------------------------------------ #

    def profile(self) -> HardwareProfile:
        """Profile local hardware.

        Delegates to macfleet.pool.agent.profile_hardware() so the engine
        returns the same gpu_cores / ram_gb / chip_name fields the
        PoolAgent would advertise. Falls back to a zero profile if the
        macOS detection helpers fail (e.g. on Linux during framework-
        agnostic CI).
        """
        try:
            from macfleet.pool.agent import profile_hardware
            hw = profile_hardware()
            hw.node_id = f"{hw.hostname}-mlx"
            hw.mlx_available = True
            return hw
        except Exception:
            return HardwareProfile(
                hostname=socket.gethostname(),
                node_id=f"{socket.gethostname()}-mlx",
                gpu_cores=0,
                ram_gb=0.0,
                memory_bandwidth_gbps=0.0,
                has_ane=True,
                chip_name="Unknown",
                mlx_available=True,
            )

    def param_count(self) -> int:
        """Total trainable parameters."""
        if self._model is None:
            return 0
        flat = _flatten_params(self._model.parameters())
        return sum(arr.size for _, arr in flat)

    def memory_usage_gb(self) -> float:
        """Current memory usage in GB (params + gradients)."""
        if self._model is None:
            return 0.0

        total = 0
        for _, arr in _flatten_params(self._model.parameters()):
            total += arr.size * arr.dtype.size

        if self._grads is not None:
            for _, arr in _flatten_params(self._grads):
                total += arr.size * arr.dtype.size

        return total / (1024**3)

    def estimated_model_memory_gb(self) -> float:
        """Estimated total memory needed (params + grads + optimizer)."""
        if self._model is None:
            return 0.0
        param_bytes = sum(
            arr.size * arr.dtype.size
            for _, arr in _flatten_params(self._model.parameters())
        )
        # MLX unified memory is more efficient; ~3x for Adam optimizer
        return param_bytes * 3.0 / (1024**3)

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_type=EngineType.MLX,
            supports_mps=False,
            supports_ane=True,
            supported_dtypes=["float32", "float16", "bfloat16"],
        )
