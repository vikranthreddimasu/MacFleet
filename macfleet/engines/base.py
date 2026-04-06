"""Engine protocol: the critical interface decoupling pool from ML frameworks.

Both TorchEngine and MLXEngine implement this protocol. The pool/comm layer
never imports torch or mlx — gradients flow as dict[str, bytes].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class EngineType(Enum):
    """Supported training engine types."""
    TORCH = "torch"
    MLX = "mlx"


class ThermalPressure(Enum):
    """Thermal pressure levels from macOS."""
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"

    @property
    def workload_multiplier(self) -> float:
        """Suggested workload multiplier based on thermal pressure."""
        return {
            ThermalPressure.NOMINAL: 1.0,
            ThermalPressure.FAIR: 0.9,
            ThermalPressure.SERIOUS: 0.7,
            ThermalPressure.CRITICAL: 0.3,
        }[self]


@dataclass
class HardwareProfile:
    """Hardware capabilities of a single Mac node."""
    hostname: str
    node_id: str  # unique identifier (hostname + random suffix)
    gpu_cores: int
    ram_gb: float
    memory_bandwidth_gbps: float
    has_ane: bool  # Apple Neural Engine
    chip_name: str  # e.g., "Apple M4 Pro"
    thermal_pressure: ThermalPressure = ThermalPressure.NOMINAL
    mps_available: bool = False
    mlx_available: bool = False

    @property
    def compute_score(self) -> float:
        """Composite score for coordinator election and scheduling.

        Higher is better. Factors in GPU cores, memory bandwidth, and RAM.
        """
        return (self.gpu_cores * 10.0) + (self.memory_bandwidth_gbps * 2.0) + self.ram_gb

    def can_fit_model(self, model_memory_gb: float, headroom: float = 0.3) -> bool:
        """Check if this node can hold a model with headroom.

        Args:
            model_memory_gb: Estimated model memory (params + grads + optimizer + activations).
            headroom: Fraction of extra memory to reserve (default 30%).
        """
        usable_ram = self.ram_gb - 4.0  # Reserve ~4GB for macOS + apps
        return usable_ram >= model_memory_gb * (1.0 + headroom)


@dataclass
class TrainingMetrics:
    """Per-step training metrics reported by the engine."""
    loss: float = 0.0
    throughput_samples_sec: float = 0.0
    memory_used_gb: float = 0.0
    step_time_sec: float = 0.0
    grad_norm: float = 0.0


@runtime_checkable
class Engine(Protocol):
    """Training engine protocol.

    Both TorchEngine and MLXEngine implement this. The pool and communication
    layers work exclusively through this interface — they never import torch or mlx.

    Gradients flow as dict[str, bytes] (parameter name -> serialized tensor bytes)
    so the pool layer is completely framework-agnostic.
    """

    def load_model(self, model: Any, optimizer: Any | None = None) -> None:
        """Load a model and optional optimizer into the engine.

        Args:
            model: Framework-specific model (torch.nn.Module or mlx equivalent).
            optimizer: Framework-specific optimizer.
        """
        ...

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run forward pass on a batch.

        Args:
            batch: Dict of input tensors (framework-specific).

        Returns:
            Loss value (framework-specific tensor).
        """
        ...

    def backward(self, loss: Any) -> None:
        """Run backward pass to compute gradients.

        Args:
            loss: Loss value from forward().
        """
        ...

    def get_gradients(self) -> dict[str, bytes]:
        """Serialize current gradients to bytes.

        Returns:
            Dict mapping parameter name to serialized gradient bytes.
            Format is engine-specific (numpy for torch, mlx.save for mlx).
        """
        ...

    def apply_gradients(self, averaged_grads: dict[str, bytes]) -> None:
        """Deserialize and apply averaged gradients from AllReduce.

        Args:
            averaged_grads: Dict mapping parameter name to serialized gradient bytes.
        """
        ...

    def step(self) -> None:
        """Run optimizer step (update model parameters)."""
        ...

    def zero_grad(self) -> None:
        """Zero out all gradients."""
        ...

    def state_dict(self) -> bytes:
        """Serialize full model + optimizer state for checkpointing.

        Returns:
            Serialized state as bytes.
        """
        ...

    def load_state_dict(self, data: bytes) -> None:
        """Load model + optimizer state from checkpoint.

        Args:
            data: Serialized state bytes from state_dict().
        """
        ...

    def profile(self) -> HardwareProfile:
        """Profile the hardware this engine is running on.

        Returns:
            HardwareProfile describing compute capabilities.
        """
        ...

    def param_count(self) -> int:
        """Total number of trainable parameters.

        Returns:
            Parameter count.
        """
        ...

    def memory_usage_gb(self) -> float:
        """Current memory usage in GB.

        Returns:
            Memory used by model, gradients, and optimizer.
        """
        ...

    def estimated_model_memory_gb(self) -> float:
        """Estimated total memory for model (params + grads + optimizer + activations).

        Used for pre-flight memory checks before distributed training.

        Returns:
            Estimated memory in GB.
        """
        ...


@dataclass
class EngineCapabilities:
    """What an engine implementation supports."""
    engine_type: EngineType
    supports_mps: bool = False
    supports_ane: bool = False
    supports_gradient_checkpointing: bool = False
    supported_dtypes: list[str] = field(default_factory=lambda: ["float32", "float16"])
