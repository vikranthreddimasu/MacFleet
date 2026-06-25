"""Configuration dataclasses for MacFleet distributed training."""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from macfleet.utils.network import validate_host, validate_ip_address, validate_port


class NodeRole(Enum):
    """Role of a node in the cluster."""
    MASTER = "master"
    WORKER = "worker"


class CompressionType(Enum):
    """Gradient compression type."""
    NONE = "none"
    TOPK = "topk"
    FP16 = "fp16"
    TOPK_FP16 = "topk_fp16"


class ThermalState(Enum):
    """Thermal state reported by macOS."""
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"


# Default configuration values
DEFAULT_GRPC_PORT = 50051
DEFAULT_TENSOR_PORT = 50052
DEFAULT_TOPK_RATIO = 0.1
DEFAULT_HEARTBEAT_INTERVAL_SEC = 2.0
DEFAULT_HEARTBEAT_TIMEOUT_SEC = 6.0
DEFAULT_CHECKPOINT_EVERY = 2
DEFAULT_AUTH_TOKEN_ENV = "MACFLEET_AUTH_TOKEN"
MIN_AUTH_TOKEN_LENGTH = 16
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def resolve_auth_token(
    auth_token: Optional[str] = None,
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
) -> Optional[str]:
    """Resolve and validate the optional control-plane auth token.

    Tokens are intentionally omitted from serialized config dictionaries.
    """
    if auth_token is None:
        if not auth_token_env:
            return None
        if not _ENV_NAME_RE.match(auth_token_env):
            raise ValueError(
                "auth_token_env must be a valid environment variable name"
            )
        auth_token = os.environ.get(auth_token_env)

    if auth_token is None or auth_token == "":
        return None

    token = str(auth_token)
    if any(ch in token for ch in "\r\n\0"):
        raise ValueError("auth_token must not contain control characters")
    if len(token) < MIN_AUTH_TOKEN_LENGTH:
        raise ValueError(
            f"auth_token must be at least {MIN_AUTH_TOKEN_LENGTH} characters"
        )
    return token


@dataclass
class NodeConfig:
    """Configuration for a single node in the cluster.

    Attributes:
        hostname: Node hostname for identification.
        ip_address: IP address for network communication.
        gpu_cores: Number of GPU cores (MPS).
        ram_gb: Total RAM in gigabytes.
        memory_bandwidth_gbps: Memory bandwidth in GB/s.
        tensor_port: Port for raw tensor transfers.
        rank: Assigned rank in the cluster (set by coordinator).
        workload_weight: Proportion of batch assigned (0.0-1.0, set by coordinator).
    """
    hostname: str
    ip_address: str
    gpu_cores: int
    ram_gb: int
    memory_bandwidth_gbps: float
    tensor_port: int = DEFAULT_TENSOR_PORT
    rank: int = -1
    workload_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.hostname or not self.hostname.strip():
            raise ValueError("hostname must not be empty")
        if any(ch in self.hostname for ch in "\r\n\t"):
            raise ValueError("hostname must not contain control characters")
        self.ip_address = validate_ip_address(self.ip_address, "ip_address")
        if self.gpu_cores < 0:
            raise ValueError(f"gpu_cores must be non-negative, got {self.gpu_cores}")
        if self.ram_gb < 0:
            raise ValueError(f"ram_gb must be non-negative, got {self.ram_gb}")
        if self.memory_bandwidth_gbps < 0:
            raise ValueError(
                "memory_bandwidth_gbps must be non-negative, "
                f"got {self.memory_bandwidth_gbps}"
            )
        self.tensor_port = validate_port(self.tensor_port, "tensor_port")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "gpu_cores": self.gpu_cores,
            "ram_gb": self.ram_gb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "tensor_port": self.tensor_port,
            "rank": self.rank,
            "workload_weight": self.workload_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeConfig":
        """Create from dictionary, ignoring unknown keys."""
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ClusterConfig:
    """Configuration for the cluster topology and communication.

    Attributes:
        role: Role of this node (master or worker).
        master_addr: IP address of the master node.
        master_port: gRPC port of the master node.
        tensor_port: Port for raw tensor transfers.
        heartbeat_interval_sec: Interval between heartbeats.
        heartbeat_timeout_sec: Timeout before marking node as dead.
        discovery_enabled: Whether to use Bonjour/zeroconf discovery.
        service_name: Bonjour service name for discovery.
    """
    role: NodeRole
    master_addr: str = ""
    master_port: int = DEFAULT_GRPC_PORT
    tensor_port: int = DEFAULT_TENSOR_PORT
    heartbeat_interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC
    heartbeat_timeout_sec: float = DEFAULT_HEARTBEAT_TIMEOUT_SEC
    discovery_enabled: bool = True
    service_name: str = "_macfleet._tcp.local."
    host: Optional[str] = None
    min_workers: int = 1  # Minimum workers required before training starts
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV
    auth_token: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.master_addr:
            self.master_addr = validate_host(self.master_addr, "master_addr")
        if self.host:
            self.host = validate_ip_address(self.host, "host")
        self.master_port = validate_port(self.master_port, "master_port")
        self.tensor_port = validate_port(self.tensor_port, "tensor_port")
        if self.heartbeat_interval_sec <= 0:
            raise ValueError(
                "heartbeat_interval_sec must be positive, "
                f"got {self.heartbeat_interval_sec}"
            )
        if self.heartbeat_timeout_sec <= 0:
            raise ValueError(
                "heartbeat_timeout_sec must be positive, "
                f"got {self.heartbeat_timeout_sec}"
            )
        if self.heartbeat_timeout_sec <= self.heartbeat_interval_sec:
            raise ValueError(
                f"heartbeat_timeout_sec ({self.heartbeat_timeout_sec}) must be greater than "
                f"heartbeat_interval_sec ({self.heartbeat_interval_sec})"
            )
        if self.min_workers < 1:
            raise ValueError(f"min_workers must be >= 1, got {self.min_workers}")
        if not self.auth_token_env or not _ENV_NAME_RE.match(self.auth_token_env):
            raise ValueError(
                "auth_token_env must be a valid environment variable name"
            )
        self.auth_token = resolve_auth_token(self.auth_token, self.auth_token_env)

    @property
    def master_grpc_address(self) -> str:
        """Full gRPC address of the master."""
        return f"{self.master_addr}:{self.master_port}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "tensor_port": self.tensor_port,
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "heartbeat_timeout_sec": self.heartbeat_timeout_sec,
            "discovery_enabled": self.discovery_enabled,
            "service_name": self.service_name,
            "host": self.host,
            "min_workers": self.min_workers,
            "auth_token_env": self.auth_token_env,
            "auth_required": self.auth_token is not None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterConfig":
        """Create from dictionary, ignoring unknown keys."""
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if "role" in data:
            data["role"] = NodeRole(data["role"])
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for distributed training.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Total batch size (split across nodes by weight).
        learning_rate: Initial learning rate.
        compression: Gradient compression type.
        topk_ratio: Ratio for Top-K sparsification (0.0-1.0).
        checkpoint_every: Checkpoint every N epochs.
        checkpoint_dir: Directory for saving checkpoints.
        calibration_steps: Number of steps for throughput calibration.
        dynamic_rebalance_threshold: Throughput ratio change to trigger rebalance.
        device: PyTorch device to use ("mps", "cpu").
    """
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.1
    compression: CompressionType = CompressionType.TOPK_FP16
    topk_ratio: float = DEFAULT_TOPK_RATIO
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    checkpoint_dir: str = "./checkpoints"
    calibration_steps: int = 10
    dynamic_rebalance_threshold: float = 0.15
    device: str = "mps"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0.0 < self.topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0.0, 1.0], got {self.topk_ratio}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if self.calibration_steps < 1:
            raise ValueError(f"calibration_steps must be >= 1, got {self.calibration_steps}")
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError(f"device must be 'mps', 'cpu', or 'cuda', got '{self.device}'")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "compression": self.compression.value,
            "topk_ratio": self.topk_ratio,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_dir": self.checkpoint_dir,
            "calibration_steps": self.calibration_steps,
            "dynamic_rebalance_threshold": self.dynamic_rebalance_threshold,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create from dictionary, ignoring unknown keys."""
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if "compression" in data:
            data["compression"] = CompressionType(data["compression"])
        return cls(**data)


@dataclass
class ClusterState:
    """Runtime state of the cluster.

    Attributes:
        world_size: Total number of nodes in the cluster.
        nodes: Registered nodes by rank.
        training_active: Whether training is currently running.
        current_epoch: Current training epoch.
        current_step: Current training step within epoch.
    """
    world_size: int = 0
    nodes: dict[int, NodeConfig] = field(default_factory=dict)
    training_active: bool = False
    current_epoch: int = 0
    current_step: int = 0

    def get_node(self, rank: int) -> Optional[NodeConfig]:
        """Get node by rank."""
        return self.nodes.get(rank)

    def add_node(self, node: NodeConfig) -> None:
        """Add or update a node."""
        self.nodes[node.rank] = node
        self.world_size = len(self.nodes)

    def remove_node(self, rank: int) -> Optional[NodeConfig]:
        """Remove a node by rank."""
        node = self.nodes.pop(rank, None)
        self.world_size = len(self.nodes)
        return node

    def get_all_tensor_addresses(self) -> list[tuple[str, int]]:
        """Get (ip, port) tuples for all nodes' tensor channels."""
        return [
            (node.ip_address, node.tensor_port)
            for node in self.nodes.values()
        ]
