"""Shared test fixtures for MacFleet tests."""

import logging

import pytest
import torch

from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def cluster_config_master():
    """Create a master cluster config for testing."""
    return ClusterConfig(role=NodeRole.MASTER)


@pytest.fixture
def cluster_config_worker():
    """Create a worker cluster config for testing."""
    return ClusterConfig(role=NodeRole.WORKER)


@pytest.fixture
def training_config():
    """Create a default training config for testing."""
    return TrainingConfig(epochs=2, batch_size=32, device="cpu")


@pytest.fixture
def node_config_factory():
    """Factory for creating NodeConfig instances."""
    def make(hostname="test-node", ip="127.0.0.1", gpu_cores=10,
             ram_gb=16, bw=200.0, tensor_port=50052, rank=-1):
        return NodeConfig(
            hostname=hostname,
            ip_address=ip,
            gpu_cores=gpu_cores,
            ram_gb=ram_gb,
            memory_bandwidth_gbps=bw,
            tensor_port=tensor_port,
            rank=rank,
        )
    return make


@pytest.fixture
def cluster_state():
    """Create an empty cluster state."""
    return ClusterState()


@pytest.fixture
def small_tensor():
    """A small tensor for testing."""
    return torch.randn(100)


@pytest.fixture
def medium_tensor():
    """A medium tensor for testing (1MB)."""
    return torch.randn(256 * 1024)  # ~1MB in float32
