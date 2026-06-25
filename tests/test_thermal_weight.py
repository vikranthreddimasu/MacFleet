"""Tests for thermal weight management.

Verifies that thermal throttling reduces weight relative to baseline
and recovers when thermal state improves.
"""

import time

from macfleet.comm.grpc_service import ClusterControlServicer
from macfleet.core.config import ClusterState, NodeConfig


class FakeContext:
    pass


class FakeRegisterRequest:
    def __init__(self, hostname, ip_address, gpu_cores=10, ram_gb=16,
                 memory_bandwidth_gbps=100.0, tensor_port=50052):
        self.hostname = hostname
        self.ip_address = ip_address
        self.gpu_cores = gpu_cores
        self.ram_gb = ram_gb
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self.tensor_port = tensor_port


class FakeHeartbeatRequest:
    def __init__(self, rank, thermal_state="nominal"):
        self.rank = rank
        self.throughput_samples_per_sec = 100.0
        self.thermal_state = thermal_state
        self.timestamp_ms = int(time.time() * 1000)
        self.current_step = 0


def _create_servicer_with_worker():
    """Create a servicer with master (rank 0) and one worker (rank 1)."""
    state = ClusterState()
    master = NodeConfig(
        hostname="master", ip_address="10.0.0.1",
        gpu_cores=16, ram_gb=24, memory_bandwidth_gbps=273.0,
        tensor_port=50052, rank=0, workload_weight=1.0,
    )
    state.add_node(master)

    servicer = ClusterControlServicer(
        cluster_state=state,
        tensor_addr="10.0.0.1",
        tensor_port=50052,
    )

    # Register a worker
    req = FakeRegisterRequest("worker", "10.0.0.2", gpu_cores=10)
    resp = servicer.Register(req, FakeContext())
    worker_rank = resp.assigned_rank
    base_weight = resp.workload_weight

    return servicer, worker_rank, base_weight


def test_thermal_nominal_preserves_weight():
    """Nominal thermal state should preserve baseline weight."""
    servicer, rank, base_weight = _create_servicer_with_worker()

    for _ in range(10):
        resp = servicer.Heartbeat(
            FakeHeartbeatRequest(rank=rank, thermal_state="nominal"),
            FakeContext(),
        )
        assert resp.new_workload_weight == base_weight


def test_thermal_throttle_uses_baseline():
    """Critical thermal state should reduce weight relative to baseline, not current."""
    servicer, rank, base_weight = _create_servicer_with_worker()

    # Send multiple "critical" heartbeats — weight should NOT spiral to zero
    for _ in range(10):
        resp = servicer.Heartbeat(
            FakeHeartbeatRequest(rank=rank, thermal_state="critical"),
            FakeContext(),
        )
        # Should always be base_weight * 0.7, not decaying further
        assert abs(resp.new_workload_weight - base_weight * 0.7) < 1e-6


def test_thermal_fair_partial_throttle():
    """Fair thermal state should reduce weight to 90% of baseline."""
    servicer, rank, base_weight = _create_servicer_with_worker()

    resp = servicer.Heartbeat(
        FakeHeartbeatRequest(rank=rank, thermal_state="fair"),
        FakeContext(),
    )
    assert abs(resp.new_workload_weight - base_weight * 0.9) < 1e-6


def test_thermal_recovery():
    """Weight should recover when thermal state returns to nominal."""
    servicer, rank, base_weight = _create_servicer_with_worker()

    # Throttle
    resp = servicer.Heartbeat(
        FakeHeartbeatRequest(rank=rank, thermal_state="critical"),
        FakeContext(),
    )
    assert resp.new_workload_weight < base_weight

    # Recover
    resp = servicer.Heartbeat(
        FakeHeartbeatRequest(rank=rank, thermal_state="nominal"),
        FakeContext(),
    )
    assert resp.new_workload_weight == base_weight
