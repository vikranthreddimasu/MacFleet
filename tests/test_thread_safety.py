"""Tests for thread safety of gRPC servicer and coordinator.

Verifies that concurrent access from multiple gRPC threads
does not cause race conditions.
"""

import threading
import time

from macfleet.comm.grpc_service import ClusterControlServicer
from macfleet.core.config import ClusterState, NodeConfig


class FakeContext:
    """Minimal mock for grpc.ServicerContext."""
    pass


class FakeRegisterRequest:
    """Minimal mock for RegisterRequest."""
    def __init__(self, hostname, ip_address, gpu_cores=10, ram_gb=16,
                 memory_bandwidth_gbps=100.0, tensor_port=50052):
        self.hostname = hostname
        self.ip_address = ip_address
        self.gpu_cores = gpu_cores
        self.ram_gb = ram_gb
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self.tensor_port = tensor_port


class FakeHeartbeatRequest:
    """Minimal mock for HeartbeatRequest."""
    def __init__(self, rank, throughput=100.0, thermal_state="nominal", current_step=0):
        self.rank = rank
        self.throughput_samples_per_sec = throughput
        self.thermal_state = thermal_state
        self.timestamp_ms = int(time.time() * 1000)
        self.current_step = current_step


class FakeBarrierRequest:
    """Minimal mock for BarrierRequest."""
    def __init__(self, rank, barrier_id="test_barrier", step=0):
        self.rank = rank
        self.barrier_id = barrier_id
        self.step = step


def test_concurrent_register_unique_ranks():
    """Verify that concurrent Register calls from multiple threads
    always produce unique rank assignments."""
    state = ClusterState()
    # Add master node as rank 0
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

    num_threads = 8
    results = [None] * num_threads
    errors = []

    def register_worker(idx):
        try:
            request = FakeRegisterRequest(
                hostname=f"worker-{idx}",
                ip_address=f"10.0.0.{idx + 10}",
                gpu_cores=10,
            )
            response = servicer.Register(request, FakeContext())
            results[idx] = response.assigned_rank
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during registration: {errors}"

    # All ranks should be unique
    ranks = [r for r in results if r is not None]
    assert len(ranks) == num_threads
    assert len(set(ranks)) == num_threads, f"Duplicate ranks found: {ranks}"

    # All ranks should be >= 1 (rank 0 is master)
    assert all(r >= 1 for r in ranks), f"Invalid rank found: {ranks}"


def test_concurrent_heartbeats():
    """Verify that concurrent heartbeats don't corrupt state."""
    state = ClusterState()
    master = NodeConfig(
        hostname="master", ip_address="10.0.0.1",
        gpu_cores=16, ram_gb=24, memory_bandwidth_gbps=273.0,
        tensor_port=50052, rank=0, workload_weight=0.6,
    )
    state.add_node(master)

    worker = NodeConfig(
        hostname="worker", ip_address="10.0.0.2",
        gpu_cores=10, ram_gb=16, memory_bandwidth_gbps=120.0,
        tensor_port=50052, rank=1, workload_weight=0.4,
    )
    state.add_node(worker)

    servicer = ClusterControlServicer(
        cluster_state=state,
        tensor_addr="10.0.0.1",
        tensor_port=50052,
    )

    errors = []

    def send_heartbeats(rank, count=50):
        try:
            for i in range(count):
                request = FakeHeartbeatRequest(rank=rank, throughput=float(i))
                response = servicer.Heartbeat(request, FakeContext())
                assert response.acknowledged
        except Exception as e:
            errors.append(e)

    # Multiple threads sending heartbeats for same and different ranks
    threads = [
        threading.Thread(target=send_heartbeats, args=(1, 50)),
        threading.Thread(target=send_heartbeats, args=(1, 50)),
        threading.Thread(target=send_heartbeats, args=(0, 50)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during heartbeats: {errors}"


def test_concurrent_barriers():
    """Verify that concurrent barrier operations are safe."""
    state = ClusterState()
    for rank in range(4):
        node = NodeConfig(
            hostname=f"node-{rank}", ip_address=f"10.0.0.{rank + 1}",
            gpu_cores=10, ram_gb=16, memory_bandwidth_gbps=120.0,
            tensor_port=50052, rank=rank, workload_weight=0.25,
        )
        state.add_node(node)

    servicer = ClusterControlServicer(
        cluster_state=state,
        tensor_addr="10.0.0.1",
        tensor_port=50052,
    )

    results = [None] * 4
    errors = []

    def reach_barrier(rank):
        try:
            request = FakeBarrierRequest(rank=rank, barrier_id="epoch_0")
            response = servicer.SyncBarrier(request, FakeContext())
            results[rank] = response
        except Exception as e:
            errors.append(e)

    # All 4 ranks reach barrier concurrently
    threads = [threading.Thread(target=reach_barrier, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during barrier: {errors}"

    # Exactly one thread should see proceed=True (the last one)
    proceed_count = sum(1 for r in results if r and r.proceed)
    assert proceed_count == 1, f"Expected 1 proceed, got {proceed_count}"

    # All responses should have valid node counts
    for r in results:
        assert r is not None
        assert 1 <= r.nodes_at_barrier <= 4
