"""Stress tests for ClusterRegistry under concurrent mutation.

Production scenario: gossip heartbeat callbacks fire from one async
task while training reads alive_nodes from another, while a fresh peer
discovery is registering nodes from a third. The registry's lock must
prevent torn reads of (hardware, status, data_port) and the bully
election must converge to one coordinator regardless of interleaving.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from macfleet.engines.base import HardwareProfile, ThermalPressure
from macfleet.pool.heartbeat import NodeStatus
from macfleet.pool.registry import ClusterRegistry, NodeRecord


def _make_node(node_id: str, gpu_cores: int = 10, ram_gb: float = 16.0) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        hostname=f"host-{node_id}",
        ip_address="127.0.0.1",
        port=50051,
        hardware=HardwareProfile(
            hostname=f"host-{node_id}",
            node_id=node_id,
            gpu_cores=gpu_cores,
            ram_gb=ram_gb,
            memory_bandwidth_gbps=100.0,
            has_ane=True,
            chip_name="Apple M4",
        ),
    )


class TestRegistryConcurrentMutation:
    """N threads hammering register/mark_failed/mark_alive at once."""

    def test_no_torn_reads_on_concurrent_register(self):
        reg = ClusterRegistry("local")
        n_threads = 8
        n_iterations = 200
        stop = threading.Event()
        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                for i in range(n_iterations):
                    if stop.is_set():
                        return
                    node = _make_node(
                        f"writer-{idx}-{i % 5}",
                        gpu_cores=random.randint(8, 32),
                    )
                    reg.register(node)
                    if random.random() < 0.3:
                        reg.mark_failed(node.node_id)
                    if random.random() < 0.3:
                        reg.mark_alive(node.node_id)
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(n_iterations):
                    if stop.is_set():
                        return
                    # alive_nodes must always return a coherent list of
                    # NodeRecords. compute_score read on a torn record
                    # could blow up if hardware was None.
                    nodes = reg.alive_nodes
                    for n in nodes:
                        # Trigger composite read across multiple fields.
                        _ = n.hardware.compute_score
                        _ = n.is_alive
                        _ = reg.coordinator_id
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i,), daemon=True)
            for i in range(n_threads)
        ]
        threads.extend(
            threading.Thread(target=reader, daemon=True)
            for _ in range(n_threads // 2)
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            if t.is_alive():
                stop.set()
                pytest.fail("registry stress test deadlocked")
        assert not errors, f"errors during stress: {errors[:3]}"

    def test_election_converges_after_churn(self):
        """After all threads stop, exactly one coordinator is elected."""
        reg = ClusterRegistry("local")
        reg.register(_make_node("local"))
        nodes_to_churn = [f"peer-{i:02d}" for i in range(20)]

        def churn() -> None:
            for nid in nodes_to_churn:
                reg.register(_make_node(nid, gpu_cores=random.randint(8, 60)))
                if random.random() < 0.5:
                    reg.mark_failed(nid)
                if random.random() < 0.3:
                    reg.mark_alive(nid)

        threads = [threading.Thread(target=churn, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Final state: every alive node must agree on the same coordinator.
        coord = reg.coordinator_id
        assert coord is not None
        alive = reg.alive_nodes
        if alive:
            # Coordinator must be one of the alive nodes.
            assert coord in [n.node_id for n in alive]
            # Coordinator must have the highest compute_score among alive.
            max_score = max(n.compute_score for n in alive)
            coord_record = reg.get_node(coord)
            assert coord_record is not None
            assert coord_record.compute_score == max_score

    def test_update_hardware_atomic_swap(self):
        """update_hardware must swap hardware AND data_port together."""
        reg = ClusterRegistry("local")
        reg.register(_make_node("local"))

        peer_id = "remote"
        reg.register(NodeRecord(
            node_id=peer_id, hostname=peer_id, ip_address="10.0.0.2",
            port=50051, data_port=50052,
            hardware=HardwareProfile(
                hostname=peer_id, node_id=peer_id,
                gpu_cores=8, ram_gb=8.0, memory_bandwidth_gbps=100.0,
                has_ane=True, chip_name="Old chip",
            ),
        ))

        n_iterations = 500
        new_hw_1 = HardwareProfile(
            hostname=peer_id, node_id=peer_id,
            gpu_cores=60, ram_gb=192.0, memory_bandwidth_gbps=800.0,
            has_ane=True, chip_name="M4 Ultra",
        )
        new_hw_2 = HardwareProfile(
            hostname=peer_id, node_id=peer_id,
            gpu_cores=12, ram_gb=24.0, memory_bandwidth_gbps=200.0,
            has_ane=True, chip_name="M4 Pro",
        )

        torn_reads: list[tuple] = []

        def updater() -> None:
            for i in range(n_iterations):
                if i % 2 == 0:
                    reg.update_hardware(peer_id, new_hw_1, new_data_port=60052)
                else:
                    reg.update_hardware(peer_id, new_hw_2, new_data_port=61052)

        def reader() -> None:
            for _ in range(n_iterations):
                rec = reg.get_node(peer_id)
                if rec is None:
                    continue
                # Coupled invariant: gpu_cores=60 ⇔ data_port=60052,
                # gpu_cores=12 ⇔ data_port=61052. Both ports stay
                # within the valid 0..65535 range so the registry's
                # port guard accepts them.
                cores = rec.hardware.gpu_cores
                port = rec.data_port
                if (cores == 60 and port != 60052) or (cores == 12 and port != 61052):
                    torn_reads.append((cores, port))

        threads = [threading.Thread(target=updater, daemon=True)]
        threads.extend(threading.Thread(target=reader, daemon=True) for _ in range(4))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20.0)

        assert not torn_reads, (
            f"observed {len(torn_reads)} torn (gpu_cores, data_port) reads — "
            f"first 3: {torn_reads[:3]}"
        )

    def test_get_ranks_stable_under_churn(self):
        """get_ranks should always return distinct ranks for distinct nodes."""
        reg = ClusterRegistry("local")
        for i in range(10):
            reg.register(_make_node(f"node-{i:02d}", gpu_cores=8 + i))

        results: list[bool] = []

        def reader() -> None:
            for _ in range(500):
                ranks = reg.get_ranks()
                # Each node should have a unique rank.
                if len(set(ranks.values())) != len(ranks):
                    results.append(False)
                    return
            results.append(True)

        def churner() -> None:
            for i in range(500):
                nid = f"node-{i % 10:02d}"
                if random.random() < 0.5:
                    reg.mark_failed(nid)
                else:
                    reg.mark_alive(nid)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        threads.extend(threading.Thread(target=churner, daemon=True) for _ in range(2))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert all(results), "get_ranks observed duplicate ranks under churn"


class TestRegistryUpdateThroughput:
    """update_throughput is also lock-protected — verify under contention."""

    def test_throughput_writes_no_torn(self):
        reg = ClusterRegistry("local")
        for i in range(5):
            reg.register(_make_node(f"n-{i}"))

        def writer(idx: int) -> None:
            for i in range(200):
                nid = f"n-{idx}"
                reg.update_throughput(nid, float(i))

        threads = [
            threading.Thread(target=writer, args=(i,), daemon=True)
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Final value must be 199 (the last write each thread did).
        for i in range(5):
            rec = reg.get_node(f"n-{i}")
            assert rec is not None
            assert rec.throughput_samples_sec == 199.0


class TestRegistryThermalUpdates:
    def test_thermal_pressure_swaps_under_contention(self):
        reg = ClusterRegistry("local")
        reg.register(_make_node("n0"))

        def writer() -> None:
            for i in range(500):
                pressure = (
                    ThermalPressure.NOMINAL if i % 2 == 0
                    else ThermalPressure.SERIOUS
                )
                reg.update_thermal("n0", pressure)

        def reader() -> None:
            for _ in range(500):
                rec = reg.get_node("n0")
                if rec is not None:
                    # Pressure must always be one of the four enum values.
                    assert rec.hardware.thermal_pressure in ThermalPressure

        threads = [threading.Thread(target=writer, daemon=True)]
        threads.extend(threading.Thread(target=reader, daemon=True) for _ in range(3))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
