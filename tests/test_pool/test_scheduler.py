"""Tests for the heterogeneous workload scheduler."""

from macfleet.engines.base import HardwareProfile, ThermalPressure
from macfleet.pool.registry import ClusterRegistry, NodeRecord
from macfleet.pool.scheduler import Scheduler, SchedulerConfig


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


class TestSchedulerWeights:
    def test_equal_nodes_equal_weights(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a", gpu_cores=10))
        reg.register(_make_node("b", gpu_cores=10))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        weights = scheduler.compute_weights()

        assert abs(weights["a"] - 0.5) < 0.01
        assert abs(weights["b"] - 0.5) < 0.01

    def test_unequal_gpu_cores(self):
        reg = ClusterRegistry("air")
        reg.register(_make_node("air", gpu_cores=10))
        reg.register(_make_node("pro", gpu_cores=20))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        weights = scheduler.compute_weights()

        # air: 10/30 = 0.333, pro: 20/30 = 0.667
        assert abs(weights["air"] - 1 / 3) < 0.01
        assert abs(weights["pro"] - 2 / 3) < 0.01

    def test_thermal_reduces_weight(self):
        reg = ClusterRegistry("a")
        node_a = _make_node("a", gpu_cores=10)
        node_b = _make_node("b", gpu_cores=10)
        node_b.hardware.thermal_pressure = ThermalPressure.SERIOUS  # 0.7 factor

        reg.register(node_a)
        reg.register(node_b)

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        weights = scheduler.compute_weights()

        # a: 10*1.0=10, b: 10*0.7=7, total=17
        # a: 10/17=0.588, b: 7/17=0.412
        assert weights["a"] > weights["b"]
        assert abs(weights["a"] - 10 / 17) < 0.01

    def test_throughput_based_weights(self):
        reg = ClusterRegistry("a")
        node_a = _make_node("a", gpu_cores=10)
        node_b = _make_node("b", gpu_cores=10)
        node_a.throughput_samples_sec = 100.0  # measured
        node_b.throughput_samples_sec = 50.0   # slower (maybe Air throttling)

        reg.register(node_a)
        reg.register(node_b)

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=True))
        weights = scheduler.compute_weights()

        # a: 100/150=0.667, b: 50/150=0.333
        assert abs(weights["a"] - 2 / 3) < 0.01


class TestSchedulerAssignment:
    def test_batch_split(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a", gpu_cores=10))
        reg.register(_make_node("b", gpu_cores=20))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        assignments = scheduler.assign(global_batch_size=128)

        assert len(assignments) == 2
        total_batch = sum(a.batch_size for a in assignments)
        assert total_batch == 128

        # Pro (20 cores) should get more
        by_id = {a.node_id: a for a in assignments}
        assert by_id["b"].batch_size > by_id["a"].batch_size

    def test_minimum_batch_guard(self):
        reg = ClusterRegistry("strong")
        reg.register(_make_node("strong", gpu_cores=100))
        reg.register(_make_node("weak", gpu_cores=1))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False, min_batch_per_node=4))
        assignments = scheduler.assign(global_batch_size=10)

        by_id = {a.node_id: a for a in assignments}
        # weak gets 1/101 * 10 ≈ 0 → not viable
        assert not by_id["weak"].is_viable

    def test_single_node(self):
        reg = ClusterRegistry("solo")
        reg.register(_make_node("solo", gpu_cores=16))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        assignments = scheduler.assign(global_batch_size=64)

        assert len(assignments) == 1
        assert assignments[0].batch_size == 64
        assert assignments[0].weight == 1.0

    def test_non_viable_detection(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a", gpu_cores=100))
        reg.register(_make_node("b", gpu_cores=1))

        scheduler = Scheduler(reg, SchedulerConfig(use_throughput=False))
        non_viable = scheduler.get_non_viable_nodes(global_batch_size=8)
        assert "b" in non_viable

    def test_rebalance_trigger(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a"))

        scheduler = Scheduler(reg, SchedulerConfig(rebalance_every_n_steps=5))
        for _ in range(4):
            assert not scheduler.should_rebalance()
        assert scheduler.should_rebalance()  # 5th call
