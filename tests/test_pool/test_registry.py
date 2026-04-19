"""Tests for the cluster registry and coordinator election."""

from macfleet.engines.base import HardwareProfile
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


class TestClusterRegistry:
    def test_register_and_world_size(self):
        reg = ClusterRegistry("node-a")
        reg.register(_make_node("node-a"))
        reg.register(_make_node("node-b"))
        assert reg.world_size == 2

    def test_deregister(self):
        reg = ClusterRegistry("node-a")
        reg.register(_make_node("node-a"))
        reg.register(_make_node("node-b"))
        reg.deregister("node-b")
        assert reg.world_size == 1

    def test_mark_failed_reduces_world_size(self):
        reg = ClusterRegistry("node-a")
        reg.register(_make_node("node-a"))
        reg.register(_make_node("node-b"))
        reg.mark_failed("node-b")
        assert reg.world_size == 1

    def test_mark_alive_restores(self):
        reg = ClusterRegistry("node-a")
        reg.register(_make_node("node-a"))
        reg.register(_make_node("node-b"))
        reg.mark_failed("node-b")
        assert reg.world_size == 1
        reg.mark_alive("node-b")
        assert reg.world_size == 2


class TestCoordinatorElection:
    def test_highest_score_wins(self):
        reg = ClusterRegistry("weak")
        reg.register(_make_node("weak", gpu_cores=8, ram_gb=8.0))    # score = 8*10 + 100*2 + 8 = 288
        reg.register(_make_node("strong", gpu_cores=20, ram_gb=48.0)) # score = 20*10 + 100*2 + 48 = 448
        assert reg.coordinator_id == "strong"

    def test_coordinator_changes_on_failure(self):
        reg = ClusterRegistry("weak")
        reg.register(_make_node("weak", gpu_cores=8))
        reg.register(_make_node("strong", gpu_cores=20))
        assert reg.coordinator_id == "strong"

        reg.mark_failed("strong")
        assert reg.coordinator_id == "weak"

    def test_coordinator_restored_on_recovery(self):
        reg = ClusterRegistry("weak")
        reg.register(_make_node("weak", gpu_cores=8))
        reg.register(_make_node("strong", gpu_cores=20))
        reg.mark_failed("strong")
        assert reg.coordinator_id == "weak"

        reg.mark_alive("strong")
        assert reg.coordinator_id == "strong"

    def test_single_node_is_coordinator(self):
        reg = ClusterRegistry("solo")
        reg.register(_make_node("solo"))
        assert reg.coordinator_id == "solo"
        assert reg.is_coordinator

    def test_empty_registry_no_coordinator(self):
        reg = ClusterRegistry("none")
        assert reg.coordinator_id is None
        assert not reg.is_coordinator


class TestRankAssignment:
    def test_ranks_by_score_descending(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a", gpu_cores=8))    # lower score
        reg.register(_make_node("b", gpu_cores=16))   # higher score
        reg.register(_make_node("c", gpu_cores=12))   # middle score

        ranks = reg.get_ranks()
        assert ranks["b"] == 0  # highest score = rank 0
        assert ranks["c"] == 1
        assert ranks["a"] == 2

    def test_ranks_exclude_failed(self):
        reg = ClusterRegistry("a")
        reg.register(_make_node("a", gpu_cores=8))
        reg.register(_make_node("b", gpu_cores=16))
        reg.mark_failed("b")

        ranks = reg.get_ranks()
        assert "b" not in ranks
        assert ranks["a"] == 0
