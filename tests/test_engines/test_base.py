"""Tests for Engine protocol and core types."""

from macfleet.engines.base import (
    EngineType,
    HardwareProfile,
    ThermalPressure,
    TrainingMetrics,
)


class TestThermalPressure:
    def test_workload_multipliers(self):
        assert ThermalPressure.NOMINAL.workload_multiplier == 1.0
        assert ThermalPressure.FAIR.workload_multiplier == 0.9
        assert ThermalPressure.SERIOUS.workload_multiplier == 0.7
        assert ThermalPressure.CRITICAL.workload_multiplier == 0.3


class TestHardwareProfile:
    def test_compute_score(self):
        profile = HardwareProfile(
            hostname="test-mac",
            node_id="test-mac-abc",
            gpu_cores=16,
            ram_gb=36.0,
            memory_bandwidth_gbps=200.0,
            has_ane=True,
            chip_name="Apple M4 Pro",
            mps_available=True,
        )
        score = profile.compute_score
        # 16*10 + 200*2 + 36 = 160 + 400 + 36 = 596
        assert score == 596.0

    def test_can_fit_model(self):
        profile = HardwareProfile(
            hostname="air",
            node_id="air-xyz",
            gpu_cores=8,
            ram_gb=16.0,
            memory_bandwidth_gbps=100.0,
            has_ane=True,
            chip_name="Apple M4",
        )
        # Usable: 16 - 4 = 12GB. With 30% headroom:
        # 8GB model needs 8 * 1.3 = 10.4GB <= 12GB -> True
        assert profile.can_fit_model(8.0)
        # 10GB model needs 10 * 1.3 = 13GB > 12GB -> False
        assert not profile.can_fit_model(10.0)

    def test_can_fit_model_custom_headroom(self):
        profile = HardwareProfile(
            hostname="studio",
            node_id="studio-xyz",
            gpu_cores=40,
            ram_gb=192.0,
            memory_bandwidth_gbps=800.0,
            has_ane=True,
            chip_name="Apple M4 Ultra",
        )
        # 188GB usable, 150GB model with 20% headroom = 180GB <= 188 -> True
        assert profile.can_fit_model(150.0, headroom=0.2)


class TestEngineType:
    def test_values(self):
        assert EngineType.TORCH.value == "torch"
        assert EngineType.MLX.value == "mlx"


class TestTrainingMetrics:
    def test_defaults(self):
        metrics = TrainingMetrics()
        assert metrics.loss == 0.0
        assert metrics.throughput_samples_sec == 0.0
