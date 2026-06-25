"""Tests for weighted distributed sampler."""

import pytest
import torch
from torch.utils.data import TensorDataset

from macfleet.training.distributed_sampler import (
    DistributedBatchSampler,
    WeightedDistributedSampler,
    compute_weights_from_gpu_cores,
    compute_weights_from_throughput,
)


class TestWeightedDistributedSampler:
    """Tests for WeightedDistributedSampler."""

    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        data = torch.randn(1000, 10)
        labels = torch.randint(0, 10, (1000,))
        return TensorDataset(data, labels)

    def test_equal_weights(self, dataset):
        """Test sampler with equal weights."""
        sampler0 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=None
        )
        sampler1 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=1, weights=None
        )

        # Each should get half
        assert len(sampler0) == 500
        assert len(sampler1) == 500

    def test_weighted_split(self, dataset):
        """Test sampler with weighted split."""
        # Pro: 16 cores, Air: 10 cores
        weights = compute_weights_from_gpu_cores([16, 10])

        sampler0 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=weights
        )
        sampler1 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=1, weights=weights
        )

        # Pro should get more samples
        assert len(sampler0) > len(sampler1)
        assert len(sampler0) == 615  # ~61.5%
        assert len(sampler1) == 385  # ~38.5%

    def test_no_overlap(self, dataset):
        """Test that sampled indices don't overlap."""
        weights = [0.6, 0.4]

        sampler0 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=weights
        )
        sampler1 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=1, weights=weights
        )

        indices0 = set(iter(sampler0))
        indices1 = set(iter(sampler1))

        # No overlap
        assert len(indices0 & indices1) == 0

    def test_full_coverage(self, dataset):
        """Test that all indices are covered."""
        weights = [0.7, 0.3]

        sampler0 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=weights
        )
        sampler1 = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=1, weights=weights
        )

        indices0 = set(iter(sampler0))
        indices1 = set(iter(sampler1))

        # All indices covered
        assert len(indices0 | indices1) == 1000

    def test_shuffle_consistency(self, dataset):
        """Test shuffle is consistent across epochs."""
        sampler = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=True, seed=42
        )

        # Same epoch should give same order
        sampler.set_epoch(1)
        indices1 = list(iter(sampler))

        sampler.set_epoch(1)
        indices2 = list(iter(sampler))

        assert indices1 == indices2

        # Different epoch should give different order
        sampler.set_epoch(2)
        indices3 = list(iter(sampler))

        assert indices1 != indices3

    def test_update_weights(self, dataset):
        """Test updating weights dynamically."""
        sampler = WeightedDistributedSampler(
            dataset, num_replicas=2, rank=0, weights=[0.5, 0.5]
        )

        assert len(sampler) == 500

        # Update to 70/30
        sampler.set_weights([0.7, 0.3])

        assert len(sampler) == 700


class TestDistributedBatchSampler:
    """Tests for DistributedBatchSampler."""

    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        data = torch.randn(1000, 10)
        labels = torch.randint(0, 10, (1000,))
        return TensorDataset(data, labels)

    def test_batch_sizes(self, dataset):
        """Test that batch sizes are weighted correctly."""
        weights = [0.6, 0.4]

        batch_sampler0 = DistributedBatchSampler(
            dataset, batch_size=100, num_replicas=2, rank=0, weights=weights
        )
        batch_sampler1 = DistributedBatchSampler(
            dataset, batch_size=100, num_replicas=2, rank=1, weights=weights
        )

        # Rank 0 should have larger batch size
        assert batch_sampler0.batch_size == 60  # 60% of 100
        assert batch_sampler1.batch_size == 40  # 40% of 100

    def test_iteration(self, dataset):
        """Test iterating over batches."""
        batch_sampler = DistributedBatchSampler(
            dataset, batch_size=100, num_replicas=2, rank=0
        )

        batches = list(batch_sampler)

        # Check batch structure
        assert len(batches) > 0
        assert all(isinstance(b, list) for b in batches)


class TestWeightComputation:
    """Tests for weight computation functions."""

    def test_weights_from_gpu_cores(self):
        """Test computing weights from GPU cores."""
        # M4 Pro (16) and M4 (10)
        weights = compute_weights_from_gpu_cores([16, 10])

        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6
        assert weights[0] > weights[1]  # Pro should have higher weight
        assert abs(weights[0] - 16/26) < 1e-6

    def test_weights_from_throughput(self):
        """Test computing weights from throughput."""
        throughputs = [1000.0, 600.0]  # samples/sec
        weights = compute_weights_from_throughput(throughputs)

        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6
        assert weights[0] > weights[1]
        assert abs(weights[0] - 1000/1600) < 1e-6

    def test_zero_throughput_handling(self):
        """Test handling of zero throughput."""
        weights = compute_weights_from_throughput([0.0, 0.0])

        # Should fall back to equal weights
        assert weights == [0.5, 0.5]

    def test_single_node(self):
        """Test with single node."""
        weights = compute_weights_from_gpu_cores([16])

        assert weights == [1.0]
