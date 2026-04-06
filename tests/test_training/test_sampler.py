"""Tests for the weighted distributed sampler."""

import pytest
import torch
from torch.utils.data import TensorDataset

from macfleet.training.sampler import (
    DistributedBatchSampler,
    WeightedDistributedSampler,
    compute_weights_from_gpu_cores,
    compute_weights_from_throughput,
)


def _make_dataset(size: int) -> TensorDataset:
    return TensorDataset(torch.arange(size))


class TestWeightedDistributedSampler:
    def test_equal_weights(self):
        ds = _make_dataset(100)
        s0 = WeightedDistributedSampler(ds, num_replicas=2, rank=0)
        s1 = WeightedDistributedSampler(ds, num_replicas=2, rank=1)
        assert len(s0) == 50
        assert len(s1) == 50

    def test_weighted_split(self):
        ds = _make_dataset(100)
        # Rank 0 gets 70%, rank 1 gets 30%
        s0 = WeightedDistributedSampler(ds, num_replicas=2, rank=0, weights=[0.7, 0.3])
        s1 = WeightedDistributedSampler(ds, num_replicas=2, rank=1, weights=[0.7, 0.3])
        assert len(s0) == 70
        assert len(s1) == 30

    def test_all_samples_covered(self):
        ds = _make_dataset(100)
        weights = [0.6, 0.25, 0.15]
        samplers = [
            WeightedDistributedSampler(ds, num_replicas=3, rank=i, weights=weights, shuffle=False)
            for i in range(3)
        ]
        all_indices = set()
        for s in samplers:
            all_indices.update(list(s))
        assert len(all_indices) == 100

    def test_no_overlap(self):
        ds = _make_dataset(100)
        s0 = WeightedDistributedSampler(ds, num_replicas=2, rank=0, shuffle=False)
        s1 = WeightedDistributedSampler(ds, num_replicas=2, rank=1, shuffle=False)
        idx0 = set(s0)
        idx1 = set(s1)
        assert len(idx0 & idx1) == 0

    def test_shuffle_reproducibility(self):
        ds = _make_dataset(100)
        s1 = WeightedDistributedSampler(ds, num_replicas=2, rank=0, seed=42)
        s2 = WeightedDistributedSampler(ds, num_replicas=2, rank=0, seed=42)
        assert list(s1) == list(s2)

    def test_different_epochs(self):
        ds = _make_dataset(100)
        s = WeightedDistributedSampler(ds, num_replicas=2, rank=0, shuffle=True)
        s.set_epoch(0)
        epoch0 = list(s)
        s.set_epoch(1)
        epoch1 = list(s)
        assert epoch0 != epoch1  # different shuffle

    def test_dynamic_weight_update(self):
        ds = _make_dataset(100)
        s = WeightedDistributedSampler(ds, num_replicas=2, rank=0, weights=[0.5, 0.5])
        assert len(s) == 50

        s.set_weights([0.8, 0.2])
        assert len(s) == 80

    def test_invalid_weights_length(self):
        ds = _make_dataset(100)
        with pytest.raises(ValueError):
            WeightedDistributedSampler(ds, num_replicas=2, rank=0, weights=[0.5])

    def test_three_nodes(self):
        ds = _make_dataset(120)
        weights = [0.5, 0.3, 0.2]
        sizes = []
        for rank in range(3):
            s = WeightedDistributedSampler(ds, num_replicas=3, rank=rank, weights=weights)
            sizes.append(len(s))
        assert sum(sizes) == 120
        assert sizes[0] == 60  # 50% of 120
        assert sizes[1] == 36  # 30% of 120


class TestDistributedBatchSampler:
    def test_batch_sizes(self):
        ds = _make_dataset(100)
        # Global batch_size=10, 2 nodes with equal weights
        bs0 = DistributedBatchSampler(ds, batch_size=10, num_replicas=2, rank=0)
        bs1 = DistributedBatchSampler(ds, batch_size=10, num_replicas=2, rank=1)
        assert bs0.batch_size == 5
        assert bs1.batch_size == 5

    def test_weighted_batch_sizes(self):
        ds = _make_dataset(100)
        # Pro gets 70% of global batch
        bs = DistributedBatchSampler(
            ds, batch_size=10, num_replicas=2, rank=0, weights=[0.7, 0.3]
        )
        assert bs.batch_size == 7

    def test_yields_batches(self):
        ds = _make_dataset(50)
        bs = DistributedBatchSampler(
            ds, batch_size=10, num_replicas=1, rank=0, shuffle=False
        )
        batches = list(bs)
        assert all(len(b) == 10 for b in batches[:-1])


class TestWeightComputations:
    def test_from_gpu_cores(self):
        weights = compute_weights_from_gpu_cores([10, 20])
        assert abs(weights[0] - 1 / 3) < 0.01
        assert abs(weights[1] - 2 / 3) < 0.01

    def test_from_throughput(self):
        weights = compute_weights_from_throughput([100.0, 50.0])
        assert abs(weights[0] - 2 / 3) < 0.01
        assert abs(weights[1] - 1 / 3) < 0.01

    def test_zero_gpu_cores(self):
        weights = compute_weights_from_gpu_cores([0, 0])
        assert weights == [0.5, 0.5]

    def test_zero_throughput(self):
        weights = compute_weights_from_throughput([0.0, 0.0])
        assert weights == [0.5, 0.5]
