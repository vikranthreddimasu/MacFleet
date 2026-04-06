"""Weighted distributed sampler for MacFleet v2.

Extends PyTorch's Sampler to support weighted splitting based on
each node's compute capacity (GPU cores, measured throughput).
Ported from MacFleet v1 with dynamic weight update support.

Example:
    Air  (10 GPU cores, weight=0.38): gets 38% of samples
    Pro  (16 GPU cores, weight=0.62): gets 62% of samples
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


class WeightedDistributedSampler(Sampler[int]):
    """Distributed sampler with weighted batch allocation.

    Unlike PyTorch's DistributedSampler which splits data equally,
    this gives each node a proportion of samples based on their
    workload weight (from GPU cores or calibrated throughput).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        weights: Optional[list[float]] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Normalize weights
        if weights is None:
            self.weights = [1.0 / num_replicas] * num_replicas
        else:
            if len(weights) != num_replicas:
                raise ValueError(
                    f"weights length {len(weights)} != num_replicas {num_replicas}"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self._recompute_counts()

    def _recompute_counts(self) -> None:
        """Recompute sample counts from weights."""
        total_size = len(self.dataset)
        self._sample_counts = self._compute_sample_counts(total_size)
        self.num_samples = self._sample_counts[self.rank]
        self.total_size = sum(self._sample_counts)

    def _compute_sample_counts(self, total_size: int) -> list[int]:
        """Compute number of samples for each rank based on weights."""
        counts = []
        remaining = total_size
        for i, weight in enumerate(self.weights):
            if i == len(self.weights) - 1:
                counts.append(remaining)
            else:
                count = int(total_size * weight)
                remaining -= count
                counts.append(count)
        return counts

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        start = sum(self._sample_counts[: self.rank])
        end = start + self._sample_counts[self.rank]
        return iter(indices[start:end])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling reproducibility."""
        self.epoch = epoch

    def set_weights(self, weights: list[float]) -> None:
        """Dynamically update weights (e.g., after scheduler rebalance)."""
        if len(weights) != self.num_replicas:
            raise ValueError(
                f"weights length {len(weights)} != num_replicas {self.num_replicas}"
            )
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self._recompute_counts()


class DistributedBatchSampler(Sampler[list[int]]):
    """Batch sampler that yields weighted batch sizes per rank.

    Instead of a fixed batch size per node, allocates batch samples
    based on node weights so each forward pass processes the
    appropriate amount of data.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        weights: Optional[list[float]] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.sampler = WeightedDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            weights=weights,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        # Compute this rank's batch size
        if weights is None:
            weights = [1.0 / num_replicas] * num_replicas
        total = sum(weights)
        normalized = [w / total for w in weights]
        self.batch_size = max(1, int(batch_size * normalized[rank]))
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #


def compute_weights_from_gpu_cores(gpu_cores: list[int]) -> list[float]:
    """Compute workload weights proportional to GPU core counts."""
    total = sum(gpu_cores)
    if total == 0:
        return [1.0 / len(gpu_cores)] * len(gpu_cores)
    return [c / total for c in gpu_cores]


def compute_weights_from_throughput(throughputs: list[float]) -> list[float]:
    """Compute workload weights proportional to measured throughput."""
    total = sum(throughputs)
    if total == 0:
        return [1.0 / len(throughputs)] * len(throughputs)
    return [t / total for t in throughputs]
