"""Weighted distributed sampler for MacFleet v2.

Extends PyTorch's Sampler to support weighted splitting based on
each node's compute capacity (GPU cores, measured throughput).
Ported from MacFleet v1 with dynamic weight update support.

Example:
    Air  (10 GPU cores, weight=0.38): gets 38% of samples
    Pro  (16 GPU cores, weight=0.62): gets 62% of samples
"""

from __future__ import annotations

import math
from collections.abc import Sized
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


def _normalize_weights(weights: list[float], num_replicas: int) -> list[float]:
    """Validate and normalize workload weights without allowing invalid shards."""
    if len(weights) != num_replicas:
        raise ValueError(f"weights length {len(weights)} != num_replicas {num_replicas}")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight <= 0
        for weight in weights
    ):
        raise ValueError("weights must be finite, positive numbers")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")
    return [float(weight) / total for weight in weights]


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
        if not isinstance(num_replicas, int) or isinstance(num_replicas, bool) or num_replicas < 1:
            raise ValueError("num_replicas must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be an integer in [0, {num_replicas}), got {rank!r}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if not isinstance(dataset, Sized):
            raise TypeError("WeightedDistributedSampler requires a sized dataset")
        self._dataset_size = len(dataset)

        # Normalize weights
        if weights is None:
            self.weights = [1.0 / num_replicas] * num_replicas
        else:
            self.weights = _normalize_weights(weights, num_replicas)

        self._recompute_counts()

    def _recompute_counts(self) -> None:
        """Recompute sample counts from weights."""
        total_size = self._dataset_size
        self._sample_counts = self._compute_sample_counts(total_size)
        self.num_samples = self._sample_counts[self.rank]
        self.total_size = sum(self._sample_counts)

    def _compute_sample_counts(self, total_size: int) -> list[int]:
        """Compute number of samples for each rank based on weights.

        With drop_last=True, every rank gets floor(total * weight) so the
        per-rank count is deterministic across ranks; the remainder is
        dropped. With drop_last=False (default), remaining samples go to the
        largest fractional remainders so no sample is skipped and rank order
        does not systematically favor the final node.
        """
        ideal_counts = [total_size * weight for weight in self.weights]
        counts = [int(count) for count in ideal_counts]
        if not self.drop_last:
            remainder = total_size - sum(counts)
            ranked_remainders = sorted(
                range(self.num_replicas),
                key=lambda rank: (-(ideal_counts[rank] - counts[rank]), rank),
            )
            for rank in ranked_remainders[:remainder]:
                counts[rank] += 1
        return counts

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self._dataset_size, generator=g).tolist()
        else:
            indices = list(range(self._dataset_size))

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
        self.weights = _normalize_weights(weights, self.num_replicas)
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
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_size < num_replicas:
            raise ValueError(
                "batch_size must be at least num_replicas so every rank has a sample"
            )
        self.sampler = WeightedDistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            weights=weights,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        # Reuse the sampler's validated, normalized weights so allocation
        # decisions cannot diverge between samples and batches.
        self.batch_size = max(1, int(batch_size * self.sampler.weights[rank]))
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


def _normalize_capacities(
    values: list[int] | list[float],
    *,
    name: str,
    integers_only: bool,
) -> list[float]:
    """Validate non-negative capacities and normalize without overflow."""
    if not values:
        return []

    capacities: list[float] = []
    for value in values:
        valid_type = (
            isinstance(value, int)
            if integers_only
            else isinstance(value, (int, float))
        )
        if isinstance(value, bool) or not valid_type:
            kind = "non-negative integers" if integers_only else "finite non-negative numbers"
            raise ValueError(f"{name} must contain {kind}")
        try:
            capacity = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite non-negative numbers") from exc
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"{name} must contain finite non-negative numbers")
        capacities.append(capacity)

    maximum = max(capacities)
    if maximum == 0:
        return [1.0 / len(capacities)] * len(capacities)

    scaled = [capacity / maximum for capacity in capacities]
    total = sum(scaled)
    return [capacity / total for capacity in scaled]


def compute_weights_from_gpu_cores(gpu_cores: list[int]) -> list[float]:
    """Compute workload weights proportional to GPU core counts."""
    return _normalize_capacities(
        gpu_cores,
        name="gpu_cores",
        integers_only=True,
    )


def compute_weights_from_throughput(throughputs: list[float]) -> list[float]:
    """Compute workload weights proportional to measured throughput."""
    return _normalize_capacities(
        throughputs,
        name="throughputs",
        integers_only=False,
    )
