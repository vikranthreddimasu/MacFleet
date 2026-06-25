"""Weighted distributed sampler for MacFleet.

Extends PyTorch's DistributedSampler to support weighted splitting
based on each node's compute capacity (GPU cores, throughput).
"""

from typing import Iterator, Optional, TypeVar

import torch
from torch.utils.data import Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)


class WeightedDistributedSampler(Sampler[int]):
    """Distributed sampler with weighted batch allocation.

    Unlike PyTorch's DistributedSampler which splits data equally,
    this sampler gives each node a proportion of samples based on
    their workload weight (derived from GPU cores or calibrated throughput).

    Example:
        Pro (16 GPU cores, weight=0.62): gets 62% of samples
        Air (10 GPU cores, weight=0.38): gets 38% of samples
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
        """Initialize the sampler.

        Args:
            dataset: Dataset to sample from.
            num_replicas: Number of nodes (world_size).
            rank: This node's rank.
            weights: Workload weight for each rank. If None, uses equal weights.
                    Must sum to 1.0.
            shuffle: Whether to shuffle indices.
            seed: Random seed for shuffling.
            drop_last: Whether to drop incomplete batch at end.
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Set weights
        if weights is None:
            self.weights = [1.0 / num_replicas] * num_replicas
        else:
            if len(weights) != num_replicas:
                raise ValueError(f"weights length {len(weights)} != num_replicas {num_replicas}")
            total = sum(weights)
            self.weights = [w / total for w in weights]  # Normalize

        # Compute sample counts for each rank
        total_size = len(dataset)
        self._sample_counts = self._compute_sample_counts(total_size)
        self.num_samples = self._sample_counts[rank]
        self.total_size = sum(self._sample_counts)

    def _compute_sample_counts(self, total_size: int) -> list[int]:
        """Compute number of samples for each rank based on weights."""
        counts = []
        remaining = total_size

        for i, weight in enumerate(self.weights):
            if i == len(self.weights) - 1:
                # Last rank gets remainder to ensure all samples used
                count = remaining
            else:
                count = int(total_size * weight)
                remaining -= count
            counts.append(count)

        return counts

    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices for this rank."""
        # Generate shuffled indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Compute start and end for this rank
        start = sum(self._sample_counts[:self.rank])
        end = start + self._sample_counts[self.rank]

        # Get this rank's portion
        rank_indices = indices[start:end]

        return iter(rank_indices)

    def __len__(self) -> int:
        """Number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling reproducibility."""
        self.epoch = epoch

    def set_weights(self, weights: list[float]) -> None:
        """Update workload weights (e.g., after rebalancing)."""
        if len(weights) != self.num_replicas:
            raise ValueError(f"weights length {len(weights)} != num_replicas {self.num_replicas}")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self._sample_counts = self._compute_sample_counts(len(self.dataset))
        self.num_samples = self._sample_counts[self.rank]
        self.total_size = sum(self._sample_counts)


class DistributedBatchSampler(Sampler[list[int]]):
    """Batch sampler that yields weighted batch sizes per rank.

    Instead of fixed batch size, allocates batch samples based on
    node weights. Useful when you want each forward pass to process
    the weighted amount.
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
        """Initialize the batch sampler.

        Args:
            dataset: Dataset to sample from.
            batch_size: Total batch size across all nodes.
            num_replicas: Number of nodes.
            rank: This node's rank.
            weights: Workload weights per rank.
            shuffle: Whether to shuffle.
            seed: Random seed.
            drop_last: Whether to drop incomplete batches.
        """
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
        """Yield batches of indices."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility."""
        self.sampler.set_epoch(epoch)


def compute_weights_from_gpu_cores(gpu_cores: list[int]) -> list[float]:
    """Compute workload weights from GPU core counts.

    Args:
        gpu_cores: List of GPU core counts per rank.

    Returns:
        Normalized weights.
    """
    total = sum(gpu_cores)
    if total == 0:
        return [1.0 / len(gpu_cores)] * len(gpu_cores)
    return [cores / total for cores in gpu_cores]


def compute_weights_from_throughput(throughputs: list[float]) -> list[float]:
    """Compute workload weights from measured throughputs.

    Args:
        throughputs: List of samples/sec per rank.

    Returns:
        Normalized weights.
    """
    total = sum(throughputs)
    if total == 0:
        return [1.0 / len(throughputs)] * len(throughputs)
    return [t / total for t in throughputs]
