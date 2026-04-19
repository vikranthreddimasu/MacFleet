"""Preflight guards for Pool.train.

v2.2 PR 9 (A4 from docs/designs/v3-cathedral.md): catch common
misconfigurations BEFORE the training loop starts, with error messages
that tell the user what to fix.

The big one is empty-or-undersized datasets: with DataParallel, each
rank gets `batch_size // world_size` samples per step. If the dataset
has fewer samples than `batch_size`, the dataloader silently produces
0 steps and the loss is NaN. If it has more than `batch_size` but less
than `batch_size * world_size`, some ranks starve and the allreduce
hangs forever. Both are terrible UX — users blame the framework for
"training not running" when they could have learned in 5 ms.
"""

from __future__ import annotations


class DatasetSizeError(ValueError):
    """Raised when a dataset is too small for the requested batch/world size."""


def check_dataset_sufficient(
    dataset_len: int,
    batch_size: int,
    world_size: int,
    *,
    min_batches: int = 1,
) -> None:
    """Fail fast if the dataset can't produce at least `min_batches` full batches.

    Args:
        dataset_len: Number of samples in the dataset.
        batch_size: Global batch size (summed across ranks).
        world_size: Number of training ranks (>= 1).
        min_batches: Minimum global batches required per epoch. Default 1
            (the least strict check — just "can we take at least one step?").

    Raises:
        DatasetSizeError: With a remediation-rich message naming the
        expected minimum and suggesting concrete fixes.

    The check handles three failure modes distinctly:
        1. empty dataset → tell the user their loader produced no samples
        2. smaller than one global batch → tell them to lower batch_size
           or use more data, with an exact minimum
        3. smaller than one per-rank batch → tell them some ranks would
           starve, with both the global and per-rank minimums
    """
    if dataset_len <= 0:
        raise DatasetSizeError(
            "Dataset is empty. Check that your DataLoader/Dataset produces "
            "samples before calling pool.train()."
        )

    if world_size < 1:
        raise DatasetSizeError(f"world_size must be >= 1, got {world_size}")

    per_rank = batch_size // world_size
    if per_rank < 1:
        raise DatasetSizeError(
            f"batch_size {batch_size} is smaller than world_size {world_size}; "
            f"each rank gets 0 samples per step. Increase batch_size to at "
            f"least {world_size}, or run on fewer nodes."
        )

    required = batch_size * min_batches
    if dataset_len < required:
        shortfall = required - dataset_len
        raise DatasetSizeError(
            f"Dataset has {dataset_len} samples but needs >= {required} to "
            f"run at least {min_batches} batch(es) of size {batch_size} "
            f"across {world_size} rank(s). "
            f"Shortfall: {shortfall} samples. "
            f"Either: (a) collect more data, "
            f"(b) reduce batch_size to {dataset_len // min_batches} or "
            f"smaller, or (c) reduce world_size."
        )
