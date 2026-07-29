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

import math


class DatasetSizeError(ValueError):
    """Raised when a dataset is too small for the requested batch/world size."""


class TrainingConfigError(ValueError):
    """Raised when training hyperparameters cannot produce a valid run."""


VALID_TRAINING_COMPRESSION = frozenset({"none", "light", "adaptive"})

# Rejected until sparse-on-wire: per-rank TopK before dense allreduce
# biases averaged gradients when ranks see different batches.
_UNSAFE_TOPK_COMPRESSION = frozenset({"moderate", "aggressive"})


def check_training_options(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    compression: str | None,
) -> str:
    """Validate core Pool.train options and return normalized compression.

    This guard runs before engine setup or DataLoader construction, so common
    mistakes fail with direct messages instead of late framework exceptions.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise TrainingConfigError(
            f"epochs must be a positive integer (>= 1), got {epochs!r}."
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise TrainingConfigError(
            f"batch_size must be a positive integer (>= 1), got {batch_size!r}."
        )
    if (
        not isinstance(lr, (int, float))
        or isinstance(lr, bool)
        or not math.isfinite(float(lr))
        or float(lr) <= 0
    ):
        raise TrainingConfigError(
            f"lr must be a positive finite number (> 0), got {lr!r}."
        )

    normalized = "none" if compression is None else compression
    if isinstance(normalized, str) and normalized in _UNSAFE_TOPK_COMPRESSION:
        raise TrainingConfigError(
            f"compression={normalized!r} uses per-rank TopK before allreduce, "
            "which biases averaged gradients until sparse-on-wire ships. "
            "Use 'none', 'light' (FP16), or 'adaptive' (dense-safe FP16/none)."
        )
    if not isinstance(normalized, str) or normalized not in VALID_TRAINING_COMPRESSION:
        valid = ", ".join(sorted(VALID_TRAINING_COMPRESSION))
        raise TrainingConfigError(
            f"compression must be one of: {valid}. Got {compression!r}."
        )
    return normalized


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
    for name, value, minimum in (
        ("dataset_len", dataset_len, 0),
        ("batch_size", batch_size, 1),
        ("world_size", world_size, 1),
        ("min_batches", min_batches, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise DatasetSizeError(f"{name} must be a {qualifier} integer, got {value!r}")

    if dataset_len <= 0:
        raise DatasetSizeError(
            "Dataset is empty. Check that your DataLoader/Dataset produces "
            "samples before calling pool.train()."
        )

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
        max_batch_size = dataset_len // min_batches
        if max_batch_size >= 1:
            remediation = (
                f"Either: (a) collect more data, or "
                f"(b) reduce batch_size to {max_batch_size} or smaller."
            )
        else:
            remediation = (
                f"Either: (a) collect at least {min_batches} samples, or "
                f"(b) lower min_batches; no positive batch_size can produce "
                f"{min_batches} full batch(es) from {dataset_len} sample(s)."
            )
        raise DatasetSizeError(
            f"Dataset has {dataset_len} samples but needs >= {required} to "
            f"run at least {min_batches} batch(es) of size {batch_size} "
            f"across {world_size} rank(s). "
            f"Shortfall: {shortfall} samples. "
            f"{remediation}"
        )
