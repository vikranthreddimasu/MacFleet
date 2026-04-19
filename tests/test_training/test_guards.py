"""Tests for Pool.train preflight guards (A4 from docs/designs/v3-cathedral.md).

The guards fail fast with messages that tell the user *what to fix*,
not just *that something is wrong*. These tests lock in the error-shape
contract so future refactors don't regress the UX.
"""

from __future__ import annotations

import pytest

from macfleet.training.guards import DatasetSizeError, check_dataset_sufficient


class TestCheckDatasetSufficient:
    def test_happy_path(self):
        """Enough samples for one batch → no raise."""
        check_dataset_sufficient(
            dataset_len=128, batch_size=128, world_size=1,
        )

    def test_large_dataset_no_raise(self):
        check_dataset_sufficient(
            dataset_len=10_000, batch_size=128, world_size=4,
        )

    def test_empty_dataset_raises(self):
        with pytest.raises(DatasetSizeError, match="empty"):
            check_dataset_sufficient(
                dataset_len=0, batch_size=32, world_size=1,
            )

    def test_empty_message_mentions_dataloader(self):
        with pytest.raises(DatasetSizeError) as exc:
            check_dataset_sufficient(0, 32, 1)
        assert "DataLoader" in str(exc.value) or "Dataset" in str(exc.value)

    def test_batch_smaller_than_world(self):
        """batch_size // world_size < 1 → some ranks starve."""
        with pytest.raises(DatasetSizeError, match="smaller than world_size"):
            check_dataset_sufficient(
                dataset_len=1000, batch_size=3, world_size=4,
            )

    def test_too_small_for_one_batch(self):
        """Dataset has some samples but < batch_size."""
        with pytest.raises(DatasetSizeError) as exc:
            check_dataset_sufficient(
                dataset_len=50, batch_size=128, world_size=1,
            )
        msg = str(exc.value)
        assert "50 samples" in msg
        assert "128" in msg  # required
        assert "Shortfall: 78" in msg

    def test_error_suggests_reduced_batch_size(self):
        with pytest.raises(DatasetSizeError) as exc:
            check_dataset_sufficient(50, 128, 1)
        assert "reduce batch_size" in str(exc.value).lower()

    def test_error_suggests_more_data(self):
        with pytest.raises(DatasetSizeError) as exc:
            check_dataset_sufficient(50, 128, 1)
        assert "collect more data" in str(exc.value).lower()

    def test_min_batches_enforced(self):
        """Require at least N batches per epoch."""
        # Enough for 1 batch, but we're asking for 10
        with pytest.raises(DatasetSizeError) as exc:
            check_dataset_sufficient(
                dataset_len=200, batch_size=128, world_size=1, min_batches=10,
            )
        assert "10 batch" in str(exc.value)

    def test_world_size_must_be_positive(self):
        with pytest.raises(DatasetSizeError, match="world_size must be >= 1"):
            check_dataset_sufficient(1000, 32, 0)

    def test_exactly_one_batch_sufficient(self):
        """dataset_len == batch_size → exactly 1 batch → OK."""
        check_dataset_sufficient(
            dataset_len=32, batch_size=32, world_size=1,
        )
