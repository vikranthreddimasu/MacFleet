"""Convenience function for distributed training.

    macfleet.train(model=MyModel(), dataset=ds, epochs=10)
"""

from typing import Any


def train(
    model: Any,
    dataset: Any,
    epochs: int = 10,
    batch_size: int = 128,
    engine: str = "torch",
    compression: str = "adaptive",
    **kwargs: Any,
) -> dict:
    """Train a model on the MacFleet pool.

    Convenience wrapper that creates a Pool, joins, and trains.

    Args:
        model: PyTorch nn.Module or MLX model.
        dataset: Training dataset.
        epochs: Number of training epochs.
        batch_size: Global batch size.
        engine: Engine type ("torch" or "mlx").
        compression: Compression type.

    Returns:
        Dict with training results: {loss, loss_history, epochs, time_sec, steps}.
    """
    from macfleet.sdk.pool import Pool

    with Pool(engine=engine) as pool:
        return pool.train(
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            compression=compression,
            **kwargs,
        )
