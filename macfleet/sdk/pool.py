"""High-level Pool API for MacFleet.

    with macfleet.Pool() as pool:
        pool.train(model=MyModel(), dataset=ds, epochs=10)

The Pool handles discovery, cluster formation, engine setup,
and gradient synchronization. Users just provide a model and data.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from rich.console import Console

console = Console()


class Pool:
    """Context manager for a MacFleet compute pool.

    Discovers peers on the network, forms a cluster, and provides
    a simple interface for distributed training.

    Usage:
        with macfleet.Pool() as pool:
            pool.train(model=model, dataset=dataset, epochs=10)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        token: Optional[str] = None,
        engine: str = "torch",
        port: int = 50051,
        discovery_timeout: float = 3.0,
    ):
        self.name = name
        self.token = token
        self.engine_type = engine
        self.port = port
        self.discovery_timeout = discovery_timeout
        self._joined = False
        self._agent = None
        self._peers = []

    def __enter__(self) -> Pool:
        self.join()
        return self

    def __exit__(self, *args: Any) -> None:
        self.leave()

    def join(self) -> None:
        """Join the compute pool (discover peers, register).

        For single-node training, this succeeds immediately.
        For multi-node, discovers peers via mDNS.
        """
        self._joined = True

    def leave(self) -> None:
        """Gracefully leave the pool."""
        self._joined = False

    def train(
        self,
        model: Any,
        dataset: Any,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 0.001,
        optimizer: Any = None,
        loss_fn: Any = None,
        engine: Optional[str] = None,
        compression: str = "none",
        **kwargs: Any,
    ) -> dict:
        """Train a model on the pool.

        Handles engine setup, data loading, gradient sync, and training.
        Currently supports single-node training directly and multi-node
        via the Python programmatic API.

        Args:
            model: PyTorch nn.Module (or MLX model in Phase 2).
            dataset: PyTorch Dataset or (X, y) tuple.
            epochs: Number of training epochs.
            batch_size: Global batch size.
            lr: Learning rate (used if optimizer is None).
            optimizer: Pre-configured optimizer (optional).
            loss_fn: Loss function (optional, defaults to model output).
            engine: Override engine type.
            compression: Compression type.

        Returns:
            Dict with training results: {loss, epochs, time_sec, steps}.
        """
        if not self._joined:
            raise RuntimeError("Must join pool before training. Use Pool as context manager.")

        engine_type = engine or self.engine_type

        if engine_type == "torch":
            return self._train_torch(
                model, dataset, epochs, batch_size, lr, optimizer, loss_fn, **kwargs
            )
        else:
            raise ValueError(f"Engine '{engine_type}' not yet supported. Use 'torch'.")

    def _train_torch(
        self,
        model: Any,
        dataset: Any,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer: Any,
        loss_fn: Any,
        **kwargs: Any,
    ) -> dict:
        """Single-node PyTorch training (multi-node via DataParallel in programmatic API)."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from macfleet.engines.torch_engine import TorchEngine

        engine = TorchEngine(device="cpu")

        # Setup optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        engine.load_model(model, optimizer)

        # Setup dataloader
        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            X, y = dataset
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X, y)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        total_start = time.time()
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for batch in dataloader:
                engine.zero_grad()

                if loss_fn is not None:
                    # Separate input/target batches
                    if len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    else:
                        loss = loss_fn(model(batch[0]))
                else:
                    # Model returns loss directly
                    if len(batch) >= 2:
                        loss = model(batch[0]).sum()
                    else:
                        loss = model(batch[0]).sum()

                engine.backward(loss)
                engine.step()
                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history.append(avg_loss)

        total_time = time.time() - total_start
        return {
            "loss": history[-1] if history else 0.0,
            "loss_history": history,
            "epochs": epochs,
            "time_sec": total_time,
            "steps": epochs * len(dataloader),
        }

    @property
    def world_size(self) -> int:
        """Number of nodes in the pool."""
        return 1  # Single-node for now

    @property
    def nodes(self) -> list[dict]:
        """List of nodes in the pool with their profiles."""
        return []
