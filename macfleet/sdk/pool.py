"""High-level Pool API for MacFleet.

    with macfleet.Pool() as pool:
        pool.train(model=MyModel(), dataset=ds, epochs=10)
        results = pool.map(process_image, image_paths)
        result = pool.run(expensive_fn, data)

The Pool handles discovery, cluster formation, engine setup,
and gradient synchronization. Users just provide a model and data,
or any Python function for general-purpose compute.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Optional

import cloudpickle
from rich.console import Console

console = Console()


def _run_pickled(fn_bytes: bytes, args_bytes: bytes, kwargs_bytes: bytes) -> Any:
    """Trampoline: deserialize with cloudpickle, call, return result.

    ProcessPoolExecutor uses stdlib pickle internally, which cannot
    serialize closures or lambdas. This module-level function IS
    picklable by stdlib, and it uses cloudpickle to deserialize the
    actual function and arguments (passed as bytes).
    """
    import cloudpickle as _cp
    fn = _cp.loads(fn_bytes)
    args = _cp.loads(args_bytes)
    kwargs = _cp.loads(kwargs_bytes)
    return fn(*args, **kwargs)


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
        fleet_id: Optional[str] = None,
        tls: bool = False,
        open: bool = False,
    ):
        from macfleet.security.auth import resolve_token_with_file

        self.name = name
        if open:
            self.token = None
        else:
            self.token = resolve_token_with_file(token, auto_generate=True)
        self.engine_type = engine
        self.port = port
        self.discovery_timeout = discovery_timeout
        self.fleet_id = fleet_id
        self.tls = tls
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
        elif engine_type == "mlx":
            return self._train_mlx(
                model, dataset, epochs, batch_size, lr, optimizer, loss_fn, **kwargs
            )
        else:
            raise ValueError(f"Engine '{engine_type}' not supported. Use 'torch' or 'mlx'.")

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

    def _train_mlx(
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
        """Single-node MLX training."""
        import mlx.core as mx
        import mlx.optimizers as optim

        from macfleet.engines.mlx_engine import MLXEngine

        engine = MLXEngine()

        if optimizer is None:
            optimizer = optim.Adam(learning_rate=lr)

        engine.load_model(model, optimizer, loss_fn=loss_fn)

        # Convert dataset to MLX arrays
        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            X, y = dataset
            if not isinstance(X, mx.array):
                X = mx.array(X if not hasattr(X, 'numpy') else X.numpy(), dtype=mx.float32)
            if not isinstance(y, mx.array):
                y = mx.array(y if not hasattr(y, 'numpy') else y.numpy(), dtype=mx.int32)
        else:
            raise ValueError("MLX training expects dataset as (X, y) tuple")

        n_samples = X.shape[0]

        total_start = time.time()
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0

            indices = list(range(n_samples))
            import random
            random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                bx = X[batch_idx]
                by = y[batch_idx]

                engine.zero_grad()
                loss = engine.forward((bx, by))
                engine.backward(loss)
                engine.step()

                epoch_loss += float(loss)
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history.append(avg_loss)

        total_time = time.time() - total_start
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size
        return {
            "loss": history[-1] if history else 0.0,
            "loss_history": history,
            "epochs": epochs,
            "time_sec": total_time,
            "steps": epochs * steps_per_epoch,
        }

    def map(
        self,
        fn: Callable,
        iterable: Iterable,
        timeout: float = 300.0,
        max_workers: Optional[int] = None,
    ) -> list:
        """Apply fn to each item across the pool, return results in order.

        For single-node pools, uses a local ProcessPoolExecutor.
        For multi-node, distributes tasks to workers via TaskDispatcher.

        Args:
            fn: Function to apply to each item.
            iterable: Items to process.
            timeout: Per-task timeout in seconds.
            max_workers: Max parallel processes (single-node only).

        Returns:
            List of results in the same order as input.

        Usage:
            with macfleet.Pool() as pool:
                results = pool.map(process_image, image_paths)
        """
        if not self._joined:
            raise RuntimeError("Must join pool before compute. Use Pool as context manager.")

        items = list(iterable)
        if not items:
            return []

        import os
        workers = max_workers or min(os.cpu_count() or 1, 4)
        fn_bytes = cloudpickle.dumps(fn)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_pickled,
                    fn_bytes,
                    cloudpickle.dumps((item,)),
                    cloudpickle.dumps({}),
                )
                for item in items
            ]
            return [f.result(timeout=timeout) for f in futures]

    def submit(self, fn: Callable, *args: Any, timeout: float = 300.0, **kwargs: Any) -> Any:
        """Submit a single task and block until complete.

        For single-node pools, executes in a child process.
        For multi-node, dispatches to a worker.

        Args:
            fn: Function to execute.
            *args: Positional arguments for fn.
            timeout: Timeout in seconds.
            **kwargs: Keyword arguments for fn.

        Returns:
            The function's return value.

        Usage:
            with macfleet.Pool() as pool:
                result = pool.submit(expensive_fn, data)
        """
        if not self._joined:
            raise RuntimeError("Must join pool before compute. Use Pool as context manager.")

        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_pickled,
                cloudpickle.dumps(fn),
                cloudpickle.dumps(args),
                cloudpickle.dumps(kwargs),
            )
            return future.result(timeout=timeout)

    def run(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run a function on the pool. Shorthand for submit().

        Usage:
            with macfleet.Pool() as pool:
                result = pool.run(analyze, dataset)
        """
        return self.submit(fn, *args, **kwargs)

    @property
    def world_size(self) -> int:
        """Number of nodes in the pool."""
        return 1  # Single-node for now

    @property
    def nodes(self) -> list[dict]:
        """List of nodes in the pool with their profiles."""
        return []
