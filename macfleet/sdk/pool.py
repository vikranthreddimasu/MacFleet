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

import asyncio
import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Optional

import cloudpickle
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def _dataset_len(dataset: Any) -> int:
    """Return the number of samples in a dataset, handling common shapes.

    Supports:
        - Objects with __len__ (PyTorch Dataset, lists, etc.)
        - (X, y) tuples where X is sized
        - Anything else: raises TypeError so the caller can skip the guard

    v2.2 PR 9 (A4): used by Pool.train's preflight guard.
    """
    if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        # (X, y) pair shape
        inner = dataset[0]
        if hasattr(inner, "__len__"):
            return len(inner)
        if hasattr(inner, "shape") and len(inner.shape) > 0:
            return int(inner.shape[0])
    if hasattr(dataset, "__len__"):
        return len(dataset)
    raise TypeError(
        f"Cannot determine size of dataset {type(dataset).__name__}; "
        f"provide a Dataset with __len__ or an (X, y) tuple."
    )


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
        data_port: Optional[int] = None,
        discovery_timeout: float = 3.0,
        fleet_id: Optional[str] = None,
        tls: bool = False,
        open: bool = False,
        # v2.2 PR 8 (Issue 1a): distributed pool wiring behind a feature flag.
        # With the flag off (default), Pool remains a single-node convenience
        # wrapper and Pool.join is a no-op. Flip to True to instantiate a
        # real PoolAgent that participates in mDNS discovery + heartbeat.
        enable_pool_distributed: bool = False,
        quorum_size: int = 1,
        quorum_timeout_sec: float = 10.0,
        peers: Optional[list[str]] = None,
    ):
        from macfleet.security.auth import resolve_token_with_file

        self.name = name
        if open:
            self.token = None
        else:
            self.token = resolve_token_with_file(token, auto_generate=True)
        self.engine_type = engine
        self.port = port
        self.data_port = data_port
        self.discovery_timeout = discovery_timeout
        self.fleet_id = fleet_id
        self.tls = tls
        self.enable_pool_distributed = enable_pool_distributed
        self.quorum_size = quorum_size
        self.quorum_timeout_sec = quorum_timeout_sec
        self._manual_peers = peers or []
        self._joined = False
        self._agent = None
        self._peers = []

        # Background event loop — keeps the async PoolAgent alive across
        # sync Pool method calls. Started lazily in join() when the
        # distributed feature flag is set.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def __enter__(self) -> Pool:
        self.join()
        return self

    def __exit__(self, *args: Any) -> None:
        self.leave()

    def join(self) -> None:
        """Join the compute pool (discover peers, register).

        Default (feature flag off): no-op. The Pool behaves as a
        single-node convenience wrapper and legacy training paths work
        unchanged.

        With `enable_pool_distributed=True`: instantiates a `PoolAgent`,
        starts mDNS discovery, and blocks until `world_size >=
        quorum_size` (including self) or `quorum_timeout_sec` elapses.
        """
        if self._joined:
            return

        if not self.enable_pool_distributed:
            self._joined = True
            return

        self._start_agent()
        self._joined = True

    def leave(self) -> None:
        """Gracefully leave the pool."""
        if not self._joined:
            return

        if self._agent is not None and self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._agent.stop(), self._loop,
                )
                fut.result(timeout=5.0)
            except Exception as e:
                logger.warning("Pool.leave: agent stop raised %s", e)
            self._agent = None

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5.0)
            self._loop = None
            self._loop_thread = None

        self._joined = False

    def _start_agent(self) -> None:
        """Spin up a background event loop and start a PoolAgent on it.

        PoolAgent is async; Pool exposes a sync API. We own a loop in a
        background thread so the agent's discovery + heartbeat tasks keep
        running between sync Pool calls. This avoids both `asyncio.run`
        (which tears down the loop) and the uvloop integration fragility
        of `nest_asyncio`.
        """
        from macfleet.pool.agent import PoolAgent

        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop, name="macfleet-pool-loop", daemon=True,
        )
        self._loop_thread.start()
        ready.wait(timeout=2.0)

        self._agent = PoolAgent(
            name=self.name,
            port=self.port,
            data_port=self.data_port,
            token=self.token,
            fleet_id=self.fleet_id,
            tls=self.tls,
            peers=self._manual_peers,
        )

        # Start the agent on the background loop. Agent startup includes
        # mDNS registration + heartbeat server bind, which can take a few
        # seconds on its own. We give it a fixed floor instead of borrowing
        # from quorum_timeout_sec (otherwise a tight quorum timeout causes
        # the wrong kind of error).
        start_fut = asyncio.run_coroutine_threadsafe(
            self._agent.start(), self._loop,
        )
        agent_start_timeout = max(10.0, self.quorum_timeout_sec)
        start_fut.result(timeout=agent_start_timeout)

        # Wait for quorum — poll the agent's registry for alive peers
        deadline = time.monotonic() + self.quorum_timeout_sec
        while time.monotonic() < deadline:
            if self._agent.registry is not None:
                if self._agent.registry.world_size >= self.quorum_size:
                    return
            time.sleep(0.1)

        observed = (
            self._agent.registry.world_size if self._agent.registry else 0
        )
        # Stop the agent cleanly before bubbling the error up
        try:
            asyncio.run_coroutine_threadsafe(
                self._agent.stop(), self._loop,
            ).result(timeout=2.0)
        except Exception:
            pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
        self._agent = None
        self._loop = None
        self._loop_thread = None

        raise TimeoutError(
            f"No quorum within {self.quorum_timeout_sec}s: saw {observed} "
            f"node(s), need {self.quorum_size}. "
            f"Run 'macfleet status' to check discovery, or pass "
            f"peers=['<ip>:{self.port}'] to connect manually."
        )

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

        # A4 preflight: reject empty / undersized datasets before we bring up
        # optimizer state + dataloader. Checking here means the user sees a
        # helpful error in microseconds instead of watching training
        # silently produce 0 batches.
        try:
            dataset_len = _dataset_len(dataset)
        except TypeError:
            # Dataset doesn't support len(); skip the guard (e.g. iterable datasets).
            # We can still rely on the DataLoader to catch edge cases downstream.
            pass
        else:
            from macfleet.training.guards import check_dataset_sufficient
            check_dataset_sufficient(
                dataset_len=dataset_len,
                batch_size=batch_size,
                world_size=self.world_size,
            )

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
        """Number of alive nodes in the pool (including self).

        v2.2 PR 8 (Issue 1a): reads from the agent's ClusterRegistry when
        `enable_pool_distributed=True`. Returns 1 for the legacy single-node
        path so existing Pool().train() code keeps working.
        """
        if self._agent is not None and self._agent.registry is not None:
            return self._agent.registry.world_size
        return 1

    @property
    def nodes(self) -> list[dict]:
        """List of alive nodes in the pool with their profiles.

        v2.2 PR 8 (Issue 1a): reads from the agent's ClusterRegistry when
        the distributed flag is on. Legacy single-node mode returns [].
        """
        if self._agent is None or self._agent.registry is None:
            return []
        out = []
        for record in self._agent.registry.alive_nodes:
            hw = record.hardware
            out.append({
                "node_id": record.node_id,
                "hostname": record.hostname,
                "ip_address": record.ip_address,
                "port": record.port,
                "data_port": record.data_port,
                "chip_name": hw.chip_name,
                "gpu_cores": hw.gpu_cores,
                "ram_gb": hw.ram_gb,
                "compute_score": hw.compute_score,
                "is_coordinator": (
                    self._agent.registry.coordinator_id == record.node_id
                ),
            })
        return out
