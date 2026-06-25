"""High-level Pool API for MacFleet.

    with macfleet.Pool() as pool:
        pool.train(model=MyModel(), dataset=ds, epochs=10)
        results = pool.map(process_image, image_paths)
        result = pool.run(expensive_fn, data)

The Pool handles discovery, cluster formation, engine setup,
and gradient synchronization. Users just provide a model and data,
or any Python function for general-purpose compute.

Distributed training is SPMD: run the SAME script on every Mac. Each
Pool joins the fleet, waits for quorum, and `pool.train(...)` forms a
training mesh over the data ports, broadcasts rank 0's initial weights,
and allreduces gradients every step:

    # identical script on each Mac
    with macfleet.Pool(enable_pool_distributed=True, quorum_size=2) as pool:
        pool.train(model=MyModel(), dataset=(X, y), epochs=10)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

if TYPE_CHECKING:
    from macfleet.pool.agent import PoolAgent

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def _dataset_len(dataset: Any) -> int:
    """Return the number of samples in a dataset, handling common shapes.

    Supports:
        - Objects with __len__ (PyTorch Dataset, lists, etc.)
        - (X, y) tuples where both halves have a .shape attribute
          (numpy array, torch tensor, mlx array, pandas DataFrame).
        - Anything else: raises TypeError so the caller can skip the guard

    Both `(X, y)` tuples and `[X, y]` lists are accepted as paired
    inputs as long as both halves expose a matching .shape[0]. This
    matches what Pool._train_torch / Pool._train_mlx accept downstream.
    A plain non-paired list (no .shape on the elements, or mismatched
    leading dims) falls through to len() like any other sized object.

    v2.2 PR 9 (A4): used by Pool.train's preflight guard.
    """
    if (
        isinstance(dataset, (tuple, list))
        and len(dataset) == 2
        and hasattr(dataset[0], "shape")
        and hasattr(dataset[1], "shape")
    ):
        x = dataset[0]
        y = dataset[1]
        n_x = x.shape[0] if hasattr(x.shape, "__len__") and len(x.shape) > 0 else None
        n_y = y.shape[0] if hasattr(y.shape, "__len__") and len(y.shape) > 0 else None
        if n_x is not None and n_y is not None and int(n_x) == int(n_y):
            return int(n_x)
    if hasattr(dataset, "__len__"):
        return len(dataset)
    raise TypeError(
        f"Cannot determine size of dataset {type(dataset).__name__}; "
        f"provide a Dataset with __len__ or an (X, y) tuple of arrays."
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


# --------------------------------------------------------------------------- #
# Distributed training runners (module-level: testable without a Pool/agent)  #
# --------------------------------------------------------------------------- #


async def _distributed_train_torch(
    *,
    local_id: str,
    nodes: list,
    security: Any = None,
    local_hw: Any = None,
    model: Any,
    dataset: Any,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: Any = None,
    loss_fn: Any = None,
    device: str = "auto",
    compression: str = "none",
    link_type: Any = None,
    rendezvous_timeout_sec: float = 60.0,
    seed: int = 0,
) -> dict:
    """N-node data-parallel PyTorch training over a freshly formed mesh.

    SPMD: every node calls this with the SAME `nodes` list and its own
    `local_id`. Ranks come from sorted node_id order (see mesh.derive_ranks).

    `batch_size` is the GLOBAL batch size — each rank processes
    `batch_size // world_size` samples per step, and gradient averaging
    makes the effective update equivalent to one global batch.

    Sharding uses equal weights with drop_last=True so every rank runs an
    IDENTICAL number of allreduce steps per epoch — mismatched step counts
    would strand the last allreduce until the recv timeout.
    """
    import hashlib

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from macfleet.engines.torch_engine import TorchEngine
    from macfleet.pool.network import LinkType
    from macfleet.training.data_parallel import DataParallel, DataParallelConfig
    from macfleet.training.mesh import form_mesh
    from macfleet.training.sampler import WeightedDistributedSampler

    mesh = await form_mesh(
        local_id,
        nodes,
        security=security,
        local_hw=local_hw,
        rendezvous_timeout_sec=rendezvous_timeout_sec,
    )
    try:
        engine = TorchEngine(device=device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        engine.load_model(model, optimizer)
        dev = engine.device

        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            X, y = dataset
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X, y)

        per_rank_batch = max(1, batch_size // mesh.world_size)
        # drop_last=True → floor(N/world) samples on EVERY rank, so the
        # per-epoch step count is identical across the fleet.
        sampler = WeightedDistributedSampler(
            dataset,
            num_replicas=mesh.world_size,
            rank=mesh.rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        dataloader = DataLoader(dataset, batch_size=per_rank_batch, sampler=sampler)

        dp = DataParallel(
            engine,
            mesh.group,
            config=DataParallelConfig(compression=compression),
            link_type=link_type if link_type is not None else LinkType.UNKNOWN,
        )
        await dp.setup()  # validates architecture, broadcasts rank 0's weights

        total_start = time.time()
        history: list[float] = []

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_loss = 0.0
            steps = 0
            for batch in dataloader:
                engine.zero_grad()

                if loss_fn is not None:
                    if len(batch) >= 2:
                        inputs, targets = batch[0].to(dev), batch[1].to(dev)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    else:
                        loss = loss_fn(model(batch[0].to(dev)))
                else:
                    loss = model(batch[0].to(dev)).sum()

                engine.backward(loss)
                await dp.sync_gradients()
                engine.step()
                epoch_loss += loss.item()
                steps += 1

            history.append(epoch_loss / max(steps, 1))

        total_time = time.time() - total_start
        # SHA of final flat params: identical across ranks iff sync held.
        params_sha = hashlib.sha256(
            engine.get_flat_parameters().tobytes()
        ).hexdigest()

        return {
            "loss": history[-1] if history else 0.0,
            "loss_history": history,
            "epochs": epochs,
            "time_sec": total_time,
            "steps": epochs * len(dataloader),
            "rank": mesh.rank,
            "world_size": mesh.world_size,
            "avg_sync_time_sec": dp.avg_sync_time_sec,
            "degraded": dp.degraded,
            "unsynced_steps": dp.unsynced_steps,
            "validation_fallback_steps": dp.validation_fallback_steps,
            "last_sync_error": dp.last_sync_error,
            "params_sha256": params_sha,
        }
    finally:
        await mesh.close()


async def _distributed_train_mlx(
    *,
    local_id: str,
    nodes: list,
    security: Any = None,
    local_hw: Any = None,
    model: Any,
    dataset: Any,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: Any = None,
    loss_fn: Any = None,
    compression: str = "none",
    link_type: Any = None,
    rendezvous_timeout_sec: float = 60.0,
    seed: int = 0,
) -> dict:
    """N-node data-parallel MLX training over a freshly formed mesh.

    Same SPMD/sharding contract as _distributed_train_torch: equal
    floor(N/world) shards per rank so step counts match exactly.
    """
    import hashlib

    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np

    from macfleet.engines.mlx_engine import MLXEngine
    from macfleet.pool.network import LinkType
    from macfleet.training.data_parallel import DataParallel, DataParallelConfig
    from macfleet.training.mesh import form_mesh

    if not (isinstance(dataset, (tuple, list)) and len(dataset) == 2):
        raise ValueError("MLX training expects dataset as (X, y) tuple")

    mesh = await form_mesh(
        local_id,
        nodes,
        security=security,
        local_hw=local_hw,
        rendezvous_timeout_sec=rendezvous_timeout_sec,
    )
    try:
        engine = MLXEngine()
        if optimizer is None:
            optimizer = optim.Adam(learning_rate=lr)
        engine.load_model(model, optimizer, loss_fn=loss_fn)

        X, y = dataset
        if not isinstance(X, mx.array):
            X = mx.array(X if not hasattr(X, "numpy") else X.numpy(), dtype=mx.float32)
        if not isinstance(y, mx.array):
            y = mx.array(y if not hasattr(y, "numpy") else y.numpy(), dtype=mx.int32)

        n_samples = X.shape[0]
        shard = n_samples // mesh.world_size  # equal on every rank
        per_rank_batch = max(1, batch_size // mesh.world_size)

        dp = DataParallel(
            engine,
            mesh.group,
            config=DataParallelConfig(compression=compression),
            link_type=link_type if link_type is not None else LinkType.UNKNOWN,
        )
        await dp.setup()

        total_start = time.time()
        history: list[float] = []

        for epoch in range(epochs):
            # Same permutation on every rank (seeded identically), then
            # disjoint contiguous slices — the numpy analogue of
            # WeightedDistributedSampler with drop_last=True.
            rng = np.random.default_rng(seed + epoch)
            perm = rng.permutation(n_samples)
            my_idx = perm[mesh.rank * shard : (mesh.rank + 1) * shard]

            epoch_loss = 0.0
            steps = 0
            for i in range(0, shard, per_rank_batch):
                batch_idx = my_idx[i : i + per_rank_batch].tolist()
                bx = X[batch_idx]
                by = y[batch_idx]

                engine.zero_grad()
                loss = engine.forward((bx, by))
                engine.backward(loss)
                await dp.sync_gradients()
                engine.step()

                epoch_loss += float(loss)
                steps += 1

            history.append(epoch_loss / max(steps, 1))

        total_time = time.time() - total_start
        steps_per_epoch = (shard + per_rank_batch - 1) // per_rank_batch
        params_sha = hashlib.sha256(
            engine.get_flat_parameters().tobytes()
        ).hexdigest()

        return {
            "loss": history[-1] if history else 0.0,
            "loss_history": history,
            "epochs": epochs,
            "time_sec": total_time,
            "steps": epochs * steps_per_epoch,
            "rank": mesh.rank,
            "world_size": mesh.world_size,
            "avg_sync_time_sec": dp.avg_sync_time_sec,
            "degraded": dp.degraded,
            "unsynced_steps": dp.unsynced_steps,
            "validation_fallback_steps": dp.validation_fallback_steps,
            "last_sync_error": dp.last_sync_error,
            "params_sha256": params_sha,
        }
    finally:
        await mesh.close()


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
        # Budget for all peers to reach pool.train() and connect their
        # data-plane transports (SPMD scripts are started by hand on each
        # Mac, so allow a human-scale delay between starts).
        rendezvous_timeout_sec: float = 60.0,
        allow_legacy_pickle: bool = False,
    ):
        from macfleet.security.auth import resolve_token_with_file

        self.name = name
        if open and token is not None:
            raise ValueError(
                "Pool(open=True) disables authentication and cannot be combined "
                "with an explicit token. Remove open=True to join a secure fleet, "
                "or remove token to create an unauthenticated local/open pool."
            )
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
        self.rendezvous_timeout_sec = rendezvous_timeout_sec
        self.allow_legacy_pickle = allow_legacy_pickle
        self._manual_peers = peers or []
        self._joined = False
        self._agent: Optional[PoolAgent] = None
        self._peers: list[str] = []

        # Background event loop — keeps the async PoolAgent alive across
        # sync Pool method calls. Started lazily in join() when the
        # distributed feature flag is set.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def __enter__(self) -> Pool:
        self.join()
        return self

    def __exit__(
        self,
        exc_type: Any = None,
        exc_val: Any = None,
        exc_tb: Any = None,
    ) -> None:
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

        self._teardown_loop()
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

        new_loop = asyncio.new_event_loop()
        self._loop = new_loop
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(new_loop)
            ready.set()
            new_loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop, name="macfleet-pool-loop", daemon=True,
        )
        self._loop_thread.start()
        if not ready.wait(timeout=2.0):
            # Worker thread never signaled ready — clean up before raising.
            self._teardown_loop()
            raise RuntimeError(
                "Pool background event loop failed to start within 2s."
            )

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
        try:
            start_fut = asyncio.run_coroutine_threadsafe(
                self._agent.start(), self._loop,
            )
            agent_start_timeout = max(10.0, self.quorum_timeout_sec)
            start_fut.result(timeout=agent_start_timeout)
        except BaseException:
            # agent.start() can raise (mDNS bind failure, port conflict).
            # Tear down the orphaned loop+thread before re-raising so a
            # subsequent join() doesn't stack a fresh loop on top.
            self._agent = None
            self._teardown_loop()
            raise

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
        self._agent = None
        self._teardown_loop()

        raise TimeoutError(
            f"No quorum within {self.quorum_timeout_sec}s: saw {observed} "
            f"node(s), need {self.quorum_size}. "
            f"Run 'macfleet status' to check discovery, or pass "
            f"peers=['<ip>:{self.port}'] to connect manually."
        )

    def _teardown_loop(self) -> None:
        """Stop the background event loop and join its thread."""
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
            # Release the selector/epoll fd (stop() alone leaks it), but only
            # once the loop has actually stopped — close() raises
            # "Cannot close a running event loop" if the join above timed out
            # while the loop was still draining tasks.
            if not self._loop.is_running() and not self._loop.is_closed():
                try:
                    self._loop.close()
                except RuntimeError:
                    pass
        self._loop = None
        self._loop_thread = None

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
        device: str = "auto",
        distributed: Optional[bool] = None,
        **kwargs: Any,
    ) -> dict:
        """Train a model on the pool.

        Handles engine setup, data loading, and the training loop. When the
        pool is distributed with live peers, training is automatically
        multi-node data-parallel: every Mac runs the SAME script, the pool
        forms a gradient mesh over the data ports, rank 0 broadcasts initial
        weights, and gradients are allreduced every step. Otherwise training
        runs single-node on this Mac's best device.

        Args:
            model: PyTorch nn.Module (or MLX model).
            dataset: PyTorch Dataset or (X, y) tuple. (MLX requires (X, y).)
            epochs: Number of training epochs.
            batch_size: GLOBAL batch size. In distributed mode each rank
                processes batch_size // world_size samples per step.
            lr: Learning rate (used if optimizer is None).
            optimizer: Pre-configured optimizer (optional).
            loss_fn: Loss function (optional, defaults to model output).
            engine: Override engine type.
            compression: Gradient compression for the distributed path —
                "none", "light", "moderate", "aggressive", "adaptive".
                Ignored by single-node training.
            device: Torch device ("auto", "mps", "cpu"). "auto" picks the
                Apple Silicon GPU (MPS) when available. Ignored by the MLX
                engine (unified memory).
            distributed: Force the training mode. None (default) = auto:
                multi-node when the pool is distributed with peers, else
                single-node. True = require multi-node (raises if the pool
                has no peers). False = force single-node even with peers.

        Returns:
            Dict with training results:
            ``{loss, loss_history, epochs, time_sec, steps}``, plus
            ``{rank, world_size, avg_sync_time_sec, params_sha256}`` in
            distributed mode (params_sha256 matches across ranks iff the
            fleet stayed in sync).
        """
        if not self._joined:
            raise RuntimeError("Must join pool before training. Use Pool as context manager.")

        engine_type = engine or self.engine_type
        if engine_type not in ("torch", "mlx"):
            raise ValueError(
                f"Engine '{engine_type}' not supported. Use 'torch' or 'mlx'."
            )
        from macfleet.training.guards import check_training_options

        compression = check_training_options(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            compression=compression,
        )

        has_peers = self.is_distributed and self.world_size > 1
        if distributed is None:
            distributed = has_peers
        elif distributed and not has_peers:
            raise RuntimeError(
                "distributed=True but this pool has no live peers. Construct "
                "Pool(enable_pool_distributed=True, quorum_size=N) and run the "
                "same training script on every Mac (see docs/guides/train.md), "
                "or drop distributed=True to train single-node."
            )

        if not distributed and compression not in (None, "none"):
            console.print(
                f"[yellow]Note:[/yellow] compression='{compression}' is ignored "
                "by single-node training; it only applies to the multi-node "
                "data-parallel path."
            )

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
                world_size=self.world_size if distributed else 1,
            )

        if distributed:
            return self._train_distributed(
                model, dataset, epochs, batch_size, lr, optimizer, loss_fn,
                engine_type=engine_type, compression=compression,
                device=device, **kwargs,
            )

        if engine_type == "torch":
            return self._train_torch(
                model, dataset, epochs, batch_size, lr, optimizer, loss_fn,
                device=device, **kwargs
            )
        return self._train_mlx(
            model, dataset, epochs, batch_size, lr, optimizer, loss_fn, **kwargs
        )

    def _train_distributed(
        self,
        model: Any,
        dataset: Any,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer: Any,
        loss_fn: Any,
        engine_type: str,
        compression: str,
        device: str = "auto",
        **kwargs: Any,
    ) -> dict:
        """Multi-node data-parallel training across the live pool members.

        Snapshots the agent's registry, derives the mesh spec, and runs the
        async training runner on a fresh event loop in THIS thread. The
        agent's background loop keeps gossiping heartbeats untouched —
        training compute must never starve liveness detection.
        """
        from macfleet.pool.network import LinkType, get_network_topology
        from macfleet.training.mesh import NodeSpec, derive_ranks

        agent = self._agent
        if agent is None or agent.registry is None:
            raise RuntimeError("Distributed training requires a running pool agent.")

        records = agent.registry.alive_nodes
        nodes = [
            NodeSpec(node_id=r.node_id, ip_address=r.ip_address, data_port=r.data_port)
            for r in records
        ]
        local_id = agent.node_id
        if local_id not in {n.node_id for n in nodes}:
            raise RuntimeError(
                "Local node missing from its own registry snapshot — "
                "agent is mid-shutdown? Re-join the pool and retry."
            )

        try:
            best = get_network_topology().best_link
            link_type = best.link_type if best else LinkType.UNKNOWN
        except Exception:
            link_type = LinkType.UNKNOWN

        rank = derive_ranks(nodes)[local_id]
        console.print(
            f"[bold blue]MacFleet[/bold blue] distributed training: "
            f"{len(nodes)} nodes, this Mac is rank {rank} "
            f"(engine={engine_type}, compression={compression})"
        )

        common: dict[str, Any] = dict(
            local_id=local_id,
            nodes=nodes,
            security=agent._security,
            local_hw=agent._local_hw_exchange(),
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            loss_fn=loss_fn,
            compression=compression,
            link_type=link_type,
            rendezvous_timeout_sec=self.rendezvous_timeout_sec,
        )
        if engine_type == "torch":
            return asyncio.run(_distributed_train_torch(device=device, **common))
        return asyncio.run(_distributed_train_mlx(**common))

    def _train_torch(
        self,
        model: Any,
        dataset: Any,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer: Any,
        loss_fn: Any,
        device: str = "auto",
        **kwargs: Any,
    ) -> dict:
        """Single-node PyTorch training (multi-node via DataParallel in programmatic API)."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from macfleet.engines.torch_engine import TorchEngine

        engine = TorchEngine(device=device)
        dev = engine.device

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
                        inputs, targets = batch[0].to(dev), batch[1].to(dev)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    else:
                        loss = loss_fn(model(batch[0].to(dev)))
                else:
                    # Model returns loss directly
                    loss = model(batch[0].to(dev)).sum()

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

        The default path requires `fn` to be decorated with @macfleet.task.
        Registered tasks route through the task registry (name + msgpack
        args, no cloudpickle). For migration-only local scripts, construct
        `Pool(..., allow_legacy_pickle=True)` to opt into the legacy
        ProcessPoolExecutor + cloudpickle fallback.

        Args:
            fn: Function to apply to each item. Prefer @macfleet.task.
            iterable: Items to process.
            timeout: Per-task timeout in seconds.
            max_workers: Max parallel workers.

        Returns:
            List of results in the same order as input.

        Usage:
            @macfleet.task
            def process(img): ...

            with macfleet.Pool() as pool:
                results = pool.map(process, image_paths)
        """
        if not self._joined:
            raise RuntimeError("Must join pool before compute. Use Pool as context manager.")

        items = list(iterable)
        if not items:
            return []

        if self._is_registered_task(fn):
            return [self._run_registered_task(fn, item, timeout=timeout) for item in items]

        self._ensure_legacy_pickle_allowed("map")

        # Legacy cloudpickle fallback (local-only, explicit opt-in).
        import os
        import warnings

        import cloudpickle

        from macfleet.security.audit import audit_event

        audit_event("compute.legacy_pickle_used", method="map")
        warnings.warn(
            "Pool.map is using the local-only cloudpickle/ProcessPool path. "
            "Decorate the callable with @macfleet.task before using a fleet "
            "or untrusted code.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        @macfleet.task-decorated functions route through the registry
        (safe, msgpack-native). Undecorated functions are rejected by
        default; construct `Pool(..., allow_legacy_pickle=True)` only for
        migration-only local scripts.

        Args:
            fn: Function to execute. Prefer @macfleet.task.
            *args: Positional arguments for fn.
            timeout: Timeout in seconds.
            **kwargs: Keyword arguments for fn.

        Returns:
            The function's return value.

        Usage:
            @macfleet.task
            def analyze(data): ...

            with macfleet.Pool() as pool:
                result = pool.submit(analyze, data)
        """
        if not self._joined:
            raise RuntimeError("Must join pool before compute. Use Pool as context manager.")

        if self._is_registered_task(fn):
            return self._run_registered_task(fn, *args, timeout=timeout, **kwargs)

        self._ensure_legacy_pickle_allowed("submit")

        # Legacy cloudpickle fallback (local-only, explicit opt-in).
        import warnings

        import cloudpickle

        from macfleet.security.audit import audit_event

        audit_event("compute.legacy_pickle_used", method="submit")
        warnings.warn(
            "Pool.submit is using the local-only cloudpickle/ProcessPool path. "
            "Decorate the callable with @macfleet.task before using a fleet "
            "or untrusted code.",
            DeprecationWarning,
            stacklevel=2,
        )
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_pickled,
                cloudpickle.dumps(fn),
                cloudpickle.dumps(args),
                cloudpickle.dumps(kwargs),
            )
            return future.result(timeout=timeout)

    @staticmethod
    def _is_registered_task(fn: Any) -> bool:
        """True iff `fn` was decorated with @macfleet.task."""
        return callable(fn) and hasattr(fn, "task_name")

    def _ensure_legacy_pickle_allowed(self, method: str) -> None:
        """Reject unsafe dynamic function execution unless explicitly enabled."""
        if self.allow_legacy_pickle:
            return
        raise ValueError(
            f"Pool.{method} requires a function decorated with @macfleet.task. "
            "Registered tasks are validated by name and encoded with msgpack; "
            "undecorated functions require Python pickle execution. Decorate "
            "the callable, or construct Pool(..., allow_legacy_pickle=True) "
            "for local-only migration code you fully trust."
        )

    def _run_registered_task(
        self, fn: Any, *args: Any, timeout: float = 300.0, **kwargs: Any,
    ) -> Any:
        """Execute a registered task by name, validating args via Pydantic schema.

        This is the secure path (no cloudpickle). For distributed mode
        with live peers we'd go through TaskDispatcher; for now we invoke
        the registered callable locally by name so Pool.submit/map stay
        functional in single-node setups without a peer mesh.

        The wire encoding happens regardless: TaskSpec.from_call validates
        args/kwargs against the Pydantic schema (if declared), then
        serializes to msgpack. We decode and invoke locally. This keeps
        the invocation shape identical to what a future distributed path
        would see.
        """
        from macfleet.compute.models import TaskSpec

        spec = TaskSpec.from_call(fn, args=args, kwargs=kwargs, timeout=timeout)
        entry = spec.resolve()
        resolved_args, resolved_kwargs = spec.validated_args(entry)
        # Invoke the registered callable in-process. A future PR wires
        # this to TaskDispatcher when pool.world_size > 1.
        return entry.fn(*resolved_args, **resolved_kwargs)

    def run(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run a function on the pool. Shorthand for submit().

        Usage:
            with macfleet.Pool() as pool:
                result = pool.run(analyze, dataset)
        """
        return self.submit(fn, *args, **kwargs)

    def dashboard_snapshot(self) -> list:
        """Return a list of NodeHealth snapshots for the current pool state.

        v2.2 PR 11 (E2): for callers that want to drive the Rich TUI
        Dashboard themselves, or for headless health checks (e.g.
        `macfleet status --json`). Returns [] if the pool is not running
        in distributed mode (no agent to snapshot).

        Example — run your own dashboard loop:

            from macfleet.monitoring.dashboard import Dashboard

            with macfleet.Pool(enable_pool_distributed=True) as pool:
                with Dashboard() as dash:
                    while training:
                        dash.update_nodes(pool.dashboard_snapshot())
                        time.sleep(2.0)
        """
        if self._agent is None:
            return []
        from macfleet.monitoring.agent_adapter import snapshot_all
        return snapshot_all(self._agent)

    @property
    def is_distributed(self) -> bool:
        """True iff the pool is running in distributed mode with a live agent.

        v2.2 PR 10 (Issue 25): callers that want to branch on "am I running
        solo or across the fleet?" should check this instead of `world_size > 1`
        directly, because world_size is 1 in both solo mode AND a distributed
        pool with no peers yet. This property captures intent.
        """
        return (
            self.enable_pool_distributed
            and self._agent is not None
            and self._agent.registry is not None
        )

    @property
    def world_size(self) -> int:
        """Number of alive nodes in the pool (including self).

        v2.2 PR 8 (Issue 1a): reads from the agent's ClusterRegistry when
        `enable_pool_distributed=True`. Returns 1 for the legacy single-node
        path so existing Pool().train() code keeps working.

        WARNING: world_size == 1 is ambiguous between solo mode and a
        distributed pool with no peers yet (or all peers transiently
        dropped). Use `pool.is_distributed` to disambiguate when the
        distinction matters.
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
