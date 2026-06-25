"""High-level Trainer API for MacFleet distributed training.

Provides a simple interface for training PyTorch models across
multiple Macs with automatic distributed coordination.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import TensorTransport
from macfleet.compression.pipeline import create_pipeline
from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)
from macfleet.training.data_parallel import MacFleetDDP
from macfleet.training.distributed_sampler import WeightedDistributedSampler
from macfleet.utils.tensor_utils import MessageType

console = Console()


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    samples_per_sec: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    compression_ratio: float = 1.0


@dataclass
class TrainerState:
    """State of the trainer for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_accuracy: float = 0.0


class Trainer:
    """High-level trainer for distributed training with MacFleet.

    Example:
        trainer = Trainer(
            model=model,
            dataset=train_dataset,
            training_config=TrainingConfig(epochs=10),
            cluster_config=ClusterConfig(role=NodeRole.MASTER),
        )
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        training_config: TrainingConfig,
        cluster_config: ClusterConfig,
        optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        criterion: Optional[nn.Module] = None,
        val_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[Callable]] = None,
        distributed: bool = False,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            dataset: Training dataset.
            training_config: Training configuration.
            cluster_config: Cluster configuration.
            optimizer_cls: Optimizer class.
            optimizer_kwargs: Optimizer keyword arguments.
            criterion: Loss function (default: CrossEntropyLoss).
            val_dataset: Optional validation dataset.
            callbacks: Optional list of callback functions.
            distributed: Whether to run distributed training.
        """
        self.model = model
        self.dataset = dataset
        self.training_config = training_config
        self.cluster_config = cluster_config
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": training_config.learning_rate}
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []
        self.distributed = distributed

        # State
        self.state = TrainerState()
        self._metrics = TrainingMetrics()

        # Distributed state
        self._rank = 0
        self._world_size = 1
        self._weights = [1.0]

        # Distributed components (initialized in setup)
        self._transport: Optional[TensorTransport] = None
        self._collective_group: Optional[CollectiveGroup] = None
        self._ddp_model: Optional[MacFleetDDP] = None
        self._optimizer: Optional[Optimizer] = None
        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[WeightedDistributedSampler] = None

        # gRPC handles (for teardown)
        self._grpc_server = None
        self._grpc_client = None

        # Device
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Get the appropriate device."""
        if self.training_config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif self.training_config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def setup(self) -> None:
        """Set up distributed training components."""
        console.print("[bold blue]Setting up training...[/bold blue]")

        # Move model to device
        self.model = self.model.to(self._device)
        console.print(f"  Model moved to {self._device}")

        # Create tensor transport (used for send/recv once connections are stored)
        self._transport = TensorTransport(
            host="0.0.0.0",
            port=self.cluster_config.tensor_port,
        )

        if not self.distributed:
            # Single-node: start background server for general use
            await self._transport.start_server()
        # For distributed: connections are managed explicitly in
        # _setup_distributed_master / _setup_distributed_worker
        console.print(f"  Tensor transport on port {self.cluster_config.tensor_port}")

        # Set up distributed or single-node
        if self.distributed:
            if self.cluster_config.role == NodeRole.MASTER:
                await self._setup_distributed_master()
            else:
                await self._setup_distributed_worker()
        else:
            self._rank = 0
            self._world_size = 1
            self._weights = [1.0]

        # Create collective group
        self._collective_group = CollectiveGroup(
            rank=self._rank,
            world_size=self._world_size,
            transport=self._transport,
        )
        self._collective_group.set_device(self._device)

        # Wire up peer connections on the collective group
        if self.distributed and hasattr(self, '_peer_info'):
            for peer_rank, conn_key in self._peer_info.items():
                self._collective_group._peer_connections[peer_rank] = conn_key

        # Verify connections with test tensor
        if self.distributed and self._world_size > 1:
            await self._verify_connections()

        # Create compression pipeline (None when no compression)
        compression_type = self.training_config.compression.value
        compression = None
        if compression_type != "none":
            compression = create_pipeline(compression_type, self.training_config.topk_ratio)

        # Wrap model in DDP
        self._ddp_model = MacFleetDDP(
            module=self.model,
            collective_group=self._collective_group,
            compression_pipeline=compression,
        )

        # Broadcast initial parameters from rank 0 to ensure all nodes
        # start with identical model weights (critical for correctness)
        if self.distributed and self._world_size > 1:
            console.print("  Broadcasting initial parameters from rank 0...")
            await self._ddp_model.broadcast_parameters()
            console.print("  [green]Parameters synchronized[/green]")

        # Create optimizer
        self._optimizer = self.optimizer_cls(
            self._ddp_model.parameters(),
            **self.optimizer_kwargs,
        )

        # Create sampler and dataloader
        self._sampler = WeightedDistributedSampler(
            dataset=self.dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            weights=self._weights,
            shuffle=True,
        )

        # Compute per-node batch size
        if self._rank < len(self._weights):
            weight = self._weights[self._rank]
        else:
            weight = 1.0 / self._world_size
        local_batch_size = max(1, int(self.training_config.batch_size * weight))

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=local_batch_size,
            sampler=self._sampler,
            num_workers=0,  # MPS doesn't work well with multiprocessing
            pin_memory=False,
        )

        # All nodes must run the same number of AllReduce steps per epoch
        # to keep the TCP protocol synchronized. Use global batch count.
        import math
        self._steps_per_epoch = math.ceil(len(self.dataset) / self.training_config.batch_size)

        console.print(f"  Rank: {self._rank}, World size: {self._world_size}")
        console.print(f"  Local batch size: {local_batch_size}")
        console.print(f"  Samples per epoch: {len(self._sampler)}")
        console.print(f"  Steps per epoch: {self._steps_per_epoch}")
        console.print("[bold green]Setup complete![/bold green]")

    async def _setup_distributed_master(self) -> None:
        """Set up master node for distributed training.

        Starts gRPC server, waits for workers to register, then
        accepts their TCP tensor connections using raw socket accept.
        """
        import socket as _socket

        from macfleet.comm.grpc_service import ClusterControlServicer, GRPCServer
        from macfleet.utils.network import (
            get_gpu_info,
            get_hostname,
            get_local_ip,
            get_memory_bandwidth,
            get_memory_info,
        )

        hostname = get_hostname()
        ip = self.cluster_config.host or get_local_ip()
        gpu_info = get_gpu_info()
        mem_info = get_memory_info()

        cluster_state = ClusterState()

        # Register self as rank 0
        self_node = NodeConfig(
            hostname=hostname,
            ip_address=ip,
            gpu_cores=gpu_info.get("gpu_cores", 10),
            ram_gb=mem_info.get("total_gb", 16),
            memory_bandwidth_gbps=get_memory_bandwidth(),
            tensor_port=self.cluster_config.tensor_port,
            rank=0,
            workload_weight=1.0,
        )
        cluster_state.add_node(self_node)

        # Create raw listening socket for tensor connections.
        # We avoid asyncio.start_server because its handle_client callback
        # doesn't fire reliably when gRPC threads are running.
        loop = asyncio.get_running_loop()
        server_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        server_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        server_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1024 * 1024)
        server_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1024 * 1024)
        server_sock.bind(("0.0.0.0", self.cluster_config.tensor_port))
        server_sock.listen(8)
        server_sock.setblocking(False)
        self._tensor_server_sock = server_sock
        console.print(f"  Tensor server listening on port {self.cluster_config.tensor_port}")

        # Wait for workers via gRPC registration
        registered_workers: list[NodeConfig] = []
        worker_event = asyncio.Event()

        def on_register(node: NodeConfig) -> None:
            registered_workers.append(node)
            # Thread-safe: gRPC callback runs on a thread pool thread
            loop.call_soon_threadsafe(worker_event.set)

        servicer = ClusterControlServicer(
            cluster_state=cluster_state,
            tensor_addr=ip,
            tensor_port=self.cluster_config.tensor_port,
            on_register=on_register,
            auth_token=self.cluster_config.auth_token,
        )

        self._grpc_server = GRPCServer(
            servicer=servicer,
            host="0.0.0.0",
            port=self.cluster_config.master_port,
        )
        self._grpc_server.start()
        console.print(f"  gRPC server started on port {self.cluster_config.master_port}")
        min_workers = self.cluster_config.min_workers
        console.print(f"  [yellow]Waiting for {min_workers} worker(s) to register...[/yellow]")

        # Wait for min_workers to register
        while len(registered_workers) < min_workers:
            worker_event.clear()
            await asyncio.wait_for(worker_event.wait(), timeout=120.0)
        # Brief stabilization period for late workers
        await asyncio.sleep(3.0)

        self._rank = 0
        self._world_size = cluster_state.world_size

        # Build weights sorted by rank
        self._weights = [0.0] * self._world_size
        for node in cluster_state.nodes.values():
            self._weights[node.rank] = node.workload_weight

        for worker in registered_workers:
            console.print(
                f"  [green]Worker registered: {worker.hostname} "
                f"at {worker.ip_address}, rank {worker.rank}[/green]"
            )

        # Accept TCP tensor connections from all workers using raw socket.
        # Workers may connect from a different IP than they registered with
        # (e.g., link-local vs manual IP on the same Thunderbolt interface),
        # so we match by IP first, then assign remaining connections by order.
        console.print(
            f"  [yellow]Waiting for {len(registered_workers)} "
            "tensor connection(s)...[/yellow]"
        )
        accepted: list[tuple] = []  # [(reader, writer, addr), ...]
        while len(accepted) < len(registered_workers):
            client_sock, addr = await asyncio.wait_for(
                loop.sock_accept(server_sock), timeout=60.0,
            )
            client_sock.setblocking(False)
            reader, writer = await asyncio.open_connection(sock=client_sock)
            accepted.append((reader, writer, addr))
            console.print(f"  TCP connection from {addr[0]}:{addr[1]}")

        # Map accepted connections to worker ranks.
        # Try IP matching first, then assign remaining by arrival order.
        self._peer_info: dict[int, str] = {}
        ip_to_conn = {conn[2][0]: conn for conn in accepted}
        unmatched_workers = []
        matched_ips = set()

        for worker in registered_workers:
            conn_data = ip_to_conn.get(worker.ip_address)
            if conn_data:
                reader, writer, addr = conn_data
                conn_key = f"{addr[0]}:{addr[1]}"
                async with self._transport._lock:
                    self._transport._connections[conn_key] = (reader, writer)
                self._peer_info[worker.rank] = conn_key
                matched_ips.add(addr[0])
                console.print(f"  Mapped rank {worker.rank} -> {conn_key} (IP match)")
            else:
                unmatched_workers.append(worker)

        # Assign remaining connections to unmatched workers by arrival order
        unmatched_conns = [c for c in accepted if c[2][0] not in matched_ips]
        for worker, conn_data in zip(unmatched_workers, unmatched_conns):
            reader, writer, addr = conn_data
            conn_key = f"{addr[0]}:{addr[1]}"
            async with self._transport._lock:
                self._transport._connections[conn_key] = (reader, writer)
            self._peer_info[worker.rank] = conn_key
            console.print(
                f"  Mapped rank {worker.rank} -> {conn_key} "
                f"(registered as {worker.ip_address})"
            )

    async def _setup_distributed_worker(self) -> None:
        """Set up worker node for distributed training.

        Registers with master via gRPC, then waits for master to
        connect to our tensor port for AllReduce communication.
        """
        from macfleet.comm.grpc_service import ClusterControlClient
        from macfleet.utils.network import (
            get_gpu_info,
            get_hostname,
            get_local_ip,
            get_memory_bandwidth,
            get_memory_info,
        )

        hostname = get_hostname()
        ip = self.cluster_config.host or get_local_ip()
        gpu_info = get_gpu_info()
        mem_info = get_memory_info()

        console.print(
            f"  Connecting to master at "
            f"{self.cluster_config.master_addr}:{self.cluster_config.master_port}"
        )

        self._grpc_client = ClusterControlClient(
            self.cluster_config.master_addr,
            self.cluster_config.master_port,
            auth_token=self.cluster_config.auth_token,
            auth_token_env=self.cluster_config.auth_token_env,
        )

        # Retry gRPC connect + register (channel is lazy, so the real
        # network I/O happens inside register(), not connect())
        for attempt in range(30):
            try:
                self._grpc_client.connect()
                rank, weight, world_size, master_tensor_addr, master_tensor_port = (
                    self._grpc_client.register(
                        hostname=hostname,
                        ip_address=ip,
                        gpu_cores=gpu_info.get("gpu_cores", 10),
                        ram_gb=mem_info.get("total_gb", 16),
                        memory_bandwidth_gbps=get_memory_bandwidth(),
                        tensor_port=self.cluster_config.tensor_port,
                    )
                )
                break
            except Exception as e:
                if attempt < 29:
                    console.print(f"  [yellow]Attempt {attempt + 1}/30 failed: {e}[/yellow]")
                    self._grpc_client.disconnect()
                    await asyncio.sleep(3.0)
                else:
                    raise RuntimeError(
                        "Failed to register with master after 30 attempts"
                    )

        console.print(f"  [green]Registered with master, assigned rank {rank}[/green]")

        self._rank = rank
        self._world_size = world_size

        # Fallback: if gRPC response has empty tensor addr, use master_addr
        if not master_tensor_addr:
            master_tensor_addr = self.cluster_config.master_addr
        if not master_tensor_port:
            master_tensor_port = self.cluster_config.tensor_port

        console.print(
            f"  Master tensor endpoint: {master_tensor_addr}:{master_tensor_port}"
        )

        # Get full cluster weights from master
        state = self._grpc_client.get_cluster_state()
        self._weights = [0.0] * world_size
        for node in state['nodes']:
            self._weights[node['rank']] = node['workload_weight']

        # Connect to master's tensor port (worker initiates TCP)
        conn_key = await self._transport.connect(
            master_tensor_addr, master_tensor_port,
        )
        self._peer_info = {0: conn_key}
        console.print("  TCP tensor connection established with master")

    async def _verify_connections(self) -> None:
        """Verify peer connections by exchanging a small test tensor.

        Master sends first, worker receives first, ensuring no deadlock.
        """
        console.print("  Verifying peer connections...")

        test_tensor = torch.tensor(
            [float(self._rank), 1.0, 2.0, 3.0], dtype=torch.float32,
        )

        for peer_rank, conn_key in self._collective_group._peer_connections.items():
            if self._rank < peer_rank:
                # Lower rank sends first, then receives
                await self._transport.send_tensor(
                    test_tensor, conn_key, MessageType.TENSOR_WEIGHTS,
                )
                recv, _ = await self._transport.recv_tensor(conn_key)
                assert recv.shape == test_tensor.shape
            else:
                # Higher rank receives first, then sends
                recv, _ = await self._transport.recv_tensor(conn_key)
                assert recv.shape == test_tensor.shape
                await self._transport.send_tensor(
                    test_tensor, conn_key, MessageType.TENSOR_WEIGHTS,
                )

            console.print(
                f"    Rank {self._rank} <-> Rank {peer_rank}: "
                f"[green]OK[/green] (received {recv.tolist()})"
            )

    async def teardown(self) -> None:
        """Clean up distributed components."""
        if self._collective_group:
            await self._collective_group.disconnect_all()
        if self._transport:
            await self._transport.stop_server()
        # Close raw tensor server socket (used by master in distributed mode)
        if hasattr(self, '_tensor_server_sock') and self._tensor_server_sock:
            self._tensor_server_sock.close()
            self._tensor_server_sock = None
        if self._grpc_server:
            self._grpc_server.stop()
            self._grpc_server = None
        if self._grpc_client:
            self._grpc_client.disconnect()
            self._grpc_client = None

    def fit(self) -> TrainerState:
        """Train the model.

        Returns:
            Final trainer state.
        """
        return asyncio.run(self._fit_async())

    async def _fit_async(self) -> TrainerState:
        """Async implementation of fit."""
        await self.setup()

        try:
            console.print("\n[bold blue]Starting training...[/bold blue]")

            for epoch in range(self.training_config.epochs):
                self.state.epoch = epoch
                self._sampler.set_epoch(epoch)

                # Train one epoch
                epoch_metrics = await self._train_epoch(epoch)

                # Clear MPS cache between epochs to prevent memory buildup
                if self._device == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()

                # Print epoch summary
                self._print_epoch_summary(epoch, epoch_metrics)

                # Validation
                if self.val_dataset is not None:
                    val_metrics = await self._validate()
                    console.print(f"  Validation loss: {val_metrics['loss']:.4f}")
                    # Clear cache after validation too
                    if self._device == "mps":
                        torch.mps.empty_cache()

                # Checkpoint
                if (epoch + 1) % self.training_config.checkpoint_every == 0:
                    self._save_checkpoint(epoch)

            # Barrier: wait for all nodes to finish before teardown.
            # The master finishes faster (higher throughput), so without this
            # it tears down the TCP connection while the worker is still syncing.
            if self.distributed and self._world_size > 1:
                console.print("[dim]Waiting for all nodes to finish...[/dim]")
                try:
                    from macfleet.comm.collectives import allreduce
                    barrier_tensor = torch.ones(1, device="cpu")
                    await allreduce(barrier_tensor, self._collective_group, op="sum")
                    console.print("[dim]All nodes synchronized.[/dim]")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # Other node already disconnected, that's OK

            console.print("\n[bold green]Training complete![/bold green]")

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            console.print(f"\n[bold red]Peer disconnected: {e}[/bold red]")
            console.print("[yellow]The other node may have run out of memory or crashed.[/yellow]")
        finally:
            await self.teardown()

        return self.state

    async def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self._ddp_model.train()

        # Sync BatchNorm buffers from rank 0 at start of each epoch
        if self.distributed and self._world_size > 1:
            await self._ddp_model.sync_buffers()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_compute_time = 0.0
        total_comm_time = 0.0

        num_batches = self._steps_per_epoch

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch + 1}", total=num_batches)

            # Use an iterator that cycles so shorter dataloaders don't run out
            # before all nodes have completed the synchronized step count.
            data_iter = iter(self._dataloader)

            for batch_idx in range(num_batches):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    # Dataloader exhausted — cycle back to start
                    data_iter = iter(self._dataloader)
                    inputs, targets = next(data_iter)

                # Move to device
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Forward pass (timed)
                compute_start = time.perf_counter()

                self._ddp_model.zero_grad(set_to_none=True)
                outputs = self._ddp_model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Sync MPS
                if self._device == "mps":
                    torch.mps.synchronize()

                compute_time = (time.perf_counter() - compute_start) * 1000

                # Gradient sync (timed)
                comm_start = time.perf_counter()
                await self._ddp_model.sync_gradients()
                comm_time = (time.perf_counter() - comm_start) * 1000

                # Optimizer step
                self._optimizer.step()

                # Metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

                # Free forward pass memory
                del outputs, loss, predicted
                total_compute_time += compute_time
                total_comm_time += comm_time

                self.state.global_step += 1

                # Periodic MPS cache clear to prevent memory buildup within epoch
                if self._device == "mps" and (batch_idx + 1) % 50 == 0:
                    torch.mps.empty_cache()

                progress.update(task, advance=1)

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        if total_compute_time > 0:
            samples_per_sec = total_samples / (total_compute_time / 1000)
        else:
            samples_per_sec = 0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples_per_sec": samples_per_sec,
            "compute_time_ms": total_compute_time,
            "comm_time_ms": total_comm_time,
        }

    async def _validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataset is None:
            return {}

        self._ddp_model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._ddp_model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
                del outputs, loss, predicted

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        }

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print epoch summary."""
        table = Table(title=f"Epoch {epoch + 1} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Loss", f"{metrics['loss']:.4f}")
        table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
        table.add_row("Throughput", f"{metrics['samples_per_sec']:.1f} samples/s")
        table.add_row("Compute Time", f"{metrics['compute_time_ms']:.0f} ms")
        table.add_row("Comm Time", f"{metrics['comm_time_ms']:.0f} ms")

        total_time = metrics['compute_time_ms'] + metrics['comm_time_ms']
        comm_overhead = metrics['comm_time_ms'] / total_time * 100 if total_time > 0 else 0.0
        table.add_row("Comm Overhead", f"{comm_overhead:.1f}%")

        console.print(table)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a training checkpoint. Only master (rank 0) saves."""
        if self._rank != 0:
            return

        checkpoint_dir = self.training_config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        # Save directly without building a full dict in memory
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.state.global_step,
                "model_state_dict": self._ddp_model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "training_config": self.training_config.to_dict(),
            },
            path,
        )
        # Clear MPS cache after checkpoint (state_dict creates copies)
        if self._device == "mps":
            torch.mps.empty_cache()
        console.print(f"[green]Checkpoint saved: {path}[/green]")

    def load_checkpoint(self, path: str, trusted: bool = False) -> None:
        """Load a training checkpoint."""
        from macfleet.training.checkpoint import safe_torch_load

        checkpoint = safe_torch_load(path, map_location=self._device, trusted=trusted)

        self._ddp_model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state.epoch = checkpoint["epoch"]
        self.state.global_step = checkpoint["global_step"]

        console.print(f"[green]Checkpoint loaded: {path}[/green]")
