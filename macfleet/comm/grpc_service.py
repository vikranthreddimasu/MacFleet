"""gRPC control plane for MacFleet cluster coordination.

Provides server and client implementations for cluster control messages
including registration, heartbeats, synchronization barriers, and training commands.
"""

import hmac
import threading
import time
from concurrent import futures
from typing import Callable, Optional

import grpc

from macfleet.comm.proto import control_pb2, control_pb2_grpc
from macfleet.core.config import (
    DEFAULT_AUTH_TOKEN_ENV,
    ClusterState,
    NodeConfig,
    TrainingConfig,
    resolve_auth_token,
)
from macfleet.utils.network import validate_host, validate_port

AUTH_METADATA_KEY = "x-macfleet-auth-token"


class ClusterControlServicer(control_pb2_grpc.ClusterControlServicer):
    """gRPC service implementation for cluster coordination.

    Runs on the master node and handles:
    - Node registration
    - Heartbeats
    - Synchronization barriers
    - Training control
    """

    def __init__(
        self,
        cluster_state: ClusterState,
        tensor_addr: str,
        tensor_port: int,
        on_register: Optional[Callable[[NodeConfig], None]] = None,
        on_heartbeat: Optional[Callable[[int, float, str], None]] = None,
        on_deregister: Optional[Callable[[int, str], None]] = None,
        auth_token: Optional[str] = None,
    ):
        """Initialize the servicer.

        Args:
            cluster_state: Shared cluster state object.
            tensor_addr: Address for tensor transfers.
            tensor_port: Port for tensor transfers.
            on_register: Callback when a node registers.
            on_heartbeat: Callback on heartbeat (rank, throughput, thermal).
            on_deregister: Callback on deregistration (rank, reason).
            auth_token: Optional shared token required for all control RPCs.
        """
        self._state = cluster_state
        self._tensor_addr = tensor_addr
        self._tensor_port = tensor_port
        self._on_register = on_register
        self._on_heartbeat = on_heartbeat
        self._on_deregister = on_deregister
        self._auth_token = resolve_auth_token(auth_token, "")
        self._next_rank = 1  # Rank 0 is reserved for master
        self._free_ranks: list[int] = []  # Reusable ranks from departed nodes
        # barrier_id -> (ranks, created_time)
        self._barriers: dict[str, tuple[set[int], float]] = {}
        self._barrier_ttl = 120.0  # seconds before stale barriers are cleaned up
        self._base_weights: dict[int, float] = {}  # rank -> baseline weight (GPU-core based)
        self._lock = threading.Lock()  # Protects all mutable state from gRPC thread pool

    def _require_authorized(self, context: grpc.ServicerContext) -> None:
        """Require the configured shared token when auth is enabled."""
        if self._auth_token is None:
            return

        provided = None
        for item in context.invocation_metadata():
            if isinstance(item, tuple):
                key, value = item
            else:
                key, value = item.key, item.value
            if key.lower() == AUTH_METADATA_KEY:
                provided = value
                break

        if provided is not None and hmac.compare_digest(str(provided), self._auth_token):
            return

        message = "missing or invalid MacFleet auth token"
        if hasattr(context, "abort"):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, message)
        raise PermissionError(message)

    def Register(
        self,
        request: control_pb2.RegisterRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.RegisterResponse:
        """Handle node registration."""
        self._require_authorized(context)

        with self._lock:
            # Assign rank (reuse freed ranks before incrementing)
            if self._free_ranks:
                rank = self._free_ranks.pop(0)
            else:
                rank = self._next_rank
                self._next_rank += 1

            # Calculate workload weight based on GPU cores
            total_cores = sum(n.gpu_cores for n in self._state.nodes.values())
            total_cores += request.gpu_cores

            # Create node config
            node = NodeConfig(
                hostname=request.hostname,
                ip_address=request.ip_address,
                gpu_cores=request.gpu_cores,
                ram_gb=request.ram_gb,
                memory_bandwidth_gbps=request.memory_bandwidth_gbps,
                tensor_port=request.tensor_port,
                rank=rank,
                workload_weight=request.gpu_cores / total_cores if total_cores > 0 else 0.5,
            )

            # Add to cluster state
            self._state.add_node(node)

            # Recalculate all weights and store baselines
            self._recalculate_weights()
            for n in self._state.nodes.values():
                self._base_weights[n.rank] = n.workload_weight

        # Callback outside lock to avoid holding lock during user code
        if self._on_register:
            self._on_register(node)

        return control_pb2.RegisterResponse(
            assigned_rank=rank,
            workload_weight=node.workload_weight,
            world_size=self._state.world_size,
            master_tensor_addr=self._tensor_addr,
            master_tensor_port=self._tensor_port,
        )

    def _recalculate_weights(self) -> None:
        """Recalculate workload weights for all nodes."""
        total_cores = sum(n.gpu_cores for n in self._state.nodes.values())
        if total_cores == 0:
            return

        for node in self._state.nodes.values():
            node.workload_weight = node.gpu_cores / total_cores

    def Heartbeat(
        self,
        request: control_pb2.HeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.HeartbeatResponse:
        """Handle heartbeat from a worker."""
        self._require_authorized(context)

        with self._lock:
            node = self._state.get_node(request.rank)
            if not node:
                return control_pb2.HeartbeatResponse(
                    acknowledged=False,
                    new_workload_weight=0.0,
                    should_stop=False,
                )

            # Apply thermal multiplier relative to baseline (not current)
            # so weight doesn't spiral toward zero on repeated heartbeats.
            base_weight = self._base_weights.get(request.rank, node.workload_weight)
            if request.thermal_state in ("serious", "critical"):
                new_weight = base_weight * 0.7
            elif request.thermal_state == "fair":
                new_weight = base_weight * 0.9
            else:
                new_weight = base_weight  # Restore when nominal

            # Write adjusted weight back to cluster state
            node.workload_weight = new_weight

        # Callback outside lock
        if self._on_heartbeat:
            self._on_heartbeat(
                request.rank,
                request.throughput_samples_per_sec,
                request.thermal_state,
            )

        return control_pb2.HeartbeatResponse(
            acknowledged=True,
            new_workload_weight=new_weight,
            should_stop=False,
        )

    def SyncBarrier(
        self,
        request: control_pb2.BarrierRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.BarrierResponse:
        """Handle synchronization barrier."""
        self._require_authorized(context)

        barrier_id = request.barrier_id
        rank = request.rank

        with self._lock:
            now = time.time()

            # Clean up stale barriers (TTL-based)
            stale = [bid for bid, (_, created) in self._barriers.items()
                     if now - created > self._barrier_ttl]
            for bid in stale:
                del self._barriers[bid]

            # Initialize or update barrier
            if barrier_id not in self._barriers:
                self._barriers[barrier_id] = (set(), now)
            ranks_set, created = self._barriers[barrier_id]
            ranks_set.add(rank)
            nodes_at_barrier = len(ranks_set)

            # Check if all nodes have reached the barrier
            proceed = nodes_at_barrier >= self._state.world_size

            if proceed:
                del self._barriers[barrier_id]

        return control_pb2.BarrierResponse(
            proceed=proceed,
            nodes_at_barrier=nodes_at_barrier,
        )

    def StartTraining(
        self,
        request: control_pb2.TrainingConfigProto,
        context: grpc.ServicerContext,
    ) -> control_pb2.Ack:
        """Handle start training command."""
        self._require_authorized(context)

        with self._lock:
            self._state.training_active = True
            self._state.current_epoch = 0
            self._state.current_step = 0

        return control_pb2.Ack(
            success=True,
            message="Training started",
        )

    def StopTraining(
        self,
        request: control_pb2.StopRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.Ack:
        """Handle stop training command."""
        self._require_authorized(context)

        with self._lock:
            self._state.training_active = False

        return control_pb2.Ack(
            success=True,
            message=f"Training stopped: {request.reason}",
        )

    def GetClusterState(
        self,
        request: control_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> control_pb2.ClusterStateProto:
        """Return current cluster state."""
        self._require_authorized(context)

        with self._lock:
            nodes = []
            for node in self._state.nodes.values():
                nodes.append(control_pb2.NodeInfoProto(
                    rank=node.rank,
                    hostname=node.hostname,
                    ip_address=node.ip_address,
                    gpu_cores=node.gpu_cores,
                    ram_gb=node.ram_gb,
                    memory_bandwidth_gbps=node.memory_bandwidth_gbps,
                    tensor_port=node.tensor_port,
                    workload_weight=node.workload_weight,
                    status="connected",
                ))

            status = "training" if self._state.training_active else "idle"

            return control_pb2.ClusterStateProto(
                world_size=self._state.world_size,
                nodes=nodes,
                training_active=self._state.training_active,
                current_epoch=self._state.current_epoch,
                current_step=self._state.current_step,
                training_status=status,
            )

    def Deregister(
        self,
        request: control_pb2.DeregisterRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.Ack:
        """Handle graceful node deregistration."""
        self._require_authorized(context)

        with self._lock:
            node = self._state.get_node(request.rank)
            if not node:
                return control_pb2.Ack(
                    success=False,
                    message=f"Unknown rank {request.rank}",
                )

            hostname = node.hostname
            self._state.remove_node(request.rank)
            self._free_ranks.append(request.rank)
            self._base_weights.pop(request.rank, None)
            self._recalculate_weights()

        # Callback outside lock
        if self._on_deregister:
            self._on_deregister(request.rank, request.reason)

        return control_pb2.Ack(
            success=True,
            message=f"Node {hostname} (rank {request.rank}) deregistered: {request.reason}",
        )

    def Broadcast(
        self,
        request: control_pb2.BroadcastRequest,
        context: grpc.ServicerContext,
    ) -> control_pb2.Ack:
        """Handle broadcast request."""
        self._require_authorized(context)

        # For now, just acknowledge - actual broadcast logic in coordinator
        return control_pb2.Ack(
            success=True,
            message=f"Broadcast {request.message_type} received",
        )


class GRPCServer:
    """gRPC server wrapper for the cluster control service."""

    def __init__(
        self,
        servicer: ClusterControlServicer,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ):
        """Initialize the server.

        Args:
            servicer: The service implementation.
            host: Host to bind to.
            port: Port to bind to.
            max_workers: Max thread pool workers.
        """
        self._servicer = servicer
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._server: Optional[grpc.Server] = None

    def start(self) -> None:
        """Start the gRPC server."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )
        control_pb2_grpc.add_ClusterControlServicer_to_server(
            self._servicer, self._server
        )
        self._server.add_insecure_port(f"{self._host}:{self._port}")
        self._server.start()

    def stop(self, grace: float = 5.0) -> None:
        """Stop the gRPC server."""
        if self._server:
            self._server.stop(grace)
            self._server = None

    def wait_for_termination(self, timeout: Optional[float] = None) -> bool:
        """Wait for the server to terminate."""
        if self._server:
            return self._server.wait_for_termination(timeout)
        return True

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None


class ClusterControlClient:
    """gRPC client for connecting to the cluster coordinator."""

    def __init__(
        self,
        master_addr: str,
        master_port: int = 50051,
        auth_token: Optional[str] = None,
        auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
    ):
        """Initialize the client.

        Args:
            master_addr: Address of the master node.
            master_port: gRPC port of the master.
            auth_token: Optional shared token for protected control planes.
            auth_token_env: Environment variable to read when auth_token is unset.
        """
        master_addr = validate_host(master_addr, "master_addr")
        master_port = validate_port(master_port, "master_port")
        self._addr = f"{master_addr}:{master_port}"
        self._auth_token = resolve_auth_token(auth_token, auth_token_env)
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[control_pb2_grpc.ClusterControlStub] = None

    def _metadata(self) -> Optional[tuple[tuple[str, str], ...]]:
        """Return gRPC metadata containing the auth token when configured."""
        if self._auth_token is None:
            return None
        return ((AUTH_METADATA_KEY, self._auth_token),)

    def connect(self) -> None:
        """Connect to the master."""
        self._channel = grpc.insecure_channel(self._addr)
        self._stub = control_pb2_grpc.ClusterControlStub(self._channel)

    def disconnect(self) -> None:
        """Disconnect from the master."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    def register(
        self,
        hostname: str,
        ip_address: str,
        gpu_cores: int,
        ram_gb: int,
        memory_bandwidth_gbps: float,
        tensor_port: int,
    ) -> tuple[int, float, int, str, int]:
        """Register this node with the cluster.

        Returns:
            Tuple of (rank, workload_weight, world_size, master_tensor_addr, master_tensor_port).
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        request = control_pb2.RegisterRequest(
            hostname=hostname,
            ip_address=ip_address,
            gpu_cores=gpu_cores,
            ram_gb=ram_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            tensor_port=tensor_port,
        )

        response = self._stub.Register(request, metadata=self._metadata())

        return (
            response.assigned_rank,
            response.workload_weight,
            response.world_size,
            response.master_tensor_addr,
            response.master_tensor_port,
        )

    def heartbeat(
        self,
        rank: int,
        throughput: float = 0.0,
        thermal_state: str = "nominal",
        current_step: int = 0,
    ) -> tuple[bool, float, bool]:
        """Send a heartbeat to the master.

        Returns:
            Tuple of (acknowledged, new_workload_weight, should_stop).
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        request = control_pb2.HeartbeatRequest(
            rank=rank,
            throughput_samples_per_sec=throughput,
            thermal_state=thermal_state,
            timestamp_ms=int(time.time() * 1000),
            current_step=current_step,
        )

        response = self._stub.Heartbeat(request, metadata=self._metadata())

        return (
            response.acknowledged,
            response.new_workload_weight,
            response.should_stop,
        )

    def sync_barrier(
        self,
        rank: int,
        step: int,
        barrier_id: str,
    ) -> tuple[bool, int]:
        """Wait at a synchronization barrier.

        Returns:
            Tuple of (proceed, nodes_at_barrier).
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        request = control_pb2.BarrierRequest(
            rank=rank,
            step=step,
            barrier_id=barrier_id,
        )

        response = self._stub.SyncBarrier(request, metadata=self._metadata())

        return (response.proceed, response.nodes_at_barrier)

    def get_cluster_state(self) -> dict:
        """Get the current cluster state.

        Returns:
            Dictionary with cluster state information.
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        response = self._stub.GetClusterState(
            control_pb2.Empty(),
            metadata=self._metadata(),
        )

        nodes = []
        for node in response.nodes:
            nodes.append({
                "rank": node.rank,
                "hostname": node.hostname,
                "ip_address": node.ip_address,
                "gpu_cores": node.gpu_cores,
                "ram_gb": node.ram_gb,
                "workload_weight": node.workload_weight,
                "status": node.status,
            })

        return {
            "world_size": response.world_size,
            "nodes": nodes,
            "training_active": response.training_active,
            "current_epoch": response.current_epoch,
            "current_step": response.current_step,
            "training_status": response.training_status,
        }

    def start_training(self, config: TrainingConfig) -> bool:
        """Signal the cluster to start training.

        Returns:
            True if successful.
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        request = control_pb2.TrainingConfigProto(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            compression=config.compression.value,
            topk_ratio=config.topk_ratio,
            checkpoint_every=config.checkpoint_every,
            checkpoint_dir=config.checkpoint_dir,
            device=config.device,
        )

        response = self._stub.StartTraining(request, metadata=self._metadata())
        return response.success

    def deregister(self, rank: int, reason: str = "Graceful shutdown") -> bool:
        """Deregister this node from the cluster.

        Returns:
            True if successful.
        """
        if not self._stub:
            return False

        try:
            request = control_pb2.DeregisterRequest(
                rank=rank,
                reason=reason,
            )
            response = self._stub.Deregister(
                request,
                timeout=5.0,
                metadata=self._metadata(),
            )
            return response.success
        except Exception:
            return False

    def stop_training(self, reason: str = "User request") -> bool:
        """Signal the cluster to stop training.

        Returns:
            True if successful.
        """
        if not self._stub:
            raise RuntimeError("Not connected to master")

        request = control_pb2.StopRequest(
            reason=reason,
            save_checkpoint=True,
        )

        response = self._stub.StopTraining(request, metadata=self._metadata())
        return response.success

    @property
    def is_connected(self) -> bool:
        """Check if connected to master."""
        return self._stub is not None
