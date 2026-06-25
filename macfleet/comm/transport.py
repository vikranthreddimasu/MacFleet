"""Async TCP tensor transport for MacFleet.

Provides high-performance tensor transfer over raw TCP sockets using asyncio.
Uses the protocol defined in Section 9.2 of DESIGN.md.
"""

import asyncio
import struct
from typing import Optional

import torch

from macfleet.utils.tensor_utils import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MessageType,
    bytes_to_tensor,
    deserialize_compressed_gradient,
    serialize_compressed_gradient,
    tensor_to_bytes,
)

# Buffer sizes for optimal Thunderbolt performance
RECV_BUFFER_SIZE = 1024 * 1024  # 1 MB receive buffer
SEND_BUFFER_SIZE = 1024 * 1024  # 1 MB send buffer

# Timeout for recv operations to avoid hanging on dead peers
RECV_TIMEOUT = 120.0  # seconds


class TensorTransport:
    """Async TCP transport for tensor data.

    Handles sending and receiving tensors over TCP with
    the binary protocol from DESIGN.md Section 9.2.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50052,
    ):
        """Initialize transport.

        Args:
            host: Host to bind/connect to.
            port: Port for tensor transfers.
        """
        self.host = host
        self.port = port
        self._server: Optional[asyncio.Server] = None
        self._connections: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._lock = asyncio.Lock()

    async def start_server(
        self,
        on_receive: Optional[callable] = None,
    ) -> None:
        """Start the tensor server.

        Args:
            on_receive: Callback for received tensors.
                       Signature: async def callback(tensor, msg_type, addr)
        """
        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            addr = writer.get_extra_info("peername")
            addr_key = f"{addr[0]}:{addr[1]}"

            async with self._lock:
                self._connections[addr_key] = (reader, writer)

            if not on_receive:
                # No callback - connection stored for explicit send/recv.
                # Don't read from the stream; recv_tensor will do that.
                return

            try:
                while True:
                    # Read header
                    header_data = await asyncio.wait_for(
                        reader.readexactly(HEADER_SIZE), timeout=RECV_TIMEOUT
                    )
                    msg_type_code, dtype_code, n_dims, payload_size = struct.unpack(
                        HEADER_FORMAT, header_data
                    )
                    msg_type = MessageType(msg_type_code)

                    if msg_type == MessageType.COMPRESSED_GRADIENT:
                        # Read compression metadata + payload
                        metadata_size = 12  # orig_numel, topk_count, orig_dtype
                        remaining = await asyncio.wait_for(
                            reader.readexactly(metadata_size + payload_size),
                            timeout=RECV_TIMEOUT,
                        )
                        full_data = header_data + remaining

                        indices, values, orig_numel, orig_dtype = deserialize_compressed_gradient(
                            full_data
                        )
                        await on_receive(
                            (indices, values, orig_numel, orig_dtype),
                            msg_type,
                            addr,
                        )
                    else:
                        # Read shape + payload
                        shape_size = n_dims * 4
                        remaining = await asyncio.wait_for(
                            reader.readexactly(shape_size + payload_size),
                            timeout=RECV_TIMEOUT,
                        )
                        full_data = header_data + remaining

                        tensor, msg_type = bytes_to_tensor(full_data)
                        await on_receive(tensor, msg_type, addr)

            except (asyncio.IncompleteReadError, ConnectionResetError, TimeoutError):
                # Client disconnected or timed out
                pass
            finally:
                async with self._lock:
                    self._connections.pop(addr_key, None)
                writer.close()
                await writer.wait_closed()

        self._server = await asyncio.start_server(
            handle_client,
            self.host,
            self.port,
        )

        # Configure socket options for performance
        for sock in self._server.sockets:
            sock.setsockopt(
                __import__("socket").SOL_SOCKET,
                __import__("socket").SO_RCVBUF,
                RECV_BUFFER_SIZE,
            )
            sock.setsockopt(
                __import__("socket").SOL_SOCKET,
                __import__("socket").SO_SNDBUF,
                SEND_BUFFER_SIZE,
            )

    async def stop_server(self) -> None:
        """Stop the tensor server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Close all connections
        async with self._lock:
            for reader, writer in self._connections.values():
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            self._connections.clear()

    async def connect(self, host: str, port: int) -> str:
        """Connect to a remote tensor server.

        Args:
            host: Remote host.
            port: Remote port.

        Returns:
            Connection key for sending.
        """
        reader, writer = await asyncio.open_connection(host, port)

        # Configure socket for performance
        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(
                __import__("socket").SOL_SOCKET,
                __import__("socket").SO_RCVBUF,
                RECV_BUFFER_SIZE,
            )
            sock.setsockopt(
                __import__("socket").SOL_SOCKET,
                __import__("socket").SO_SNDBUF,
                SEND_BUFFER_SIZE,
            )

        conn_key = f"{host}:{port}"
        async with self._lock:
            self._connections[conn_key] = (reader, writer)

        return conn_key

    async def wait_for_incoming(self, from_ip: str, timeout: float = 30.0) -> str:
        """Wait for an incoming connection from a specific IP.

        Args:
            from_ip: IP address to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            Connection key for the accepted connection.

        Raises:
            TimeoutError: If no connection arrives within timeout.
        """
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            async with self._lock:
                for conn_key in self._connections:
                    if conn_key.startswith(from_ip + ":"):
                        return conn_key
            await asyncio.sleep(0.1)
        raise TimeoutError(f"No incoming connection from {from_ip} within {timeout}s")

    async def disconnect(self, conn_key: str) -> None:
        """Disconnect from a remote server.

        Args:
            conn_key: Connection key from connect().
        """
        async with self._lock:
            conn = self._connections.pop(conn_key, None)
            if conn:
                _, writer = conn
                try:
                    writer.close()
                    await writer.wait_closed()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # Peer already disconnected

    async def send_tensor(
        self,
        tensor: torch.Tensor,
        conn_key: str,
        msg_type: MessageType = MessageType.TENSOR_GRADIENT,
    ) -> None:
        """Send a tensor to a connected peer.

        Args:
            tensor: Tensor to send.
            conn_key: Connection key from connect().
            msg_type: Message type for the tensor.
        """
        async with self._lock:
            conn = self._connections.get(conn_key)
            if not conn:
                raise ConnectionError(f"Not connected to {conn_key}")
            _, writer = conn

        data = tensor_to_bytes(tensor, msg_type)
        writer.write(data)
        await writer.drain()

    async def recv_tensor(
        self,
        conn_key: str,
        device: Optional[str] = None,
    ) -> tuple[torch.Tensor, MessageType]:
        """Receive a tensor from a connected peer.

        Args:
            conn_key: Connection key.
            device: Target device for the tensor.

        Returns:
            Tuple of (tensor, message_type).
        """
        async with self._lock:
            conn = self._connections.get(conn_key)
            if not conn:
                raise ConnectionError(f"Not connected to {conn_key}")
            reader, _ = conn

        # Read header (with timeout to avoid hanging on dead peers)
        header_data = await asyncio.wait_for(
            reader.readexactly(HEADER_SIZE), timeout=RECV_TIMEOUT
        )
        msg_type_code, dtype_code, n_dims, payload_size = struct.unpack(
            HEADER_FORMAT, header_data
        )
        msg_type = MessageType(msg_type_code)

        # Read shape + payload
        shape_size = n_dims * 4
        remaining = await asyncio.wait_for(
            reader.readexactly(shape_size + payload_size), timeout=RECV_TIMEOUT
        )
        full_data = header_data + remaining

        tensor, msg_type = bytes_to_tensor(full_data, device)
        return tensor, msg_type

    async def send_compressed_gradient(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        original_numel: int,
        original_dtype: torch.dtype,
        conn_key: str,
    ) -> None:
        """Send a compressed gradient.

        Args:
            indices: Sparse indices.
            values: Sparse values (FP16).
            original_numel: Original gradient size.
            original_dtype: Original gradient dtype.
            conn_key: Connection key.
        """
        async with self._lock:
            conn = self._connections.get(conn_key)
            if not conn:
                raise ConnectionError(f"Not connected to {conn_key}")
            _, writer = conn

        data = serialize_compressed_gradient(indices, values, original_numel, original_dtype)
        writer.write(data)
        await writer.drain()

    async def recv_compressed_gradient(
        self,
        conn_key: str,
        device: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
        """Receive a compressed gradient.

        Args:
            conn_key: Connection key.
            device: Target device.

        Returns:
            Tuple of (indices, values, original_numel, original_dtype).
        """
        async with self._lock:
            conn = self._connections.get(conn_key)
            if not conn:
                raise ConnectionError(f"Not connected to {conn_key}")
            reader, _ = conn

        # Read header (with timeout to avoid hanging on dead peers)
        header_data = await asyncio.wait_for(
            reader.readexactly(HEADER_SIZE), timeout=RECV_TIMEOUT
        )
        msg_type_code, _, _, payload_size = struct.unpack(HEADER_FORMAT, header_data)

        if msg_type_code != MessageType.COMPRESSED_GRADIENT:
            raise ValueError(f"Expected compressed gradient, got {msg_type_code}")

        # Read compression metadata + payload
        metadata_size = 12
        remaining = await asyncio.wait_for(
            reader.readexactly(metadata_size + payload_size), timeout=RECV_TIMEOUT
        )
        full_data = header_data + remaining

        return deserialize_compressed_gradient(full_data, device)


async def send_tensor(
    tensor: torch.Tensor,
    host: str,
    port: int,
    msg_type: MessageType = MessageType.TENSOR_GRADIENT,
) -> None:
    """Convenience function to send a single tensor.

    Opens a connection, sends the tensor, and closes.

    Args:
        tensor: Tensor to send.
        host: Remote host.
        port: Remote port.
        msg_type: Message type.
    """
    reader, writer = await asyncio.open_connection(host, port)
    try:
        data = tensor_to_bytes(tensor, msg_type)
        writer.write(data)
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def recv_tensor(
    reader: asyncio.StreamReader,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, MessageType]:
    """Receive a tensor from a stream reader.

    Args:
        reader: Async stream reader.
        device: Target device.

    Returns:
        Tuple of (tensor, message_type).
    """
    # Read header
    header_data = await reader.readexactly(HEADER_SIZE)
    msg_type_code, dtype_code, n_dims, payload_size = struct.unpack(
        HEADER_FORMAT, header_data
    )

    # Read shape + payload
    shape_size = n_dims * 4
    remaining = await reader.readexactly(shape_size + payload_size)
    full_data = header_data + remaining

    return bytes_to_tensor(full_data, device)


class TensorServer:
    """Simple async tensor server for receiving tensors.

    Example usage:
        server = TensorServer("0.0.0.0", 50052)
        async def handle(tensor, msg_type, addr):
            print(f"Received {tensor.shape} from {addr}")
        await server.start(handle)
        # ... server runs ...
        await server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 50052):
        self.host = host
        self.port = port
        self._server: Optional[asyncio.Server] = None
        self._handler: Optional[callable] = None

    async def start(self, handler: callable) -> None:
        """Start the server with a message handler.

        Args:
            handler: Async callback for received tensors.
                    Signature: async def handler(tensor, msg_type, addr)
        """
        self._handler = handler

        async def client_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            addr = writer.get_extra_info("peername")
            try:
                while True:
                    tensor, msg_type = await recv_tensor(reader)
                    if self._handler:
                        await self._handler(tensor, msg_type, addr)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        self._server = await asyncio.start_server(
            client_handler,
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None and self._server.is_serving()
