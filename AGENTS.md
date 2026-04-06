# AGENTS.md — AI Agent Guide for MacFleet

This file helps AI coding assistants (Claude, Copilot, Cursor, Cody, etc.) understand the MacFleet codebase quickly. If you're an AI agent working on this project, read this first.

## What is MacFleet?

MacFleet pools multiple Apple Silicon Macs into a single distributed ML training cluster. Users run `macfleet join` on each Mac, and the framework handles discovery, gradient synchronization, and workload scheduling automatically.

- **Version**: 2.0.0
- **Language**: Python 3.11+
- **Platform**: macOS only (Apple Silicon)
- **Tests**: 371, run with `make test` or `python -m pytest tests/ -v`
- **Package**: `pip install macfleet`

## Critical Architecture Rule

**The communication layer NEVER imports torch or mlx.** Gradients flow as numpy arrays between nodes. Each engine (TorchEngine, MLXEngine) converts to/from numpy at the boundary. This is the core design constraint — do not break it.

```
 TorchEngine                                    MLXEngine
  (torch)                                        (mlx)
     |                                              |
     v                                              v
 get_flat_gradients() -> np.ndarray    get_flat_gradients() -> np.ndarray
     |                                              |
     +----> CollectiveGroup.allreduce() <-----------+
                (numpy only, no framework imports)
     |                                              |
     v                                              v
 apply_flat_gradients(np.ndarray)      apply_flat_gradients(np.ndarray)
```

## Project Structure

```
macfleet/
  __init__.py              # Version, lazy imports for Pool/train/distributed
  cli/
    main.py                # Click CLI: join, leave, status, info, train, bench, diagnose
  comm/
    protocol.py            # 24-byte binary wire protocol with CRC32
    transport.py           # TCP peer connections with handshake, per-direction locks
    collectives.py         # Ring AllReduce, broadcast, scatter, gather (numpy arrays)
  compression/
    pipeline.py            # Torch-based compression pipeline (v1 compat)
    topk.py                # TopK sparsification with error feedback (torch)
    quantize.py            # FP16/INT8 quantization (torch)
    adaptive.py            # Numpy-native adaptive compression (v2, bandwidth-aware)
  engines/
    base.py                # Engine protocol (typing.Protocol), HardwareProfile, enums
    torch_engine.py        # PyTorch+MPS implementation
    mlx_engine.py          # Apple MLX implementation
  monitoring/
    thermal.py             # macOS thermal state via pmset/ioreg/sysctl
    health.py              # Composite node health score
    throughput.py           # Per-step training throughput tracker
    dashboard.py           # Rich TUI dashboard
  pool/
    agent.py               # Pool agent daemon (runs on every Mac)
    discovery.py           # Zeroconf/mDNS peer discovery
    registry.py            # Cluster state, coordinator election (bully algorithm)
    network.py             # Network interface detection, link type classification
    scheduler.py           # Weighted batch allocation, thermal-aware rebalancing
    heartbeat.py           # Gossip-based peer liveness detection
  security/
    auth.py                # SecurityConfig, HMAC challenge-response, TLS, gradient validation
    __init__.py            # Re-exports security primitives
  sdk/
    pool.py                # macfleet.Pool() context manager
    train.py               # macfleet.train() convenience function
    decorators.py          # @macfleet.distributed decorator
  training/
    data_parallel.py       # N-node gradient sync with optional adaptive compression
    loop.py                # Composable training loop
    sampler.py             # Weighted distributed sampler
  utils/
    __init__.py            # Shared utilities
tests/
  test_comm/               # Transport, protocol, collectives tests
  test_compression/        # TopK, FP16, adaptive compression tests
  test_engines/            # TorchEngine, MLXEngine conformance tests
  test_integration/        # End-to-end multi-node training tests
  test_monitoring/         # Health, throughput, dashboard tests
  test_pool/               # Registry, scheduler, heartbeat, network tests
  test_sdk/                # Pool SDK, train(), @distributed tests
  test_training/           # DataParallel, sampler tests
  conftest.py              # Shared fixtures
```

## Key Interfaces

### Engine Protocol (`macfleet/engines/base.py`)

Both TorchEngine and MLXEngine implement this. Any new engine must too.

```python
class Engine(Protocol):
    def load_model(self, model, optimizer=None) -> None
    def forward(self, batch) -> Any                    # returns loss
    def backward(self, loss) -> None
    def get_flat_gradients(self) -> np.ndarray          # 1D float32
    def apply_flat_gradients(self, flat: np.ndarray)    # from allreduce
    def get_flat_parameters(self) -> np.ndarray          # for broadcast
    def apply_flat_parameters(self, flat: np.ndarray)    # after broadcast
    def step(self) -> None                               # optimizer step
    def zero_grad(self) -> None
    def state_dict(self) -> bytes                        # checkpoint
    def load_state_dict(self, data: bytes) -> None
    def profile(self) -> HardwareProfile
    def param_count(self) -> int
    def memory_usage_gb(self) -> float
```

### DataParallel (`macfleet/training/data_parallel.py`)

Ties an engine to a CollectiveGroup for gradient sync:

```python
dp = DataParallel(engine, group, link_type=LinkType.ETHERNET)
await dp.setup()              # broadcasts initial params from rank 0
await dp.sync_gradients()     # allreduce after backward, before step
```

Config options: `compression` ("none", "light", "moderate", "aggressive", "adaptive"), `compression_warmup_steps`, `broadcast_params_on_start`.

### CollectiveGroup (`macfleet/comm/collectives.py`)

Framework-agnostic collective operations over TCP:

```python
group = CollectiveGroup(rank=0, world_size=3, transport=transport, rank_to_peer={1: "peer-1", 2: "peer-2"})
averaged = await group.allreduce(np_array, op="mean")
synced = await group.broadcast(np_array, src=0)
```

Algorithms: direct exchange for N=2, ring allreduce for N>=3.

### AdaptiveCompressor (`macfleet/compression/adaptive.py`)

Numpy-native gradient compression that adapts to network quality:

```python
compressor = AdaptiveCompressor(link_type=LinkType.WIFI)        # auto: TopK 1% + FP16
compressor = AdaptiveCompressor(link_type=LinkType.ETHERNET)    # auto: TopK 10% + FP16
compressor = AdaptiveCompressor(link_type=LinkType.THUNDERBOLT) # auto: no compression
compressed = compressor.compress(gradient_array)
restored = compressor.decompress(compressed)
```

TopK compressor has error feedback (residual accumulation) for convergence.

### PeerTransport (`macfleet/comm/transport.py`)

TCP transport with handshake-based peer identification:

```python
transport = PeerTransport(local_id="node-0")
await transport.start_server("0.0.0.0", 50052)
await transport.connect("peer-1", "192.168.1.10", 50052)
await transport.send("peer-1", payload_bytes)
data = await transport.recv("peer-1")
```

Per-direction locks (`_send_lock`, `_recv_lock`) allow safe concurrent allreduce. Buffer sizes adapt to link type (WiFi=1MB, Ethernet=2MB, TB4=4MB).

## Common Tasks

### Adding a new engine

1. Create `macfleet/engines/my_engine.py` implementing the Engine protocol
2. Key methods: `get_flat_gradients() -> np.ndarray` and `apply_flat_gradients(np.ndarray)` — these are what DataParallel calls
3. Add tests in `tests/test_engines/test_my_engine.py`
4. Register in `macfleet/engines/__init__.py` (optional import like MLX)
5. Add support in `macfleet/sdk/pool.py` `_train_*` method

### Adding a new CLI command

1. Add a `@cli.command()` function in `macfleet/cli/main.py`
2. Use Click decorators for options/arguments
3. Use `rich.console.Console` for output formatting

### Adding a new compression strategy

1. For numpy-native: add to `macfleet/compression/adaptive.py`
2. For torch-based: add a new `Compressor` subclass in `macfleet/compression/pipeline.py`
3. The adaptive compressor is what DataParallel uses — torch pipeline is v1 compat

### Running tests

```bash
make test                                        # all 268 tests
python -m pytest tests/test_engines/ -v          # just engine tests
python -m pytest tests/ -k "allreduce" -v        # tests matching keyword
python -m pytest tests/test_engines/test_mlx_engine.py -v  # single file
```

MLX tests auto-skip if MLX is not installed (`pytest.importorskip`).

## Known Constraints

- **MLX backward pass**: MLX uses functional `nn.value_and_grad()`, not imperative `loss.backward()`. The MLXEngine stores the batch in `forward()` and recomputes with gradients in `backward()`.
- **Compression pipeline split**: `macfleet/compression/pipeline.py` (torch tensors, v1) vs `macfleet/compression/adaptive.py` (numpy arrays, v2). DataParallel uses the numpy-native adaptive compressor.
- **Single-node Pool.train()**: The `Pool.train()` SDK method currently runs single-node training. Multi-node requires the programmatic API (PeerTransport + CollectiveGroup + DataParallel).
- **asyncio everywhere**: Transport, collectives, and heartbeat are all async. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Don'ts

- Don't import torch or mlx in `macfleet/comm/`, `macfleet/pool/`, or `macfleet/compression/adaptive.py`
- Don't add `grpcio` or protobuf — v2 replaced gRPC with msgpack + raw TCP
- Don't use `list.pop(0)` — use `collections.deque` for O(1) operations
- Don't call `asyncio.Event.set()` from a non-asyncio thread without `loop.call_soon_threadsafe()`

## Dependencies

Core (always installed): `zeroconf`, `rich`, `click`, `numpy`, `msgpack`
Optional: `torch` (torch engine), `mlx` (mlx engine), `pyyaml` (config files)
Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pytest-cov`
