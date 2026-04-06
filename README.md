# MacFleet

**Pool Apple Silicon Macs into a distributed ML training cluster.**

Turn your spare MacBooks, Mac Minis, and Mac Studios into one big GPU. MacFleet connects them over WiFi, Ethernet, or Thunderbolt and splits training across all of them automatically.

```
  macfleet join                macfleet join               macfleet join
 ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
 │  MacBook Pro  │◄────────►│  MacBook Air  │◄────────►│  Mac Studio   │
 │  M4 Pro       │  WiFi /  │  M4           │  WiFi /  │  M4 Ultra     │
 │  16 GPU cores │  ETH /   │  10 GPU cores │  ETH /   │  60 GPU cores │
 │  48 GB RAM    │  TB4     │  16 GB RAM    │  TB4     │  192 GB RAM   │
 │  weight: 0.35 │           │  weight: 0.15 │           │  weight: 0.50 │
 └──────────────┘           └──────────────┘           └──────────────┘
         ▲                          ▲                          ▲
         └──────────────────────────┴──────────────────────────┘
                        Ring AllReduce (gradient sync)
```

---

## Install

```bash
pip install macfleet
```

With PyTorch support:
```bash
pip install macfleet[torch]
```

With Apple MLX support:
```bash
pip install macfleet[mlx]
```

With everything:
```bash
pip install macfleet[all]
```

---

## Quick Start

### 1. Join the pool (on each Mac)

```bash
macfleet join
```

That's it. Each Mac auto-discovers the others on the same network using mDNS/Bonjour. No IP addresses, no config files.

### 2. Train a model

**Python SDK (recommended):**

```python
import macfleet
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

with macfleet.Pool() as pool:
    result = pool.train(
        model=model,
        dataset=(X_train, y_train),
        epochs=10,
        batch_size=128,
        loss_fn=nn.CrossEntropyLoss(),
    )

print(f"Final loss: {result['loss']:.4f}")
```

**One-liner:**

```python
macfleet.train(model=model, dataset=(X, y), epochs=10)
```

**Decorator:**

```python
@macfleet.distributed(engine="torch")
def my_training():
    # your training code here
    ...
```

**CLI:**

```bash
macfleet train                        # demo training on synthetic data
macfleet train my_script.py           # run your training script
```

### 3. Train with MLX (Apple's native framework)

```python
import mlx.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def __call__(self, x):
        return self.linear2(nn.relu(self.linear1(x)))

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")

with macfleet.Pool(engine="mlx") as pool:
    result = pool.train(
        model=MyModel(),
        dataset=(X_train, y_train),
        epochs=10,
        loss_fn=loss_fn,
    )
```

---

## CLI Reference

| Command             | What it does                                          |
|---------------------|-------------------------------------------------------|
| `macfleet join`     | Join the pool. Discovers peers via mDNS automatically |
| `macfleet leave`    | Gracefully leave the pool                             |
| `macfleet status`   | Show all pool members, network links, thermals        |
| `macfleet info`     | Show local hardware (chip, GPU cores, RAM, thermals)  |
| `macfleet train`    | Run demo training or submit a training script         |
| `macfleet bench`    | Benchmark compute, network throughput, or allreduce   |
| `macfleet diagnose` | Health check: MPS, thermals, network, memory          |

Options for common commands:

```bash
macfleet join --name "studio"          # custom node name
macfleet join --port 50051             # custom port
macfleet bench --type compute          # benchmark GPU compute
macfleet bench --type network          # benchmark network throughput
macfleet bench --type allreduce        # benchmark distributed allreduce
```

---

## How It Works

MacFleet uses **data parallelism**: every Mac holds a full copy of the model, trains on a portion of the data, and averages gradients after each step.

```
Step 1: Split batch across nodes (weighted by GPU power)
   Node A gets 35% of batch (16 GPU cores)
   Node B gets 15% of batch (10 GPU cores)
   Node C gets 50% of batch (60 GPU cores)

Step 2: Each node runs forward + backward pass locally

Step 3: AllReduce averages gradients across all nodes
   Uses Ring AllReduce for efficient N-node communication

Step 4: Each node applies the same averaged gradients
   Models stay identical across all nodes
```

### Key design decisions

- **Framework-agnostic core.** The communication layer only uses numpy. It never imports PyTorch or MLX. This means both engines work through the same pool/network/compression infrastructure.

- **Adaptive compression.** Gradient compression auto-selects based on your network:

  | Network       | Compression  | Ratio | 100MB gradients become |
  |---------------|-------------|-------|------------------------|
  | Thunderbolt 4 | None        | 1x    | 100 MB                 |
  | Ethernet      | TopK 10% + FP16 | ~20x  | ~5 MB              |
  | WiFi          | TopK 1% + FP16  | ~200x | ~500 KB            |

- **Heterogeneous scheduling.** Faster Macs get bigger batches. The scheduler continuously re-profiles throughput and adjusts for thermal throttling:

  | Thermal State | Workload |
  |--------------|----------|
  | Nominal      | 100%     |
  | Fair         | 90%      |
  | Serious      | 70%      |
  | Critical     | 30%      |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI: macfleet join | status | train | bench | info | diagnose  │
│  SDK: macfleet.Pool() | macfleet.train() | @distributed         │
├─────────────────────────────────────────────────────────────────┤
│  Training: DataParallel | TrainingLoop | WeightedSampler        │
├─────────────────────────────────────────────────────────────────┤
│  Engines: TorchEngine (PyTorch+MPS) | MLXEngine (Apple MLX)    │
├─────────────────────────────────────────────────────────────────┤
│  Compression: TopK + FP16 + Adaptive (bandwidth-aware)          │
├─────────────────────────────────────────────────────────────────┤
│  Pool: Agent | Registry | Discovery | Scheduler | Heartbeat     │
├─────────────────────────────────────────────────────────────────┤
│  Communication: PeerTransport | WireProtocol | Collectives      │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring: Thermal | Health | Throughput | Dashboard           │
└─────────────────────────────────────────────────────────────────┘
```

### Project structure

```
macfleet/
  cli/            CLI commands (Click)
  comm/           TCP transport, wire protocol, ring allreduce
  compression/    TopK sparsification, FP16 quantization, adaptive pipeline
  engines/        Engine protocol, TorchEngine, MLXEngine
  monitoring/     Thermal, health, throughput tracking, Rich dashboard
  pool/           Agent, mDNS discovery, registry, scheduler, heartbeat
  sdk/            Pool API, train(), @distributed decorator
  training/       DataParallel, training loop, weighted sampler
```

---

## Python SDK Reference

### `macfleet.Pool`

```python
with macfleet.Pool(engine="torch") as pool:
    result = pool.train(
        model=model,              # PyTorch nn.Module or MLX nn.Module
        dataset=dataset,          # TensorDataset or (X, y) tuple
        epochs=10,                # number of epochs
        batch_size=128,           # global batch size (split across nodes)
        lr=0.001,                 # learning rate (if no optimizer provided)
        optimizer=optimizer,      # optional pre-configured optimizer
        loss_fn=loss_fn,          # optional loss function
        compression="none",       # "none", "light", "moderate", "aggressive"
    )

# result dict:
# {
#     "loss": 0.1234,
#     "loss_history": [0.9, 0.5, 0.3, 0.1],
#     "epochs": 10,
#     "time_sec": 45.2,
#     "steps": 500,
# }
```

Pool options:

| Parameter            | Default    | Description                          |
|---------------------|------------|--------------------------------------|
| `engine`            | `"torch"`  | `"torch"` or `"mlx"`                |
| `name`              | `None`     | Custom node name                     |
| `port`              | `50051`    | Communication port                   |
| `token`             | `None`     | Pool authentication token            |
| `discovery_timeout` | `3.0`      | mDNS discovery timeout (seconds)     |

### `macfleet.train()`

Convenience wrapper that creates a Pool, joins, trains, and returns results:

```python
result = macfleet.train(model=model, dataset=(X, y), epochs=10, engine="torch")
```

### `@macfleet.distributed`

Decorator that wraps your function in a Pool context:

```python
@macfleet.distributed(engine="torch", compression="adaptive")
def train():
    # your training code runs inside a Pool
    ...
```

---

## Advanced: Programmatic Distributed Training

For full control over multi-node training:

```python
import asyncio
from macfleet.engines.torch_engine import TorchEngine
from macfleet.comm.transport import PeerTransport
from macfleet.comm.collectives import CollectiveGroup
from macfleet.training.data_parallel import DataParallel

async def distributed_train():
    # 1. Setup engine
    engine = TorchEngine(device="auto")  # auto-selects MPS or CPU
    engine.load_model(model, optimizer)

    # 2. Setup communication
    transport = PeerTransport(local_id="node-0")
    await transport.start_server("0.0.0.0", 50052)
    await transport.connect("node-1", "192.168.1.100", 50052)

    # 3. Create collective group
    group = CollectiveGroup(
        rank=0, world_size=2,
        transport=transport,
        rank_to_peer={1: "node-1"},
    )

    # 4. Create data parallel strategy
    dp = DataParallel(engine, group, link_type=LinkType.ETHERNET)
    await dp.setup()  # broadcasts initial parameters

    # 5. Training loop
    for batch in dataloader:
        engine.zero_grad()
        loss = engine.forward(batch)
        engine.backward(loss)
        await dp.sync_gradients()  # allreduce + optional compression
        engine.step()

asyncio.run(distributed_train())
```

---

## When to Use MacFleet

MacFleet works best when:

- Your Macs are within ~3x compute capability of each other
- The model fits in the weakest Mac's RAM (with 30% headroom)
- You have Ethernet or better (WiFi works but reduces efficiency)
- Each Mac can sustain training for the full run (plugged in, lid open)

**Model size limits (data parallel):**

| Machine         | Usable RAM | Max Model Size |
|-----------------|-----------|----------------|
| Air 16GB        | ~10 GB    | ~800M params   |
| Pro 36GB        | ~28 GB    | ~3B params     |
| Pro 48GB        | ~40 GB    | ~5B params     |
| Studio 192GB    | ~180 GB   | ~20B+ params   |

**Quick check:** Run `macfleet bench` before training. If the predicted speedup is < 1.3x, a single powerful Mac may be more efficient.

---

## Development

```bash
git clone https://github.com/vikranthreddimasu/MacFleet.git
cd MacFleet
pip install -e ".[dev,all]"
```

```bash
make test          # 268 tests
make bench         # compute + network + allreduce benchmarks
make lint          # ruff + mypy
make format        # auto-format code
make clean         # remove caches and build artifacts
```

---

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- PyTorch 2.1+ (for torch engine)
- MLX 0.5+ (optional, for mlx engine)

---

## License

MIT
