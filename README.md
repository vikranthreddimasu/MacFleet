# MacFleet

**Distributed ML training across Apple Silicon Macs over Thunderbolt 4**

MacFleet turns multiple Apple Silicon MacBooks into a distributed training cluster, enabling data-parallel PyTorch training with automatic gradient synchronization, compression, and fault tolerance.

```
┌────────────────────────┐      Thunderbolt 4       ┌────────────────────────┐
│   MacBook Pro (M4 Pro) │     (~4 GB/s, <1ms)      │   MacBook Air (M4)     │
│        MASTER          │◄────────────────────────►│        WORKER          │
│                        │                          │                        │
│  • 16 GPU cores        │                          │  • 10 GPU cores        │
│  • 24 GB RAM           │                          │  • 16 GB RAM           │
│  • 62% workload        │                          │  • 38% workload        │
└────────────────────────┘                          └────────────────────────┘
```

## Features

- **Data Parallel Training**: Automatically split batches based on each node's compute capacity
- **Gradient Compression**: Top-K sparsification + FP16 for ~20x bandwidth reduction
- **Fault Tolerance**: Automatic detection and recovery when nodes disconnect
- **Zero Config Discovery**: Bonjour/zeroconf for automatic node discovery
- **MPS Optimized**: Native Apple Silicon GPU acceleration via Metal Performance Shaders

## Quick Start

### Installation

```bash
pip install macfleet
```

Or from source:

```bash
git clone https://github.com/macfleet/macfleet.git
cd macfleet
pip install -e .
```

### Hardware Setup

1. Connect two Macs with a Thunderbolt 4 cable
2. Configure the Thunderbolt Bridge network:
   - System Settings → Network → Thunderbolt Bridge
   - Mac Pro: IP `10.0.0.1`, Subnet `255.255.255.0`
   - Mac Air: IP `10.0.0.2`, Subnet `255.255.255.0`
3. Verify: `ping 10.0.0.2` from Pro, `ping 10.0.0.1` from Air

### Launch Cluster

On the **MacBook Pro** (master):
```bash
macfleet launch --role master
```

On the **MacBook Air** (worker):
```bash
macfleet launch --role worker --master 10.0.0.1
```

### Train a Model

```python
import torch
import torchvision
from macfleet import Trainer, ClusterConfig, TrainingConfig, NodeRole

# Standard PyTorch model
model = torchvision.models.resnet50()

# Configure training
training_config = TrainingConfig(
    epochs=10,
    batch_size=128,           # Split across nodes by weight
    learning_rate=0.1,
    compression="topk_fp16",  # 20x compression
)

# Configure cluster
cluster_config = ClusterConfig(
    role=NodeRole.MASTER,     # or NodeRole.WORKER
    master_addr="10.0.0.1",
)

# Create dataset
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=torchvision.transforms.ToTensor()
)

# Train!
trainer = Trainer(
    model=model,
    dataset=dataset,
    training_config=training_config,
    cluster_config=cluster_config,
)
trainer.fit()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MacFleet Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Trainer    │───►│  DDP Wrapper │───►│  Collectives │                   │
│  │              │    │ (MacFleetDDP │    │  (AllReduce) │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                    │                          │
│         ▼                   ▼                    ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Weighted   │    │  Compression │    │   Tensor     │                   │
│  │   Sampler    │    │  (TopK+FP16) │    │  Transport   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                  │                          │
│                                                  ▼                          │
│                             ┌────────────────────────────────────┐          │
│                             │     Thunderbolt 4 Bridge           │          │
│                             │     (~4 GB/s, <1ms latency)        │          │
│                             └────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Trainer** | High-level API for distributed training |
| **MacFleetDDP** | Distributed Data Parallel wrapper with gradient sync |
| **Collectives** | AllReduce, Broadcast, Scatter, Gather operations |
| **Compression** | Top-K sparsification + FP16 quantization |
| **Transport** | Async TCP for raw tensor transfers |
| **Discovery** | Bonjour/zeroconf for automatic node discovery |

## CLI Commands

```bash
# Launch nodes
macfleet launch --role master --port 50051
macfleet launch --role worker --master 10.0.0.1

# Check cluster status
macfleet status --master 10.0.0.1

# Diagnose local readiness before launching
macfleet diagnose --host 10.0.0.1
macfleet diagnose --master 10.0.0.1:50051

# Run benchmarks
macfleet benchmark --type bandwidth
macfleet benchmark --type allreduce

# System info
macfleet info
```

## Security

MacFleet can protect the gRPC control plane with a shared token. Set the same
token on every node before launching the coordinator, workers, or status command:

```bash
export MACFLEET_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
macfleet launch --role master --host 10.0.0.1
macfleet launch --role worker --master 10.0.0.1:50051
macfleet status --master 10.0.0.1:50051
```

If `MACFLEET_AUTH_TOKEN` is not set, MacFleet keeps legacy open control-plane
behavior for local experiments. Use `macfleet diagnose` to confirm whether auth
is configured and whether the local control/tensor ports are available.

## Benchmarks

### Tensor Transfer Throughput

| Size | Serialize | Deserialize | Throughput |
|------|-----------|-------------|------------|
| 1 MB | 1.1 ms | 0.8 ms | 1.0 GB/s |
| 10 MB | 1.5 ms | 0.8 ms | 8.7 GB/s |
| 50 MB | 6.7 ms | 6.2 ms | 7.6 GB/s |
| 100 MB | 13.1 ms | 12.4 ms | 7.8 GB/s |

### Gradient Compression

| Compression | Ratio | Overhead |
|-------------|-------|----------|
| None | 100% | 0 ms |
| Top-K (10%) | 10% | ~1 ms |
| FP16 | 50% | <1 ms |
| Top-K + FP16 | ~15% | ~2 ms |

### Estimated AllReduce Times

| Model | Params | Uncompressed | TopK+FP16 |
|-------|--------|--------------|-----------|
| ResNet-50 | 25M | 12.5 ms | 1.9 ms |
| GPT-2 (124M) | 124M | 62 ms | 9.3 ms |
| ViT-Base | 86M | 43 ms | 6.5 ms |

## Examples

### ResNet-50 on CIFAR-10

```bash
python examples/train_resnet.py --epochs 10 --batch-size 128
```

### GPT-2 Fine-tuning

```bash
python examples/train_gpt2.py --epochs 3 --batch-size 8
```

## Configuration

### TrainingConfig

```python
TrainingConfig(
    epochs=10,                    # Training epochs
    batch_size=128,               # Total batch (split by node weights)
    learning_rate=0.1,            # Initial learning rate
    compression="topk_fp16",      # "none", "topk", "fp16", "topk_fp16"
    topk_ratio=0.1,               # Keep top 10% of gradients
    checkpoint_every=2,           # Checkpoint every N epochs
    device="mps",                 # "mps", "cpu"
)
```

### ClusterConfig

```python
ClusterConfig(
    role=NodeRole.MASTER,         # MASTER or WORKER
    master_addr="10.0.0.1",       # Master IP address
    master_port=50051,            # gRPC control port
    tensor_port=50052,            # Tensor transfer port
    discovery_enabled=True,       # Use Bonjour discovery
)
```

## Requirements

- macOS 14.0+ (Sonoma) or 15.0+ (Sequoia)
- Python 3.11+
- PyTorch 2.1+ with MPS support
- Thunderbolt 4 cable for multi-node

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_transport.py

# Run with coverage
pytest --cov=macfleet tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## Roadmap

- [ ] Ring-AllReduce for N>2 nodes
- [ ] Pipeline parallelism for large models
- [ ] Communication/computation overlap
- [ ] Metal kernel optimization
- [ ] Multi-model training support
