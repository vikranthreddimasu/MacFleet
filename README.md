# MacFleet

**Pool Apple Silicon Macs into a distributed ML training cluster.**

Turn spare MacBooks, Mac Minis, and Mac Studios into one big GPU. MacFleet connects them over Thunderbolt, Ethernet, or WiFi and splits training across all of them automatically.

```
  macfleet join              macfleet join            macfleet join
 ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
 │  MacBook Pro │◄────────►│  MacBook Air │◄────────►│  Mac Studio  │
 │  M4 Pro      │  WiFi /  │  M4          │  WiFi /  │  M4 Ultra    │
 │  16 GPU cores│  ETH /   │  10 GPU cores│  ETH /   │  60 GPU cores│
 │  48 GB RAM   │  TB4     │  16 GB RAM   │  TB4     │  192 GB RAM  │
 └──────────────┘          └──────────────┘          └──────────────┘
         ▲                          ▲                          ▲
         └──────────────────────────┴──────────────────────────┘
                        Ring AllReduce (gradient sync)
```

## Install

```bash
pip install macfleet            # core
pip install macfleet[torch]     # + PyTorch
pip install macfleet[mlx]       # + Apple MLX
pip install macfleet[all]       # everything
```

## Quick Start

**1. Join the pool** (run on each Mac):

```bash
macfleet join
```

No config files, no IP addresses. Macs find each other automatically via mDNS/Bonjour.

**2. Train:**

```python
import macfleet
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

with macfleet.Pool() as pool:
    result = pool.train(model=model, dataset=(X_train, y_train), epochs=10)
```

## Features

- **Dual engine** — PyTorch (MPS) and Apple MLX, same pool infrastructure
- **Zero config** — mDNS discovery, no coordinator setup, no config files
- **Adaptive compression** — auto-selects TopK + FP16 based on link speed (1x–200x reduction)
- **Heterogeneous scheduling** — faster Macs get bigger batches, adjusts for thermal throttling
- **Secure** — HMAC mutual authentication, mandatory TLS, fleet-scoped discovery, gradient validation
- **Framework-agnostic core** — communication layer uses only numpy, never imports torch or mlx

## Security

Protect your fleet with a shared token:

```bash
macfleet join --token my-secret-token
macfleet join --token my-secret-token --fleet-id research-team
```

When a token is set:
- **Fleet isolation** — nodes with different tokens are invisible to each other on the network
- **Mutual authentication** — HMAC-SHA256 challenge-response on every connection
- **Encryption** — TLS enabled automatically (mandatory with auth)
- **Authenticated heartbeat** — HMAC-signed liveness probes, replay-resistant
- **Gradient validation** — rejects NaN, Inf, and extreme magnitudes (anti-poisoning)

No token = open fleet, fully backward compatible.

## CLI

```
macfleet join       Join the pool (auto-discovers peers)
macfleet status     Show pool members and network info
macfleet info       Show local hardware profile
macfleet train      Run training (demo or custom script)
macfleet bench      Benchmark compute, network, or allreduce
macfleet diagnose   System health check
```

## How It Works

MacFleet uses **data parallelism**: every Mac holds a full copy of the model, trains on a weighted portion of the data, and averages gradients via Ring AllReduce after each step.

| Network       | Compression     | 100 MB gradients become |
|---------------|-----------------|-------------------------|
| Thunderbolt 4 | None            | 100 MB                  |
| Ethernet      | TopK 10% + FP16 | ~5 MB                   |
| WiFi          | TopK 1% + FP16  | ~500 KB                 |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- PyTorch 2.1+ or MLX 0.5+

## Development

```bash
git clone https://github.com/vikranthreddimasu/MacFleet.git
cd MacFleet
pip install -e ".[dev,all]"
make test       # 373 tests
make lint       # ruff + mypy
```

## License

MIT
