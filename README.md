# MacFleet

**Distributed ML training on Apple Silicon.** Pool your Macs into a
cluster in 5 seconds, run PyTorch or MLX across them, keep zero cloud
spend.

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

## Why MacFleet

Apple Silicon is everywhere. Every researcher, student, and founder
has a serious ML machine on their desk. What's missing is a way to
team them up.

- **PyTorch on MPS has no distributed story.** NCCL is CUDA-only.
  Gloo is broken on MPS. Single-GPU-on-MPS only.
- **MLX is native** but most researchers' code is still PyTorch.
- **Cloud is expensive** and the iteration loop is slow.

MacFleet fills that gap. Any two Macs on the same WiFi can pool their
GPUs. Security is baked in (HMAC + TLS). Adaptive compression keeps
WiFi viable for gradient sync. The framework-agnostic core lets you
pick your engine (`torch` or `mlx`) per call.

## Install

```bash
pip install macfleet                    # core
pip install "macfleet[torch]"           # + PyTorch
pip install "macfleet[mlx]"             # + Apple MLX
pip install "macfleet[all]"             # everything
```

## The 5-minute path

**1. Scaffold a starter script:**

```bash
macfleet quickstart
# Wrote my_macfleet_demo.py
```

**2. Run it:**

```bash
python my_macfleet_demo.py
# Pool world size: 1
# Training done: {'loss': 0.31, 'epochs': 10, 'time_sec': 1.4}
```

**3. Pair a second Mac:**

On Mac #1:

```bash
macfleet join
# first run auto-generates a fleet token and prints a QR code +
# pairing URL (also copied to the pasteboard). Use --bootstrap to
# re-print pairing info later.
```

On Mac #2 (same Apple ID → Handoff pasteboard sync):

```bash
macfleet pair && macfleet join
```

Or: scan the QR from Mac #1 with your iPhone camera. Tap the link.
Done.

**4. Set `enable_pool_distributed=True` and run the same script on both
Macs** — training now spans both: the pool forms a gradient mesh, rank 0
broadcasts initial weights, and every step's gradients are averaged
across the fleet. The result dict's `params_sha256` matches on both
Macs when the fleet stayed in sync.

## Features

- **Dual engine** — PyTorch (MPS) and Apple MLX, same pool infrastructure
- **Zero config** — mDNS discovery, no coordinator setup, no config files
- **Safe task dispatch** — `@macfleet.task` registry + msgpack args
  (no cloudpickle on the wire)
- **Adaptive compression** — auto-selects TopK + FP16 based on link
  speed. On 2-Mac fleets the sparse payload itself crosses the wire:
  ~60x less gradient traffic on WiFi (TopK 1% + FP16). N≥3 rings still
  transmit dense (sparse ring merge: TODOS.md Issue 3)
- **Heterogeneous scheduling** — faster Macs get bigger batches,
  adjusts for thermal throttling
- **Secure by default** — auto-generated fleet tokens (scrypt-derived
  keys), client-first HMAC mutual auth (servers reveal nothing to
  unauthenticated peers), mandatory TLS with channel-bound handshakes
  (MITM-relay resistant), per-IP rate limiting
- **Framework-agnostic core** — communication layer uses only numpy,
  never imports torch or mlx

## Security

Security is on by default. The first `macfleet join` auto-generates a
fleet token at `~/.macfleet/token` (mode 0600). See the
[security reference](docs/reference/security.md) for the full threat
model.

Short version:

- **Fleet isolation** — nodes with different tokens can't see each
  other on the network (mDNS service type is scoped by fleet hash)
- **Mutual authentication** — HMAC-SHA256 challenge-response on every
  connection, plus signed hardware profile exchange (v2.2)
- **Encryption** — TLS mandatory whenever auth is enabled
- **Rate limiting** — 5 failed auth attempts per IP → 5-minute ban,
  exponential backoff in between (heartbeat read timeout tightened to
  1s to stop slowloris)
- **No cloudpickle over the wire** — `@macfleet.task` routes
  registered callables by name, not by pickled closures

## CLI

```
macfleet join         Join the pool (auto-discovers peers)
macfleet pair         Read a pairing URL from pasteboard / stdin
macfleet status       Show pool members and network info
macfleet info         Show local hardware profile
macfleet train        Run training (demo or custom script)
macfleet bench        Benchmark compute, network, or allreduce
macfleet doctor       System health check
macfleet quickstart   Write a starter training script
```

## How it works

MacFleet uses **data parallelism**: every Mac holds a full copy of the
model, trains on a weighted portion of the data, and averages
gradients via Ring AllReduce after each step.

The compression layer (TopK + FP16) is applied locally before the
allreduce; v2.2 transmits dense gradients on the wire (sparse
allreduce is on the v2.3 roadmap as Issue 3). The bandwidth savings
table below describes the **target** ratios once sparse-on-wire ships:

| Network       | Compression     | 100 MB gradients (v2.3 target) |
|---------------|-----------------|--------------------------------|
| Thunderbolt 4 | None            | 100 MB                         |
| Ethernet      | TopK 10% + FP16 | ~5 MB                          |
| WiFi          | TopK 1% + FP16  | ~500 KB                        |

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- PyTorch 2.1+ or MLX 0.5+ (optional, pick your engine)

## Documentation

Full docs: run `mkdocs serve` after `pip install "macfleet[docs]"`, or
read the Markdown source in `docs/`:

- [Quickstart](docs/getting-started/quickstart.md)
- [Pairing flows](docs/getting-started/pairing.md)
- [Pool.train API](docs/guides/train.md)
- [@macfleet.task](docs/guides/tasks.md)
- [Dashboard](docs/guides/dashboard.md)
- [Security](docs/reference/security.md)
- [Wire protocol](docs/reference/protocol.md)

## Development

```bash
git clone https://github.com/vikranthreddimasu/MacFleet.git
cd MacFleet
pip install -e ".[dev,all]"
make test       # 447 tests
make lint       # ruff + mypy
```

## License

MIT
