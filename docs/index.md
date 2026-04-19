# MacFleet

**Distributed ML training on Apple Silicon.** Pool your Macs into a cluster
in 5 seconds, run PyTorch or MLX across them, keep zero cloud spend.

```python
import macfleet

with macfleet.Pool(enable_pool_distributed=True) as pool:
    pool.train(my_model, dataset, epochs=3)
```

## Why MacFleet

Apple Silicon is everywhere. Every researcher, student, and founder with
a MacBook Pro, Mac mini, or Mac Studio has a serious ML machine on their
desk. What's missing: a way to team them up.

- **PyTorch on MPS** has no distributed story. NCCL is CUDA-only. Gloo
  is broken on MPS. Single-GPU-on-MPS only.
- **MLX is native** but most researchers' code is still PyTorch.
- **Cloud is expensive** and the iteration loop is slow.

MacFleet fills that gap. Any two Macs on the same WiFi can pool their
GPUs and run a training loop together. Security is baked in (HMAC +
TLS). Adaptive compression keeps WiFi viable for gradient sync. The
framework-agnostic core lets you pick your engine (`torch` or `mlx`)
per call.

## The five-minute path

1. `pip install macfleet`
2. On Mac #1: `macfleet join --bootstrap`
3. On Mac #2: scan the QR with iPhone camera → tap → done
4. Both Macs: `python train.py`

## Current state: v2.2

- `Pool.join` / `Pool.nodes` / `Pool.world_size` wired to live discovery
- `@macfleet.task` registry replaces cloudpickle on the wire
- APING v2 handshake + rate-limited heartbeat
- Token pairing via QR + pasteboard
- Dashboard wired to agent state

See the [changelog](https://github.com/vikranthreddimasu/MacFleet/releases)
for details.
