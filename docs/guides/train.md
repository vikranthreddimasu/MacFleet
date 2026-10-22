# Training with Pool.train

Set `enable_pool_distributed=True` on `Pool()` to allreduce gradients across peers.

The core API:

```python
pool.train(
    model,                    # nn.Module (torch) or mlx model
    dataset,                  # sized Dataset or (X, y) tuple
    epochs=10,
    batch_size=128,           # global batch; split across ranks
    lr=0.001,
    optimizer=None,           # default: Adam(lr)
    loss_fn=None,             # default: model returns loss directly
    engine=None,              # "torch" or "mlx"; None → Pool's default
    compression="none",       # "none" | "light" | "adaptive"
    distributed=None,         # None=follow Pool distributed intent,
                              # True=require peers, False=force single-node
)
```

Returns a dict: `{"loss": ..., "epochs": ..., "time_sec": ..., "steps": ...}`.
In distributed mode it also includes `rank`, `world_size`,
`avg_sync_time_sec`, `params_sha256`, `degraded`, `unsynced_steps`,
`validation_fallback_steps`, and `last_sync_error`. The hash matches
across ranks iff the fleet stayed in sync (same check
`tools/two_mac_real_train.py` uses).

`degraded=True` means at least one step used a local fallback instead of
a clean synchronized gradient. `unsynced_steps` counts allreduce or
connection failures. `validation_fallback_steps` counts NaN/Inf,
metadata, or shape-validation failures that were discarded before they
could poison the model.

Torch accepts a sized dataset, an input tensor, or paired `(X, y)` arrays.
Input-only batches pass all samples to the model; a custom loss receives
`loss_fn(outputs)`. Paired batches use `loss_fn(outputs, targets)`. MLX requires
paired arrays and uses `loss_fn(model, inputs, targets)`. Both engines reject
paired arrays with different sample counts.

## Mutation semantics

`pool.train` mutates `model` in place. This matches PyTorch's idiom —
after `loss.backward(); optimizer.step()`, the module's parameters
have moved. The returned dict is the training summary, not the final
state. If you want to save weights, save from the same `model`
object after `pool.train` returns.

## Batching across ranks

`batch_size` is the **global** batch size. If your fleet has 4
Macs, each rank gets `batch_size // world_size` samples per step, and
the gradient is averaged across all 4 after each step.

This means you typically want `batch_size >= world_size` (so every
rank gets at least one sample). The A4 guard will raise
`DatasetSizeError` before training starts if you violate this — see
[Dataset guard](#dataset-guard) below.

## Dataset guard

Before the optimizer and dataloader come up, `pool.train` checks
that your dataset can produce at least one global batch. Three cases
get distinct errors:

### Empty dataset

```
DatasetSizeError: Dataset is empty. Check that your DataLoader/Dataset
produces samples before calling pool.train().
```

### Batch size smaller than world size

```
DatasetSizeError: batch_size 3 is smaller than world_size 4; each
rank gets 0 samples per step. Increase batch_size to at least 4, or
run on fewer nodes.
```

### Dataset has some samples but fewer than batch_size

```
DatasetSizeError: Dataset has 50 samples but needs >= 128 to run at
least 1 batch(es) of size 128 across 1 rank(s). Shortfall: 78 samples.
Either: (a) collect more data, (b) reduce batch_size to 50 or
smaller, or (c) reduce world_size.
```

## Atomic checkpoints

`macfleet.utils.atomic_write` gives you crash-safe saves:

```python
from macfleet.utils.atomic_write import atomic_write_via
import torch

def save_checkpoint(epoch, loss):
    atomic_write_via(
        f"runs/exp-42/epoch-{epoch}.pt",
        lambda p: torch.save({"epoch": epoch, "loss": loss,
                               "state_dict": model.state_dict()}, p),
    )
```

The save either succeeds completely or leaves the previous checkpoint
untouched. No half-written .pt files.

If training crashes mid-save, your previous epoch checkpoint is still
loadable after an interrupted write. Use `fsync_dir=True` when saving if the
directory entry also needs to be synced for durability across power loss.

## Single-node vs distributed

The feature flag `enable_pool_distributed` gates distributed behavior.

**Single-node** (flag off, default):

```python
with macfleet.Pool() as pool:
    pool.train(...)     # runs locally, no agent, no mDNS
```

**Distributed** (flag on, requires paired Macs):

```python
with macfleet.Pool(
    enable_pool_distributed=True,
    quorum_size=2,       # wait for at least 2 nodes
    quorum_timeout_sec=30,
) as pool:
    pool.train(...)     # ring allreduce across pool.world_size nodes
```

Distributed training is **SPMD: run the same script on every Mac.**
Each Pool joins the fleet and waits for quorum; `pool.train()` then
forms a gradient mesh over the data ports, broadcasts rank 0's
model, buffers and optimizer state, and allreduces gradients after every backward pass.
Ranks are derived from sorted node ids, so no extra coordination
round is needed.

The Macs don't have to call `pool.train()` at the same instant — the
mesh rendezvous retries for `rendezvous_timeout_sec` (Pool kwarg,
default 60s) while you start the script on each machine. If a peer
never shows up:

```
MeshFormationError: Could not connect to peer mac-mini-9f2e at
192.168.1.10:50052 within 60s. Check that the peer is running the
same training script, and that 'macfleet status' shows it as alive.
```

A wrong fleet token fails immediately (`PeerAuthError`) instead of
retrying until the deadline.

In distributed mode, `pool.join()` blocks until quorum is met. If the
timeout expires, you get:

```
TimeoutError: No quorum within 30.0s: saw 1 node(s), need 2.
Run 'macfleet status' to check discovery, or pass
peers=['<ip>:50051'] to connect manually.
```

## Failure and compatibility contract

Distributed intent never silently falls back to local execution. Pass
`distributed=False` explicitly for a local run. `@macfleet.distributed` is
removed as an execution path because it never synchronized gradients; use the
explicit Pool context above.

Setup checks model layout, engine, dataset size, epoch count and batch schedule.
These checks do not prove two arbitrary Python programs have the same semantics.
Use identical scripts, data ordering, engine versions and loss functions.
Rank 0 supplies initial model and optimizer state. Training uses equal shards;
capacity-weighted rescheduling is a separate low-level API.

Peer loss, malformed gradients and non-finite gradients abort by default. Rejoin
and restart all ranks from the same checkpoint after a failure; in-flight training
does not elastically shrink or resume on rediscovery. The result hash compares
parameters, not every buffer or arbitrary Python model attribute.

`TrainingConfig.max_grad_norm` clips synchronized gradients. Automatic checkpoint
intervals are unsupported and rejected; use `on_step` and `engine.state_dict()`.
Unknown `Pool.train` keyword arguments are rejected instead of ignored.


The global batch size must be divisible by the number of participating nodes.
MacFleet rejects a non-divisible size instead of silently rounding it down.
Pool listeners currently use IPv4. On recent macOS versions, allow the Python
runtime or its launching application under **Privacy & Security → Local Network**.
If `nc` can reach a peer but Python reports `No route to host`, check that permission
before changing fleet tokens or retrying training. IPv6-only fleets are unsupported.


Engine checkpoints now validate optimizer identity and configuration. Torch
checkpoints include the parameter names and order for each optimizer group;
MLX checkpoints include optimizer settings. Checkpoints from older MacFleet
versions that lack this metadata cannot safely restore optimizer state and
are rejected. For a deliberate weights-only migration, load the old weights
with the framework's native model APIs, create a fresh optimizer, and save a
new MacFleet checkpoint. This restarts optimizer history; it is not an exact resume.


The low-level `DistributedBatchSampler` divides each global batch by the same
weights on every rank, with at least one sample per rank. It rounds allocations
so their sum equals the global batch size. With `drop_last=True`, it drops the
last incomplete global batch. Otherwise, a final batch smaller than the number
of ranks repeats samples from that batch so every rank can participate.

The composable `training_loop` checks batch counts, epoch counts and gradient
clipping settings across ranks before training, and advances either the batch
sampler or ordinary sampler each epoch.
Distributed loaders must expose their length. Use identical weights, seed and
data ordering on every rank. `DataParallel` averages gradients equally across
ranks; with unequal batch sizes, callers must scale their loss gradients if they
need a mean weighted by sample count. `Pool.train` uses equal per-rank batches.
