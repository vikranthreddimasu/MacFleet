# Training with Pool.train

The core API:

```python
pool.train(
    model,                    # nn.Module (torch) or mlx model
    dataset,                  # Dataset, (X, y) tuple, or iterable
    epochs=10,
    batch_size=128,           # global batch; split across ranks
    lr=0.001,
    optimizer=None,           # default: Adam(lr)
    loss_fn=None,             # default: model returns loss directly
    engine=None,              # "torch" or "mlx"; None → Pool's default
    compression="none",       # "none" | "topk" | "fp8" (auto-picked from link)
)
```

Returns a dict: `{"loss": ..., "epochs": ..., "time_sec": ..., "steps": ...}`.

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
loadable. Ctrl-C is safe. Power loss is safe.

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

In distributed mode, `pool.join()` blocks until quorum is met. If the
timeout expires, you get:

```
TimeoutError: No quorum within 30.0s: saw 1 node(s), need 2.
Run 'macfleet status' to check discovery, or pass
peers=['<ip>:50051'] to connect manually.
```
