# Quickstart

Your first distributed training run, start to finish.

## Single-Mac sanity check

Before adding a second Mac, verify the basics work solo:

```python
import macfleet
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2),
        )
    def forward(self, x):
        return self.net(x)

X = torch.randn(1000, 4)
y = torch.randint(0, 2, (1000,))

with macfleet.Pool() as pool:
    result = pool.train(
        model=MLP(),
        dataset=(X, y),
        epochs=5,
        batch_size=64,
        loss_fn=nn.CrossEntropyLoss(),
    )

print(result)  # {"loss": 0.4, "epochs": 5, "time_sec": 2.1, "steps": 80}
```

That runs single-node, no agent, no discovery. Good baseline.

## Two-Mac setup

### Mac #1 — the first on the fleet

```bash
macfleet join --bootstrap
```

That starts the agent and prints a QR code:

```
Fleet pairing URL: macfleet://pair?token=...&fleet=default
Scan this QR from a second Mac's iPhone camera:
████ ▄▄▄▄▄ █▀█ █▄▀▄▀ ██▄ ▄▄▄▄▄ ████
...
```

Leave the terminal open. The agent runs until you Ctrl-C.

### Mac #2 — the second Mac

Option A: point your iPhone camera at the QR on Mac #1. Tap the
notification. The URL is copied.

Option B: your Macs share a pasteboard via Handoff. Just run:

```bash
macfleet pair    # reads URL from pasteboard
macfleet join    # joins the fleet using the token now on disk
```

After `pair`, both Macs show each other in `macfleet status`:

```
$ macfleet status
Node                    IP                Chip             GPU  Fleet
mac-mini-studio         192.168.1.10      Apple M2 Max     30   (coordinator)
macbook-pro             192.168.1.11      Apple M1 Pro     16
```

### From Python

Now your Python code can use `enable_pool_distributed=True`:

```python
with macfleet.Pool(
    enable_pool_distributed=True,
    quorum_size=2,          # wait for both Macs
    quorum_timeout_sec=30,  # bail if second Mac isn't up in 30s
) as pool:
    print(f"World size: {pool.world_size}")
    for n in pool.nodes:
        print(f"  {n['hostname']}: {n['chip_name']}, {n['gpu_cores']} GPU cores")

    pool.train(model, dataset, epochs=3)
```

## Dispatching one-off tasks

Not every job is a training loop. Register any function as a task:

```python
@macfleet.task
def resize(image_path: str) -> dict:
    # runs on any node in the fleet
    ...
    return {"width": w, "height": h}

with macfleet.Pool(enable_pool_distributed=True) as pool:
    results = pool.map(resize, image_paths)
```

One gotcha: both Macs must have imported the module that declares
`resize` before the dispatch, otherwise the remote worker throws
`TaskNotRegisteredError`. See [Registered tasks](../guides/tasks.md)
for the full story.

## Next steps

- [Training guide](../guides/train.md) — full API, optimizers, callbacks
- [Dashboard](../guides/dashboard.md) — live Rich TUI
- [Pairing](pairing.md) — all three pairing flows (QR / pasteboard / URL)
