# Dashboard

MacFleet ships a Rich TUI dashboard that renders cluster health, per-
node thermal state, training progress, and network stats.

## One-liner

```python
import macfleet
import time
from macfleet.monitoring.dashboard import Dashboard

with macfleet.Pool(enable_pool_distributed=True) as pool:
    with Dashboard() as dash:
        while True:
            dash.update_nodes(pool.dashboard_snapshot())
            time.sleep(2.0)
```

You get:

```
┌──────────────────────── MacFleet Cluster Dashboard ────────────────────────┐
│ Node               Chip             GPU   Health   Status                 │
│ mac-mini-studio    Apple M2 Max     30    1.00     HEALTHY                │
│ macbook-pro        Apple M1 Pro     16    0.92     HEALTHY                │
├────────────────────── Training ────────────────────────────────────────────┤
│ Epoch 3/10   Step 240   Loss 0.4231   Throughput 1280 samples/s           │
│ Sync time: 12ms avg    Compression: 5.2x                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

## Wiring it into your training loop

`pool.dashboard_snapshot()` returns a list of `NodeHealth` objects.
`Dashboard.update_nodes(list)` rerenders immediately. `update_training`
and `update_network` take finer-grained slices:

```python
from macfleet.monitoring.dashboard import Dashboard

with Dashboard() as dash:
    for epoch in range(10):
        for step, batch in enumerate(loader):
            loss = training_step(batch)
            if step % 10 == 0:
                dash.update_training(
                    epoch=epoch,
                    total_epochs=10,
                    step=step,
                    loss=loss,
                )
                dash.update_nodes(pool.dashboard_snapshot())
```

## Health score

Each node gets a composite 0.0..1.0 score. The `agent_adapter` bucket
maps it to a status:

| Score | Status | Color |
|-------|--------|-------|
| >= 0.8 | HEALTHY | green |
| >= 0.5 | DEGRADED | yellow |
| < 0.5 | UNHEALTHY | red |

Penalties compound:
- Thermal throttling: workload multiplier from `ThermalState`
- High memory pressure: >80% → progressive penalty
- Low battery (unplugged): <20% soft penalty, <10% hard penalty
- Connection failures: each failure cuts 10% off the score

See `macfleet/monitoring/health.py::NodeHealth.health_score` for the
full formula.

## Headless health checks

Not all workflows want a TUI. `pool.dashboard_snapshot()` returns the
same data as a list of dataclasses — dump it to JSON for a Slack bot,
a PagerDuty hook, or a simple CI gate:

```python
import json
from dataclasses import asdict

with macfleet.Pool(enable_pool_distributed=True) as pool:
    snap = pool.dashboard_snapshot()
    print(json.dumps([asdict(n) for n in snap], default=str, indent=2))
```

## What's not here yet

- **Peer thermal/memory gossip**: peer health shows thermal state from
  the last heartbeat, but not live memory or battery. Issue 12 in the
  roadmap expands the heartbeat payload to carry more.
- **Web dashboard**: terminal-only today. A FastAPI + HTMX version
  is on the v2.3 roadmap.
