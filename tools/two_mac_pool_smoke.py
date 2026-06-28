"""Beginner-friendly two-Mac Pool.train smoke test.

Run this same file on both Macs after they have been paired. It exercises the
real high-level SDK path:

    Pool(enable_pool_distributed=True) -> PoolAgent -> discovery -> mesh
    -> DataParallel -> CollectiveGroup allreduce -> identical final params

Success means both Macs print the same params_sha256 and degraded=False.
"""

from __future__ import annotations

import json
import os
import socket

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import macfleet


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _peers_from_env() -> list[str]:
    raw = os.environ.get("MACFLEET_PEERS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_dataset(n_samples: int = 512) -> TensorDataset:
    torch.manual_seed(1234)
    x = torch.randn(n_samples, 8)
    y = (x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2] > 0).long()
    return TensorDataset(x, y)


def make_model() -> nn.Module:
    # Different seed per host proves rank 0's setup broadcast is working.
    host_seed = sum(socket.gethostname().encode("utf-8")) % 10_000
    torch.manual_seed(9000 + host_seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


def main() -> None:
    name = os.environ.get("MACFLEET_NAME", socket.gethostname().split(".")[0])
    device = os.environ.get("DEVICE", "cpu")
    compression = os.environ.get("COMPRESSION", "none")

    with macfleet.Pool(
        name=name,
        enable_pool_distributed=True,
        quorum_size=_env_int("QUORUM_SIZE", 2),
        quorum_timeout_sec=_env_float("QUORUM_TIMEOUT", 45.0),
        rendezvous_timeout_sec=_env_float("RENDEZVOUS_TIMEOUT", 90.0),
        peers=_peers_from_env(),
    ) as pool:
        print(f"[{name}] joined pool: world_size={pool.world_size}")
        for node in sorted(pool.nodes, key=lambda item: str(item["node_id"])):
            print(
                f"[{name}] node {node['hostname']} "
                f"{node['chip_name']} data:{node['data_port']}"
            )

        result = pool.train(
            model=make_model(),
            dataset=make_dataset(),
            epochs=_env_int("EPOCHS", 3),
            batch_size=_env_int("BATCH_SIZE", 64),
            lr=float(os.environ.get("LR", "0.03")),
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
            compression=compression,
            distributed=True,
        )

    summary = {
        "rank": result.get("rank"),
        "world_size": result.get("world_size"),
        "steps": result.get("steps"),
        "loss_history": result.get("loss_history"),
        "avg_sync_time_sec": result.get("avg_sync_time_sec"),
        "degraded": result.get("degraded"),
        "unsynced_steps": result.get("unsynced_steps"),
        "last_sync_error": result.get("last_sync_error"),
        "params_sha256": result.get("params_sha256"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print("PASS when both Macs show:")
    print("  - world_size == 2")
    print("  - degraded == false")
    print("  - unsynced_steps == 0")
    print("  - identical params_sha256")


if __name__ == "__main__":
    main()
