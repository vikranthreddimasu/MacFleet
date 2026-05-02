"""Forensic-trace distributed training — every step logged to JSONL.

Writes /tmp/macfleet_trace_rank{N}.jsonl with one line per training
step capturing:
  - epoch / step / batch indices processed by THIS rank
  - local pre-sync gradient: first-5 values, L2 norm, SHA1
  - byte counters (cumulative sent/received) before + after the step
  - sync_gradients() wall time
  - synced post-allreduce gradient: first-5 values, L2 norm, SHA1
  - parameters after optimizer.step(): SHA1
  - timestamp (host-local monotonic + wall-clock)
  - loss, train accuracy

After both Macs finish, copy both .jsonl files to one Mac and run:
    python3 tools/compare_traces.py /tmp/macfleet_trace_rank0.jsonl /tmp/macfleet_trace_rank1.jsonl

The compare script does row-by-row checks:
  - local_grad_sha1 differs (different shards → different gradients)
  - synced_grad_sha1 matches (allreduce averaged them)
  - params_after_sha1 matches (identical update applied)
  - batch_indices differ (sampler gave each rank distinct samples)
  - byte counts mirror (rank 0 sent ≈ rank 1 received)

Optional verbose MacFleet internal logging:
    MACFLEET_LOG=DEBUG RANK=0 python3 tools/two_mac_traced_train.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Configure logging BEFORE importing macfleet so its loggers inherit our level
_LOG_LEVEL = os.environ.get("MACFLEET_LOG", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.engines.torch_engine import TorchEngine
from macfleet.training.data_parallel import DataParallel
from macfleet.training.sampler import WeightedDistributedSampler

PORT = int(os.environ.get("PORT", "60001"))
RANK = int(os.environ.get("RANK", "0"))
PEER_IP = os.environ.get("PEER_IP", "")
WORLD_SIZE = 2

# Smaller model than two_mac_real_train so the trace is easy to read.
EPOCHS = 4
N_SAMPLES = 2400
BATCH_SIZE = 32
INPUT_DIM = 16
HIDDEN = 64
N_CLASSES = 4

TRACE_PATH = f"/tmp/macfleet_trace_rank{RANK}.jsonl"

CONFIG = TransportConfig(connect_timeout_sec=15.0, recv_timeout_sec=30.0)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


_centers_cache: dict = {}


def _get_centers() -> torch.Tensor:
    if "centers" not in _centers_cache:
        g = torch.Generator().manual_seed(4242)
        _centers_cache["centers"] = torch.randn(N_CLASSES, INPUT_DIM, generator=g) * 2.5
    return _centers_cache["centers"]


def make_dataset(n_samples: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    centers = _get_centers()
    per_class = n_samples // N_CLASSES
    X = torch.zeros(per_class * N_CLASSES, INPUT_DIM)
    y = torch.zeros(per_class * N_CLASSES, dtype=torch.long)
    for c in range(N_CLASSES):
        idx = slice(c * per_class, (c + 1) * per_class)
        X[idx] = centers[c] + torch.randn(per_class, INPUT_DIM, generator=g) * 0.7
        y[idx] = c
    perm = torch.randperm(len(X), generator=g)
    return TensorDataset(X[perm], y[perm])


class IndexedDataLoader:
    """Wraps a DataLoader so we can capture which sample indices the
    sampler hands to this rank — proves both Macs see different data."""

    def __init__(self, dataset: TensorDataset, sampler, batch_size: int):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        idx_buf: list[int] = []
        for idx in self.sampler:
            idx_buf.append(int(idx))
            if len(idx_buf) == self.batch_size:
                xs = torch.stack([self.dataset[i][0] for i in idx_buf])
                ys = torch.stack([self.dataset[i][1] for i in idx_buf])
                yield idx_buf, xs, ys
                idx_buf = []
        if idx_buf:
            xs = torch.stack([self.dataset[i][0] for i in idx_buf])
            ys = torch.stack([self.dataset[i][1] for i in idx_buf])
            yield idx_buf, xs, ys

    def __len__(self) -> int:
        n = len(self.sampler)
        return (n + self.batch_size - 1) // self.batch_size


def _grad_summary(arr: np.ndarray) -> dict:
    return {
        "sha1": hashlib.sha1(arr.tobytes()).hexdigest(),
        "first5": [round(float(v), 6) for v in arr.flatten()[:5].tolist()],
        "norm": round(float(np.linalg.norm(arr)), 6),
        "size": int(arr.size),
    }


async def main() -> None:
    if RANK not in (0, 1):
        print(f"RANK must be 0 or 1, got {RANK}")
        sys.exit(1)
    if RANK == 1 and not PEER_IP:
        print("RANK=1 requires PEER_IP")
        sys.exit(1)

    transport = PeerTransport(local_id=f"node-{RANK}", config=CONFIG)
    if RANK == 0:
        await transport.start_server("0.0.0.0", PORT)
        print(f"[rank 0] listening on 0.0.0.0:{PORT} — waiting for rank 1...")
        while transport.connection_count < 1:
            await asyncio.sleep(0.5)
        print(f"[rank 0] rank 1 connected")
        rank_to_peer = {1: "node-1"}
        peer_id = "node-1"
    else:
        await transport.connect("node-0", PEER_IP, PORT)
        print(f"[rank 1] connected to {PEER_IP}:{PORT}")
        rank_to_peer = {0: "node-0"}
        peer_id = "node-0"

    group = CollectiveGroup(
        rank=RANK, world_size=WORLD_SIZE,
        transport=transport, rank_to_peer=rank_to_peer,
    )

    torch.manual_seed(42)
    model = TinyClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)

    dp = DataParallel(engine, group)
    await dp.setup()

    train_set = make_dataset(N_SAMPLES, seed=42)
    sampler = WeightedDistributedSampler(
        train_set, num_replicas=WORLD_SIZE, rank=RANK,
        shuffle=True, seed=0,
    )
    loader = IndexedDataLoader(train_set, sampler, BATCH_SIZE)

    n_params = sum(p.numel() for p in model.parameters())
    n_batches_per_rank = len(loader)
    total_steps = EPOCHS * n_batches_per_rank

    print(f"[rank {RANK}] model params: {n_params:,}")
    print(f"[rank {RANK}] {EPOCHS} epochs × {n_batches_per_rank} batches/epoch = {total_steps} steps")
    print(f"[rank {RANK}] writing trace to {TRACE_PATH}")

    # Truncate the trace file
    open(TRACE_PATH, "w").close()

    global_step = 0
    train_start = time.monotonic()

    with open(TRACE_PATH, "a") as trace:
        # Write a header record with run config
        header = {
            "type": "header",
            "rank": RANK,
            "world_size": WORLD_SIZE,
            "epochs": EPOCHS,
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_params": n_params,
            "wall_clock_start": time.time(),
            "monotonic_start": train_start,
            "model": "TinyClassifier(16→64→4)",
            "optimizer": "Adam(lr=0.005)",
            "peer_id": peer_id,
        }
        trace.write(json.dumps(header) + "\n")

        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)
            for batch_indices, batch_x, batch_y in loader:
                conn = transport.get_connection(peer_id)
                bytes_sent_before = conn.bytes_sent if conn else 0
                bytes_recv_before = conn.bytes_received if conn else 0

                t_step_start = time.monotonic()

                engine.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                engine.backward(loss)

                local_grad = engine.get_flat_gradients().copy()
                local_summary = _grad_summary(local_grad)

                t_sync_start = time.monotonic()
                await dp.sync_gradients()
                sync_ms = (time.monotonic() - t_sync_start) * 1000.0

                synced_grad = engine.get_flat_gradients().copy()
                synced_summary = _grad_summary(synced_grad)

                engine.step()

                bytes_sent_after = conn.bytes_sent if conn else 0
                bytes_recv_after = conn.bytes_received if conn else 0

                params_after = engine.get_flat_parameters()
                params_sha1 = hashlib.sha1(params_after.tobytes()).hexdigest()

                acc = float((logits.argmax(1) == batch_y).float().mean())

                record = {
                    "type": "step",
                    "step": global_step,
                    "epoch": epoch,
                    "rank": RANK,
                    "monotonic": time.monotonic() - train_start,
                    "wall_clock": time.time(),
                    "batch_indices": batch_indices,
                    "batch_size_actual": len(batch_indices),
                    "loss": round(float(loss), 6),
                    "train_acc": round(acc, 4),
                    "local_grad": local_summary,
                    "synced_grad": synced_summary,
                    "params_after_sha1": params_sha1,
                    "bytes_sent_delta": bytes_sent_after - bytes_sent_before,
                    "bytes_received_delta": bytes_recv_after - bytes_recv_before,
                    "bytes_sent_total": bytes_sent_after,
                    "bytes_received_total": bytes_recv_after,
                    "sync_ms": round(sync_ms, 3),
                    "step_ms": round((time.monotonic() - t_step_start) * 1000.0, 3),
                }
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                global_step += 1

            print(f"[rank {RANK}] epoch {epoch + 1}/{EPOCHS} done — {global_step} steps logged")

        # Footer record
        footer = {
            "type": "footer",
            "rank": RANK,
            "total_steps": global_step,
            "total_wall_clock_sec": time.monotonic() - train_start,
            "final_params_sha1": hashlib.sha1(
                engine.get_flat_parameters().tobytes(),
            ).hexdigest(),
            "total_bytes_sent": (transport.get_connection(peer_id).bytes_sent
                                  if transport.get_connection(peer_id) else 0),
            "total_bytes_received": (transport.get_connection(peer_id).bytes_received
                                       if transport.get_connection(peer_id) else 0),
        }
        trace.write(json.dumps(footer) + "\n")

    print()
    print(f"[rank {RANK}] === DONE ===")
    print(f"[rank {RANK}] trace: {TRACE_PATH}")
    print(f"[rank {RANK}] steps logged: {global_step}")
    print(f"[rank {RANK}] wall-clock: {time.monotonic() - train_start:.1f}s")
    print(f"[rank {RANK}] copy this file to a single Mac then run:")
    print(f"    python3 tools/compare_traces.py macfleet_trace_rank0.jsonl macfleet_trace_rank1.jsonl")

    await transport.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
