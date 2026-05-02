"""Real distributed training across two Macs — ~1 minute of compute.

Trains a 3-layer MLP on a synthetic 8-class Gaussian-mixture dataset
(8000 samples, batch_size=64, 12 epochs). Each Mac processes its own
shard of the data via WeightedDistributedSampler, runs forward +
backward locally, then allreduces gradients across the wire before
each optimizer step. After 12 epochs, both Macs hold byte-identical
model parameters AND should report ~90%+ test accuracy.

Per-epoch output shows:
    - per-rank loss + train accuracy (different per rank — different shards)
    - compute time vs sync time (where the wall-clock went)
    - cumulative bytes exchanged with the peer

End-of-run summary:
    - total wall-clock training time
    - test accuracy on a held-out synthetic test set
    - total bytes sent / received
    - SHA1 of final params (must match across Macs)

Run on Mac #1 (listener):
    RANK=0 python3 tools/two_mac_real_train.py

Run on Mac #2 (connector):
    RANK=1 PEER_IP=<Mac1-ip> python3 tools/two_mac_real_train.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.engines.torch_engine import TorchEngine
from macfleet.training.data_parallel import DataParallel
from macfleet.training.sampler import WeightedDistributedSampler

PORT = int(os.environ.get("PORT", "60001"))
RANK = int(os.environ.get("RANK", "0"))
PEER_IP = os.environ.get("PEER_IP", "")
WORLD_SIZE = 2

# Sized so each Mac runs ~60-90s wall clock on a phone-hotspot link.
# Bumped epochs+model size after observing a 22s run on M4 hardware.
EPOCHS = 30
N_SAMPLES = 12000
TEST_SAMPLES = 3000
BATCH_SIZE = 64
INPUT_DIM = 32
HIDDEN = 192
N_CLASSES = 8

CONFIG = TransportConfig(connect_timeout_sec=15.0, recv_timeout_sec=60.0)


class Classifier(nn.Module):
    """3-layer MLP with ~21K parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Cluster centers are generated ONCE from a fixed seed. Train + test
# splits draw fresh samples around the SAME centers, so the test set
# is genuinely held-out (not seen during training) but lives in the
# same distribution the model was trained on. Earlier version used
# different center seeds — train hit 100% but test was ~3% because
# the test distribution was a totally different problem.
_CENTER_SEED = 4242
_centers_cache = {}


def _get_centers() -> torch.Tensor:
    if "centers" not in _centers_cache:
        g = torch.Generator().manual_seed(_CENTER_SEED)
        _centers_cache["centers"] = torch.randn(N_CLASSES, INPUT_DIM, generator=g) * 2.5
    return _centers_cache["centers"]


def make_dataset(n_samples: int, seed: int) -> TensorDataset:
    """Synthetic Gaussian-mixture classification — visibly separable.

    Same cluster centers across train/test splits; only sample noise
    differs. Ensures a held-out test set actually evaluates generalization
    on the trained distribution.
    """
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
        print("[rank 0] rank 1 connected — starting training")
        rank_to_peer = {1: "node-1"}
        peer_id = "node-1"
    else:
        await transport.connect("node-0", PEER_IP, PORT)
        print(f"[rank 1] connected to {PEER_IP}:{PORT} — starting training")
        rank_to_peer = {0: "node-0"}
        peer_id = "node-0"

    group = CollectiveGroup(
        rank=RANK, world_size=WORLD_SIZE,
        transport=transport, rank_to_peer=rank_to_peer,
    )

    torch.manual_seed(42)
    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[rank {RANK}] model: 3-layer MLP, {n_params:,} params "
        f"({n_params * 4 / 1024:.1f} KB per allreduce)"
    )

    dp = DataParallel(engine, group)
    await dp.setup()  # rank 0 broadcasts initial weights

    train_set = make_dataset(N_SAMPLES, seed=42)
    sampler = WeightedDistributedSampler(
        train_set, num_replicas=WORLD_SIZE, rank=RANK,
        shuffle=True, seed=0,
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)

    # Same cluster centers as train — fresh samples — proper held-out set.
    test_set = make_dataset(TEST_SAMPLES, seed=999)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    print(
        f"[rank {RANK}] training: {EPOCHS} epochs × ~{len(train_loader)} "
        f"batches/epoch on this rank = ~{EPOCHS * len(train_loader)} "
        f"allreduces total"
    )
    print()

    total_start = time.monotonic()

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        ep_start = time.monotonic()
        compute_t = 0.0
        sync_t = 0.0
        loss_sum = 0.0
        correct = 0
        seen = 0
        model.train()

        for batch_x, batch_y in train_loader:
            t1 = time.monotonic()
            engine.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            engine.backward(loss)
            t2 = time.monotonic()

            await dp.sync_gradients()
            t3 = time.monotonic()
            engine.step()

            compute_t += (t2 - t1)
            sync_t += (t3 - t2)
            loss_sum += loss.item() * len(batch_y)
            correct += (logits.argmax(1) == batch_y).sum().item()
            seen += len(batch_y)

        ep_elapsed = time.monotonic() - ep_start
        avg_loss = loss_sum / seen
        train_acc = correct / seen

        # Cumulative bytes after this epoch
        conn = transport.get_connection(peer_id)
        kb_sent = (conn.bytes_sent / 1024) if conn else 0
        kb_recv = (conn.bytes_received / 1024) if conn else 0

        print(
            f"[rank {RANK}] epoch {epoch + 1:2d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  acc={train_acc:.1%}  "
            f"compute={compute_t:5.2f}s  sync={sync_t:5.2f}s  "
            f"total={ep_elapsed:5.2f}s  "
            f"net_cum=↑{kb_sent:6.1f}KB ↓{kb_recv:6.1f}KB"
        )

    total_elapsed = time.monotonic() - total_start

    # Held-out test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += len(y)
    test_acc = test_correct / test_total

    conn = transport.get_connection(peer_id)
    bytes_sent = conn.bytes_sent if conn else 0
    bytes_recv = conn.bytes_received if conn else 0
    final_hash = hashlib.sha1(engine.get_flat_parameters().tobytes()).hexdigest()

    print()
    print(f"[rank {RANK}] {'=' * 60}")
    print(f"[rank {RANK}] TRAINING COMPLETE")
    print(f"[rank {RANK}] {'=' * 60}")
    print(f"[rank {RANK}] wall-clock training time: {total_elapsed:.1f}s")
    print(f"[rank {RANK}] held-out test accuracy:    {test_acc:.1%}")
    print(f"[rank {RANK}] bytes sent to peer:    {bytes_sent / 1024:>9.1f} KB")
    print(f"[rank {RANK}] bytes received:        {bytes_recv / 1024:>9.1f} KB")
    print(f"[rank {RANK}] final params SHA1:     {final_hash}")
    print(f"[rank {RANK}] {'=' * 60}")
    print(
        f"[rank {RANK}] PASS condition:\n"
        f"[rank {RANK}]   - both Macs see same final params SHA1\n"
        f"[rank {RANK}]   - test accuracy > 80% on both Macs\n"
        f"[rank {RANK}]   - bytes sent/received roughly mirror "
        f"between Macs (sent on rank 0 ≈ received on rank 1, vice versa)"
    )

    await transport.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
