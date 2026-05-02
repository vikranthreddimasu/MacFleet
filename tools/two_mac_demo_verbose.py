"""Two-Mac distributed gradient-sync demo — VERBOSE.

Same setup as two_mac_demo.py but prints concrete per-step receipts:

  - LOCAL grad hash (different on each rank — proof both Macs computed
    different gradients on different inputs)
  - SYNCED grad hash (identical on both ranks — proof allreduce averaged
    them across the wire)
  - per-rank loss
  - bytes flowing between the Macs (final summary from PeerConnection
    counters)

Optional env var SKIP_SYNC=1 turns off the sync_gradients call entirely
to show the negative case — params diverge, post-sync hashes differ.

Run on Mac #1 (listener):
    RANK=0 python3 tools/two_mac_demo_verbose.py

Run on Mac #2 (connector):
    RANK=1 PEER_IP=<Mac1-ip> python3 tools/two_mac_demo_verbose.py

Compare side-by-side:
    PASS — pre-sync hashes differ AND post-sync hashes match every step
    FAIL — pre-sync and post-sync hashes match each other (no sync ran)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.engines.torch_engine import TorchEngine
from macfleet.training.data_parallel import DataParallel

PORT = int(os.environ.get("PORT", "60001"))
RANK = int(os.environ.get("RANK", "0"))
PEER_IP = os.environ.get("PEER_IP", "")
SKIP_SYNC = os.environ.get("SKIP_SYNC", "") == "1"
WORLD_SIZE = 2
STEPS = 12

CONFIG = TransportConfig(connect_timeout_sec=15.0, recv_timeout_sec=30.0)


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


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
        print("[rank 0] rank 1 connected")
        rank_to_peer = {1: "node-1"}
        peer_id = "node-1"
    else:
        await transport.connect("node-0", PEER_IP, PORT)
        print(f"[rank 1] connected to {PEER_IP}:{PORT}")
        rank_to_peer = {0: "node-0"}
        peer_id = "node-0"

    if SKIP_SYNC:
        print(f"[rank {RANK}] SKIP_SYNC=1 — gradient sync DISABLED (negative-case run)")

    group = CollectiveGroup(
        rank=RANK, world_size=WORLD_SIZE,
        transport=transport, rank_to_peer=rank_to_peer,
    )

    torch.manual_seed(42)
    model = nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)

    dp = DataParallel(engine, group)
    if not SKIP_SYNC:
        await dp.setup()

    torch.manual_seed(1000 + RANK)

    print(f"[rank {RANK}] {'step':>4}  {'loss':>10}  {'local_grad':>14}  "
          f"{'synced_grad':>14}  {'params_after':>14}")

    for step in range(STEPS):
        engine.zero_grad()
        x = torch.randn(8, 4)
        loss = model(x).sum()
        engine.backward(loss)

        local_grad = engine.get_flat_gradients().copy()
        local_h = _hash(local_grad)

        if SKIP_SYNC:
            synced_h = "(skipped)"
        else:
            await dp.sync_gradients()
            synced_grad = engine.get_flat_gradients()
            synced_h = _hash(synced_grad)

        engine.step()
        params_h = _hash(engine.get_flat_parameters())

        print(f"[rank {RANK}] {step:>4}  {float(loss):>10.4f}  "
              f"{local_h:>14}  {synced_h:>14}  {params_h:>14}")

    # Final byte counters — concrete network proof
    conn = transport.get_connection(peer_id)
    if conn is not None:
        bytes_sent = conn.bytes_sent
        bytes_recv = conn.bytes_received
        print()
        print(f"[rank {RANK}] === network bytes ===")
        print(f"[rank {RANK}] sent to peer:     {bytes_sent:>10} bytes")
        print(f"[rank {RANK}] received from peer:{bytes_recv:>10} bytes")
        if SKIP_SYNC:
            print(f"[rank {RANK}] (only handshake bytes — no grad sync ran)")
        else:
            print(f"[rank {RANK}] (handshake + {STEPS} rounds of allreduce)")

    print()
    print(f"[rank {RANK}] === reading the output ===")
    print(f"[rank {RANK}] PASS: 'local_grad' column should DIFFER between Macs each step")
    print(f"[rank {RANK}]       'synced_grad' column should MATCH between Macs each step")
    print(f"[rank {RANK}]       'params_after' column should MATCH between Macs each step")

    await transport.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
