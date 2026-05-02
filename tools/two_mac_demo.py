"""Two-Mac distributed gradient-sync demo.

Run on Mac #1 (the listener) with RANK=0:
    RANK=0 python3 tools/two_mac_demo.py

Run on Mac #2 (the connector) with RANK=1 and PEER_IP=<Mac1-ip>:
    RANK=1 PEER_IP=10.172.215.146 python3 tools/two_mac_demo.py

What it does:
    1. Establishes a 2-rank PeerTransport mesh (Mac #1 listens on
       :60001, Mac #2 connects to it).
    2. Both ranks initialize the same model from seed=42, then call
       DataParallel.setup() which broadcasts rank 0's params so both
       ranks start identical.
    3. Each rank seeds its random inputs DIFFERENTLY (1000+rank), so
       the local gradients on each rank are different.
    4. Each step: forward + backward + sync_gradients + optimizer step.
       The allreduce averages gradients across ranks, so both ranks
       apply the SAME averaged gradient to the (initially identical)
       model — params stay identical step after step.
    5. After 20 steps, prints a SHA1 of the final flat parameters.

PASS condition: BOTH Macs print the SAME SHA1.
FAIL condition: hashes differ — gradients didn't sync.

This runs in open mode (no token) on its own port (60001), so it
doesn't conflict with `macfleet join` agents already on 50051/50052.
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
WORLD_SIZE = 2
STEPS = 20

CONFIG = TransportConfig(connect_timeout_sec=15.0, recv_timeout_sec=30.0)


async def main() -> None:
    if RANK not in (0, 1):
        print(f"RANK must be 0 or 1, got {RANK}")
        sys.exit(1)
    if RANK == 1 and not PEER_IP:
        print("RANK=1 requires PEER_IP to be set (Mac #1's IP)")
        sys.exit(1)

    transport = PeerTransport(local_id=f"node-{RANK}", config=CONFIG)

    if RANK == 0:
        await transport.start_server("0.0.0.0", PORT)
        print(f"[rank 0] listening on 0.0.0.0:{PORT} — waiting for rank 1...")
        # Block until rank 1 connects
        while transport.connection_count < 1:
            await asyncio.sleep(0.5)
        print(f"[rank 0] rank 1 connected — starting training")
        rank_to_peer = {1: "node-1"}
    else:
        print(f"[rank 1] connecting to {PEER_IP}:{PORT}...")
        await transport.connect("node-0", PEER_IP, PORT)
        print(f"[rank 1] connected — starting training")
        rank_to_peer = {0: "node-0"}

    group = CollectiveGroup(
        rank=RANK,
        world_size=WORLD_SIZE,
        transport=transport,
        rank_to_peer=rank_to_peer,
    )

    # Same seed → same initial model weights on both ranks.
    torch.manual_seed(42)
    model = nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)

    dp = DataParallel(engine, group)
    await dp.setup()  # rank 0 broadcasts its params; both ranks aligned

    # Re-seed differently per rank so each rank's inputs differ.
    torch.manual_seed(1000 + RANK)

    for step in range(STEPS):
        engine.zero_grad()
        x = torch.randn(8, 4)
        loss = model(x).sum()
        engine.backward(loss)
        sync_t = await dp.sync_gradients()
        engine.step()
        if step % 5 == 0:
            params_now = engine.get_flat_parameters()
            short_hash = hashlib.sha1(params_now.tobytes()).hexdigest()[:8]
            print(
                f"[rank {RANK}] step {step:2d}  "
                f"loss={float(loss):.4f}  "
                f"sync={sync_t * 1000:.1f}ms  "
                f"params_sha1={short_hash}"
            )

    final = engine.get_flat_parameters()
    final_hash = hashlib.sha1(final.tobytes()).hexdigest()
    print()
    print(f"[rank {RANK}] FINAL param SHA1 (full): {final_hash}")
    print(f"[rank {RANK}] {'='*60}")
    print(f"[rank {RANK}] PASS condition: BOTH Macs print this same hash.")
    print(f"[rank {RANK}] If hashes differ, gradient sync is broken.")

    await transport.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
