"""Verify authenticated discovery and fleet isolation across real processes."""

import asyncio
import multiprocessing
import queue
import secrets
import time

import pytest


def _agent_worker(index, token, snapshots, stop):
    from macfleet.pool.agent import PoolAgent

    async def run():
        agent = PoolAgent(name=f"test-node-{index}", port=0, data_port=0, token=token)
        await agent.start()
        previous = None
        try:
            while not stop.is_set():
                members = sorted(node.node_id for node in agent.registry.alive_nodes)
                if members != previous:
                    snapshots.put((index, agent.node_id, members))
                    previous = members
                await asyncio.sleep(0.05)
        finally:
            await agent.stop()

    asyncio.run(run())


@pytest.mark.parametrize("groups", [(0, 0, 0), (0, 0, 1)], ids=["discovery", "fleet-isolation"])
def test_agents_discover_only_authenticated_fleet_members(groups):
    ctx = multiprocessing.get_context("spawn")
    tokens = [secrets.token_hex(32) for _ in range(2)]
    snapshots = ctx.Queue()
    stop = ctx.Event()
    processes = [
        ctx.Process(target=_agent_worker, args=(i, tokens[group], snapshots, stop), daemon=True)
        for i, group in enumerate(groups)
    ]
    identities = {}
    observed = {}
    expected = {}
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 20
        matched_at = None
        while time.monotonic() < deadline:
            assert all(p.is_alive() for p in processes), [p.exitcode for p in processes]
            try:
                index, node_id, members = snapshots.get(timeout=0.1)
                identities[index] = node_id
                observed[index] = set(members)
            except queue.Empty:
                pass
            if len(identities) == len(groups):
                expected = {
                    i: {identities[j] for j in identities if groups[j] == group}
                    for i, group in enumerate(groups)
                }
                if observed == expected:
                    matched_at = matched_at or time.monotonic()
                    if time.monotonic() - matched_at >= 1:
                        return
                else:
                    matched_at = None
        pytest.fail(f"Authenticated membership never stabilized: {observed=}, {expected=}")
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        snapshots.close()
        snapshots.join_thread()
