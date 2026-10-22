"""The public Pool API must form a secure fleet and train across processes."""

import asyncio
import faulthandler
import importlib.util
import multiprocessing
import queue
import secrets
import subprocess
import sys
import time
import traceback

import pytest

_PHASES = (
    "spawned", "initializing runtime", "pool enter", "train 0 begin", "train 0 end",
    "train 1 begin", "train 1 end", "pool leave", "complete", "failed",
)


def _dump_async_tasks(index):
    print(f"Training child {index}: asyncio tasks at {time.monotonic():.6f}", flush=True)
    for task in asyncio.all_tasks():
        task.print_stack(file=sys.stdout)
        coroutine = getattr(task.get_coro(), "cr_await", None)
        while coroutine is not None:
            frame = getattr(coroutine, "cr_frame", None)
            if frame is not None:
                traceback.print_stack(frame, limit=1, file=sys.stdout)
            coroutine = getattr(coroutine, "cr_await", None)
    sys.stdout.flush()


def _train_worker(index, engine, world_size, token, results, phases=None):
    import numpy as np

    import macfleet.sdk.pool as sdk_pool
    from macfleet.sdk.pool import Pool

    def phase(name):
        if phases is not None:
            phases[index] = _PHASES.index(name)
        print(f"Training child {index}: {name} at {time.monotonic():.6f}", flush=True)

    faulthandler.dump_traceback_later(40, repeat=True)
    try:
        phase("initializing runtime")
        x = np.random.default_rng(42).normal(size=(192, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int32)
        if engine == "torch":
            import torch

            torch.manual_seed(index)
            model = torch.nn.Linear(4, 2)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            import mlx.core as mx
            import mlx.nn as nn

            print(
                f"Training child {index}: MLX {mx.__version__}, "
                f"default device {mx.default_device()}, Metal available {mx.metal.is_available()}, "
                f"device info {mx.device_info()}",
                flush=True,
            )
            original_train = sdk_pool._distributed_train_mlx

            async def watched_train(**kwargs):
                watchdog = asyncio.get_running_loop().call_later(35, _dump_async_tasks, index)
                try:
                    return await original_train(**kwargs)
                finally:
                    watchdog.cancel()

            sdk_pool._distributed_train_mlx = watched_train
            mx.random.seed(index)
            model = nn.Linear(4, 2)

            def loss_fn(model, inputs, targets):
                return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")

        phase("pool enter")
        with Pool(
            name=f"training-{index}", token=token, engine=engine, port=0, data_port=0,
            enable_pool_distributed=True, quorum_size=world_size,
            quorum_timeout_sec=20, rendezvous_timeout_sec=20,
        ) as pool:
            reports = []
            for call in range(2):
                phase(f"train {call} begin")
                reports.append(
                    pool.train(model, (x, y), epochs=3, batch_size=48, lr=0.1, loss_fn=loss_fn, device="cpu")
                )
                phase(f"train {call} end")
            phase("pool leave")
        phase("complete")
        results.put(reports)
    except BaseException:
        phase("failed")
        results.put({"error": traceback.format_exc()})
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("engine", ["torch", "mlx"])
@pytest.mark.parametrize("world_size", [2, 3])
def test_pool_training_across_processes(engine, world_size):
    if importlib.util.find_spec(engine) is None:
        pytest.skip(f"{engine} is not installed")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    phases = ctx.Array("i", world_size, lock=False)
    token = secrets.token_hex(32)
    processes = [
        ctx.Process(target=_train_worker, args=(i, engine, world_size, token, results, phases))
        for i in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        try:
            runs = [results.get(timeout=60) for _ in processes]
        except queue.Empty:
            for index, process in enumerate(processes):
                print(
                    f"Training child {index}: pid={process.pid}, exitcode={process.exitcode}, "
                    f"last phase={_PHASES[phases[index]]}",
                    flush=True,
                )
                if sys.platform == "darwin" and process.is_alive():
                    try:
                        sample = subprocess.run(
                            ["sample", str(process.pid), "1"], capture_output=True,
                            text=True, timeout=5, check=False,
                        )
                        print(sample.stdout, sample.stderr, flush=True)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        print(f"Native sample unavailable: {error}", flush=True)
            raise
        assert all(isinstance(reports, list) and len(reports) == 2 for reports in runs), runs
        for reports in zip(*runs, strict=True):
            assert {report["rank"] for report in reports} == set(range(world_size))
            assert len({report["params_sha256"] for report in reports}) == 1
            for report in reports:
                assert report["world_size"] == world_size
                assert report["steps"] == 12
                assert report["loss_history"][-1] < report["loss_history"][0]
                assert not report["degraded"]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        results.close()
        results.join_thread()
