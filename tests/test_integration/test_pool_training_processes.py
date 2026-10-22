"""The public Pool API must form a secure fleet and train across processes."""

import importlib.util
import multiprocessing
import secrets
import traceback

import pytest


def _train_worker(index, engine, token, results):
    import numpy as np

    from macfleet.sdk.pool import Pool

    try:
        x = np.random.default_rng(42).normal(size=(128, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int32)
        if engine == "torch":
            import torch

            torch.manual_seed(index)
            model = torch.nn.Linear(4, 2)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            import mlx.core as mx
            import mlx.nn as nn

            mx.random.seed(index)
            model = nn.Linear(4, 2)

            def loss_fn(model, inputs, targets):
                return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")

        with Pool(
            name=f"training-{index}", token=token, engine=engine, port=0, data_port=0,
            enable_pool_distributed=True, quorum_size=2,
            quorum_timeout_sec=20, rendezvous_timeout_sec=20,
        ) as pool:
            result = pool.train(model, (x, y), epochs=3, batch_size=32, lr=0.1, loss_fn=loss_fn, device="cpu")
        results.put(result)
    except BaseException:
        results.put({"error": traceback.format_exc()})
        raise


@pytest.mark.parametrize("engine", ["torch", "mlx"])
def test_pool_training_across_processes(engine):
    if importlib.util.find_spec(engine) is None:
        pytest.skip(f"{engine} is not installed")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    token = secrets.token_hex(32)
    processes = [ctx.Process(target=_train_worker, args=(i, engine, token, results)) for i in range(2)]
    try:
        for process in processes:
            process.start()
        reports = [results.get(timeout=50) for _ in processes]
        assert all("error" not in report for report in reports), reports
        assert {report["rank"] for report in reports} == {0, 1}
        assert len({report["params_sha256"] for report in reports}) == 1
        for report in reports:
            assert report["world_size"] == 2
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
