#!/usr/bin/env python3
"""Test distributed training on loopback (127.0.0.1).

Runs master and worker trainers in the same process using asyncio tasks
to verify TCP peer connections, test tensor exchange, and AllReduce work.
"""

import asyncio
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macfleet.core.config import ClusterConfig, CompressionType, NodeRole, TrainingConfig
from macfleet.training.trainer import Trainer


def make_tiny_model():
    """Create a tiny model for testing."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def make_tiny_dataset(n=64):
    """Create a tiny dataset for testing."""
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return TensorDataset(X, y)


async def run_master(port: int, tensor_port: int):
    """Run master trainer."""
    model = make_tiny_model()
    dataset = make_tiny_dataset()

    training_config = TrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        compression=CompressionType.NONE,
        device="cpu",
    )

    cluster_config = ClusterConfig(
        role=NodeRole.MASTER,
        master_addr="127.0.0.1",
        master_port=port,
        tensor_port=tensor_port,
        host="127.0.0.1",
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        training_config=training_config,
        cluster_config=cluster_config,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
        distributed=True,
    )

    await trainer.setup()
    print("\n[MASTER] Setup complete!")

    # Run a couple training steps to test AllReduce
    print("[MASTER] Running training steps...")
    trainer._ddp_model.train()
    for batch_idx, (inputs, targets) in enumerate(trainer._dataloader):
        if batch_idx >= 2:
            break
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")
        trainer._ddp_model.zero_grad()
        outputs = trainer._ddp_model(inputs)
        loss = trainer.criterion(outputs, targets)
        loss.backward()
        await trainer._ddp_model.sync_gradients()
        trainer._optimizer.step()
        print(f"[MASTER] Step {batch_idx}: loss={loss.item():.4f}")

    print("[MASTER] Training steps done, tearing down...")
    await trainer.teardown()
    print("[MASTER] Done.")


async def run_worker(master_port: int, tensor_port: int, master_tensor_port: int):
    """Run worker trainer."""
    # Small delay to let master start first
    await asyncio.sleep(2.0)

    model = make_tiny_model()
    dataset = make_tiny_dataset()

    training_config = TrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        compression=CompressionType.NONE,
        device="cpu",
    )

    cluster_config = ClusterConfig(
        role=NodeRole.WORKER,
        master_addr="127.0.0.1",
        master_port=master_port,
        tensor_port=tensor_port,
        host="127.0.0.1",
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        training_config=training_config,
        cluster_config=cluster_config,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
        distributed=True,
    )

    await trainer.setup()
    print("\n[WORKER] Setup complete!")

    # Run matching training steps
    print("[WORKER] Running training steps...")
    trainer._ddp_model.train()
    for batch_idx, (inputs, targets) in enumerate(trainer._dataloader):
        if batch_idx >= 2:
            break
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")
        trainer._ddp_model.zero_grad()
        outputs = trainer._ddp_model(inputs)
        loss = trainer.criterion(outputs, targets)
        loss.backward()
        await trainer._ddp_model.sync_gradients()
        trainer._optimizer.step()
        print(f"[WORKER] Step {batch_idx}: loss={loss.item():.4f}")

    print("[WORKER] Training steps done, tearing down...")
    await trainer.teardown()
    print("[WORKER] Done.")


async def main():
    print("=" * 60)
    print("Distributed Training Loopback Test")
    print("=" * 60)

    grpc_port = 50061
    master_tensor_port = 50062
    worker_tensor_port = 50063

    # Run master and worker concurrently
    try:
        await asyncio.gather(
            run_master(grpc_port, master_tensor_port),
            run_worker(grpc_port, worker_tensor_port, master_tensor_port),
        )
        print("\n" + "=" * 60)
        print("TEST PASSED: Distributed setup + peer verification succeeded!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
