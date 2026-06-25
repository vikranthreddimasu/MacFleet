"""Distributed checkpointing for MacFleet.

Handles saving and loading training state for resumable training,
including model weights, optimizer state, and training progress.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from rich.console import Console
from torch.optim import Optimizer

console = Console()


def _atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    """Write a torch checkpoint via temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_dump(data: Dict[str, Any], path: Path) -> None:
    """Write JSON metadata via temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def safe_torch_load(
    path: str | Path,
    map_location: str = "cpu",
    *,
    trusted: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint using PyTorch's safer weights-only mode by default."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=not trusted,
        )
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    epoch: int
    global_step: int
    timestamp: float
    rank: int
    world_size: int
    loss: float
    accuracy: float
    training_config: Dict[str, Any]


class CheckpointManager:
    """Manage saving and loading of distributed training checkpoints.

    Handles:
    - Model state_dict
    - Optimizer state_dict
    - Training progress (epoch, step)
    - Sampler state
    - Compression residuals
    - Training configuration
    """

    def __init__(
        self,
        checkpoint_dir: str,
        rank: int = 0,
        world_size: int = 1,
        max_checkpoints: int = 5,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            rank: This node's rank.
            world_size: Total number of nodes.
            max_checkpoints: Maximum checkpoints to keep (older ones deleted).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.world_size = world_size
        self.max_checkpoints = max_checkpoints

        if not str(self.checkpoint_dir).strip():
            raise ValueError("checkpoint_dir must not be empty")
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be >= 1")

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        loss: float = 0.0,
        accuracy: float = 0.0,
        training_config: Optional[Dict[str, Any]] = None,
        sampler_state: Optional[Dict[str, Any]] = None,
        compression_residuals: Optional[Dict[str, torch.Tensor]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch.
            global_step: Current global step.
            loss: Current loss.
            accuracy: Current accuracy.
            training_config: Training configuration dict.
            sampler_state: Sampler state for resuming.
            compression_residuals: Compression residuals for error feedback.
            extra: Any additional data to save.

        Returns:
            Path to saved checkpoint.
        """
        timestamp = time.time()

        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            timestamp=timestamp,
            rank=self.rank,
            world_size=self.world_size,
            loss=loss,
            accuracy=accuracy,
            training_config=training_config or {},
        )

        # Build checkpoint dict
        checkpoint = {
            "metadata": asdict(metadata),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }

        if sampler_state is not None:
            checkpoint["sampler_state"] = sampler_state

        if compression_residuals is not None:
            # Convert tensors to CPU for saving
            checkpoint["compression_residuals"] = {
                k: v.cpu() for k, v in compression_residuals.items()
            }

        if extra is not None:
            checkpoint["extra"] = extra

        # Generate filename
        filename = f"checkpoint_epoch{epoch:03d}_step{global_step:06d}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint atomically so interrupted writes do not corrupt the
        # last known-good file.
        _atomic_torch_save(checkpoint, filepath)

        # Save metadata as JSON for quick inspection
        metadata_path = self.checkpoint_dir / f"{filename}.json"
        _atomic_json_dump(asdict(metadata), metadata_path)

        console.print(f"[green]Checkpoint saved: {filepath}[/green]")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(filepath)

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        trusted: bool = False,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            checkpoint_path: Path to checkpoint. If None, loads latest.
            device: Device to load tensors to.
            trusted: Allow unsafe pickle loading for checkpoints from trusted sources.

        Returns:
            Loaded checkpoint data.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")

        console.print(f"[blue]Loading checkpoint: {checkpoint_path}[/blue]")

        checkpoint = safe_torch_load(checkpoint_path, map_location=device, trusted=trusted)

        # Load model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move compression residuals to device
        if "compression_residuals" in checkpoint:
            checkpoint["compression_residuals"] = {
                k: v.to(device) for k, v in checkpoint["compression_residuals"].items()
            }

        console.print(
            f"[green]Loaded checkpoint from epoch {checkpoint['epoch']}, "
            f"step {checkpoint['global_step']}[/green]"
        )

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]  # Last one is the latest

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint files sorted by step.

        Returns:
            List of checkpoint paths.
        """
        pattern = "checkpoint_epoch*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        return [str(p) for p in checkpoints]

    def get_checkpoint_metadata(self, checkpoint_path: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a checkpoint without loading the full file.

        Args:
            checkpoint_path: Path to checkpoint.

        Returns:
            CheckpointMetadata or None if not found.
        """
        json_path = f"{checkpoint_path}.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
                return CheckpointMetadata(**data)

        # Fall back to loading from checkpoint
        try:
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
            if "metadata" in checkpoint:
                return CheckpointMetadata(**checkpoint["metadata"])
        except Exception:
            pass

        return None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay under max_checkpoints."""
        checkpoints = self.list_checkpoints()

        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            try:
                os.remove(oldest)
                json_path = f"{oldest}.json"
                if os.path.exists(json_path):
                    os.remove(json_path)
                console.print(f"[yellow]Removed old checkpoint: {oldest}[/yellow]")
            except OSError:
                pass

    def exists(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.list_checkpoints()) > 0


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    **kwargs,
) -> None:
    """Convenience function to save a single checkpoint.

    Args:
        path: Path to save checkpoint.
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch.
        step: Current step.
        **kwargs: Additional data to save.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        **kwargs,
    }
    _atomic_torch_save(checkpoint, Path(path))


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str = "cpu",
    trusted: bool = False,
) -> Dict[str, Any]:
    """Convenience function to load a single checkpoint.

    Args:
        path: Path to checkpoint.
        model: Model to load into.
        optimizer: Optional optimizer to load into.
        device: Device to load to.
        trusted: Allow unsafe pickle loading for checkpoints from trusted sources.

    Returns:
        Loaded checkpoint data.
    """
    checkpoint = safe_torch_load(path, map_location=device, trusted=trusted)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AutoCheckpointer:
    """Automatic checkpointing based on interval or improvement.

    Saves checkpoints either:
    - Every N epochs/steps
    - When loss improves (best model tracking)
    """

    def __init__(
        self,
        manager: CheckpointManager,
        model: nn.Module,
        optimizer: Optimizer,
        save_every_epochs: int = 1,
        save_every_steps: Optional[int] = None,
        save_on_improvement: bool = True,
    ):
        """Initialize auto-checkpointer.

        Args:
            manager: Checkpoint manager.
            model: Model to save.
            optimizer: Optimizer to save.
            save_every_epochs: Save every N epochs.
            save_every_steps: Save every N steps (optional).
            save_on_improvement: Save when loss improves.
        """
        self.manager = manager
        self.model = model
        self.optimizer = optimizer
        self.save_every_epochs = save_every_epochs
        self.save_every_steps = save_every_steps
        self.save_on_improvement = save_on_improvement

        self.best_loss = float("inf")
        self._last_epoch_saved = -1
        self._last_step_saved = -1

    def step(
        self,
        epoch: int,
        global_step: int,
        loss: float,
        accuracy: float = 0.0,
        **kwargs,
    ) -> Optional[str]:
        """Check if checkpoint should be saved and save if needed.

        Args:
            epoch: Current epoch.
            global_step: Current global step.
            loss: Current loss.
            accuracy: Current accuracy.
            **kwargs: Additional data for checkpoint.

        Returns:
            Path to saved checkpoint if saved, None otherwise.
        """
        should_save = False
        is_best = False

        # Check epoch interval
        if (epoch > self._last_epoch_saved and
            (epoch + 1) % self.save_every_epochs == 0):
            should_save = True
            self._last_epoch_saved = epoch

        # Check step interval
        if (self.save_every_steps is not None and
            global_step > self._last_step_saved and
            global_step % self.save_every_steps == 0):
            should_save = True
            self._last_step_saved = global_step

        # Check improvement
        if self.save_on_improvement and loss < self.best_loss:
            self.best_loss = loss
            is_best = True
            should_save = True

        if should_save:
            path = self.manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=loss,
                accuracy=accuracy,
                extra={"is_best": is_best, **kwargs},
            )
            return path

        return None
