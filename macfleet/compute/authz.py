"""Authorization policy for remote task execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


class TaskAuthorizationError(PermissionError):
    """Raised when a worker refuses to execute a task."""


@dataclass(frozen=True)
class TaskAuthorizationPolicy:
    """Worker-side authorization policy for registered tasks.

    `None` for `allowed_tasks` means "any registered remote task". Supplying a
    set turns the worker into an explicit allowlist.
    """

    role: str = "worker"
    allowed_tasks: Optional[frozenset[str]] = None
    denied_tasks: frozenset[str] = field(default_factory=frozenset)
    max_timeout_sec: float = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be a non-empty string")
        for name, task_names in (
            ("allowed_tasks", self.allowed_tasks),
            ("denied_tasks", self.denied_tasks),
        ):
            if task_names is not None and any(
                not isinstance(task_name, str) or not task_name for task_name in task_names
            ):
                raise ValueError(f"{name} must contain only non-empty task names")
        if (
            isinstance(self.max_timeout_sec, bool)
            or not isinstance(self.max_timeout_sec, (int, float))
            or not math.isfinite(float(self.max_timeout_sec))
            or self.max_timeout_sec <= 0
        ):
            raise ValueError("max_timeout_sec must be a positive finite number")

    def authorize(self, spec, entry) -> None:
        """Raise TaskAuthorizationError if this task is not permitted."""
        name = entry.name
        if name in self.denied_tasks:
            raise TaskAuthorizationError(f"Task {name!r} is denied by worker policy")
        if self.allowed_tasks is not None and name not in self.allowed_tasks:
            raise TaskAuthorizationError(f"Task {name!r} is not in the worker allowlist")
        if not entry.remote:
            raise TaskAuthorizationError(f"Task {name!r} is not allowed to run remotely")
        if self.role not in entry.roles:
            raise TaskAuthorizationError(
                f"Task {name!r} does not allow worker role {self.role!r}"
            )
        if spec.timeout_sec <= 0 or spec.timeout_sec > self.max_timeout_sec:
            raise TaskAuthorizationError(
                f"Task {name!r} requested timeout {spec.timeout_sec}s, "
                f"max allowed is {self.max_timeout_sec}s"
            )


DEFAULT_TASK_POLICY = TaskAuthorizationPolicy()
