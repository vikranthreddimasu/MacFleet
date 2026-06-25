"""Authorization policy for remote task execution."""

from __future__ import annotations

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
