"""Authorization policy for remote task execution."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional


class TaskAuthorizationError(PermissionError):
    """Raised when a worker refuses to execute a task."""


def _normalize_task_names(
    field_name: str,
    value: Optional[Iterable[str]],
    *,
    allow_none: bool,
) -> Optional[frozenset[str]]:
    if value is None:
        if allow_none:
            return None
        return frozenset()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must be a collection of task names")
    normalized = frozenset(value)
    if any(not isinstance(task_name, str) or not task_name for task_name in normalized):
        raise ValueError(f"{field_name} must contain only non-empty task names")
    return normalized


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
        allowed_tasks = _normalize_task_names(
            "allowed_tasks",
            self.allowed_tasks,
            allow_none=True,
        )
        denied_tasks = _normalize_task_names(
            "denied_tasks",
            self.denied_tasks,
            allow_none=False,
        )
        assert denied_tasks is not None
        overlap = sorted((allowed_tasks or frozenset()) & denied_tasks)
        if overlap:
            raise ValueError(
                "allowed_tasks and denied_tasks overlap: "
                f"{', '.join(overlap)}"
            )
        object.__setattr__(self, "allowed_tasks", allowed_tasks)
        object.__setattr__(self, "denied_tasks", denied_tasks)
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
