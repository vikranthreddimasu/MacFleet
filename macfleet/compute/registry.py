"""Task registry and @macfleet.task decorator.

v2.2 PR 7 (Issue 20 + A2 + A8): eliminate cloudpickle-over-wire for task
dispatch. Instead of shipping pickled callables to workers and letting them
deserialize arbitrary code, tasks must be declared upfront via `@macfleet.task`
on both sides. The coordinator sends a *name* ("my_module.square"); the worker
looks that name up in its own registry. If the name is unknown, the task is
rejected before any user code runs.

This is the same design principle as Celery's `@app.task`, Ray's `@ray.remote`,
and Temporal's workflow/activity registration. It closes the RCE vector from
untrusted pickle and makes the wire format auditable.

Usage:

    from macfleet import task

    @task
    def square(n: int) -> int:
        return n * n

    # With a schema for structured args / validated return values:
    from pydantic import BaseModel

    class TrainArgs(BaseModel):
        epochs: int
        lr: float

    @task(schema=TrainArgs)
    def train(args: TrainArgs) -> dict:
        return {"loss": 0.1, "epochs_done": args.epochs}

The decorator:
    1. Registers the function globally under its name
    2. Leaves the callable usable directly from Python (for local test loops)
    3. Attaches `.task_name` and `.schema` for introspection
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskEntry:
    """A registered task: name → callable + optional Pydantic schema."""
    name: str
    fn: Callable[..., Any]
    schema: Optional[Type[BaseModel]] = None


class TaskRegistry:
    """Thread-safe singleton registry of named tasks.

    Workers and coordinators maintain their own process-local registry.
    For a task to run on a peer, both sides must have `@task` registered
    the same callable under the same name.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> TaskEntry:
        """Register a callable under a name. Returns the TaskEntry.

        If `name` is None, defaults to `f"{fn.__module__}.{fn.__qualname__}"`.
        Re-registering the same name replaces the old entry (useful for
        REPL/notebook workflows where the function is redefined).
        """
        task_name = name or f"{fn.__module__}.{fn.__qualname__}"
        entry = TaskEntry(name=task_name, fn=fn, schema=schema)
        with self._lock:
            if task_name in self._tasks:
                old = self._tasks[task_name]
                if old.fn is not fn:
                    logger.info(
                        "Task %s re-registered (old fn at %s:%s)",
                        task_name,
                        getattr(old.fn, "__code__", None) and old.fn.__code__.co_filename,
                        getattr(old.fn, "__code__", None) and old.fn.__code__.co_firstlineno,
                    )
            self._tasks[task_name] = entry
        return entry

    def get(self, name: str) -> Optional[TaskEntry]:
        """Look up a registered task. Returns None if the name is unknown."""
        with self._lock:
            return self._tasks.get(name)

    def names(self) -> list[str]:
        """List all registered task names (for debugging / introspection)."""
        with self._lock:
            return sorted(self._tasks)

    def clear(self) -> None:
        """Clear all registered tasks (used by tests)."""
        with self._lock:
            self._tasks.clear()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tasks


# Process-wide default registry. `@task` registers here unless an explicit
# registry is passed (advanced multi-tenant use cases).
_default_registry = TaskRegistry()


def get_default_registry() -> TaskRegistry:
    """Return the process-wide default task registry."""
    return _default_registry


def task(
    fn: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    schema: Optional[Type[BaseModel]] = None,
    registry: Optional[TaskRegistry] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Decorator that registers a callable as a MacFleet task.

    Can be used bare `@task` or parameterized `@task(name=..., schema=...)`.

    The wrapped function remains directly callable from Python (so users can
    test it locally without spinning up a pool). After registration the
    callable also exposes:
        .task_name  — the registered name
        .schema     — the Pydantic schema class (or None)
    """
    target_registry = registry or _default_registry

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        entry = target_registry.register(func, name=name, schema=schema)
        # Attach introspection handles directly on the function object
        func.task_name = entry.name  # type: ignore[attr-defined]
        func.schema = entry.schema  # type: ignore[attr-defined]
        return func

    if fn is not None:
        # Bare @task usage
        return _wrap(fn)
    # Parameterized @task(name=..., schema=...) usage
    return _wrap
