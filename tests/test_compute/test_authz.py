"""Tests for worker task authorization policy validation."""

from __future__ import annotations

import pytest

from macfleet.compute.authz import TaskAuthorizationPolicy


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": ""}, "role"),
        ({"allowed_tasks": frozenset({""})}, "allowed_tasks"),
        ({"allowed_tasks": "task.name"}, "allowed_tasks"),
        ({"allowed_tasks": 42}, "allowed_tasks"),
        ({"denied_tasks": frozenset({""})}, "denied_tasks"),
        ({"denied_tasks": "task.name"}, "denied_tasks"),
        (
            {
                "allowed_tasks": frozenset({"task.a"}),
                "denied_tasks": frozenset({"task.a"}),
            },
            "overlap",
        ),
        ({"max_timeout_sec": 0}, "max_timeout_sec"),
        ({"max_timeout_sec": float("nan")}, "max_timeout_sec"),
    ],
)
def test_policy_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TaskAuthorizationPolicy(**kwargs)


def test_policy_normalizes_task_name_collections():
    policy = TaskAuthorizationPolicy(
        allowed_tasks=["task.a", "task.b"],
        denied_tasks=("task.c",),
    )

    assert policy.allowed_tasks == frozenset({"task.a", "task.b"})
    assert policy.denied_tasks == frozenset({"task.c"})
