"""Tests for worker task authorization policy validation."""

from __future__ import annotations

import pytest

from macfleet.compute.authz import TaskAuthorizationPolicy


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": ""}, "role"),
        ({"allowed_tasks": frozenset({""})}, "allowed_tasks"),
        ({"denied_tasks": frozenset({""})}, "denied_tasks"),
        ({"max_timeout_sec": 0}, "max_timeout_sec"),
        ({"max_timeout_sec": float("nan")}, "max_timeout_sec"),
    ],
)
def test_policy_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TaskAuthorizationPolicy(**kwargs)
