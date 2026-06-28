"""CI-safe coverage for the two-Mac verification harness.

The harness in ``tools/two_mac_verify.py`` is normally run by a human across
two Macs. Its ``--self-check`` mode, however, exercises only single-machine
plumbing (import, interface detection, deterministic topology selection, link
serialization, single-node ``Pool.train``) and therefore runs in CI on one
machine. These tests pin that behavior so a regression in the verifier — or in
the plumbing it checks — is caught by ``make test``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Self-check's single_node_train check needs torch.
pytest.importorskip("torch")

_HARNESS_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "two_mac_verify.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("two_mac_verify", _HARNESS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution (the module uses
    # `from __future__ import annotations`) can find the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_self_check_all_pass():
    harness = _load_harness()
    report = harness.run_self_check()

    assert report.mode == "self-check"
    # Every check must pass on a healthy single machine.
    failed = [c.check_id for c in report.checks if not c.passed]
    assert failed == [], f"self-check failures: {failed}"
    assert report.failures == 0


def test_self_check_covers_expected_checks():
    harness = _load_harness()
    report = harness.run_self_check()

    ids = {c.check_id for c in report.checks}
    assert ids == {
        "import_and_version",
        "interface_detection",
        "topology_selection",
        "link_serialization",
        "single_node_train",
    }


def test_single_node_train_actually_reduces_loss():
    harness = _load_harness()
    report = harness.run_self_check()

    train = report.info.get("self_train")
    assert isinstance(train, dict)
    assert train["world_size"] == 1
    assert train["steps"] > 0
    # Linearly separable data: the model must learn.
    assert train["loss_last"] < train["loss_first"]


def test_main_self_check_exit_code_zero():
    harness = _load_harness()
    assert harness.main(["--self-check"]) == 0
