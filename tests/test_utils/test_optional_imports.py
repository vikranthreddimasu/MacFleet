"""Framework-independent entry points must stay independent in a fresh process."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("statement", [
    "import macfleet.comm.transport",
    "import macfleet.pool.agent",
    "import macfleet.compression.adaptive",
    "import macfleet.training.data_parallel",
    "from macfleet.pool.agent import profile_hardware; profile_hardware()",
])
def test_core_import_does_not_load_training_frameworks(statement):
    result = subprocess.run(
        [sys.executable, "-c", (
            f"{statement}; import sys; "
            "assert not {'torch', 'mlx', 'mlx.core'} & sys.modules.keys()"
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_optional_public_exports_still_resolve():
    pytest.importorskip("torch")
    from macfleet.compression import CompressionPipeline, FP16Quantizer, TopKCompressor
    from macfleet.compression.pipeline import CompressionPipeline as Pipeline
    from macfleet.compression.quantize import FP16Quantizer as Quantizer
    from macfleet.compression.topk import TopKCompressor as TopK
    from macfleet.engines import TorchEngine
    from macfleet.engines.torch_engine import TorchEngine as Engine

    assert (CompressionPipeline, FP16Quantizer, TopKCompressor, TorchEngine) == (Pipeline, Quantizer, TopK, Engine)
