"""Gradient compression utilities for MacFleet.

Two pipelines:
- `adaptive.py` (numpy-native, v2) — required, no optional deps. Used by
  DataParallel's sync_gradients flow.
- `pipeline.py` + `topk.py` + `quantize.py` (torch-based, v1 compat) —
  loaded only when explicitly requested, keeping the numpy path independent.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

# Numpy-native adaptive compression (v2) — no torch required
from macfleet.compression.adaptive import (
    AdaptiveCompressionConfig,
    AdaptiveCompressor,
    CompressedArray,
    CompressionLevel,
    NumpyFP16Compressor,
    NumpyTopKCompressor,
)

__all__ = [
    # Numpy-native adaptive compression (v2, always available)
    "AdaptiveCompressor",
    "AdaptiveCompressionConfig",
    "CompressedArray",
    "CompressionLevel",
    "NumpyTopKCompressor",
    "NumpyFP16Compressor",
]

if TYPE_CHECKING:
    from macfleet.compression.pipeline import (  # noqa: F401
        CompressedGradient,
        CompressionPipeline,
        Compressor,
        FP16Stage,
        TopKStage,
        create_pipeline,
    )
    from macfleet.compression.quantize import FP16Quantizer, Int8Quantizer  # noqa: F401
    from macfleet.compression.topk import TopKCompressor  # noqa: F401

_TORCH_EXPORTS = {
    "CompressionPipeline": "pipeline",
    "CompressedGradient": "pipeline",
    "Compressor": "pipeline",
    "FP16Stage": "pipeline",
    "TopKStage": "pipeline",
    "create_pipeline": "pipeline",
    "TopKCompressor": "topk",
    "FP16Quantizer": "quantize",
    "Int8Quantizer": "quantize",
}
if find_spec("torch") is not None:
    __all__.extend(_TORCH_EXPORTS)


def __getattr__(name: str):
    if name in _TORCH_EXPORTS and name in __all__:
        return getattr(import_module(f"{__name__}.{_TORCH_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
