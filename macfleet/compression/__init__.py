"""Gradient compression utilities for MacFleet.

Two pipelines:
- `adaptive.py` (numpy-native, v2) — required, no optional deps. Used by
  DataParallel's sync_gradients flow.
- `pipeline.py` + `topk.py` + `quantize.py` (torch-based, v1 compat) —
  scheduled for deletion in v2.3 per TODOS.md. Imports are guarded by
  try/except so the numpy path stays usable when torch isn't installed.
"""

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

# Torch-based pipeline (v1 compat) — optional, requires `pip install macfleet[torch]`
try:
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

    __all__.extend([
        "CompressionPipeline",
        "CompressedGradient",
        "Compressor",
        "FP16Stage",
        "TopKStage",
        "create_pipeline",
        "TopKCompressor",
        "FP16Quantizer",
        "Int8Quantizer",
    ])
except ImportError:
    pass
