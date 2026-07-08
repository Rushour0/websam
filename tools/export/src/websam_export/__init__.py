"""websam-export: SAM3 / EdgeTAM -> ONNX export pipeline for WebSAM.

Public surface:

* :mod:`websam_export.spec` — the executable export contract
  (:class:`ExportSpec` / :class:`GraphSpec` / :class:`TensorSpec`, the
  memory-bank constants, the :data:`TIERS` registry, and the
  :func:`tpos_index` temporal-position rule).
* :mod:`websam_export.manifest` — :func:`build_manifest`,
  :func:`validate_manifest`, and the streamed :func:`sha256_file` helper.
"""

from .manifest import SCHEMA_VERSION, build_manifest, sha256_file, validate_manifest
from .spec import (
    COND_TPOS_INDEX,
    EDGETAM_1024,
    MAX_COND_FRAMES,
    MAX_MEMORY_MAPS,
    MAX_OBJECT_POINTERS,
    NUM_RECENT,
    PTR_TOKENS,
    SAM3_560,
    SAM3_1008,
    TIERS,
    ExportSpec,
    GraphSpec,
    TensorSpec,
    tpos_index,
)

__all__ = [
    "COND_TPOS_INDEX",
    "EDGETAM_1024",
    "ExportSpec",
    "GraphSpec",
    "MAX_COND_FRAMES",
    "MAX_MEMORY_MAPS",
    "MAX_OBJECT_POINTERS",
    "NUM_RECENT",
    "PTR_TOKENS",
    "SAM3_1008",
    "SAM3_560",
    "SCHEMA_VERSION",
    "TIERS",
    "TensorSpec",
    "build_manifest",
    "sha256_file",
    "tpos_index",
    "validate_manifest",
]
