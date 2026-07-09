"""Browser-runtime manifest builder for the EdgeTAM video tier.

Emits the EXACT shape `packages/core/src/weights/manifest.ts::parseModelManifest`
parses (schemaVersion 1, `graphs.<role>.inputs/outputs` keyed by SEMANTIC
name, `video` section with the corrected M2 constants) — this is a different,
browser-facing shape from `websam_export.manifest.build_manifest` (which
serves the `ExportSpec`/multi-tier SAM3+EdgeTAM *design* manifest, not a
runtime artifact manifest). Mirrors `tools/goldens/fetch-models.mjs`'s
`buildManifest` for the M1 image tier.

The semantic-key tables below are the ONE place ONNX tensor names are bound
to the keys `docs/m2-internal-contracts.md` §2.2 and `video-engine.ts` read —
change an ONNX input/output name in `wrappers/edgetam.py` and update the
matching entry here, nowhere else.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

SCHEMA_VERSION = 1
TIER = "edgetam"
OPSET = 18

# EdgeTAM memory-bank constants (FINDINGS.md-corrected; also
# `websam_export.spec.EDGETAM_1024`, kept independent here since the TS
# manifest shape differs field-for-field from `ExportSpec`).
MAX_COND_FRAMES = 1
NUM_RECENT = 6
MAX_MEMORY_MAPS = MAX_COND_FRAMES + NUM_RECENT  # 7
TOKENS_PER_MEMORY_MAP = 512
PTR_TOKENS = 64
MAX_OBJECT_POINTERS = 16
KV_LEN = MAX_MEMORY_MAPS * TOKENS_PER_MEMORY_MAP + PTR_TOKENS  # 3648
MEM_DIM = 64
EMBED_DIM = 256
GRID_SIZE = 64
IMAGE_SIZE = 1024


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _t(name: str, dtype: str, shape: list) -> dict[str, Any]:
    return {"name": name, "dtype": dtype, "shape": shape}


def _file_ref(path: pathlib.Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# Semantic-key IO tables (docs/m2-internal-contracts.md §2.2).
# ---------------------------------------------------------------------------

_VIDEO_ENCODER_IO = {
    "inputs": {
        "pixels": _t("pixel_values", "float16", [1, 3, IMAGE_SIZE, IMAGE_SIZE]),
    },
    "outputs": {
        "visionFeatures": _t("vision_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        "visionPos": _t("vision_pos_embed", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        "highRes0": _t("high_res_features_0", "float16", [1, 32, GRID_SIZE * 4, GRID_SIZE * 4]),
        "highRes1": _t("high_res_features_1", "float16", [1, 64, GRID_SIZE * 2, GRID_SIZE * 2]),
    },
}

_NO_MEM_CONDITION_IO = {
    "inputs": {
        "visionFeatures": _t("vision_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
    },
    "outputs": {
        "conditionedFeatures": _t("conditioned_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
    },
}

_MEMORY_ATTENTION_IO = {
    "inputs": {
        "queries": _t("current_vision_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        "queriesPos": _t("current_vision_pos_embed", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        "memorySpatial": _t("memory_spatial", "float16", [1, MAX_MEMORY_MAPS, TOKENS_PER_MEMORY_MAP, MEM_DIM]),
        "memorySpatialPos": _t("memory_spatial_pos", "float16", [1, MAX_MEMORY_MAPS, TOKENS_PER_MEMORY_MAP, MEM_DIM]),
        "tposIndices": _t("tpos_indices", "int64", [1, MAX_MEMORY_MAPS]),
        "memoryMask": _t("memory_mask", "bool", [1, KV_LEN]),
        "objectPointers": _t("object_pointers", "float16", [1, MAX_OBJECT_POINTERS, EMBED_DIM]),
        "pointerDeltas": _t("pointer_deltas", "int64", [1, MAX_OBJECT_POINTERS]),
        "pointerMask": _t("pointer_mask", "bool", [1, MAX_OBJECT_POINTERS]),
    },
    "outputs": {
        "conditionedFeatures": _t("conditioned_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
    },
}

_MASK_DECODER_VIDEO_IO = {
    "inputs": {
        "conditionedFeatures": _t("conditioned_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        "highRes0": _t("high_res_features_0", "float16", [1, 32, GRID_SIZE * 4, GRID_SIZE * 4]),
        "highRes1": _t("high_res_features_1", "float16", [1, 64, GRID_SIZE * 2, GRID_SIZE * 2]),
        "points": _t("point_coords", "float16", [1, 1, "num_points", 2]),
        "labels": _t("point_labels", "int64", [1, 1, "num_points"]),
    },
    "outputs": {
        "maskLogits": _t("low_res_masks", "float16", [1, 1, GRID_SIZE * 4, GRID_SIZE * 4]),
        "highResMaskLogits": _t("high_res_masks", "float16", [1, 1, IMAGE_SIZE, IMAGE_SIZE]),
        "objectPointer": _t("object_pointer", "float16", [1, 1, EMBED_DIM]),
        "objectScoreLogits": _t("object_score_logits", "float16", [1, 1, 1]),
        "iouScores": _t("iou_scores", "float16", [1, 1, 3]),
    },
}

_MEMORY_ENCODER_IO = {
    "inputs": {
        "visionFeatures": _t("vision_features", "float16", [1, 256, GRID_SIZE, GRID_SIZE]),
        # LOW-res (PIN-7 reconciliation): the SAME tensor as maskDecoderVideo's
        # `maskLogits` output; the graph upsamples to 1024x1024 in-graph.
        "maskLogits": _t("mask_logits", "float16", [1, 1, GRID_SIZE * 4, GRID_SIZE * 4]),
        "isPrompted": _t("is_prompted", "float16", [1]),
    },
    "outputs": {
        "memoryFeatures": _t("memory_features", "float16", [1, TOKENS_PER_MEMORY_MAP, MEM_DIM]),
        "memoryPos": _t("memory_pos_embed", "float16", [1, TOKENS_PER_MEMORY_MAP, MEM_DIM]),
    },
}

GRAPH_IO: dict[str, dict[str, dict]] = {
    "videoEncoder": _VIDEO_ENCODER_IO,
    "noMemCondition": _NO_MEM_CONDITION_IO,
    "memoryAttention": _MEMORY_ATTENTION_IO,
    "maskDecoderVideo": _MASK_DECODER_VIDEO_IO,
    "memoryEncoder": _MEMORY_ENCODER_IO,
}

# graph role -> exported artifact basename (tools/export/dist/edgetam/<name>.onnx)
GRAPH_FILES: dict[str, str] = {
    "videoEncoder": "vision_encoder",
    "noMemCondition": "no_mem_embed",
    "memoryAttention": "memory_attention",
    "maskDecoderVideo": "mask_decoder_video",
    "memoryEncoder": "memory_encoder",
}


def build_manifest(
    onnx_dir: pathlib.Path,
    *,
    toolchain: dict[str, str],
) -> dict[str, Any]:
    """Build the schemaVersion-1 EdgeTAM manifest. `onnx_dir` holds the fp16
    `.onnx` files produced by `export_edgetam.export_all`."""
    graphs: dict[str, Any] = {}
    for role, io in GRAPH_IO.items():
        basename = GRAPH_FILES[role]
        path = onnx_dir / f"{basename}.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"manifest role {role!r}: missing artifact {path}")
        ref = _file_ref(path)
        graphs[role] = {
            "files": {"fp16": ref},
            "inputs": io["inputs"],
            "outputs": io["outputs"],
        }

    return {
        "schemaVersion": SCHEMA_VERSION,
        "tier": TIER,
        "opset": OPSET,
        "graphs": graphs,
        "toolchain": {"exporter": "websam_export.export_edgetam (M2 wave-3 production)", **toolchain},
        "preprocess": {
            "mode": "square-stretch",
            "inputSize": IMAGE_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "maskSize": GRID_SIZE * 4,
        },
        "video": {
            "maxCondFrames": MAX_COND_FRAMES,
            "numRecent": NUM_RECENT,
            "tokensPerMemoryMap": TOKENS_PER_MEMORY_MAP,
            "ptrTokens": PTR_TOKENS,
            "maxObjectPointers": MAX_OBJECT_POINTERS,
            "kvLen": KV_LEN,
            "memDim": MEM_DIM,
            "embedDim": EMBED_DIM,
            "gridSize": GRID_SIZE,
            "multiObjectBatch": False,
            "initPath": "noMemGraph",
            "tposDelivery": "indices",
            "occlusionThreshold": 0.0,
        },
    }
