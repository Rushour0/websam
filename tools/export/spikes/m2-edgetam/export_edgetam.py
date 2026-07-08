"""Export the five EdgeTAM video graphs to ONNX (fp32, opset 18).

Primary path: torch.onnx.export(dynamo=True) + onnxslim.
Fallback per-graph: TorchScript exporter (dynamo=False).
Artifacts land in ./onnx/ (gitignored). Run:

    uv run --extra export python export_edgetam.py [graph ...]
"""

from __future__ import annotations

import pathlib
import sys
import traceback

import torch
from transformers import EdgeTamVideoModel

from websam_export.wrappers.edgetam import (
    GRID,
    HIDDEN,
    IMAGE_SIZE,
    KV_LEN,
    MEM_DIM,
    EdgeTamMaskDecoderVideoWrapper,
    EdgeTamMemoryAttentionWrapper,
    EdgeTamMemoryEncoderWrapper,
    EdgeTamNoMemEmbedWrapper,
    EdgeTamVisionEncoderWrapper,
)

HERE = pathlib.Path(__file__).parent
ONNX_DIR = HERE / "onnx"
ONNX_DIR.mkdir(exist_ok=True)

MODEL_ID = "yonigozlan/EdgeTAM-hf"
OPSET = 18


def build_graph_defs(model):
    """(name, wrapper, example_inputs, input_names, output_names, dynamic_shapes)"""
    return {
        "vision_encoder": (
            EdgeTamVisionEncoderWrapper(model),
            (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),),
            ["pixel_values"],
            ["vision_features", "vision_pos_embed", "high_res_features_0", "high_res_features_1"],
            None,
        ),
        "no_mem_embed": (
            EdgeTamNoMemEmbedWrapper(model),
            (torch.randn(1, HIDDEN, GRID, GRID),),
            ["vision_features"],
            ["conditioned_features"],
            None,
        ),
        "memory_attention": (
            EdgeTamMemoryAttentionWrapper(model),
            (
                torch.randn(1, HIDDEN, GRID, GRID),
                torch.randn(1, HIDDEN, GRID, GRID),
                torch.randn(1, KV_LEN, MEM_DIM),
                torch.randn(1, KV_LEN, MEM_DIM),
                torch.zeros(1, 1, 1, KV_LEN),
            ),
            ["current_vision_features", "current_vision_pos_embed", "memory",
             "memory_pos_embed", "attn_bias"],
            ["conditioned_features"],
            None,
        ),
        "mask_decoder_video": (
            EdgeTamMaskDecoderVideoWrapper(model),
            (
                torch.randn(1, HIDDEN, GRID, GRID),
                torch.randn(1, 32, GRID * 4, GRID * 4),
                torch.randn(1, 64, GRID * 2, GRID * 2),
                torch.rand(1, 1, 2, 2) * IMAGE_SIZE,
                torch.tensor([[[1, 0]]], dtype=torch.int64),
            ),
            ["conditioned_features", "high_res_features_0", "high_res_features_1",
             "point_coords", "point_labels"],
            ["low_res_masks", "high_res_masks", "object_pointer",
             "object_score_logits", "iou_scores"],
            {
                "conditioned_features": None,
                "high_res_features_0": None,
                "high_res_features_1": None,
                "point_coords": {2: torch.export.Dim("num_points", min=1, max=16)},
                "point_labels": {2: torch.export.Dim("num_points", min=1, max=16)},
            },
        ),
        "memory_encoder": (
            EdgeTamMemoryEncoderWrapper(model),
            (
                torch.randn(1, HIDDEN, GRID, GRID),
                torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE),
                torch.ones(1),
            ),
            ["vision_features", "high_res_masks", "is_prompted"],
            ["memory_features", "memory_pos_embed"],
            None,
        ),
    }


# Graphs where ONNXProgram.optimize() (onnxscript 0.7.1) produces an INVALID
# graph (dangling value refs like "val_215_1", from de-duplicating the
# perceiver layers that are called twice — once per 1D/2D branch). The
# unoptimized dynamo graph is valid and parity-clean; onnxslim still runs.
SKIP_ONNXSCRIPT_OPTIMIZE = {"memory_encoder"}

# onnxslim 0.1.94 breaks the same graph a second way: its MatMul+Add->Gemm
# fusion emits node_linear_7 with a bias shape Gemm cannot broadcast
# (runtime error "Gemm: Invalid bias shape for broadcast"). Ship this graph
# straight from the dynamo exporter.
SKIP_ONNXSLIM = {"memory_encoder"}


def export_one(name, wrapper, example_inputs, input_names, output_names, dynamic_shapes) -> str:
    """Returns 'dynamo' | 'torchscript' on success; raises on double failure."""
    wrapper.eval()
    path = ONNX_DIR / f"{name}.onnx"
    try:
        onnx_prog = torch.onnx.export(
            wrapper,
            example_inputs,
            dynamo=True,
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )
        if name not in SKIP_ONNXSCRIPT_OPTIMIZE:
            onnx_prog.optimize()
        onnx_prog.save(str(path))
        mode = "dynamo"
    except Exception:
        print(f"[{name}] dynamo export FAILED:\n{traceback.format_exc()}", file=sys.stderr)
        dynamic_axes = None
        if dynamic_shapes:
            dynamic_axes = {
                k: {ax: d.__name__ if hasattr(d, "__name__") else "dyn" for ax, d in v.items()}
                for k, v in dynamic_shapes.items() if v
            }
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(path),
            dynamo=False,
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        mode = "torchscript"

    if name not in SKIP_ONNXSLIM:
        try:
            import onnxslim

            slim = onnxslim.slim(str(path))
            import onnx

            onnx.save(slim, str(path))
        except Exception:
            print(f"[{name}] onnxslim failed (keeping unslimmed graph):\n{traceback.format_exc()}",
                  file=sys.stderr)

    size_mb = path.stat().st_size / 1e6
    print(f"[{name}] exported via {mode}: {path.name} ({size_mb:.1f} MB)")
    return mode


def main() -> None:
    torch.manual_seed(0)
    model = EdgeTamVideoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    defs = build_graph_defs(model)
    wanted = sys.argv[1:] or list(defs)
    results = {}
    for name in wanted:
        results[name] = export_one(name, *defs[name])
    print("\nexport summary:", results)


if __name__ == "__main__":
    main()
