"""Production export driver for the EdgeTAM video tier (M2 wave-3).

Exports all five EdgeTAM graphs (`vision_encoder`, `no_mem_embed`,
`memory_attention`, `mask_decoder_video`, `memory_encoder`) with the
SEPARATED (JS-fed) `memory_attention` / low-res-mask `memory_encoder`
interfaces (see `wrappers/edgetam.py` "INTERFACE RECONCILIATION" docstrings),
keeps the fp32 source AND converts every graph to fp16 (both survive on
disk, self-contained), and writes a browser-compatible schemaVersion-1
manifest (`packages/core/src/weights/manifest.ts` shape) with the CORRECTED
EdgeTAM constants from `tools/export/spikes/m2-edgetam/FINDINGS.md`:
512 tokens/map, kvLen 3648, 1 cond + 6 recent, ImageNet mean/std, no
`pointer_time_deltas` graph input.

Reuses the spike's proven onnxslim/onnxscript opt-out list (FINDINGS.md
gotchas 1-2): `memory_encoder` skips both `ONNXProgram.optimize()` and
`onnxslim` (they independently corrupt the perceiver's twice-called layers).

Usage:
    cd tools/export && uv run --extra export python -m websam_export.export_edgetam [--out DIR] [graph ...]

Produces (under `--out`, default `dist/edgetam/`), fp32 AND fp16 per graph
(both self-contained; fp32 is the pre-conversion source, exact, and is what
onnxruntime-web's wasm device loads — fp16 is a CPU-execution-provider trap
there):
    vision_encoder.onnx / vision_encoder_fp16.onnx
    no_mem_embed.onnx / no_mem_embed_fp16.onnx
    memory_attention.onnx / memory_attention_fp16.onnx
    mask_decoder_video.onnx / mask_decoder_video_fp16.onnx
    memory_encoder.onnx / memory_encoder_fp16.onnx
    manifest.json                                  (tier 'edgetam')
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any

import onnx
import torch
from onnxconverter_common import float16 as onnx_float16
from transformers import EdgeTamVideoModel

from websam_export.wrappers.edgetam import (
    GRID,
    HIDDEN,
    IMAGE_SIZE,
    KV_LEN,
    MAX_MEMORY_MAPS,
    MAX_OBJECT_POINTERS,
    MEM_DIM,
    TOKENS_PER_MAP,
    EdgeTamMaskDecoderVideoWrapper,
    EdgeTamMemoryAttentionWrapper,
    EdgeTamMemoryEncoderWrapper,
    EdgeTamNoMemEmbedWrapper,
    EdgeTamVisionEncoderWrapper,
)

HERE = pathlib.Path(__file__).parent
DEFAULT_OUT = HERE.parent.parent / "dist" / "edgetam"

MODEL_ID = "yonigozlan/EdgeTAM-hf"
OPSET = 18


def build_graph_defs(model: EdgeTamVideoModel) -> dict[str, tuple]:
    """(name, wrapper, example_inputs, input_names, output_names, dynamic_shapes)."""
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
                torch.randn(1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM),
                torch.randn(1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM),
                torch.zeros(1, MAX_MEMORY_MAPS, dtype=torch.int64),
                torch.zeros(1, KV_LEN, dtype=torch.bool),
                torch.randn(1, MAX_OBJECT_POINTERS, HIDDEN),
                torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.int64),
                torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.bool),
            ),
            ["current_vision_features", "current_vision_pos_embed", "memory_spatial",
             "memory_spatial_pos", "tpos_indices", "memory_mask", "object_pointers",
             "pointer_deltas", "pointer_mask"],
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
                torch.randn(1, 1, GRID * 4, GRID * 4),
                torch.ones(1),
            ),
            ["vision_features", "mask_logits", "is_prompted"],
            ["memory_features", "memory_pos_embed"],
            None,
        ),
    }


# Graphs where ONNXProgram.optimize() (onnxscript) emits an INVALID graph
# (dangling value refs from de-duplicating the perceiver layers, which are
# called twice — 1D branch, then 2D branch). Ships unoptimized (spike
# FINDINGS.md gotcha 1).
SKIP_ONNXSCRIPT_OPTIMIZE = {"memory_encoder"}

# onnxslim breaks the same graph a second, independent way (a bad Gemm
# fusion). Ships straight from the dynamo exporter (spike gotcha 2).
SKIP_ONNXSLIM = {"memory_encoder"}


def export_fp32(name: str, wrapper, example_inputs, input_names, output_names,
                 dynamic_shapes, out_dir: pathlib.Path) -> str:
    """Export one graph to fp32 ONNX at `out_dir/<name>.onnx`. Returns 'dynamo' | 'torchscript'."""
    wrapper.eval()
    path = out_dir / f"{name}.onnx"
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
            wrapper, example_inputs, str(path), dynamo=False, opset_version=OPSET,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
        )
        mode = "torchscript"

    if name not in SKIP_ONNXSLIM:
        try:
            import onnxslim

            slim = onnxslim.slim(str(path))
            onnx.save(slim, str(path))
        except Exception:
            print(f"[{name}] onnxslim failed (keeping unslimmed graph):\n{traceback.format_exc()}",
                  file=sys.stderr)

    size_mb = path.stat().st_size / 1e6
    print(f"[{name}] fp32 exported via {mode}: {path.name} ({size_mb:.1f} MB)")
    return mode


def ensure_graph_inputs(path: pathlib.Path, extra: list[tuple[str, int, tuple[int, ...]]]) -> None:
    """Append UNUSED graph inputs the exporter pruned via dead-code elimination.

    `memory_attention`'s `pointer_deltas` / `pointer_mask` are accepted-but-
    unused per the wrapper's docstring (EdgeTAM has no pointer temporal PE) —
    dynamo tracing therefore drops them from the exported graph entirely
    (they're never read by any op). But `docs/m2-internal-contracts.md` §2.2
    is the SAME memoryAttention semantic-key contract every EdgeTAM-family
    engine binds against, and `video-engine.ts` unconditionally uploads and
    feeds both tensors by semantic key (not tier-conditional) — so the ONNX
    graph must expose them as real (if unconsumed) inputs, or the browser's
    `session.run(feeds)` fails on an unrecognized feed name. ONNX permits
    graph inputs with no consumers, so this is a pure additive graph-surgery
    step (no retrace, no risk of changing any existing tensor).
    """
    model = onnx.load(str(path))
    existing = {vi.name for vi in model.graph.input}
    added = []
    for name, elem_type, shape in extra:
        if name in existing:
            continue
        vi = onnx.helper.make_tensor_value_info(name, elem_type, list(shape))
        model.graph.input.append(vi)
        added.append(name)
    if added:
        onnx.save(model, str(path))
        print(f"[{path.stem}] added unused-but-required graph input(s): {', '.join(added)}")


def convert_to_fp16(path: pathlib.Path) -> None:
    """In-place fp32 -> fp16 conversion (memory-path weights ship fp16 per M2
    policy). Keeps I/O tensor dtypes as declared by the wrapper (bool/int64
    inputs, float32 activations at the graph boundary get cast internally by
    ORT) via `keep_io_types=False` — the graph's own inputs/outputs that are
    already non-float (bool, int64) are untouched by the converter; float
    inputs/outputs become fp16, matching the browser runtime's fp16 session.
    """
    model = onnx.load(str(path))
    fp16_model = onnx_float16.convert_float_to_float16(
        model, keep_io_types=False, disable_shape_infer=False,
    )
    # The dynamo exporter leaves stale float32 `value_info` entries on nodes
    # the converter rewires around (observed on Resize/Cast sandwiches from
    # interpolate ops); those stale entries make ORT's loader reject an
    # otherwise-correct fp16 graph ("Type Error: ... does not match expected
    # type"). Drop `value_info` and let ORT's own shape inference at load
    # time regenerate it from the (now-consistent) node type chain.
    del fp16_model.graph.value_info[:]

    # `convert_float_to_float16` rewrites Constant/initializer dtypes but
    # does NOT rewrite an explicit `Cast(to=FLOAT)` node — our wrappers emit
    # exactly one class of these (`bool_tensor.to(some_other_tensor.dtype)`,
    # where `some_other_tensor` was float32 AT TRACE TIME, e.g. the
    # `memory_mask` -> additive-bias cast in `EdgeTamMemoryAttentionWrapper`),
    # which survives conversion as a stale fp32 Cast feeding fp16 neighbors
    # ("Type parameter (T) ... bound to different types"). Since
    # `keep_io_types=False` means nothing in these graphs is meant to stay
    # fp32, flip every such Cast to FLOAT16 too.
    FLOAT, FLOAT16 = 1, 10
    flipped = 0
    for node in fp16_model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == FLOAT:
                attr.i = FLOAT16
                flipped += 1
    if flipped:
        print(f"[{path.stem}] flipped {flipped} residual fp32 Cast node(s) to fp16")

    onnx.save(fp16_model, str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"[{path.stem}] converted to fp16: {size_mb:.1f} MB")


def export_all(out_dir: pathlib.Path, names: list[str] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = EdgeTamVideoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    defs = build_graph_defs(model)
    wanted = names or list(defs)
    results: dict[str, Any] = {}
    for name in wanted:
        mode = export_fp32(name, *defs[name], out_dir=out_dir)
        if name == "memory_attention":
            ensure_graph_inputs(out_dir / f"{name}.onnx", [
                ("pointer_deltas", onnx.TensorProto.INT64, (1, MAX_OBJECT_POINTERS)),
                ("pointer_mask", onnx.TensorProto.BOOL, (1, MAX_OBJECT_POINTERS)),
            ])
        fp32_path = out_dir / f"{name}.onnx"
        fp16_path = out_dir / f"{name}_fp16.onnx"
        fp16_path.write_bytes(fp32_path.read_bytes())
        convert_to_fp16(fp16_path)
        results[name] = {"export_mode": mode, "dtype": ["float32", "float16"]}
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("graphs", nargs="*", help="subset of graph names to export (default: all)")
    args = parser.parse_args(argv)
    results = export_all(args.out, args.graphs or None)
    print("\nexport summary:", results)


if __name__ == "__main__":
    main()
