"""Per-graph parity: ONNX Runtime vs PyTorch eager, on REAL activations
captured from the HF EdgeTAM video run (capture_golden.py).

Three comparisons per graph, base tolerance fp32 max_abs < 1e-4:

  1. wrapper-eager vs HF-captured output  — proves the wrapper (incl. the
     padded-KV re-implementation of memory attention) equals HF's math;
  2. ORT vs wrapper-eager                 — proves the exported graph equals
     the wrapper;
  3. (memory_attention only) padding-inertness: re-run ORT with the padded
     region filled with garbage instead of zeros; output must not move.

Two tensors carry a justified, LOOSER max_abs bound (analysis in FINDINGS.md;
in both cases the eager wrapper is BIT-EXACT vs HF, so the deviation is
ORT-vs-PyTorch fp32 kernel non-associativity amplified by network gain, not a
miscompilation — verified by staged re-runs feeding exact eager intermediates):

  * vision_features (deepest RepViT+FPN path): 5e-4 abs, relative ~1e-4;
  * memory_encoder memory_features: 5e-3 abs + cosine >= 0.99999
    (sigmoid 4e-6 -> memory-fuser ~4e-5 -> perceiver layer-norm gain ~1e-3).

Run:  uv run --extra export python parity_graphs.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort
import torch
from transformers import EdgeTamVideoModel

from clip_util import NUM_FRAMES, PROMPT_FRAME
from websam_export.wrappers.edgetam import (
    KV_LEN,
    MAX_MEMORY_MAPS,
    ATTN_BIAS_NEG,
    PTR_TOKENS,
    TOKENS_PER_MAP,
    EdgeTamMaskDecoderVideoWrapper,
    EdgeTamMemoryAttentionWrapper,
    EdgeTamMemoryEncoderWrapper,
    EdgeTamNoMemEmbedWrapper,
    EdgeTamVisionEncoderWrapper,
)

HERE = pathlib.Path(__file__).parent
MODEL_ID = "yonigozlan/EdgeTAM-hf"
TOL = 1e-4

FAILURES: list[str] = []


def check(name: str, got: np.ndarray, want: np.ndarray, tol: float = TOL) -> None:
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    status = "OK " if err < tol else "FAIL"
    if err >= tol:
        FAILURES.append(f"{name}: max_abs={err:.3e} (tol {tol})")
    print(f"  [{status}] {name}: max_abs={err:.3e}")


def check_cosine(name: str, got, want, min_cos: float = 0.99999) -> None:
    a = np.asarray(got).ravel().astype(np.float64)
    b = np.asarray(want).ravel().astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    status = "OK " if cos >= min_cos else "FAIL"
    if cos < min_cos:
        FAILURES.append(f"{name}: cosine={cos:.7f} (min {min_cos})")
    print(f"  [{status}] {name}: cosine={cos:.7f}")


def sess(name: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(HERE / "onnx" / f"{name}.onnx"),
                                providers=["CPUExecutionProvider"])


def pad_memory_attention_inputs(g: dict, t: int, garbage: bool = False):
    """Build the frozen-KV padded inputs from HF's unpadded captured tensors.

    Layout contract (what the JS engine must replicate):
      [0 : n_maps*512)            real memory maps, HF order (cond first, then
                                  recent offsets 6..1 oldest-first)
      [n_maps*512 : 3584)         padded map slots (masked)
      [3584 : 3584+n_ptr_tokens)  real pointer tokens
      [.. : 3648)                 padded pointer slots (masked)
    """
    mem = g[f"f{t}/ma_memory"][:, 0, :]        # (S, 64)
    pos = g[f"f{t}/ma_memory_pos"][:, 0, :]
    n_maps = int(g[f"f{t}/ma_num_spatial_maps"])
    n_ptr = int(g[f"f{t}/ma_num_ptr_tokens"])
    assert mem.shape[0] == n_maps * TOKENS_PER_MAP + n_ptr, (mem.shape, n_maps, n_ptr)

    fill = np.float32(1e3) if garbage else np.float32(0)
    memory = np.full((1, KV_LEN, 64), fill, dtype=np.float32)
    pos_pad = np.full((1, KV_LEN, 64), fill, dtype=np.float32)
    bias = np.full((1, 1, 1, KV_LEN), ATTN_BIAS_NEG, dtype=np.float32)

    spatial = n_maps * TOKENS_PER_MAP
    memory[0, :spatial] = mem[:spatial]
    pos_pad[0, :spatial] = pos[:spatial]
    bias[0, 0, 0, :spatial] = 0.0
    ptr_base = MAX_MEMORY_MAPS * TOKENS_PER_MAP
    memory[0, ptr_base:ptr_base + n_ptr] = mem[spatial:]
    pos_pad[0, ptr_base:ptr_base + n_ptr] = pos[spatial:]
    bias[0, 0, 0, ptr_base:ptr_base + n_ptr] = 0.0

    feats = g[f"f{t}/ma_current_vision_features"][:, 0, :].T.reshape(1, 256, 64, 64)
    fpos = g[f"f{t}/ma_current_vision_pos"][:, 0, :].T.reshape(1, 256, 64, 64)
    return {
        "current_vision_features": np.ascontiguousarray(feats),
        "current_vision_pos_embed": np.ascontiguousarray(fpos),
        "memory": memory,
        "memory_pos_embed": pos_pad,
        "attn_bias": bias,
    }


def main() -> None:
    g = dict(np.load(HERE / "activations" / "golden.npz"))
    model = EdgeTamVideoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    # ---- vision_encoder -----------------------------------------------------
    print("== vision_encoder ==")
    w = EdgeTamVisionEncoderWrapper(model).eval()
    s = sess("vision_encoder")
    for t in (0, 4):
        px = g[f"f{t}/pixel_values"]
        with torch.no_grad():
            eager = [o.numpy() for o in w(torch.from_numpy(px))]
        # HF reference: decoder inputs captured from the video run. On the
        # initial conditioning frame pix_feat additionally carries
        # no_memory_embedding, so compare raw feats only vs the encoder inputs
        # of the memory path (enc_vision_features is always raw).
        check(f"f{t} eager vision_features vs HF", eager[0], g[f"f{t}/enc_vision_features"])
        check(f"f{t} eager high_res_0 vs HF", eager[2], g[f"f{t}/dec_high_res_0"])
        check(f"f{t} eager high_res_1 vs HF", eager[3], g[f"f{t}/dec_high_res_1"])
        got = s.run(None, {"pixel_values": px})
        for i, nm in enumerate(["vision_features", "vision_pos_embed",
                                "high_res_features_0", "high_res_features_1"]):
            # vision_features: justified 5e-4 (kernel-level fp32 noise on the
            # deepest path; persists with ORT_DISABLE_ALL, relative ~1e-4).
            tol = 5e-4 if nm == "vision_features" else TOL
            check(f"f{t} ORT vs eager {nm}", got[i], eager[i], tol=tol)

    # ---- no_mem_embed -------------------------------------------------------
    print("== no_mem_embed ==")
    w = EdgeTamNoMemEmbedWrapper(model).eval()
    s = sess("no_mem_embed")
    t = PROMPT_FRAME
    raw = g[f"f{t}/enc_vision_features"]
    with torch.no_grad():
        eager = w(torch.from_numpy(raw)).numpy()
    check(f"f{t} eager vs HF dec_pix_feat", eager, g[f"f{t}/dec_pix_feat"])
    got = s.run(None, {"vision_features": raw})[0]
    check(f"f{t} ORT vs eager", got, eager)

    # ---- memory_attention ---------------------------------------------------
    print("== memory_attention ==")
    w = EdgeTamMemoryAttentionWrapper(model).eval()
    s = sess("memory_attention")
    for t in range(PROMPT_FRAME + 1, NUM_FRAMES):
        feed = pad_memory_attention_inputs(g, t)
        with torch.no_grad():
            eager = w(*[torch.from_numpy(v) for v in feed.values()]).numpy()
        check(f"f{t} eager(padded) vs HF", eager, g[f"f{t}/ma_output"])
        got = s.run(None, feed)[0]
        check(f"f{t} ORT vs eager", got, eager)
    # padding-inertness: garbage in the masked region must not change output
    t = NUM_FRAMES - 1
    base = s.run(None, pad_memory_attention_inputs(g, t))[0]
    poisoned = s.run(None, pad_memory_attention_inputs(g, t, garbage=True))[0]
    check(f"f{t} ORT padding inertness (garbage-in-masked)", poisoned, base)

    # ---- mask_decoder_video -------------------------------------------------
    print("== mask_decoder_video ==")
    w = EdgeTamMaskDecoderVideoWrapper(model).eval()
    s = sess("mask_decoder_video")
    for t in range(NUM_FRAMES):
        feed = {
            "conditioned_features": g[f"f{t}/dec_pix_feat"],
            "high_res_features_0": g[f"f{t}/dec_high_res_0"],
            "high_res_features_1": g[f"f{t}/dec_high_res_1"],
            "point_coords": g[f"f{t}/dec_point_coords"].astype(np.float32),
            "point_labels": g[f"f{t}/dec_point_labels"].astype(np.int64),
        }
        with torch.no_grad():
            eager = [o.numpy() for o in w(*[torch.from_numpy(v) for v in feed.values()])]
        names = ["low_res_masks", "high_res_masks", "object_pointer",
                 "object_score_logits", "iou_scores"]
        hf_ref = [g[f"f{t}/dec_low_res_masks"], g[f"f{t}/dec_high_res_masks"],
                  g[f"f{t}/dec_object_pointer"], g[f"f{t}/dec_object_score_logits"],
                  g[f"f{t}/dec_iou_scores"]]
        got = s.run(None, feed)
        for i, nm in enumerate(names):
            if t in (0, 4):  # keep the log short: full check, sampled print
                check(f"f{t} eager vs HF {nm}", eager[i], hf_ref[i])
            check_quiet = check if t in (0, 4) else _quiet_check
            check_quiet(f"f{t} ORT vs eager {nm}", got[i], eager[i])

    # ---- memory_encoder -----------------------------------------------------
    print("== memory_encoder ==")
    w = EdgeTamMemoryEncoderWrapper(model).eval()
    s = sess("memory_encoder")
    for t in (0, 1, 5):
        feed = {
            "vision_features": g[f"f{t}/enc_vision_features"],
            "high_res_masks": g[f"f{t}/enc_high_res_masks"],
            "is_prompted": g[f"f{t}/enc_is_prompted"],
        }
        with torch.no_grad():
            eager = [o.numpy() for o in w(*[torch.from_numpy(v) for v in feed.values()])]
        check(f"f{t} eager vs HF memory_features", eager[0], g[f"f{t}/enc_memory_features"])
        check(f"f{t} eager vs HF memory_pos", eager[1], g[f"f{t}/enc_memory_pos"])
        got = s.run(None, feed)
        # memory_features: justified 5e-3 + cosine bound (perceiver layer-norm
        # gain amplifies ~4e-6 upstream kernel noise ~300x; see FINDINGS.md).
        check(f"f{t} ORT vs eager memory_features", got[0], eager[0], tol=5e-3)
        check_cosine(f"f{t} ORT vs eager memory_features (cos)", got[0], eager[0])
        check(f"f{t} ORT vs eager memory_pos", got[1], eager[1])

    print()
    if FAILURES:
        print(f"PARITY FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(" -", f)
        raise SystemExit(1)
    print("ALL PARITY CHECKS PASSED (fp32 max_abs < 1e-4; documented looser "
          "bounds on vision_features and memory_features, see module docstring)")


def _quiet_check(name, got, want, tol=TOL):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    if err >= tol:
        FAILURES.append(f"{name}: max_abs={err:.3e} (tol {tol})")
        print(f"  [FAIL] {name}: max_abs={err:.3e}")


if __name__ == "__main__":
    main()
