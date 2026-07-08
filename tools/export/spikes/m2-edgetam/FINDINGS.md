# M2 EdgeTAM export spike — FINDINGS

Date: 2026-07-09. Go/no-go spike for websam M2 (EdgeTAM video tier).
Toolchain: transformers 5.13.0, torch 2.12.1, onnxscript 0.7.1, onnxslim 0.1.94,
onnxruntime 1.27.x (CPU EP), timm 1.0.27. Checkpoint: `yonigozlan/EdgeTAM-hf`
(the repo transformers' auto_docstrings name as canonical; `facebook/EdgeTAM`
only hosts the original `edgetam.pt`). Model: 13.90M params fp32.

All line citations: `transformers/models/edgetam_video/modeling_edgetam_video.py`
at v5.13.0 unless noted.

## VERDICT: GO — all five graphs export, pass parity, and the pure-ORT loop reproduces HF end-to-end

| Graph | Export path | Per-graph parity (ORT vs eager, real activations) | Verdict |
| --- | --- | --- | --- |
| `vision_encoder` | dynamo + optimize + onnxslim | max_abs 2.7–3.4e-4 on `vision_features` (justified, see gotcha 3); ≤1.5e-5 on the other 3 outputs | **GO** |
| `no_mem_embed` | dynamo | 0.0 (bit-exact) | **GO** |
| `memory_attention` | dynamo + optimize + onnxslim | ≤3.5e-6 all 7 tracked frames; wrapper-eager vs HF **bit-exact**; padding-inertness bit-exact | **GO** |
| `mask_decoder_video` | dynamo + optimize + onnxslim | ≤4.7e-5 across all outputs, all 8 frames; wrapper-eager vs HF bit-exact | **GO** |
| `memory_encoder` | dynamo, **no optimize, no onnxslim** (both break it — gotchas 1, 2) | ≤1.3e-3 max_abs / cosine ≥0.9999999 (justified, gotcha 4); wrapper-eager vs HF bit-exact | **GO** |

**Mini e2e (the real gate): pure-ORT 8-frame loop vs HF PyTorch+processor
end-to-end — IoU = 1.0000 on every frame** (binarized masks at original
640x480 resolution, gate was ≥0.95). `e2e_loop.py` is the executable spec.

Reproduce (all in this directory):

```sh
uv run --extra export python capture_golden.py   # HF golden + activations
uv run --extra export python export_edgetam.py   # 5 x .onnx
uv run --extra export python dump_constants.py   # tpos_table.npy
uv run --extra export python parity_graphs.py    # per-graph gates
uv run --extra export python e2e_loop.py         # pure-ORT loop + IoU gate
uv run --extra export --group dev pytest tests/test_wrappers_edgetam.py  # wrapper math vs HF (network-free)
```

## Artifact tensor tables (fp32, opset 18, ir 10)

### vision_encoder.onnx — 23.9 MB
RepViT-M1 (timm) backbone + FPN neck + the mask decoder's `conv_s0/s1` skip
projections pre-applied per frame (mirrors `get_image_features`, L2244-2273).

| Dir | Name | Dtype | Shape |
| --- | --- | --- | --- |
| in | `pixel_values` | float32 | (1, 3, 1024, 1024) |
| out | `vision_features` | float32 | (1, 256, 64, 64) — RAW (no no-mem embed) |
| out | `vision_pos_embed` | float32 | (1, 256, 64, 64) |
| out | `high_res_features_0` | float32 | (1, 32, 256, 256) |
| out | `high_res_features_1` | float32 | (1, 64, 128, 128) |

### no_mem_embed.onnx — 1.3 KB
| Dir | Name | Dtype | Shape |
| --- | --- | --- | --- |
| in | `vision_features` | float32 | (1, 256, 64, 64) |
| out | `conditioned_features` | float32 | (1, 256, 64, 64) |

Per-channel add of the learned `no_memory_embedding` (L2843-2852); used ONLY
on initial conditioning frames instead of memory attention. (Trivially
inlinable in JS if we ship the 256-float vector; kept as a graph for now.)

### memory_attention.onnx — 23.9 MB
| Dir | Name | Dtype | Shape |
| --- | --- | --- | --- |
| in | `current_vision_features` | float32 | (1, 256, 64, 64) — RAW frame feats |
| in | `current_vision_pos_embed` | float32 | (1, 256, 64, 64) |
| in | `memory` | float32 | (1, 3648, 64) — padded KV, layout below |
| in | `memory_pos_embed` | float32 | (1, 3648, 64) |
| in | `attn_bias` | float32 | (1, 1, 1, 3648) — 0 valid / −1e4 padding |
| out | `conditioned_features` | float32 | (1, 256, 64, 64) |

Frozen: `KV_LEN = 7 maps x 512 tokens + 64 ptr tokens = 3648`;
`rope_k_repeat = 7`, `num_k_exclude_rope = 64`. Bias is added to the
cross-attention score matrix only (self-attention over the 4096 queries is
never masked, matching HF). Padding proven inert (garbage in masked slots
changes nothing, bit-exact).

### mask_decoder_video.onnx — 26.8 MB
Fused prompt encoder + two-way-transformer decoder + occlusion clamp +
best-of-3 selection + object-pointer head (`_single_frame_forward`,
L2305-2501), `multimask_output=True` frozen (exact for ≤1-point prompts AND
tracked frames — `_use_multimask` L2900-2908).

| Dir | Name | Dtype | Shape |
| --- | --- | --- | --- |
| in | `conditioned_features` | float32 | (1, 256, 64, 64) |
| in | `high_res_features_0` | float32 | (1, 32, 256, 256) |
| in | `high_res_features_1` | float32 | (1, 64, 128, 128) |
| in | `point_coords` | float32 | (1, 1, **num_points** [dynamic], 2) — 1024-space px |
| in | `point_labels` | int64 | (1, 1, **num_points**) — 1 pos / 0 neg / −1 none |
| out | `low_res_masks` | float32 | (1, 1, 256, 256) logits (best mask, NO_OBJ clamped) |
| out | `high_res_masks` | float32 | (1, 1, 1024, 1024) logits |
| out | `object_pointer` | float32 | (1, 1, 256) — occlusion-blended |
| out | `object_score_logits` | float32 | (1, 1, 1) |
| out | `iou_scores` | float32 | (1, 1, 3) |

`num_points` is dynamic **on purpose**, not P=8-padded as planned: HF's `-10`
padding tokens embed to zero vectors but still participate as attention
tokens, so a frozen P=8 would NOT be numerically equivalent to HF's unpadded
runs (masks would deviate on every frame). Exact per-frame prompt shapes:
click frame `((x,y) scaled, label 1)`, tracked frames `((0,0), label −1)`
(L2414-2419). The prompt encoder itself appends one more (0,0)/−1 pad point
internally (L1654-1656) — inside the graph, JS never sees it.

### memory_encoder.onnx — 8.7 MB
Mask prep + memory encoder + **2D Spatial Perceiver** (the perceiver lives
here, compressing each encoded memory frame before storage — NOT in the
attention graph). `_encode_new_memory`, L3033-3073.

| Dir | Name | Dtype | Shape |
| --- | --- | --- | --- |
| in | `vision_features` | float32 | (1, 256, 64, 64) — RAW frame feats |
| in | `high_res_masks` | float32 | (1, 1, 1024, 1024) decoder logits |
| in | `is_prompted` | float32 | (1,) — 1.0 prompted frame (binarize), 0.0 tracked (sigmoid) |
| out | `memory_features` | float32 | (1, 512, 64) — one bank entry |
| out | `memory_pos_embed` | float32 | (1, 512, 64) — **frame-independent constant** (cache it) |

Sizes: 83.4 MB total fp32 (quant ladder is a later milestone).

## Key divergences from the SAM3-tracker semantics in spec.py

`spec.py`'s `EDGETAM_1024` tier is **wrong in two constants** and needs
updating:

1. **512 tokens per memory map, not 256.** The perceiver emits 256 *1D*
   latents (global, no spatial structure, positional encoding = zeros,
   L1493-1516) **concatenated with** 256 *2D* latents (16x16 window grid,
   sine positional encoding, L1518-1545). `perceiver_resampler_num_latents=256`
   AND `perceiver_resampler_num_latents_2d=256`. Empirically confirmed:
   stored `maskmem_features` are (1, 512, 64).
   → `kv_len = 7*512 + 64 = 3648`, not 1856.
2. **Conditioning frames are UNLIMITED, not capped at 4 (or 1).**
   `max_cond_frame_num = -1` → `_select_closest_cond_frames` (L2549-2563)
   returns ALL conditioning frames, and every one contributes a 512-token map
   AND competes for pointer slots. The "4 cond + 6 recent" SAM3 bank model
   does not apply. Consequence for the JS engine: with >1 prompted frame the
   HF KV grows beyond our frozen 7 maps. M2 policy decision needed —
   recommended: cap cond maps at `7 − min(6, #recent available)` using the
   `_select_closest_cond_frames` algorithm (closest-before, closest-after,
   then by |Δt|) with an explicit max, and document the deviation from HF
   unlimited-cond behavior. Single-prompt sessions (the M2 demo path) are
   exactly equivalent.
3. **tpos rule matches spec.py but via different mechanics.** Cond frames are
   gathered at `temporal_offset 0` (L2615) and the embedding index is
   `offset − 1 = −1`, i.e. Python-wraps to the LAST row (6) of the
   (7,1,1,64) `memory_temporal_positional_encoding` (L2663-2667). Recent
   offset k → row k−1. Net effect identical to spec.py's `tpos_index`
   (cond→6, k→k−1) — the JS engine must NOT index by physical slot.
4. **Object pointers need NO temporal position input.**
   `enable_temporal_pos_encoding_for_object_pointers = False` → pointer
   positional embeddings are ZEROS (L2766-2769). The plan's
   `pointer_time_deltas [16] int64` runtime input is a SAM3-only concern;
   EdgeTAM's graph has no pointer-time input at all. Pointer bank arithmetic
   is otherwise as spec'd: ≤16 pointers x 4 splits of 64 = 64 KV tokens,
   appended LAST (RoPE-excluded by trailing count).
5. **No occlusion spatial embedding.** `enable_occlusion_spatial_embedding =
   False` → the occluded-frame memory add (L3062-3066) is dead code;
   `object_score_logits` is NOT a memory-encoder input. (Occlusion still
   affects the decoder: NO_OBJ_SCORE=−1024 mask clamp + no-object pointer
   blend, both inside `mask_decoder_video`.)
6. **Streaming vs offline pointer cap is a non-issue here.** Offline HF uses
   `min(num_frames, 16)` pointers vs streaming 16 (L2691-2694), but since
   pointer temporal PE is disabled the cap only limits how many pointers are
   *gathered* — and the gather loop (offsets 1..15, break at frame 0,
   L2716-2721) yields identical sets for both modes. The JS engine can use
   streaming semantics unconditionally.
7. **Memory-attention RoPE is asymmetric.** Queries get 64x64-grid axial RoPE;
   memory keys get 16x16-grid RoPE applied ONLY to the 2D-latent half of each
   map (the 1D-latent half is treated as "temporal" and skipped, L419-433),
   repeated per map. Query features enter as `feats + 0.1 * pos` (L1253) —
   note the 0.1 factor.
8. **Preprocessing differs from SAM3.** EdgeTAM video uses `Sam2VideoProcessor`:
   square-stretch resize to **1024x1024**, bilinear antialias, ImageNet
   normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) — NOT SAM3's
   0.5/0.5. Prompts scale per-axis by 1024/srcW, 1024/srcH.
9. **Frame-0 conditioning masks are cached, not recomputed.** HF's `forward`
   (L2178-2186) returns the stored conditioning-frame output when propagation
   revisits it; the JS engine should do the same (decode once at click time).

## What the JS video engine must implement

The complete per-frame recipe with citations is the docstring of
`e2e_loop.py` (executable, IoU 1.0 vs HF). Summary:

* **memory-bank.ts**: two insertion-ordered stores (`cond` never evicted;
  `recent` ring of 6 — entries older than t−6 unreachable). Entry =
  512x64 memory features + 256-f32 object pointer. `maskmem_pos_enc` is a
  constant — fetch once (or bake into assets with `tpos_table` (7x64)).
* **KV assembly** per tracked frame into fixed (1,3648,64) buffers:
  cond maps (insertion order, +tpos row 6) → recent maps oldest-first
  (offset k → +tpos row k−1; skip cond/missing frames) → zero padding to
  3584 → pointer tokens (cond-past pointers in insertion order, then tracked
  offsets 1..15; 4 tokens each, pos rows zero) → zero padding to 3648.
  attn_bias: 0 valid / −1e4 padded. Whole-map (512-token) padding granularity
  is mandatory; pointer region is fixed at [3584, 3648).
* **Graph chaining**: vision → (no_mem_embed | memory_attention) →
  mask_decoder_video → memory_encoder → bank store. All IO already BCHW /
  batch-first — zero transposes between graphs (deliberate divergence from
  spec.py's seq-first convention; update spec.py when productionizing).
* **Prompts**: exact count (dynamic axis), no −10 padding. Tracked frames pass
  ((0,0), −1). Multi-point refinement (>1 point) needs a second decoder
  export with `multimask_output=False` (routes through
  `_dynamic_multimask_via_stability`, L1929-1931) — out of M2 scope,
  decision recorded.

## Export gotchas hit (and workarounds)

1. **`ONNXProgram.optimize()` (onnxscript 0.7.1) emits an invalid graph for
   the memory encoder** — dangling value refs (`val_215_1`-style) from
   deduplicating the perceiver layers, which are *called twice* (1D branch,
   then 2D branch, L1475-1483). ORT refuses to load ("Node input ... is not a
   graph input, initializer, or output of a previous node"). Workaround: skip
   `optimize()` for that graph (`SKIP_ONNXSCRIPT_OPTIMIZE` in
   `export_edgetam.py`). The unoptimized dynamo graph is valid.
2. **onnxslim 0.1.94 breaks the same graph a second, independent way**: its
   MatMul+Add→Gemm fusion produces `Gemm: Invalid bias shape for broadcast`
   at run time (node_linear_7, in the perceiver). Workaround: `SKIP_ONNXSLIM`
   for `memory_encoder`; graph ships straight from the dynamo exporter
   (8.7 MB vs 8.0 slimmed — negligible).
3. **`vision_features` ORT-vs-torch max_abs ≈ 2.7–3.4e-4 (> the 1e-4 bar), rel
   ≈ 1.1e-4.** Not a miscompilation: persists unchanged with
   `ORT_DISABLE_ALL` graph optimizations, and the wrapper is bit-exact vs HF
   in eager — it is fp32 kernel non-associativity (ORT vs PyTorch conv
   kernels) on the deepest backbone+FPN path. Accepted at 5e-4 with this
   justification; the e2e IoU gate (1.0000) is the arbiter.
4. **`memory_features` ORT-vs-torch up to 1.3e-3 (cosine ≥0.9999999).** Same
   class, amplified: staged isolation shows sigmoid noise ~4e-6 → memory-fuser
   ~4e-5 → perceiver ~1e-3; each stage's layer norms multiply tiny absolute
   input perturbations ~10-30x. Perceiver alone, fed exact eager inputs, is
   at 1.5e-5. Accepted at 5e-3 + cosine bound; e2e gate decisive.
5. **`timm` and `onnxscript` are hard deps of the export path** (RepViT
   backbone is a `TimmWrapper`; dynamo exporter imports onnxscript). Added to
   the `export` extra in pyproject.
6. **Frozen `-10`-padded points are NOT equivalent to unpadded prompts** (zero
   embeddings still attend) — resolved by exporting `num_points` as a dynamic
   axis (dynamo `torch.export.Dim`, worked first try). See decoder table.
7. What did NOT bite (pre-scoped as top risk): dynamo export of memory
   attention worked first try, including the padded-KV group-RoPE reshapes and
   the additive-bias threading. No TorchScript fallback needed anywhere.

## File inventory

| File | Role |
| --- | --- |
| `src/websam_export/wrappers/edgetam.py` | 5 reusable export wrappers (constants + line-cited re-implementations) |
| `tests/test_wrappers_edgetam.py` | network-free wrapper-vs-HF math tests (8 passing) |
| `spikes/m2-edgetam/capture_golden.py` | HF e2e run + activation capture (`activations/golden.npz`) |
| `spikes/m2-edgetam/export_edgetam.py` | export driver (dynamo primary, TS fallback, per-graph opt-out lists) |
| `spikes/m2-edgetam/parity_graphs.py` | per-graph ORT/eager/HF parity + padding-inertness gates |
| `spikes/m2-edgetam/e2e_loop.py` | pure-ORT loop = executable JS-engine spec (IoU 1.0 vs HF) |
| `spikes/m2-edgetam/dump_constants.py` | ships `tpos_table.npy` (7x64) |
| `spikes/m2-edgetam/clip_util.py` | synthetic 8-frame 640x480 test clip |

Heavy artifacts (`onnx/`, `activations/`) are gitignored; nothing committed.
