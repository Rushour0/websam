# S0 Findings — reuse of `onnx-community/sam3-tracker-ONNX` vision encoder

Date: 2026-07-08. Spike for websam M1-S0.
Artifacts inspected: `onnx/vision_encoder.onnx` + `onnx/prompt_encoder_mask_decoder.onnx`
(fp32 graph protobufs, downloaded WITHOUT external `.onnx_data` weights; loaded via
`onnx.load(..., load_external_data=False)` — see `inspect_graphs.py`).

## Verdict: REUSE-OK

The community vision encoder emits **all three FPN levels** (high-res feats included),
its output names/shapes match the decoder's inputs exactly, and the preprocessing
contract is fully pinned from source: **square-stretch** resize to 1008×1008,
mean/std = 0.5, prompts rescaled per-axis by `1008/srcW` and `1008/srcH`.
Nothing needs re-exporting for the single-image encode→decode path.

Caveats (why not more than OK):

- Fp32 vision encoder weights are **1.87 GB** (`vision_encoder.onnx_data`); fp16 is
  935 MB, q4f16 is 296 MB. Browser deployment will want a quantized variant
  (`transformers.js_config` in `config.json` sets `use_external_data_format: true`).
- These two graphs cover the **per-frame image path only**. The repo's video/memory
  components (memory attention, memory encoder — `num_maskmem: 7`,
  `memory_attention_rope_feat_sizes: [72, 72]` in `config.json`) are not in these
  two graphs; tracking across frames needs the other exported graphs or our own export.
- No `mask_input`/previous-logits input on this decoder export (see table below) —
  iterative mask-prompt refinement (contract rule 5) is NOT exposed by this
  community decoder. If M-later needs mask prompts, the decoder (not the encoder)
  must be re-exported. The vision encoder itself remains reusable regardless.

## Tensor tables

Both graphs: ONNX ir_version 8, **opset 18** (ai.onnx), producer pytorch 2.9.0.
Model config: `image_size: 1008`, `model_type: sam3_tracker`, exported at
`transformers_version: 5.0.0.dev0`.

### vision_encoder.onnx

| Direction | Name | Dtype | Shape |
| --- | --- | --- | --- |
| input | `pixel_values` | float32 | `[batch_size, 3, 1008, 1008]` |
| output | `image_embeddings.0` | float32 | `[batch_size, ?, ?, ?]` (symbolic; = `[B, 32, 288, 288]`, pinned by decoder input below) |
| output | `image_embeddings.1` | float32 | `[batch_size, ?, ?, ?]` (= `[B, 64, 144, 144]`) |
| output | `image_embeddings.2` | float32 | `[batch_size, ?, ?, ?]` (= `[B, 256, 72, 72]`) |

**Answer to the critical question: YES — the high-res FPN feature maps are emitted**
in addition to the main embedding. The encoder's own output shapes are symbolic
(`Reshapeimage_embeddings.N_dim_*`), but the decoder graph declares them with
concrete dims, which fixes the contract:
`image_embeddings.0 = [B,32,288,288]` (high-res), `image_embeddings.1 = [B,64,144,144]`
(mid-res), `image_embeddings.2 = [B,256,72,72]` (main embedding). Output names match
decoder input names 1:1, so features can be piped straight through.

### prompt_encoder_mask_decoder.onnx

| Direction | Name | Dtype | Shape |
| --- | --- | --- | --- |
| input | `input_points` | float32 | `[batch_size, 1, num_points_per_image, 2]` |
| input | `input_labels` | int64 | `[batch_size, 1, num_points_per_image]` |
| input | `input_boxes` | float32 | `[batch_size, num_boxes_per_image, 4]` |
| input | `image_embeddings.0` | float32 | `[batch_size, 32, 288, 288]` |
| input | `image_embeddings.1` | float32 | `[batch_size, 64, 144, 144]` |
| input | `image_embeddings.2` | float32 | `[batch_size, 256, 72, 72]` |
| output | `iou_scores` | float32 | `[batch_size, num_boxes_or_points, 3]` |
| output | `pred_masks` | float32 | `[batch_size, num_boxes_or_points, num_masks, H, W]` (H/W symbolic; decoder logit grid) |
| output | `object_score_logits` | float32 | `[batch_size, num_boxes_or_points, 1]` |

Yes — the decoder **consumes the high-res feats** (`image_embeddings.0/.1`) alongside
the main embedding. Point convention: `(x, y)` pairs in **model-input pixel space**
(0..1008, see below), labels int64 (SAM convention: 1 = foreground, 0 = background;
padding points use pad value −10, handled by the processor). Boxes are
`(x1, y1, x2, y2)` in the same 1008-pixel space (rescaled as two corner points).
There is **no `mask_input` / `has_mask_input`** and no `orig_im_size` input.

## Preprocessing contract (with citations)

The checkpoint's `preprocessor_config.json` names `Sam3ImageProcessorFast`. That class
lives at `src/transformers/models/sam3/image_processing_sam3_fast.py` in transformers
**v5.0.0** (the export-era version, matching `transformers_version: 5.0.0.dev0`); on
**main** (checked at commit `bce8fd08f6d9`, 2026-07-08) it was consolidated into
`src/transformers/models/sam3/image_processing_sam3.py` as `Sam3ImageProcessor`
(TorchvisionBackend) — the auto-mapping `("sam3_tracker", {"torchvision": "Sam3ImageProcessor"})`
in `models/auto/image_processing_auto.py:142` routes the checkpoint there. Both
versions are behaviorally identical for everything below. Local copies of the fetched
sources are in this directory (`image_processing_sam3.py`, `image_processing_sam3_fast_v5.0.0.py`,
`image_processing_backends.py`, `processing_sam3_tracker.py`).

### (a) Resize mode: SQUARE STRETCH (anisotropic), no padding

- `Sam3ImageProcessorFast` (v5.0.0, `image_processing_sam3_fast.py:399-415`) /
  `Sam3ImageProcessor` (main, `image_processing_sam3.py:401-416`) class defaults:
  `size = {"height": 1008, "width": 1008}` and — under the literal comment
  `# disable SAM padding logic` — `do_pad = None`, `pad_size = None`,
  `mask_pad_size = None`. (Checkpoint `preprocessor_config.json` agrees:
  `"do_pad": null`, `"pad_size": null`, `"size": {"height": 1008, "width": 1008}`.)
- The inherited resize takes the height+width branch — v5.0.0
  `image_processing_utils_fast.py:458-459` / main `image_processing_backends.py:259-260`:

  ```python
  elif size.height and size.width:
      new_size = (size.height, size.width)
  ```

  then (v5.0.0 line 470 / main line 270):

  ```python
  return tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)
  ```

  `torchvision` `resize` with an explicit `(h, w)` tuple does a direct anisotropic
  resize — **no aspect-ratio preservation, no letterbox, no padding**. A 1280×720
  frame is stretched non-uniformly to 1008×1008.
- Interpolation: `resample = PILImageResampling.BILINEAR` (v5.0.0 fast file line 400;
  checkpoint config `"resample": 2`) → torchvision BILINEAR with `antialias=True`.

### (b) Normalization

- `rescale_factor = 1/255` (`0.00392156862745098` in `preprocessor_config.json`).
- `image_mean = image_std = [0.5, 0.5, 0.5]` (IMAGENET_STANDARD_MEAN/STD; class
  defaults at v5.0.0 fast file lines 401-402, confirmed by `preprocessor_config.json`).
- Net: `pixel = (raw/255 − 0.5) / 0.5 = raw/127.5 − 1` → range **[−1, 1]**, RGB,
  channels-first (`"data_format": "channels_first"`).

### (c) Prompt rescaling in `Sam3TrackerProcessor`

`processing_sam3_tracker.py` (identical at v5.0.0 and main):

- `target_size` defaults to `image_processor.size["height"]` = **1008** (line 50).
- `_normalize_coordinates` (v5.0.0 lines 173-196 / main 174-197):

  ```python
  old_h, old_w = original_size
  new_h, new_w = target_size, target_size
  ...
  coords[..., 0] = coords[..., 0] * (new_w / old_w)
  coords[..., 1] = coords[..., 1] * (new_h / old_h)
  ```

  i.e. x is scaled by `1008/srcW` and y **independently** by `1008/srcH` — the exact
  prompt-side mirror of the anisotropic square-stretch image resize. **No offset is
  ever added** (no padding to account for). Boxes are reshaped `(-1, 2, 2)` and each
  corner rescaled the same way (`is_bounding_box=True` path). Output coordinates are
  **pixels in the 1008×1008 model-input grid**, not normalized to [0,1].
- Padding points (batching ragged prompt lists) use `point_pad_value = -10` and are
  excluded from rescaling (`_normalize_tensor_coordinates`, `preserve_padding=True`).

## Update needed to `packages/core/docs/coordinate-contract.md`

Rule 3 currently reads "The mode is pinned empirically in M1-S0 — currently
UNRESOLVED." Change to pin the mode:

- The correct `TransformMode` from `packages/core/src/coords.ts` is
  **`'square-stretch'`** — anisotropic resize to `1008 × 1008` with zero padding;
  `scaleX = 1008/srcW`, `scaleY = 1008/srcH`, `padX = padY = 0`.
- `modelSize = 1008`; decoder mask grid follows `mask_size = 288 × 288`
  (`preprocessor_config.json`) — note this equals the `image_embeddings.0` spatial
  size, and the doc's "e.g. 256×256" example for decoder-logit space should become
  288×288 for this model.
- Evidence class: source-pinned at both transformers v5.0.0 (export-era) and main —
  image resize (`(h,w)`-tuple `tvF.resize`, `do_pad=None`) and prompt rescale
  (independent per-axis `1008/old_w`, `1008/old_h`, no offset) are both anisotropic
  and mutually consistent. Per the contract's own bar, the golden non-square-image
  test remains the final empirical confirmation; it requires downloading the weight
  data (1.87 GB fp32 / 296 MB q4f16), which was out of scope for this graph-only
  spike. There is no letterbox branch anywhere in the code path for prompts to
  disagree with.

## Surprises / notes

1. Vision-encoder output shapes are symbolic in its own graph; the concrete FPN
   shapes are only declared on the decoder's inputs. Anyone validating the encoder
   in isolation should pin against the decoder's declared shapes.
2. The community decoder has **no mask-prompt input** (`mask_input`) — contract
   rule 5 (feeding previous logits back) cannot be exercised with this export.
3. `preprocessor_config.json` says `"processor_class": "Sam2Processor"` (not Sam3) —
   harmless (the tracker processor shares SAM2's conventions) but worth knowing
   when loading via AutoProcessor.
4. `iou_scores` last dim is 3 (three mask candidates), matching
   `multimask_output_in_sam: true`; `pred_masks` has a separate `num_masks` dim.
5. `image_processing_sam3_fast.py` no longer exists on transformers main — the v5
   backend refactor merged fast processors into the plain file
   (`Sam3ImageProcessor(TorchvisionBackend)`); behavior verified identical.
