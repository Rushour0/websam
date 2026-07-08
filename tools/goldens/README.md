# websam-goldens

Golden fixture generator for the **M1 gate**: the browser pipeline's mask for a
point prompt on a **non-square** image must reach **IoU >= 0.9** against these
transformers.js reference masks.

This directory is intentionally **outside the pnpm workspace** (the workspace
globs are `packages/*`, `apps/*`, `apps/bundler-matrix/*`). Install its deps
with plain `npm install` here — do not add it to `pnpm-workspace.yaml`.

## Reference model

- `onnx-community/sam3-tracker-ONNX` @ revision `429305c8a5b3de597243d919a07e4e6bdcd00ef7`
- transformers.js (`@huggingface/transformers`) 4.2.0, `Sam3TrackerModel` + `AutoProcessor` + `RawImage`
- dtype `q4f16` (`vision_encoder_q4f16.onnx` ~296 MB + `prompt_encoder_mask_decoder_q4f16.onnx` ~5 MB), device `cpu` (onnxruntime-node)
- Preprocessing (from `Sam3ImageProcessorFast`): the source image is resized
  (bilinear, **non-aspect-preserving**) to **1008x1008**, rescaled 1/255,
  normalized mean/std 0.5. There is no padding: `reshaped_input_sizes` is
  `[1008, 1008]` even for the 640x427 source. The decoder emits 288x288 mask
  logits which `post_process_masks` upsamples back to source resolution and
  binarizes at threshold **0.0**.
- Point prompts are given in **source-image pixel coords** `[x, y]`; the
  processor rescales them by `reshaped/original` per axis. Labels: 1 =
  positive, 0 = negative.
- The model returns 3 multimask candidates; the golden uses the candidate with
  the highest predicted `iou_scores` (recorded per fixture in the meta JSON).

## Fixtures (committed)

| file | what |
| --- | --- |
| `fixtures/scene-640x427.png` | deterministic synthetic scene (flat bg + orange circle + green rect + blue circle), geometry documented in `make-scene.mjs` |
| `fixtures/golden-mask-point1.{png,rle.json}` | mask for a single positive point at `(180, 210)` (big circle center) |
| `fixtures/golden-mask-point2.{png,rle.json}` | mask for a 2-point prompt: positive `(465, 220)` (rect center) + negative `(180, 210)` |
| `fixtures/golden-meta-point{1,2}.json` | full per-prompt metadata: revision, dtype, prompts, preprocessing sizes, iou scores, selected mask index, area/centroid |
| `fixtures/golden-meta.json` | combined summary of the above |

PNG masks are single-channel binary 0/255 at source resolution (640x427).

RLE format (`*.rle.json`): `{width, height, counts}` where `counts` are
alternating run lengths over a **row-major** scan (`index = y*width + x`),
starting with a run of **zeros** (`counts[0]` may be 0). `sum(counts) ==
width*height`. The generator asserts the RLE decodes back to the PNG
pixel-for-pixel.

## Regenerating

```sh
cd tools/goldens
npm install
. ~/.nvm/nvm.sh && nvm use 22
node make-scene.mjs   # deterministic; only needed if the scene definition changed
node generate.mjs     # downloads ~300 MB of model weights into ./models (gitignored)
```

`generate.mjs` runs built-in sanity checks (RLE==PNG roundtrip, area fraction
in (1%, 90%), centroid inside the prompted shape's bbox) and fails loudly if
any fixture is degenerate. Override the quantization with `DTYPE=fp16 node
generate.mjs` (recorded in the meta files).

If you regenerate with a different model revision, dtype, or transformers.js
version, the masks may shift slightly — commit the new fixtures and meta
together, and re-run the M1 gate.
