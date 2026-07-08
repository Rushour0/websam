# The Coordinate Contract

Every coordinate bug in interactive segmentation comes from the same place:
two pieces of code disagreeing about which space a number lives in. This
document pins the spaces and the one transform between them. It is normative
for every websam package.

## Spaces

| Space | Units | Who lives here |
| --- | --- | --- |
| **Source-pixel space** | pixels of the user's image/video frame (`srcW × srcH`) | ALL user-facing prompts (points, boxes), all returned mask geometry |
| **Model-input space** | pixels of the square model input (`modelSize × modelSize`) | encoder inputs, prompt coordinates fed to the prompt encoder |
| **Decoder-logit space** | the mask decoder's low-res logit grid (288×288 for the SAM3 tracker export) | mask prompts (previous-mask logits fed back into the decoder) |

## Rules

1. **Prompts are in source-pixel space.** Users click on their image; they
   never see `modelSize`. Any API that accepts a point or box accepts source
   pixels.
2. **There is ONE source↔model transform.** It is computed by
   `computeTransform(srcW, srcH, modelSize, mode)` in `src/coords.ts` and
   applied by `sourceToModel` / `modelToSource`. No other code may derive
   scales or padding from image sizes. It mirrors the Hugging Face
   `image_processing_sam3_fast` preprocessing, because the exported graphs
   are traced against exactly that preprocessing.
3. **The mode is PINNED (M1-S0, 2026-07): `'square-stretch'`.**
   `Sam3ImageProcessorFast` does an anisotropic resize to 1008×1008 with
   `do_pad=None` (no letterbox branch exists), and
   `Sam3TrackerProcessor._normalize_coordinates` scales x by `1008/srcW`
   and y by `1008/srcH` independently with no offset. Pinned twice
   independently: from the HF source at the export-era tag AND at main
   (`tools/export/spikes/s0/FINDINGS.md`), and empirically via the
   transformers.js golden run on a 640×427 image
   (`tools/goldens/fixtures/golden-meta.json`: `reshaped_input_sizes`
   `[[1008,1008]]`, `pad_size: null`). Normalization: `raw/255`, then
   mean=std=0.5 (i.e. `raw/127.5 − 1`), RGB, bilinear+antialias. The
   `'letterbox'` mode in `src/coords.ts` stays implemented/tested for
   future model families (e.g. SAM1/SAM2-style exports).
4. **Every MaskResult carries its transform.** Results embed the
   `CoordinateTransform` they were produced under (mode, scales, padding,
   sizes), so downstream consumers map mask geometry back to source pixels
   without re-deriving anything — even across a future mode re-pin.
5. **Mask prompts live in decoder-logit space.** When feeding a previous
   mask back into the decoder, it is the decoder's low-res LOGIT grid, not a
   source-resolution binary mask. Compositing code upsamples logits to
   source space using rule 2's transform; prompt-side code never does.

## Golden test requirement

M1-S0 must land a golden test with a NON-SQUARE image (e.g. 1280×720):

- a fixed prompt point in source-pixel space,
- the reference mask produced by the HF reference pipeline,
- an assertion that our pipeline (with the pinned mode) reproduces it.

Square test images are forbidden as the only coverage — square inputs make
`square-stretch` and `letterbox` produce identical transforms
(`scaleX === scaleY`, zero padding), hiding exactly the bug this contract
exists to prevent. `src/coords.test.ts` already enforces the non-square
distinction at the math level.
