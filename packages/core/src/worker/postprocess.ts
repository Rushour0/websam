/**
 * Worker-side postprocessing: decoder logit grid → source-resolution binary
 * mask.
 *
 * Mirrors the HF `post_process_masks` semantics the goldens were generated
 * under (tools/goldens/README.md): the decoder's low-res logits (288×288 for
 * the SAM3 tracker export) are bilinearly upsampled to the model square and
 * mapped back to source resolution, then binarized at threshold **0.0**.
 *
 * Instead of materializing the intermediate `modelSize²` grid, this samples
 * the logit grid directly once per SOURCE pixel: map the pixel center
 * through the session {@link CoordinateTransform} into the model square,
 * scale by `gridSize / modelSize` into logit-grid coordinates
 * (pixel-center/`align_corners=False` convention, matching torch
 * `interpolate`), bilinear-sample, and threshold `> 0` (§3.4).
 */

import type { CoordinateTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';

/**
 * Convert one candidate's logit grid into a row-major 0/1 mask at source
 * resolution (`transform.srcW × transform.srcH`).
 *
 * @param logits - Row-major `gridSize * gridSize` decoder logits.
 * @param gridSize - Logit grid side length (e.g. 288 = `preprocess.maskSize`).
 * @param transform - The session's source↔model transform; also carries the
 * source dims the mask maps onto.
 */
export function logitsToSourceMask(
  logits: Float32Array,
  gridSize: number,
  transform: CoordinateTransform,
): Uint8Array {
  if (logits.length !== gridSize * gridSize) {
    throw new InvalidStateError(
      `logitsToSourceMask: expected ${gridSize * gridSize} logits for a ` +
        `${gridSize}x${gridSize} grid, got ${logits.length}`,
    );
  }
  const { srcW, srcH, scaleX, scaleY, padX, padY, modelSize } = transform;
  const gridScale = gridSize / modelSize;
  const maxG = gridSize - 1;
  const out = new Uint8Array(srcW * srcH);

  // Per-column sampling coordinates are y-invariant: precompute once.
  const colLo = new Int32Array(srcW);
  const colHi = new Int32Array(srcW);
  const colFrac = new Float32Array(srcW);
  for (let x = 0; x < srcW; x++) {
    // Source pixel center → model square → logit grid (align_corners=False).
    const gx = Math.min(Math.max(((x + 0.5) * scaleX + padX) * gridScale - 0.5, 0), maxG);
    const lo = Math.floor(gx);
    colLo[x] = lo;
    colHi[x] = Math.min(lo + 1, maxG);
    colFrac[x] = gx - lo;
  }

  for (let y = 0; y < srcH; y++) {
    const gy = Math.min(Math.max(((y + 0.5) * scaleY + padY) * gridScale - 0.5, 0), maxG);
    const yLo = Math.floor(gy);
    const yHi = Math.min(yLo + 1, maxG);
    const fy = gy - yLo;
    const rowLo = yLo * gridSize;
    const rowHi = yHi * gridSize;
    const outRow = y * srcW;
    for (let x = 0; x < srcW; x++) {
      const xLo = colLo[x] as number;
      const xHi = colHi[x] as number;
      const fx = colFrac[x] as number;
      const top =
        (logits[rowLo + xLo] as number) * (1 - fx) + (logits[rowLo + xHi] as number) * fx;
      const bottom =
        (logits[rowHi + xLo] as number) * (1 - fx) + (logits[rowHi + xHi] as number) * fx;
      out[outRow + x] = top * (1 - fy) + bottom * fy > 0 ? 1 : 0;
    }
  }
  return out;
}
