/**
 * Worker-side preprocessing: bitmap → normalized CHW pixel tensor, and
 * prompts (SOURCE pixels) → decoder input tensors (model-input pixels).
 *
 * Pixel pipeline (mirrors HF `Sam3ImageProcessorFast`, values from
 * `manifest.preprocess` — never TS constants, per
 * docs/m1-internal-contracts.md §7): resize via OffscreenCanvas 2D
 * `drawImage` (square-stretch: the whole square; letterbox: the scaled
 * content rect), then RGBA → Float32 CHW with `(v/255 − mean) / std`.
 * Letterbox pad regions are explicitly zeroed POST-normalization for
 * exactness (canvas transparency would otherwise normalize to −1).
 *
 * Prompt pipeline: every point/box coordinate goes through
 * {@link sourceToModel} — the ONE transform of the coordinate contract —
 * producing `input_points [1,1,N,2]` f32, `input_labels [1,1,N]` i64, and
 * `input_boxes [1,Nb,4]` f32 (empty `[1,0,4]` when no box prompt exists,
 * §1.1.1). Mask prompts are decoder-logit space and land in M2.
 */

import { sourceToModel, type CoordinateTransform } from '../coords.js';
import { InvalidStateError, NotImplementedError } from '../errors.js';
import type { Prompt } from '../segmenter.js';
import type { ModelManifest } from '../weights/manifest.js';

/** The manifest's preprocessing block (mode, inputSize, mean, std, maskSize). */
export type PreprocessSpec = ModelManifest['preprocess'];

/**
 * Convert row-major RGBA bytes (canvas `getImageData` layout) into a
 * normalized channels-first Float32 tensor:
 * `out[c*size² + y*size + x] = (rgba[(y*size + x)*4 + c] / 255 − mean[c]) / std[c]`.
 * Alpha is ignored. Pure math — unit-testable in node.
 *
 * @param rgba - `size * size * 4` bytes, RGBA interleaved.
 * @param size - Square side length in pixels.
 * @param mean - Per-channel mean (RGB), applied after the 1/255 rescale.
 * @param std - Per-channel std (RGB).
 */
export function rgbaToChw(
  rgba: Uint8ClampedArray | Uint8Array,
  size: number,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array {
  const pixels = size * size;
  if (rgba.length !== pixels * 4) {
    throw new InvalidStateError(
      `rgbaToChw: expected ${pixels * 4} RGBA bytes for ${size}x${size}, got ${rgba.length}`,
    );
  }
  const out = new Float32Array(3 * pixels);
  for (let c = 0; c < 3; c++) {
    const m = mean[c] as number;
    const s = std[c] as number;
    const plane = c * pixels;
    for (let i = 0; i < pixels; i++) {
      out[plane + i] = ((rgba[i * 4 + c] as number) / 255 - m) / s;
    }
  }
  return out;
}

/**
 * Zero every pixel of a normalized CHW tensor that lies OUTSIDE the
 * letterboxed content rect (`padX/padY` on each side, per the transform's
 * symmetric padding). No-op for square-stretch transforms (`pad = 0`).
 * Mutates `chw` in place.
 */
export function zeroPadRegions(chw: Float32Array, transform: CoordinateTransform): void {
  const size = transform.modelSize;
  const pixels = size * size;
  // Content rect [x0, x1) × [y0, y1); padding is symmetric by construction.
  const x0 = Math.round(transform.padX);
  const x1 = size - x0;
  const y0 = Math.round(transform.padY);
  const y1 = size - y0;
  if (x0 === 0 && y0 === 0) return;
  for (let c = 0; c < 3; c++) {
    const plane = c * pixels;
    for (let y = 0; y < size; y++) {
      const row = plane + y * size;
      if (y < y0 || y >= y1) {
        chw.fill(0, row, row + size);
      } else {
        if (x0 > 0) chw.fill(0, row, row + x0);
        if (x1 < size) chw.fill(0, row + x1, row + size);
      }
    }
  }
}

/**
 * Resize a bitmap into the model square via OffscreenCanvas and produce the
 * normalized CHW Float32 tensor the vision encoder eats.
 *
 * The caller keeps ownership of `bitmap` (the engine closes it — worker owns
 * it post-transfer, §3.4). Requires a worker/browser context; throws
 * {@link InvalidStateError} where `OffscreenCanvas` does not exist.
 *
 * @param bitmap - Decoded source image.
 * @param transform - The session transform (already computed from the bitmap
 * dims + `preprocess.mode`/`inputSize` by the engine).
 * @param preprocess - Manifest preprocessing block (mean/std, size).
 */
export function bitmapToTensor(
  bitmap: ImageBitmap,
  transform: CoordinateTransform,
  preprocess: PreprocessSpec,
): Float32Array {
  const size = preprocess.inputSize;
  if (transform.modelSize !== size) {
    throw new InvalidStateError(
      `bitmapToTensor: transform.modelSize ${transform.modelSize} != preprocess.inputSize ${size}`,
    );
  }
  if (typeof OffscreenCanvas !== 'function') {
    throw new InvalidStateError(
      'bitmapToTensor: OffscreenCanvas is not available in this environment ' +
        '(requires a browser window or worker context)',
    );
  }
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new InvalidStateError('bitmapToTensor: OffscreenCanvas 2d context unavailable');
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  // square-stretch: pad = 0 and the content rect is the whole square, so one
  // drawImage covers both modes.
  ctx.drawImage(
    bitmap,
    transform.padX,
    transform.padY,
    transform.srcW * transform.scaleX,
    transform.srcH * transform.scaleY,
  );
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const chw = rgbaToChw(rgba, size, preprocess.mean, preprocess.std);
  if (transform.mode === 'letterbox') zeroPadRegions(chw, transform);
  return chw;
}

/** Flat prompt tensors ready for `Backend.uploadTensor` (§1.1.1 shapes). */
export interface PromptTensors {
  /** `[1, 1, pointCount, 2]` float32, model-input pixel coords. */
  points: Float32Array;
  pointCount: number;
  /** `[1, 1, pointCount]` int64 (SAM convention: 1 = foreground, 0 = background). */
  labels: BigInt64Array;
  /** `[1, boxCount, 4]` float32 `(x1, y1, x2, y2)`, model-input pixel coords. */
  boxes: Float32Array;
  boxCount: number;
}

/**
 * Map prompts from SOURCE-pixel space into model-input space (via
 * {@link sourceToModel}, the coordinate contract's one transform) and pack
 * them into the decoder's tensor layouts. When no box prompt exists the
 * boxes tensor is the empty `[1, 0, 4]` feed (§1.1.1).
 *
 * @throws NotImplementedError — mask prompts (decoder-logit space feedback)
 * land in M2; the community decoder export exposes no mask input (S0).
 */
export function buildPromptTensors(
  prompts: readonly Prompt[],
  transform: CoordinateTransform,
): PromptTensors {
  const points: number[] = [];
  const labels: bigint[] = [];
  const boxes: number[] = [];
  for (const prompt of prompts) {
    if (prompt.type === 'point') {
      const p = sourceToModel({ x: prompt.x, y: prompt.y }, transform);
      points.push(p.x, p.y);
      labels.push(BigInt(prompt.label));
    } else if (prompt.type === 'box') {
      const a = sourceToModel({ x: prompt.x1, y: prompt.y1 }, transform);
      const b = sourceToModel({ x: prompt.x2, y: prompt.y2 }, transform);
      boxes.push(a.x, a.y, b.x, b.y);
    } else {
      throw new NotImplementedError('mask prompts, lands in M2');
    }
  }
  return {
    points: Float32Array.from(points),
    pointCount: points.length / 2,
    labels: BigInt64Array.from(labels),
    boxes: Float32Array.from(boxes),
    boxCount: boxes.length / 4,
  };
}
