/**
 * Coordinate transforms between source-pixel space and model-input space.
 *
 * THE contract (see docs/coordinate-contract.md): all user-facing prompts
 * (points, boxes) are expressed in SOURCE-pixel space; exactly one transform
 * — computed once per (source size, model) pair by {@link computeTransform}
 * — maps them into model-input space, mirroring Hugging Face's
 * `image_processing_sam3_fast` preprocessing.
 *
 * Which {@link TransformMode} SAM3 actually uses (`'square-stretch'` vs
 * `'letterbox'`) is UNRESOLVED at M0 and will be pinned empirically in
 * M1-S0 against a golden non-square image; both modes are implemented and
 * tested here so pinning is a one-line constant change, not new code.
 */

/**
 * How a `srcW × srcH` image is fitted into the square `modelSize × modelSize`
 * model input.
 *
 * - `'square-stretch'` — anisotropic resize straight to the square;
 *   aspect ratio is NOT preserved, no padding (`padX = padY = 0`).
 * - `'letterbox'` — aspect-preserving resize by one uniform scale so the
 *   longer side equals `modelSize`, then center the image with symmetric
 *   padding on the short axis.
 */
export type TransformMode = 'square-stretch' | 'letterbox';

/** A 2D point. In websam APIs, prompt points are in source-pixel space. */
export interface Point {
  x: number;
  y: number;
}

/**
 * The one true source↔model transform. Carried on every MaskResult so masks
 * can always be mapped back to source pixels regardless of which mode M1-S0
 * pins (see docs/coordinate-contract.md).
 */
export interface CoordinateTransform {
  /** Horizontal scale from source pixels to model pixels. */
  scaleX: number;
  /** Vertical scale from source pixels to model pixels (=== scaleX when letterboxed). */
  scaleY: number;
  /** Horizontal padding (model pixels) added left of the image content. Always 0 for square-stretch. */
  padX: number;
  /** Vertical padding (model pixels) added above the image content. Always 0 for square-stretch. */
  padY: number;
  /** Source image width in pixels. */
  srcW: number;
  /** Source image height in pixels. */
  srcH: number;
  /** Model input side length in pixels (model inputs are square). */
  modelSize: number;
  /** The fitting mode this transform was computed with. */
  mode: TransformMode;
}

/**
 * Compute the source→model transform for an image.
 *
 * @param srcW - Source image width in pixels (must be > 0).
 * @param srcH - Source image height in pixels (must be > 0).
 * @param modelSize - Model input side length in pixels (must be > 0).
 * @param mode - Fitting mode; see {@link TransformMode}. Which one SAM3 uses
 * is pinned empirically in M1-S0 (docs/coordinate-contract.md).
 */
export function computeTransform(
  srcW: number,
  srcH: number,
  modelSize: number,
  mode: TransformMode,
): CoordinateTransform {
  if (!Number.isFinite(srcW) || srcW <= 0 || !Number.isFinite(srcH) || srcH <= 0) {
    throw new RangeError(`computeTransform: source size must be positive, got ${srcW}x${srcH}`);
  }
  if (!Number.isFinite(modelSize) || modelSize <= 0) {
    throw new RangeError(`computeTransform: modelSize must be positive, got ${modelSize}`);
  }

  if (mode === 'square-stretch') {
    return {
      scaleX: modelSize / srcW,
      scaleY: modelSize / srcH,
      padX: 0,
      padY: 0,
      srcW,
      srcH,
      modelSize,
      mode,
    };
  }

  // letterbox: one uniform scale, longer side fills the square, centered.
  const scale = modelSize / Math.max(srcW, srcH);
  return {
    scaleX: scale,
    scaleY: scale,
    padX: (modelSize - srcW * scale) / 2,
    padY: (modelSize - srcH * scale) / 2,
    srcW,
    srcH,
    modelSize,
    mode,
  };
}

/**
 * Map a source-pixel-space point into model-input space:
 * `model = source * scale + pad` per axis. Continuous coordinates — no
 * rounding, so round-trips through {@link modelToSource} are exact up to
 * floating point.
 */
export function sourceToModel(point: Point, transform: CoordinateTransform): Point {
  return {
    x: point.x * transform.scaleX + transform.padX,
    y: point.y * transform.scaleY + transform.padY,
  };
}

/**
 * Inverse of {@link sourceToModel}: map a model-input-space point back to
 * source pixels: `source = (model - pad) / scale` per axis.
 */
export function modelToSource(point: Point, transform: CoordinateTransform): Point {
  return {
    x: (point.x - transform.padX) / transform.scaleX,
    y: (point.y - transform.padY) / transform.scaleY,
  };
}
