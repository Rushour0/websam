/**
 * The concrete {@link MaskResult} implementation handed to callers by the
 * image (and later video) sessions.
 *
 * Immutability model (M1 decision, see docs/m1-internal-contracts.md §5):
 * **copy on construction, NOT pooling**. The constructor unconditionally
 * copies `binaryMask`, so no reference to caller memory is retained and no
 * websam code ever mutates the copy — which makes memoizing {@link
 * MaskResultImpl.toRLE} legal. Pooling arrives with the video path.
 *
 * Pure main-thread module: no worker/ORT imports. The RLE paths are fully
 * unit-testable in node; the DOM paths ({@link MaskResultImpl.toImageData},
 * {@link MaskResultImpl.toBitmap}) feature-detect their globals and throw
 * {@link InvalidStateError} in environments without them.
 */

import type { CoordinateTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';
import type { MaskResult } from '../segmenter.js';
import { encodeRLE, toCocoRLE, type CocoRLE, type RLEMask } from './rle.js';

/** Constructor payload for {@link MaskResultImpl}. */
export interface MaskResultInit {
  objectId: number;
  score: number;
  width: number;
  height: number;
  /** Row-major 0/1 bytes, length width*height (validated). */
  binaryMask: Uint8Array;
  transform: CoordinateTransform;
}

/**
 * Immutable segmentation result. See {@link MaskResult} for the public
 * contract; this class adds the {@link MaskResultImpl.transform} member per
 * coordinate-contract rule 4.
 */
export class MaskResultImpl implements MaskResult {
  readonly objectId: number;
  /** Predicted IoU / confidence score for this mask. */
  readonly score: number;
  /** Source-image width the mask maps onto. */
  readonly width: number;
  /** Source-image height the mask maps onto. */
  readonly height: number;
  /**
   * Coordinate-contract rule 4: every result carries the source↔model
   * transform it was decoded under. Extra member beyond the frozen public
   * {@link MaskResult} interface — structurally compatible; surfaced on the
   * interface post-M1.
   */
  readonly transform: CoordinateTransform;

  /** Private copy of the mask bytes; never exposed, never mutated. */
  readonly #mask: Uint8Array;
  /** Memoized {@link toRLE} result (legal: the instance is immutable). */
  #rle: RLEMask | undefined;

  /**
   * @param init - Result fields; `init.binaryMask` is **always copied**
   * (no reference to caller memory is retained) and its length is validated
   * against `width * height`.
   */
  constructor(init: MaskResultInit) {
    const { width, height, binaryMask } = init;
    if (!Number.isInteger(width) || width < 0 || !Number.isInteger(height) || height < 0) {
      throw new RangeError(`MaskResultImpl: invalid dimensions ${width}x${height}`);
    }
    if (binaryMask.length !== width * height) {
      throw new InvalidStateError(
        `MaskResultImpl: mask length ${binaryMask.length} does not match ` +
          `${width}x${height}=${width * height}`,
      );
    }
    this.objectId = init.objectId;
    this.score = init.score;
    this.width = width;
    this.height = height;
    this.transform = init.transform;
    this.#mask = binaryMask.slice();
  }

  /**
   * Render the mask as RGBA {@link ImageData}: set pixels become opaque
   * white `[255,255,255,255]`, unset pixels transparent black `[0,0,0,0]`.
   * Returns a fresh (caller-owned, mutable) instance every call.
   *
   * Throws {@link InvalidStateError} where the `ImageData` constructor does
   * not exist (e.g. plain Node without a DOM).
   */
  toImageData(): ImageData {
    if (typeof ImageData !== 'function') {
      throw new InvalidStateError(
        'MaskResult.toImageData: ImageData is not available in this environment ' +
          '(requires a browser window or worker context)',
      );
    }
    const { width, height } = this;
    const rgba = new Uint8ClampedArray(width * height * 4);
    const mask = this.#mask;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) rgba.fill(255, i * 4, i * 4 + 4);
    }
    return new ImageData(rgba, width, height);
  }

  /**
   * Render the mask as an {@link ImageBitmap} via
   * `createImageBitmap(this.toImageData())` — convenient for
   * `drawImage`-based canvas overlays.
   *
   * Rejects with {@link InvalidStateError} where `createImageBitmap` (or
   * `ImageData`) does not exist.
   */
  async toBitmap(): Promise<ImageBitmap> {
    if (typeof createImageBitmap !== 'function') {
      throw new InvalidStateError(
        'MaskResult.toBitmap: createImageBitmap is not available in this environment ' +
          '(requires a browser window or worker context)',
      );
    }
    return createImageBitmap(this.toImageData());
  }

  /**
   * Our documented row-major RLE (NOT COCO — use {@link toCocoRLE}).
   * Memoized: repeated calls return the same {@link RLEMask} instance, which
   * callers must treat as read-only.
   */
  toRLE(): RLEMask {
    this.#rle ??= encodeRLE(this.#mask, this.width, this.height);
    return this.#rle;
  }

  /** Real COCO RLE: column-major, pycocotools-compressed string counts. */
  toCocoRLE(): CocoRLE {
    return toCocoRLE(this.toRLE());
  }

  /** Row-major 0/1 bytes, `width * height`. Defensive copy every call. */
  toBinary(): Uint8Array {
    return this.#mask.slice();
  }
}
