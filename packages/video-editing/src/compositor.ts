import { NotImplementedError } from '@websam/core';
import type { RLEMask } from '@websam/core';

/**
 * How masks are blended over the source frame.
 *
 * - `'highlight'` — tint the masked region with a per-object color over
 *   the original frame (the interactive-editing look).
 * - `'cutout'` — keep only the masked pixels; everything else transparent.
 * - `'matte'` — render the mask itself as a grayscale alpha matte.
 * - `'background-dim'` — keep masked pixels at full strength and darken
 *   the rest of the frame.
 */
export type CompositeMode = 'highlight' | 'cutout' | 'matte' | 'background-dim';

/**
 * Options for a single {@link MaskCompositor.composite} call.
 */
export interface CompositeOptions {
  /** Blend mode. See {@link CompositeMode}. */
  mode: CompositeMode;
  /**
   * Per-object CSS colors for `'highlight'` mode, keyed by object id.
   * Objects without an entry get an auto-assigned color.
   */
  colors?: ReadonlyMap<string, string>;
  /** Overlay opacity in `[0, 1]` for `'highlight'` and `'background-dim'`. Defaults to `0.5`. */
  opacity?: number;
}

/**
 * Composites RLE masks over video frames onto a canvas.
 *
 * M0 contract only: the surface is typed, but {@link composite} throws
 * {@link NotImplementedError} until the render path lands in M4.
 */
export class MaskCompositor {
  /** The canvas all composite calls draw into. */
  readonly canvas: OffscreenCanvas | HTMLCanvasElement;

  constructor(canvas: OffscreenCanvas | HTMLCanvasElement) {
    this.canvas = canvas;
  }

  /**
   * Draw `frame` onto the canvas with `masks` blended per `options.mode`.
   *
   * @param frame - The source frame to composite over.
   * @param masks - Masks for this frame keyed by object id, e.g. from
   * `MaskTimeline.getAll(frameIndex)`.
   * @throws NotImplementedError — always, at M0.
   */
  composite(
    frame: CanvasImageSource,
    masks: ReadonlyMap<string, RLEMask>,
    options: CompositeOptions,
  ): void {
    void frame;
    void masks;
    void options;
    throw new NotImplementedError('MaskCompositor, lands in M4');
  }
}
