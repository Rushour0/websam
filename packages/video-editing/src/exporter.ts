import { NotImplementedError } from '@websam/core';
import type { MaskTimeline } from './timeline.js';

/**
 * What the exported pixels represent.
 *
 * - `'matte'` — a grayscale alpha matte: white where the object is, black
 *   elsewhere. Composable in any NLE.
 * - `'cutout'` — the source video's RGB with alpha punched from the mask,
 *   i.e. the object cut out over transparency.
 */
export type ExportMode = 'matte' | 'cutout';

/**
 * Container format for the export.
 *
 * - `'png-sequence'` — a zip of one PNG per frame. **The portable
 *   default**: lossless, alpha-capable, and importable by every editor and
 *   browser.
 * - `'webm-vp9-alpha'` — a single WebM video with a VP9 alpha channel.
 *   Compact and convenient, but encoding relies on WebCodecs VP9
 *   alpha support, which is **Chrome-only** today.
 * - `'auto'` — pick `'webm-vp9-alpha'` when the running browser can encode
 *   it, otherwise fall back to `'png-sequence'`.
 */
export type ExportFormat = 'png-sequence' | 'webm-vp9-alpha' | 'auto';

/**
 * Options for {@link AlphaMatteExporter.export}.
 */
export interface ExportOptions {
  /** What the exported pixels represent. See {@link ExportMode}. */
  mode: ExportMode;
  /**
   * Container format. Defaults to `'auto'`. See {@link ExportFormat} for
   * the portability trade-offs.
   */
  format?: ExportFormat;
  /**
   * Source frames, in presentation order, matching the timeline's frame
   * indexing. Required for `'cutout'` mode (the matte alone has no RGB);
   * ignored for `'matte'` mode.
   */
  source?: AsyncIterable<VideoFrame | ImageBitmap>;
  /** Progress callback, called with the number of frames finished so far. */
  onProgress?: (framesDone: number, frameCount: number) => void;
}

/**
 * The result of a finished export.
 */
export interface ExportResult {
  /** The encoded artifact: a zip for `'png-sequence'`, a WebM for `'webm-vp9-alpha'`. */
  blob: Blob;
  /** The format actually used (resolves `'auto'` to a concrete format). */
  format: Exclude<ExportFormat, 'auto'>;
  /** Suggested file name, e.g. `matte.zip` or `cutout.webm`. */
  suggestedFileName: string;
}

/**
 * Exports a {@link MaskTimeline} as alpha mattes or cutouts.
 *
 * M0 contract only: the surface is typed, but {@link export} throws
 * {@link NotImplementedError} until the PNG-sequence path lands in M2 and
 * the VP9-alpha path lands in M4.
 */
export class AlphaMatteExporter {
  /** The timeline whose masks will be exported. */
  readonly timeline: MaskTimeline;

  constructor(timeline: MaskTimeline) {
    this.timeline = timeline;
  }

  /**
   * Render and encode the timeline's masks.
   *
   * @throws NotImplementedError — always, at M0.
   */
  async export(options: ExportOptions): Promise<ExportResult> {
    void options;
    throw new NotImplementedError('AlphaMatteExporter, lands in M2 (PNG-zip) / M4 (VP9-alpha)');
  }
}
