import { decodeRLE, InvalidStateError, NotImplementedError, type RLEMask } from '@websam/core';
import { Zip, ZipPassThrough } from 'fflate';
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
  /**
   * Progress callback. Called once per timeline frame index with the number
   * of frame indices processed so far and the total frame count — sparse
   * (hole) frames advance the count too, so progress tracks reality.
   */
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
  /**
   * Number of PNG frames actually written across all objects. Sparse
   * timelines (e.g. from a cancelled propagation) export fewer frames than
   * `timeline.frameCount`; holes are skipped, not failed.
   */
  framesExported: number;
}

/**
 * Per-object frame index lists plus timeline geometry, written as
 * `timeline.json` at the zip root so importers can reassemble the sequence.
 */
interface TimelineManifest {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  /** Object id → ascending list of frame indices that have a PNG. */
  objects: Record<string, number[]>;
}

/**
 * Exports a {@link MaskTimeline} as alpha mattes or cutouts.
 *
 * M2: `'matte'` + `'png-sequence'` (and `'auto'`, which resolves to
 * `'png-sequence'`) are real — a store-mode zip of one white-on-black PNG per
 * tracked frame plus a `timeline.json` index. `'cutout'` and
 * `'webm-vp9-alpha'` still throw {@link NotImplementedError} (they land in
 * M4). PNG rendering needs `OffscreenCanvas`/`ImageData`, so the matte path is
 * browser/worker-only.
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
   * @throws NotImplementedError — for `'cutout'` mode or `'webm-vp9-alpha'`
   * format (both land in M4).
   * @throws InvalidStateError — when `OffscreenCanvas`/`ImageData` are
   * unavailable (i.e. outside a browser or worker).
   */
  async export(options: ExportOptions): Promise<ExportResult> {
    if (options.mode === 'cutout') {
      throw new NotImplementedError('cutout export, lands in M4');
    }
    const format = options.format ?? 'auto';
    if (format === 'webm-vp9-alpha') {
      throw new NotImplementedError('webm-vp9-alpha export, lands in M4');
    }
    // mode === 'matte', format is 'png-sequence' or 'auto'; 'auto' resolves to
    // 'png-sequence' at M2 (VP9-alpha detection lands in M4).
    return this.exportPngSequence(options);
  }

  private async exportPngSequence(options: ExportOptions): Promise<ExportResult> {
    const { timeline } = this;
    const objectIds = timeline.objectIds();
    const multiObject = objectIds.length > 1;

    // Streaming store-mode zip: fflate emits chunks as we push each entry, so
    // we never hold every PNG plus the whole zip in memory at once.
    // Concrete-ArrayBuffer copies: fflate reuses its output buffers between
    // callbacks, and the copy also narrows to `Uint8Array<ArrayBuffer>` (a
    // valid `BlobPart`).
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let resolveBlob!: (blob: Blob) => void;
    let rejectBlob!: (err: unknown) => void;
    const blobPromise = new Promise<Blob>((resolve, reject) => {
      resolveBlob = resolve;
      rejectBlob = reject;
    });
    const zip = new Zip((err, data, final) => {
      if (err) {
        rejectBlob(err);
        return;
      }
      chunks.push(new Uint8Array(data));
      if (final) resolveBlob(new Blob(chunks, { type: 'application/zip' }));
    });

    const manifest: TimelineManifest = {
      fps: timeline.fps,
      frameCount: timeline.frameCount,
      width: timeline.width,
      height: timeline.height,
      objects: {},
    };
    for (const id of objectIds) manifest.objects[id] = [];

    let framesExported = 0;
    try {
      for (let frameIndex = 0; frameIndex < timeline.frameCount; frameIndex++) {
        for (const id of objectIds) {
          const rle = timeline.get(id, frameIndex);
          if (rle === undefined) continue; // hole — skipped, not failed
          const png = await renderMatte(rle);
          const name = multiObject
            ? `obj-${id}/frame-${pad6(frameIndex)}.png`
            : `frame-${pad6(frameIndex)}.png`;
          const entry = new ZipPassThrough(name);
          zip.add(entry);
          entry.push(png, true);
          manifest.objects[id]?.push(frameIndex);
          framesExported++;
        }
        options.onProgress?.(frameIndex + 1, timeline.frameCount);
      }

      const metaEntry = new ZipPassThrough('timeline.json');
      zip.add(metaEntry);
      metaEntry.push(new TextEncoder().encode(JSON.stringify(manifest)), true);
      zip.end();
    } catch (err) {
      // Surface the failure through the thrown error; `blobPromise` stays
      // pending (never rejected) so there is no unhandled rejection to leak.
      throw err;
    }

    const blob = await blobPromise;
    return {
      blob,
      format: 'png-sequence',
      suggestedFileName: 'matte.zip',
      framesExported,
    };
  }
}

/** Zero-pad a frame index to six digits (`frame-000042.png`). */
function pad6(n: number): string {
  return String(n).padStart(6, '0');
}

/**
 * Render one mask into PNG bytes: a white-on-black opaque RGBA matte encoded
 * via `OffscreenCanvas.convertToBlob({type:'image/png'})`.
 *
 * @throws InvalidStateError — when `OffscreenCanvas`/`ImageData` are absent.
 */
async function renderMatte(rle: RLEMask): Promise<Uint8Array> {
  if (typeof OffscreenCanvas === 'undefined' || typeof ImageData === 'undefined') {
    throw new InvalidStateError(
      'AlphaMatteExporter: PNG export needs OffscreenCanvas/ImageData (browser or worker only)',
    );
  }
  const { width, height } = rle;
  const binary = decodeRLE(rle);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < binary.length; i++) {
    const value = binary[i] ? 255 : 0;
    const offset = i * 4;
    rgba[offset] = value;
    rgba[offset + 1] = value;
    rgba[offset + 2] = value;
    rgba[offset + 3] = 255; // opaque
  }
  const image = new ImageData(rgba, width, height);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (ctx === null) {
    throw new InvalidStateError('AlphaMatteExporter: OffscreenCanvas 2D context unavailable');
  }
  ctx.putImageData(image, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return new Uint8Array(await blob.arrayBuffer());
}
