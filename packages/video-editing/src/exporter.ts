import { InvalidStateError, NotImplementedError, decodeRLE } from '@websam/core';
import type { RLEMask } from '@websam/core';
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
   * Progress callback, called after each frame index is processed with the
   * number of frame indices finished so far and the timeline's total
   * `frameCount`. Frames with no stored mask (holes) still count as
   * finished — they are skipped, not failed.
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
   * How many mask images were actually exported (one per stored
   * object×frame). Sparse timelines are normal — a cancelled propagation
   * leaves holes, which are skipped — so this reports reality rather than
   * `frameCount × objects`.
   */
  framesExported: number;
}

/**
 * Sidecar metadata written as `timeline.json` at the zip root of a
 * `'png-sequence'` export: the timeline dimensions plus, per object, the
 * frame indices that were actually exported.
 */
interface TimelineSidecar {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  /** Ascending frame indices exported for each object, keyed by object id. */
  objects: Record<string, number[]>;
}

/**
 * Exports a {@link MaskTimeline} as alpha mattes or cutouts.
 *
 * M2 surface: `'matte'` + `'png-sequence'` is real (`'auto'` resolves to
 * `'png-sequence'`); `'cutout'` and `'webm-vp9-alpha'` still throw
 * {@link NotImplementedError} until M4.
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
   * `'matte'` + `'png-sequence'` produces a **store-mode** zip (PNGs are
   * already DEFLATE-compressed; fflate is used purely as a streaming
   * container writer). Layout: with a single tracked object, PNGs sit at
   * the zip root as `frame-%06d.png`; with multiple objects each gets a
   * folder, `obj-<id>/frame-%06d.png`. A `timeline.json` sidecar records
   * fps, frameCount, width, height, and the per-object frame index lists.
   * Frames with no stored mask are **skipped, not failed** —
   * {@link ExportResult.framesExported} and
   * {@link ExportOptions.onProgress} report reality.
   *
   * @throws NotImplementedError — `'cutout'` mode and `'webm-vp9-alpha'`
   * format, both landing in M4.
   * @throws InvalidStateError — no `OffscreenCanvas` available (the PNG
   * path is browser/worker-only) or a mask's dimensions do not match the
   * timeline's.
   */
  async export(options: ExportOptions): Promise<ExportResult> {
    if (options.mode === 'cutout') {
      throw new NotImplementedError('cutout export, lands in M4');
    }
    const format = options.format ?? 'auto';
    if (format === 'webm-vp9-alpha') {
      throw new NotImplementedError('webm-vp9-alpha export, lands in M4');
    }
    // 'matte' + ('png-sequence' | 'auto'): VP9-alpha detection lands in M4,
    // so 'auto' resolves to 'png-sequence' at M2.
    return this.#exportPngSequence(options);
  }

  async #exportPngSequence(options: ExportOptions): Promise<ExportResult> {
    if (typeof OffscreenCanvas === 'undefined') {
      throw new InvalidStateError(
        'png-sequence export requires OffscreenCanvas — run in a browser window or worker',
      );
    }
    const { timeline } = this;
    const canvas = new OffscreenCanvas(timeline.width, timeline.height);
    const ctx = canvas.getContext('2d');
    if (ctx === null) {
      throw new InvalidStateError('png-sequence export: OffscreenCanvas 2d context unavailable');
    }

    // Streaming store-mode zip: chunks accumulate as they are produced, so
    // at no point do all rendered PNGs AND the finished zip coexist twice.
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let zip!: Zip;
    const zipDone = new Promise<void>((resolve, reject) => {
      zip = new Zip((err, chunk, final) => {
        if (err) {
          reject(err);
          return;
        }
        // fflate emits Uint8Array<ArrayBufferLike>; every chunk it produces
        // here is freshly allocated or one we handed it, never reused.
        chunks.push(chunk as Uint8Array<ArrayBuffer>);
        if (final) resolve();
      });
    });
    const addEntry = (name: string, bytes: Uint8Array): void => {
      // ZipPassThrough = store mode: no compression work on PNG bytes.
      const entry = new ZipPassThrough(name);
      zip.add(entry);
      entry.push(bytes, true);
    };

    const objectIds = timeline.objectIds();
    const singleObject = objectIds.length === 1;
    const exportedFrames = new Map<string, number[]>(objectIds.map((id) => [id, []]));
    let framesExported = 0;

    for (let frameIndex = 0; frameIndex < timeline.frameCount; frameIndex++) {
      for (const objectId of objectIds) {
        const rle = timeline.get(objectId, frameIndex);
        if (rle === undefined) continue; // hole: skipped, not failed
        const prefix = singleObject ? '' : `obj-${objectId}/`;
        const name = `${prefix}frame-${String(frameIndex).padStart(6, '0')}.png`;
        addEntry(name, await renderMattePng(canvas, ctx, rle, timeline));
        exportedFrames.get(objectId)?.push(frameIndex);
        framesExported += 1;
      }
      options.onProgress?.(frameIndex + 1, timeline.frameCount);
    }

    const sidecar: TimelineSidecar = {
      fps: timeline.fps,
      frameCount: timeline.frameCount,
      width: timeline.width,
      height: timeline.height,
      objects: Object.fromEntries(exportedFrames),
    };
    addEntry('timeline.json', new TextEncoder().encode(JSON.stringify(sidecar)));
    zip.end();
    await zipDone;

    return {
      blob: new Blob(chunks, { type: 'application/zip' }),
      format: 'png-sequence',
      suggestedFileName: 'matte.zip',
      framesExported,
    };
  }
}

/**
 * Render one RLE mask as a white-on-black, fully opaque RGBA PNG via the
 * shared OffscreenCanvas.
 */
async function renderMattePng(
  canvas: OffscreenCanvas,
  ctx: OffscreenCanvasRenderingContext2D,
  rle: RLEMask,
  dims: { width: number; height: number },
): Promise<Uint8Array> {
  if (rle.width !== dims.width || rle.height !== dims.height) {
    throw new InvalidStateError(
      `png-sequence export: mask is ${rle.width}x${rle.height}, timeline is ${dims.width}x${dims.height}`,
    );
  }
  const mask = decodeRLE(rle);
  const imageData = ctx.createImageData(dims.width, dims.height);
  const pixels = imageData.data;
  for (let i = 0; i < mask.length; i++) {
    const value = mask[i] ? 255 : 0;
    const offset = i * 4;
    pixels[offset] = value;
    pixels[offset + 1] = value;
    pixels[offset + 2] = value;
    pixels[offset + 3] = 255; // opaque matte: value lives in RGB, not alpha
  }
  ctx.putImageData(imageData, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return new Uint8Array(await blob.arrayBuffer());
}
