import { describe, expect, it } from 'vitest';
import type { RLEMask } from '@websam3/core';
import { unzipSync } from 'fflate';
import { AlphaMatteExporter } from './exporter.js';
import { MaskTimeline } from './timeline.js';

// The video-editing package has a single (node) vitest project; this file
// self-skips there and only runs where the real PNG codec exists (a browser or
// worker lane with OffscreenCanvas + createImageBitmap). It exercises the REAL
// OffscreenCanvas.convertToBlob('image/png') path the node test stubs out.
const hasCanvas =
  typeof OffscreenCanvas !== 'undefined' &&
  typeof ImageData !== 'undefined' &&
  typeof createImageBitmap !== 'undefined';

function rle(width: number, height: number, counts: number[]): RLEMask {
  return { width, height, counts: Uint32Array.from(counts) };
}

/** Decode a PNG entry back into flat RGBA via createImageBitmap + a 2D canvas. */
async function pngToRgba(entry: Uint8Array): Promise<Uint8ClampedArray> {
  const bitmap = await createImageBitmap(new Blob([new Uint8Array(entry)], { type: 'image/png' }));
  // Capture dims before close(): closing an ImageBitmap resets width/height to 0.
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (ctx === null) throw new Error('no 2D context');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, width, height).data;
}

describe.skipIf(!hasCanvas)('AlphaMatteExporter — real PNG round-trip', () => {
  it('encodes a real PNG whose pixels decode back to the white-on-black matte', async () => {
    // 2x2 mask [1,0,0,1] → white, black, black, white (all opaque).
    const timeline = new MaskTimeline({ frameCount: 1, fps: 30, width: 2, height: 2 });
    timeline.set('a', 0, rle(2, 2, [0, 1, 2, 1]));

    const result = await new AlphaMatteExporter(timeline).export({
      mode: 'matte',
      format: 'png-sequence',
    });
    expect(result.format).toBe('png-sequence');
    expect(result.framesExported).toBe(1);

    const entries = unzipSync(new Uint8Array(await result.blob.arrayBuffer()));
    const entry = entries['frame-000000.png'];
    expect(entry).toBeDefined();
    // Real PNG signature.
    expect(Array.from(entry!.subarray(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

    const rgba = await pngToRgba(entry!);
    expect(Array.from(rgba)).toEqual([
      255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ]);
  });

  it('produces a multi-object zip whose entries all decode as PNGs', async () => {
    const timeline = new MaskTimeline({ frameCount: 2, fps: 30, width: 2, height: 2 });
    timeline.set('a', 0, rle(2, 2, [4]));
    timeline.set('b', 1, rle(2, 2, [0, 4]));

    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte' });
    const entries = unzipSync(new Uint8Array(await result.blob.arrayBuffer()));
    expect(Object.keys(entries).sort()).toEqual([
      'obj-a/frame-000000.png',
      'obj-b/frame-000001.png',
      'timeline.json',
    ]);
    // obj-b frame is all-white (mask [0,4] → every pixel set).
    const rgba = await pngToRgba(entries['obj-b/frame-000001.png']!);
    expect(Array.from(rgba)).toEqual([
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ]);
  });
});
