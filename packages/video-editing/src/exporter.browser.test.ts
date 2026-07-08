import { describe, expect, it } from 'vitest';
import { unzipSync } from 'fflate';
import { decodeRLE, encodeRLE } from '@websam/core';
import type { RLEMask } from '@websam/core';
import { AlphaMatteExporter } from './exporter.js';
import { MaskTimeline } from './timeline.js';

// Non-square on purpose: a row/column-major mixup would scramble the pixels.
const WIDTH = 8;
const HEIGHT = 6;

/** A deterministic non-trivial binary mask: a filled rectangle plus a stripe. */
function patternMask(): { rle: RLEMask; bits: Uint8Array } {
  const bits = new Uint8Array(WIDTH * HEIGHT);
  for (let y = 1; y < 4; y++) {
    for (let x = 2; x < 7; x++) bits[y * WIDTH + x] = 1;
  }
  for (let x = 0; x < WIDTH; x++) bits[5 * WIDTH + x] = x % 2;
  return { rle: encodeRLE(bits, WIDTH, HEIGHT), bits };
}

/** Decode one PNG zip entry back to RGBA bytes via the real browser codec. */
async function decodePngEntry(entry: Uint8Array): Promise<Uint8ClampedArray> {
  const bitmap = await createImageBitmap(
    new Blob([entry as Uint8Array<ArrayBuffer>], { type: 'image/png' }),
  );
  expect(bitmap.width).toBe(WIDTH);
  expect(bitmap.height).toBe(HEIGHT);
  const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
  const ctx = canvas.getContext('2d');
  if (ctx === null) throw new Error('2d context unavailable');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, WIDTH, HEIGHT).data;
}

/** The white-on-black opaque RGBA a matte PNG must decode to for `rle`. */
function expectedRgba(rle: RLEMask): Uint8ClampedArray {
  const bits = decodeRLE(rle);
  const rgba = new Uint8ClampedArray(bits.length * 4);
  for (let i = 0; i < bits.length; i++) {
    const value = bits[i] ? 255 : 0;
    rgba[i * 4] = value;
    rgba[i * 4 + 1] = value;
    rgba[i * 4 + 2] = value;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

describe('AlphaMatteExporter png-sequence (real browser PNG codec)', () => {
  it('exports PNGs whose decoded pixels round-trip the mask exactly', async () => {
    const { rle } = patternMask();
    const timeline = new MaskTimeline({ frameCount: 3, fps: 30, width: WIDTH, height: HEIGHT });
    timeline.set('1', 0, rle);
    timeline.set('1', 2, rle); // hole at frame 1

    const result = await new AlphaMatteExporter(timeline).export({
      mode: 'matte',
      format: 'png-sequence',
    });
    expect(result.format).toBe('png-sequence');
    expect(result.suggestedFileName).toBe('matte.zip');
    expect(result.framesExported).toBe(2);

    const entries = unzipSync(new Uint8Array(await result.blob.arrayBuffer()));
    expect(Object.keys(entries).sort()).toEqual([
      'frame-000000.png',
      'frame-000002.png',
      'timeline.json',
    ]);

    const png = entries['frame-000000.png'];
    if (png === undefined) throw new Error('frame-000000.png missing from zip');
    const rgba = await decodePngEntry(png);
    // PNG is lossless and the matte is opaque, so the round-trip is exact.
    expect(Array.from(rgba)).toEqual(Array.from(expectedRgba(rle)));
  });

  it('nests per-object folders and a parseable timeline.json for multi-object timelines', async () => {
    const { rle } = patternMask();
    const timeline = new MaskTimeline({ frameCount: 2, fps: 24, width: WIDTH, height: HEIGHT });
    timeline.set('1', 0, rle);
    timeline.set('2', 1, rle);

    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte' });
    const entries = unzipSync(new Uint8Array(await result.blob.arrayBuffer()));
    expect(Object.keys(entries).sort()).toEqual([
      'obj-1/frame-000000.png',
      'obj-2/frame-000001.png',
      'timeline.json',
    ]);

    const sidecarBytes = entries['timeline.json'];
    if (sidecarBytes === undefined) throw new Error('timeline.json missing from zip');
    const sidecar = JSON.parse(new TextDecoder().decode(sidecarBytes)) as {
      fps: number;
      frameCount: number;
      width: number;
      height: number;
      objects: Record<string, number[]>;
    };
    expect(sidecar).toEqual({
      fps: 24,
      frameCount: 2,
      width: WIDTH,
      height: HEIGHT,
      objects: { '1': [0], '2': [1] },
    });
  });
});
