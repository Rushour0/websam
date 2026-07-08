import { describe, expect, it } from 'vitest';
import { computeTransform } from '../coords.js';
import { buildPromptTensors, bitmapToTensor, rgbaToChw, zeroPadRegions } from './preprocess.js';

describe('rgbaToChw', () => {
  // Synthetic 2x2 image: pure red, pure green, pure blue, and a mixed pixel.
  const rgba = Uint8ClampedArray.from([
    255, 0, 0, 255, //   (0,0) red
    0, 255, 0, 255, //   (1,0) green
    0, 0, 255, 255, //   (0,1) blue
    128, 64, 32, 255, // (1,1) mixed
  ]);
  const mean: [number, number, number] = [0.5, 0.5, 0.5];
  const std: [number, number, number] = [0.5, 0.5, 0.5];

  it('normalizes with (v/255 - mean) / std into [-1, 1]', () => {
    const chw = rgbaToChw(rgba, 2, mean, std);
    // Red channel, pixel (0,0): (255/255 - 0.5) / 0.5 = 1.
    expect(chw[0]).toBeCloseTo(1, 6);
    // Red channel, pixel (1,1): (128/255 - 0.5) / 0.5.
    expect(chw[3]).toBeCloseTo((128 / 255 - 0.5) / 0.5, 6);
    // Green channel, pixel (1,1): (64/255 - 0.5) / 0.5.
    expect(chw[7]).toBeCloseTo((64 / 255 - 0.5) / 0.5, 6);
    // Blue channel, pixel (1,1): (32/255 - 0.5) / 0.5.
    expect(chw[11]).toBeCloseTo((32 / 255 - 0.5) / 0.5, 6);
  });

  it('lays planes out channels-first: out[c*4 + y*2 + x]', () => {
    const chw = rgbaToChw(rgba, 2, mean, std);
    expect(chw).toHaveLength(12);
    // R plane: red pixel is 1, green/blue pixels are -1.
    expect([...chw.slice(0, 4)].map((v) => Math.sign(v))).toEqual([1, -1, -1, 1 /* 128 > 127.5 */]);
    // G plane: only the green pixel is positive-one; mixed (64) is negative.
    expect(chw[4]).toBeCloseTo(-1, 6);
    expect(chw[5]).toBeCloseTo(1, 6);
    expect(chw[6]).toBeCloseTo(-1, 6);
    // B plane: only the blue pixel is positive-one.
    expect(chw[8]).toBeCloseTo(-1, 6);
    expect(chw[9]).toBeCloseTo(-1, 6);
    expect(chw[10]).toBeCloseTo(1, 6);
  });

  it('ignores alpha', () => {
    const transparent = Uint8ClampedArray.from(rgba);
    transparent[3] = 0;
    transparent[7] = 0;
    expect(rgbaToChw(transparent, 2, mean, std)).toEqual(rgbaToChw(rgba, 2, mean, std));
  });

  it('applies per-channel mean/std independently', () => {
    const chw = rgbaToChw(rgba, 2, [0, 0.5, 1], [1, 0.5, 0.25]);
    // (0,0) red pixel: R=(1-0)/1, G=(0-0.5)/0.5, B=(0-1)/0.25.
    expect(chw[0]).toBeCloseTo(1, 6);
    expect(chw[4]).toBeCloseTo(-1, 6);
    expect(chw[8]).toBeCloseTo(-4, 6);
  });

  it('rejects a byte length that does not match the size', () => {
    expect(() => rgbaToChw(rgba, 3, mean, std)).toThrowError(
      expect.objectContaining({ code: 'INVALID_STATE' }),
    );
  });
});

describe('zeroPadRegions', () => {
  it('zeroes only the letterbox bands, in every channel', () => {
    // 4x2 source into a 4x4 square: scale 1, padY = 1 (rows 0 and 3 are pad).
    const transform = computeTransform(4, 2, 4, 'letterbox');
    expect(transform.padY).toBe(1);
    const chw = new Float32Array(3 * 16).fill(0.5);
    zeroPadRegions(chw, transform);
    for (let c = 0; c < 3; c++) {
      const plane = c * 16;
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          const expected = y === 0 || y === 3 ? 0 : 0.5;
          expect(chw[plane + y * 4 + x]).toBe(expected);
        }
      }
    }
  });

  it('is a no-op for square-stretch transforms', () => {
    const transform = computeTransform(4, 2, 4, 'square-stretch');
    const chw = new Float32Array(3 * 16).fill(-1);
    zeroPadRegions(chw, transform);
    expect(chw.every((v) => v === -1)).toBe(true);
  });
});

describe('bitmapToTensor', () => {
  it('throws InvalidStateError where OffscreenCanvas does not exist (node)', () => {
    const transform = computeTransform(4, 2, 4, 'square-stretch');
    const preprocess: import('./preprocess.js').PreprocessSpec = {
      mode: 'square-stretch',
      inputSize: 4,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      maskSize: 2,
    };
    expect(() =>
      bitmapToTensor({ width: 4, height: 2 } as unknown as ImageBitmap, transform, preprocess),
    ).toThrowError(expect.objectContaining({ code: 'INVALID_STATE' }));
  });
});

describe('buildPromptTensors', () => {
  // Non-square on purpose (coordinate contract): 4x2 source into an 8-square
  // stretches anisotropically — scaleX = 2, scaleY = 4.
  const transform = computeTransform(4, 2, 8, 'square-stretch');

  it('maps points through sourceToModel and packs [1,1,N,2] f32 + [1,1,N] i64', () => {
    const tensors = buildPromptTensors(
      [
        { type: 'point', x: 1, y: 1, label: 1 },
        { type: 'point', x: 2, y: 0.5, label: 0 },
      ],
      transform,
    );
    expect(tensors.pointCount).toBe(2);
    expect([...tensors.points]).toEqual([2, 4, 4, 2]);
    expect(tensors.labels).toBeInstanceOf(BigInt64Array);
    expect([...tensors.labels]).toEqual([1n, 0n]);
    expect(tensors.boxCount).toBe(0);
    expect(tensors.boxes).toHaveLength(0);
  });

  it('maps box corners independently per axis into [1,Nb,4]', () => {
    const tensors = buildPromptTensors([{ type: 'box', x1: 0, y1: 0, x2: 2, y2: 1 }], transform);
    expect(tensors.boxCount).toBe(1);
    expect([...tensors.boxes]).toEqual([0, 0, 4, 4]);
    expect(tensors.pointCount).toBe(0);
    expect(tensors.points).toHaveLength(0);
    expect(tensors.labels).toHaveLength(0);
  });

  it('applies letterbox padding offsets to prompt coordinates', () => {
    // 4x2 into a 4-square letterbox: scale 1, padY 1.
    const letterbox = computeTransform(4, 2, 4, 'letterbox');
    const tensors = buildPromptTensors([{ type: 'point', x: 1, y: 1, label: 1 }], letterbox);
    expect([...tensors.points]).toEqual([1, 2]);
  });

  it('produces the empty [1,0,4] boxes feed when no box prompt exists', () => {
    const tensors = buildPromptTensors([{ type: 'point', x: 0, y: 0, label: 1 }], transform);
    expect(tensors.boxCount).toBe(0);
    expect(tensors.boxes).toBeInstanceOf(Float32Array);
    expect(tensors.boxes).toHaveLength(0);
  });

  it('rejects mask prompts with NotImplementedError (lands in M2)', () => {
    expect(() =>
      buildPromptTensors(
        [{ type: 'mask', mask: { width: 2, height: 2, counts: Uint32Array.from([4]) } }],
        transform,
      ),
    ).toThrowError(expect.objectContaining({ code: 'NOT_IMPLEMENTED' }));
  });
});
