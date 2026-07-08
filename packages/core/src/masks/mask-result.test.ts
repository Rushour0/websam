import { afterEach, describe, expect, it, vi } from 'vitest';
import { computeTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';
import { MaskResultImpl, type MaskResultInit } from './mask-result.js';
import { decodeRLE, encodeRLE, toCocoRLE } from './rle.js';

/** Build a MaskResultInit around a row-major 0/1 mask. */
function makeInit(
  rows: number[][],
  overrides: Partial<MaskResultInit> = {},
): MaskResultInit {
  const height = rows.length;
  const width = rows[0]?.length ?? 0;
  return {
    objectId: 7,
    score: 0.93,
    width,
    height,
    binaryMask: Uint8Array.from(rows.flat()),
    transform: computeTransform(width || 1, height || 1, 1008, 'square-stretch'),
    ...overrides,
  };
}

const DIAGONAL = [
  [1, 0],
  [0, 1],
];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('MaskResultImpl construction', () => {
  it('exposes objectId, score, dimensions, and the transform', () => {
    const init = makeInit(DIAGONAL);
    const result = new MaskResultImpl(init);
    expect(result.objectId).toBe(7);
    expect(result.score).toBe(0.93);
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    // Coordinate-contract rule 4: the exact transform rides along.
    expect(result.transform).toBe(init.transform);
  });

  it('copies binaryMask: mutating the caller buffer after construction changes nothing', () => {
    const init = makeInit(DIAGONAL);
    const result = new MaskResultImpl(init);
    init.binaryMask.fill(1);
    expect([...result.toBinary()]).toEqual([1, 0, 0, 1]);
    expect([...result.toRLE().counts]).toEqual([0, 1, 2, 1]);
  });

  it('rejects a mask whose length does not match width*height', () => {
    expect(
      () => new MaskResultImpl(makeInit(DIAGONAL, { binaryMask: new Uint8Array(5) })),
    ).toThrow(InvalidStateError);
  });

  it('rejects non-integer or negative dimensions', () => {
    expect(() => new MaskResultImpl(makeInit(DIAGONAL, { width: -2 }))).toThrow(RangeError);
    expect(() => new MaskResultImpl(makeInit(DIAGONAL, { height: 1.5 }))).toThrow(RangeError);
  });

  it('accepts an empty 0x0 mask', () => {
    const result = new MaskResultImpl(makeInit([]));
    expect(result.toBinary().length).toBe(0);
    expect([...result.toRLE().counts]).toEqual([0]);
  });
});

describe('toBinary', () => {
  it('returns a fresh defensive copy every call', () => {
    const result = new MaskResultImpl(makeInit(DIAGONAL));
    const first = result.toBinary();
    first.fill(0);
    const second = result.toBinary();
    expect(second).not.toBe(first);
    expect([...second]).toEqual([1, 0, 0, 1]);
  });
});

describe('toRLE / toCocoRLE', () => {
  it('round-trips through decodeRLE back to the binary mask', () => {
    // Deterministic pseudo-random non-square mask (xorshift32).
    const width = 37;
    const height = 23;
    const mask = new Uint8Array(width * height);
    let state = 0x9e3779b9;
    for (let i = 0; i < mask.length; i++) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      mask[i] = (state >>> 0) % 3 === 0 ? 1 : 0;
    }
    const result = new MaskResultImpl(
      makeInit([], {
        width,
        height,
        binaryMask: mask,
        transform: computeTransform(width, height, 1008, 'square-stretch'),
      }),
    );
    expect(decodeRLE(result.toRLE())).toEqual(mask);
    expect(decodeRLE(result.toRLE())).toEqual(result.toBinary());
  });

  it('memoizes toRLE: repeated calls return the same instance', () => {
    const result = new MaskResultImpl(makeInit(DIAGONAL));
    const first = result.toRLE();
    expect(result.toRLE()).toBe(first);
    expect(result.toRLE()).toBe(first);
  });

  it('matches encodeRLE/toCocoRLE on the same bytes (pycocotools vector)', () => {
    const init = makeInit(DIAGONAL);
    const result = new MaskResultImpl(init);
    const reference = encodeRLE(Uint8Array.from(DIAGONAL.flat()), 2, 2);
    expect(result.toRLE()).toEqual(reference);
    // Known pycocotools vector for the 2x2 diagonal (see rle.test.ts).
    const coco = result.toCocoRLE();
    expect(coco).toEqual(toCocoRLE(reference));
    expect(coco.size).toEqual([2, 2]);
    expect(coco.counts).toBe('0120');
  });
});

describe('DOM-dependent paths', () => {
  it('toImageData throws InvalidStateError when ImageData is absent (node)', () => {
    expect(typeof ImageData).toBe('undefined'); // precondition: bare node
    const result = new MaskResultImpl(makeInit(DIAGONAL));
    expect(() => result.toImageData()).toThrow(InvalidStateError);
    expect(() => result.toImageData()).toThrow(/ImageData/);
  });

  it('toBitmap rejects with InvalidStateError when createImageBitmap is absent (node)', async () => {
    expect(typeof createImageBitmap).toBe('undefined'); // precondition: bare node
    const result = new MaskResultImpl(makeInit(DIAGONAL));
    await expect(result.toBitmap()).rejects.toThrow(InvalidStateError);
    await expect(result.toBitmap()).rejects.toThrow(/createImageBitmap/);
  });

  it('toImageData emits opaque-white RGBA for set pixels, transparent black otherwise', () => {
    // Minimal stand-in so the pure pixel-fill path runs in node; the real
    // constructor is exercised by the wave-3 browser e2e.
    class FakeImageData {
      constructor(
        readonly data: Uint8ClampedArray,
        readonly width: number,
        readonly height: number,
      ) {}
    }
    vi.stubGlobal('ImageData', FakeImageData);

    const result = new MaskResultImpl(makeInit(DIAGONAL));
    const image = result.toImageData() as unknown as FakeImageData;
    expect(image.width).toBe(2);
    expect(image.height).toBe(2);
    expect([...image.data]).toEqual([
      255, 255, 255, 255, /**/ 0, 0, 0, 0,
      0, 0, 0, 0, /*     */ 255, 255, 255, 255,
    ]);
  });

  it('toImageData returns a fresh instance every call (no memoization of mutable output)', () => {
    class FakeImageData {
      constructor(
        readonly data: Uint8ClampedArray,
        readonly width: number,
        readonly height: number,
      ) {}
    }
    vi.stubGlobal('ImageData', FakeImageData);

    const result = new MaskResultImpl(makeInit(DIAGONAL));
    const first = result.toImageData();
    const second = result.toImageData();
    expect(second).not.toBe(first);
    expect(second.data).not.toBe(first.data);
  });

  it('toBitmap forwards this.toImageData() through createImageBitmap', async () => {
    class FakeImageData {
      constructor(
        readonly data: Uint8ClampedArray,
        readonly width: number,
        readonly height: number,
      ) {}
    }
    const bitmap = { close: () => {} };
    const createImageBitmapMock = vi.fn(async (_image: unknown) => bitmap);
    vi.stubGlobal('ImageData', FakeImageData);
    vi.stubGlobal('createImageBitmap', createImageBitmapMock);

    const result = new MaskResultImpl(makeInit(DIAGONAL));
    await expect(result.toBitmap()).resolves.toBe(bitmap);
    expect(createImageBitmapMock).toHaveBeenCalledTimes(1);
    expect(createImageBitmapMock.mock.calls[0]?.[0]).toBeInstanceOf(FakeImageData);
  });
});
