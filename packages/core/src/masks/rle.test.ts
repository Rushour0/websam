import { describe, expect, it } from 'vitest';
import { InvalidStateError } from '../errors.js';
import { decodeRLE, encodeRLE, toCocoRLE } from './rle.js';

/** Build a row-major Uint8Array mask from rows of 0/1. */
function fromRows(rows: number[][]): { mask: Uint8Array; width: number; height: number } {
  const height = rows.length;
  const width = rows[0]?.length ?? 0;
  return { mask: Uint8Array.from(rows.flat()), width, height };
}

describe('encodeRLE / decodeRLE (row-major)', () => {
  it('starts counts with the zero-run (0 when first pixel is set)', () => {
    const { mask, width, height } = fromRows([
      [1, 0],
      [0, 1],
    ]);
    const rle = encodeRLE(mask, width, height);
    // Row-major sequence 1,0,0,1 → 0 zeros, 1 one, 2 zeros, 1 one.
    expect([...rle.counts]).toEqual([0, 1, 2, 1]);
    expect(rle.width).toBe(2);
    expect(rle.height).toBe(2);
  });

  it('encodes all-zero and all-one masks', () => {
    expect([...encodeRLE(new Uint8Array(6), 3, 2).counts]).toEqual([6]);
    expect([...encodeRLE(new Uint8Array(6).fill(1), 3, 2).counts]).toEqual([0, 6]);
    expect([...encodeRLE(new Uint8Array(0), 0, 0).counts]).toEqual([0]);
  });

  it('normalizes any non-zero byte to 1', () => {
    const rle = encodeRLE(Uint8Array.from([0, 255, 7, 0]), 4, 1);
    expect([...decodeRLE(rle)]).toEqual([0, 1, 1, 0]);
  });

  it('round-trips a deterministic pseudo-random non-square mask', () => {
    const width = 37;
    const height = 23;
    const mask = new Uint8Array(width * height);
    let state = 0x12345678;
    for (let i = 0; i < mask.length; i++) {
      // xorshift32 — deterministic, no test flakiness.
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      mask[i] = (state >>> 0) % 3 === 0 ? 1 : 0;
    }
    const rle = encodeRLE(mask, width, height);
    expect([...rle.counts].reduce((a, b) => a + b, 0)).toBe(width * height);
    expect(decodeRLE(rle)).toEqual(mask);
  });

  it('rejects a mask whose length does not match its dimensions', () => {
    expect(() => encodeRLE(new Uint8Array(5), 2, 2)).toThrow(InvalidStateError);
    expect(() => encodeRLE(new Uint8Array(4), -2, -2)).toThrow(RangeError);
  });

  it('rejects decoding when run lengths do not sum to width*height', () => {
    expect(() =>
      decodeRLE({ width: 2, height: 2, counts: Uint32Array.from([1, 1]) }),
    ).toThrow(InvalidStateError);
  });
});

describe('toCocoRLE (column-major + pycocotools string compression)', () => {
  /**
   * Known vectors generated with pycocotools:
   *   pycocotools.mask.encode(np.asfortranarray(rows, dtype=np.uint8))
   */
  const vectors: { rows: number[][]; counts: string }[] = [
    // 2x2 diagonal
    { rows: [[1, 0], [0, 1]], counts: '0120' },
    // 3x4 asymmetric blob — exercises negative deltas ('M' = -3)
    {
      rows: [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
      ],
      counts: '441M1',
    },
    // all zeros / all ones
    { rows: [[0, 0, 0], [0, 0, 0]], counts: '6' },
    { rows: [[1, 1, 1], [1, 1, 1]], counts: '06' },
    // 8x8 top half set — long alternating delta-zero tail
    {
      rows: [...Array.from({ length: 4 }, () => Array(8).fill(1)), ...Array.from({ length: 4 }, () => Array(8).fill(0))],
      counts: '04400000000000000',
    },
    // 1x40 all ones — run > 31 forces a LEB128 continuation char ('X')
    { rows: [Array(40).fill(1)], counts: '0X1' },
  ];

  it.each(vectors)('matches pycocotools for counts $counts', ({ rows, counts }) => {
    const { mask, width, height } = fromRows(rows);
    const coco = toCocoRLE(encodeRLE(mask, width, height));
    expect(coco.size).toEqual([height, width]);
    expect(coco.counts).toBe(counts);
  });

  it('size is [height, width] per COCO convention (not [w, h])', () => {
    const { mask, width, height } = fromRows([[1, 0, 0]]); // 1 row, 3 cols
    const coco = toCocoRLE(encodeRLE(mask, width, height));
    expect(coco.size).toEqual([1, 3]);
  });
});
