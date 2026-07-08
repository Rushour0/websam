/**
 * Run-length encoding for binary masks.
 *
 * Two formats live here:
 *
 * 1. websam's internal RLE ({@link RLEMask}) — ROW-major run lengths over a
 *    `width × height` binary mask, alternating zero-runs and one-runs and
 *    always starting with the zero-run count (which may be 0). Compact,
 *    O(runs) to composite, and cheap to produce from decoder output.
 * 2. COCO RLE ({@link toCocoRLE}) — the pycocotools interchange format:
 *    COLUMN-major (Fortran-order) run lengths, compressed to the LEB128-style
 *    ASCII string pycocotools emits. Provided for export/eval compatibility.
 */

import { InvalidStateError } from '../errors.js';

/** websam's internal row-major RLE mask. */
export interface RLEMask {
  /** Mask width in pixels. */
  width: number;
  /** Mask height in pixels. */
  height: number;
  /**
   * Alternating run lengths in ROW-major pixel order, starting with the
   * count of leading zeros (0 if the first pixel is 1). Sums to width*height.
   */
  counts: Uint32Array;
}

/** COCO-format compressed RLE, byte-compatible with pycocotools `encode`. */
export interface CocoRLE {
  /** `[height, width]` — COCO convention. */
  size: [number, number];
  /** pycocotools-compressed column-major run lengths (ASCII string). */
  counts: string;
}

/**
 * Encode a binary mask into websam's row-major RLE.
 *
 * @param mask - Row-major binary mask; any non-zero byte counts as 1.
 * Length must be exactly `width * height`.
 */
export function encodeRLE(mask: Uint8Array, width: number, height: number): RLEMask {
  if (!Number.isInteger(width) || width < 0 || !Number.isInteger(height) || height < 0) {
    throw new RangeError(`encodeRLE: invalid dimensions ${width}x${height}`);
  }
  if (mask.length !== width * height) {
    throw new InvalidStateError(
      `encodeRLE: mask length ${mask.length} does not match ${width}x${height}=${width * height}`,
    );
  }
  const counts: number[] = [];
  let current = 0; // runs always start by counting zeros
  let run = 0;
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 1 : 0;
    if (v === current) {
      run++;
    } else {
      counts.push(run);
      current = v;
      run = 1;
    }
  }
  counts.push(run); // final run (or the sole 0-run for an empty mask)
  return { width, height, counts: Uint32Array.from(counts) };
}

/**
 * Decode websam RLE back into a row-major binary mask of 0/1 bytes.
 * Exact inverse of {@link encodeRLE} (up to non-zero → 1 normalization).
 */
export function decodeRLE(rle: RLEMask): Uint8Array {
  const total = rle.width * rle.height;
  const mask = new Uint8Array(total);
  let pos = 0;
  let value = 0; // first run counts zeros
  for (let i = 0; i < rle.counts.length; i++) {
    const run = rle.counts[i] ?? 0;
    if (value === 1) {
      mask.fill(1, pos, pos + run);
    }
    pos += run;
    value ^= 1;
  }
  if (pos !== total) {
    throw new InvalidStateError(
      `decodeRLE: run lengths sum to ${pos}, expected ${rle.width}x${rle.height}=${total}`,
    );
  }
  return mask;
}

/**
 * Convert websam RLE to COCO compressed RLE, byte-compatible with
 * pycocotools: run lengths are recomputed in COLUMN-major (Fortran) order —
 * pixel index `x * height + y` — starting with the zero-run, then compressed
 * with pycocotools' LEB128-style scheme (5 value bits per ASCII char offset
 * by 48, continuation bit 0x20, and counts after the first two delta-encoded
 * against `counts[i-2]`).
 */
export function toCocoRLE(rle: RLEMask): CocoRLE {
  const { width, height } = rle;
  const mask = decodeRLE(rle);

  // Column-major run lengths, starting with the count of zeros.
  const counts: number[] = [];
  let current = 0;
  let run = 0;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      // Iterate pixels in Fortran order; `mask` itself is row-major.
      const v = mask[y * width + x] ? 1 : 0;
      if (v === current) {
        run++;
      } else {
        counts.push(run);
        current = v;
        run = 1;
      }
    }
  }
  counts.push(run);

  return { size: [height, width], counts: compressCocoCounts(counts) };
}

/**
 * pycocotools `rleToString`: each count (delta-encoded against the count two
 * places back, from the third onward) is emitted in 5-bit groups, low bits
 * first, as ASCII `chr(48 + group)`, with 0x20 marking continuation. Signed
 * arithmetic (JS `>>` is an arithmetic shift) exactly matches the C code.
 */
function compressCocoCounts(counts: readonly number[]): string {
  let s = '';
  for (let i = 0; i < counts.length; i++) {
    let x = counts[i] ?? 0;
    if (i > 2) x -= counts[i - 2] ?? 0;
    let more = true;
    while (more) {
      let c = x & 0x1f;
      x >>= 5;
      more = (c & 0x10) !== 0 ? x !== -1 : x !== 0;
      if (more) c |= 0x20;
      s += String.fromCharCode(c + 48);
    }
  }
  return s;
}
