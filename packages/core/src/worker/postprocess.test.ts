import { describe, expect, it } from 'vitest';
import { computeTransform } from '../coords.js';
import { logitsToSourceMask } from './postprocess.js';

describe('logitsToSourceMask', () => {
  it('reduces to per-pixel thresholding when grid == model == source size', () => {
    // scale 1, gridScale 1: pixel center (x+0.5) maps to grid coord x exactly.
    const transform = computeTransform(4, 4, 4, 'square-stretch');
    const logits = Float32Array.from([
      2, 2, -3, -3,
      2, 2, -3, -3,
      -3, -3, 0, 5,
      -3, -3, 5, 5,
    ]);
    const mask = logitsToSourceMask(logits, 4, transform);
    expect([...mask]).toEqual([
      1, 1, 0, 0,
      1, 1, 0, 0,
      0, 0, 0, 1, // logit exactly 0 is NOT set (strict > 0)
      0, 0, 1, 1,
    ]);
  });

  it('bilinearly upsamples a low-res grid onto a non-square source (square-stretch)', () => {
    // 4x2 source, model square 4, grid 2. gridScale = 0.5.
    // gx = ((x+0.5)*1)*0.5 - 0.5 → [0 (clamped), 0.25, 0.75, 1 (clamped)]
    // gy = ((y+0.5)*2)*0.5 - 0.5 → [0, 1] exactly.
    const transform = computeTransform(4, 2, 4, 'square-stretch');
    const logits = Float32Array.from([
      -1, 1, // grid row 0
      1, -1, // grid row 1
    ]);
    const mask = logitsToSourceMask(logits, 2, transform);
    // Row 0 interpolates -1→1 across x: [-1, -0.5, 0.5, 1] → [0,0,1,1].
    // Row 1 interpolates 1→-1 across x: [1, 0.5, -0.5, -1] → [1,1,0,0].
    expect([...mask]).toEqual([0, 0, 1, 1, 1, 1, 0, 0]);
  });

  it('accounts for letterbox padding when sampling', () => {
    // 2x1 source into a 2-square letterbox: scale 1, padY = 0.5.
    // gy = ((0+0.5)*1 + 0.5)*1 - 0.5 = 0.5 → midway between grid rows.
    const transform = computeTransform(2, 1, 2, 'letterbox');
    const logits = Float32Array.from([
      3, -3, // grid row 0
      -1, 1, // grid row 1
    ]);
    const mask = logitsToSourceMask(logits, 2, transform);
    // x=0: gx clamps to 0 → (3 + -1)/2 = 1 → set.
    // x=1: gx clamps to 1 → (-3 + 1)/2 = -1 → unset.
    expect([...mask]).toEqual([1, 0]);
  });

  it('emits a row-major mask of exactly srcW*srcH bytes', () => {
    const transform = computeTransform(5, 3, 8, 'square-stretch');
    const logits = new Float32Array(4 * 4).fill(1);
    const mask = logitsToSourceMask(logits, 4, transform);
    expect(mask).toHaveLength(15);
    expect(mask.every((v) => v === 1)).toBe(true);
  });

  it('rejects a logits buffer that does not match the grid size', () => {
    const transform = computeTransform(2, 2, 2, 'square-stretch');
    expect(() => logitsToSourceMask(new Float32Array(3), 2, transform)).toThrowError(
      expect.objectContaining({ code: 'INVALID_STATE' }),
    );
  });
});
