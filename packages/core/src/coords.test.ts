import { describe, expect, it } from 'vitest';
import { computeTransform, modelToSource, sourceToModel, type Point } from './coords.js';

const SRC_W = 1280;
const SRC_H = 720;
const MODEL = 1008;

describe('computeTransform', () => {
  it('square-stretch scales each axis independently with zero padding', () => {
    const t = computeTransform(SRC_W, SRC_H, MODEL, 'square-stretch');
    expect(t).toMatchObject({ srcW: SRC_W, srcH: SRC_H, modelSize: MODEL, mode: 'square-stretch' });
    expect(t.scaleX).toBeCloseTo(MODEL / SRC_W, 12);
    expect(t.scaleY).toBeCloseTo(MODEL / SRC_H, 12);
    expect(t.padX).toBe(0);
    expect(t.padY).toBe(0);
    // Non-square input MUST produce anisotropic scales in this mode.
    expect(t.scaleX).not.toBeCloseTo(t.scaleY, 6);
  });

  it('letterbox uses one uniform scale and centers the short axis', () => {
    const t = computeTransform(SRC_W, SRC_H, MODEL, 'letterbox');
    const scale = MODEL / SRC_W; // width is the long side
    expect(t.scaleX).toBeCloseTo(scale, 12);
    expect(t.scaleY).toBeCloseTo(scale, 12);
    expect(t.padX).toBe(0); // long side fills the square exactly
    expect(t.padY).toBeCloseTo((MODEL - SRC_H * scale) / 2, 12);
    // Content is centered: pad + content + pad === modelSize.
    expect(2 * t.padY + SRC_H * scale).toBeCloseTo(MODEL, 9);
  });

  it('letterbox pads X for portrait images', () => {
    const t = computeTransform(SRC_H, SRC_W, MODEL, 'letterbox'); // 720x1280 portrait
    expect(t.padY).toBe(0);
    expect(t.padX).toBeGreaterThan(0);
    expect(2 * t.padX + SRC_H * t.scaleX).toBeCloseTo(MODEL, 9);
  });

  it('square input makes both modes identical (why golden tests must be non-square)', () => {
    const stretch = computeTransform(512, 512, MODEL, 'square-stretch');
    const letter = computeTransform(512, 512, MODEL, 'letterbox');
    expect(stretch.scaleX).toBeCloseTo(letter.scaleX, 12);
    expect(stretch.scaleY).toBeCloseTo(letter.scaleY, 12);
    expect(letter.padX).toBe(0);
    expect(letter.padY).toBe(0);
  });

  it('rejects non-positive dimensions', () => {
    expect(() => computeTransform(0, 720, MODEL, 'letterbox')).toThrow(RangeError);
    expect(() => computeTransform(1280, -1, MODEL, 'letterbox')).toThrow(RangeError);
    expect(() => computeTransform(1280, 720, 0, 'square-stretch')).toThrow(RangeError);
    expect(() => computeTransform(Number.NaN, 720, MODEL, 'letterbox')).toThrow(RangeError);
  });
});

describe('sourceToModel / modelToSource', () => {
  const points: Point[] = [
    { x: 0, y: 0 }, // top-left corner
    { x: SRC_W - 1, y: SRC_H - 1 }, // last pixel
    { x: SRC_W, y: SRC_H }, // exclusive edge
    { x: SRC_W / 2, y: SRC_H / 2 }, // center
    { x: 1, y: 719 }, // near-edge asymmetric
    { x: 640.25, y: 359.75 }, // sub-pixel
  ];

  for (const mode of ['square-stretch', 'letterbox'] as const) {
    it(`${mode}: round-trips non-square points exactly`, () => {
      const t = computeTransform(SRC_W, SRC_H, MODEL, mode);
      for (const p of points) {
        const back = modelToSource(sourceToModel(p, t), t);
        expect(back.x).toBeCloseTo(p.x, 9);
        expect(back.y).toBeCloseTo(p.y, 9);
      }
    });
  }

  it('square-stretch maps the source extent to the full model square', () => {
    const t = computeTransform(SRC_W, SRC_H, MODEL, 'square-stretch');
    expect(sourceToModel({ x: 0, y: 0 }, t)).toEqual({ x: 0, y: 0 });
    const edge = sourceToModel({ x: SRC_W, y: SRC_H }, t);
    expect(edge.x).toBeCloseTo(MODEL, 9);
    expect(edge.y).toBeCloseTo(MODEL, 9);
  });

  it('letterbox maps the source extent inside the padded band', () => {
    const t = computeTransform(SRC_W, SRC_H, MODEL, 'letterbox');
    const topLeft = sourceToModel({ x: 0, y: 0 }, t);
    expect(topLeft.x).toBeCloseTo(0, 9);
    expect(topLeft.y).toBeCloseTo(t.padY, 9);
    const bottomRight = sourceToModel({ x: SRC_W, y: SRC_H }, t);
    expect(bottomRight.x).toBeCloseTo(MODEL, 9);
    expect(bottomRight.y).toBeCloseTo(MODEL - t.padY, 9);
    // Vertical center of the content is the center of the model square.
    const center = sourceToModel({ x: SRC_W / 2, y: SRC_H / 2 }, t);
    expect(center.y).toBeCloseTo(MODEL / 2, 9);
  });

  it('the two modes disagree on non-square images (the M1-S0 pin matters)', () => {
    const p = { x: 100, y: 100 };
    const a = sourceToModel(p, computeTransform(SRC_W, SRC_H, MODEL, 'square-stretch'));
    const b = sourceToModel(p, computeTransform(SRC_W, SRC_H, MODEL, 'letterbox'));
    expect(a.y).not.toBeCloseTo(b.y, 3);
  });
});
