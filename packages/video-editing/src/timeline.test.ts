import { describe, expect, it } from 'vitest';
import type { RLEMask } from '@websam/core';
import { MaskTimeline } from './timeline.js';

const WIDTH = 4;
const HEIGHT = 4;

/** Build a tiny 4x4 RLE mask from raw counts. */
function mask(counts: number[]): RLEMask {
  return { width: WIDTH, height: HEIGHT, counts: Uint32Array.from(counts) };
}

function makeTimeline(frameCount = 6): MaskTimeline {
  return new MaskTimeline({ frameCount, fps: 30, width: WIDTH, height: HEIGHT });
}

describe('MaskTimeline set/get', () => {
  it('round-trips a mask and returns undefined for unset frames', () => {
    const t = makeTimeline();
    const m = mask([3, 5, 8]);
    expect(t.set('obj-1', 2, m)).toBe(true);
    expect(t.get('obj-1', 2)).toBe(m);
    expect(t.get('obj-1', 3)).toBeUndefined();
    expect(t.get('nobody', 2)).toBeUndefined();
  });

  it('overwrites an existing frame in place', () => {
    const t = makeTimeline();
    t.set('a', 1, mask([16]));
    const replacement = mask([0, 16]);
    t.set('a', 1, replacement);
    expect(t.get('a', 1)).toBe(replacement);
  });

  it('rejects out-of-range frame indices', () => {
    const t = makeTimeline(6);
    expect(() => t.set('a', 6, mask([16]))).toThrow(RangeError);
    expect(() => t.set('a', -1, mask([16]))).toThrow(RangeError);
    expect(() => t.get('a', 1.5)).toThrow(RangeError);
  });

  it('getAll returns only the objects with a mask at that frame', () => {
    const t = makeTimeline();
    const a2 = mask([1, 15]);
    const b2 = mask([2, 14]);
    t.set('a', 2, a2);
    t.set('b', 2, b2);
    t.set('b', 4, mask([16]));

    const at2 = t.getAll(2);
    expect(at2.size).toBe(2);
    expect(at2.get('a')).toBe(a2);
    expect(at2.get('b')).toBe(b2);
    expect(t.getAll(0).size).toBe(0);
  });

  it('tracks object ids across writes', () => {
    const t = makeTimeline();
    t.set('a', 0, mask([16]));
    t.set('b', 1, mask([16]));
    expect(t.objectIds().sort()).toEqual(['a', 'b']);
  });
});

describe('MaskTimeline epoch invalidation', () => {
  it('invalidateAfter drops strictly later frames and keeps earlier ones', () => {
    const t = makeTimeline(6);
    for (const i of [0, 1, 2, 3, 4, 5]) t.set('a', i, mask([i, 16 - i]));

    const newEpoch = t.invalidateAfter('a', 2);
    expect(newEpoch).toBe(1);
    expect(t.epoch('a')).toBe(1);
    expect(t.get('a', 0)).toBeDefined();
    expect(t.get('a', 2)).toBeDefined();
    expect(t.get('a', 3)).toBeUndefined();
    expect(t.get('a', 5)).toBeUndefined();
  });

  it('rejects stale writes from the previous epoch and accepts current-epoch writes', () => {
    const t = makeTimeline(6);
    t.set('a', 0, mask([16]), 0);
    const epoch = t.invalidateAfter('a', 0); // epoch 0 -> 1

    // Straggler propagation result still carrying epoch 0: must be dropped.
    expect(t.set('a', 3, mask([4, 12]), 0)).toBe(false);
    expect(t.get('a', 3)).toBeUndefined();

    // Re-propagation under the new epoch: accepted.
    const fresh = mask([8, 8]);
    expect(t.set('a', 3, fresh, epoch)).toBe(true);
    expect(t.get('a', 3)).toBe(fresh);
  });

  it('a write with a newer epoch advances the current epoch', () => {
    const t = makeTimeline(6);
    t.set('a', 0, mask([16]), 5);
    expect(t.epoch('a')).toBe(5);
    expect(t.set('a', 1, mask([16]), 4)).toBe(false);
  });

  it('epochs are independent per object', () => {
    const t = makeTimeline(6);
    t.set('a', 1, mask([16]));
    t.set('b', 1, mask([16]));
    t.invalidateAfter('a', 0);
    expect(t.epoch('a')).toBe(1);
    expect(t.epoch('b')).toBe(0);
    expect(t.get('b', 1)).toBeDefined();
  });
});

describe('MaskTimeline holes', () => {
  it('finds unmasked frames across the whole timeline', () => {
    const t = makeTimeline(6);
    t.set('a', 0, mask([16]));
    t.set('a', 1, mask([16]));
    t.set('a', 4, mask([16]));
    expect(t.holes('a')).toEqual([2, 3, 5]);
  });

  it('respects a half-open range', () => {
    const t = makeTimeline(6);
    t.set('a', 0, mask([16]));
    t.set('a', 4, mask([16]));
    expect(t.holes('a', { start: 0, end: 5 })).toEqual([1, 2, 3]);
    expect(t.holes('a', { start: 4 })).toEqual([5]);
  });

  it('treats an unknown object as fully unmasked and validates the range', () => {
    const t = makeTimeline(3);
    expect(t.holes('ghost')).toEqual([0, 1, 2]);
    expect(() => t.holes('ghost', { end: 4 })).toThrow(RangeError);
    expect(() => t.holes('ghost', { start: 2, end: 1 })).toThrow(RangeError);
  });

  it('reports no holes once every frame is covered', () => {
    const t = makeTimeline(3);
    for (const i of [0, 1, 2]) t.set('a', i, mask([16]));
    expect(t.holes('a')).toEqual([]);
  });
});

describe('MaskTimeline JSON round-trip', () => {
  it('serializes counts as base64 strings and survives JSON.stringify', () => {
    const t = makeTimeline(6);
    t.set('a', 2, mask([3, 5, 8]));
    const json = t.toJSON();
    const serialized = json.objects['a']?.frames['2'];
    expect(typeof serialized?.counts).toBe('string');
    // Round-trips through real JSON text, not just structured clone.
    expect(() => JSON.parse(JSON.stringify(json))).not.toThrow();
  });

  it('restores dimensions, masks, and epochs exactly', () => {
    const t = makeTimeline(6);
    t.set('a', 0, mask([16]));
    t.set('a', 3, mask([3, 5, 8]));
    t.set('b', 5, mask([0, 1, 15]));
    t.invalidateAfter('b', 5); // epoch 1, no frames dropped (5 is the last frame)

    const revived = MaskTimeline.fromJSON(
      JSON.parse(JSON.stringify(t.toJSON())) as ReturnType<MaskTimeline['toJSON']>,
    );

    expect(revived.frameCount).toBe(6);
    expect(revived.fps).toBe(30);
    expect(revived.width).toBe(WIDTH);
    expect(revived.height).toBe(HEIGHT);
    expect(revived.objectIds().sort()).toEqual(['a', 'b']);
    expect(revived.get('a', 3)).toEqual(mask([3, 5, 8]));
    expect(revived.get('b', 5)).toEqual(mask([0, 1, 15]));
    expect(revived.epoch('a')).toBe(0);
    expect(revived.epoch('b')).toBe(1);
    expect(revived.holes('a')).toEqual(t.holes('a'));
  });

  it('round-trips large run counts through base64 without loss', () => {
    const t = makeTimeline(2);
    const big = mask([0, 70_000_000, 1, 4_294_967_295]);
    t.set('a', 0, big);
    const revived = MaskTimeline.fromJSON(t.toJSON());
    expect(revived.get('a', 0)).toEqual(big);
  });

  it('stale-write rejection still holds after revival', () => {
    const t = makeTimeline(4);
    t.set('a', 0, mask([16]));
    t.invalidateAfter('a', 0);
    const revived = MaskTimeline.fromJSON(t.toJSON());
    expect(revived.set('a', 1, mask([16]), 0)).toBe(false);
    expect(revived.set('a', 1, mask([16]), 1)).toBe(true);
  });
});
