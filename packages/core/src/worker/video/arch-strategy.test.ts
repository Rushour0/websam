import { describe, expect, it } from 'vitest';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import { strategyFor } from './arch-strategy.js';

/** EdgeTAM video section (FINDINGS.md values); override per test. */
function video(overrides: Partial<VideoManifestSection> = {}): VideoManifestSection {
  return {
    maxCondFrames: 1,
    numRecent: 6,
    tokensPerMemoryMap: 512,
    ptrTokens: 64,
    maxObjectPointers: 16,
    kvLen: 3648,
    memDim: 64,
    embedDim: 256,
    gridSize: 64,
    multiObjectBatch: true,
    initPath: 'noMemGraph',
    tposDelivery: 'precombined',
    occlusionThreshold: 0,
    ...overrides,
  };
}

const edgetam = () => strategyFor('edgetam', video());

describe('strategyFor', () => {
  it('builds the EdgeTAM strategy', () => {
    const s = edgetam();
    expect(s.arch).toBe('edgetam');
    // e2e_loop.py commits memory on every tracked frame (no occlusion gate).
    expect(s.commitOccludedMemory).toBe(true);
  });

  it('defers SAM3 tracker to M3', () => {
    expect(() => strategyFor('sam3-tracker', video())).toThrow(NotImplementedError);
  });
});

describe('EdgeTAM tposIndex — the spec.py::tpos_index case table', () => {
  const s = edgetam();

  it('maps conditioning memories to the dedicated last slot (numRecent)', () => {
    expect(s.tposIndex({ isCond: true })).toBe(6);
    // recentOffset is ignored for cond entries.
    expect(s.tposIndex({ isCond: true, recentOffset: 3 })).toBe(6);
  });

  it('maps recent offset k to row k-1 across the whole window', () => {
    for (let k = 1; k <= 6; k++) {
      expect(s.tposIndex({ isCond: false, recentOffset: k })).toBe(k - 1);
    }
  });

  it('throws on a missing recentOffset for a non-conditioning memory', () => {
    expect(() => s.tposIndex({ isCond: false })).toThrow(InvalidStateError);
  });

  it('throws on an out-of-range or non-integer recentOffset', () => {
    expect(() => s.tposIndex({ isCond: false, recentOffset: 0 })).toThrow(InvalidStateError);
    expect(() => s.tposIndex({ isCond: false, recentOffset: 7 })).toThrow(InvalidStateError);
    expect(() => s.tposIndex({ isCond: false, recentOffset: -1 })).toThrow(InvalidStateError);
    expect(() => s.tposIndex({ isCond: false, recentOffset: 2.5 })).toThrow(InvalidStateError);
  });

  it('honours a different numRecent from the manifest (upper bound + cond slot move)', () => {
    const s10 = strategyFor('edgetam', video({ numRecent: 10 }));
    expect(s10.tposIndex({ isCond: true })).toBe(10);
    expect(s10.tposIndex({ isCond: false, recentOffset: 10 })).toBe(9);
    expect(() => s10.tposIndex({ isCond: false, recentOffset: 11 })).toThrow(InvalidStateError);
  });
});

describe('EdgeTAM selectCondFrames', () => {
  const s = edgetam();

  it('returns every frame when at or under the cap (the single-prompt M2 path)', () => {
    expect(s.selectCondFrames([0], 5, 1)).toEqual([0]);
    expect(s.selectCondFrames([2, 7], 5, 2)).toEqual([2, 7]);
    expect(s.selectCondFrames([], 5, 1)).toEqual([]);
    // max <= 0 disables capping (EdgeTAM's max_cond_frame_num = -1).
    expect(s.selectCondFrames([1, 2, 3], 2, 0)).toEqual([1, 2, 3]);
  });

  it('max=1 keeps the single closest to the current frame', () => {
    expect(s.selectCondFrames([0, 4, 9], 8, 1)).toEqual([9]);
    expect(s.selectCondFrames([0, 4, 9], 3, 1)).toEqual([4]);
  });

  it('max=1 breaks ties toward the LOWER frameIdx', () => {
    // frames 2 and 8 are equidistant from 5 → keep 2.
    expect(s.selectCondFrames([2, 8], 5, 1)).toEqual([2]);
  });

  it('max>=2 keeps a before-anchor and an after-anchor, filling by distance', () => {
    // current=5: before-anchor 4, after-anchor 6; fill (max 3) picks 3 over 9.
    expect(s.selectCondFrames([0, 3, 4, 6, 9], 5, 3)).toEqual([3, 4, 6]);
  });

  it('preserves insertion order among the winners', () => {
    const winners = s.selectCondFrames([9, 1, 5, 3], 4, 2);
    // anchors: before=3, after=5 → returned in input order [5, 3] filtered → [5,3]?
    // input order is [9,1,5,3] so filtered winners keep that order.
    expect(winners).toEqual([5, 3]);
  });
});

describe('EdgeTAM pointerTimeDeltas', () => {
  it('is inert (all-zero) and padded to maxObjectPointers (pointer temporal PE disabled)', () => {
    const s = edgetam();
    const deltas = s.pointerTimeDeltas([9, 7, 3], 10);
    expect(deltas).toBeInstanceOf(BigInt64Array);
    expect(deltas.length).toBe(16);
    expect([...deltas].every((d) => d === 0n)).toBe(true);
  });

  it('tracks a non-default maxObjectPointers', () => {
    const s = strategyFor('edgetam', video({ maxObjectPointers: 8, ptrTokens: 32 }));
    expect(s.pointerTimeDeltas([], 0).length).toBe(8);
  });
});
