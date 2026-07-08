import { describe, expect, it } from 'vitest';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import { strategyFor } from './arch-strategy.js';

/** Real EdgeTAM constants (tools/export/src/websam_export/spec.py EDGETAM_1024). */
function edgetamVideo(): VideoManifestSection {
  return {
    maxCondFrames: 1,
    numRecent: 6,
    tokensPerMemoryMap: 256,
    ptrTokens: 64,
    maxObjectPointers: 16,
    kvLen: 7 * 256 + 64, // 1856
    memDim: 64,
    embedDim: 256,
    gridSize: 64,
    multiObjectBatch: true,
    initPath: 'noMemFlag',
    tposDelivery: 'indices',
    occlusionThreshold: 0,
  };
}

describe('strategyFor', () => {
  it('returns an edgetam strategy carrying the arch tag', () => {
    const strategy = strategyFor('edgetam', edgetamVideo());
    expect(strategy.arch).toBe('edgetam');
  });

  it('throws NotImplementedError for sam3-tracker (branch lands in M3)', () => {
    let caught: unknown;
    try {
      strategyFor('sam3-tracker', edgetamVideo());
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(NotImplementedError);
    expect((caught as NotImplementedError).code).toBe('NOT_IMPLEMENTED');
    expect((caught as NotImplementedError).message).toContain('M3');
  });

  it('commitOccludedMemory is a strategy field (⚠ PIN-8), true for edgetam pending the spike', () => {
    expect(strategyFor('edgetam', edgetamVideo()).commitOccludedMemory).toBe(true);
  });
});

describe('tposIndex (spec.py::tpos_index case table)', () => {
  const strategy = strategyFor('edgetam', edgetamVideo());

  it('conditioning memories use the dedicated last slot: numRecent (6)', () => {
    expect(strategy.tposIndex({ isCond: true })).toBe(6);
    // recentOffset is irrelevant for cond entries (spec.py ignores it too).
    expect(strategy.tposIndex({ isCond: true, recentOffset: 3 })).toBe(6);
  });

  it('recent memory at recency rank k → index k-1, for every valid k', () => {
    for (let k = 1; k <= 6; k++) {
      expect(strategy.tposIndex({ isCond: false, recentOffset: k })).toBe(k - 1);
    }
  });

  it('throws when recentOffset is missing for a non-cond memory', () => {
    expect(() => strategy.tposIndex({ isCond: false })).toThrow(InvalidStateError);
  });

  it('throws when recentOffset is out of [1, numRecent] or not an integer', () => {
    expect(() => strategy.tposIndex({ isCond: false, recentOffset: 0 })).toThrow(
      InvalidStateError,
    );
    expect(() => strategy.tposIndex({ isCond: false, recentOffset: 7 })).toThrow(
      InvalidStateError,
    );
    expect(() => strategy.tposIndex({ isCond: false, recentOffset: 1.5 })).toThrow(
      InvalidStateError,
    );
  });

  it('respects the manifest numRecent, never a TS constant', () => {
    const video = { ...edgetamVideo(), numRecent: 3, kvLen: 4 * 256 + 64 };
    const s = strategyFor('edgetam', video);
    expect(s.tposIndex({ isCond: true })).toBe(3);
    expect(s.tposIndex({ isCond: false, recentOffset: 3 })).toBe(2);
    expect(() => s.tposIndex({ isCond: false, recentOffset: 4 })).toThrow(InvalidStateError);
  });
});

describe('selectCondFrames (HF _select_closest_cond_frames replication, ⚠ PIN-6)', () => {
  const strategy = strategyFor('edgetam', edgetamVideo());

  it('keeps everything (sorted ascending) when the set fits', () => {
    expect(strategy.selectCondFrames([], 5, 1)).toEqual([]);
    expect(strategy.selectCondFrames([9], 5, 1)).toEqual([9]);
    expect(strategy.selectCondFrames([10, 2], 5, 4)).toEqual([2, 10]);
  });

  it('max=1 (EdgeTAM): the single frame closest to currentFrame', () => {
    expect(strategy.selectCondFrames([0, 4, 9], 5, 1)).toEqual([4]);
    expect(strategy.selectCondFrames([0, 4, 9], 8, 1)).toEqual([9]);
    // The frame equal to currentFrame is distance 0 and always wins.
    expect(strategy.selectCondFrames([0, 5, 9], 5, 1)).toEqual([5]);
  });

  it('max=1 ties break toward the LOWER frameIdx', () => {
    expect(strategy.selectCondFrames([3, 7], 5, 1)).toEqual([3]);
    expect(strategy.selectCondFrames([7, 3], 5, 1)).toEqual([3]); // input order irrelevant
  });

  it('max>=2: closest-before + closest-at/after are always kept (HF rule)', () => {
    expect(strategy.selectCondFrames([0, 5, 10], 6, 2)).toEqual([5, 10]);
    // currentFrame before all candidates: no "before", closest-after + fill.
    expect(strategy.selectCondFrames([5, 10, 20], 2, 2)).toEqual([5, 10]);
    // currentFrame after all candidates: no "after", closest-before + fill.
    expect(strategy.selectCondFrames([1, 5, 10], 20, 2)).toEqual([5, 10]);
  });

  it('max>=2: remaining capacity fills by ascending temporal distance', () => {
    // before=5, after=10 kept; remaining {0 (dist 6), 20 (dist 14)} → 0.
    expect(strategy.selectCondFrames([0, 5, 10, 20], 6, 3)).toEqual([0, 5, 10]);
  });

  it('rejects a non-positive max', () => {
    expect(() => strategy.selectCondFrames([1, 2], 3, 0)).toThrow(InvalidStateError);
  });
});

describe('pointerTimeDeltas (streaming rule, ⚠ PIN-9)', () => {
  const strategy = strategyFor('edgetam', edgetamVideo());

  it('emits currentFrame - ptrFrame, most-recent-first, zero-padded to maxObjectPointers', () => {
    const deltas = strategy.pointerTimeDeltas([3, 5, 4], 7);
    expect(deltas).toBeInstanceOf(BigInt64Array);
    expect(deltas.length).toBe(16);
    expect([...deltas.slice(0, 3)]).toEqual([2n, 3n, 4n]); // frames 5, 4, 3
    expect([...deltas.slice(3)]).toEqual(new Array(13).fill(0n));
  });

  it('returns all zeros for an empty pointer bank', () => {
    expect([...strategy.pointerTimeDeltas([], 9)]).toEqual(new Array(16).fill(0n));
  });

  it('truncates to the most recent maxObjectPointers frames', () => {
    const frames = Array.from({ length: 20 }, (_, i) => i); // 0..19
    const deltas = strategy.pointerTimeDeltas(frames, 20);
    expect(deltas.length).toBe(16);
    // Most recent 16 frames are 19..4 → deltas 1..16.
    expect([...deltas]).toEqual(Array.from({ length: 16 }, (_, i) => BigInt(i + 1)));
  });
});
