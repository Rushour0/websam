/**
 * Per-arch video-memory semantics, isolated in ONE strategy object so SAM3
 * lands at M3 by adding a branch here — the memory bank and video engine
 * never contain arch conditionals.
 *
 * Mirrors the executable export spec (`tools/export/src/websam_export/spec.py`):
 * the tpos rule, cond-frame selection, and pointer-delta packing below are
 * the same rules the Python golden generator runs, so JS and the export
 * pipeline share one contract (docs/m2-internal-contracts.md §3.1).
 *
 * Every semantic constant (numRecent, maxObjectPointers, …) comes from the
 * manifest's {@link VideoManifestSection} — never a TS constant.
 */

import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { ModelSpec } from '../../registry.js';
import type { VideoManifestSection } from '../../weights/manifest.js';

/** A memory-bank entry reference: which frame produced it, and whether it was prompted. */
export interface MemoryEntryRef {
  frameIdx: number;
  isCond: boolean;
}

/** Arch-specific memory semantics consumed by the memory bank and video engine. */
export interface VideoArchStrategy {
  readonly arch: ModelSpec['arch'];
  /**
   * Temporal-position embedding index for one memory entry (spec.py tpos_index):
   * cond → numRecent; recent at offset k (1 = most recent valid entry) → k-1.
   * `recentOffset` is the 1-based RECENCY RANK among valid recent slots
   * (descending frameIdx), not the raw frame distance.            ⚠ PIN-5
   */
  tposIndex(entry: { isCond: boolean; recentOffset?: number }): number;
  /**
   * Which cond frames stay when the region overflows / at assembly, replicating
   * HF `_select_closest_cond_frames` tie-breaking exactly (EdgeTAM max=1 → the
   * single closest to `currentFrame`; ties break toward the LOWER frameIdx). ⚠ PIN-6
   */
  selectCondFrames(condFrames: readonly number[], currentFrame: number, max: number): number[];
  /**
   * Streaming pointer deltas (`currentFrame - ptrFrame`), most-recent-first,
   * zero-padded to maxObjectPointers; int64 because it is a graph input.  ⚠ PIN-9
   */
  pointerTimeDeltas(ptrFrames: readonly number[], currentFrame: number): BigInt64Array;
  /** Whether an occluded frame's memory is still committed to the bank. ⚠ PIN-8 */
  readonly commitOccludedMemory: boolean;
}

/**
 * Replicates HF `_select_closest_cond_frames`: keep everything when it fits;
 * otherwise keep the closest cond frame strictly before `currentFrame`, the
 * closest at/after it, then fill remaining capacity by ascending absolute
 * distance. Candidates are sorted ascending first, so the stable sort breaks
 * distance ties toward the LOWER frameIdx — which is also what makes the
 * `max === 1` extension (EdgeTAM) "single closest, ties toward lower". ⚠ PIN-6
 */
function selectClosestCondFrames(
  condFrames: readonly number[],
  currentFrame: number,
  max: number,
): number[] {
  if (!Number.isInteger(max) || max < 1) {
    throw new InvalidStateError(`selectCondFrames: max must be a positive integer, got ${max}`);
  }
  const ascending = [...condFrames].sort((a, b) => a - b);
  if (ascending.length <= max) return ascending;

  if (max === 1) {
    // HF asserts max >= 2; EdgeTAM's maxCondFrames = 1 extends the rule to
    // "the single closest to currentFrame", ties toward the lower frameIdx.
    let winner = ascending[0]!;
    for (const f of ascending) {
      if (Math.abs(f - currentFrame) < Math.abs(winner - currentFrame)) winner = f;
    }
    return [winner];
  }

  const selected = new Set<number>();
  // Closest cond frame strictly before currentFrame.
  const before = ascending.filter((f) => f < currentFrame);
  if (before.length > 0) selected.add(before[before.length - 1]!);
  // Closest cond frame at/after currentFrame.
  const after = ascending.find((f) => f >= currentFrame);
  if (after !== undefined) selected.add(after);
  // Fill the remaining capacity by temporal closeness (stable → lower wins ties).
  const remaining = ascending
    .filter((f) => !selected.has(f))
    .sort((a, b) => Math.abs(a - currentFrame) - Math.abs(b - currentFrame));
  for (const f of remaining) {
    if (selected.size >= max) break;
    selected.add(f);
  }
  return [...selected].sort((a, b) => a - b);
}

/**
 * Build the strategy for one architecture. EdgeTAM is the only M2 arch;
 * `'sam3-tracker'` gains its branch at M3.
 *
 * @param arch - Architecture family from the registry {@link ModelSpec}.
 * @param video - The manifest's video section (all semantic constants).
 * @throws NotImplementedError — arch without an M2 video strategy.
 */
export function strategyFor(
  arch: ModelSpec['arch'],
  video: VideoManifestSection,
): VideoArchStrategy {
  if (arch !== 'edgetam') {
    throw new NotImplementedError(`strategyFor('${arch}') video strategy, lands in M3`);
  }
  const { numRecent, maxObjectPointers } = video;
  return {
    arch,
    tposIndex(entry) {
      // spec.py::tpos_index — cond → the dedicated last slot (numRecent);
      // recent at recency rank k (1 = most recent) → k - 1.
      if (entry.isCond) return numRecent;
      const k = entry.recentOffset;
      if (k === undefined) {
        throw new InvalidStateError('tposIndex: recentOffset is required for non-cond memories');
      }
      if (!Number.isInteger(k) || k < 1 || k > numRecent) {
        throw new InvalidStateError(`tposIndex: recentOffset must be in [1, ${numRecent}], got ${k}`);
      }
      return k - 1;
    },
    selectCondFrames(condFrames, currentFrame, max) {
      return selectClosestCondFrames(condFrames, currentFrame, max);
    },
    pointerTimeDeltas(ptrFrames, currentFrame) {
      // ⚠ PIN-9: EdgeTAM streaming rule assumed `currentFrame - ptrFrame`,
      // most-recent-first; the spike may re-pin this as manifest/strategy data.
      const deltas = new BigInt64Array(maxObjectPointers); // zero-padded tail
      const recentFirst = [...ptrFrames].sort((a, b) => b - a).slice(0, maxObjectPointers);
      for (const [i, frame] of recentFirst.entries()) {
        deltas[i] = BigInt(currentFrame - frame);
      }
      return deltas;
    },
    // ⚠ PIN-8: HF SAM2/EdgeTAM commits the encoded (empty-mask) memory even
    // for occluded frames; the spike may flip this to false.
    commitOccludedMemory: true,
  };
}
