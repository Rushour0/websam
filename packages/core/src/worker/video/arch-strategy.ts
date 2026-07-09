/**
 * Per-architecture video (memory-bank) semantics, isolated in ONE strategy
 * object so a second tracker family (SAM3) lands at M3 by adding a branch —
 * never by editing the memory bank or engine.
 *
 * Every rule here mirrors the executable export spec
 * (`tools/export/src/websam_export/spec.py`) and the EdgeTAM export spike
 * (`tools/export/spikes/m2-edgetam/{FINDINGS.md,e2e_loop.py}`), which is the
 * authoritative source for the EdgeTAM numbers. Where FINDINGS.md corrected an
 * earlier assumption the divergence is called out inline.
 */

import type { ModelSpec } from '../../registry.js';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';

/** A memory-bank entry's frame identity and conditioning status. */
export interface MemoryEntryRef {
  frameIdx: number;
  isCond: boolean;
}

/** Architecture-specific bookkeeping the memory bank and engine defer to. */
export interface VideoArchStrategy {
  /** The architecture this strategy serves. */
  readonly arch: ModelSpec['arch'];

  /**
   * Temporal-position embedding index for one memory entry (mirrors
   * `spec.py::tpos_index`): a **conditioning** map uses the dedicated last
   * slot `numRecent`; a **recent** map at `recentOffset = k` uses `k - 1`.
   *
   * `recentOffset` is the RAW FRAME DISTANCE `currentFrame − entry.frameIdx`
   * (FINDINGS.md gotcha 3 + e2e_loop.py: HF gathers recent maps by temporal
   * offset `k = t − prev`, not by dense recency rank — the two coincide for
   * gapless streaming but diverge when a conditioning frame falls inside the
   * recent window, where the cond frame is skipped yet still consumes an
   * offset). Must satisfy `1 <= recentOffset <= numRecent`.
   *
   * @throws InvalidStateError — `recentOffset` missing (non-cond) or out of range.
   */
  tposIndex(entry: { isCond: boolean; recentOffset?: number }): number;

  /**
   * Which conditioning frames survive when the cond region would overflow,
   * replicating HF `_select_closest_cond_frames`: keep the closest-before and
   * closest-after anchors, then fill by `|Δt|` to `currentFrame`, ties broken
   * toward the LOWER frameIdx. Winners are returned in the input's insertion
   * order. `max <= 0` or `condFrames.length <= max` returns every frame
   * unchanged (EdgeTAM's `max_cond_frame_num = -1` maps to a large `max`; the
   * single-prompt M2 path never overflows).
   */
  selectCondFrames(condFrames: readonly number[], currentFrame: number, max: number): number[];

  /**
   * Streaming pointer temporal deltas (`currentFrame − ptrFrame`),
   * most-recent-first, zero-padded to `maxObjectPointers`; int64 because it is
   * a graph input on architectures that consume it.
   *
   * EdgeTAM DISABLES pointer temporal position encoding
   * (`enable_temporal_pos_encoding_for_object_pointers = False`, FINDINGS.md
   * divergence 4) — its exported graph has no pointer-time input at all, so
   * this returns an all-zero (inert) array of length `maxObjectPointers`.
   */
  pointerTimeDeltas(ptrFrames: readonly number[], currentFrame: number): BigInt64Array;

  /** Whether an occluded frame's encoded memory is still committed to the bank. */
  readonly commitOccludedMemory: boolean;
}

/**
 * EdgeTAM strategy, pinned to the M2 export spike. Notable divergences from the
 * SAM3-tracker semantics in `spec.py` (all confirmed by `e2e_loop.py`,
 * IoU 1.0 vs HF):
 *
 * - 512 tokens per memory map (256 1D + 256 2D perceiver latents), 1 cond + 6
 *   recent maps, `kvLen = 3648` — all delivered via the manifest.
 * - Recent-map tpos indexed by raw frame distance (see {@link tposIndex}).
 * - Object pointers carry NO temporal position (deltas are inert).
 * - Memory is committed on every tracked frame regardless of occlusion — the
 *   e2e loop never gates the memory encoder on `object_score_logits`
 *   (FINDINGS.md divergence 5), so {@link commitOccludedMemory} is `true`.
 */
class EdgetamStrategy implements VideoArchStrategy {
  readonly arch = 'edgetam' as const;
  readonly commitOccludedMemory = true;

  readonly #numRecent: number;
  readonly #maxObjectPointers: number;

  constructor(video: VideoManifestSection) {
    this.#numRecent = video.numRecent;
    this.#maxObjectPointers = video.maxObjectPointers;
  }

  tposIndex(entry: { isCond: boolean; recentOffset?: number }): number {
    if (entry.isCond) return this.#numRecent;
    const k = entry.recentOffset;
    if (k === undefined) {
      throw new InvalidStateError('tposIndex: recentOffset is required for non-conditioning memories');
    }
    if (!Number.isInteger(k) || k < 1 || k > this.#numRecent) {
      throw new InvalidStateError(`tposIndex: recentOffset must be in [1, ${this.#numRecent}], got ${k}`);
    }
    return k - 1;
  }

  selectCondFrames(condFrames: readonly number[], currentFrame: number, max: number): number[] {
    if (max <= 0 || condFrames.length <= max) return [...condFrames];

    const selected = new Set<number>();
    // HF forces one anchor on each side of the current frame when max >= 2.
    if (max >= 2) {
      let before = -Infinity;
      let after = Infinity;
      for (const t of condFrames) {
        if (t < currentFrame && t > before) before = t;
        if (t >= currentFrame && t < after) after = t;
      }
      if (before !== -Infinity) selected.add(before);
      if (after !== Infinity) selected.add(after);
    }
    // Fill the rest by |Δt|, ties toward the lower frameIdx.
    const remaining = condFrames
      .filter((t) => !selected.has(t))
      .sort((a, b) => Math.abs(a - currentFrame) - Math.abs(b - currentFrame) || a - b);
    for (const t of remaining) {
      if (selected.size >= max) break;
      selected.add(t);
    }
    return condFrames.filter((t) => selected.has(t)).slice(0, max);
  }

  pointerTimeDeltas(_ptrFrames: readonly number[], _currentFrame: number): BigInt64Array {
    // EdgeTAM: pointer temporal PE disabled → deltas are never read; return the
    // fixed-length inert (zero) vector so callers can bind a stable shape.
    return new BigInt64Array(this.#maxObjectPointers);
  }
}

/**
 * Build the {@link VideoArchStrategy} for a model architecture from its video
 * manifest section.
 *
 * @throws NotImplementedError — for architectures whose video strategy has not
 * landed yet (SAM3 tracker arrives at M3).
 */
export function strategyFor(arch: ModelSpec['arch'], video: VideoManifestSection): VideoArchStrategy {
  switch (arch) {
    case 'edgetam':
      return new EdgetamStrategy(video);
    case 'sam3-tracker':
      throw new NotImplementedError('strategyFor(sam3-tracker), lands in M3');
    default: {
      const exhaustive: never = arch;
      throw new NotImplementedError(`strategyFor(${String(exhaustive)})`);
    }
  }
}
