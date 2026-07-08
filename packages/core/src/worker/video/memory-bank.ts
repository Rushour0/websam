/**
 * Per-object video memory bank (docs/m2-internal-contracts.md §3.2).
 *
 * One instance per tracked object. Storage goes through the {@link Backend}
 * interface exclusively: on webgpu the rings are `allocTensor(..., 'device')`
 * GPUBuffers written with `copyRegion`; on wasm they are cpu tensors — the
 * bank cannot tell (`location` is decided by the engine).
 *
 * Slot model: a fixed ring of `M = maxCondFrames + numRecent` spatial memory
 * maps, slots `[0, maxCondFrames)` forming the pinned cond region and
 * `[maxCondFrames, M)` the recent ring. Slots are bound IN PLACE: physical
 * slot order is not temporal order — memory attention is permutation-
 * invariant given correct per-slot pos/tpos and mask bits — so a frame step
 * performs ZERO spatial copies beyond the single `commit` copyRegion.
 *
 * The object-pointer bank stays cpu-side (16×256 f32 = 16 KB, uploaded per
 * frame): pointers are graph inputs anyway (`pointerDeltas` is data-
 * dependent) so a device ring buys nothing.
 *
 * Attn-bias assembly is MASK-BIT assembly: the mask enters the graph as
 * bool validity bits and the additive `-1e4` fp16-safe bias is built
 * in-graph — JS never assembles bias values (§2.2).
 */

import type { Backend, DeviceTensor } from '../../backend/backend.js';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import type { VideoArchStrategy } from './arch-strategy.js';

/** Metadata for one spatial-memory slot. */
export interface MemorySlotMeta {
  /** Source frame of the committed memory; -1 when invalid. */
  frameIdx: number;
  /** Whether the slot belongs to the pinned cond region. */
  isCond: boolean;
  /** Whether the slot currently holds a committed memory. */
  valid: boolean;
}

/** Everything `memoryAttention` needs for one object at one frame (§2.2 feeds). */
export interface MemoryAssembly {
  /** Persistent rings, BORROWED (never dispose): [M, T, memDim]. */
  memorySpatial: DeviceTensor;
  memorySpatialPos: DeviceTensor;
  /** [M] per-slot tpos indices (strategy.tposIndex), −1 for invalid slots. */
  tposIndices: BigInt64Array;
  /** [kvLen] validity bits (spatial region per slot, then pointer region). */
  memoryMask: Uint8Array;
  /** [P, embedDim] zero-padded pointer bank (cpu-side, uploaded per frame). */
  objectPointers: Float32Array;
  /** [P] streaming pointer time deltas, most-recent-first, zero-padded. */
  pointerDeltas: BigInt64Array;
  /** [P] pointer validity bits, aligned with `objectPointers` rows. */
  pointerMask: Uint8Array;
  /** Count of valid spatial maps (0 → engine takes the init/no-mem path). */
  validMaps: number;
}

/** One committed object pointer (cpu-side ring entry). */
interface PointerEntry {
  frameIdx: number;
  data: Float32Array;
}

/**
 * Fixed-slot memory bank for one tracked object.
 *
 * Streaming rule (normative): `assemble(N)` only counts slots and pointers
 * with `frameIdx < N` as valid — frame N never sees its own memory; assembly
 * happens BEFORE commit each frame.
 */
export class MemoryBank {
  readonly #backend: Backend;
  readonly #video: VideoManifestSection;
  readonly #strategy: VideoArchStrategy;
  readonly #slots: MemorySlotMeta[];
  readonly #memorySpatial: DeviceTensor;
  readonly #memorySpatialPos: DeviceTensor;
  /** Insertion-ordered pointer ring; eviction removes the smallest frameIdx. */
  #pointers: PointerEntry[] = [];
  #disposed = false;

  constructor(init: {
    backend: Backend;
    video: VideoManifestSection;
    strategy: VideoArchStrategy;
    location: 'device' | 'cpu';
  }) {
    const { backend, video, strategy, location } = init;
    if (video.tposDelivery !== 'indices') {
      // ⚠ PIN-3: the 'precombined' fallback (cpu pos+tpos sums uploaded per
      // frame) is implemented only if the spike forces it — 'indices' is the
      // strongly preferred path (§3.2).
      throw new NotImplementedError(
        `MemoryBank tposDelivery:'${video.tposDelivery}', lands only if the M2 spike forces it`,
      );
    }
    this.#backend = backend;
    this.#video = video;
    this.#strategy = strategy;
    const maps = video.maxCondFrames + video.numRecent;
    /** Slot layout: [0, maxCondFrames) = cond region; [maxCondFrames, M) = recent ring. */
    this.#slots = Array.from({ length: maps }, (_, i) => ({
      frameIdx: -1,
      isCond: i < video.maxCondFrames,
      valid: false,
    }));
    const ringShape = [maps, video.tokensPerMemoryMap, video.memDim] as const;
    this.#memorySpatial = backend.allocTensor(ringShape, 'float32', location);
    try {
      this.#memorySpatialPos = backend.allocTensor(ringShape, 'float32', location);
    } catch (err) {
      this.#memorySpatial.dispose();
      throw err;
    }
  }

  /** Slot layout: [0, maxCondFrames) = cond region; [maxCondFrames, M) = recent ring. */
  get slots(): readonly MemorySlotMeta[] {
    return this.#slots;
  }

  #assertLive(method: string): void {
    if (this.#disposed) {
      throw new InvalidStateError(`MemoryBank.${method} called after dispose()`);
    }
  }

  #assertFrameIdx(method: string, frameIdx: number): void {
    if (!Number.isInteger(frameIdx) || frameIdx < 0) {
      throw new InvalidStateError(
        `MemoryBank.${method}: frameIdx must be a non-negative integer, got ${frameIdx}`,
      );
    }
  }

  /** True iff assemble(currentFrameIdx) would yield validMaps > 0 (streaming rule applied). */
  hasMemory(currentFrameIdx: number): boolean {
    this.#assertLive('hasMemory');
    return this.#slots.some((s) => s.valid && s.frameIdx < currentFrameIdx);
  }

  /** Pick the cond-region slot to overwrite for a new cond commit at `frameIdx`. */
  #chooseCondSlot(frameIdx: number): number {
    const { maxCondFrames } = this.#video;
    // Re-prompting a frame already in the cond region refreshes its slot.
    for (let i = 0; i < maxCondFrames; i++) {
      if (this.#slots[i]!.valid && this.#slots[i]!.frameIdx === frameIdx) return i;
    }
    for (let i = 0; i < maxCondFrames; i++) {
      if (!this.#slots[i]!.valid) return i;
    }
    // Region full: keep the strategy's winners, overwrite a loser. The new
    // frame is always a winner (distance 0 to itself), so a loser exists.
    const pool = [
      ...this.#slots.slice(0, maxCondFrames).map((s) => s.frameIdx),
      frameIdx,
    ];
    const winners = new Set(this.#strategy.selectCondFrames(pool, frameIdx, maxCondFrames));
    for (let i = 0; i < maxCondFrames; i++) {
      if (!winners.has(this.#slots[i]!.frameIdx)) return i;
    }
    /* v8 ignore next 4 -- unreachable: the new frame always wins selection. */
    throw new InvalidStateError(
      `MemoryBank.commit: no cond slot to evict for frame ${frameIdx}`,
    );
  }

  /** Pick the recent-ring slot for a new non-cond commit at `frameIdx`. */
  #chooseRecentSlot(frameIdx: number): number {
    const { maxCondFrames } = this.#video;
    // Re-committing the same frame (e.g. after invalidate/re-track) refreshes in place.
    for (let i = maxCondFrames; i < this.#slots.length; i++) {
      if (this.#slots[i]!.valid && this.#slots[i]!.frameIdx === frameIdx) return i;
    }
    for (let i = maxCondFrames; i < this.#slots.length; i++) {
      if (!this.#slots[i]!.valid) return i;
    }
    // Ring full: evict the oldest (smallest frameIdx).
    let oldest = maxCondFrames;
    for (let i = maxCondFrames + 1; i < this.#slots.length; i++) {
      if (this.#slots[i]!.frameIdx < this.#slots[oldest]!.frameIdx) oldest = i;
    }
    return oldest;
  }

  #assertSlotShaped(method: string, name: string, tensor: DeviceTensor): void {
    const { tokensPerMemoryMap, memDim } = this.#video;
    const elements = tensor.shape.reduce((a, b) => a * b, 1);
    if (tensor.dtype !== 'float32' || elements !== tokensPerMemoryMap * memDim) {
      throw new InvalidStateError(
        `MemoryBank.${method}: ${name} must be float32 with ${tokensPerMemoryMap * memDim} ` +
          `elements ([T, memDim]), got ${tensor.dtype} [${tensor.shape.join(', ')}]`,
      );
    }
  }

  /**
   * Commit one frame's encoded memory. `memoryFeatures`/`memoryPos` are the
   * memoryEncoder outputs for THIS object ([T, memDim] after batch-slicing);
   * the bank copyRegions them into the chosen slot and does NOT take
   * ownership (caller disposes per §4.5). One `copyRegion` per ring — the
   * only spatial copies of a frame step.
   */
  commit(
    frameIdx: number,
    isCond: boolean,
    memoryFeatures: DeviceTensor,
    memoryPos: DeviceTensor,
  ): void {
    this.#assertLive('commit');
    this.#assertFrameIdx('commit', frameIdx);
    this.#assertSlotShaped('commit', 'memoryFeatures', memoryFeatures);
    this.#assertSlotShaped('commit', 'memoryPos', memoryPos);
    const slot = isCond ? this.#chooseCondSlot(frameIdx) : this.#chooseRecentSlot(frameIdx);
    this.#backend.copyRegion(memoryFeatures, this.#memorySpatial, slot);
    this.#backend.copyRegion(memoryPos, this.#memorySpatialPos, slot);
    const meta = this.#slots[slot]!;
    meta.frameIdx = frameIdx;
    meta.valid = true;
  }

  /** Push an object pointer (cpu Float32Array[embedDim]); ring of maxObjectPointers, oldest evicted. */
  commitPointer(frameIdx: number, pointer: Float32Array): void {
    this.#assertLive('commitPointer');
    this.#assertFrameIdx('commitPointer', frameIdx);
    const { embedDim, maxObjectPointers } = this.#video;
    if (pointer.length !== embedDim) {
      throw new InvalidStateError(
        `MemoryBank.commitPointer: pointer must have ${embedDim} elements, got ${pointer.length}`,
      );
    }
    // Re-committing a frame (refine path) replaces its pointer in place.
    const existing = this.#pointers.findIndex((p) => p.frameIdx === frameIdx);
    if (existing !== -1) this.#pointers.splice(existing, 1);
    this.#pointers.push({ frameIdx, data: pointer.slice() });
    while (this.#pointers.length > maxObjectPointers) {
      let oldest = 0;
      for (let i = 1; i < this.#pointers.length; i++) {
        if (this.#pointers[i]!.frameIdx < this.#pointers[oldest]!.frameIdx) oldest = i;
      }
      this.#pointers.splice(oldest, 1);
    }
  }

  /**
   * Build the frame-N attention feeds. STREAMING RULE: only slots/pointers
   * with frameIdx < currentFrameIdx count as valid (frame N never sees its
   * own memory); assembly happens BEFORE commit each frame. Slots are bound
   * in place — mask bits and per-slot tpos indices carry the temporal
   * semantics, not slot order.
   */
  assemble(currentFrameIdx: number): MemoryAssembly {
    this.#assertLive('assemble');
    this.#assertFrameIdx('assemble', currentFrameIdx);
    const { tokensPerMemoryMap: T, kvLen, maxObjectPointers, embedDim } = this.#video;
    const maps = this.#slots.length;

    // Per-slot visibility for this frame (streaming rule).
    const visible = this.#slots.map((s) => s.valid && s.frameIdx < currentFrameIdx);

    // Recency ranks among visible RECENT slots: descending frameIdx, 1-based.
    const recentByAge = this.#slots
      .map((meta, slot) => ({ meta, slot }))
      .filter(({ meta, slot }) => !meta.isCond && visible[slot])
      .sort((a, b) => b.meta.frameIdx - a.meta.frameIdx);
    const rankBySlot = new Map<number, number>();
    for (const [i, { slot }] of recentByAge.entries()) rankBySlot.set(slot, i + 1);

    const tposIndices = new BigInt64Array(maps).fill(-1n);
    const memoryMask = new Uint8Array(kvLen);
    let validMaps = 0;
    for (let slot = 0; slot < maps; slot++) {
      if (!visible[slot]) continue;
      validMaps++;
      const meta = this.#slots[slot]!;
      const tpos = meta.isCond
        ? this.#strategy.tposIndex({ isCond: true })
        : this.#strategy.tposIndex({ isCond: false, recentOffset: rankBySlot.get(slot)! });
      tposIndices[slot] = BigInt(tpos);
      memoryMask.fill(1, slot * T, (slot + 1) * T);
    }

    // Pointer bank: most-recent-first packing (aligned with pointerTimeDeltas).
    const visiblePointers = this.#pointers
      .filter((p) => p.frameIdx < currentFrameIdx)
      .sort((a, b) => b.frameIdx - a.frameIdx)
      .slice(0, maxObjectPointers);
    const objectPointers = new Float32Array(maxObjectPointers * embedDim);
    const pointerMask = new Uint8Array(maxObjectPointers);
    for (const [i, entry] of visiblePointers.entries()) {
      objectPointers.set(entry.data, i * embedDim);
      pointerMask[i] = 1;
    }
    const pointerDeltas = this.#strategy.pointerTimeDeltas(
      visiblePointers.map((p) => p.frameIdx),
      currentFrameIdx,
    );
    // Pointer region of the KV mask ([M*T, kvLen)): the ptrTokens projected
    // tokens are derived from the whole pointer bank, so they are valid iff
    // at least one pointer is. ⚠ PIN-1/9: re-pinned if pointers enter fused.
    if (visiblePointers.length > 0) memoryMask.fill(1, maps * T, kvLen);

    return {
      memorySpatial: this.#memorySpatial,
      memorySpatialPos: this.#memorySpatialPos,
      tposIndices,
      memoryMask,
      objectPointers,
      pointerDeltas,
      pointerMask,
      validMaps,
    };
  }

  /** Refine support: invalidate every NON-cond slot and pointer with frameIdx > `frameIdx`. */
  invalidateAfter(frameIdx: number): void {
    this.#assertLive('invalidateAfter');
    for (const meta of this.#slots) {
      if (!meta.isCond && meta.valid && meta.frameIdx > frameIdx) {
        meta.valid = false;
        meta.frameIdx = -1;
      }
    }
    this.#pointers = this.#pointers.filter((p) => p.frameIdx <= frameIdx);
  }

  /** All slots/pointers invalid; rings retained (no allocation churn). */
  reset(): void {
    this.#assertLive('reset');
    for (const meta of this.#slots) {
      meta.valid = false;
      meta.frameIdx = -1;
    }
    this.#pointers = [];
  }

  /** Rings disposed; further use (including a second dispose) → InvalidStateError. */
  dispose(): void {
    this.#assertLive('dispose');
    this.#disposed = true;
    this.#memorySpatial.dispose();
    this.#memorySpatialPos.dispose();
    this.#pointers = [];
  }
}
