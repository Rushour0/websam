/**
 * Per-object memory bank for the EdgeTAM video path.
 *
 * ONE instance tracks ONE object. Storage goes through the {@link Backend}
 * interface exclusively: on webgpu the spatial rings are device
 * {@link Backend.allocTensor} GPUBuffers written with {@link Backend.copyRegion};
 * on wasm they are cpu tensors. The bank cannot tell which — `location` is
 * decided by the engine (`backend.kind === 'webgpu' ? 'device' : 'cpu'`).
 *
 * Slot layout (mirrors `spec.py` / `e2e_loop.py`): `M = maxCondFrames + numRecent`
 * physical slots; `[0, maxCondFrames)` is the pinned conditioning region (never
 * evicted), `[maxCondFrames, M)` is the recent ring. Object pointers live in a
 * separate cpu-side bank (they are graph inputs anyway — a device ring buys
 * nothing).
 *
 * PERMUTATION-INVARIANT BINDING: EdgeTAM's memory attention RoPEs keys in
 * `M` groups of `tokensPerMemoryMap` identically (FINDINGS.md divergence 7 /
 * e2e_loop.py step 4), so physical slot order need not be temporal order.
 * Committed maps are bound IN PLACE and never repacked; each frame step
 * performs exactly one `copyRegion` per ring (the commit), and {@link assemble}
 * only rewrites the small cpu-side per-slot `tposIndices` + validity bits.
 *
 * STREAMING RULE: {@link assemble} for frame N sees only memories from frames
 * `< N` (frame N never attends to its own memory); the engine assembles BEFORE
 * committing frame N.
 */

import type { Backend, DeviceTensor, TensorLocation } from '../../backend/backend.js';
import { InvalidStateError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import type { VideoArchStrategy } from './arch-strategy.js';

/** Metadata for one physical memory slot. */
export interface MemorySlotMeta {
  /** Frame this slot holds, or `-1` when invalid. */
  frameIdx: number;
  /** True for the conditioning region; false for the recent ring. */
  isCond: boolean;
  /** Whether the slot currently holds a committed memory. */
  valid: boolean;
}

/** The per-frame attention feeds produced by {@link MemoryBank.assemble}. */
export interface MemoryAssembly {
  /**
   * Persistent spatial-feature ring `[M, tokensPerMemoryMap, memDim]`,
   * BORROWED — the engine must never dispose it.
   */
  memorySpatial: DeviceTensor;
  /** Persistent spatial positional-encoding ring `[M, tokensPerMemoryMap, memDim]`, BORROWED. */
  memorySpatialPos: DeviceTensor;
  /**
   * `[M]` per-slot temporal-position indices (`strategy.tposIndex`), `-1n` for
   * invalid / out-of-window slots. On `tposDelivery: 'indices'` these feed the
   * graph directly; on `'precombined'` the engine adds `tpos[idx]` to the
   * matching `memorySpatialPos` slot before upload.
   */
  tposIndices: BigInt64Array;
  /**
   * `[kvLen]` validity bits — spatial region (per physical slot,
   * `tokensPerMemoryMap` bits each) then the pointer region (`ptrTokens` bits).
   * `1` valid, `0` padding. The engine turns these into the graph's fp16-safe
   * additive attention bias (`0` / `-1e4`); the bank never assembles bias values.
   */
  memoryMask: Uint8Array;
  /** `[maxObjectPointers, embedDim]` zero-padded pointer bank (cpu-side, uploaded per frame). */
  objectPointers: Float32Array;
  /** `[maxObjectPointers]` streaming pointer deltas (all-zero on EdgeTAM). */
  pointerDeltas: BigInt64Array;
  /** `[maxObjectPointers]` pointer validity bits. */
  pointerMask: Uint8Array;
  /** Count of valid spatial maps; `0` → the engine takes the init / no-mem path. */
  validMaps: number;
}

/** Constructor arguments for {@link MemoryBank}. */
export interface MemoryBankInit {
  backend: Backend;
  video: VideoManifestSection;
  strategy: VideoArchStrategy;
  location: TensorLocation;
}

export class MemoryBank {
  readonly #backend: Backend;
  readonly #video: VideoManifestSection;
  readonly #strategy: VideoArchStrategy;

  /** Physical slots: `[0, maxCondFrames)` cond, `[maxCondFrames, M)` recent. */
  readonly #slots: MemorySlotMeta[];
  /** `[M, T, memDim]` spatial-feature ring. */
  readonly #memorySpatial: DeviceTensor;
  /** `[M, T, memDim]` spatial positional-encoding ring. */
  readonly #memorySpatialPos: DeviceTensor;

  /** Conditioning-frame pointers: never ring-evicted (mirror the cond maps). */
  readonly #condPointers = new Map<number, Float32Array>();
  /** Recent tracked-frame pointers: ring of `maxObjectPointers`, oldest evicted. */
  readonly #recentPointers = new Map<number, Float32Array>();

  #disposed = false;

  constructor(init: MemoryBankInit) {
    this.#backend = init.backend;
    this.#video = init.video;
    this.#strategy = init.strategy;

    const { maxCondFrames, numRecent, tokensPerMemoryMap, memDim } = this.#video;
    const maps = maxCondFrames + numRecent;
    const ringShape = [maps, tokensPerMemoryMap, memDim] as const;
    this.#memorySpatial = init.backend.allocTensor(ringShape, 'float32', init.location);
    this.#memorySpatialPos = init.backend.allocTensor(ringShape, 'float32', init.location);

    this.#slots = Array.from({ length: maps }, (_v, i): MemorySlotMeta => ({
      frameIdx: -1,
      isCond: i < maxCondFrames,
      valid: false,
    }));
  }

  /** Physical slots as a read-only snapshot (copies; mutation cannot leak in). */
  get slots(): readonly MemorySlotMeta[] {
    return this.#slots.map((s) => ({ ...s }));
  }

  #assertLive(method: string): void {
    if (this.#disposed) throw new InvalidStateError(`MemoryBank.${method} called after dispose()`);
  }

  /** Whether slot `s` is a valid map for frame `currentFrameIdx` under the streaming + window rules. */
  #slotRow(s: MemorySlotMeta, currentFrameIdx: number): number | null {
    if (!s.valid || s.frameIdx >= currentFrameIdx) return null;
    if (s.isCond) return this.#strategy.tposIndex({ isCond: true });
    const offset = currentFrameIdx - s.frameIdx;
    if (offset < 1 || offset > this.#video.numRecent) return null; // beyond the recent window
    return this.#strategy.tposIndex({ isCond: false, recentOffset: offset });
  }

  /** True iff {@link assemble} at `currentFrameIdx` would yield `validMaps > 0`. */
  hasMemory(currentFrameIdx: number): boolean {
    this.#assertLive('hasMemory');
    return this.#slots.some((s) => this.#slotRow(s, currentFrameIdx) !== null);
  }

  #condFrames(): number[] {
    const out: number[] = [];
    for (const s of this.#slots) if (s.isCond && s.valid) out.push(s.frameIdx);
    return out;
  }

  /**
   * Commit one frame's encoded memory into a slot (a single `copyRegion` per
   * ring). `memoryFeatures` / `memoryPos` are the memoryEncoder outputs for
   * THIS object (`[T, memDim]` after batch-slicing); the bank does NOT take
   * ownership — the caller disposes them.
   *
   * Slot choice:
   * - cond → first invalid cond slot, else the loser of
   *   `strategy.selectCondFrames(existing + this, thisFrame, maxCondFrames)`;
   * - recent → first invalid recent slot, else the oldest (smallest frameIdx).
   */
  commit(
    frameIdx: number,
    isCond: boolean,
    memoryFeatures: DeviceTensor,
    memoryPos: DeviceTensor,
  ): void {
    this.#assertLive('commit');
    const slot = isCond ? this.#pickCondSlot(frameIdx) : this.#pickRecentSlot();
    this.#backend.copyRegion(memoryFeatures, this.#memorySpatial, slot);
    this.#backend.copyRegion(memoryPos, this.#memorySpatialPos, slot);
    this.#slots[slot] = { frameIdx, isCond, valid: true };
  }

  #condSlotRange(): readonly [number, number] {
    return [0, this.#video.maxCondFrames];
  }

  #pickCondSlot(frameIdx: number): number {
    const [start, end] = this.#condSlotRange();
    for (let i = start; i < end; i++) if (!this.#slots[i]!.valid) return i;

    // Cond region full: keep the winners, overwrite a loser slot.
    const existing = this.#condFrames();
    const winners = new Set(
      this.#strategy.selectCondFrames([...existing, frameIdx], frameIdx, this.#video.maxCondFrames),
    );
    for (let i = start; i < end; i++) {
      if (!winners.has(this.#slots[i]!.frameIdx)) return i;
    }
    // Degenerate fallback (all existing survive AND this frame is not a winner):
    // evict the temporally-farthest cond frame.
    let victim = start;
    let farthest = -1;
    for (let i = start; i < end; i++) {
      const d = Math.abs(this.#slots[i]!.frameIdx - frameIdx);
      if (d > farthest) {
        farthest = d;
        victim = i;
      }
    }
    return victim;
  }

  #pickRecentSlot(): number {
    const [, condEnd] = this.#condSlotRange();
    const M = this.#slots.length;
    for (let i = condEnd; i < M; i++) if (!this.#slots[i]!.valid) return i;
    // Ring full: evict the oldest (smallest frameIdx).
    let oldest = condEnd;
    for (let i = condEnd + 1; i < M; i++) {
      if (this.#slots[i]!.frameIdx < this.#slots[oldest]!.frameIdx) oldest = i;
    }
    return oldest;
  }

  /**
   * Push one object pointer (`cpu Float32Array[embedDim]`). Conditioning-frame
   * pointers (frame currently held by a valid cond slot) are pinned; tracked
   * pointers form a ring of `maxObjectPointers`, oldest evicted. The array is
   * copied — the caller may reuse its buffer.
   */
  commitPointer(frameIdx: number, pointer: Float32Array): void {
    this.#assertLive('commitPointer');
    if (pointer.length !== this.#video.embedDim) {
      throw new InvalidStateError(
        `commitPointer: pointer length ${pointer.length} != embedDim ${this.#video.embedDim}`,
      );
    }
    const copy = pointer.slice();
    const [start, end] = this.#condSlotRange();
    const isCond = this.#slots
      .slice(start, end)
      .some((s) => s.valid && s.frameIdx === frameIdx);
    if (isCond) {
      this.#condPointers.set(frameIdx, copy);
      return;
    }
    this.#recentPointers.set(frameIdx, copy);
    if (this.#recentPointers.size > this.#video.maxObjectPointers) {
      let oldest = Infinity;
      for (const f of this.#recentPointers.keys()) if (f < oldest) oldest = f;
      this.#recentPointers.delete(oldest);
    }
  }

  /**
   * Build the frame-N attention feeds. Assembly is a cpu-side rewrite of the
   * small per-slot indices/bits + the pointer bank; the spatial rings are
   * returned borrowed, unmodified (see class docstring: zero spatial copies).
   */
  assemble(currentFrameIdx: number): MemoryAssembly {
    this.#assertLive('assemble');
    const { tokensPerMemoryMap: T, kvLen, maxObjectPointers: P, embedDim, ptrTokens } = this.#video;
    const M = this.#slots.length;
    const splits = ptrTokens / P; // KV tokens per pointer (embedDim / memDim)

    const tposIndices = new BigInt64Array(M).fill(-1n);
    const memoryMask = new Uint8Array(kvLen);
    let validMaps = 0;

    for (let s = 0; s < M; s++) {
      const row = this.#slotRow(this.#slots[s]!, currentFrameIdx);
      if (row === null) continue;
      tposIndices[s] = BigInt(row);
      memoryMask.fill(1, s * T, s * T + T);
      validMaps++;
    }

    // Object pointers: cond (past-or-equal, insertion order) then tracked
    // offsets 1..P-1 (most-recent-first), capped at P (mirrors e2e_loop.py).
    const ordered = this.#assemblePointers(currentFrameIdx);
    const objectPointers = new Float32Array(P * embedDim);
    const pointerMask = new Uint8Array(P);
    const ptrBase = M * T;
    for (let i = 0; i < ordered.length; i++) {
      objectPointers.set(ordered[i]!.ptr, i * embedDim);
      pointerMask[i] = 1;
      const tok = ptrBase + i * splits;
      memoryMask.fill(1, tok, tok + splits);
    }
    const pointerDeltas = this.#strategy.pointerTimeDeltas(
      ordered.map((p) => p.frameIdx),
      currentFrameIdx,
    );

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

  #assemblePointers(currentFrameIdx: number): { frameIdx: number; ptr: Float32Array }[] {
    const out: { frameIdx: number; ptr: Float32Array }[] = [];
    for (const [frameIdx, ptr] of this.#condPointers) {
      if (frameIdx <= currentFrameIdx) out.push({ frameIdx, ptr });
    }
    for (let d = 1; d < this.#video.maxObjectPointers; d++) {
      const ref = currentFrameIdx - d;
      const ptr = this.#recentPointers.get(ref);
      if (ptr) out.push({ frameIdx: ref, ptr });
    }
    return out.slice(0, this.#video.maxObjectPointers);
  }

  /**
   * Refine support: invalidate every NON-cond slot and tracked pointer whose
   * frameIdx is strictly greater than `frameIdx` (cond memories persist —
   * mirrors `MaskTimeline.invalidateAfter`).
   */
  invalidateAfter(frameIdx: number): void {
    this.#assertLive('invalidateAfter');
    for (const s of this.#slots) {
      if (!s.isCond && s.valid && s.frameIdx > frameIdx) {
        s.valid = false;
        s.frameIdx = -1;
      }
    }
    for (const f of [...this.#recentPointers.keys()]) {
      if (f > frameIdx) this.#recentPointers.delete(f);
    }
  }

  /** Invalidate all slots + pointers; the device rings are retained (reusable). */
  reset(): void {
    this.#assertLive('reset');
    const [, condEnd] = this.#condSlotRange();
    for (let i = 0; i < this.#slots.length; i++) {
      this.#slots[i] = { frameIdx: -1, isCond: i < condEnd, valid: false };
    }
    this.#condPointers.clear();
    this.#recentPointers.clear();
  }

  /** Dispose the device rings. Further use throws {@link InvalidStateError}. */
  dispose(): void {
    this.#assertLive('dispose');
    this.#disposed = true;
    this.#memorySpatial.dispose();
    this.#memorySpatialPos.dispose();
    this.#condPointers.clear();
    this.#recentPointers.clear();
  }
}
