import { beforeEach, describe, expect, it } from 'vitest';
import type {
  Backend,
  BackendSession,
  DeviceTensor,
  DType,
  GraphAsset,
  IOBindingPlan,
  TensorLocation,
} from '../../backend/backend.js';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import { strategyFor } from './arch-strategy.js';
import { MemoryBank } from './memory-bank.js';

// ---------------------------------------------------------------------------
// FakeBackend: cpu-backed tensors so slot arithmetic is asserted on real bytes,
// with recorded copyRegion calls and a debugStats census. Mirrors the wasm
// backend's cpu semantics (copyRegion === TypedArray.set) — the M2 primitive
// the real backend agent implements.
// ---------------------------------------------------------------------------

function elemCount(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

class FakeTensor implements DeviceTensor {
  #disposed = false;
  constructor(
    readonly shape: readonly number[],
    readonly dtype: DType,
    readonly location: TensorLocation,
    readonly data: Float32Array,
    private readonly onDispose: (t: FakeTensor) => void,
  ) {}
  get disposed(): boolean {
    return this.#disposed;
  }
  dispose(): void {
    if (this.#disposed) throw new InvalidStateError('FakeTensor already disposed');
    this.#disposed = true;
    this.onDispose(this);
  }
}

class FakeBackend implements Backend {
  readonly kind = 'wasm' as const;
  readonly live = new Set<FakeTensor>();
  readonly copyRegionCalls: { slotIndex: number; srcElems: number; dstShape: number[] }[] = [];

  async init(): Promise<void> {}
  async createSession(_g: GraphAsset, _p?: IOBindingPlan): Promise<BackendSession> {
    throw new NotImplementedError('FakeBackend.createSession');
  }

  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    const t = new FakeTensor(
      [...shape],
      dtype,
      location,
      new Float32Array(elemCount(shape)),
      (x) => this.live.delete(x),
    );
    this.live.add(t);
    return t;
  }

  uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor {
    const src = data as Float32Array;
    const t = new FakeTensor([...shape], dtype, 'cpu', Float32Array.from(src), (x) =>
      this.live.delete(x),
    );
    this.live.add(t);
    return t;
  }

  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    const slotElems = elemCount(dst.shape.slice(1));
    const srcElems = elemCount(src.shape);
    if (srcElems !== slotElems) {
      throw new InvalidStateError(`copyRegion byte-count mismatch: src ${srcElems} != slot ${slotElems}`);
    }
    if (src.dtype !== dst.dtype) throw new InvalidStateError('copyRegion dtype mismatch');
    if (slotIndex < 0 || slotIndex >= dst.shape[0]!) throw new InvalidStateError('copyRegion slot out of bounds');
    (dst as FakeTensor).data.set((src as FakeTensor).data, slotIndex * slotElems);
    this.copyRegionCalls.push({ slotIndex, srcElems, dstShape: [...dst.shape] });
  }

  reshape(tensor: DeviceTensor, shape: readonly number[]): DeviceTensor {
    // Non-owning view: shares data, not tracked in `live`, dispose is a no-op —
    // so it never perturbs the debugStats census.
    return new FakeTensor([...shape], tensor.dtype, tensor.location, (tensor as FakeTensor).data, () => {});
  }

  async readback(tensor: DeviceTensor): Promise<ArrayBufferView> {
    return (tensor as FakeTensor).data;
  }

  debugStats(): { liveTensors: number; liveBytes: number } {
    let liveBytes = 0;
    for (const t of this.live) liveBytes += t.data.byteLength;
    return { liveTensors: this.live.size, liveBytes };
  }

  async dispose(): Promise<void> {
    for (const t of [...this.live]) t.dispose();
  }
}

// ---------------------------------------------------------------------------
// Small-but-faithful video section: EdgeTAM slot arithmetic (1 cond + 6 recent,
// 16-pointer bank, 4 tokens/pointer) with tiny token/dim sizes so ring bytes
// are easy to assert. kvLen = (1+6)*2 + 64 = 78.
// ---------------------------------------------------------------------------

const T = 2; // tokensPerMemoryMap
const MEM_DIM = 1;
const EMBED_DIM = 4; // embedDim / memDim = 4 = ptrTokens / maxObjectPointers (splits)
const M = 7; // maxCondFrames + numRecent

function video(overrides: Partial<VideoManifestSection> = {}): VideoManifestSection {
  return {
    maxCondFrames: 1,
    numRecent: 6,
    tokensPerMemoryMap: T,
    ptrTokens: 64,
    maxObjectPointers: 16,
    kvLen: M * T + 64,
    memDim: MEM_DIM,
    embedDim: EMBED_DIM,
    gridSize: 1,
    multiObjectBatch: true,
    initPath: 'noMemGraph',
    tposDelivery: 'precombined',
    occlusionThreshold: 0,
    ...overrides,
  };
}

function makeBank(backend: FakeBackend, overrides?: Partial<VideoManifestSection>) {
  const v = video(overrides);
  return new MemoryBank({ backend, video: v, strategy: strategyFor('edgetam', v), location: 'device' });
}

/**
 * Sentinel feature/pos tensors filled with `value` so slot writes are
 * checkable. Tagged `'float16'` — the ring `MemoryBank` allocates is
 * `'float16'` (the real memoryEncoder outputs it commits are), and
 * `copyRegion` requires src/dst dtypes to match; the fake backend's `data`
 * stays a plain `Float32Array` regardless since this harness never encodes
 * real half-float bits.
 */
function feat(backend: FakeBackend, value: number, shape: readonly number[] = [T, MEM_DIM]): DeviceTensor {
  const data = new Float32Array(elemCount(shape)).fill(value);
  return backend.uploadTensor(data, shape, 'float16');
}

function pointerVec(value: number): Float32Array {
  return new Float32Array(EMBED_DIM).fill(value);
}

/** Commit cond frame then tracked frames in order (with pointers), reproducing the streaming loop. */
function commitSequence(bank: MemoryBank, backend: FakeBackend, cond: number, tracked: number[]): void {
  bank.commit(cond, true, feat(backend, cond), feat(backend, 1000 + cond));
  bank.commitPointer(cond, pointerVec(cond));
  for (const f of tracked) {
    bank.commit(f, false, feat(backend, f), feat(backend, 1000 + f));
    bank.commitPointer(f, pointerVec(f));
  }
}

let backend: FakeBackend;
beforeEach(() => {
  backend = new FakeBackend();
});

describe('MemoryBank construction', () => {
  it('allocates exactly the two spatial rings and regions the slots', () => {
    const bank = makeBank(backend);
    expect(backend.live.size).toBe(2);
    for (const t of backend.live) {
      expect(t.shape).toEqual([M, T, MEM_DIM]);
      expect(t.location).toBe('device');
    }
    const slots = bank.slots;
    expect(slots.length).toBe(M);
    expect(slots[0]).toEqual({ frameIdx: -1, isCond: true, valid: false });
    expect(slots.slice(1).every((s) => !s.isCond)).toBe(true);
  });

  it('exposes slots as a defensive snapshot (mutation cannot leak in)', () => {
    const bank = makeBank(backend);
    const slots = bank.slots;
    slots[0]!.frameIdx = 999;
    expect(bank.slots[0]!.frameIdx).toBe(-1);
  });
});

describe('MemoryBank.commit — slot selection & eviction', () => {
  it('pins the cond frame in slot 0 and fills the recent ring in order', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6]);
    const slots = bank.slots;
    expect(slots[0]).toEqual({ frameIdx: 0, isCond: true, valid: true });
    expect(slots.slice(1).map((s) => s.frameIdx)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('evicts the oldest recent slot (smallest frameIdx) once the ring is full', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6]);
    // Frame 7 evicts frame 1 (oldest), reusing its physical slot.
    bank.commit(7, false, feat(backend, 7), feat(backend, 1007));
    const recent = bank.slots.slice(1).map((s) => s.frameIdx);
    expect(recent).toEqual([7, 2, 3, 4, 5, 6]);
    expect(recent).not.toContain(1);
  });

  it('writes committed features into the chosen physical slot (real bytes)', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2]);
    const asm = bank.assemble(3);
    const spatial = (asm.memorySpatial as FakeTensor).data;
    const pos = (asm.memorySpatialPos as FakeTensor).data;
    // slot 0 (cond frame 0) → value 0; slot 1 (frame 1) → value 1; slot 2 → 2.
    expect([...spatial.slice(0, T)]).toEqual([0, 0]);
    expect([...spatial.slice(T, 2 * T)]).toEqual([1, 1]);
    expect([...spatial.slice(2 * T, 3 * T)]).toEqual([2, 2]);
    // pos ring got the 1000+frame sentinel.
    expect([...pos.slice(T, 2 * T)]).toEqual([1001, 1001]);
  });

  it('accepts the relaxed byte-count copyRegion rule (leading batch dim on the source)', () => {
    const bank = makeBank(backend);
    // memoryEncoder output is [1, T, memDim]; element count still equals one slot.
    bank.commit(0, true, feat(backend, 5, [1, T, MEM_DIM]), feat(backend, 6, [1, T, MEM_DIM]));
    expect(bank.slots[0]!.frameIdx).toBe(0);
    const asm = bank.assemble(1);
    expect([...(asm.memorySpatial as FakeTensor).data.slice(0, T)]).toEqual([5, 5]);
  });

  it('issues exactly two copyRegion calls per commit and none during assemble', () => {
    const bank = makeBank(backend);
    bank.commit(0, true, feat(backend, 0), feat(backend, 0));
    expect(backend.copyRegionCalls.length).toBe(2);
    bank.assemble(1);
    expect(backend.copyRegionCalls.length).toBe(2); // assemble does not repack
    expect(backend.copyRegionCalls.every((c) => c.slotIndex === 0)).toBe(true);
  });
});

describe('MemoryBank.assemble — streaming rule & tpos indices', () => {
  it('excludes frame N from its own assembly (assemble happens before commit)', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2]);
    // Assemble AT frame 2: frame 2 was committed, but streaming excludes >= N.
    const asm = bank.assemble(2);
    // Valid maps: cond frame 0 + recent frame 1. Frame 2 (== N) excluded.
    expect(asm.validMaps).toBe(2);
    expect(asm.tposIndices[2]).toBe(-1n); // slot 2 holds frame 2 → excluded
  });

  it('assigns cond→row 6 and recent maps by RAW frame distance (e2e_loop bookkeeping)', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6]);
    const asm = bank.assemble(7);
    expect(asm.validMaps).toBe(7);
    // Physical slot i holds frame i; tpos: slot0 cond→6, slot i→ (7-i)-1 = 6-i.
    expect([...asm.tposIndices].map(Number)).toEqual([6, 5, 4, 3, 2, 1, 0]);
  });

  it('drops recent maps beyond the numRecent window while keeping the cond map', () => {
    const bank = makeBank(backend);
    // Only a cond frame (0) and one stale tracked frame (1); assemble far ahead.
    commitSequence(bank, backend, 0, [1]);
    const asm = bank.assemble(10); // frame 1 offset 9 > numRecent 6 → dropped
    expect(asm.validMaps).toBe(1); // only the cond map survives
    expect(asm.tposIndices[0]).toBe(6n);
    expect(asm.tposIndices[1]).toBe(-1n);
  });

  it('builds the spatial mask at whole-slot granularity for valid maps only', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2]);
    const asm = bank.assemble(3);
    // slots 0,1,2 valid (frames 0,1,2 all < 3) → their T-token windows are 1.
    for (let s = 0; s < 3; s++) {
      expect([...asm.memoryMask.slice(s * T, s * T + T)]).toEqual([1, 1]);
    }
    // slots 3..6 invalid → zero.
    for (let s = 3; s < M; s++) {
      expect([...asm.memoryMask.slice(s * T, s * T + T)]).toEqual([0, 0]);
    }
  });

  it('hasMemory tracks whether assemble would yield any valid map', () => {
    const bank = makeBank(backend);
    expect(bank.hasMemory(0)).toBe(false);
    bank.commit(0, true, feat(backend, 0), feat(backend, 0));
    expect(bank.hasMemory(0)).toBe(false); // streaming: frame 0 cannot see its own memory
    expect(bank.hasMemory(1)).toBe(true);
  });
});

describe('MemoryBank pointer bank — cond pinned + recent ring', () => {
  it('orders pointers cond-first then tracked offsets 1..P-1 (e2e_loop order)', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6]);
    const asm = bank.assemble(7);
    // e2e: [cond 0, offset1→6, offset2→5, ... offset6→1] = 7 pointers.
    const rows = [...Array(7)].map((_v, i) => asm.objectPointers[i * EMBED_DIM]);
    expect(rows).toEqual([0, 6, 5, 4, 3, 2, 1]);
    expect([...asm.pointerMask.slice(0, 7)]).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(asm.pointerMask[7]).toBe(0);
    // Pointer-region mask: 4 tokens per pointer, 7 pointers → 28 valid bits.
    const base = M * T;
    expect([...asm.memoryMask.slice(base, base + 28)].every((b) => b === 1)).toBe(true);
    expect(asm.memoryMask[base + 28]).toBe(0);
  });

  it('keeps a pointer reachable at an offset the map ring has already evicted', () => {
    const bank = makeBank(backend);
    // Commit through frame 7: the map ring evicts frame 1, but its POINTER (a
    // 16-deep ring) survives and is still reachable at offset 7 from frame 8.
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6, 7]);
    const asm = bank.assemble(8);
    // Maps: frame 1 gone (evicted); cond 0 + frames 2..7 → 7 maps.
    expect(asm.validMaps).toBe(7);
    // Pointers: cond 0 + offsets 1..7 → frames 7,6,5,4,3,2,1 → includes frame 1.
    const rows = [...Array(8)].map((_v, i) => asm.objectPointers[i * EMBED_DIM]);
    expect(rows).toEqual([0, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('evicts the oldest tracked pointer once the pointer ring overflows', () => {
    const bank = makeBank(backend, { maxObjectPointers: 4, ptrTokens: 16 });
    // 1 cond + tracked 1..5; recent-pointer ring holds 4 → frame 1 evicted.
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5]);
    const asm = bank.assemble(6);
    // cond 0 + offsets 1..3 (ring cap 4-1) → frames 5,4,3 ; capped total = 4.
    const rows = [...Array(4)].map((_v, i) => asm.objectPointers[i * EMBED_DIM]);
    expect(rows).toEqual([0, 5, 4, 3]);
    expect([...asm.pointerMask]).toEqual([1, 1, 1, 1]);
  });

  it('rejects a pointer whose length is not embedDim', () => {
    const bank = makeBank(backend);
    expect(() => bank.commitPointer(0, new Float32Array(EMBED_DIM + 1))).toThrow(InvalidStateError);
  });

  it('EdgeTAM pointer deltas are inert (all zero), padded to maxObjectPointers', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3]);
    const asm = bank.assemble(4);
    expect(asm.pointerDeltas.length).toBe(16);
    expect([...asm.pointerDeltas].every((d) => d === 0n)).toBe(true);
  });
});

describe('MemoryBank.invalidateAfter — refine support', () => {
  it('drops recent maps + tracked pointers after the frame, keeping cond', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4]);
    bank.invalidateAfter(2); // keep cond(0) + recent 1,2; drop 3,4
    const asm = bank.assemble(5);
    // Maps: cond 0 (row6), frame 1 (offset4→row3), frame 2 (offset3→row2).
    expect(asm.validMaps).toBe(3);
    const validFrames = bank.slots.filter((s) => s.valid).map((s) => s.frameIdx).sort((a, b) => a - b);
    expect(validFrames).toEqual([0, 1, 2]);
    // Pointers: cond 0 + offsets to frames 2,1 (3 & 4 removed).
    const rows: number[] = [];
    for (let i = 0; i < 16; i++) if (asm.pointerMask[i]) rows.push(asm.objectPointers[i * EMBED_DIM]!);
    expect(rows).toEqual([0, 2, 1]);
  });

  it('leaves the cond map/pointer untouched even when after the cut', () => {
    const bank = makeBank(backend);
    // Cond frame is 5 (later than the cut); it must survive invalidateAfter(3).
    bank.commit(5, true, feat(backend, 5), feat(backend, 1005));
    bank.commitPointer(5, pointerVec(5));
    bank.invalidateAfter(3);
    expect(bank.slots[0]).toEqual({ frameIdx: 5, isCond: true, valid: true });
    const asm = bank.assemble(6);
    expect(asm.validMaps).toBe(1);
    expect(asm.objectPointers[0]).toBe(5);
  });
});

describe('MemoryBank cond overflow (replace-on-new-prompt)', () => {
  it('overwrites the single cond slot when a second cond frame is committed', () => {
    const bank = makeBank(backend);
    bank.commit(0, true, feat(backend, 0), feat(backend, 0));
    // maxCondFrames = 1 → the new prompt replaces the old cond map.
    bank.commit(4, true, feat(backend, 4), feat(backend, 4));
    expect(bank.slots[0]).toEqual({ frameIdx: 4, isCond: true, valid: true });
    const asm = bank.assemble(5);
    expect([...(asm.memorySpatial as FakeTensor).data.slice(0, T)]).toEqual([4, 4]);
  });
});

describe('MemoryBank.reset & dispose', () => {
  it('reset invalidates all slots + pointers but retains the rings', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3]);
    const rings = bank.assemble(4);
    bank.reset();
    expect(bank.slots.every((s) => !s.valid && s.frameIdx === -1)).toBe(true);
    expect(bank.assemble(4).validMaps).toBe(0);
    // The device rings survive reset (not disposed) and are reusable.
    expect((rings.memorySpatial as FakeTensor).disposed).toBe(false);
    expect((rings.memorySpatialPos as FakeTensor).disposed).toBe(false);
    bank.commit(0, true, feat(backend, 9), feat(backend, 9));
    expect(bank.slots[0]!.frameIdx).toBe(0);
  });

  it('dispose releases both rings and poisons further use', () => {
    const bank = makeBank(backend);
    bank.dispose();
    expect(backend.live.size).toBe(0);
    expect(() => bank.commit(0, true, feat(backend, 0), feat(backend, 0))).toThrow(InvalidStateError);
    expect(() => bank.assemble(0)).toThrow(InvalidStateError);
    expect(() => bank.hasMemory(0)).toThrow(InvalidStateError);
  });

  it('assemble allocates no backend tensors — feeds are cpu-side typed arrays', () => {
    const bank = makeBank(backend);
    commitSequence(bank, backend, 0, [1, 2, 3, 4, 5, 6, 7, 8]);
    const liveBeforeAssemble = backend.live.size;
    const asm = bank.assemble(9);
    // Borrowed rings + cpu-side arrays only; no new device allocation.
    expect(backend.live.size).toBe(liveBeforeAssemble);
    expect(asm.tposIndices).toBeInstanceOf(BigInt64Array);
    expect(asm.memoryMask).toBeInstanceOf(Uint8Array);
    expect(asm.objectPointers).toBeInstanceOf(Float32Array);
  });
});
