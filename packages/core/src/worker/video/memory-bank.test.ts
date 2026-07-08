import { describe, expect, it } from 'vitest';
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
// FakeBackend: cpu typed arrays + a REAL copyRegion (contiguous byte copy per
// the §1.1 relaxed rule) so slot arithmetic is asserted on real bytes.
// ---------------------------------------------------------------------------

function typedArrayFor(dtype: DType, elements: number): Float32Array | BigInt64Array | Uint8Array | Int32Array | Uint16Array {
  switch (dtype) {
    case 'float32':
      return new Float32Array(elements);
    case 'int64':
      return new BigInt64Array(elements);
    case 'uint8':
    case 'bool':
      return new Uint8Array(elements);
    case 'int32':
      return new Int32Array(elements);
    case 'float16':
      return new Uint16Array(elements);
  }
}

class FakeTensor implements DeviceTensor {
  disposed = false;
  constructor(
    readonly shape: readonly number[],
    readonly dtype: DType,
    readonly location: TensorLocation,
    readonly data: ReturnType<typeof typedArrayFor>,
    private readonly onDispose: (t: FakeTensor) => void,
  ) {}
  dispose(): void {
    if (this.disposed) throw new InvalidStateError('FakeTensor disposed twice');
    this.disposed = true;
    this.onDispose(this);
  }
}

interface CopyRegionCall {
  src: FakeTensor;
  dst: FakeTensor;
  slotIndex: number;
}

class FakeBackend implements Backend {
  readonly kind = 'wasm' as const;
  readonly live = new Set<FakeTensor>();
  readonly allocCalls: { shape: readonly number[]; dtype: DType; location: TensorLocation }[] = [];
  readonly copyRegionCalls: CopyRegionCall[] = [];

  async init(): Promise<void> {}
  async createSession(_graph: GraphAsset, _plan?: IOBindingPlan): Promise<BackendSession> {
    throw new Error('FakeBackend.createSession is not used by the bank');
  }

  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.allocCalls.push({ shape: [...shape], dtype, location });
    const elements = shape.reduce((a, b) => a * b, 1);
    const tensor = new FakeTensor(
      [...shape],
      dtype,
      location,
      typedArrayFor(dtype, elements),
      (t) => this.live.delete(t),
    );
    this.live.add(tensor);
    return tensor;
  }

  uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor {
    const elements = shape.reduce((a, b) => a * b, 1);
    const copy = typedArrayFor(dtype, elements);
    copy.set(data as never);
    const tensor = new FakeTensor([...shape], dtype, 'cpu', copy, (t) => this.live.delete(t));
    this.live.add(tensor);
    return tensor;
  }

  /** §1.1 relaxed rule: src element count == one dst slot's, same dtype; contiguous copy. */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    const s = src as FakeTensor;
    const d = dst as FakeTensor;
    if (s.disposed || d.disposed) throw new InvalidStateError('copyRegion on disposed tensor');
    if (s.dtype !== d.dtype) throw new InvalidStateError('copyRegion dtype mismatch');
    const slotElements = d.shape.slice(1).reduce((a, b) => a * b, 1);
    const srcElements = s.shape.reduce((a, b) => a * b, 1);
    if (srcElements !== slotElements) {
      throw new InvalidStateError(
        `copyRegion element-count mismatch: src ${srcElements} vs slot ${slotElements}`,
      );
    }
    const slotCount = d.shape[0] ?? 0;
    if (!Number.isInteger(slotIndex) || slotIndex < 0 || slotIndex >= slotCount) {
      throw new InvalidStateError(`copyRegion slot ${slotIndex} out of [0, ${slotCount})`);
    }
    (d.data as Float32Array).set(s.data as Float32Array, slotIndex * slotElements);
    this.copyRegionCalls.push({ src: s, dst: d, slotIndex });
  }

  async readback(tensor: DeviceTensor): Promise<ArrayBufferView> {
    return (tensor as FakeTensor).data;
  }

  async dispose(): Promise<void> {
    for (const t of [...this.live]) t.dispose();
  }
}

// ---------------------------------------------------------------------------
// Fixtures. The tiny section keeps byte-level assertions readable; the
// EdgeTAM section checks the real spec.py constants. Both satisfy the kvLen
// identity the manifest parser enforces.
// ---------------------------------------------------------------------------

/** Tiny section: M = 1 cond + 3 recent = 4 slots, T=2 tokens, memDim=3 → 6-float slots. */
function tinyVideo(): VideoManifestSection {
  return {
    maxCondFrames: 1,
    numRecent: 3,
    tokensPerMemoryMap: 2,
    ptrTokens: 5,
    maxObjectPointers: 4,
    kvLen: 4 * 2 + 5, // 13
    memDim: 3,
    embedDim: 2,
    gridSize: 8,
    multiObjectBatch: true,
    initPath: 'noMemFlag',
    tposDelivery: 'indices',
    occlusionThreshold: 0,
  };
}

/** Real EdgeTAM constants (spec.py EDGETAM_1024): 1 cond + 6 recent, 256 latents, kvLen 1856. */
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

function makeBank(video: VideoManifestSection = tinyVideo()) {
  const backend = new FakeBackend();
  const strategy = strategyFor('edgetam', video);
  const bank = new MemoryBank({ backend, video, strategy, location: 'cpu' });
  return { backend, strategy, bank, video };
}

/** A [T, memDim] float32 tensor filled with `value` (distinct per commit in tests). */
function slotTensor(backend: FakeBackend, video: VideoManifestSection, value: number): DeviceTensor {
  const elements = video.tokensPerMemoryMap * video.memDim;
  return backend.uploadTensor(
    new Float32Array(elements).fill(value),
    [video.tokensPerMemoryMap, video.memDim],
    'float32',
  );
}

/** Commit frame memory filled with `frameIdx + 0.5` (features) / `-(frameIdx + 0.5)` (pos). */
function commitFrame(
  backend: FakeBackend,
  bank: MemoryBank,
  video: VideoManifestSection,
  frameIdx: number,
  isCond: boolean,
): void {
  const features = slotTensor(backend, video, frameIdx + 0.5);
  const pos = slotTensor(backend, video, -(frameIdx + 0.5));
  bank.commit(frameIdx, isCond, features, pos);
  features.dispose();
  pos.dispose();
}

/** The ring's slot `slot` as a plain number array (byte-level witness). */
function ringSlot(ring: DeviceTensor, video: VideoManifestSection, slot: number): number[] {
  const elements = video.tokensPerMemoryMap * video.memDim;
  const data = (ring as FakeTensor).data as Float32Array;
  return [...data.subarray(slot * elements, (slot + 1) * elements)];
}

const filled = (video: VideoManifestSection, value: number) =>
  new Array(video.tokensPerMemoryMap * video.memDim).fill(value);

describe('MemoryBank construction', () => {
  it('allocates exactly two [M, T, memDim] float32 rings at the requested location', () => {
    const { backend } = makeBank();
    expect(backend.allocCalls).toEqual([
      { shape: [4, 2, 3], dtype: 'float32', location: 'cpu' },
      { shape: [4, 2, 3], dtype: 'float32', location: 'cpu' },
    ]);
    expect(backend.live.size).toBe(2);
  });

  it('uses the real EdgeTAM ring shape [7, 256, 64] for the edgetam section', () => {
    const backend = new FakeBackend();
    const video = edgetamVideo();
    new MemoryBank({ backend, video, strategy: strategyFor('edgetam', video), location: 'device' });
    expect(backend.allocCalls[0]).toEqual({
      shape: [7, 256, 64],
      dtype: 'float32',
      location: 'device',
    });
  });

  it('lays slots out as [0, maxCondFrames) cond then the recent ring, all invalid', () => {
    const { bank } = makeBank();
    expect(bank.slots.map((s) => s.isCond)).toEqual([true, false, false, false]);
    expect(bank.slots.every((s) => !s.valid && s.frameIdx === -1)).toBe(true);
  });

  it("throws NotImplementedError for tposDelivery:'precombined' without leaking rings (⚠ PIN-3)", () => {
    const backend = new FakeBackend();
    const video: VideoManifestSection = { ...tinyVideo(), tposDelivery: 'precombined' };
    expect(
      () =>
        new MemoryBank({
          backend,
          video,
          strategy: strategyFor('edgetam', { ...video, tposDelivery: 'indices' }),
          location: 'cpu',
        }),
    ).toThrow(NotImplementedError);
    expect(backend.live.size).toBe(0);
  });
});

describe('MemoryBank slot selection and eviction', () => {
  it('cond commits land in the cond region (slot 0) and write real bytes to both rings', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    expect(bank.slots[0]).toEqual({ frameIdx: 0, isCond: true, valid: true });
    const asm = bank.assemble(1);
    expect(ringSlot(asm.memorySpatial, video, 0)).toEqual(filled(video, 0.5));
    expect(ringSlot(asm.memorySpatialPos, video, 0)).toEqual(filled(video, -0.5));
  });

  it('recent commits fill the first invalid recent slot, in order', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 1, false);
    commitFrame(backend, bank, video, 2, false);
    commitFrame(backend, bank, video, 3, false);
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([-1, 1, 2, 3]);
    const asm = bank.assemble(4);
    expect(ringSlot(asm.memorySpatial, video, 1)).toEqual(filled(video, 1.5));
    expect(ringSlot(asm.memorySpatial, video, 2)).toEqual(filled(video, 2.5));
    expect(ringSlot(asm.memorySpatial, video, 3)).toEqual(filled(video, 3.5));
  });

  it('a full recent ring evicts the slot with the smallest frameIdx (oldest)', () => {
    const { backend, bank, video } = makeBank();
    for (const f of [1, 2, 3]) commitFrame(backend, bank, video, f, false);
    commitFrame(backend, bank, video, 4, false); // evicts frame 1 (slot 1)
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([-1, 4, 2, 3]);
    const asm = bank.assemble(5);
    expect(ringSlot(asm.memorySpatial, video, 1)).toEqual(filled(video, 4.5));
    commitFrame(backend, bank, video, 5, false); // now frame 2 (slot 2) is oldest
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([-1, 4, 5, 3]);
  });

  it('re-committing the same recent frame refreshes its slot in place', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 1, false);
    commitFrame(backend, bank, video, 2, false);
    commitFrame(backend, bank, video, 1, false); // refresh, not a new slot
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([-1, 1, 2, -1]);
  });

  it('cond region full (EdgeTAM max=1): a new prompt overwrites the single cond slot', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    commitFrame(backend, bank, video, 5, true);
    expect(bank.slots[0]).toEqual({ frameIdx: 5, isCond: true, valid: true });
    const asm = bank.assemble(6);
    expect(ringSlot(asm.memorySpatial, video, 0)).toEqual(filled(video, 5.5));
    // The recent ring is untouched by cond eviction.
    expect(bank.slots.slice(1).every((s) => !s.valid)).toBe(true);
  });

  it('cond commits never spill into the recent ring', () => {
    const { backend, bank, video } = makeBank();
    for (const f of [0, 2, 4]) commitFrame(backend, bank, video, f, true);
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([4, -1, -1, -1]);
  });

  it('rejects wrong-shaped or wrong-dtype memory tensors', () => {
    const { backend, bank, video } = makeBank();
    const wrongShape = backend.uploadTensor(new Float32Array(4), [4], 'float32');
    expect(() => bank.commit(0, true, wrongShape, wrongShape)).toThrow(InvalidStateError);
    const wrongDtype = backend.uploadTensor(
      new Uint8Array(video.tokensPerMemoryMap * video.memDim),
      [video.tokensPerMemoryMap, video.memDim],
      'uint8',
    );
    expect(() => bank.commit(0, true, wrongDtype, wrongDtype)).toThrow(InvalidStateError);
  });

  it('rejects a negative or non-integer frameIdx', () => {
    const { backend, bank, video } = makeBank();
    const t = slotTensor(backend, video, 1);
    expect(() => bank.commit(-1, true, t, t)).toThrow(InvalidStateError);
    expect(() => bank.commit(1.5, false, t, t)).toThrow(InvalidStateError);
  });

  it('does NOT take ownership of committed tensors (caller disposes per §4.5)', () => {
    const { backend, bank, video } = makeBank();
    const features = slotTensor(backend, video, 1);
    const pos = slotTensor(backend, video, -1);
    bank.commit(0, true, features, pos);
    expect((features as FakeTensor).disposed).toBe(false);
    expect((pos as FakeTensor).disposed).toBe(false);
    features.dispose();
    pos.dispose();
  });

  it('issues exactly one copyRegion per ring per commit (the frame step budget)', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    expect(backend.copyRegionCalls.length).toBe(2);
    expect(backend.copyRegionCalls.map((c) => c.slotIndex)).toEqual([0, 0]);
  });
});

describe('MemoryBank temporal-id assignment (tposIndices)', () => {
  it('assigns cond → numRecent and recent recency rank k → k-1 (spec.py rule)', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    for (const f of [1, 2, 3]) commitFrame(backend, bank, video, f, false);
    const asm = bank.assemble(4);
    // Slot order: [cond@0, recent@1, recent@2, recent@3].
    // Ranks (descending frameIdx): 3→1, 2→2, 1→3; tpos = rank-1; cond → 3 (= numRecent).
    expect([...asm.tposIndices]).toEqual([3n, 2n, 1n, 0n]);
  });

  it('ranks by RECENCY among valid slots, not raw frame distance (⚠ PIN-5)', () => {
    const { backend, bank, video } = makeBank();
    // Sparse frames (gaps as after a refine): 2, 7, 9.
    for (const f of [2, 7, 9]) commitFrame(backend, bank, video, f, false);
    const asm = bank.assemble(20);
    // Ranks: 9→1, 7→2, 2→3 regardless of the distance gaps.
    expect([...asm.tposIndices]).toEqual([-1n, 2n, 1n, 0n]);
  });

  it('marks invalid slots with -1', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 1, false);
    const asm = bank.assemble(2);
    expect([...asm.tposIndices]).toEqual([-1n, 0n, -1n, -1n]);
  });
});

describe('MemoryBank streaming rule + hasMemory', () => {
  it('assemble(N) never sees frame N’s own memory (assembly precedes commit)', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 1, false);
    commitFrame(backend, bank, video, 4, false);
    const at4 = bank.assemble(4);
    expect(at4.validMaps).toBe(1); // only frame 1 visible
    expect([...at4.tposIndices]).toEqual([-1n, 0n, -1n, -1n]);
    const at5 = bank.assemble(5);
    expect(at5.validMaps).toBe(2);
    // Ranks: 4→rank1→tpos 0, 1→rank2→tpos 1 (slot 1 holds frame 1, slot 2 frame 4).
    expect([...at5.tposIndices]).toEqual([-1n, 1n, 0n, -1n]);
  });

  it('hasMemory applies the same streaming cutoff', () => {
    const { backend, bank, video } = makeBank();
    expect(bank.hasMemory(0)).toBe(false);
    commitFrame(backend, bank, video, 2, true);
    expect(bank.hasMemory(2)).toBe(false); // own frame does not count
    expect(bank.hasMemory(3)).toBe(true);
  });

  it('an empty bank assembles to validMaps 0 with an all-zero mask and all -1 tpos', () => {
    const { bank, video } = makeBank();
    const asm = bank.assemble(0);
    expect(asm.validMaps).toBe(0);
    expect([...asm.memoryMask]).toEqual(new Array(video.kvLen).fill(0));
    expect([...asm.tposIndices]).toEqual([-1n, -1n, -1n, -1n]);
    expect([...asm.pointerMask]).toEqual([0, 0, 0, 0]);
  });
});

describe('MemoryBank mask-bit assembly (partial banks)', () => {
  it('sets exactly the valid slots’ [i*T, (i+1)*T) spatial ranges, kvLen total', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true); // slot 0
    commitFrame(backend, bank, video, 2, false); // slot 1
    const asm = bank.assemble(3);
    expect(asm.memoryMask.length).toBe(video.kvLen); // 13
    // Slots 0 and 1 valid (T=2 each); slots 2, 3 invalid; ptr region (5) zero.
    expect([...asm.memoryMask]).toEqual([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(asm.validMaps).toBe(2);
  });

  it('pointer region bits flip on iff at least one pointer is visible', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    bank.commitPointer(0, new Float32Array(video.embedDim).fill(7));
    const asm = bank.assemble(1);
    expect([...asm.memoryMask]).toEqual([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
  });

  it('lays out the real EdgeTAM kvLen 1856 = 7×256 + 64', () => {
    const video = edgetamVideo();
    const backend = new FakeBackend();
    const bank = new MemoryBank({
      backend,
      video,
      strategy: strategyFor('edgetam', video),
      location: 'cpu',
    });
    commitFrame(backend, bank, video, 0, true); // slot 0
    commitFrame(backend, bank, video, 1, false); // slot 1
    bank.commitPointer(1, new Float32Array(256));
    const asm = bank.assemble(2);
    expect(asm.memoryMask.length).toBe(1856);
    const on = (from: number, to: number) =>
      asm.memoryMask.subarray(from, to).every((b) => b === 1);
    const off = (from: number, to: number) =>
      asm.memoryMask.subarray(from, to).every((b) => b === 0);
    expect(on(0, 512)).toBe(true); // cond slot 0 + recent slot 1
    expect(off(512, 7 * 256)).toBe(true); // slots 2..6 invalid
    expect(on(7 * 256, 1856)).toBe(true); // ptr region: one pointer visible
    expect([...asm.tposIndices]).toEqual([6n, 0n, -1n, -1n, -1n, -1n, -1n]);
  });
});

describe('MemoryBank pointer ring', () => {
  it('packs pointers most-recent-first with aligned deltas and mask', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    bank.commitPointer(0, Float32Array.from([10, 11]));
    bank.commitPointer(1, Float32Array.from([20, 21]));
    bank.commitPointer(2, Float32Array.from([30, 31]));
    const asm = bank.assemble(3);
    // Most-recent-first: frame 2, 1, 0; fourth row zero-padded.
    expect([...asm.objectPointers]).toEqual([30, 31, 20, 21, 10, 11, 0, 0]);
    expect([...asm.pointerDeltas]).toEqual([1n, 2n, 3n, 0n]);
    expect([...asm.pointerMask]).toEqual([1, 1, 1, 0]);
  });

  it('evicts the oldest pointer beyond maxObjectPointers', () => {
    const { bank, video } = makeBank();
    for (let f = 0; f < video.maxObjectPointers + 1; f++) {
      bank.commitPointer(f, Float32Array.from([f, f]));
    }
    const asm = bank.assemble(10);
    // Frames 1..4 survive (0 evicted); most-recent-first.
    expect([...asm.pointerDeltas]).toEqual([6n, 7n, 8n, 9n]);
    expect([...asm.objectPointers]).toEqual([4, 4, 3, 3, 2, 2, 1, 1]);
    expect([...asm.pointerMask]).toEqual([1, 1, 1, 1]);
  });

  it('applies the streaming rule to pointers (frame N’s pointer invisible at N)', () => {
    const { bank } = makeBank();
    bank.commitPointer(2, Float32Array.from([1, 2]));
    bank.commitPointer(5, Float32Array.from([3, 4]));
    const asm = bank.assemble(5);
    expect([...asm.pointerMask]).toEqual([1, 0, 0, 0]);
    expect([...asm.objectPointers]).toEqual([1, 2, 0, 0, 0, 0, 0, 0]);
    expect([...asm.pointerDeltas]).toEqual([3n, 0n, 0n, 0n]);
  });

  it('replaces a re-committed frame’s pointer in place (refine path)', () => {
    const { bank } = makeBank();
    bank.commitPointer(1, Float32Array.from([1, 1]));
    bank.commitPointer(1, Float32Array.from([9, 9]));
    const asm = bank.assemble(2);
    expect([...asm.objectPointers]).toEqual([9, 9, 0, 0, 0, 0, 0, 0]);
    expect([...asm.pointerMask]).toEqual([1, 0, 0, 0]);
  });

  it('copies the pointer defensively and rejects a wrong-width pointer', () => {
    const { bank } = makeBank();
    const data = Float32Array.from([5, 6]);
    bank.commitPointer(0, data);
    data.fill(0); // caller mutation must not reach the bank
    expect([...bank.assemble(1).objectPointers].slice(0, 2)).toEqual([5, 6]);
    expect(() => bank.commitPointer(1, new Float32Array(3))).toThrow(InvalidStateError);
  });
});

describe('MemoryBank invalidateAfter / reset / dispose', () => {
  it('invalidateAfter drops non-cond slots and pointers strictly after frameIdx; cond is pinned', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    for (const f of [1, 2, 3]) commitFrame(backend, bank, video, f, false);
    for (const f of [0, 1, 2, 3]) bank.commitPointer(f, Float32Array.from([f, f]));
    bank.invalidateAfter(1);
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([0, 1, -1, -1]);
    expect(bank.slots.map((s) => s.valid)).toEqual([true, true, false, false]);
    const asm = bank.assemble(5);
    expect([...asm.pointerMask]).toEqual([1, 1, 0, 0]); // pointers 0, 1 kept
    expect([...asm.pointerDeltas]).toEqual([4n, 5n, 0n, 0n]);
  });

  it('cond slots survive invalidateAfter even when their frameIdx is larger', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 8, true);
    bank.invalidateAfter(2);
    expect(bank.slots[0]).toEqual({ frameIdx: 8, isCond: true, valid: true });
  });

  it('invalidated recent slots are reused first by later commits (post-refine reuse)', () => {
    const { backend, bank, video } = makeBank();
    for (const f of [1, 2, 3]) commitFrame(backend, bank, video, f, false);
    bank.invalidateAfter(1); // slots for frames 2, 3 freed
    commitFrame(backend, bank, video, 6, false);
    // First invalid recent slot (the old frame-2 slot) is reused, no eviction.
    expect(bank.slots.map((s) => s.frameIdx)).toEqual([-1, 1, 6, -1]);
    const asm = bank.assemble(7);
    expect(ringSlot(asm.memorySpatial, video, 2)).toEqual(filled(video, 6.5));
  });

  it('reset invalidates everything but retains the rings (no realloc, no dispose)', () => {
    const { backend, bank, video } = makeBank();
    commitFrame(backend, bank, video, 0, true);
    bank.commitPointer(0, new Float32Array(video.embedDim));
    bank.reset();
    expect(bank.slots.every((s) => !s.valid && s.frameIdx === -1)).toBe(true);
    expect(bank.hasMemory(99)).toBe(false);
    expect(backend.live.size).toBe(2); // rings still alive
    expect(backend.allocCalls.length).toBe(2); // and not reallocated
    commitFrame(backend, bank, video, 1, true); // bank still usable
    expect(bank.hasMemory(2)).toBe(true);
  });

  it('dispose releases both rings; every further use throws InvalidStateError', () => {
    const { backend, bank, video } = makeBank();
    const features = slotTensor(backend, video, 1);
    const pos = slotTensor(backend, video, -1);
    bank.dispose();
    expect(backend.live.size).toBe(2); // only the two undisposed slot tensors remain
    expect(() => bank.commit(0, true, features, pos)).toThrow(InvalidStateError);
    expect(() => bank.assemble(0)).toThrow(InvalidStateError);
    expect(() => bank.hasMemory(0)).toThrow(InvalidStateError);
    expect(() => bank.commitPointer(0, new Float32Array(video.embedDim))).toThrow(
      InvalidStateError,
    );
    expect(() => bank.invalidateAfter(0)).toThrow(InvalidStateError);
    expect(() => bank.reset()).toThrow(InvalidStateError);
    expect(() => bank.dispose()).toThrow(InvalidStateError);
  });
});

describe('multi-object batch offsets (§4.4 engine binding over §1.1 relaxed copyRegion)', () => {
  it('each object’s ring copies contiguously into its batch slot of [B, M*T, memDim]', () => {
    const video = tinyVideo();
    const backend = new FakeBackend();
    const strategy = strategyFor('edgetam', video);
    const bankA = new MemoryBank({ backend, video, strategy, location: 'cpu' });
    const bankB = new MemoryBank({ backend, video, strategy, location: 'cpu' });
    commitFrame(backend, bankA, video, 0, true); // A: slot 0 = 0.5s
    commitFrame(backend, bankB, video, 1, false); // B: slot 1 = 1.5s

    const maps = video.maxCondFrames + video.numRecent;
    const ringElements = maps * video.tokensPerMemoryMap * video.memDim; // 24
    // Engine-owned batched graph input: B=2 objects on the leading dim.
    const batched = backend.allocTensor(
      [2, maps * video.tokensPerMemoryMap, video.memDim],
      'float32',
      'cpu',
    );
    // The §1.1 relaxed rule: a whole [M, T, D] ring is one [M*T, D] batch
    // slot's worth of bytes — contiguous copy, no reshape.
    backend.copyRegion(bankA.assemble(2).memorySpatial, batched, 0);
    backend.copyRegion(bankB.assemble(2).memorySpatial, batched, 1);

    const data = (batched as FakeTensor).data as Float32Array;
    const slotBytes = video.tokensPerMemoryMap * video.memDim; // 6 floats per map
    // Object A at batch offset 0: its cond slot 0 payload, rest zero.
    expect([...data.subarray(0, slotBytes)]).toEqual(filled(video, 0.5));
    expect(data.subarray(slotBytes, ringElements).every((v) => v === 0)).toBe(true);
    // Object B at batch offset ringElements: slot 1 payload at map offset 1.
    const b = ringElements;
    expect(data.subarray(b, b + slotBytes).every((v) => v === 0)).toBe(true);
    expect([...data.subarray(b + slotBytes, b + 2 * slotBytes)]).toEqual(filled(video, 1.5));
    expect(data.subarray(b + 2 * slotBytes, 2 * ringElements).every((v) => v === 0)).toBe(true);
  });

  it('per-object banks stay fully independent (no shared ring state)', () => {
    const video = tinyVideo();
    const backend = new FakeBackend();
    const strategy = strategyFor('edgetam', video);
    const bankA = new MemoryBank({ backend, video, strategy, location: 'cpu' });
    const bankB = new MemoryBank({ backend, video, strategy, location: 'cpu' });
    commitFrame(backend, bankA, video, 0, true);
    expect(bankA.hasMemory(1)).toBe(true);
    expect(bankB.hasMemory(1)).toBe(false);
    expect(bankB.assemble(1).validMaps).toBe(0);
    expect(bankA.assemble(1).memorySpatial).not.toBe(bankB.assemble(1).memorySpatial);
  });
});
