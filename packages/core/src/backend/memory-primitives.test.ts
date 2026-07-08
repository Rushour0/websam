/**
 * Unit tests for the M2 backend memory primitives: real `copyRegion`
 * (typed-array region copy on wasm, buffer-to-buffer command on webgpu),
 * device-located `allocTensor`, and the `debugStats()` tensor census. ort
 * and the GPUDevice are structural fakes — the wasm fakes carry real bytes
 * so slot arithmetic is asserted on data, and the fake GPU device applies
 * `copyBufferToBuffer` at submit so the webgpu offsets are asserted on
 * bytes too. Real-ort coverage lives in memory-primitives.browser.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { InvalidStateError, OutOfMemoryError } from '../errors.js';
import { OrtDeviceTensor } from '../runtime/ort-tensor.js';
import type { DeviceTensor } from './backend.js';
import { WasmBackend } from './wasm-backend.js';
import { WebGpuBackend, type OrtModule } from './webgpu-backend.js';

/* ----------------------------- fake WebGPU ------------------------------ */

class FakeGpuBuffer {
  readonly data: Uint8Array;
  destroyed = false;
  constructor(
    readonly size: number,
    readonly usage: number,
    readonly label?: string,
  ) {
    this.data = new Uint8Array(size); // zero-initialized, like real WebGPU
  }
  destroy(): void {
    this.destroyed = true;
  }
  async mapAsync(_mode: number): Promise<void> {}
  getMappedRange(): ArrayBuffer {
    return this.data.buffer as ArrayBuffer;
  }
  unmap(): void {}
}

interface RecordedCopy {
  src: FakeGpuBuffer;
  srcOffset: number;
  dst: FakeGpuBuffer;
  dstOffset: number;
  size: number;
}

/** Byte-accurate GPUDevice fake: copies apply at queue.submit, like real WebGPU. */
class FakeGpuDevice {
  buffers: FakeGpuBuffer[] = [];
  copies: RecordedCopy[] = [];
  submits = 0;
  failNextCreateBuffer = false;

  createBuffer(descriptor: { size: number; usage: number; label?: string }): FakeGpuBuffer {
    if (this.failNextCreateBuffer) {
      this.failNextCreateBuffer = false;
      throw new Error('fake device out of memory');
    }
    const buffer = new FakeGpuBuffer(descriptor.size, descriptor.usage, descriptor.label);
    this.buffers.push(buffer);
    return buffer;
  }

  createCommandEncoder(): {
    copyBufferToBuffer: (
      src: FakeGpuBuffer,
      srcOffset: number,
      dst: FakeGpuBuffer,
      dstOffset: number,
      size: number,
    ) => void;
    finish: () => object;
  } {
    const pending: (() => void)[] = [];
    const record = this.copies;
    return {
      copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => {
        record.push({ src, srcOffset, dst, dstOffset, size });
        pending.push(() => dst.data.set(src.data.subarray(srcOffset, srcOffset + size), dstOffset));
      },
      finish: () => ({ pending }),
    };
  }

  readonly queue = {
    submit: (commandBuffers: readonly object[]): void => {
      this.submits += 1;
      for (const commands of commandBuffers) {
        for (const apply of (commands as { pending: (() => void)[] }).pending) {
          apply();
        }
      }
    },
  };
}

/* ------------------------------- fake ort -------------------------------- */

class FakeOrtTensor {
  location = 'cpu';
  disposeCalls = 0;
  gpuBuffer: FakeGpuBuffer | undefined;
  #disposer: (() => void) | undefined;
  #downloader: (() => Promise<unknown>) | undefined;

  constructor(
    readonly type: string,
    public data: unknown,
    readonly dims: readonly number[] = [],
  ) {}

  static fromGpuBuffer(
    buffer: FakeGpuBuffer,
    options: {
      dataType: string;
      dims: readonly number[];
      download?: () => Promise<unknown>;
      dispose?: () => void;
    },
  ): FakeOrtTensor {
    const tensor = new FakeOrtTensor(options.dataType, undefined, options.dims);
    tensor.location = 'gpu-buffer';
    tensor.gpuBuffer = buffer;
    tensor.#disposer = options.dispose;
    tensor.#downloader = options.download;
    return tensor;
  }

  dispose(): void {
    this.disposeCalls += 1;
    this.#disposer?.();
    this.location = 'none';
  }

  async getData(): Promise<unknown> {
    if (this.#downloader) {
      const data = await this.#downloader();
      // Mirrors ort: a download pins the handle to cpu afterwards.
      this.location = 'cpu';
      this.data = data;
      this.#downloader = undefined;
      return data;
    }
    return this.data;
  }
}

class FakeInferenceSession {
  releaseCalls = 0;
  async run(): Promise<Record<string, FakeOrtTensor>> {
    return { out: new FakeOrtTensor('float32', Float32Array.from([3]), [1]) };
  }
  async release(): Promise<void> {
    this.releaseCalls += 1;
  }
}

interface FakeOrt {
  ort: OrtModule;
  device: FakeGpuDevice;
  sessions: FakeInferenceSession[];
}

function makeFakeOrt(): FakeOrt {
  const device = new FakeGpuDevice();
  const sessions: FakeInferenceSession[] = [];
  const ort = {
    Tensor: FakeOrtTensor,
    env: { webgpu: { device: Promise.resolve(device) } },
    InferenceSession: {
      create: async () => {
        const session = new FakeInferenceSession();
        sessions.push(session);
        return session;
      },
    },
  } as unknown as OrtModule;
  return { ort, device, sessions };
}

function stubGpu(): void {
  vi.stubGlobal('navigator', {
    gpu: { requestAdapter: async () => ({ features: new Set(['shader-f16']) }) },
  });
}

const graphBytes = Uint8Array.from([8, 18]);

afterEach(() => {
  vi.unstubAllGlobals();
});

/* --------------------------- WasmBackend (cpu) --------------------------- */

describe('WasmBackend M2 memory primitives (cpu bytes)', () => {
  let backend: WasmBackend;

  beforeEach(async () => {
    backend = new WasmBackend(makeFakeOrt().ort);
    await backend.init();
  });

  const dataOf = (tensor: DeviceTensor): unknown => (tensor as OrtDeviceTensor).ortTensor.data;

  it('copyRegion writes exactly one slot of the ring', () => {
    const dst = backend.allocTensor([3, 2, 2], 'float32', 'cpu');
    const src = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [2, 2], 'float32');

    backend.copyRegion(src, dst, 1);

    expect(dataOf(dst)).toEqual(Float32Array.from([0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0]));
  });

  it('accepts byte-count-equal shapes (reshape-free): a flat [4] src fills a [2,2] slot', () => {
    // The engine's batched-KV case in miniature: ring rows copy into a
    // differently-shaped batch slot with the same element count.
    const dst = backend.allocTensor([2, 2, 2], 'float32', 'cpu');
    const src = backend.uploadTensor(Float32Array.from([5, 6, 7, 8]), [4], 'float32');

    backend.copyRegion(src, dst, 0);

    expect(dataOf(dst)).toEqual(Float32Array.from([5, 6, 7, 8, 0, 0, 0, 0]));
  });

  it('copies first and last slots at the right offsets (int64 path)', () => {
    const dst = backend.allocTensor([3, 2], 'int64', 'cpu');
    const first = backend.uploadTensor(BigInt64Array.from([1n, 2n]), [2], 'int64');
    const last = backend.uploadTensor(BigInt64Array.from([8n, 9n]), [2], 'int64');

    backend.copyRegion(first, dst, 0);
    backend.copyRegion(last, dst, 2);

    expect(dataOf(dst)).toEqual(BigInt64Array.from([1n, 2n, 0n, 0n, 8n, 9n]));
  });

  it('overwriting an occupied slot replaces its contents', () => {
    const dst = backend.allocTensor([2, 2], 'float32', 'cpu');
    const a = backend.uploadTensor(Float32Array.from([1, 1]), [2], 'float32');
    const b = backend.uploadTensor(Float32Array.from([2, 2]), [2], 'float32');

    backend.copyRegion(a, dst, 1);
    backend.copyRegion(b, dst, 1);

    expect(dataOf(dst)).toEqual(Float32Array.from([0, 0, 2, 2]));
  });

  it('validates dtype, element count, slot bounds, and operand liveness', () => {
    const dst = backend.allocTensor([3, 2, 2], 'float32', 'cpu');
    const src = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [2, 2], 'float32');

    const wrongDtype = backend.uploadTensor(Int32Array.from([1, 2, 3, 4]), [2, 2], 'int32');
    expect(() => backend.copyRegion(wrongDtype, dst, 0)).toThrow(InvalidStateError);

    const wrongCount = backend.uploadTensor(Float32Array.from([1, 2]), [2], 'float32');
    expect(() => backend.copyRegion(wrongCount, dst, 0)).toThrow(InvalidStateError);

    for (const slotIndex of [-1, 3, 1.5, Number.NaN]) {
      expect(() => backend.copyRegion(src, dst, slotIndex)).toThrow(InvalidStateError);
    }

    expect(() => backend.copyRegion({} as DeviceTensor, dst, 0)).toThrow(InvalidStateError);
    expect(() => backend.copyRegion(src, {} as DeviceTensor, 0)).toThrow(InvalidStateError);

    const disposed = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [2, 2], 'float32');
    disposed.dispose();
    expect(() => backend.copyRegion(disposed, dst, 0)).toThrow(InvalidStateError);
    expect(() => backend.copyRegion(src, disposed, 0)).toThrow(InvalidStateError);
  });

  it('throws InvalidStateError before init()', () => {
    const fresh = new WasmBackend(makeFakeOrt().ort);
    expect(() => fresh.copyRegion({} as DeviceTensor, {} as DeviceTensor, 0)).toThrow(
      InvalidStateError,
    );
    expect(() => fresh.allocTensor([1], 'float32', 'device')).toThrow(InvalidStateError);
  });

  it("allocTensor('device') degrades to a zeroed cpu tensor ('device' === 'cpu' on wasm)", () => {
    const tensor = backend.allocTensor([2, 3], 'float32', 'device');
    expect(tensor.location).toBe('cpu');
    expect(tensor.shape).toEqual([2, 3]);
    expect(dataOf(tensor)).toEqual(new Float32Array(6));
  });
});

/* ------------------------- WebGpuBackend (device) ------------------------ */

describe('WebGpuBackend M2 memory primitives (fake GPUDevice)', () => {
  let fake: FakeOrt;
  let backend: WebGpuBackend;

  beforeEach(async () => {
    stubGpu();
    fake = makeFakeOrt();
    backend = new WebGpuBackend(fake.ort);
    await backend.init();
  });

  const allocDevice = async (): Promise<void> => {
    // ort owns the GPUDevice; it becomes readable after the first session.
    await backend.createSession({ name: 'g', bytes: graphBytes });
  };

  it("allocTensor('device') requires ort's device (first session) — InvalidStateError before", () => {
    expect(() => backend.allocTensor([2, 2], 'float32', 'device')).toThrow(InvalidStateError);
  });

  it("allocTensor('device') creates a zeroed, device-located GPUBuffer tensor", async () => {
    await allocDevice();
    const tensor = backend.allocTensor([2, 3], 'float32', 'device');

    expect(tensor.location).toBe('device');
    expect(tensor.dtype).toBe('float32');
    expect(tensor.shape).toEqual([2, 3]);

    const buffer = fake.device.buffers.at(-1)!;
    expect(buffer.size).toBe(32); // 24 logical bytes padded to 16-byte multiple
    expect(buffer.usage & 0x0080).toBeTruthy(); // STORAGE
    expect(buffer.usage & 0x0004).toBeTruthy(); // COPY_SRC
    expect(buffer.usage & 0x0008).toBeTruthy(); // COPY_DST

    const view = (await backend.readback(tensor)) as Float32Array;
    expect(view).toEqual(new Float32Array(6));
  });

  it("allocTensor('device') dispose destroys the GPUBuffer; failures map to OutOfMemoryError", async () => {
    await allocDevice();
    const tensor = backend.allocTensor([4], 'float32', 'device');
    const buffer = fake.device.buffers.at(-1)!;
    tensor.dispose();
    expect(buffer.destroyed).toBe(true);

    fake.device.failNextCreateBuffer = true;
    expect(() => backend.allocTensor([4], 'float32', 'device')).toThrow(OutOfMemoryError);

    expect(() => backend.allocTensor([2, -1], 'float32', 'device')).toThrow(RangeError);
  });

  it('copyRegion issues one contiguous copyBufferToBuffer at slotIndex * slotBytes', async () => {
    await allocDevice();
    const dst = backend.allocTensor([3, 2, 2], 'float32', 'device'); // slotBytes = 16
    const src = backend.allocTensor([2, 2], 'float32', 'device');
    const srcBuffer = fake.device.buffers.at(-1)!;
    const pattern = Float32Array.from([1, 2, 3, 4]);
    srcBuffer.data.set(new Uint8Array(pattern.buffer)); // poke device bytes directly

    backend.copyRegion(src, dst, 2);

    const copy = fake.device.copies.at(-1)!;
    expect(copy.srcOffset).toBe(0);
    expect(copy.dstOffset).toBe(2 * 16);
    expect(copy.size).toBe(16);
    expect(fake.device.submits).toBeGreaterThan(0);

    const view = (await backend.readback(dst)) as Float32Array;
    expect(view).toEqual(Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]));
  });

  it("copyRegion rejects 'cpu'-located operands (upload first) and misaligned slots", async () => {
    await allocDevice();
    const dst = backend.allocTensor([3, 2, 2], 'float32', 'device');
    const cpuSrc = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [2, 2], 'float32');
    expect(() => backend.copyRegion(cpuSrc, dst, 0)).toThrow(InvalidStateError);

    const cpuDst = backend.allocTensor([3, 2, 2], 'float32', 'cpu');
    const devSrc = backend.allocTensor([2, 2], 'float32', 'device');
    expect(() => backend.copyRegion(devSrc, cpuDst, 0)).toThrow(InvalidStateError);

    // uint8 slot of 2 bytes violates WebGPU's 4-byte copy alignment.
    const narrowDst = backend.allocTensor([2, 2], 'uint8', 'device');
    const narrowSrc = backend.allocTensor([2], 'uint8', 'device');
    expect(() => backend.copyRegion(narrowSrc, narrowDst, 1)).toThrow(InvalidStateError);
  });
});

/* ------------------------------ debugStats ------------------------------- */

describe.each([
  ['WasmBackend', (ort: OrtModule) => new WasmBackend(ort)],
  ['WebGpuBackend', (ort: OrtModule) => new WebGpuBackend(ort)],
] as const)('%s debugStats() census', (_name, make) => {
  let fake: FakeOrt;
  let backend: WasmBackend | WebGpuBackend;

  beforeEach(async () => {
    stubGpu();
    fake = makeFakeOrt();
    backend = make(fake.ort);
    await backend.init();
  });

  it('counts alloc/upload tensors with logical byte sizes and tracks dispose', () => {
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });

    const a = backend.allocTensor([2, 3], 'float32', 'cpu'); // 24 B
    const b = backend.uploadTensor(BigInt64Array.from([1n, 2n]), [2], 'int64'); // 16 B
    const c = backend.uploadTensor(Uint8Array.from([1]), [1], 'bool'); // 1 B
    expect(backend.debugStats()).toEqual({ liveTensors: 3, liveBytes: 41 });

    b.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 2, liveBytes: 25 });
    a.dispose();
    c.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
  });

  it('counts run() outputs and prunes them once the caller disposes', async () => {
    const session = await backend.createSession({ name: 'g', bytes: graphBytes });
    const outputs = await session.run({}); // fake returns one float32 [1] output
    expect(backend.debugStats()).toEqual({ liveTensors: 1, liveBytes: 4 });

    outputs['out']!.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
  });

  it('reports zeros after backend.dispose() (callable in any state)', async () => {
    backend.allocTensor([8], 'float32', 'cpu');
    const session = await backend.createSession({ name: 'g', bytes: graphBytes });
    await session.run({});
    expect(backend.debugStats().liveTensors).toBe(2);

    await backend.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
  });

  it('dispose() sweeps live run outputs without double-disposing caller-disposed ones', async () => {
    const session = await backend.createSession({ name: 'g', bytes: graphBytes });
    const first = await session.run({});
    const second = await session.run({});
    first['out']!.dispose(); // caller-disposed; stays in the census until pruned

    await backend.dispose(); // must not throw on the already-disposed output

    expect((second['out'] as OrtDeviceTensor).disposed).toBe(true);
    const firstOrt = (first['out'] as OrtDeviceTensor).ortTensor as unknown as FakeOrtTensor;
    expect(firstOrt.disposeCalls).toBe(1);
  });
});

/* ----------------- census wrap keeps M1 run semantics ----------------- */

describe('CensusOrtBackendSession run semantics', () => {
  it('still rejects foreign feeds and returns OrtDeviceTensor outputs', async () => {
    const fake = makeFakeOrt();
    const backend = new WasmBackend(fake.ort);
    await backend.init();
    const session = await backend.createSession({ name: 'g', bytes: graphBytes });

    await expect(session.run({ x: {} as DeviceTensor })).rejects.toBeInstanceOf(
      InvalidStateError,
    );

    const outputs = await session.run({});
    expect(outputs['out']).toBeInstanceOf(OrtDeviceTensor);
    expect((outputs['out'] as OrtDeviceTensor).ortTensor).toBeInstanceOf(FakeOrtTensor);
  });
});
