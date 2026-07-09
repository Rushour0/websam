/**
 * Unit tests for the M2 memory-bank backend primitives (backend-video §1.1):
 * device/cpu `allocTensor`, `copyRegion` slot arithmetic + validation, and
 * the `debugStats` tensor census. ort itself is a structural fake — the
 * WASM path exercises real typed-array bytes on CPU, and the WebGPU path
 * asserts the command-encoder copy arithmetic against a fake GPU device.
 * Real onnxruntime-web (and a real adapter) runs in
 * memory-primitives.browser.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DeviceTensor } from './backend.js';
import { InvalidStateError, OutOfMemoryError } from '../errors.js';
import { OrtDeviceTensor } from '../runtime/ort-tensor.js';
import { WasmBackend } from './wasm-backend.js';
import { WebGpuBackend, type OrtModule } from './webgpu-backend.js';

type OrtTensor = import('onnxruntime-web').Tensor;

interface CopyCall {
  src: FakeGpuBuffer;
  srcOffset: number;
  dst: FakeGpuBuffer;
  dstOffset: number;
  size: number;
}

class FakeGpuBuffer {
  destroyCalls = 0;
  constructor(
    readonly size: number,
    readonly usage: number,
  ) {}
  destroy(): void {
    this.destroyCalls += 1;
  }
}

class FakeCommandEncoder {
  constructor(private readonly log: CopyCall[]) {}
  copyBufferToBuffer(
    src: FakeGpuBuffer,
    srcOffset: number,
    dst: FakeGpuBuffer,
    dstOffset: number,
    size: number,
  ): void {
    this.log.push({ src, srcOffset, dst, dstOffset, size });
  }
  finish(): unknown {
    return { commandBuffer: this.log.length };
  }
}

class FakeGpuDevice {
  readonly buffers: FakeGpuBuffer[] = [];
  readonly copies: CopyCall[] = [];
  readonly submitted: unknown[][] = [];
  failNextCreate = false;
  createBuffer(descriptor: { size: number; usage: number }): FakeGpuBuffer {
    if (this.failNextCreate) throw new Error('simulated GPU OOM');
    const buffer = new FakeGpuBuffer(descriptor.size, descriptor.usage);
    this.buffers.push(buffer);
    return buffer;
  }
  createCommandEncoder(): FakeCommandEncoder {
    return new FakeCommandEncoder(this.copies);
  }
  readonly queue = {
    submit: (commandBuffers: unknown[]): void => {
      this.submitted.push(commandBuffers);
    },
  };
}

class FakeOrtTensor {
  location = 'cpu';
  disposeCalls = 0;
  gpuBuffer: FakeGpuBuffer | undefined;
  constructor(
    readonly type: string,
    readonly data: unknown,
    readonly dims: readonly number[] = [],
  ) {}
  static fromGpuBuffer(
    buffer: FakeGpuBuffer,
    options: { dataType: string; dims: readonly number[] },
  ): FakeOrtTensor {
    const tensor = new FakeOrtTensor(options.dataType, new Uint8Array(0), options.dims);
    tensor.location = 'gpu-buffer';
    tensor.gpuBuffer = buffer;
    return tensor;
  }
  dispose(): void {
    this.disposeCalls += 1;
    this.location = 'none';
  }
  async getData(): Promise<unknown> {
    return this.data;
  }
}

interface Fake {
  ort: OrtModule;
  device: FakeGpuDevice;
}

function makeFakeOrt(): Fake {
  const device = new FakeGpuDevice();
  const ort = {
    Tensor: FakeOrtTensor,
    env: { webgpu: { device } },
    InferenceSession: {
      create: async (): Promise<{ run(): Promise<unknown>; release(): Promise<void> }> => ({
        run: async () => ({}),
        release: async () => undefined,
      }),
    },
  } as unknown as OrtModule;
  return { ort, device };
}

function stubGpu(): void {
  vi.stubGlobal('navigator', {
    gpu: { requestAdapter: async () => ({ features: new Set<string>() }) },
  });
}

/** Read the backing typed array of an ort-fake cpu tensor. */
function dataOf(tensor: DeviceTensor): unknown {
  return (tensor as OrtDeviceTensor).ortTensor.data;
}

describe('WasmBackend memory primitives (cpu path — real bytes)', () => {
  let backend: WasmBackend;

  beforeEach(async () => {
    backend = new WasmBackend(makeFakeOrt().ort);
    await backend.init();
  });

  it("allocTensor('device') degrades to a zeroed cpu tensor", () => {
    const tensor = backend.allocTensor([3, 4], 'float32', 'device');
    expect(tensor.location).toBe('cpu');
    expect(tensor.dtype).toBe('float32');
    expect(dataOf(tensor)).toEqual(new Float32Array(12));
  });

  it('copyRegion writes src into the right slot offset via typed-array set', () => {
    const dst = backend.allocTensor([3, 4], 'float32', 'device'); // 3 slots of 4
    const src = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [4], 'float32');
    backend.copyRegion(src, dst, 1);
    expect(Array.from(dataOf(dst) as Float32Array)).toEqual([0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0]);
  });

  it('copyRegion honors the relaxed byte-count rule (shape need not equal one slot)', () => {
    const dst = backend.allocTensor([2, 2, 3], 'float32', 'cpu'); // 2 slots of 6
    const src = backend.uploadTensor(Float32Array.from([9, 8, 7, 6, 5, 4]), [6], 'float32'); // flat
    backend.copyRegion(src, dst, 0);
    expect(Array.from(dataOf(dst) as Float32Array).slice(0, 6)).toEqual([9, 8, 7, 6, 5, 4]);
  });

  it('copyRegion of int64 slots uses BigInt64Array.set', () => {
    const dst = backend.allocTensor([2, 2], 'int64', 'cpu');
    const src = backend.uploadTensor(BigInt64Array.from([5n, 6n]), [2], 'int64');
    backend.copyRegion(src, dst, 1);
    expect(Array.from(dataOf(dst) as BigInt64Array)).toEqual([0n, 0n, 5n, 6n]);
  });

  describe('copyRegion validation (InvalidStateError)', () => {
    it('rejects an out-of-bounds slot index', () => {
      const dst = backend.allocTensor([3, 4], 'float32', 'cpu');
      const src = backend.uploadTensor(new Float32Array(4), [4], 'float32');
      expect(() => backend.copyRegion(src, dst, 3)).toThrow(InvalidStateError);
      expect(() => backend.copyRegion(src, dst, -1)).toThrow(InvalidStateError);
    });

    it('rejects a dtype mismatch', () => {
      const dst = backend.allocTensor([3, 4], 'float32', 'cpu');
      const src = backend.uploadTensor(new Int32Array(4), [4], 'int32');
      expect(() => backend.copyRegion(src, dst, 0)).toThrow(InvalidStateError);
    });

    it('rejects a src element count that is not exactly one slot', () => {
      const dst = backend.allocTensor([3, 4], 'float32', 'cpu');
      const src = backend.uploadTensor(new Float32Array(2), [2], 'float32');
      expect(() => backend.copyRegion(src, dst, 0)).toThrow(InvalidStateError);
    });

    it('rejects a scalar (no slot axis) destination', () => {
      const dst = backend.allocTensor([], 'float32', 'cpu');
      const src = backend.uploadTensor(new Float32Array(1), [1], 'float32');
      expect(() => backend.copyRegion(src, dst, 0)).toThrow(InvalidStateError);
    });
  });

  describe('debugStats census', () => {
    it('starts at zero', () => {
      expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
    });

    it('counts uploaded + allocated tensors and their aggregate bytes', () => {
      backend.uploadTensor(new Float32Array(4), [2, 2], 'float32'); // 16 B
      backend.allocTensor([10], 'float32', 'cpu'); // 40 B
      backend.uploadTensor(BigInt64Array.from([1n]), [1], 'int64'); // 8 B
      expect(backend.debugStats()).toEqual({ liveTensors: 3, liveBytes: 64 });
    });

    it('decrements when a tensor is disposed', () => {
      const a = backend.uploadTensor(new Float32Array(4), [2, 2], 'float32');
      backend.allocTensor([10], 'float32', 'cpu');
      a.dispose();
      expect(backend.debugStats()).toEqual({ liveTensors: 1, liveBytes: 40 });
    });
  });

  it('every primitive throws InvalidStateError before init()', () => {
    const fresh = new WasmBackend(makeFakeOrt().ort);
    expect(() => fresh.allocTensor([1], 'float32', 'device')).toThrow(InvalidStateError);
    expect(() => fresh.copyRegion({} as DeviceTensor, {} as DeviceTensor, 0)).toThrow(
      InvalidStateError,
    );
    // The census is safe to read at any lifecycle point (empty before init).
    expect(fresh.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
  });
});

describe('WebGpuBackend memory primitives (device path — fake GPU device)', () => {
  let fake: Fake;
  let backend: WebGpuBackend;

  beforeEach(async () => {
    stubGpu();
    fake = makeFakeOrt();
    backend = new WebGpuBackend(fake.ort);
    await backend.init();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("allocTensor('device') creates a zeroed GPUBuffer and wraps it as a device tensor", () => {
    const tensor = backend.allocTensor([4, 3], 'float32', 'device');
    expect(tensor.location).toBe('device');
    expect(tensor.shape).toEqual([4, 3]);
    expect(fake.device.buffers).toHaveLength(1);
    // STORAGE | COPY_SRC | COPY_DST, size = 4*3*4 = 48 B (already 4-aligned).
    expect(fake.device.buffers[0]?.size).toBe(48);
    expect(fake.device.buffers[0]?.usage).toBe(0x0080 | 0x0004 | 0x0008);
  });

  it("allocTensor('device') pads sub-4-byte tensors up to a 4-byte multiple", () => {
    backend.allocTensor([3], 'uint8', 'device'); // 3 B → padded to 4
    expect(fake.device.buffers[0]?.size).toBe(4);
  });

  it("allocTensor('cpu') still returns a zeroed cpu tensor (no GPU buffer)", () => {
    const tensor = backend.allocTensor([2, 2], 'float32', 'cpu');
    expect(tensor.location).toBe('cpu');
    expect(fake.device.buffers).toHaveLength(0);
  });

  it("allocTensor('device') throws OutOfMemoryError when buffer creation fails", () => {
    fake.device.failNextCreate = true;
    expect(() => backend.allocTensor([4, 3], 'float32', 'device')).toThrow(OutOfMemoryError);
  });

  it('copyRegion issues a copyBufferToBuffer at the correct slot offset + size', () => {
    const dst = backend.allocTensor([4, 3], 'float32', 'device'); // slot = 3 floats = 12 B
    const src = backend.allocTensor([3], 'float32', 'device');
    backend.copyRegion(src, dst, 2);
    expect(fake.device.copies).toHaveLength(1);
    const copy = fake.device.copies[0];
    expect(copy?.srcOffset).toBe(0);
    expect(copy?.dstOffset).toBe(24); // slotIndex 2 * 12 B
    expect(copy?.size).toBe(12);
    expect(fake.device.submitted).toHaveLength(1);
  });

  it('copyRegion rejects a cpu-located operand (upload-first contract)', () => {
    const dst = backend.allocTensor([4, 3], 'float32', 'device');
    const cpuSrc = backend.uploadTensor(new Float32Array(3), [3], 'float32');
    expect(() => backend.copyRegion(cpuSrc, dst, 0)).toThrow(InvalidStateError);
    // Validation ran but no copy was issued.
    expect(fake.device.copies).toHaveLength(0);
  });

  it('copyRegion validates the byte-count rule before touching the device', () => {
    const dst = backend.allocTensor([4, 3], 'float32', 'device');
    const src = backend.allocTensor([2], 'float32', 'device'); // wrong slot size
    expect(() => backend.copyRegion(src, dst, 0)).toThrow(InvalidStateError);
    expect(fake.device.copies).toHaveLength(0);
  });

  it('disposing a device tensor destroys its GPUBuffer and untracks it in the census', () => {
    const tensor = backend.allocTensor([4, 3], 'float32', 'device'); // 48 B
    expect(backend.debugStats()).toEqual({ liveTensors: 1, liveBytes: 48 });
    const buffer = fake.device.buffers[0];
    tensor.dispose();
    expect(buffer?.destroyCalls).toBe(1);
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
  });

  it('debugStats counts a mix of cpu and device tensors', () => {
    backend.allocTensor([4, 3], 'float32', 'device'); // 48 B
    backend.uploadTensor(new Float32Array(4), [2, 2], 'float32'); // 16 B
    expect(backend.debugStats()).toEqual({ liveTensors: 2, liveBytes: 64 });
  });

  it('dispose sweeps device tensors (destroying their buffers) and resets the census', async () => {
    backend.allocTensor([4, 3], 'float32', 'device');
    backend.allocTensor([2], 'float32', 'device');
    await backend.dispose();
    expect(fake.device.buffers.every((b) => b.destroyCalls === 1)).toBe(true);
  });
});

describe('OrtDeviceTensor value import stays wired', () => {
  it('is the concrete wrapper both backends produce', () => {
    const backend = new WasmBackend(makeFakeOrt().ort);
    return backend.init().then(() => {
      const tensor = backend.allocTensor([1], 'float32', 'cpu');
      expect(tensor).toBeInstanceOf(OrtDeviceTensor);
      const ort = (tensor as OrtDeviceTensor).ortTensor as unknown as OrtTensor;
      expect(ort).toBeDefined();
    });
  });
});
