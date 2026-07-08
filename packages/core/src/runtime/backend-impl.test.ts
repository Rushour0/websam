/**
 * Unit tests for the M1 backend bodies (construction, state guards, error
 * paths, tracking). ort itself is a structural fake here — real inference
 * runs in backend-impl.browser.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Backend, DeviceTensor } from '../backend/backend.js';
import { WasmBackend } from '../backend/wasm-backend.js';
import { WebGpuBackend, type OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError, NotImplementedError } from '../errors.js';
import { OrtDeviceTensor } from './ort-tensor.js';

type OrtTensor = import('onnxruntime-web').Tensor;

class FakeOrtTensor {
  location = 'cpu';
  disposeCalls = 0;
  constructor(
    readonly type: string,
    readonly data: unknown,
    readonly dims: readonly number[] = [],
  ) {}
  dispose(): void {
    this.disposeCalls += 1;
    this.location = 'none';
  }
  async getData(): Promise<unknown> {
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
  created: { bytes: Uint8Array; options: Record<string, unknown> }[];
  sessions: FakeInferenceSession[];
}

function makeFakeOrt(): FakeOrt {
  const created: FakeOrt['created'] = [];
  const sessions: FakeInferenceSession[] = [];
  const ort = {
    Tensor: FakeOrtTensor,
    InferenceSession: {
      create: async (bytes: Uint8Array, options: Record<string, unknown>) => {
        created.push({ bytes, options });
        const session = new FakeInferenceSession();
        sessions.push(session);
        return session;
      },
    },
  } as unknown as OrtModule;
  return { ort, created, sessions };
}

function stubGpu(f16 = true): void {
  vi.stubGlobal('navigator', {
    gpu: {
      requestAdapter: async () => ({ features: new Set(f16 ? ['shader-f16'] : []) }),
    },
  });
}

const graphBytes = Uint8Array.from([8, 18]);

describe.each([
  ['WasmBackend', 'wasm', (ort: OrtModule): Backend => new WasmBackend(ort)],
  ['WebGpuBackend', 'webgpu', (ort: OrtModule): Backend => new WebGpuBackend(ort)],
] as const)('%s M1 bodies', (_name, kind, make) => {
  let fake: FakeOrt;
  let backend: Backend;

  beforeEach(() => {
    stubGpu();
    fake = makeFakeOrt();
    backend = make(fake.ort);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it(`has kind '${kind}'`, () => {
    expect(backend.kind).toBe(kind);
  });

  describe('before init()', () => {
    it('every M1 method throws InvalidStateError; copyRegion stays NotImplementedError', async () => {
      await expect(backend.createSession({ name: 'g', bytes: graphBytes })).rejects.toBeInstanceOf(
        InvalidStateError,
      );
      expect(() => backend.uploadTensor(new Float32Array(1), [1], 'float32')).toThrow(
        InvalidStateError,
      );
      expect(() => backend.allocTensor([1], 'float32', 'cpu')).toThrow(InvalidStateError);
      await expect(backend.readback({} as DeviceTensor)).rejects.toBeInstanceOf(InvalidStateError);
      await expect(backend.dispose()).rejects.toBeInstanceOf(InvalidStateError);
      expect(() =>
        backend.copyRegion({} as DeviceTensor, {} as DeviceTensor, 0),
      ).toThrow(NotImplementedError);
    });
  });

  describe('after init()', () => {
    beforeEach(async () => {
      await backend.init();
    });

    it('createSession compiles graph.bytes on the right execution provider', async () => {
      const session = await backend.createSession({ name: 'visionEncoder', bytes: graphBytes });
      expect(session).toBeDefined();
      expect(fake.created).toHaveLength(1);
      expect(fake.created[0]?.bytes).toBe(graphBytes);
      expect(fake.created[0]?.options['executionProviders']).toEqual([kind]);
    });

    it('createSession rejects url graphs with NotImplementedError (M2)', async () => {
      await expect(
        backend.createSession({ name: 'g', url: 'https://example.com/g.onnx' }),
      ).rejects.toBeInstanceOf(NotImplementedError);
    });

    it('uploadTensor creates a cpu-located tensor over the given host data', () => {
      const data = BigInt64Array.from([1n, 0n]);
      const tensor = backend.uploadTensor(data, [1, 1, 2], 'int64');
      expect(tensor.location).toBe('cpu');
      expect(tensor.dtype).toBe('int64');
      expect(tensor.shape).toEqual([1, 1, 2]);
      expect((tensor as OrtDeviceTensor).ortTensor.data).toBe(data);
    });

    it("allocTensor('cpu') returns a zeroed tensor; 'device' stays NotImplementedError (M2)", () => {
      const tensor = backend.allocTensor([2, 2], 'float32', 'cpu');
      expect(tensor.location).toBe('cpu');
      expect((tensor as OrtDeviceTensor).ortTensor.data).toEqual(new Float32Array(4));
      expect(() => backend.allocTensor([2, 2], 'float32', 'device')).toThrow(NotImplementedError);
    });

    it('copyRegion still throws NotImplementedError (M2 memory-bank primitive)', () => {
      const t = backend.allocTensor([1], 'float32', 'cpu');
      expect(() => backend.copyRegion(t, t, 0)).toThrow(NotImplementedError);
    });

    it('readback returns a view over cpu data and getData() for device tensors', async () => {
      const data = Float32Array.from([1, 2]);
      const cpuTensor = backend.uploadTensor(data, [2], 'float32');
      await expect(backend.readback(cpuTensor)).resolves.toBe(data);

      const onGpu = new FakeOrtTensor('float32', Float32Array.from([9]), [1]);
      onGpu.location = 'gpu-buffer';
      const getData = vi.spyOn(onGpu, 'getData');
      const deviceTensor = OrtDeviceTensor.wrap(onGpu as unknown as OrtTensor);
      await expect(backend.readback(deviceTensor)).resolves.toEqual(Float32Array.from([9]));
      expect(getData).toHaveBeenCalledOnce();
    });

    it('readback rejects foreign and disposed tensors with InvalidStateError', async () => {
      await expect(backend.readback({} as DeviceTensor)).rejects.toBeInstanceOf(InvalidStateError);
      const tensor = backend.uploadTensor(new Float32Array(1), [1], 'float32');
      tensor.dispose();
      await expect(backend.readback(tensor)).rejects.toBeInstanceOf(InvalidStateError);
    });

    it('dispose sweeps tracked tensors + sessions and resets initialized', async () => {
      const tensor = backend.uploadTensor(new Float32Array(1), [1], 'float32');
      const alloc = backend.allocTensor([1], 'float32', 'cpu');
      await backend.createSession({ name: 'g', bytes: graphBytes });

      await backend.dispose();

      expect((tensor as OrtDeviceTensor).disposed).toBe(true);
      expect(((tensor as OrtDeviceTensor).ortTensor as unknown as FakeOrtTensor).disposeCalls).toBe(1);
      expect((alloc as OrtDeviceTensor).disposed).toBe(true);
      expect(fake.sessions[0]?.releaseCalls).toBe(1);
      // Reset to the uninitialized state: everything (incl. a second dispose) throws again.
      expect((backend as WasmBackend | WebGpuBackend).initialized).toBe(false);
      await expect(backend.dispose()).rejects.toBeInstanceOf(InvalidStateError);
      expect(() => backend.uploadTensor(new Float32Array(1), [1], 'float32')).toThrow(
        InvalidStateError,
      );
    });

    it('dispose skips tensors and sessions the caller already released', async () => {
      const tensor = backend.uploadTensor(new Float32Array(1), [1], 'float32');
      const session = await backend.createSession({ name: 'g', bytes: graphBytes });
      tensor.dispose();
      await session.dispose();

      await backend.dispose(); // must not double-dispose either

      expect(((tensor as OrtDeviceTensor).ortTensor as unknown as FakeOrtTensor).disposeCalls).toBe(1);
      expect(fake.sessions[0]?.releaseCalls).toBe(1);
    });
  });
});

describe('WebGpuBackend ioPlan mapping', () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards the plan as preferredOutputLocation ('device' → 'gpu-buffer') — device is ort's, never injected", async () => {
    stubGpu();
    const fake = makeFakeOrt();
    const backend = new WebGpuBackend(fake.ort);
    await backend.init();
    await backend.createSession(
      { name: 'visionEncoder', bytes: graphBytes },
      { outputLocations: { 'image_embeddings.0': 'device', iou_scores: 'cpu' } },
    );
    expect(fake.created[0]?.options['preferredOutputLocation']).toEqual({
      'image_embeddings.0': 'gpu-buffer',
      iou_scores: 'cpu',
    });
  });

  it('init discovers shader-f16 on the adapter', async () => {
    stubGpu(false);
    const fake = makeFakeOrt();
    const backend = new WebGpuBackend(fake.ort);
    await backend.init();
    expect(backend.features.f16).toBe(false);
  });
});

describe('WasmBackend ioPlan handling', () => {
  it('ignores any ioPlan — everything is cpu on wasm', async () => {
    const fake = makeFakeOrt();
    const backend = new WasmBackend(fake.ort);
    await backend.init();
    await backend.createSession(
      { name: 'g', bytes: graphBytes },
      { outputLocations: { out: 'device' } },
    );
    expect(fake.created[0]?.options).toEqual({ executionProviders: ['wasm'] });
  });
});
