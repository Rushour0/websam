import { describe, expect, it, vi } from 'vitest';
import type { DeviceTensor } from '../backend/backend.js';
import type { OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError } from '../errors.js';
import { createOrtSession, OrtBackendSession } from './ort-session.js';
import { createCpuTensor } from './ort-tensor.js';

type OrtInferenceSession = import('onnxruntime-web').InferenceSession;

class FakeOrtTensor {
  location = 'cpu';
  constructor(
    readonly type: string,
    readonly data: unknown,
    readonly dims: readonly number[] = [],
  ) {}
  dispose(): void {}
  async getData(): Promise<unknown> {
    return this.data;
  }
}

class FakeInferenceSession {
  releaseCalls = 0;
  runCalls: { feeds: Record<string, unknown>; fetches: readonly string[] | undefined }[] = [];
  results: Record<string, FakeOrtTensor> = {
    out: new FakeOrtTensor('float32', Float32Array.from([3]), [1]),
  };

  async run(
    feeds: Record<string, unknown>,
    fetches?: readonly string[],
  ): Promise<Record<string, FakeOrtTensor>> {
    this.runCalls.push({ feeds, fetches });
    return this.results;
  }

  async release(): Promise<void> {
    this.releaseCalls += 1;
  }
}

function makeFakeOrt(): {
  ort: OrtModule;
  created: { bytes: Uint8Array; options: Record<string, unknown> }[];
  sessions: FakeInferenceSession[];
} {
  const created: { bytes: Uint8Array; options: Record<string, unknown> }[] = [];
  const sessions: FakeInferenceSession[] = [];
  const ort = {
    Tensor: FakeOrtTensor,
    InferenceSession: {
      create: vi.fn(async (bytes: Uint8Array, options: Record<string, unknown>) => {
        created.push({ bytes, options });
        const session = new FakeInferenceSession();
        sessions.push(session);
        return session;
      }),
    },
  } as unknown as OrtModule;
  return { ort, created, sessions };
}

const bytes = Uint8Array.from([1, 2, 3]);

describe('createOrtSession', () => {
  it("webgpu: uses the ['webgpu'] provider and maps the ioPlan to preferredOutputLocation", async () => {
    const { ort, created } = makeFakeOrt();
    await createOrtSession(ort, 'webgpu', bytes, {
      ioPlan: { outputLocations: { embed0: 'device', maskLogits: 'cpu' } },
    });
    expect(created).toHaveLength(1);
    expect(created[0]?.bytes).toBe(bytes);
    expect(created[0]?.options).toEqual({
      executionProviders: ['webgpu'],
      preferredOutputLocation: { embed0: 'gpu-buffer', maskLogits: 'cpu' },
    });
  });

  it('webgpu: omits preferredOutputLocation without an ioPlan', async () => {
    const { ort, created } = makeFakeOrt();
    await createOrtSession(ort, 'webgpu', bytes);
    expect(created[0]?.options).toEqual({ executionProviders: ['webgpu'] });
  });

  it("wasm: uses the ['wasm'] provider and ignores any ioPlan (everything is cpu)", async () => {
    const { ort, created } = makeFakeOrt();
    await createOrtSession(ort, 'wasm', bytes, {
      ioPlan: { outputLocations: { embed0: 'device' } },
    });
    expect(created[0]?.options).toEqual({ executionProviders: ['wasm'] });
  });
});

describe('OrtBackendSession.run', () => {
  it('unwraps OrtDeviceTensor feeds, forwards fetches, and wraps outputs', async () => {
    const { ort } = makeFakeOrt();
    const inner = new FakeInferenceSession();
    const session = new OrtBackendSession(inner as unknown as OrtInferenceSession);

    const a = createCpuTensor(ort, Float32Array.from([1]), [1], 'float32');
    const outputs = await session.run({ a }, ['out']);

    expect(inner.runCalls).toHaveLength(1);
    expect(inner.runCalls[0]?.feeds['a']).toBe(a.ortTensor);
    expect(inner.runCalls[0]?.fetches).toEqual(['out']);

    const out = outputs['out'];
    expect(out).toBeDefined();
    expect(out?.dtype).toBe('float32');
    expect(out?.shape).toEqual([1]);
    expect(out?.location).toBe('cpu');
  });

  it('omits the fetches argument when none is given (defaults to all outputs)', async () => {
    const { ort } = makeFakeOrt();
    const inner = new FakeInferenceSession();
    const session = new OrtBackendSession(inner as unknown as OrtInferenceSession);
    const a = createCpuTensor(ort, Float32Array.from([1]), [1], 'float32');
    await session.run({ a });
    expect(inner.runCalls[0]?.fetches).toBeUndefined();
  });

  it('rejects feeds that are not OrtDeviceTensor instances', async () => {
    const inner = new FakeInferenceSession();
    const session = new OrtBackendSession(inner as unknown as OrtInferenceSession);
    const foreign = { shape: [1], dtype: 'float32', location: 'cpu', dispose() {} } as DeviceTensor;
    await expect(session.run({ x: foreign })).rejects.toBeInstanceOf(InvalidStateError);
    expect(inner.runCalls).toHaveLength(0);
  });

  it('rejects run() on a disposed session', async () => {
    const { ort } = makeFakeOrt();
    const inner = new FakeInferenceSession();
    const session = new OrtBackendSession(inner as unknown as OrtInferenceSession);
    await session.dispose();
    const a = createCpuTensor(ort, Float32Array.from([1]), [1], 'float32');
    await expect(session.run({ a })).rejects.toBeInstanceOf(InvalidStateError);
  });
});

describe('OrtBackendSession.dispose', () => {
  it('releases the ort session once, notifies onDispose, and is idempotent', async () => {
    const inner = new FakeInferenceSession();
    const onDispose = vi.fn();
    const session = new OrtBackendSession(inner as unknown as OrtInferenceSession, onDispose);
    await session.dispose();
    await session.dispose();
    expect(inner.releaseCalls).toBe(1);
    expect(session.disposed).toBe(true);
    expect(onDispose).toHaveBeenCalledExactlyOnceWith(session);
  });
});
