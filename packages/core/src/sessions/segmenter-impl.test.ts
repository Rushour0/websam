import { describe, expect, it, vi } from 'vitest';
import { InvalidStateError, UnsupportedDeviceError } from '../errors.js';
import type { LoadProgressEvent, SegmenterConfig } from '../index.js';
import { registerModel, type ModelSpec } from '../registry.js';
import type { WorkerInitRequest, WorkerInitResult } from '../worker/protocol.js';
import { ImageSessionImpl } from './image-session.js';
import { createSegmenterImpl } from './segmenter-impl.js';
import type { RemoteEngine, WorkerHandle } from './spawn-worker.js';
import { VideoSessionImpl } from './video-session.js';

/** Video RPC stubs for the M1/image-path tests that don't exercise them. */
function videoStubs(): Pick<
  RemoteEngine,
  | 'createVideoSession'
  | 'attachVideoSource'
  | 'addVideoObject'
  | 'refineVideoObject'
  | 'removeVideoObject'
  | 'propagateVideo'
  | 'resetVideoSession'
  | 'closeVideoSession'
> {
  const nope = async (): Promise<never> => {
    throw new Error('video RPC: not under test');
  };
  return {
    createVideoSession: vi.fn(nope),
    attachVideoSource: vi.fn(nope),
    addVideoObject: vi.fn(nope),
    refineVideoObject: vi.fn(nope),
    removeVideoObject: vi.fn(nope),
    propagateVideo: vi.fn(nope),
    resetVideoSession: vi.fn(nope),
    closeVideoSession: vi.fn(nope),
  };
}

// Node has WebAssembly and no navigator.gpu, so the main-thread probes
// resolve `device: 'auto'` to 'wasm' with quantPreference ['int8', 'fp32'].

/** A registered tier the node probes can actually resolve (wasm-capable). */
const WASM_TIER: ModelSpec = {
  id: 'test-wasm-tier',
  displayName: 'Test tier (wasm ok)',
  arch: 'edgetam',
  inputSize: 64,
  supportsVideo: false,
  license: 'apache-2.0',
  manifestUrl: 'https://models.invalid/test-wasm-tier/manifest.json',
  devices: { webgpu: true, wasm: true },
};

/** A tier whose support matrix excludes wasm (the only device node resolves). */
const WEBGPU_ONLY_TIER: ModelSpec = {
  id: 'test-webgpu-only-tier',
  displayName: 'Test tier (webgpu only)',
  arch: 'sam3-tracker',
  inputSize: 64,
  supportsVideo: false,
  license: 'apache-2.0',
  manifestUrl: 'https://models.invalid/test-webgpu-only-tier/manifest.json',
  devices: { webgpu: true, wasm: false },
};

/** A wasm-resolvable, permissive tier that DOES advertise video support. */
const VIDEO_TIER: ModelSpec = {
  id: 'test-video-tier',
  displayName: 'Test tier (video, wasm ok)',
  arch: 'edgetam',
  inputSize: 64,
  supportsVideo: true,
  license: 'apache-2.0',
  manifestUrl: 'https://models.invalid/test-video-tier/manifest.json',
  devices: { webgpu: true, wasm: true },
};

registerModel(WASM_TIER);
registerModel(WEBGPU_ONLY_TIER);
registerModel(VIDEO_TIER);

const INIT_RESULT: WorkerInitResult = { device: 'wasm', quant: 'int8', totalBytes: 4096 };

interface FakeWorker {
  engine: RemoteEngine;
  handle: WorkerHandle;
  spawn: ReturnType<typeof vi.fn> & ((workerUrl?: string | URL) => WorkerHandle);
  terminate: ReturnType<typeof vi.fn>;
}

function fakeWorker(engineOverrides: Partial<RemoteEngine> = {}): FakeWorker {
  const engine: RemoteEngine = {
    init: vi.fn(async () => INIT_RESULT),
    createSession: vi.fn(async () => 7),
    encodeImage: vi.fn(async () => {
      throw new Error('encodeImage: not under test');
    }),
    decode: vi.fn(async () => []),
    closeSession: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
    ...videoStubs(),
    ...engineOverrides,
  };
  const terminate = vi.fn();
  const handle: WorkerHandle = { engine, terminate };
  const spawn = vi.fn((_workerUrl?: string | URL) => handle);
  return { engine, handle, spawn, terminate };
}

function create(config: SegmenterConfig = {}, worker: FakeWorker = fakeWorker()) {
  return createSegmenterImpl({ model: WASM_TIER.id, ...config }, { spawnWorker: worker.spawn });
}

describe('createSegmenterImpl — config validation', () => {
  it('rejects an invalid device value', async () => {
    await expect(
      createSegmenterImpl({ device: 'cuda' as never }),
    ).rejects.toThrow(InvalidStateError);
  });

  it('rejects an invalid quant value', async () => {
    await expect(createSegmenterImpl({ quant: 'fp8' as never })).rejects.toThrow(
      InvalidStateError,
    );
  });

  it("rejects an acceptLicense value other than 'sam'", async () => {
    await expect(
      createSegmenterImpl({ acceptLicense: 'gpl' as never }),
    ).rejects.toThrow(/acceptLicense/);
  });

  it('rejects an unknown model id with InvalidStateError naming the id', async () => {
    await expect(createSegmenterImpl({ model: 'no-such-tier' })).rejects.toThrow(
      InvalidStateError,
    );
    await expect(createSegmenterImpl({ model: 'no-such-tier' })).rejects.toThrow(
      /no-such-tier/,
    );
  });
});

describe('createSegmenterImpl — registry default + license gate', () => {
  it("defaults to 'edgetam' (permissive, wasm-capable) — no license gate", async () => {
    // DEFAULT_MODEL_ID flipped to the Apache-2.0 EdgeTAM tier at M2; the default
    // path resolves on node's wasm device with no acceptLicense required.
    const worker = fakeWorker();
    const segmenter = await createSegmenterImpl({}, { spawnWorker: worker.spawn });
    expect(segmenter.model.spec.id).toBe('edgetam');
  });

  it('license-gated tier (sam3-tracker) rejects without acceptLicense (InvalidStateError names the key)', async () => {
    await expect(createSegmenterImpl({ model: 'sam3-tracker' })).rejects.toThrow(InvalidStateError);
    await expect(createSegmenterImpl({ model: 'sam3-tracker' })).rejects.toThrow(/acceptLicense/);
  });

  it('license-gated tier + acceptLicense passes the gate (fails later on device support)', async () => {
    // sam3-tracker is webgpu-only; node resolves wasm — so passing the license
    // gate surfaces as the UnsupportedDeviceError cross-check, not a license error.
    await expect(createSegmenterImpl({ model: 'sam3-tracker', acceptLicense: 'sam' })).rejects.toThrow(
      UnsupportedDeviceError,
    );
  });

  it('permissive-license tiers need no acceptLicense', async () => {
    const worker = fakeWorker();
    await expect(create({}, worker)).resolves.toBeDefined();
  });
});

describe('createSegmenterImpl — device resolution cross-check', () => {
  it('throws UnsupportedDeviceError when the tier does not support the resolved device', async () => {
    await expect(
      createSegmenterImpl({ model: WEBGPU_ONLY_TIER.id }, { spawnWorker: fakeWorker().spawn }),
    ).rejects.toThrow(UnsupportedDeviceError);
  });

  it("explicit device 'webgpu' throws UnsupportedDeviceError in node (probe not granted)", async () => {
    await expect(create({ device: 'webgpu' })).rejects.toThrow(UnsupportedDeviceError);
  });
});

describe('createSegmenterImpl — worker spawn + init', () => {
  it('spawns with the configured workerUrl and inits with the resolved request', async () => {
    const worker = fakeWorker();
    const url = new URL('https://example.invalid/custom-worker.js');
    await create({ workerUrl: url, modelBaseUrl: 'https://cdn.invalid/m/', wasmPaths: '/ort/' }, worker);

    expect(worker.spawn).toHaveBeenCalledExactlyOnceWith(url);
    expect(worker.engine.init).toHaveBeenCalledOnce();
    const req = vi.mocked(worker.engine.init).mock.calls[0]?.[0] as WorkerInitRequest;
    expect(req.spec.id).toBe(WASM_TIER.id);
    expect(req.device).toBe('wasm');
    expect(req.quantPreference).toEqual(['int8', 'fp32']);
    expect(req.cache).toBe(true);
    expect(req.modelBaseUrl).toBe('https://cdn.invalid/m/');
    expect(req.wasmPaths).toBe('/ort/');
  });

  it('forwards cache: false', async () => {
    const worker = fakeWorker();
    await create({ cache: false }, worker);
    const req = vi.mocked(worker.engine.init).mock.calls[0]?.[0] as WorkerInitRequest;
    expect(req.cache).toBe(false);
  });

  it("proxies onProgress to init and emits {phase:'ready'} after init resolves", async () => {
    const events: LoadProgressEvent[] = [];
    const worker = fakeWorker({
      init: vi.fn(async (_req: WorkerInitRequest, onProgress?: (e: LoadProgressEvent) => void) => {
        onProgress?.({ phase: 'download', loaded: 1, total: 2, file: 'a.onnx' });
        return INIT_RESULT;
      }),
    });
    await create({ onProgress: (e) => events.push(e) }, worker);
    expect(events.map((e) => e.phase)).toEqual(['download', 'ready']);
  });

  it('passes undefined progress when no callback is configured', async () => {
    const worker = fakeWorker();
    await create({}, worker);
    expect(vi.mocked(worker.engine.init).mock.calls[0]?.[1]).toBeUndefined();
  });

  it('terminates the worker and propagates the error when init rejects', async () => {
    const boom = new Error('weights unreachable');
    const worker = fakeWorker({
      init: vi.fn(async () => {
        throw boom;
      }),
    });
    await expect(create({}, worker)).rejects.toBe(boom);
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});

describe('Segmenter facade', () => {
  it('exposes the resolved device and model info from the init result', async () => {
    const segmenter = await create();
    expect(segmenter.device).toBe('wasm');
    expect(segmenter.model.spec.id).toBe(WASM_TIER.id);
    expect(segmenter.model.quant).toBe('int8');
    expect(segmenter.model.totalBytes).toBe(4096);
  });

  it('createImageSession opens a worker session slot and returns an ImageSessionImpl', async () => {
    const worker = fakeWorker();
    const segmenter = await create({}, worker);
    const session = await segmenter.createImageSession();
    expect(session).toBeInstanceOf(ImageSessionImpl);
    expect(worker.engine.createSession).toHaveBeenCalledOnce();
  });

  it('createVideoSession returns a VideoSessionImpl on a video-capable tier', async () => {
    const worker = fakeWorker({ createVideoSession: vi.fn(async () => 42) });
    const segmenter = await create({ model: VIDEO_TIER.id }, worker);
    const session = await segmenter.createVideoSession();
    expect(session).toBeInstanceOf(VideoSessionImpl);
    expect(worker.engine.createVideoSession).toHaveBeenCalledOnce();
  });

  it('createVideoSession throws UnsupportedDeviceError on an image-only tier', async () => {
    // WASM_TIER has supportsVideo: false.
    const segmenter = await create();
    await expect(segmenter.createVideoSession()).rejects.toThrow(UnsupportedDeviceError);
  });

  it('dispose chains engine.dispose → worker.terminate, in that order', async () => {
    const order: string[] = [];
    const worker = fakeWorker({
      dispose: vi.fn(async () => {
        order.push('engine.dispose');
      }),
    });
    worker.terminate.mockImplementation(() => order.push('terminate'));
    const segmenter = await create({}, worker);
    await segmenter.dispose();
    expect(order).toEqual(['engine.dispose', 'terminate']);
  });

  it('terminates even when engine.dispose rejects', async () => {
    const worker = fakeWorker({
      dispose: vi.fn(async () => {
        throw new Error('worker already gone');
      }),
    });
    const segmenter = await create({}, worker);
    await expect(segmenter.dispose()).rejects.toThrow('worker already gone');
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('double-dispose is a no-op', async () => {
    const worker = fakeWorker();
    const segmenter = await create({}, worker);
    await segmenter.dispose();
    await segmenter.dispose();
    expect(worker.engine.dispose).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('use-after-dispose throws InvalidStateError', async () => {
    const segmenter = await create();
    await segmenter.dispose();
    await expect(segmenter.createImageSession()).rejects.toThrow(InvalidStateError);
    await expect(segmenter.createVideoSession()).rejects.toThrow(InvalidStateError);
  });
});
