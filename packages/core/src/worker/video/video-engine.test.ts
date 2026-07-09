import { afterEach, beforeEach, describe, expect, it } from 'vitest';
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
import type { ModelSpec } from '../../registry.js';
import type { DecodedFrame, FrameRange, FrameSource, VideoSourceInfo } from '../../video/frame-source.js';
import type { GraphManifestEntry, ModelManifest, VideoManifestSection } from '../../weights/manifest.js';
import { VideoEngine, type VideoEngineGraphs } from './video-engine.js';

// ---------------------------------------------------------------------------
// bitmapToTensor needs OffscreenCanvas (browser-only); stub a minimal one so
// the M2 video path can run in node unit tests. Pixel content is irrelevant
// here — these tests assert SEQUENCING and the dispose/census schedule, not
// preprocessing numerics (already covered by preprocess.test.ts's pure math).
// ---------------------------------------------------------------------------
class FakeCanvasContext {
  imageSmoothingEnabled = true;
  imageSmoothingQuality = 'high';
  drawImage(..._args: unknown[]): void {}
  getImageData(_x: number, _y: number, w: number, h: number): { data: Uint8ClampedArray } {
    return { data: new Uint8ClampedArray(w * h * 4) };
  }
}
class FakeOffscreenCanvas {
  constructor(
    public width: number,
    public height: number,
  ) {}
  getContext(_kind: string, _opts?: unknown): FakeCanvasContext {
    return new FakeCanvasContext();
  }
}

// ---------------------------------------------------------------------------
// FakeBackend: mirrors memory-bank.test.ts's FakeBackend (cpu tensors,
// debugStats census). ADDITIONALLY tracks a SEPARATE "session outputs" set —
// real backends explicitly exclude BackendSession run() outputs from
// debugStats() ("owned by their BackendSession"), so this harness models
// that split faithfully: only alloc/uploadTensor calls count toward
// debugStats(), matching backend.ts's contract.
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
  readonly sessionOutputsAlive = new Set<FakeTensor>();
  readonly copyRegionCalls: number[] = [];

  async init(): Promise<void> {}
  async createSession(_g: GraphAsset, _p?: IOBindingPlan): Promise<BackendSession> {
    throw new NotImplementedError('FakeBackend.createSession');
  }

  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    const t = new FakeTensor([...shape], dtype, location, (x) => this.live.delete(x));
    this.live.add(t);
    return t;
  }

  uploadTensor(_data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor {
    const t = new FakeTensor([...shape], dtype, 'cpu', (x) => this.live.delete(x));
    this.live.add(t);
    return t;
  }

  /** Used only by {@link FakeSession} to mint outputs OUTSIDE the backend's own census. */
  makeOutputTensor(shape: readonly number[], dtype: DType, location: TensorLocation = 'cpu'): FakeTensor {
    const t = new FakeTensor([...shape], dtype, location, (x) => this.sessionOutputsAlive.delete(x));
    this.sessionOutputsAlive.add(t);
    return t;
  }

  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    const slotElems = elemCount(dst.shape.slice(1));
    if (elemCount(src.shape) !== slotElems) throw new InvalidStateError('copyRegion byte-count mismatch');
    if (src.dtype !== dst.dtype) throw new InvalidStateError('copyRegion dtype mismatch');
    if (slotIndex < 0 || slotIndex >= (dst.shape[0] ?? 0)) throw new InvalidStateError('copyRegion slot OOB');
    this.copyRegionCalls.push(slotIndex);
  }

  reshape(tensor: DeviceTensor, shape: readonly number[]): DeviceTensor {
    // Non-owning view: not tracked in `live`, dispose is a no-op — never
    // perturbs the census-flatness assertions.
    return new FakeTensor([...shape], tensor.dtype, tensor.location, () => {});
  }

  async readback(tensor: DeviceTensor): Promise<ArrayBufferView> {
    return new Float32Array(elemCount(tensor.shape)).fill(0);
  }

  debugStats(): { liveTensors: number; liveBytes: number } {
    let liveBytes = 0;
    for (const t of this.live) liveBytes += elemCount(t.shape) * 4;
    return { liveTensors: this.live.size, liveBytes };
  }

  async dispose(): Promise<void> {
    for (const t of [...this.live]) t.dispose();
  }
}

/** A `BackendSession` whose outputs are scripted per semantic role. */
class FakeSession implements BackendSession {
  calls = 0;
  constructor(
    private readonly backend: FakeBackend,
    private readonly makeOutputs: (
      feeds: Record<string, DeviceTensor>,
      callIndex: number,
    ) => Record<string, DeviceTensor>,
  ) {}
  async run(feeds: Record<string, DeviceTensor>): Promise<Record<string, DeviceTensor>> {
    this.calls += 1;
    return this.makeOutputs(feeds, this.calls);
  }
  async dispose(): Promise<void> {}
}

// ---------------------------------------------------------------------------
// A tiny-but-faithful EdgeTAM-shaped video manifest section (same convention
// as memory-bank.test.ts: small token/dim sizes so byte math stays legible).
// ---------------------------------------------------------------------------

const T = 2; // tokensPerMemoryMap
const MEM_DIM = 1;
const EMBED_DIM = 4;
const M = 7; // maxCondFrames(1) + numRecent(6)
const PTR_TOKENS = 64;
const MAX_PTRS = 16;
const GRID = 1; // logit grid side = GRID*4 = 4

function videoSection(overrides: Partial<VideoManifestSection> = {}): VideoManifestSection {
  return {
    maxCondFrames: 1,
    numRecent: 6,
    tokensPerMemoryMap: T,
    ptrTokens: PTR_TOKENS,
    maxObjectPointers: MAX_PTRS,
    kvLen: M * T + PTR_TOKENS,
    memDim: MEM_DIM,
    embedDim: EMBED_DIM,
    gridSize: GRID,
    multiObjectBatch: true,
    initPath: 'noMemGraph',
    tposDelivery: 'indices',
    occlusionThreshold: 0,
    ...overrides,
  };
}

function tensorSpec(name: string, dtype: DType, shape: (number | string)[]): { name: string; dtype: DType; shape: (number | string)[] } {
  return { name, dtype, shape };
}

function graphEntry(inputs: Record<string, string>, outputs: Record<string, string>): GraphManifestEntry {
  const spec = (n: string): ReturnType<typeof tensorSpec> => tensorSpec(n, 'float32', []);
  return {
    files: { fp32: { path: 'x', sha256: '0'.repeat(64), bytes: 1 } },
    inputs: Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, spec(v)])),
    outputs: Object.fromEntries(Object.entries(outputs).map(([k, v]) => [k, spec(v)])),
  };
}

function manifest(video: VideoManifestSection): ModelManifest {
  const graphs: ModelManifest['graphs'] = {
    videoEncoder: graphEntry(
      { pixels: 'pixels' },
      { visionFeatures: 'visionFeatures', visionPos: 'visionPos', highRes0: 'highRes0', highRes1: 'highRes1' },
    ),
    memoryAttention: graphEntry(
      {
        queries: 'queries',
        queriesPos: 'queriesPos',
        memorySpatial: 'memorySpatial',
        memorySpatialPos: 'memorySpatialPos',
        tposIndices: 'tposIndices',
        memoryMask: 'memoryMask',
        objectPointers: 'objectPointers',
        pointerDeltas: 'pointerDeltas',
        pointerMask: 'pointerMask',
        noMem: 'noMem',
      },
      { conditionedFeatures: 'conditionedFeatures' },
    ),
    maskDecoderVideo: graphEntry(
      { conditionedFeatures: 'conditionedFeatures', highRes0: 'highRes0', highRes1: 'highRes1', points: 'points', labels: 'labels' },
      { maskLogits: 'maskLogits', iouScores: 'iouScores', objectPointer: 'objectPointer', objectScoreLogits: 'objectScoreLogits' },
    ),
    memoryEncoder: graphEntry(
      { visionFeatures: 'visionFeatures', maskLogits: 'maskLogits', isPrompted: 'is_prompted' },
      { memoryFeatures: 'memoryFeatures', memoryPos: 'memoryPos' },
    ),
    noMemCondition: graphEntry({ visionFeatures: 'visionFeatures' }, { conditionedFeatures: 'conditionedFeatures' }),
  };
  return {
    schemaVersion: 1,
    tier: 'edgetam',
    opset: 18,
    graphs,
    toolchain: { exporter: 'test' },
    preprocess: { mode: 'square-stretch', inputSize: 4, mean: [0, 0, 0], std: [1, 1, 1], maskSize: GRID * 4 },
    video,
  };
}

const spec: ModelSpec = {
  id: 'edgetam',
  displayName: 'EdgeTAM',
  arch: 'edgetam',
  inputSize: 4,
  supportsVideo: true,
  license: 'apache-2.0',
  manifestUrl: 'https://example.test/manifest.json',
  devices: { webgpu: true, wasm: true },
};

// ---------------------------------------------------------------------------
// FakeFrameSource: deterministic frames, closes tracked.
// ---------------------------------------------------------------------------

class FakeVideoFrame {
  closed = false;
  close(): void {
    if (this.closed) throw new InvalidStateError('double close');
    this.closed = true;
  }
}

class FakeFrameSource implements FrameSource {
  readonly info: VideoSourceInfo;
  closeCalls = 0;
  constructor(frameCount: number) {
    this.info = { frameCount, fps: 1, width: 4, height: 4, durationUs: frameCount * 1e6, codec: 'test' };
  }
  async frameAt(frameIndex: number): Promise<VideoFrame> {
    if (frameIndex < 0 || frameIndex >= this.info.frameCount) throw new InvalidStateError('OOB');
    return new FakeVideoFrame() as unknown as VideoFrame;
  }
  async *read(range?: Partial<FrameRange>): AsyncIterableIterator<DecodedFrame> {
    const start = range?.startFrame ?? 0;
    const end = range?.endFrame ?? this.info.frameCount;
    for (let i = start; i < end; i++) {
      yield { frame: new FakeVideoFrame() as unknown as VideoFrame, frameIndex: i, timestampUs: i * 1e6 };
    }
  }
  async close(): Promise<void> {
    this.closeCalls += 1;
  }
}

// ---------------------------------------------------------------------------
// Wiring: build a VideoEngine whose four (+noMemCondition) graphs are
// FakeSessions with deterministic, cheaply-scripted outputs.
// ---------------------------------------------------------------------------

function buildGraphs(
  backend: FakeBackend,
  video: VideoManifestSection,
): { graphs: VideoEngineGraphs; sessions: { videoEncoder: FakeSession; memoryAttention: FakeSession; maskDecoderVideo: FakeSession; memoryEncoder: FakeSession; noMemCondition: FakeSession } } {
  const g = GRID;
  const videoEncoder = new FakeSession(backend, () => ({
    visionFeatures: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
    visionPos: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
    highRes0: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
    highRes1: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
  }));
  const memoryAttention = new FakeSession(backend, () => ({
    conditionedFeatures: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
  }));
  const maskDecoderVideo = new FakeSession(backend, () => ({
    maskLogits: backend.makeOutputTensor([1, 1, g * 4, g * 4], 'float32', 'cpu'),
    iouScores: backend.makeOutputTensor([1, 3], 'float32', 'cpu'),
    objectPointer: backend.makeOutputTensor([1, video.embedDim], 'float32', 'cpu'),
    // occlusion is driven by `occlusionThreshold` vs FakeBackend.readback's
    // constant-0 objectScoreLogits (see the dedicated occlusion test) — no
    // per-frame scripting needed here.
    objectScoreLogits: backend.makeOutputTensor([1, 1], 'float32', 'cpu'),
  }));
  const memoryEncoder = new FakeSession(backend, () => ({
    memoryFeatures: backend.makeOutputTensor([video.tokensPerMemoryMap, video.memDim], 'float32', 'cpu'),
    memoryPos: backend.makeOutputTensor([video.tokensPerMemoryMap, video.memDim], 'float32', 'cpu'),
  }));
  const noMemCondition = new FakeSession(backend, () => ({
    conditionedFeatures: backend.makeOutputTensor([1, 1, g, g], 'float32', 'cpu'),
  }));
  return {
    graphs: { videoEncoder, memoryAttention, maskDecoderVideo, memoryEncoder, noMemCondition },
    sessions: { videoEncoder, memoryAttention, maskDecoderVideo, memoryEncoder, noMemCondition },
  };
}

describe('VideoEngine', () => {
  let realOffscreenCanvas: unknown;

  beforeEach(() => {
    realOffscreenCanvas = (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas;
    (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas = FakeOffscreenCanvas;
  });
  afterEach(() => {
    (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas = realOffscreenCanvas;
  });

  it('rejects a manifest with no video section', () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const m = manifest(video);
    delete m.video;
    const { graphs } = buildGraphs(backend, video);
    expect(() => new VideoEngine({ backend, manifest: m, spec, graphs })).toThrow(InvalidStateError);
  });

  it("requires a noMemCondition graph when initPath is 'noMemGraph'", () => {
    const backend = new FakeBackend();
    const video = videoSection({ initPath: 'noMemGraph' });
    const { graphs } = buildGraphs(backend, video);
    const g2 = { ...graphs, noMemCondition: undefined };
    expect(() => new VideoEngine({ backend, manifest: manifest(video), spec, graphs: g2 })).toThrow(InvalidStateError);
  });

  it('addObject on the first frame uses the init (no-memory) path, not memoryAttention', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs, sessions } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(5));

    const result = await engine.addObject({ frameIndex: 0, prompts: [{ type: 'point', x: 1, y: 1, label: 1 }] });

    expect(result.objectId).toBe(1);
    expect(sessions.noMemCondition.calls).toBe(1);
    expect(sessions.memoryAttention.calls).toBe(0);
    expect(sessions.videoEncoder.calls).toBe(1);
    expect(sessions.maskDecoderVideo.calls).toBe(1);
    expect(sessions.memoryEncoder.calls).toBe(1);
    // The mask is a real 4x4 (srcW*srcH) binary buffer, transferred as an ArrayBuffer.
    expect(result.mask.binaryMask.byteLength).toBe(4 * 4);
  });

  it('auto-assigns sequential objectIds and rejects duplicates', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(5));

    const a = await engine.addObject({ frameIndex: 0, prompts: [] });
    const b = await engine.addObject({ frameIndex: 0, prompts: [] });
    expect(a.objectId).toBe(1);
    expect(b.objectId).toBe(2);
    await expect(engine.addObject({ frameIndex: 0, prompts: [], objectId: 1 })).rejects.toThrow(InvalidStateError);
  });

  it('propagate runs memoryAttention (not the init path) once memory exists, in frameIndex order', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs, sessions } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(6));
    await engine.addObject({ frameIndex: 0, prompts: [{ type: 'point', x: 1, y: 1, label: 1 }] });

    const seen: number[] = [];
    await engine.propagate(
      { startFrame: 1, endFrame: 5, epoch: 7 },
      async (frame) => {
        seen.push(frame.frameIndex);
        expect(frame.epoch).toBe(7);
        expect(frame.masks).toHaveLength(1);
        expect(frame.masks[0]?.objectId).toBe(1);
      },
      () => false,
    );

    expect(seen).toEqual([1, 2, 3, 4]);
    // 1 no-mem call from addObject; every propagated frame has memory (the
    // cond frame committed at addObject time), so all 4 go through memoryAttention.
    expect(sessions.noMemCondition.calls).toBe(1);
    expect(sessions.memoryAttention.calls).toBe(4);
    expect(sessions.videoEncoder.calls).toBe(5); // 1 (addObject) + 4 (propagate)
  });

  it('GPU-memory flatness (§4.5): backend.debugStats() is identical at every frame boundary from frame 3 on', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(10));
    await engine.addObject({ frameIndex: 0, prompts: [{ type: 'point', x: 1, y: 1, label: 1 }] });

    const statsAtFrame: { liveTensors: number; liveBytes: number }[] = [];
    await engine.propagate(
      { startFrame: 1, endFrame: 9, epoch: 0 },
      async () => {
        statsAtFrame.push(backend.debugStats());
      },
      () => false,
    );

    expect(statsAtFrame.length).toBe(8);
    const fromFrame3 = statsAtFrame.slice(2); // frame index 3 onward (1-based: 3rd propagated frame)
    for (const s of fromFrame3) {
      expect(s).toEqual(fromFrame3[0]);
    }
    // No leaked session-run-output tensors either: everything the fake
    // sessions minted was disposed by the engine each frame.
    expect(backend.sessionOutputsAlive.size).toBe(0);
  });

  it('every run-output tensor is disposed after each addObject call (no leaks)', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(3));
    await engine.addObject({ frameIndex: 0, prompts: [] });
    expect(backend.sessionOutputsAlive.size).toBe(0);
  });

  it('occlusion gating: EdgeTAM (commitOccludedMemory=true, e2e_loop.py divergence 5) still commits an occluded frame', async () => {
    const backend = new FakeBackend();
    // occlusionThreshold > 0 trips on FakeBackend's constant-0 readback of
    // objectScoreLogits, so every propagated frame looks "occluded" —
    // arch-strategy.ts's EdgetamStrategy.commitOccludedMemory is `true`
    // (the e2e loop never gates the memory encoder on object_score_logits),
    // so #encodeAndCommit's `shouldCommit` must still be true: 2 more
    // copyRegion calls (memoryFeatures + memoryPos ring writes) per frame.
    const video = videoSection({ occlusionThreshold: 1 });
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(5));
    await engine.addObject({ frameIndex: 0, prompts: [{ type: 'point', x: 1, y: 1, label: 1 }] });
    const before = backend.copyRegionCalls.length;

    await engine.propagate({ startFrame: 1, endFrame: 3, epoch: 0 }, async () => {}, () => false);

    // 2 frames propagated * 2 copyRegion calls each (commit writes both rings).
    expect(backend.copyRegionCalls.length).toBe(before + 4);
  });

  it("throws NotImplementedError for tposDelivery: 'precombined'", async () => {
    const backend = new FakeBackend();
    const video = videoSection({ tposDelivery: 'precombined' });
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(3));
    await expect(engine.addObject({ frameIndex: 0, prompts: [] })).rejects.toThrow(NotImplementedError);
  });

  it('removeObject disposes the bank and forgets the id', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(3));
    const { objectId } = await engine.addObject({ frameIndex: 0, prompts: [] });
    const before = backend.debugStats().liveTensors;
    engine.removeObject(objectId);
    expect(backend.debugStats().liveTensors).toBeLessThan(before);
    // Idempotent-ish: removing an unknown/already-removed id is a no-op, not a throw.
    expect(() => engine.removeObject(objectId)).not.toThrow();
  });

  it('propagate stops (without emitting) once isCancelled() is true', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    engine.attach(new FakeFrameSource(8));
    await engine.addObject({ frameIndex: 0, prompts: [] });

    const seen: number[] = [];
    await engine.propagate(
      { startFrame: 1, endFrame: 6, epoch: 0 },
      async (f) => {
        seen.push(f.frameIndex);
      },
      () => seen.length >= 2,
    );
    expect(seen).toEqual([1, 2]);
  });

  it('dispose() releases every object bank and closes the source', async () => {
    const backend = new FakeBackend();
    const video = videoSection();
    const { graphs } = buildGraphs(backend, video);
    const engine = new VideoEngine({ backend, manifest: manifest(video), spec, graphs });
    const source = new FakeFrameSource(3);
    engine.attach(source);
    await engine.addObject({ frameIndex: 0, prompts: [] });

    await engine.dispose();
    expect(source.closeCalls).toBe(1);
    expect(backend.debugStats().liveTensors).toBe(0);
    await expect(engine.addObject({ frameIndex: 0, prompts: [] })).rejects.toThrow(InvalidStateError);
  });
});
