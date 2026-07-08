import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { InvalidStateError, WeightVerifyError } from '../errors.js';
import type { LoadProgressEvent } from '../index.js';
import type { ModelSpec } from '../registry.js';
import { loadModelAssets, type LoadAssetsConfig } from './load-model-assets.js';
import type { Quant } from './manifest.js';
import { MemoryWeightStore } from './weight-store.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();
const ENCODER_BYTES = encoder.encode('vision-encoder-onnx-bytes-'.repeat(20));
const DECODER_BYTES = encoder.encode('prompt-decoder-onnx-bytes-'.repeat(10));

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

const SPEC: ModelSpec = {
  id: 'test-tier',
  displayName: 'Test Tier',
  arch: 'sam3-tracker',
  inputSize: 1008,
  supportsVideo: false,
  license: 'apache-2.0',
  manifestUrl: 'https://models.example.test/test-tier/manifest.json',
  devices: { webgpu: true, wasm: true },
};

const ENCODER_PATH = 'vision_encoder.onnx';
const DECODER_PATH = 'prompt_decoder.onnx';

function fileRef(path: string, bytes: Uint8Array): { path: string; sha256: string; bytes: number } {
  return { path, sha256: sha256(bytes), bytes: bytes.byteLength };
}

/** Manifest JSON where fp32 covers both roles and fp16 covers only the encoder. */
function manifestJson(): Record<string, unknown> {
  return {
    schemaVersion: 1,
    tier: SPEC.id,
    opset: 18,
    graphs: {
      visionEncoder: {
        files: {
          fp32: fileRef(ENCODER_PATH, ENCODER_BYTES),
          fp16: fileRef('vision_encoder_fp16.onnx', ENCODER_BYTES),
        },
        inputs: { pixels: { name: 'pixel_values', dtype: 'float32', shape: ['batch_size', 3, 1008, 1008] } },
        outputs: { embed0: { name: 'image_embeddings.0', dtype: 'float32', shape: ['batch_size', 32, 288, 288] } },
      },
      promptDecoder: {
        files: { fp32: fileRef(DECODER_PATH, DECODER_BYTES) },
        inputs: { points: { name: 'input_points', dtype: 'float32', shape: ['batch_size', 1, 'num_points', 2] } },
        outputs: { maskLogits: { name: 'pred_masks', dtype: 'float32', shape: ['batch_size', 'num', 3, 288, 288] } },
      },
    },
    toolchain: { exporter: 'test' },
    preprocess: { mode: 'square-stretch', inputSize: 1008, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], maskSize: 288 },
  };
}

function streamOf(...chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller): void {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

interface RecordedCall {
  url: string;
  init: RequestInit | undefined;
}

/** Wire up a fetch mock: url → handler(callIndexForThatUrl, init) → Response. */
function mockFetch(
  routes: Record<string, (call: number, init?: RequestInit) => Response | Promise<Response>>,
): { fetchImpl: typeof fetch; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const counts = new Map<string, number>();
  const fetchImpl = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    calls.push({ url, init });
    const handler = routes[url];
    if (!handler) throw new TypeError(`fetch failed: no route for ${url}`);
    const call = counts.get(url) ?? 0;
    counts.set(url, call + 1);
    return handler(call, init);
  }) as typeof fetch;
  return { fetchImpl, calls };
}

/** Standard happy-path routes: manifest + both fp32 files, encoder served in 3 chunks. */
function happyRoutes(): Record<string, (call: number, init?: RequestInit) => Response> {
  return {
    [SPEC.manifestUrl]: () => new Response(JSON.stringify(manifestJson())),
    [`https://models.example.test/test-tier/${ENCODER_PATH}`]: () =>
      new Response(
        streamOf(ENCODER_BYTES.subarray(0, 100), ENCODER_BYTES.subarray(100, 400), ENCODER_BYTES.subarray(400)),
      ),
    [`https://models.example.test/test-tier/${DECODER_PATH}`]: () => new Response(streamOf(DECODER_BYTES)),
  };
}

/**
 * A Response that delivers `chunk` and then fails mid-stream. Pull-based on
 * purpose: with `start()`, undici drops the queued chunk when the stream
 * errors, which a real browser fetch does not do.
 */
function flakyResponse(chunk: Uint8Array): Response {
  let pulls = 0;
  return new Response(
    new ReadableStream<Uint8Array>({
      pull(controller): void {
        if (pulls++ === 0) controller.enqueue(chunk);
        else controller.error(new TypeError('connection reset'));
      },
    }),
  );
}

function baseConfig(fetchImpl: typeof fetch, overrides?: Partial<LoadAssetsConfig>): LoadAssetsConfig {
  return {
    cache: false,
    quantPreference: ['fp32'],
    roles: ['visionEncoder', 'promptDecoder'],
    fetchImpl,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('loadModelAssets', () => {
  it('downloads, verifies, and returns ORT-ready bytes per role', async () => {
    const { fetchImpl } = mockFetch(happyRoutes());
    const assets = await loadModelAssets(SPEC, baseConfig(fetchImpl));

    expect(assets.quant).toBe('fp32');
    expect(assets.totalBytes).toBe(ENCODER_BYTES.byteLength + DECODER_BYTES.byteLength);
    expect(assets.manifest.tier).toBe('test-tier');
    expect(assets.manifest.preprocess.mode).toBe('square-stretch');
    expect([...assets.graphs.keys()]).toEqual(['visionEncoder', 'promptDecoder']);
    expect(assets.graphs.get('visionEncoder')).toEqual(ENCODER_BYTES);
    expect(assets.graphs.get('promptDecoder')).toEqual(DECODER_BYTES);
  });

  it('emits the LoadPhase sequence: manifest → per-file download… → verify', async () => {
    const { fetchImpl } = mockFetch(happyRoutes());
    const events: LoadProgressEvent[] = [];
    await loadModelAssets(SPEC, baseConfig(fetchImpl), (e) => events.push(e));

    expect(events[0]?.phase).toBe('manifest');
    expect(events.some((e) => e.phase === 'offline-cache')).toBe(false);

    const encDownloads = events.filter((e) => e.phase === 'download' && e.file === ENCODER_PATH);
    expect(encDownloads.length).toBeGreaterThan(0);
    // Monotonic loaded, constant total, ends complete.
    let prev = 0;
    for (const e of encDownloads) {
      expect(e.total).toBe(ENCODER_BYTES.byteLength);
      expect(e.loaded ?? 0).toBeGreaterThan(prev);
      prev = e.loaded ?? 0;
    }
    expect(prev).toBe(ENCODER_BYTES.byteLength);

    // Strict interleaving: enc downloads < enc verify < dec downloads < dec verify.
    const index = (predicate: (e: LoadProgressEvent) => boolean): number => events.findIndex(predicate);
    const encVerify = index((e) => e.phase === 'verify' && e.file === ENCODER_PATH);
    const decFirstDownload = index((e) => e.phase === 'download' && e.file === DECODER_PATH);
    const decVerify = index((e) => e.phase === 'verify' && e.file === DECODER_PATH);
    let encLastDownload = -1;
    for (const [i, e] of events.entries()) {
      if (e.phase === 'download' && e.file === ENCODER_PATH) encLastDownload = i;
    }
    expect(encLastDownload).toBeLessThan(encVerify);
    expect(encVerify).toBeLessThan(decFirstDownload);
    expect(decFirstDownload).toBeLessThan(decVerify);
    expect(events.filter((e) => e.phase === 'verify')).toHaveLength(2);
  });

  it('serves cached files from the WeightStore with offline-cache events (no re-download)', async () => {
    const store = new MemoryWeightStore();
    const first = mockFetch(happyRoutes());
    await loadModelAssets(SPEC, baseConfig(first.fetchImpl, { store }));

    const second = mockFetch(happyRoutes());
    const events: LoadProgressEvent[] = [];
    const assets = await loadModelAssets(SPEC, baseConfig(second.fetchImpl, { store }), (e) => events.push(e));

    // Only the manifest goes to the network the second time.
    expect(second.calls.map((c) => c.url)).toEqual([SPEC.manifestUrl]);
    expect(events.filter((e) => e.phase === 'download')).toHaveLength(0);
    const cachedEvents = events.filter((e) => e.phase === 'offline-cache');
    expect(cachedEvents).toEqual([
      { phase: 'offline-cache', file: ENCODER_PATH, loaded: ENCODER_BYTES.byteLength, total: ENCODER_BYTES.byteLength },
      { phase: 'offline-cache', file: DECODER_PATH, loaded: DECODER_BYTES.byteLength, total: DECODER_BYTES.byteLength },
    ]);
    expect(assets.graphs.get('visionEncoder')).toEqual(ENCODER_BYTES);
  });

  it('picks the first quant available for ALL roles (fp16 encoder-only → fp32)', async () => {
    const { fetchImpl } = mockFetch(happyRoutes());
    const quantPreference: readonly Quant[] = ['fp16', 'fp32'];
    const assets = await loadModelAssets(SPEC, baseConfig(fetchImpl, { quantPreference }));
    expect(assets.quant).toBe('fp32');
  });

  it('rejects with InvalidStateError naming available quants when none match', async () => {
    const { fetchImpl } = mockFetch(happyRoutes());
    const promise = loadModelAssets(SPEC, baseConfig(fetchImpl, { quantPreference: ['int8', 'q4f16'] }));
    await expect(promise).rejects.toThrow(InvalidStateError);
    await expect(
      loadModelAssets(SPEC, baseConfig(mockFetch(happyRoutes()).fetchImpl, { quantPreference: ['int8'] })),
    ).rejects.toThrow(/available: \[fp32\]/);
  });

  it('rejects with InvalidStateError when a requested role is missing from the manifest', async () => {
    const { fetchImpl } = mockFetch(happyRoutes());
    await expect(
      loadModelAssets(SPEC, baseConfig(fetchImpl, { roles: ['visionEncoder', 'memoryAttention'] })),
    ).rejects.toThrow(/memoryAttention/);
  });

  it('rejects with WeightVerifyError on a manifest tier mismatch', async () => {
    const wrongTier = { ...manifestJson(), tier: 'other-tier' };
    const { fetchImpl } = mockFetch({ [SPEC.manifestUrl]: () => new Response(JSON.stringify(wrongTier)) });
    await expect(loadModelAssets(SPEC, baseConfig(fetchImpl))).rejects.toThrow(WeightVerifyError);
  });

  it('rejects with WeightVerifyError on an invalid manifest and on manifest HTTP errors', async () => {
    const invalid = { ...manifestJson(), schemaVersion: 2 };
    const badSchema = mockFetch({ [SPEC.manifestUrl]: () => new Response(JSON.stringify(invalid)) });
    await expect(loadModelAssets(SPEC, baseConfig(badSchema.fetchImpl))).rejects.toThrow(WeightVerifyError);

    const notFound = mockFetch({ [SPEC.manifestUrl]: () => new Response('nope', { status: 404 }) });
    await expect(loadModelAssets(SPEC, baseConfig(notFound.fetchImpl))).rejects.toThrow(/HTTP 404/);
  });

  it('propagates network TypeErrors untouched', async () => {
    const failure = new TypeError('fetch failed: DNS');
    const fetchImpl = (async () => {
      throw failure;
    }) as unknown as typeof fetch;
    await expect(loadModelAssets(SPEC, baseConfig(fetchImpl))).rejects.toBe(failure);
  });

  it('rejects with WeightVerifyError on digest mismatch and caches nothing', async () => {
    const routes = happyRoutes();
    routes[`https://models.example.test/test-tier/${ENCODER_PATH}`] = () =>
      new Response(streamOf(encoder.encode('corrupted bytes'))); // wrong content
    const store = new MemoryWeightStore();
    const { fetchImpl } = mockFetch(routes);

    await expect(loadModelAssets(SPEC, baseConfig(fetchImpl, { store }))).rejects.toThrow(WeightVerifyError);
    const encRef = fileRef(ENCODER_PATH, ENCODER_BYTES);
    expect(await store.has(encRef)).toBe(false);
  });

  it('rebases the manifest AND weight files onto modelBaseUrl (trailing slash optional)', async () => {
    const base = 'https://cdn.example.test/models';
    const { fetchImpl, calls } = mockFetch({
      [`${base}/manifest.json`]: () => new Response(JSON.stringify(manifestJson())),
      [`${base}/${ENCODER_PATH}`]: () => new Response(streamOf(ENCODER_BYTES)),
      [`${base}/${DECODER_PATH}`]: () => new Response(streamOf(DECODER_BYTES)),
    });
    const assets = await loadModelAssets(SPEC, baseConfig(fetchImpl, { modelBaseUrl: base }));
    expect(assets.graphs.get('promptDecoder')).toEqual(DECODER_BYTES);
    expect(calls.map((c) => c.url)).toEqual([
      `${base}/manifest.json`,
      `${base}/${ENCODER_PATH}`,
      `${base}/${DECODER_PATH}`,
    ]);
  });

  it('resumes a mid-stream failure with a Range request (206 continuation)', async () => {
    const routes = happyRoutes();
    let rangeHeader: string | undefined;
    routes[`https://models.example.test/test-tier/${ENCODER_PATH}`] = (call, init) => {
      if (call === 0) return flakyResponse(ENCODER_BYTES.subarray(0, 128));
      rangeHeader = (init?.headers as Record<string, string> | undefined)?.['Range'];
      const offset = Number(/^bytes=(\d+)-$/.exec(rangeHeader ?? '')?.[1] ?? Number.NaN);
      return new Response(streamOf(ENCODER_BYTES.subarray(offset)), { status: 206 });
    };
    const { fetchImpl } = mockFetch(routes);
    const events: LoadProgressEvent[] = [];
    const assets = await loadModelAssets(SPEC, baseConfig(fetchImpl), (e) => events.push(e));

    expect(rangeHeader).toBe('bytes=128-');
    expect(assets.graphs.get('visionEncoder')).toEqual(ENCODER_BYTES);
    // Progress stays contiguous across the resume and verify fires once.
    const encDownloads = events.filter((e) => e.phase === 'download' && e.file === ENCODER_PATH);
    expect(encDownloads.at(-1)?.loaded).toBe(ENCODER_BYTES.byteLength);
    expect(events.filter((e) => e.phase === 'verify' && e.file === ENCODER_PATH)).toHaveLength(1);
  });

  it('handles servers that ignore Range (200 full body on retry) by skipping the received prefix', async () => {
    const routes = happyRoutes();
    let sawRange = false;
    routes[`https://models.example.test/test-tier/${ENCODER_PATH}`] = (call, init) => {
      if (call === 0) return flakyResponse(ENCODER_BYTES.subarray(0, 200));
      sawRange = (init?.headers as Record<string, string> | undefined)?.['Range'] !== undefined;
      return new Response(streamOf(ENCODER_BYTES), { status: 200 }); // whole file again
    };
    const { fetchImpl } = mockFetch(routes);
    const assets = await loadModelAssets(SPEC, baseConfig(fetchImpl));
    expect(sawRange).toBe(true); // the resume DID ask for a range; the server ignored it
    expect(assets.graphs.get('visionEncoder')).toEqual(ENCODER_BYTES);
  });

  it('gives up after exhausting resume attempts and propagates the stream error', async () => {
    const routes = happyRoutes();
    routes[`https://models.example.test/test-tier/${ENCODER_PATH}`] = () =>
      new Response(
        new ReadableStream<Uint8Array>({
          start(controller): void {
            controller.enqueue(ENCODER_BYTES.subarray(0, 10));
            controller.error(new TypeError('connection reset'));
          },
        }),
      );
    const { fetchImpl, calls } = mockFetch(routes);
    await expect(loadModelAssets(SPEC, baseConfig(fetchImpl))).rejects.toThrow('connection reset');
    // 1 manifest + 1 initial + 2 resume attempts.
    expect(calls.filter((c) => c.url.endsWith(ENCODER_PATH))).toHaveLength(3);
  });
});
