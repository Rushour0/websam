/**
 * M1 browser gate (docs/m1-internal-contracts.md §6.2): the REAL pipeline —
 * `createSegmenter` → module worker → weight verify → ort compile → encode →
 * decode — must reproduce the committed transformers.js reference masks on a
 * NON-square image (640×427, per the coordinate contract's golden
 * requirement) with IoU ≥ 0.9.
 *
 * Runbook (model weights are NOT committed):
 *
 *   1. `. ~/.nvm/nvm.sh && nvm use 22`
 *   2. `node tools/goldens/fetch-models.mjs`
 *      — pulls the pinned q4f16 community graphs (~300 MB) into
 *        `tools/goldens/models-cache/` (gitignored, CI-cacheable), merges the
 *        ONNX external data into self-contained files, and emits the
 *        schemaVersion-1 manifests this test loads.
 *   3. `cd packages/core && pnpm exec vitest run src/e2e/image-golden.browser.test.ts --project browser`
 *      (or `pnpm test:browser`, which runs the whole browser project).
 *
 * Models absent → the suite FAILS loudly naming fetch-models.mjs; it never
 * silently skips. The wasm leg always runs (deterministic in CI) and is slow:
 * the q4f16 vision encoder on single-threaded wasm (no COOP/COEP in the test
 * server → ort forces numThreads=1) takes minutes — hence the 10-minute
 * per-test timeout. The webgpu leg soft-passes when no adapter exists or the
 * EP cannot initialize (headless SwiftShader lanes), mirroring
 * `src/ort.browser.test.ts`.
 */
import { describe, expect, it } from 'vitest';
import {
  createSegmenter,
  decodeRLE,
  registerModel,
  type CoordinateTransform,
  type MaskResult,
  type Prompt,
} from '../index.js';
// Served URL of the committed golden scene; also anchors the URLs of its
// sibling RLE fixtures and of ../models-cache/ (out-of-root /@fs/ serving,
// allowed via server.fs.allow in vitest.config.ts).
import sceneUrl from '../../../../tools/goldens/fixtures/scene-640x427.png?url';

/** 10 minutes: the wasm encoder run alone can take several minutes. */
const GATE_TIMEOUT_MS = 600_000;

/** Must match `E2E_TIER` in tools/goldens/fetch-models.mjs (manifest-e2e.json). */
const E2E_MODEL_ID = 'sam3-tracker-e2e';

const MIN_IOU = 0.9;

/** Golden prompts (tools/goldens/fixtures/golden-meta.json), source pixels. */
const GOLDENS: { name: string; rleFile: string; prompts: Prompt[] }[] = [
  {
    name: 'point1 (positive point at the big circle center)',
    rleFile: 'golden-mask-point1.rle.json',
    prompts: [{ type: 'point', x: 180, y: 210, label: 1 }],
  },
  {
    name: 'point2 (positive rect center + negative circle center)',
    rleFile: 'golden-mask-point2.rle.json',
    prompts: [
      { type: 'point', x: 465, y: 220, label: 1 },
      { type: 'point', x: 180, y: 210, label: 0 },
    ],
  },
];

// URL anchors, derived from the one ?url asset import (query stripped so
// sibling resolution and raw static fetches stay clean).
const sceneAbsUrl = new URL(sceneUrl, globalThis.location.href);
sceneAbsUrl.search = '';
const fixturesBaseUrl = new URL('./', sceneAbsUrl);
const modelsBaseUrl = new URL('../models-cache/', sceneAbsUrl);

// The dev-served worker entry: in dist, index.js spawns its sibling
// worker.js, but browser-mode tests run from SOURCE where that sibling does
// not exist — `workerUrl` is the documented escape hatch (§4.2). The shim
// stubs vitest's dynamic-import wrapper before loading the real entry.
const workerUrl = new URL('./worker-shim.ts', import.meta.url);

// Structural view of WebGPU (lib.dom has no navigator.gpu typing).
const gpu = (
  globalThis.navigator as unknown as
    | { gpu?: { requestAdapter(): Promise<object | null> } }
    | undefined
)?.gpu;

// The built-in 'sam3-tracker' tier is webgpu-only and points at the
// production CDN; the gate registers its own tier (matching manifest-e2e.json)
// so the SAME manifest drives both device legs, including wasm.
registerModel({
  id: E2E_MODEL_ID,
  displayName: 'SAM 3 Tracker (M1 e2e gate)',
  arch: 'sam3-tracker',
  inputSize: 1008,
  supportsVideo: false,
  license: 'sam-license',
  requiresLicenseAcceptance: true,
  manifestUrl: 'https://models.websam.invalid/manifest-e2e.json',
  devices: { webgpu: true, wasm: true },
});

/** The gate must never silently skip: missing models is a loud failure. */
async function requireModels(): Promise<void> {
  const manifestUrl = new URL('manifest-e2e.json', modelsBaseUrl);
  const res = await fetch(manifestUrl).catch(() => undefined);
  if (!res?.ok) {
    throw new Error(
      `M1 gate models are missing (HTTP ${res?.status ?? 'unreachable'} for ${manifestUrl.href}). ` +
        'Run `node tools/goldens/fetch-models.mjs` (Node 22) to populate tools/goldens/models-cache/.',
    );
  }
}

interface GoldenRleJson {
  width: number;
  height: number;
  counts: number[];
}

async function fetchGoldenMask(rleFile: string): Promise<{ width: number; height: number; mask: Uint8Array }> {
  const res = await fetch(new URL(rleFile, fixturesBaseUrl));
  if (!res.ok) throw new Error(`failed to fetch golden RLE ${rleFile}: HTTP ${res.status}`);
  const rle = (await res.json()) as GoldenRleJson;
  return {
    width: rle.width,
    height: rle.height,
    mask: decodeRLE({ width: rle.width, height: rle.height, counts: Uint32Array.from(rle.counts) }),
  };
}

function intersectionOverUnion(a: Uint8Array, b: Uint8Array): number {
  expect(a.length).toBe(b.length);
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ? 1 : 0;
    const bv = b[i] ? 1 : 0;
    intersection += av & bv;
    union += av | bv;
  }
  return union === 0 ? 1 : intersection / union;
}

/** Boot the real pipeline on `device` and assert both goldens at ≥ MIN_IOU. */
async function runGoldenGate(device: 'webgpu' | 'wasm'): Promise<void> {
  // Phase logging: minutes-long CI runs need a heartbeat to debug timeouts.
  let lastPhase = '';
  let lastLoggedMb = -64;
  const segmenter = await createSegmenter({
    model: E2E_MODEL_ID,
    device,
    acceptLicense: 'sam',
    modelBaseUrl: modelsBaseUrl.href,
    cache: false, // hermetic: no OPFS/Cache-API state carried across runs
    workerUrl,
    onProgress: (e) => {
      const mb = (e.loaded ?? 0) / (1024 * 1024);
      if (e.phase !== lastPhase || mb - lastLoggedMb >= 64) {
        lastPhase = e.phase;
        lastLoggedMb = mb;
        console.log(
          `[image-golden] ${device} load: ${e.phase}${e.file ? ` ${e.file}` : ''}` +
            `${e.loaded !== undefined ? ` ${mb.toFixed(0)}/${((e.total ?? 0) / (1024 * 1024)).toFixed(0)} MB` : ''}`,
        );
      }
    },
  });
  console.log(`[image-golden] ${device} segmenter ready (quant=${segmenter.model.quant})`);
  try {
    expect(segmenter.device).toBe(device);
    const session = await segmenter.createImageSession();

    const sceneRes = await fetch(sceneAbsUrl);
    if (!sceneRes.ok) throw new Error(`failed to fetch scene png: HTTP ${sceneRes.status}`);
    const bitmap = await createImageBitmap(await sceneRes.blob());
    const encoded = await session.encode(bitmap);
    expect(encoded.width).toBe(640);
    expect(encoded.height).toBe(427);
    console.log(`[image-golden] ${device} encodeMs=${encoded.encodeMs.toFixed(0)}`);

    for (const golden of GOLDENS) {
      const reference = await fetchGoldenMask(golden.rleFile);
      const results = await session.decode(golden.prompts);
      expect(results).toHaveLength(1);
      const result = results[0] as MaskResult;
      expect(result.width).toBe(reference.width);
      expect(result.height).toBe(reference.height);

      // Coordinate-contract rule 4: the result carries its transform, and it
      // must be the S0-pinned anisotropic square-stretch (no padding). A
      // letterbox/mode drift would silently tank the IoU — fail on it by name.
      const transform = (result as MaskResult & { transform?: CoordinateTransform }).transform;
      expect(transform?.mode).toBe('square-stretch');
      expect(transform?.padX).toBe(0);
      expect(transform?.padY).toBe(0);

      const binary = result.toBinary();
      // Prompt-level sanity in source pixels: positive points sit inside the
      // mask, negative points outside (cheap, catches x/y or axis-scale swaps).
      for (const prompt of golden.prompts) {
        if (prompt.type !== 'point') continue;
        const at = binary[prompt.y * reference.width + prompt.x];
        expect(at, `label-${prompt.label} point (${prompt.x},${prompt.y}) vs mask`).toBe(prompt.label);
      }

      const iou = intersectionOverUnion(binary, reference.mask);
      console.log(
        `[image-golden] ${device} ${golden.name}: IoU=${iou.toFixed(4)} score=${result.score.toFixed(3)}`,
      );
      expect(iou, `${golden.name} on ${device}`).toBeGreaterThanOrEqual(MIN_IOU);
    }

    session.dispose();
  } finally {
    await segmenter.dispose();
  }
}

describe('M1 image golden gate (real worker pipeline vs transformers.js reference)', () => {
  it(
    'reproduces both golden masks at IoU >= 0.9 on the wasm device',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      await requireModels();
      await runGoldenGate('wasm');
    },
  );

  // Never fails on a missing adapter: skipped without navigator.gpu, and a
  // denied adapter / EP-init failure downgrades to a logged soft pass (same
  // pattern as src/ort.browser.test.ts). Once the pipeline boots, the IoU
  // assertions are as hard as the wasm leg's.
  it.skipIf(!gpu)(
    'reproduces both golden masks at IoU >= 0.9 on the webgpu device when an adapter exists',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      await requireModels();
      const adapter = await gpu!.requestAdapter().catch(() => null);
      if (!adapter) {
        console.log('[image-golden] webgpu: soft pass (navigator.gpu present but no adapter)');
        return;
      }
      try {
        await runGoldenGate('webgpu');
      } catch (err) {
        // Adapter exists but the EP could not initialize/compile (headless
        // driver quirks) — log and soft-pass rather than flake CI. Assertion
        // failures (IoU, shapes) are NOT swallowed.
        if (err instanceof Error && 'matcherResult' in err) throw err;
        console.log('[image-golden] webgpu: soft pass (pipeline failed to boot)', err);
      }
    },
  );
});
