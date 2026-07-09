/**
 * M2 browser gate (docs/m2-internal-contracts.md §6 + §9.3): the REAL video
 * pipeline — `createSegmenter({model:'edgetam'})` → module worker →
 * `createVideoSession` → `attachSource` (real mp4box demux + WebCodecs
 * decode) → `addObject` (prompt on frame 0) → `propagate()` (the real
 * memory-attention loop: videoEncoder → memoryAttention/noMemCondition →
 * maskDecoderVideo → memoryEncoder → `MemoryBank.commit`, per frame) — must
 * reproduce the committed HF `EdgeTamVideoModel` fp32 reference masks
 * (`tools/goldens/make-video-golden.py`) at per-frame IoU >= the golden
 * meta's `iouGate` (0.90). This is the first in-browser SAM-family
 * memory-attention VIDEO tracking test.
 *
 * Golden protocol (mirrors make-video-golden.py exactly):
 *   - frame 0's golden mask is the model's response to the INITIAL PROMPT
 *     (our `addObject({frameIndex: 0, prompts: [...]})` call) — NOT a
 *     `propagate()` frame.
 *   - frames 1..9 are `model(session, frame_idx=t)` with no further
 *     prompts — our `propagate({startFrame: 1})` loop.
 *
 * Runbook (model weights are NOT committed — gitignored, regenerable):
 *
 *   1. `. ~/.nvm/nvm.sh && nvm use 22`
 *   2. `cd tools/goldens && ../export/.venv/bin/python make-video-golden.py`
 *      — regenerates `models-cache/edgetam/*.onnx` + `manifest.json` (tier
 *        'edgetam', fp16, self-contained — no external-data merge needed,
 *        unlike the M1 community-graph gate) and the committed
 *        `fixtures/video/*` golden clip + per-frame RLEs + meta.
 *   3. `cd packages/core && pnpm exec vitest run src/e2e/video-golden.browser.test.ts --project browser`
 *      (or `pnpm test:browser`, which runs the whole browser project).
 *
 * Models absent → the suite FAILS loudly naming the regen command; it never
 * silently skips (same M1 rule). Forces `device: 'wasm'` for determinism in
 * CI (no COOP/COEP in the test server → ort forces numThreads=1): the video
 * loop runs 4 graphs (videoEncoder/noMemCondition-or-memoryAttention/
 * maskDecoderVideo/memoryEncoder) x 10 frames on a tiny 256x256 clip, but
 * each graph is itself compiled + run at the manifest's `inputSize` (1024)
 * on single-threaded wasm — budget the full 15 minutes. WebGPU would be far
 * faster but headless Chromium has no adapter; the webgpu leg soft-skips
 * exactly like the M1 gate and `src/ort.browser.test.ts` when no adapter
 * exists or the EP fails to initialize.
 */
import { describe, expect, it } from 'vitest';
import { createSegmenter, decodeRLE, type MaskResult, type Prompt } from '../index.js';
// Served URL of the committed golden clip; also anchors the URLs of its
// sibling RLE fixtures, the meta json, and ../../models-cache/edgetam/ (the
// gitignored weights directory — out-of-root /@fs/ serving, allowed via
// server.fs.allow in vitest.config.ts, same as the M1 gate).
import clipUrl from '../../../../tools/goldens/fixtures/video/clip-256.mp4?url';

/** 15 minutes: 10 frames x 4 graphs on single-threaded wasm at 1024 input. */
const GATE_TIMEOUT_MS = 900_000;

const MODEL_ID = 'edgetam';

const clipAbsUrl = new URL(clipUrl, globalThis.location.href);
clipAbsUrl.search = '';
const fixturesBaseUrl = new URL('./', clipAbsUrl);
const modelsBaseUrl = new URL('../../models-cache/edgetam/', clipAbsUrl);

// The dev-served worker entry (see worker-shim.ts doc for why this shim
// exists — vitest-browser dynamic-import + module-eval-order quirks).
const workerUrl = new URL('./worker-shim.ts', import.meta.url);

// Structural view of WebGPU (lib.dom has no navigator.gpu typing).
const gpu = (
  globalThis.navigator as unknown as
    | { gpu?: { requestAdapter(): Promise<object | null> } }
    | undefined
)?.gpu;

interface GoldenRleJson {
  width: number;
  height: number;
  counts: number[];
}

interface GoldenVideoMeta {
  prompt: { frameIndex: number; type: 'point'; x: number; y: number; label: 0 | 1 };
  iouGate: number;
  masks: string[];
  clip: { numFrames: number; width: number; height: number; fps: number };
}

/** The gate must never silently skip: missing models is a loud failure. */
async function requireModels(): Promise<void> {
  const manifestUrl = new URL('manifest.json', modelsBaseUrl);
  const res = await fetch(manifestUrl).catch(() => undefined);
  if (!res?.ok) {
    throw new Error(
      `M2 video gate weights are missing (HTTP ${res?.status ?? 'unreachable'} for ${manifestUrl.href}). ` +
        'Run `cd tools/goldens && ../export/.venv/bin/python make-video-golden.py` to populate ' +
        'tools/goldens/models-cache/edgetam/ (requires the tools/export venv + ffmpeg on PATH).',
    );
  }
}

async function fetchJson<T>(url: URL): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url.href}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

async function fetchGoldenMask(rleFile: string): Promise<{ width: number; height: number; mask: Uint8Array }> {
  const rle = await fetchJson<GoldenRleJson>(new URL(rleFile, fixturesBaseUrl));
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

/** Boot the real video pipeline on `device` and assert all 10 golden frames at >= iouGate. */
async function runVideoGoldenGate(device: 'webgpu' | 'wasm'): Promise<void> {
  const meta = await fetchJson<GoldenVideoMeta>(new URL('golden-video-meta.json', fixturesBaseUrl));

  let lastPhase = '';
  let lastLoggedMb = -64;
  const segmenter = await createSegmenter({
    model: MODEL_ID,
    device,
    modelBaseUrl: modelsBaseUrl.href,
    cache: false, // hermetic: no OPFS/Cache-API state carried across runs
    workerUrl,
    onProgress: (e) => {
      const mb = (e.loaded ?? 0) / (1024 * 1024);
      if (e.phase !== lastPhase || mb - lastLoggedMb >= 64) {
        lastPhase = e.phase;
        lastLoggedMb = mb;
        console.log(
          `[video-golden] ${device} load: ${e.phase}${e.file ? ` ${e.file}` : ''}` +
            `${e.loaded !== undefined ? ` ${mb.toFixed(0)}/${((e.total ?? 0) / (1024 * 1024)).toFixed(0)} MB` : ''}`,
        );
      }
    },
  });
  console.log(`[video-golden] ${device} segmenter ready (quant=${segmenter.model.quant})`);
  try {
    expect(segmenter.device).toBe(device);
    const session = await segmenter.createVideoSession();

    const clipRes = await fetch(clipAbsUrl);
    if (!clipRes.ok) throw new Error(`failed to fetch clip mp4: HTTP ${clipRes.status}`);
    const clipBlob = await clipRes.blob();

    const info = await session.attachSource(clipBlob);
    expect(info.width).toBe(meta.clip.width);
    expect(info.height).toBe(meta.clip.height);
    expect(info.frameCount).toBe(meta.clip.numFrames);

    const prompt: Prompt = {
      type: 'point',
      x: meta.prompt.x,
      y: meta.prompt.y,
      label: meta.prompt.label,
    };

    // Frame 0's golden is the INITIAL PROMPT response (addObject), not a
    // propagate() frame — make-video-golden.py's `out0 = model(session,
    // frame_idx=PROMPT_FRAME)` runs immediately after adding the point.
    const added = await session.addObject({ frameIndex: meta.prompt.frameIndex, prompts: [prompt] });
    const perFrameIou: number[] = new Array(meta.clip.numFrames).fill(Number.NaN);

    {
      const reference = await fetchGoldenMask(meta.masks[meta.prompt.frameIndex] as string);
      const binary = added.mask.toBinary();
      expect(added.mask.width).toBe(reference.width);
      expect(added.mask.height).toBe(reference.height);
      const iou = intersectionOverUnion(binary, reference.mask);
      perFrameIou[meta.prompt.frameIndex] = iou;
      console.log(`[video-golden] ${device} frame ${meta.prompt.frameIndex} (addObject): IoU=${iou.toFixed(4)}`);
      expect(iou, `frame ${meta.prompt.frameIndex} (addObject) on ${device}`).toBeGreaterThanOrEqual(meta.iouGate);
    }

    // Frames [1, numFrames) via the real propagate() memory-attention loop.
    let seen = 0;
    for await (const frame of session.propagate({ startFrame: meta.prompt.frameIndex + 1 })) {
      expect(frame.masks).toHaveLength(1);
      const result = frame.masks[0] as MaskResult;
      expect(result.objectId).toBe(added.objectId);
      const reference = await fetchGoldenMask(meta.masks[frame.frameIndex] as string);
      expect(result.width).toBe(reference.width);
      expect(result.height).toBe(reference.height);
      const binary = result.toBinary();
      const iou = intersectionOverUnion(binary, reference.mask);
      perFrameIou[frame.frameIndex] = iou;
      console.log(`[video-golden] ${device} frame ${frame.frameIndex} (propagate): IoU=${iou.toFixed(4)}`);
      expect(iou, `frame ${frame.frameIndex} (propagate) on ${device}`).toBeGreaterThanOrEqual(meta.iouGate);
      seen++;
    }
    expect(seen).toBe(meta.clip.numFrames - meta.prompt.frameIndex - 1);
    expect(perFrameIou.every((v) => Number.isFinite(v))).toBe(true);
    console.log(
      `[video-golden] ${device} all ${meta.clip.numFrames} frames: ` +
        perFrameIou.map((v, i) => `f${i}=${v.toFixed(3)}`).join(' '),
    );

    session.dispose();
  } finally {
    await segmenter.dispose();
  }
}

describe('M2 video golden gate (real worker pipeline vs HF EdgeTamVideoModel reference)', () => {
  it(
    'reproduces all 10 golden per-frame masks at IoU >= gate on the wasm device',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      await requireModels();
      await runVideoGoldenGate('wasm');
    },
  );

  // Never fails on a missing adapter: skipped without navigator.gpu, and a
  // denied adapter / EP-init failure downgrades to a logged soft pass (same
  // pattern as src/ort.browser.test.ts and the M1 image gate). Once the
  // pipeline boots, the IoU assertions are as hard as the wasm leg's.
  it.skipIf(!gpu)(
    'reproduces all 10 golden per-frame masks at IoU >= gate on the webgpu device when an adapter exists',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      await requireModels();
      const adapter = await gpu!.requestAdapter().catch(() => null);
      if (!adapter) {
        console.log('[video-golden] webgpu: soft pass (navigator.gpu present but no adapter)');
        return;
      }
      try {
        await runVideoGoldenGate('webgpu');
      } catch (err) {
        // Adapter exists but the EP could not initialize/compile (headless
        // driver quirks) — log and soft-pass rather than flake CI. Assertion
        // failures (IoU, shapes) are NOT swallowed.
        if (err instanceof Error && 'matcherResult' in err) throw err;
        console.log('[video-golden] webgpu: soft pass (pipeline failed to boot)', err);
      }
    },
  );
});
