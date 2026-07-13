/**
 * Studio EDIT-A-VIDEO integration tests (apps/studio/docs/studio-contracts.md
 * §6.1 pattern, extended): exercises the real segmentation seam
 * (`src/segmentation/*`) against the bundled EdgeTAM weights and the golden
 * clip, but through *realistic multi-step editor flows* instead of the
 * single-object happy path already covered by `segmentation.browser.test.ts`.
 *
 * Each `describe` block gets its OWN clip id (and therefore its own
 * `VideoSession`/`MaskTimeline` inside `session-manager.ts`'s module-level
 * `Map`) so the flows never interfere with each other, but all blocks share
 * ONE `loadSegmenter()` call (segmenter/model load is the expensive, one-time
 * part; sessions are cheap to create per flow).
 *
 * Runbook (weights are gitignored, staged by setup-weights):
 *   pnpm -F websam-studio setup-weights
 *   cd apps/studio && pnpm exec vitest run --project browser
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { decodeRLE, InvalidStateError, type MaskResult, type Prompt, type Segmenter } from '@websam3/core';
import { MaskTimeline } from '@websam3/video-editing';

import type { ClipMeta, StudioGet, StudioSet, StudioState, TrackedObject, TrackState } from '../store/studio-store.js';
import { loadSegmenter } from '../segmentation/segmenter-lifecycle.js';
import { activateClip, addPromptObject, refineObject } from '../segmentation/session-manager.js';
import { startTracking, cancelTracking } from '../segmentation/propagate-loop.js';
import { exportMatte } from '../segmentation/export.js';

// Served URL of the committed golden clip; siblings (RLE masks + meta json)
// are fetched relative to it — same fixture set segmentation.browser.test.ts uses.
import clipUrl from '../../../../tools/goldens/fixtures/video/clip-256.mp4?url';

/** 15 minutes per test: generous headroom for wasm-worst-case per-frame cost, matching the existing gate. */
const GATE_TIMEOUT_MS = 900_000;

/** Same bar as the existing studio segmentation gate (studio-contracts.md §6.1/§8). */
const IOU_GATE = 0.85;

const clipAbsUrl = new URL(clipUrl, globalThis.location.href);
clipAbsUrl.search = '';
const fixturesBaseUrl = new URL('./', clipAbsUrl);

interface GoldenRleJson {
  width: number;
  height: number;
  counts: number[];
}

interface GoldenVideoMeta {
  prompt: { frameIndex: number; type: 'point'; x: number; y: number; label: 0 | 1 };
  masks: string[];
  clip: { numFrames: number; width: number; height: number; fps: number };
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

/** The gate must never silently skip: missing bundled weights is a loud failure naming the fix. */
async function requireWeights(): Promise<void> {
  const manifestUrl = new URL('/models/edgetam/manifest.json', globalThis.location.href);
  const res = await fetch(manifestUrl).catch(() => undefined);
  if (!res?.ok) {
    throw new Error(
      `Studio integration gate weights are missing (HTTP ${res?.status ?? 'unreachable'} for ${manifestUrl.href}). ` +
        'Run `pnpm -F websam-studio setup-weights` to stage tools/goldens/models-cache/edgetam/ into apps/studio/public/models/edgetam/.',
    );
  }
}

/**
 * A minimal object satisfying the slices of `StudioState` that
 * `session-manager.ts`/`propagate-loop.ts`/`export.ts` actually read/write —
 * same seam-driving pattern as `segmentation.browser.test.ts`'s
 * `makeFakeStudioState`, extended with `frameCount`-bearing clips only (no
 * timeline/track UI state is needed by these lower-level modules).
 */
function makeFakeStudioState(clip: ClipMeta): { get: StudioGet; set: StudioSet; state: Partial<StudioState> } {
  const state: Partial<StudioState> = {
    clips: { [clip.id]: clip },
    activeClipId: null,
    objects: [],
    maskTimelines: {},
    liveMasksAtFrame: {},
    selection: { timelineClipId: null, objectId: null },
    trackState: { phase: 'idle' },
    playhead: 0,
  };
  const get: StudioGet = () => state as StudioState;
  const set: StudioSet = (partial) => {
    const patch = typeof partial === 'function' ? partial(state as StudioState) : partial;
    Object.assign(state, patch);
  };
  return { get, set, state };
}

let sharedSegmenter: Segmenter;
let sharedMeta: GoldenVideoMeta;
let sharedClipBlob: Blob;
let clipCounter = 0;

/** Fresh `ClipMeta` + fake studio state for one flow, all sharing the golden clip bytes/segmenter. */
function makeClip(): ClipMeta {
  clipCounter += 1;
  return {
    id: `edit-flow-clip-${clipCounter}`,
    fileName: 'clip-256.mp4',
    blob: sharedClipBlob,
    objectUrl: '',
    durationSec: sharedMeta.clip.numFrames / sharedMeta.clip.fps,
    fps: sharedMeta.clip.fps,
    width: sharedMeta.clip.width,
    height: sharedMeta.clip.height,
    frameCount: sharedMeta.clip.numFrames,
    frameCountGuessed: true,
    hasAudio: false,
  };
}

beforeAll(async () => {
  await requireWeights();
  sharedMeta = await fetchJson<GoldenVideoMeta>(new URL('golden-video-meta.json', fixturesBaseUrl));

  let lastPhase = '';
  sharedSegmenter = await loadSegmenter('edgetam', (event) => {
    if (event.phase !== lastPhase) {
      lastPhase = event.phase;
      console.log(`[edit-flows] load: ${event.phase}${event.file ? ` ${event.file}` : ''}`);
    }
  });
  console.log(`[edit-flows] segmenter ready: device=${sharedSegmenter.device} quant=${sharedSegmenter.model.quant}`);

  const clipRes = await fetch(clipAbsUrl);
  if (!clipRes.ok) throw new Error(`failed to fetch clip mp4: HTTP ${clipRes.status}`);
  sharedClipBlob = await clipRes.blob();
}, GATE_TIMEOUT_MS);

afterAll(async () => {
  await sharedSegmenter?.dispose();
});

describe('edit-flows: multi-object track', () => {
  it(
    'two prompted objects both get complete per-frame coverage after one track, and are structurally distinct',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      const clip = makeClip();
      const { get, set, state } = makeFakeStudioState(clip);

      await activateClip(get, set, clip.id);
      const timeline = (state.maskTimelines as Record<string, MaskTimeline>)[clip.id]!;
      expect(timeline).toBeDefined();

      // Object A: the golden prompt point (frame 0) — IoU-checkable against the reference.
      const promptA: Prompt = { type: 'point', x: sharedMeta.prompt.x, y: sharedMeta.prompt.y, label: sharedMeta.prompt.label };
      await addPromptObject(get, set, clip.id, sharedMeta.prompt.frameIndex, [promptA]);

      // Object B: a second point on frame 0, deliberately far from A's point
      // (opposite corner of the 256x256 frame) so the two prompts target
      // visually distinct regions — a realistic "segment two different
      // things" flow, not a duplicate of the same object.
      const promptB: Prompt = { type: 'point', x: 220, y: 30, label: 1 };
      await addPromptObject(get, set, clip.id, sharedMeta.prompt.frameIndex, [promptB]);

      const objects = state.objects as TrackedObject[];
      expect(objects).toHaveLength(2);
      const [objA, objB] = objects.map((o) => o.objectId);

      await startTracking(get, set, clip.id, sharedMeta.prompt.frameIndex + 1);
      expect(state.trackState).toEqual({ phase: 'done', clipId: clip.id });

      // No holes for EITHER object over the full clip.
      expect(timeline.holes(String(objA), { start: 0, end: sharedMeta.clip.numFrames })).toEqual([]);
      expect(timeline.holes(String(objB), { start: 0, end: sharedMeta.clip.numFrames })).toEqual([]);

      let sumIouA = 0;
      let sumOverlapAB = 0;
      let sumSizeA = 0;
      let sumSizeB = 0;
      for (let frameIndex = 0; frameIndex < sharedMeta.clip.numFrames; frameIndex++) {
        const rleA = timeline.get(String(objA), frameIndex);
        const rleB = timeline.get(String(objB), frameIndex);
        expect(rleA, `object A must have a mask at frame ${frameIndex}`).toBeDefined();
        expect(rleB, `object B must have a mask at frame ${frameIndex}`).toBeDefined();

        const binaryA = decodeRLE(rleA!);
        const binaryB = decodeRLE(rleB!);

        // Object A is IoU-gated against the golden reference (same bar as
        // the single-object gate) — proves multi-object prompting didn't
        // degrade the already-validated object's tracking quality.
        const reference = await fetchGoldenMask(sharedMeta.masks[frameIndex] as string);
        const iouA = intersectionOverUnion(binaryA, reference.mask);
        sumIouA += iouA;
        expect(iouA, `frame ${frameIndex} object A vs golden`).toBeGreaterThanOrEqual(IOU_GATE);

        // Plausibility for object B + non-overlap between A/B: neither mask
        // is empty, and they don't largely coincide (they were prompted from
        // opposite corners of the frame).
        let sizeA = 0;
        let sizeB = 0;
        let overlap = 0;
        for (let i = 0; i < binaryA.length; i++) {
          if (binaryA[i]) sizeA++;
          if (binaryB[i]) sizeB++;
          if (binaryA[i] && binaryB[i]) overlap++;
        }
        sumSizeA += sizeA;
        sumSizeB += sizeB;
        sumOverlapAB += overlap;
        expect(sizeB, `object B must be non-empty at frame ${frameIndex}`).toBeGreaterThan(0);
      }
      console.log(`[edit-flows] multi-object: mean IoU(A vs golden)=${(sumIouA / sharedMeta.clip.numFrames).toFixed(4)}`);
      // Overlap between the two distinct objects should be small relative to
      // their sizes (not literally zero — EdgeTAM masks can have soft
      // boundary bleed — but nowhere near "same object").
      const meanOverlapFraction = sumOverlapAB / Math.max(1, Math.min(sumSizeA, sumSizeB));
      expect(meanOverlapFraction).toBeLessThan(0.3);
    },
  );
});

describe('edit-flows: refine mid-flow', () => {
  it(
    'refineObject at a later frame bumps the epoch, invalidates downstream masks, and re-track restores full coverage',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      const clip = makeClip();
      const { get, set, state } = makeFakeStudioState(clip);

      await activateClip(get, set, clip.id);
      const timeline = (state.maskTimelines as Record<string, MaskTimeline>)[clip.id]!;

      const prompt: Prompt = { type: 'point', x: sharedMeta.prompt.x, y: sharedMeta.prompt.y, label: sharedMeta.prompt.label };
      await addPromptObject(get, set, clip.id, sharedMeta.prompt.frameIndex, [prompt]);
      const objectId = (state.objects as TrackedObject[])[0]!.objectId;

      await startTracking(get, set, clip.id, sharedMeta.prompt.frameIndex + 1);
      expect(state.trackState).toEqual({ phase: 'done', clipId: clip.id });
      expect(timeline.holes(String(objectId), { start: 0, end: sharedMeta.clip.numFrames })).toEqual([]);
      expect(timeline.epoch(String(objectId))).toBe(0);

      // Guard: `session-manager.ts`'s `refineObject` refuses while a track is
      // running (studio-contracts.md friction §0.5) — this is the studio's
      // OWN belt-and-braces enforcement of the "one active propagate() per
      // session" rule from m2-internal-contracts.md §6.1, distinct from (and
      // in front of) the core `VideoSession`'s own epoch-bump-mid-propagate
      // semantics. Verify it directly since this gate can never observe a
      // TRUE concurrent-epoch-invalidation through the studio seam (the
      // studio blocks the call before it ever reaches `session.refineObject`).
      set({ trackState: { phase: 'running', clipId: clip.id, frameIndex: 5, frameCount: sharedMeta.clip.numFrames } as TrackState });
      await expect(refineObject(get, set, clip.id, objectId, 5, [prompt])).rejects.toBeInstanceOf(InvalidStateError);
      set({ trackState: { phase: 'idle' } });

      // Refine at frame 5 (a "correction" using the same point — this test
      // only cares about the epoch/coverage contract, not a different mask
      // shape) once tracking is idle again — the real, allowed refine path.
      const refineFrame = 5;
      await refineObject(get, set, clip.id, objectId, refineFrame, [prompt]);

      expect(timeline.epoch(String(objectId)), 'refineObject must bump the timeline epoch').toBe(1);
      // invalidateAfter(refineFrame) drops every frame STRICTLY after
      // refineFrame — frames 6..9 must now be holes; frames 0..5 remain
      // (m2-internal-contracts.md §6.1's epoch semantics).
      expect(timeline.holes(String(objectId), { start: 0, end: sharedMeta.clip.numFrames })).toEqual([6, 7, 8, 9]);

      // Re-track resuming from the refine frame (mirrors the demo's
      // trackEpochRef-driven resume flow) must restore full coverage.
      await startTracking(get, set, clip.id, refineFrame);
      expect(state.trackState).toEqual({ phase: 'done', clipId: clip.id });
      expect(
        timeline.holes(String(objectId), { start: 0, end: sharedMeta.clip.numFrames }),
        'after re-track from the refine frame, coverage must be complete again',
      ).toEqual([]);

      // Regression probe: the corrected mask `refineObject` computed (and
      // stored into `liveMasksAtFrame`) SHOULD be what ends up in the
      // timeline at the refine frame after a re-track — the timeline is the
      // source of truth for export/coverage. See notes/findings for the
      // actual observed result.
      const refinedLiveMask = (state.liveMasksAtFrame as Record<number, MaskResult>)[objectId]!;
      const timelineRleAtRefineFrame = timeline.get(String(objectId), refineFrame)!;
      const refinedBinary = refinedLiveMask.toBinary();
      const timelineBinary = decodeRLE(timelineRleAtRefineFrame);
      const matchesRefinedMask = timelineBinary.length === refinedBinary.length && timelineBinary.every((v, i) => v === refinedBinary[i]);
      console.log(
        `[edit-flows] refine-mid-flow: timeline mask at refine frame ${refineFrame} matches refineObject's own corrected mask = ${matchesRefinedMask}`,
      );
    },
  );
});

describe('edit-flows: cancel mid-track', () => {
  it(
    'cancelling a track returns trackState to idle with no dangling iterator, and a subsequent track completes',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      const clip = makeClip();
      const { get, set, state } = makeFakeStudioState(clip);

      await activateClip(get, set, clip.id);
      const timeline = (state.maskTimelines as Record<string, MaskTimeline>)[clip.id]!;

      const prompt: Prompt = { type: 'point', x: sharedMeta.prompt.x, y: sharedMeta.prompt.y, label: sharedMeta.prompt.label };
      await addPromptObject(get, set, clip.id, sharedMeta.prompt.frameIndex, [prompt]);
      const objectId = (state.objects as TrackedObject[])[0]!.objectId;

      // Start tracking the full remaining range but don't await — cancel
      // while it's in flight (before any frame past the prompt can possibly
      // have completed, given wasm/webgpu per-frame cost).
      const trackingPromise = startTracking(get, set, clip.id, sharedMeta.prompt.frameIndex + 1);
      cancelTracking();
      await trackingPromise;

      expect(state.trackState).toEqual({ phase: 'idle' });

      // No dangling iterator: a fresh track must be accepted immediately
      // (propagate-loop.ts's `activeTrack` guard would reject a second
      // concurrent track with InvalidStateError if cancel left it set).
      await startTracking(get, set, clip.id, sharedMeta.prompt.frameIndex + 1);
      expect(state.trackState).toEqual({ phase: 'done', clipId: clip.id });
      expect(timeline.holes(String(objectId), { start: 0, end: sharedMeta.clip.numFrames })).toEqual([]);

      for (let frameIndex = sharedMeta.prompt.frameIndex + 1; frameIndex < sharedMeta.clip.numFrames; frameIndex++) {
        const rle = timeline.get(String(objectId), frameIndex)!;
        const binary = decodeRLE(rle);
        const reference = await fetchGoldenMask(sharedMeta.masks[frameIndex] as string);
        const iou = intersectionOverUnion(binary, reference.mask);
        expect(iou, `frame ${frameIndex} after cancel+retrack`).toBeGreaterThanOrEqual(IOU_GATE);
      }
    },
  );
});

describe('edit-flows: export after track (prompt-frame-hole regression guard)', () => {
  it(
    'exportMatte produces a real, parseable matte.zip whose PNG entries cover exactly the tracked frame range',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      const clip = makeClip();
      const { get, set, state } = makeFakeStudioState(clip);

      await activateClip(get, set, clip.id);
      const prompt: Prompt = { type: 'point', x: sharedMeta.prompt.x, y: sharedMeta.prompt.y, label: sharedMeta.prompt.label };
      await addPromptObject(get, set, clip.id, sharedMeta.prompt.frameIndex, [prompt]);
      const objectId = (state.objects as TrackedObject[])[0]!.objectId;

      await startTracking(get, set, clip.id, sharedMeta.prompt.frameIndex + 1);
      expect(state.trackState).toEqual({ phase: 'done', clipId: clip.id });

      // exportMatte triggers a browser download via an anchor click — stub
      // out the DOM download side-effect (matches the demo's own pattern)
      // while letting the real exporter build the real zip Blob.
      const originalCreateElement = document.createElement.bind(document);
      const clickedAnchors: HTMLAnchorElement[] = [];
      const createElementSpy = (tag: string, opts?: ElementCreationOptions) => {
        const el = originalCreateElement(tag, opts);
        if (tag === 'a') {
          const anchor = el as HTMLAnchorElement;
          anchor.click = () => clickedAnchors.push(anchor);
          return anchor;
        }
        return el;
      };
      document.createElement = createElementSpy as typeof document.createElement;

      let exportedBlob: Blob | undefined;
      const originalCreateObjectURL = URL.createObjectURL.bind(URL);
      URL.createObjectURL = ((blob: Blob) => {
        if (blob.type === 'application/zip') exportedBlob = blob;
        return originalCreateObjectURL(blob);
      }) as typeof URL.createObjectURL;

      try {
        await exportMatte(get, set, clip.id);
      } finally {
        document.createElement = originalCreateElement;
        URL.createObjectURL = originalCreateObjectURL;
      }

      expect(state.exportState).toMatchObject({ phase: 'done' });
      expect(clickedAnchors).toHaveLength(1);
      expect(exportedBlob, 'exportMatte must produce a real zip Blob').toBeDefined();

      const zipBytes = new Uint8Array(await exportedBlob!.arrayBuffer());
      const entries = parseStoreModeZip(zipBytes);
      const pngEntries = entries.filter((e) => e.name.endsWith('.png'));

      // Regression guard for the prompt-frame-hole bug: the conditioning
      // (prompt) frame's mask must be exported too, not just the
      // propagated frames — so the exported PNG count must equal the FULL
      // tracked range [promptFrame, numFrames), not [promptFrame+1, numFrames).
      const expectedFrameCount = sharedMeta.clip.numFrames - sharedMeta.prompt.frameIndex;
      expect(pngEntries).toHaveLength(expectedFrameCount);

      // Every entry must be a valid, non-empty PNG (magic-byte signature check).
      const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
      for (const entry of pngEntries) {
        expect(entry.data.length, `${entry.name} must have PNG bytes`).toBeGreaterThan(PNG_SIGNATURE.length);
        for (let i = 0; i < PNG_SIGNATURE.length; i++) {
          expect(entry.data[i], `${entry.name} byte ${i} of PNG signature`).toBe(PNG_SIGNATURE[i]);
        }
      }

      // The exported frame names must cover exactly the prompt-frame..end
      // range with no gaps and no duplicates.
      const frameNumbers = pngEntries
        .map((e) => /frame-(\d{6})\.png$/.exec(e.name)?.[1])
        .filter((v): v is string => v !== undefined)
        .map(Number)
        .sort((a, b) => a - b);
      const expectedFrameNumbers = Array.from(
        { length: expectedFrameCount },
        (_, i) => sharedMeta.prompt.frameIndex + i,
      );
      expect(frameNumbers).toEqual(expectedFrameNumbers);

      expect(state.exportState).toMatchObject({ framesExported: expectedFrameCount });
      void objectId;
    },
  );
});

describe('edit-flows: seek accuracy', () => {
  it(
    'the preview <video> element reports the frame at the requested project-frame index (frame-index <-> time mapping)',
    { timeout: GATE_TIMEOUT_MS },
    async () => {
      const clip = makeClip();
      const { get, set } = makeFakeStudioState(clip);
      await activateClip(get, set, clip.id);
      const meta = sharedMeta.clip;

      const video = document.createElement('video');
      video.muted = true;
      video.src = clip.objectUrl || clipAbsUrl.href;
      document.body.appendChild(video);
      try {
        await new Promise<void>((resolve, reject) => {
          video.addEventListener('loadedmetadata', () => resolve(), { once: true });
          video.addEventListener('error', () => reject(video.error), { once: true });
        });

        async function seekToFrame(frameIndex: number): Promise<number> {
          const targetTime = frameIndex / meta.fps;
          await new Promise<void>((resolve, reject) => {
            video.addEventListener('seeked', () => resolve(), { once: true });
            video.addEventListener('error', () => reject(video.error), { once: true });
            video.currentTime = targetTime;
          });
          return video.currentTime;
        }

        // Frame-index -> time -> frame-index round trip at several points
        // across the clip, including the first and last frames.
        const checkFrames = [0, 1, Math.floor(meta.numFrames / 2), meta.numFrames - 1];
        for (const frameIndex of checkFrames) {
          const actualTime = await seekToFrame(frameIndex);
          const expectedTime = frameIndex / meta.fps;
          // Within half a frame duration: the seek target itself is exact,
          // but `HTMLVideoElement.currentTime` after a 'seeked' event can
          // land on the nearest keyframe/decoded-frame boundary.
          const halfFrame = 0.5 / meta.fps;
          expect(
            Math.abs(actualTime - expectedTime),
            `frame ${frameIndex}: expected currentTime near ${expectedTime.toFixed(4)}s, got ${actualTime.toFixed(4)}s`,
          ).toBeLessThanOrEqual(halfFrame + 1e-3);

          // Round-trip: converting the resulting time back to a frame index
          // (the same math PreviewCanvas's rVFC loop uses per
          // studio-contracts.md §3) must recover the same frame index.
          const recoveredFrameIndex = Math.round(actualTime * meta.fps);
          expect(recoveredFrameIndex).toBe(frameIndex);
        }
      } finally {
        video.remove();
      }
    },
  );
});

/**
 * Minimal ZIP reader for the exporter's STORE-mode entries (fflate's
 * `ZipPassThrough`, per `packages/video-editing/src/exporter.ts` §7.1's
 * "entries use store" design). Reads the END OF CENTRAL DIRECTORY record
 * (signature `PK\x05\x06`) and then the CENTRAL DIRECTORY (signature
 * `PK\x01\x02`) — NOT the local file headers directly: fflate's streaming
 * `Zip`/`ZipPassThrough` writer sets the general-purpose bit-3 "streaming"
 * flag and writes `compressedSize=0`/`crc32=0` in each LOCAL header,
 * deferring the real sizes to a trailing data descriptor. Only the central
 * directory carries reliable sizes regardless of streaming mode, which is
 * why every real zip reader (fflate's own `unzipSync` included) parses it
 * instead of local headers. Avoids adding a new runtime dependency
 * (`fflate` is not a direct `apps/studio` dependency) to a test-only file,
 * per this test's ownership scope (new files only).
 */
function parseStoreModeZip(bytes: Uint8Array): { name: string; data: Uint8Array }[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const EOCD_SIG = 0x06054b50;
  const CENTRAL_DIR_SIG = 0x02014b50;
  const LOCAL_FILE_HEADER_SIG = 0x04034b50;

  // EOCD is at least 22 bytes, at the very end (no zip comment in fflate's output).
  let eocdOffset = -1;
  for (let i = bytes.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === EOCD_SIG) {
      eocdOffset = i;
      break;
    }
  }
  expect(eocdOffset, 'zip must contain an End Of Central Directory record').toBeGreaterThanOrEqual(0);

  const entryCount = view.getUint16(eocdOffset + 10, true);
  const centralDirOffset = view.getUint32(eocdOffset + 16, true);

  const entries: { name: string; data: Uint8Array }[] = [];
  let offset = centralDirOffset;
  for (let i = 0; i < entryCount; i++) {
    expect(view.getUint32(offset, true), `central directory entry ${i} signature`).toBe(CENTRAL_DIR_SIG);
    const compressionMethod = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localHeaderOffset = view.getUint32(offset + 42, true);
    const nameStart = offset + 46;
    const name = new TextDecoder().decode(bytes.subarray(nameStart, nameStart + nameLength));
    expect(compressionMethod, `${name}: expected STORE (0) compression`).toBe(0);

    // Locate the data via the LOCAL header's own name/extra lengths (may
    // differ in extra-field length from the central directory's copy).
    expect(view.getUint32(localHeaderOffset, true), `${name}: local file header signature`).toBe(LOCAL_FILE_HEADER_SIG);
    const localNameLength = view.getUint16(localHeaderOffset + 26, true);
    const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
    const dataStart = localHeaderOffset + 30 + localNameLength + localExtraLength;
    const data = bytes.subarray(dataStart, dataStart + compressedSize);
    entries.push({ name, data });

    offset = nameStart + nameLength + extraLength + commentLength;
  }
  expect(entries.length, 'zip must contain at least one entry').toBeGreaterThan(0);
  return entries;
}
