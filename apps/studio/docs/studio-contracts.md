# apps/studio — implementation contracts

*Design doc only — no app code. Written against the REAL `@websam3/core` /
`@websam3/video-editing` public surfaces and `docs/plans/studio-and-fabri-product.md`
Phase A (authoritative). Mirrors `apps/demo/src/VideoTab.tsx`'s working patterns.*

## 0. API friction flagged up front

1. **React major mismatch.** `apps/demo` is pinned to React 19, but react-konva's
   React-18 major (`^18.2.16`, confirmed via `npm view react-konva@18 version`) has
   no React-19 release. `apps/studio` is a separate app/`package.json` — pin it to
   React `^18.3.1` explicitly; do not copy the demo's version. Flag: if studio and
   demo ever need to share a component, this major split blocks it.
2. **`MaskTimeline.collect` is a static factory only** — no instance method resumes
   draining an iterator into an *existing* timeline. The demo's `VideoTab.tsx`
   works around this with a hand-rolled `drainInto` helper; the studio must
   reimplement the same helper (§4.3). Flag for upstream: an instance
   `timeline.drain(iterator, {epoch, onFrame})` would remove this duplication.
3. **`AlphaMatteExporter` cutout + `webm-vp9-alpha` still throw `NotImplementedError`.**
   MVP export is `{mode:'matte', format:'png-sequence'}` only. MP4 cutout (stretch)
   must hand-composite matte+source into an MP4 via mediabunny — it cannot call
   `exporter.export({mode:'cutout'})`.
4. **`VideoSession.attachSource` only accepts `Blob | HTMLVideoElement`**, and
   `HTMLVideoElement` is `NotImplementedError` until M4. The studio MUST attach the
   raw `File`/`Blob`, never the preview `<video>` element — two independent decode
   paths (worker-side mp4box+WebCodecs; main-thread `<video>` for rVFC preview) for
   the same bytes, by design.
5. **One active `propagate()` iterator per session** — a second call while one is
   in flight rejects `InvalidStateError`. The store must make Track mutually
   exclusive with prompting (mirrors the demo's `trackState.status !== 'running'` guard).
6. **`@websam3/react`'s hooks are stubs.** `src/segmentation/` wraps
   `@websam3/core` directly (matching the demo), not `@websam3/react`.

---

## 1. Package + scripts

`apps/studio/package.json` (new; wave-1 scaffold agent owns it):

```jsonc
{
  "name": "websam-studio", "version": "0.0.0", "private": true, "type": "module",
  "license": "MIT", "engines": { "node": ">=20" },
  "scripts": {
    "dev": "vite", "build": "tsc --noEmit && vite build", "preview": "vite preview",
    "setup-weights": "node scripts/setup-weights.mjs",
    "test": "vitest run --project unit", "test:browser": "vitest run --project browser"
  },
  "dependencies": {
    "@websam3/core": "workspace:^", "@websam3/video-editing": "workspace:^",
    "react": "^18.3.1", "react-dom": "^18.3.1",
    "react-konva": "^18.2.16", "konva": "^9.3.20",
    "@dnd-kit/core": "^6.3.1", "@dnd-kit/sortable": "^10.0.0", "@dnd-kit/utilities": "^3.2.2",
    "zustand": "^5.0.3", "react-resizable-panels": "^2.1.7", "mediabunny": "^1.3.0",
    "class-variance-authority": "^0.7.1", "clsx": "^2.1.1", "tailwind-merge": "^2.6.0",
    "lucide-react": "^0.469.0",
    "@radix-ui/react-slot": "^1.1.1", "@radix-ui/react-dialog": "^1.1.4",
    "@radix-ui/react-select": "^2.1.4", "@radix-ui/react-slider": "^1.2.2",
    "@radix-ui/react-tooltip": "^1.1.6"
  },
  "devDependencies": {
    "@types/react": "^18.3.18", "@types/react-dom": "^18.3.5",
    "@vitejs/plugin-react": "^4.3.4", "@vitest/browser": "^3.2.7", "playwright": "^1.61.1",
    "vitest": "^3.2.7", "typescript": "^5.9.3", "vite": "^6.0.7",
    "tailwindcss": "^3.4.17", "postcss": "^8.5.1", "autoprefixer": "^10.4.20"
  }
}
```

Notes: `vite ^6` (not `apps/demo`'s `^8`) — matches shadcn CLI + Tailwind v3
tooling; bump only after a deliberate check. `mediabunny` (MPL-2.0) is used only
by the stretch MP4-export path; the MVP gate doesn't need it. shadcn/ui isn't an
npm dep — its CLI copies component source into `src/components/ui/*` at scaffold
time; the Radix peers above let `tsc` succeed before the CLI runs.
`tsconfig.json`: standard Vite React-TS strict, `"jsx":"react-jsx"`, path alias
`"@/*":["./src/*"]` (shadcn convention).

---

## 2. The zustand store — `apps/studio/src/store/studio-store.ts`

**Single-owner file.** Every component reads/writes only through this store; no
component holds segmentation state itself except transient DOM refs.

```ts
import type { MaskResult, Prompt } from '@websam3/core';
import { MaskTimeline } from '@websam3/video-editing';

export interface ClipMeta {
  id: string; fileName: string;
  blob: Blob;                    // ORIGINAL file — attachSource needs this exact Blob
  objectUrl: string;             // URL.createObjectURL(blob), for <video src>
  durationSec: number; fps: number; width: number; height: number;
  frameCount: number; frameCountGuessed: boolean;
}

export interface TimelineClip {
  id: string; clipId: string; trackId: string;
  startFrame: number;            // position on the track, project frames
  inFrame: number; outFrame: number; // trim range, source-clip frame indices
}

export interface Track { id: string; order: number; clipIds: string[] }

export type ToolMode = 'select' | 'point-add' | 'point-remove' | 'box' | 'pan';

export interface TrackedObject {
  objectId: number; clipId: string; color: string; label: string; promptFrame: number;
}

export type ModelStatus =
  | { phase: 'idle' }
  | { phase: 'loading'; progress?: { phase: string; loaded?: number; total?: number; file?: string } }
  | { phase: 'ready'; device: 'webgpu' | 'wasm'; quant: string; totalBytes: number }
  | { phase: 'error'; message: string; code?: string };

export type TrackState =
  | { phase: 'idle' }
  | { phase: 'running'; clipId: string; frameIndex: number; frameCount: number }
  | { phase: 'done'; clipId: string }
  | { phase: 'error'; message: string; code?: string };

export type ExportState =
  | { phase: 'idle' }
  | { phase: 'running'; framesDone: number; frameCount: number; kind: 'matte' | 'mp4' }
  | { phase: 'done'; fileName: string; framesExported: number }
  | { phase: 'error'; message: string };

export interface StudioState {
  // media + timeline
  clips: Record<string, ClipMeta>;
  tracks: Track[];
  timelineClips: Record<string, TimelineClip>;
  selection: { timelineClipId: string | null; objectId: number | null };
  playhead: number;              // project frame index (MVP: single track timebase)
  isPlaying: boolean;
  zoom: number;                  // timeline px-per-frame

  // prompting / segmentation UI
  tool: ToolMode;
  activeClipId: string | null;   // clip bound to PreviewCanvas + the live session
  objects: TrackedObject[];
  maskTimelines: Record<string, MaskTimeline>; // by ClipMeta.id
  liveMasksAtFrame: Record<number, MaskResult>; // objectId -> current-frame mask (overlay only)

  // segmenter lifecycle
  modelStatus: ModelStatus;
  resolvedDevice: 'webgpu' | 'wasm' | null;
  trackState: TrackState;
  exportState: ExportState;
  notice: { title: string; detail: string; kind: 'error' | 'warn' } | null;

  // actions
  importClip: (file: File) => Promise<void>;          // metadata probe only; does NOT attachSource
  removeClip: (clipId: string) => void;

  addTrack: () => string;
  addTimelineClip: (clipId: string, trackId: string, startFrame: number) => string;
  moveTimelineClip: (timelineClipId: string, trackId: string, startFrame: number) => void;
  trimTimelineClip: (timelineClipId: string, inFrame: number, outFrame: number) => void;
  removeTimelineClip: (timelineClipId: string) => void;
  reorderTracks: (trackIds: string[]) => void;

  setPlayhead: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setZoom: (pxPerFrame: number) => void;

  setTool: (tool: ToolMode) => void;
  selectTimelineClip: (id: string | null) => void;
  selectObject: (objectId: number | null) => void;

  loadModel: () => Promise<void>;                      // -> segmenter-lifecycle.ts

  activateClip: (clipId: string) => Promise<void>;      // -> session-manager.ts (ensure session + attachSource)
  addPromptObject: (clipId: string, frameIndex: number, prompts: Prompt[]) => Promise<void>;
  refineObject: (clipId: string, objectId: number, frameIndex: number, prompts: Prompt[]) => Promise<void>;
  removeObject: (clipId: string, objectId: number) => void;
  startTracking: (clipId: string, startFrame?: number) => Promise<void>;
  cancelTracking: () => void;

  exportMatte: (clipId: string) => Promise<void>;       // -> export.ts
  exportMp4Cutout: (clipId: string) => Promise<void>;   // stretch; friendly-notice on failure

  setNotice: (notice: StudioState['notice']) => void;
  clearNotice: () => void;
}
```

**Store composition:** zustand with `subscribeWithSelector` (needed for
`PreviewCanvas`'s high-frequency `playhead` subscription — components select
narrow slices, e.g. `useStudioStore((s) => s.playhead)`, to avoid whole-tree
re-renders on every rVFC tick). `MaskResult`/`MaskTimeline` instances live in the
store as **plain object references** (zustand does not deep-clone); mutate a
`MaskTimeline` in place (`.set(...)`) then `set({ maskTimelines: {...get().maskTimelines} })`
(new outer object, same inner instance) to trigger subscribers.
`Segmenter`/`VideoSession` instances are **not** store state — they live in a
module-level `Map<clipId, ClipSession>` inside `session-manager.ts` (§4.2),
keeping the store itself devtools/JSON-inspectable.

---

## 3. Components

All under `apps/studio/src/components/`, each file single-owner (wave 2).

- **`App.tsx`** (wiring owner). Boots the store; mounts `Toolbar` + a
  `react-resizable-panels` layout (`MediaLibrary | PreviewCanvas | PropertiesPanel`
  on top, `Timeline` below). Owns the `<video>`/canvas ref handoff between
  `PreviewCanvas` and `session-manager.ts` (passed as props, not store state).
  Owns the single `<DndContext>` (see DnD note below). Global error boundary +
  `notice` toast rendering.

- **`Toolbar.tsx`** — no props, reads store directly. Tool radio group → `setTool`;
  play/pause → `setIsPlaying`; **Track** → `startTracking` (disabled unless
  `objects.length > 0` and not already running); **Cancel** → `cancelTracking`;
  **Export** dropdown → `exportMatte`/`exportMp4Cutout`; model-status pill with a
  "Load model" CTA when idle; resolved-device badge ("webgpu" or "wasm (slow)").

- **`MediaLibrary.tsx`** — no props. File-drop/pick zone → `importClip`; renders
  `clips` as draggable cards (`@dnd-kit/core` `useDraggable`) with a first-frame
  thumbnail, filename, duration. Drop target is `Timeline`'s tracks via the
  shared `DndContext`.

- **`PreviewCanvas.tsx`** — props `{ videoRef, stageContainerRef }` (refs from
  `App` so the segmentation module and canvas share one `<video>` without a store
  round-trip). Renders an offscreen `<video>` bound to `activeClipId`'s
  `objectUrl` plus a `react-konva` `<Stage>`/`<Layer>` sized to the clip. A
  `requestVideoFrameCallback` loop (rAF fallback) syncs a local frame ref while
  playing, converts video time → project frame via `fps`, calls `setPlayhead`.
  Draws the base frame as a Konva `<Image>` fed directly from the `<video>`
  element (Konva accepts `HTMLVideoElement` as an image source — no intermediate
  canvas). Renders `liveMasksAtFrame` as semi-transparent colored overlay
  `<Image>` nodes (`MaskResult.toImageData()` → `ImageBitmap`, cached per
  `${objectId}:${frameIndex}` to avoid re-decoding every tick). Renders
  point/box prompts for the selected object as Konva `<Circle>`/`<Rect>` (green
  positive, red negative, dashed box-in-progress). Pointer handlers translate
  stage-local → source-pixel coordinates (`stageX * (clipWidth/stageDisplayWidth)`,
  mirroring the demo's canvas-scale math) and dispatch per `tool`:
  `point-add`/`point-remove` → `addPromptObject`/`refineObject`; `box` →
  drag-to-draw, emits `{type:'box',...}` on pointerup; `select`/`pan` → no
  segmentation call.

- **`PropertiesPanel.tsx`** — no props. Shows properties of `selection`: trim
  in/out steppers → `trimTimelineClip` for a selected clip; object label, mask
  opacity slider (default 0.5, matches the demo's alpha=128), **Remove object** →
  `removeObject` for a selected object. Export-settings sub-panel mirrors the
  Toolbar's Export actions (same store calls, no duplicated logic).

- **`Timeline.tsx`** — no props. Renders `tracks` as lanes with `timelineClips`
  as `@dnd-kit/sortable` items (multi-container sortable: `useDroppable` per
  track + `useSortable` per clip, via the App-owned `DndContext`). Custom trim
  handles (hand-rolled pointer events, not dnd-kit — it does drag/reorder, not
  resize): `onPointerDown/Move/Up` compute frame deltas from `zoom` and call
  `trimTimelineClip`. Draggable playhead scrubber synced to `playhead`
  (`setIsPlaying(false)` on scrub-start). Ruler ticks derived from `zoom`.

**DnD ownership:** `App.tsx` owns the single `<DndContext>` and its `onDragEnd`
(dispatches `addTimelineClip` for MediaLibrary→Track, `moveTimelineClip` for
Track→Track/reorder, keyed on `event.active.data.current.kind`).
`MediaLibrary.tsx`/`Timeline.tsx` only register draggables/droppables — neither
owns `onDragEnd`, avoiding a two-file split-brain over one drag session.

---

## 4. `src/segmentation/` — the `@websam3/core` integration seam

**4.1 `segmenter-lifecycle.ts`** — owns `createSegmenter()` once per app session
(not per clip). Exports `loadSegmenter(onProgress): Promise<Segmenter>`, memoized
module-level so `store.loadModel()` is idempotent. Config mirrors the demo:
`{ model: 'edgetam', device: 'auto', modelBaseUrl: '/models/', workerUrl:
segmenterWorkerUrl /* from '@websam3/core/worker?worker&url' */, onProgress }`.
On success sets `modelStatus = {phase:'ready', device: segmenter.device, ...}`
and `resolvedDevice = segmenter.device`.

**4.2 `session-manager.ts`** — module-level `Map<clipId, {session, attached}>`.
`activateClip(clipId)` gets-or-creates a `VideoSession` via
`segmenter.createVideoSession()`, then `session.attachSource(clipMeta.blob)`
(the original Blob — friction §0.4). Creates the clip's `MaskTimeline` on first
activation from `attachSource`'s returned `{frameCount, fps, width, height}`.
Only one clip is "active" (bound to the shared preview `<video>`/session) at a
time in MVP; switching clips doesn't dispose — sessions stay warm in the Map,
disposed only on `removeClip`/unmount. `addPromptObject` calls
`session.addObject(...)`, pushes a `TrackedObject`, sets
`liveMasksAtFrame[objectId]`. `refineObject` calls `session.refineObject(...)`
then `maskTimeline.invalidateAfter(String(objectId), frameIndex)`, stashing the
returned epoch (module-level `Map<objectId, epoch>`) for the next
`startTracking` resume — mirrors the demo's `trackEpochRef`. `removeObject`
calls `session.removeObject(objectId)` and filters `store.objects`.

**4.3 `propagate-loop.ts`** — `drainInto(frames, timeline, epoch, onFrame)`: a
verbatim reimplementation of the demo's helper (friction §0.2) — for each
yielded `FramePropagationResult`, `timeline.set(String(mask.objectId),
frame.frameIndex, mask.toRLE(), epoch)`, then `onFrame(frame)`.
`startTracking(clipId, startFrame?)`: (1) aborts+clears any existing
`AbortController` first; refuses if already running (friction §0.5); (2) new
`AbortController` kept in a module-level ref (not store state — not
serializable); (3) `session.propagate({startFrame: startFrame ?? playhead,
signal})`; (4) `drainInto(iterator, maskTimeline, epochFor(clipId), frame =>
{ update playhead, liveMasksAtFrame, trackState.frameIndex })`; (5) catches
`EpochInvalidatedError` (friendly notice, `trackState→idle`), `AbortError`
(`trackState→idle`, silent), else `trackState→error` + notice.
`cancelTracking()` calls `controller.abort()`.

**4.4 `src/video/frame-source.ts`** — read-only probing only; the worker's
mp4box+WebCodecs pipeline inside `@websam3/core` stays the source of truth for
segmentation-relevant decoding. `probeClipMeta(file)` — used at `importClip`
time (before any session exists) via a throwaway `<video>`'s `loadedmetadata`,
so `MediaLibrary` can show duration/thumbnails pre-model-load; `fps` is
estimated here and corrected once `attachSource` runs (`frameCountGuessed`,
mirrors the demo's `Math.round(duration*fps)` fallback). `captureFrameBitmap(video,
atTime)` — seek + `createImageBitmap(video)` for thumbnails, same pattern as
the demo's `seekVideoTo`. No WebCodecs `VideoDecoder` code lives here for MVP;
flagged as where a future frame-accurate multi-clip timeline would grow.

**4.5 `export.ts`** — `exportMatte(clipId)`:
`new AlphaMatteExporter(maskTimelines[clipId]).export({mode:'matte',
format:'png-sequence', onProgress})` → download blob, exactly the demo's
`handleExport`. This is the MVP path and the only one the integration test
depends on. `exportMp4Cutout(clipId)` (**stretch**, does not block the A4
gate): does NOT call the exporter's cutout mode (friction §0.3) — instead
captures source frames via `<video>`+rVFC (or WebCodecs `VideoDecoder` if perf
demands), composites RGBA per frame in a 2D canvas using
`MaskTimeline.get(objectId, frameIndex)` → `decodeRLE` → alpha (same by-hand
compositing the demo uses in place of the still-unimplemented
`MaskCompositor`), then muxes via mediabunny's `Output`/`CanvasSource` into an
MP4. Isolated in its own file/try-catch so a failure here never touches
`trackState` or the matte path.

---

## 5. Weight setup script + vite config

**`scripts/setup-weights.mjs`** (Node ESM, `node:fs/promises`+`node:path` only):
1. `SRC = repoRoot/tools/goldens/models-cache/edgetam`.
2. Fail loudly (non-zero exit, message naming the regen command) if `SRC` or its
   `manifest.json` is missing — same never-silently-skip rule as
   `packages/core/src/e2e/video-golden.browser.test.ts`'s `requireModels()`:
   `Run: cd tools/goldens && ../export/.venv/bin/python make-video-golden.py`.
3. `fs.cp(SRC, apps/studio/public/models/edgetam, {recursive:true, force:true})`
   — clean overwrite.
4. Print total bytes + manifest tier/quant.

`apps/studio/public/models/` is gitignored (`apps/studio/.gitignore`: `models/`).
Wired as `pnpm -F websam-studio setup-weights`; a required step before
`dev`/`build`/`test:browser` (documented in `apps/studio/README.md`, wave 3).

**`vite.config.ts`**:
```ts
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
} as const;

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },  // module workers for @websam3/core's worker (onnxruntime-web dynamic import)
  server: { headers: crossOriginIsolationHeaders },
  preview: { headers: crossOriginIsolationHeaders },
});
```
Same rationale as `apps/demo/vite.config.ts` (COOP/COEP → `crossOriginIsolated`
→ SharedArrayBuffer → multithreaded WASM). `modelBaseUrl` stays `'/models/'`
(not `'/models/edgetam/'`) — the registry appends the model-id subpath itself;
mirror the demo's constant exactly.

Tailwind: standard `tailwind.config.ts` content globs over `index.html` +
`src/**/*.{ts,tsx}`; `postcss.config.js` with tailwindcss+autoprefixer;
shadcn/ui `components.json` → `src/components/ui`, `src/lib/utils.ts` (the
`cn()` helper), CSS-var theme in `src/index.css`.

---

## 6. Integration test plan

### 6.1 Browser vitest gate — `src/segmentation/segmentation.browser.test.ts`

New `apps/studio/vitest.config.ts`, mirroring `packages/core/vitest.config.ts`'s
`unit`/`browser` project split (studio has no `e2e` project — the weights
dependency is handled via a loud precondition check instead, see below):
```ts
projects: [
  { test: { name: 'unit', environment: 'node', include: ['src/**/*.test.ts'], exclude: ['src/**/*.browser.test.ts'] } },
  {
    server: { fs: { allow: [repoRoot] } },        // out-of-root fixture serving, same as packages/core
    optimizeDeps: { exclude: ['onnxruntime-web'] },
    test: { name: 'browser', include: ['src/**/*.browser.test.ts'],
      browser: { enabled: true, provider: 'playwright', headless: true, instances: [{ browser: 'chromium' }] } },
  },
]
```
`repoRoot = fileURLToPath(new URL('../..', import.meta.url))` (same relative
depth from `apps/studio/` as `packages/core/`).

The test boots the studio's OWN `src/segmentation/` module (proving the seam,
not just core), against:
- **Weights**: `apps/studio/public/models/edgetam/` — requires
  `setup-weights` to have run; the test fetches `/models/edgetam/manifest.json`
  first and throws naming the setup command if missing (never silently skip).
- **Clip**: `tools/goldens/fixtures/video/clip-256.mp4` via
  `import clipUrl from '../../../../tools/goldens/fixtures/video/clip-256.mp4?url'`.
- **Golden masks**: `golden-mask-f0..f9.rle.json` + `golden-video-meta.json`
  (prompt point + `iouGate`), fetched the same way as the core gate.

**Assertions:**
1. `loadSegmenter()` resolves; `segmenter.device` is `'webgpu'` or `'wasm'`.
2. `activateClip` attaches successfully; `frameCount === meta.clip.numFrames`.
3. `addPromptObject` at `meta.prompt.frameIndex` with the golden point produces
   a mask with IoU vs `golden-mask-f{promptFrame}` **>= 0.85** — looser than
   core's 0.90, deliberately, to allow the studio's extra coordinate round-trip
   (stage-to-source scaling) and fps-estimate path.
4. `startTracking` drains `propagate()` into the `MaskTimeline` via
   `drainInto`; every yielded frame's IoU vs its golden mask is **>= 0.85**.
5. `maskTimeline.holes(String(objectId))` is empty over `[0, frameCount)` after
   tracking — a structural non-empty-tracked-mask assertion, not just "some
   mask exists".
6. Timeout **900_000 ms** (15 min), matching the core gate's wasm-worst-case
   budget (10 frames x 4 graphs at 1024 input, single-threaded wasm); `device:
   'auto'` may resolve `webgpu` and finish faster in a GPU-enabled runner.
7. Like the core e2e gate, this project is **not** part of `apps/studio`'s
   CI-required `build`. CI runs `pnpm -F websam-studio build` only (no
   weights needed for typecheck+bundle). `test:browser` is a local/manual
   gate, run after `setup-weights` — the plan's A4 "manual Chrome run"
   acceptance check.

### 6.2 Store-logic unit tests (node) — `src/store/studio-store.test.ts`

Pure-node `vitest` (`unit` project), no browser, no segmenter — fake
`MaskTimeline`/`MaskResult`-shaped stubs where needed (the async segmentation
calls themselves are covered by §6.1, not re-tested here):
- `addTrack`/`addTimelineClip`/`moveTimelineClip`/`trimTimelineClip`/
  `removeTimelineClip`: correct shape, no orphaned ids after removal.
- `trimTimelineClip` clamps `inFrame < outFrame` within `[0, frameCount)`.
- `setPlayhead` clamps to `[0, projectDuration)`.
- `selectTimelineClip`/`selectObject` are mutually exclusive.
- `setTool` is a no-op for `point-add`/`point-remove`/`box` while
  `trackState.phase === 'running'` — encodes friction §0.5 as a store
  invariant, testable without a real session.
- `exportState`/`trackState`/`modelStatus` transitions (idle→running→done/error).

Run via `pnpm -F websam-studio test` (CI-safe, no weights, no browser).

---

## 7. Ownership table + wave plan

| File | Wave | Owner |
|---|---|---|
| `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `tailwind.config.ts`, `postcss.config.js`, `src/index.css`, `src/main.tsx`, `.gitignore` | 1 | scaffold agent |
| `scripts/setup-weights.mjs` | 1 | weights agent |
| `src/store/studio-store.ts` | 1 | store agent |
| `src/segmentation/segmenter-lifecycle.ts`, `session-manager.ts`, `propagate-loop.ts`, `export.ts`, `src/video/frame-source.ts` | 1 | segmentation agent |
| `src/components/Toolbar.tsx` | 2 | toolbar agent |
| `src/components/MediaLibrary.tsx` | 2 | media-library agent |
| `src/components/PreviewCanvas.tsx` | 2 | preview agent |
| `src/components/PropertiesPanel.tsx` | 2 | properties agent |
| `src/components/Timeline.tsx` | 2 | timeline agent |
| `src/App.tsx`, `src/components/ui/*` (shadcn-generated) | 2 | **App/wiring owner** |
| `vitest.config.ts`, `src/segmentation/segmentation.browser.test.ts`, `src/store/studio-store.test.ts`, `README.md` | 3 | integration-test agent |

**Wave 1 (parallel, disjoint):** scaffold agent (project skeleton + a minimal
placeholder `App.tsx` so `tsc --noEmit` resolves all imports); weights agent
(`setup-weights.mjs`); store agent (`studio-store.ts` — the seam every wave-2
component needs, must land this wave, reviewed against §2 exactly);
segmentation agent (`src/segmentation/*` + `src/video/frame-source.ts` —
depends only on the store's *type* shape from §2, not on any component).
Gate: `pnpm -F websam-studio build` green with the placeholder `App.tsx`.

**Wave 2 (parallel, disjoint, depend on wave-1 store + segmentation types):**
Toolbar/MediaLibrary/Preview/PropertiesPanel/Timeline agents — five
components, each reading/writing the store per §3, no cross-component
imports except shared `src/components/ui/*` primitives (scaffold agent
pre-generates these via the shadcn CLI in wave 1 to avoid a wave-2 shared-file
race). App/wiring owner writes the real `App.tsx` (wires all five + the
`DndContext` + ref handoff) — can run in the same wave-2 batch since it only
needs this doc's fixed prop contracts, not the components' implementations.
Gate: `pnpm -F websam-studio build` green (real components); manual `pnpm -F
websam-studio dev` smoke click-through (import a clip → prompt a point → see
a mask).

**Wave 3 (parallel, disjoint):** integration-test agent
(`vitest.config.ts` + both test files + README). Optional: an
`/agentic-design` polish pass per the global CLAUDE.md default for UI-craft
work (out of scope for this doc to spec further).
Gate (= the plan's A4 milestone gate): `pnpm -F websam-studio build` green in
CI; `pnpm -F websam-studio test` (unit) green in CI; `pnpm -F websam-studio
setup-weights && pnpm -F websam-studio test:browser` green manually
(weights-dependent, not CI — same split as `packages/core`'s `test:browser`
vs `test:e2e`).

---

## 8. Key decisions summary

- React 18 (not 19) for `apps/studio`, independent of `apps/demo`'s React 19 —
  react-konva's constraint.
- One `VideoSession` per source clip (module-level `Map`), lazily activated;
  MVP activates (binds to the shared preview `<video>`) only one clip at a time.
- `attachSource` always gets the original `Blob`, never the preview `<video>`
  element (friction §0.4) — two independent decode paths by design.
- MVP export is PNG-sequence matte only; MP4 cutout is an isolated,
  best-effort stretch module that hand-composites rather than depending on
  the still-unimplemented exporter cutout mode.
- `drainInto` is duplicated (not imported) from the demo's pattern because
  `MaskTimeline` has no public resume-into-existing-timeline instance method
  (friction §0.2) — flagged for upstream.
- Single shared `<DndContext>` owned by `App.tsx` for both
  MediaLibrary→Timeline and intra-Timeline drags.
- Integration-test IoU gate: 0.85 (studio) vs 0.90 (core), justified by the
  extra coordinate/estimation round-trips the studio adds atop the core's
  already-gated pipeline.
