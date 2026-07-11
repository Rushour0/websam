# websam-studio

An in-browser interactive video segmentation editor built on `@websam3/core`
(SAM-family segmentation) and `@websam3/video-editing` (mask timelines +
export). Import a clip, prompt objects with point clicks, track them across
the clip, and export an alpha matte — all client-side, no server round-trip
for inference.

Design doc: [`docs/studio-contracts.md`](./docs/studio-contracts.md) — read
that first for the store shape, component ownership, and the segmentation
integration seam. This README covers running and testing the app.

## Quick start

```sh
pnpm install                       # from the repo root
pnpm -F websam-studio setup-weights
pnpm -F websam-studio dev
```

Open the printed local URL in a Chromium-based browser (WebGPU support is
Chromium-first as of this writing; Firefox/Safari fall back to WASM, see
below).

### `setup-weights` — why it's a separate, required step

Model weights are **not committed** to the repo (`apps/studio/public/models/`
is gitignored — see `apps/studio/.gitignore`). `setup-weights` copies the
EdgeTAM ONNX graphs + `manifest.json` from the shared golden models cache
(`tools/goldens/models-cache/edgetam/`) into `apps/studio/public/models/edgetam/`
so the dev server / build / browser test suite can fetch them from
`/models/edgetam/*` at runtime. It fails loudly (non-zero exit, naming the
regen command) if the source cache is missing — it never silently produces
an empty models directory. Run it once before `dev`, `build`, or
`test:browser`; re-run it any time the golden cache is regenerated.

If the source cache itself doesn't exist yet:

```sh
cd tools/goldens && ../export/.venv/bin/python make-video-golden.py
```

(requires the `tools/export` Python venv + `ffmpeg` on `PATH`).

## COOP/COEP — why the dev server sets cross-origin-isolation headers

`apps/studio/vite.config.ts` sets `Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: require-corp` on both `server` and
`preview`. These are required for `crossOriginIsolated === true`, which in
turn unlocks `SharedArrayBuffer` and therefore **multithreaded WASM**
inference. Without them, `onnxruntime-web`'s WASM execution provider falls
back to a single thread — segmentation still works, just slower. This mirrors
`apps/demo/vite.config.ts`'s setup exactly. If you reverse-proxy the dev
server or deploy behind a CDN, you must forward these headers yourself or
WASM inference silently degrades (no error — just slower).

## WebGPU vs. WASM

`segmenter-lifecycle.ts` always requests `device: 'auto'`: `@websam3/core`
probes for a WebGPU adapter and falls back to WASM if none is available or
adapter creation fails. The resolved device is surfaced in the store as
`resolvedDevice` and shown as a badge in the Toolbar ("webgpu" or "wasm
(slow)"). WebGPU is substantially faster for the video memory-attention loop
(4 graphs × N frames per `Track` run); WASM is the universal fallback and
what CI-style headless testing exercises deterministically. There is
currently **no user-facing override** to force one device over the other —
flagged in `docs/studio-contracts.md` §4.1 as a gap if a future need arises
(e.g. a forced-WASM debug mode, or a forced-WASM leg in the browser test —
see "Known limitation" under Testing below).

## The bundled model

The studio ships a single tier: **EdgeTAM** (Apache-2.0, no license-acceptance
gate — unlike SAM2-licensed tiers). `MODEL_BASE_URL` defaults to
`/models/edgetam/`, overridable via `VITE_WEBSAM_MODELS` for
self-hosted/CDN deployments. The segmenter is loaded once per app session
(`loadSegmenter()` in `src/segmentation/segmenter-lifecycle.ts` is a
memoized singleton), not once per clip — switching the active clip reuses
the already-loaded model and simply attaches a new `VideoSession`.

## Running the tests

Two independent vitest projects (`apps/studio/vitest.config.ts`):

```sh
# Unit: pure-node store reducer tests. No browser, no model weights, no
# segmenter — safe to run in CI on every push.
cd apps/studio && pnpm exec vitest run --project unit
# equivalently: pnpm -F websam-studio test

# Browser: the real end-to-end segmentation integration gate. Headless
# Chromium via Playwright, the bundled EdgeTAM weights, and the committed
# golden clip. NOT part of CI (weights-dependent) — run locally/manually.
pnpm -F websam-studio setup-weights   # if not already staged
cd apps/studio && pnpm exec vitest run --project browser
# equivalently: pnpm -F websam-studio test:browser
```

### `src/store/studio-store.test.ts` (unit)

Pure reducer-logic tests of the zustand store per
`docs/studio-contracts.md` §6.2: timeline clip CRUD (`addTimelineClip` /
`moveTimelineClip` / `trimTimelineClip` / `removeTimelineClip`), track
reordering, `removeClip` leaving no orphaned references, playback/zoom
setters, tool/selection state, and the `modelStatus` / `trackState` /
`exportState` state-machine shapes. The async segmentation actions
themselves (`activateClip`, `addPromptObject`, `startTracking`, ...) are
**not** re-tested here — they're covered end-to-end by the browser gate.

**Two tests in this file currently FAIL against the real store** — they
encode contract invariants from `docs/studio-contracts.md` §6.2 that the
current `src/store/studio-store.ts` implementation does not yet satisfy:

1. *"`setTool` is a no-op for prompting tools while tracking is running"*
   (§6.2, encodes friction §0.5) — the real `setTool` unconditionally sets
   the tool regardless of `trackState.phase`.
2. *"`selectTimelineClip`/`selectObject` are mutually exclusive"* (§6.2) —
   the real actions each only touch their own field of `selection`, so
   selecting an object while a timeline clip is selected leaves both set.

These are real gaps in `studio-store.ts` against its own contract, not test
bugs — left failing on purpose (never weakened to pass) so they're visible
and actionable; see the PROGRESS/handoff notes for who owns the fix.

### `src/segmentation/segmentation.browser.test.ts` (browser)

The real end-to-end pipeline: `segmenter-lifecycle.ts`'s `loadSegmenter()`
→ `session-manager.ts`'s `activateClip`/`addPromptObject` → the golden prompt
point (`x=60, y=128, label=1` on frame 0, same as `packages/core`'s M2 gate)
→ `propagate-loop.ts`'s `startTracking` (which drains `session.propagate()`
into a `MaskTimeline` via its own `drainInto` reimplementation) — compared
frame-by-frame against the committed HF `EdgeTamVideoModel` fp32 reference
masks (`tools/goldens/fixtures/video/golden-mask-f{0..9}.rle.json`) at
**IoU ≥ 0.85** (looser than `packages/core`'s 0.90 — the studio seam adds an
extra coordinate round-trip atop core's already-gated pipeline).

**Latest run (WASM, this environment): every frame 0.989–0.995 IoU** — the
segmentation pipeline itself is solid, well above the 0.85 studio bar.

**One structural assertion currently FAILS** (`docs/studio-contracts.md`
§6.1 assertion 5: `maskTimeline.holes(objectId, [0, frameCount))` must be
empty after tracking): it returns `[0]` — frame 0 (the prompt frame) is
missing from the `MaskTimeline`. Root cause, precisely: `session-manager.ts`'s
`addPromptObject` stores the newly-prompted object's mask only in
`liveMasksAtFrame` (the live overlay cache) — it never calls
`maskTimeline.set(...)`. Only `propagate-loop.ts`'s `drainInto` writes into
the `MaskTimeline`, and `propagate()` is (correctly, per
`packages/core`'s own M2 gate) only ever driven from `promptFrame + 1`
onward — see the next paragraph for why. The net effect: **every freshly
prompted object's own prompt frame is a permanent gap in its exported mask
timeline** — the PNG-sequence matte export would be missing that frame.

**A second, independent pipeline bug was found while isolating the above**:
calling `session.propagate({startFrame: promptFrame})` *immediately* after
`session.addObject({frameIndex: promptFrame})` — i.e. resuming `Track` from
the exact frame that was just prompted, with the playhead unmoved — is
documented as the real usage pattern (`apps/demo/src/VideoTab.tsx`'s Track
button always passes `currentFrameIndexRef.current`, and
`packages/core/docs/m2-internal-contracts.md` §8 states "Track resumes with
`startFrame = currentFrame`"), but returns an **effectively empty mask**
(IoU ≈ 0.0 vs. the golden reference) instead of reproducing `addObject`'s own
0.989-IoU mask at that identical frame. This test deliberately starts
`startTracking` at `promptFrame + 1` (matching `packages/core`'s own M2
gate) specifically to avoid cascading that bug into every subsequent IoU
assertion, so the `holes()` failure above is reported in isolation.

Both are real, reproducible defects in the segmentation/session-manager
layer (not in this test) — reported for the orchestrator to route to the
owning file(s), not silently patched around here.

**Known limitation — device coverage.** `segmenter-lifecycle.ts` hardcodes
`device: 'auto'` with no override knob, so this gate cannot force a
deterministic `'wasm'` leg the way `packages/core`'s M2 gate does (that gate
calls `createSegmenter` directly, bypassing any app-level singleton). It
hard-gates whichever device `'auto'` resolves in the running environment
(observed here: `wasm`, `quant=int8`) — satisfying
`docs/studio-contracts.md` §6.1 assertion 1 ("`segmenter.device` is
`'webgpu'` or `'wasm'`") exactly, but not the stronger two-leg
(forced-wasm + soft-skip-webgpu) structure `packages/core`'s gate has. A
`device` override surfaced from `loadSegmenter()` (or an env knob, mirroring
`VITE_WEBSAM_MODELS`) would let a future revision add that without
bypassing the studio's own segmentation seam.

## Honest state (what works, what's a stretch, what's caveated)

- **Works, browser-gate-verified:** the interactive segmentation pipeline
  itself — prompt a point, get a mask, track it across every frame — at
  0.99 IoU against the HF reference on this environment's resolved device.
- **Works:** PNG-sequence alpha-matte export (`exportMatte`,
  `src/segmentation/export.ts`) — the MVP export path, a direct port of
  `apps/demo/src/VideoTab.tsx`'s `handleExport`.
- **Stretch, not implemented:** MP4 cutout export (`exportMp4Cutout`) always
  throws `NotImplementedError`, surfaced as a friendly warning notice — it
  never touches `trackState`/blocks the matte path. The exporter's own
  `'cutout'` mode and `'webm-vp9-alpha'` format both still throw
  `NotImplementedError` upstream in `@websam3/video-editing`; a real MP4
  cutout would need to hand-composite matte+source via `mediabunny` instead.
- **Caveat — `probeClipMeta` in headless Chromium:** `src/video/frame-source.ts`'s
  `probeClipMeta` (used by `importClip` for pre-model-load duration/dimension
  display) drives an `HTMLVideoElement`'s `loadedmetadata` event, which is
  flaky in headless Chromium (no real display compositor/decoder backing in
  some CI sandboxes). The segmentation browser gate above deliberately
  avoids this path entirely — it constructs `ClipMeta` directly from the
  golden fixture's known metadata and drives `activateClip`/`addPromptObject`/
  `startTracking` straight through `src/segmentation/*`, the proven-headless
  path (mp4box + WebCodecs inside the worker). If you see `importClip` hang
  or mis-report duration/dimensions specifically in headless/CI browser runs
  (not in a real browser), suspect this probe first.
- **Two real, precisely-reproducible bugs found by the integration gate**
  (see above): (1) `addPromptObject` never writes the prompt frame's mask
  into the clip's `MaskTimeline`, and (2) `propagate({startFrame})` at the
  exact frame just prompted returns a near-empty mask instead of the
  correct one. Both are reported for the orchestrator to route, not fixed
  in this wave (wave 3 owns tests only — `docs/studio-contracts.md` §7).
