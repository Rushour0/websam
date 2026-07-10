# websam — progress log & resume handoff

**Status as of 2026-07-09: M0, M1, M2 all COMPLETE, gated green, committed and pushed to `main`.**
Packages are published to npm as `@websam3/{core,react,video-editing}@0.0.1`. The project is
otherwise paused; next up is weight hosting (see TODO below) so the published packages are
runnable out of the box, then M3.

Repo: https://github.com/Rushour0/websam (public, MIT).
Plan of record: `~/.claude/plans/lets-build-something-from-goofy-dragon.md` (plan v2, post 4-angle
adversarial review).

## Operating rules for resuming

- **All subagents/workflow agents run on Sonnet 5** (`model: 'sonnet'`), per user directive, until
  **2026-07-14**. Not Fable.
- Node **22** required (nvm: `. ~/.nvm/nvm.sh && nvm use 22`). Node 23 breaks jsdom/tsdown engine
  ranges. `.nvmrc` pins 22.
- Delivery pattern: spikes/design first → parallel disjoint-file apply agents → one consolidating
  gate (build + typecheck + unit + browser + Python) → commit. Never one agent editing shared files
  concurrently.
- Gate command set: `pnpm build`, `pnpm typecheck`, `pnpm test`, `cd packages/core && pnpm test:browser`,
  `pnpm -F @websam3/core test:e2e` (real-weights golden gate, run locally — weights are gitignored,
  not fetched in CI), `cd tools/export && uv run --group dev pytest -q`,
  `pnpm -r --filter './packages/*' exec publint`, `pnpm -r --filter './packages/*' exec attw --pack . --profile esm-only`.
- Test project split: vitest `unit` + `browser` projects run in CI (`pnpm test` /
  `test:browser`); the `e2e` project (real model weights, golden-mask parity) is **not** wired into
  CI — it needs weight files present locally (`tools/goldens/fetch-models.mjs` /
  `make-video-golden.py`) and is run by hand before milestone gates.
- CI (`.github/workflows/ci.yml`): `build-and-test` (build, publint, attw, unit tests, demo/bundler-matrix
  app builds) + `browser-tests` (vitest browser project, xvfb+SwiftShader for the WebGPU lane). Both
  green on `main`.
- Release workflow (`.github/workflows/release.yml`) is **`workflow_dispatch`-only** — no
  auto-publish on push — until npm trusted publishing is wired up at M4.

## Milestone status

### M0 — scaffold + contracts ✅ (commit `fa8a296`)

pnpm monorepo; `@websam3/core` (Backend abstraction, error taxonomy, coords, RLE+COCO, registry,
Segmenter/session contracts), `@websam3/video-editing` (MaskTimeline), `@websam3/react`
(StrictMode-safe `useSegmenter`), demo shell w/ capability probes, bundler-matrix (Vite+webpack),
`tools/export` (Python/uv). Gate green: build+typecheck+64 unit+browser ONNX smoke+42 py+publint+attw.
CI uses xvfb+SwiftShader for the WebGPU browser lane.

### M1 — interactive image path ✅ (commits `68bb431` prep, `ff30dde` impl)

- **S0 spike verdict REUSE-OK**: community `onnx-community/sam3-tracker-ONNX` vision encoder emits
  all 3 FPN outputs the decoder needs. Coordinate mode **pinned = square-stretch** (anisotropic
  1008×1008, mean=std=0.5, per-axis prompt scaling); decoder logits 288×288. See
  `tools/export/spikes/s0/FINDINGS.md`.
- **Golden fixtures**: deterministic transformers.js (q4f16) reference masks on a 640×427
  non-square scene → `tools/goldens/fixtures/`.
- Full pipeline real: manifest-driven weight loading (streaming SHA-256, OPFS/Cache-API
  content-addressed stores, Range-resume), ORT runtime + device/quant resolution + real
  WebGPU/WASM Backend bodies, Comlink module worker (preprocess → encoder → per-click decoder →
  288→source mask), `createSegmenter`/`ImageSession`, `MaskResultImpl`, demo image tab.
- **Gate green**: 258 unit + 20 browser tests. **M1 golden gate PASSED**: pipeline reproduces
  transformers.js masks at **IoU 0.9957 / 0.9962** (bar 0.9) on the wasm EP. `dist/worker.js`
  ships as `@websam3/core/worker`.

### M2 — EdgeTAM video + rotobrush demo ✅ (commits `d727d2f`, `5ac6566`, `b92bb95`, `2032554`;
scope rename `368d65f`)

In-browser EdgeTAM video object tracking, end to end, real weights:

- **wave 1** (`d727d2f`) — backend memory primitives (`copyRegion`/`debugStats`), memory-bank +
  tracking strategy pattern, WebCodecs-in-worker frame source, PNG-zip alpha-matte exporter
  scaffolding.
- **wave 2** (`5ac6566`) — video engine (memory-attention loop) + `VideoSession` (pull-credit
  MessagePort `propagate()` protocol, epoch state machine — `refineObject` bumps the epoch, an
  in-flight iterator throws `EpochInvalidatedError` rather than silently stopping).
- **wave 3a** (`b92bb95`) — production EdgeTAM export pipeline + goldens (`make-video-golden.py`,
  `EdgeTamVideoModel` reference, `tools/export/src/websam_export/wrappers/edgetam.py`), demo
  video tab (`apps/demo/src/VideoTab.tsx`: file → click frame 0 → track → live overlay → refine →
  export matte.zip).
- **wave 3b** (`2032554`) — **in-browser video tracking GATE GREEN**: the real websam pipeline
  reproduces the HF `EdgeTamVideoModel` reference at **IoU 0.989–0.995 across 10 frames** (bar
  0.90/frame), plus CI fixes.
- **scope rename** (`368d65f`) — npm scope moved `@websam` → `@websam3` (the org actually owned);
  all package names, imports, and manifests updated.
- Model registry facts: EdgeTAM (`id: 'edgetam'`) is Apache-2.0, no license gate, `webgpu`+`wasm`
  device support, 1024px square-stretch + ImageNet mean/std preprocessing, 512-token memory
  (KV_LEN = 7×512+64 = 3648). Contracts: `packages/core/docs/m2-internal-contracts.md`.

### Published to npm

`@websam3/core@0.0.1`, `@websam3/react@0.0.1`, `@websam3/video-editing@0.0.1` — public access, no
provenance yet (trusted publishing lands at M4). Weights are **not** part of the published
packages (code-only, MIT).

## Known API surface gaps (accurate as of this doc)

- `useImageSession` / `useVideoSession` in `@websam3/react` are **still stubs** — they throw
  `NotImplementedError`. Only `useSegmenter` is implemented. Consumers call
  `segmenter.createImageSession()` / `createVideoSession()` directly (see
  `apps/demo/src/{ImageTab,VideoTab}.tsx`, `examples/quickstart/src/main.ts`).
- `AlphaMatteExporter.export({ mode: 'cutout' })` and `{ format: 'webm-vp9-alpha' }` both throw
  `NotImplementedError` — only `{ mode: 'matte', format: 'png-sequence' | 'auto' }` works today
  (`'auto'` resolves to `'png-sequence'` until VP9-alpha detection lands at M4).
- No model weights are hosted publicly — `modelBaseUrl` needs a local or self-hosted manifest
  (see root `README.md` → "Providing weights", `examples/quickstart/README.md`).

## TODO — resume order

1. **Weight hosting** (blocks the published packages actually running for anyone): convert/merge
   external-data graphs (the export spike script already does this), publish EdgeTAM weights to a
   public host (Apache-2.0 — clean to host; SAM3 stays user-supplied pending legal check), wire
   `modelBaseUrl` defaults / a documented CDN.
2. **M3** — SAM3-tracker video tier (the research-grade spike ladder S0–S6, off launch critical
   path).
3. **M4** — full editing layer: WebGPU compositor, cutout mode, VP9-alpha export, `useImageSession`/
   `useVideoSession` real implementations, docs polish, `0.1.0` publish. Needs the `@websam3` npm
   scope + trusted publishing wired (unblocks non-`workflow_dispatch` releases) and a SAM3 weight
   re-hosting legal check.
4. **M5** — [fabri](https://github.com/Rushour0) integration via `OrtNodeBackend` + VLM
   text→prompt.

## Known open items / risks

- SAM3 weight re-hosting legally unsettled — EdgeTAM-first neutralizes; SAM3 default = user runs
  the export pipeline on their own gated download.
- 51.9k-token SAM3 memory attention may be bandwidth-bound on WebGPU (M3 S5 measures; 560 is the
  multi-object default).
- iOS Safari = EdgeTAM-only experimental; WASM video = EdgeTAM-int8 only; PNG-zip is the portable
  matte export (VP9-alpha is Chrome-only, and not implemented yet regardless — M4).
- `examples/quickstart` (`db1d736`) — minimal vanilla-TS + Vite getting-started app on the real
  `@websam3/core` API; now built by CI (`build-and-test` builds `./apps/**` + `./examples/**`).

## Active workstream — Studio product + fabri (started 2026-07-10)

Direction set by the user: build a **small in-browser video Studio** (React) where someone can
view / edit / drag videos on a timeline, drive segmentation (rotobrush), and export — and land it
as a **full product with fabri agent orchestration in it** (fabri = the user's agent engine at
`/Users/rushour0/gba/fabri`, Python/BUSL-1.1). This promotes M5 to active and adds a Studio app.

Scope in flight (Sonnet scoping agents running as of 2026-07-10):
- **fabri internals** — how its tool/domain system + agent loop work, and the cleanest way to add a
  `video_editing` tool domain that drives websam WITHOUT touching fabri core (BUSL).
- **React studio stack** — pick high-support packages (drag/timeline, frame-accurate playback,
  canvas overlay, state, export) and an MVP component architecture (MediaLibrary / Timeline /
  Preview-Canvas / PropertiesPanel / Toolbar).
- **Integration architecture** — the crux: how Python fabri drives a browser studio/lib
  (candidates: OrtNodeBackend for a Node path, an edit-plan JSON the studio executes, or headless
  browser). To be decided from the scoping results before building.

Deliverables targeted: `apps/studio` (React video editor on `@websam3/*`), `integrations/fabri`
(video_editing tool domain + text→prompt via a vision LLM), and the bridge between them. Weight
hosting (below) is a hard dependency for a runnable product.
