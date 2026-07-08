# websam — progress log & resume handoff

**Status as of 2026-07-09: PAUSED mid-M2.** M0 and M1 are complete, gated green, committed and pushed. M2 (EdgeTAM video) prep is done and the export spike passed; M2 implementation wave 1 was started but **stopped mid-flight** and its output is uncommitted WIP (see below).

Repo: https://github.com/Rushour0/websam (public, MIT).
Plan of record: `~/.claude/plans/lets-build-something-from-goofy-dragon.md` (plan v2, post 4-angle adversarial review).

## Operating rules for resuming
- **All subagents/workflow agents run on Opus 4.8** (`model: 'opus'`), per user directive 2026-07-09. Not Fable.
- Node **22** required (nvm: `. ~/.nvm/nvm.sh && nvm use 22`). Node 23 breaks jsdom/tsdown engine ranges. `.nvmrc` pins 22.
- Delivery pattern: spikes/design first → parallel disjoint-file apply agents → one consolidating gate (build + typecheck + unit + browser + Python) → commit. Never one agent editing shared files concurrently.
- Gate command set: `pnpm build`, `pnpm typecheck`, `pnpm test`, `cd packages/core && pnpm test:browser`, `cd tools/export && uv run --group dev pytest -q`, `pnpm -r --filter './packages/*' exec publint`, `pnpm -r --filter './packages/*' exec attw --pack . --profile esm-only`.

## Milestone status

### M0 — scaffold + contracts ✅ (commit `fa8a296`)
pnpm monorepo; `@websam/core` (Backend abstraction, error taxonomy, coords, RLE+COCO, registry, Segmenter/session contracts), `@websam/video-editing` (MaskTimeline), `@websam/react` (StrictMode-safe useSegmenter), demo shell w/ capability probes, bundler-matrix (Vite+webpack), `tools/export` (Python/uv). Gate green: build+typecheck+64 unit+browser ONNX smoke+42 py+publint+attw. CI uses xvfb+SwiftShader for the WebGPU browser lane.

### M1 — interactive image path ✅ (commits `68bb431` prep, `ff30dde` impl)
- **S0 spike verdict REUSE-OK**: community `onnx-community/sam3-tracker-ONNX` vision encoder emits all 3 FPN outputs the decoder needs. Coordinate mode **pinned = square-stretch** (anisotropic 1008×1008, mean=std=0.5, per-axis prompt scaling); decoder logits 288×288. See `tools/export/spikes/s0/FINDINGS.md`.
- **Golden fixtures**: deterministic transformers.js (q4f16) reference masks on a 640×427 non-square scene → `tools/goldens/fixtures/`.
- Full pipeline real: manifest-driven weight loading (streaming SHA-256, OPFS/Cache-API content-addressed stores, Range-resume), ORT runtime + device/quant resolution + real WebGPU/WASM Backend bodies, Comlink module worker (preprocess → encoder → per-click decoder → 288→source mask), `createSegmenter`/`ImageSession`, `MaskResultImpl`, demo image tab.
- **Gate green**: 258 unit + 20 browser tests. **M1 golden gate PASSED**: our pipeline reproduces transformers.js masks at **IoU 0.9957 / 0.9962** (bar 0.9) on the wasm EP. `dist/worker.js` ships as `@websam/core/worker`.

### M2 — EdgeTAM video + rotobrush demo 🟡 IN PROGRESS (paused)
**Done & solid (uncommitted WIP — see "Working tree" below):**
- **EdgeTAM export spike: GO** (`tools/export/spikes/m2-edgetam/`). All 5 graphs exported (dynamo, first try, no TorchScript fallback), per-graph fp32 parity passed, and the pure-ORT Python loop reproduces HF PyTorch at **IoU 1.0000 on all 8 frames** (bar 0.95). Canonical checkpoint = `yonigozlan/EdgeTAM-hf`. Executable spec for the JS engine = `spikes/m2-edgetam/e2e_loop.py`. Reusable wrappers in `tools/export/src/websam_export/wrappers/edgetam.py` (+ 8 tests, suite 50 passed).
- **Spike corrected spec.py constants** (already fixed + tests green): EdgeTAM = **512 tokens/memory map** (256 1D + 256 2D), so **KV_LEN = 7×512+64 = 3648** (was wrongly 256/1856). Unlimited conditioning frames in this checkpoint (JS caps via closest-cond selection). No `pointer_time_deltas` input (pointer temporal PE disabled). No occlusion spatial embedding. Preprocessing = 1024 square-stretch + **ImageNet mean/std** (NOT 0.5). `num_points` exported **dynamic** (HF −10 pad tokens attend, so P=8 padding would be wrong). All graph IO is BCHW/batch-first.
- **M2 internal contracts doc**: `packages/core/docs/m2-internal-contracts.md` (690 lines, normative). Memory-bank strategy pattern, pull-credit MessagePort propagate protocol, epoch state machine, WebCodecs-in-worker frame source, PNG-zip matte export, per-file ownership + wave plan + PIN registry (PIN-1..10 now largely resolved by the spike).

**Started then STOPPED (partial/unreliable WIP):**
- M2 apply **wave 1** (backend copyRegion/debugStats, memory-bank + strategy, WebCodecs frame source, PNG-zip exporter) was launched on Fable and killed mid-run. 2 of 4 agents completed; 2 were interrupted. **Do not trust these files — discard and re-run wave 1 on Opus.** Affected uncommitted paths: `packages/core/src/backend/*.ts` + `memory-primitives*.test.ts`, `packages/core/src/weights/manifest.ts` + `.test.ts`, `packages/core/src/worker/video/`, `packages/core/src/video/`, `packages/core/package.json` (mp4box dep), `packages/video-editing/*` (exporter + fflate dep), `pnpm-lock.yaml`.

## TODO — resume order

1. **Reset M2 wave-1 impl surface** (keep the committed prep): discard the uncommitted wave-1 implementation files listed above (they were Fable, partial). KEEP: `tools/export/spikes/m2-edgetam/`, `tools/export/src/websam_export/wrappers/`, `tools/export/tests/test_wrappers_edgetam.py`, `tools/export/pyproject.toml`, `spec.py`+`test_spec.py` corrections, `packages/core/docs/m2-internal-contracts.md`. (These are committed by the WIP commit that accompanies this log — so "discard" = revert just those impl paths, or re-run wave 1 and let it overwrite.)
2. **M2 wave 1 (Opus, 4 parallel disjoint agents)**: backend-video, memory-bank, frame-source, editing — per `m2-internal-contracts.md` ownership table.
3. **M2 wave 2 (Opus, 2 agents)**: worker-video (video-engine + propagation-port + protocol/preprocess extensions), video-session (VideoSessionImpl + epoch iterator + `createVideoSession` body + flip default model to `edgetam`).
4. **M2 wave 3 (Opus)**: demo video tab (scrub-less rotobrush: file → click frame 0 → Track live overlay → refine → export matte.zip), video golden gate (`tools/export` `make-video-golden.py` → committed per-frame golden RLEs from `e2e_loop.py`; browser e2e IoU ≥0.90/frame + debugStats tensor-census flatness).
5. **Real EdgeTAM weights hosting**: convert/merge external-data graphs (spike script does this), publish to our HF repo (EdgeTAM is Apache-2.0 — clean), wire manifest.
6. **Consolidating gate + commit M2**, then **launch rotobrush demo**.

Then later: **M3** (SAM3-tracker video tier — the research-grade spike ladder S0–S6, off launch critical path), **M4** (full editing layer: WebGPU compositor, VP9-alpha export, docs, 0.1.0 publish — needs `@websam` npm scope claimed + SAM License legal check), **M5** (fabri integration via OrtNodeBackend + VLM text→prompt).

## Known open items / risks
- `@websam` npm scope not yet claimed (do before M4 publish).
- SAM3 weight re-hosting legally unsettled — EdgeTAM-first neutralizes; SAM3 default = user runs export pipeline on their own gated download.
- 51.9k-token SAM3 memory attention may be bandwidth-bound on WebGPU (M3 S5 measures; 560 is multi-object default).
- iOS Safari = EdgeTAM-only experimental; WASM video = EdgeTAM-int8 only; PNG-zip is the portable matte export (VP9-alpha Chrome-only).
