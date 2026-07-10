# Plan — websam Studio + fabri product

*Status: PLAN ONLY (build later). Written 2026-07-10. Supersedes the M4/M5 sketch in the
original plan for the studio+fabri direction. Two scoping deep-dives (fabri internals, React studio
stack) + the ludexel-gba integration pattern back every choice here.*

## Context & goal

Turn websam from "a published library" into a **product**: a human-facing in-browser **video Studio**
(view / edit / drag videos, drive segmentation, export) AND **fabri agent orchestration** that can
make and edit videos autonomously. Both are consumers of the SAME EdgeTAM segmentation core we already
shipped (`@websam3/core`, in-browser video tracking, gate-green at IoU 0.989–0.995).

Key architectural insight: **one core, two runtimes.**
- **Human Studio** → the browser core (`@websam3/core` WebGPU/WASM) in a React app.
- **fabri agent** → the SAME exported EdgeTAM ONNX graphs run in **Python via onnxruntime**, reusing
  our existing `tools/export/spikes/m2-edgetam/e2e_loop.py` (the pure-ORT tracking loop). No browser,
  no headless Playwright, no TS-in-Node backend required for the agent path.

## Decisions locked (from the user, 2026-07-10)

1. **Studio MVP first**, then layer fabri.
2. **Bundle EdgeTAM fp16 weights in the app for now** (defer a CDN). ~42 MB, all files <25 MB.
3. **fabri drives segmentation via sandbox tools** (the ludexel pattern), NOT headless browser and NOT
   edit-plan-only. Text→prompt grounding is a *tool* (extract frame → vision LLM → box JSON); fabri's
   LLM never needs image input. Confirmed viable against `fabri/src/fabri/sandbox/` (LocalSandbox →
   DockerSandbox ABC with sync_in/sync_out) and `ludexel-gba/.agent/fabri_agent.yaml` +
   `ludexel-gba/tools/agent_tools/*`.

## Architecture

```
                     ┌─────────────────────────── ONE CORE ───────────────────────────┐
                     │  Exported EdgeTAM graphs + manifest (tools/export, fp16+fp32)    │
                     │  memory-bank / video-engine semantics (spec.py + e2e_loop.py)   │
                     └───────────────┬─────────────────────────────────┬───────────────┘
        Human product ↓                                                 ↓ Agent product
  ┌──────────────────────────────┐                        ┌──────────────────────────────────┐
  │ apps/studio (React)          │                        │ integrations/fabri               │
  │  @websam3/core VideoSession  │                        │  .agent/fabri_agent.yaml         │
  │  dnd-kit + react-konva +     │                        │  .agent/prompts/domains/*.md     │
  │  zustand + shadcn + mediabunny│                       │  tools/video_editing/*.json+.py  │
  │  bundled fp16 weights        │                        │   → onnxruntime (Python) segment │
  └──────────────────────────────┘                        │   → vision-LLM text→box grounding│
                                                           │   → ffmpeg frame extract/compose │
                                                           └──────────────────────────────────┘
```

## Phase A — Studio MVP (build first)

`apps/studio` — a Vite + React + TS app; the human video editor.

### Stack (all MIT unless noted; from the studio-stack research)
- **Drag/reorder:** `@dnd-kit/core` + `@dnd-kit/sortable`; hand-built trim handles (pointer events).
- **Playback:** native `<video>` + `requestVideoFrameCallback` (playhead/preview sync); WebCodecs
  `VideoDecoder` for frame-accurate extraction into `VideoSession`.
- **Overlay:** `react-konva` + `konva` (prompt points/boxes + mask overlay image).
- **State:** `zustand` (`subscribeWithSelector` for the high-frequency playhead).
- **Layout/UI:** shadcn/ui + Radix + Tailwind + `react-resizable-panels`.
- **Export:** `mediabunny` (MPL-2.0 — permissive; the maintained successor to the deprecated
  mp4-muxer/webm-muxer) for MP4/WebM; PNG-zip matte via `@websam3/video-editing` as the cross-browser
  default. VP9-alpha WebM = Chrome/Firefox-only "bonus" (validate alpha survives the mux).
- **Consumes:** `@websam3/core` (createSegmenter → VideoSession → propagate) + `@websam3/video-editing`
  (MaskTimeline, AlphaMatteExporter). AVOID Remotion-based editors (company-size paid license).

### Component architecture (zustand store as the seam)
```
<VideoStudio>  (COOP/COEP headers via vite.config; worker.format 'es')
 ├─ <Toolbar>          tool toggle (select/point/box/pan), play/pause, Track, Export
 ├─ <PanelGroup> (react-resizable-panels)
 │   ├─ <MediaLibrary> imported clips; drag into timeline (dnd-kit)
 │   ├─ <PreviewCanvas> <video> + rVFC-synced konva <Stage>: frame + prompts + mask overlay
 │   ├─ <PropertiesPanel> selected clip/prompt props, mask opacity, export settings
 │   └─ <Timeline>     dnd-kit tracks; draggable/sortable clips + custom trim handles; playhead
 └─ export flow: VideoSession.propagate() → masks → mediabunny (MP4/WebM) | PNG-zip matte
```
Store shape: `{ clips, tracks, selection, playhead, tool, prompts: Record<clipId,Prompt[]>,
maskCache, exportSettings }`.

### Weights (bundled for now)
- `apps/studio/scripts/setup-weights.mjs`: copies the fp16 EdgeTAM graphs + manifest from
  `tools/goldens/models-cache/edgetam/` (already generated locally) into
  `apps/studio/public/models/edgetam/` (gitignored in-app). Studio sets `modelBaseUrl: '/models/'`.
  Documented fallback: run `make-video-golden.py` to (re)generate the cache. (Later: R2/HF host +
  default `modelBaseUrl`.)

### MVP scope (vertical slice, shippable)
Import an mp4 → scrub on a single-track timeline → click frame to add an object (point prompt) →
**Track** (propagate) → live mask overlay per frame → **Export** matte.zip (PNG-seq) and/or MP4
cutout via mediabunny. Multi-clip drag/trim/reorder + a second object are fast-follows.

### Milestones & gate
- A1 scaffold (Vite+React+TS+deps, tailwind/shadcn, zustand store, panel layout, weight script).
- A2 preview+prompts (video + rVFC + konva point/box; wire `@websam3/core` image/first-frame).
- A3 timeline+tracking (dnd-kit track, playhead, VideoSession.propagate → overlay, refine).
- A4 export (PNG-zip matte + mediabunny MP4 cutout) + polish pass (agentic-design).
- **Gate:** `pnpm -F websam-studio build` green + a browser smoke test that loads bundled weights,
  tracks a committed 10-frame clip, and asserts a non-empty mask (reuse the M2 golden clip). Manual
  Chrome run. CI builds `apps/studio` (already covers `./apps/**`).

## Phase B — fabri video_editing integration (build second)

Mirror the ludexel-gba pattern exactly (config + prompts + a tools dir; never touch fabri BUSL core).

### Layout (`integrations/fabri/`)
```
.agent/fabri_agent.yaml            orchestrator config (Gemini default; per-role LLM; budgets)
.agent/prompts/orchestrator.md     lists the video_editing domain + when to call each tool
.agent/prompts/domains/video_editing.md
tools/video_editing/               manifest(.json)+executable(.py) pairs, name-prefixed vid_*:
  vid_extract_frame.{json,py}      ffmpeg: video+time → PNG frame path (sandbox-jailed)
  vid_ground_text.{json,py}        frame + phrase ("the red car") → box/points JSON via a
                                     VISION LLM (Gemini/Claude vision) — the text→prompt step,
                                     done in-tool so fabri's text-only loop never needs image input
  vid_segment.{json,py}            frame + point/box → mask (onnxruntime Python, EdgeTAM decoder)
  vid_track.{json,py}              video + prompt on frame k → per-frame masks (reuse
                                     tools/export/spikes/m2-edgetam/e2e_loop.py; long job →
                                     submit→job_id→vid_poll_job to fit fabri's sync-per-call model)
  vid_export_matte.{json,py}       masks → matte.zip / MP4 cutout (ffmpeg / mediabunny-node)
  vid_composite.{json,py}          overlay/cutout/background-swap
  vid_trim.{json,py}, vid_concat.{json,py}  ffmpeg edit ops
```
- **Segmentation in Python:** reuse the exported EdgeTAM graphs + the pure-ORT loop — no browser,
  no OrtNodeBackend. Same graphs the studio uses; single source of truth.
- **Artifacts:** file paths jailed under `$FABRI_SANDBOX_ROOT` (`sandbox_root: project`), passed
  between tools by convention; final `agent.response_schema` returns `{clips[], masks[], outputPath}`
  so a host UI gets machine-checkable pointers, not prose.
- **Sandbox:** `LocalSandbox` for dev; `DockerSandbox` (fabri's ABC + `sync_in/sync_out`) when
  deployed/multi-tenant — frame extraction + vision + inference all run inside the sandbox, exactly
  the "sandbox runtimes + frame extraction for vision" the user described.
- **Live progress:** fabri's trace JSONL + `fabri serve` tailer → SSE into the Studio (ludexel routes
  `ask_user`/events to its React workspace the same way) so the Studio can show "agent is tracking…".
- **BUSL:** config + `tools/` + optional MCP server only. If native vision-input or streaming is ever
  wanted in fabri core, the user is the licensor (`pataderushikesh@gmail.com`) — but the plan avoids
  needing it by grounding in a tool.

### Fabri milestones & gate
- B1 tool contracts + manifests (schemas) + agent.yaml + domain prompt.
- B2 implement vid_extract_frame / vid_ground_text / vid_segment / vid_track (Python-ORT).
- B3 vid_export_matte / composite / trim / concat + response_schema + poll-job for long tracks.
- B4 end-to-end fabri run: "track the <described object> and export a matte" on a test clip →
  produces a matte.zip; structured output validated. Optional: wire SSE progress into the Studio.
- **Gate:** a fabri e2e test (LocalSandbox) completing the above on the committed golden clip, mask
  IoU vs the M2 goldens ≥ 0.90/frame (same bar), `uv` pytest green.

## Cross-cutting

- **Weight hosting** (bundle now → later): promote to R2 (zero egress, needs CF creds) or a public HF
  repo (Apache EdgeTAM, needs HF write token); set a default `modelBaseUrl`. Blocks "npm install →
  just works" but not the bundled Studio.
- **OrtNodeBackend (TS) — OPTIONAL, deferred:** only needed if we ever want the TS engine itself in
  Node (e.g. exact parity with the browser, or a JS-only server). The Backend interface is already
  defined/tested for it, but Python-ORT makes it unnecessary for fabri. Keep as a "nice to have."
- **React package gaps to close during this work:** `@websam3/react`'s `useImageSession`/
  `useVideoSession` are still stubs — implement them (the Studio will want real hooks).
  `AlphaMatteExporter` cutout + VP9-alpha still `NotImplementedError` (needed by the Studio export).

## Risks / sequencing notes
- Studio is independent of fabri and independent of weight *hosting* (bundled) — lowest-risk first
  slice, exactly the chosen order.
- Frame-accurate WebCodecs seeking + rVFC sync is the trickiest Studio piece; budget a spike.
- fabri long-running track jobs must use submit→poll (fabri is synchronous-per-tool-call).
- Model for subagents stays **Sonnet 5** until 2026-07-14 (global override); Node 22; per-wave gates.
- Execution: multi-agent delivery per phase (design contracts → parallel disjoint-file apply →
  consolidating gate → commit), same as M0–M2.
```
