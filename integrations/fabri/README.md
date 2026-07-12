# websam-video — fabri video-editing agent

A [fabri](https://github.com/) agent that turns plain-English video edit
requests ("cut out the red car and put it on a green background") into
calls against nine `vid_*` tools: extract a frame, ground a text phrase
to pixel coordinates, segment or track an object, and export/composite/
trim/concat the result.

**One core, two runtimes.** This is the *same* EdgeTAM ONNX graphs the
in-browser websam Studio runs (`tools/goldens/models-cache/edgetam/`,
`tools/goldens/fixtures/video/`) — here they run headless in Python via
`onnxruntime` (`tools/video_editing/_websam_ort.py`, reusing
`tools/export/spikes/m2-edgetam/e2e_loop.py`'s `OrtEngine`/`MemoryBank`),
instead of in a browser via WebGPU/ORT-web. One weights/fixtures source of
truth, two places it executes.

Design is documented in full in [`docs/fabri-contracts.md`](docs/fabri-contracts.md)
(layout, agent config schema, tool contracts, test plan).

## What's here

```
.agent/
├── fabri_agent.yaml              agent config: model, budgets, tool allow-list, response schema
└── prompts/
    ├── orchestrator.md           system prompt: tool table, submit->poll pattern, artifact passing
    └── domains/video_editing.md  single-domain specialization (near-identical; see file header)
tools/video_editing/
├── _websam_ort.py                 shared ORT core (session cache, encode/track/decode) — not a tool
├── vid_extract_frame.{json,py}    video -> still frame PNG
├── vid_ground_text.{json,py}      text phrase -> pixel point/box (Gemini vision, or a deterministic stub)
├── vid_segment.{json,py}          one frame + prompt -> one mask
├── vid_track.{json,py}            submit: track an object across a whole video (async)
├── vid_poll_job.{json,py}         poll: check a vid_track job's status/result
├── vid_export_matte.{json,py}     package masks -> matte.zip or an alpha-cutout mp4
├── vid_composite.{json,py}        apply a mask -> cutout / highlight / background-replace
├── vid_trim.{json,py}             cut a video to [startSec, endSec]
├── vid_concat.{json,py}           concatenate clips
└── _track_worker.py               detached worker vid_track spawns for the async tracking loop
tests/test_websam_ort.py           IoU-gated correctness tests against golden fixtures
```

## Install

fabri itself is BUSL-licensed and lives in a separate checkout — this
directory is **config + tools only**, exactly the `ludexel-gba` pattern;
no fabri core code is touched or vendored here.

```bash
pip install -e /Users/rushour0/gba/fabri     # or: pip install fabri, once published

cd integrations/fabri
uv sync
```

## Environment

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required for the real `vid_ground_text` path (Gemini vision, via `google-genai`) **and** the agent's own orchestrator LLM calls. |
| `WEBSAM_GROUND_TEXT_STUB` | Set (to any truthy value) to force `vid_ground_text` onto a deterministic offline stub instead of calling Gemini — no network, no key needed. Used by the test suite and any offline dev loop. |
| `FABRI_SANDBOX_ROOT` | Set by the fabri runner/service at launch; every `vid_*` script independently re-checks it and refuses paths that escape it. Not something you normally set by hand. |

## Run it

```bash
cd integrations/fabri
fabri run --config .agent/fabri_agent.yaml \
           --system-prompt-file .agent/prompts/orchestrator.md \
           "cut out the red car from clip.mp4 and export it as a transparent mp4"
```

You can also invoke any `vid_*` tool directly, exactly as the fabri
runner would (stdin JSON in, stdout JSON out) — useful for debugging a
single step without spinning up the full agent loop:

```bash
echo '{"video": "clip.mp4", "timeSec": 0.3}' | uv run python tools/video_editing/vid_extract_frame.py
```

## Tools

`vid_extract_frame`, `vid_ground_text`, `vid_segment`, `vid_track`,
`vid_poll_job`, `vid_export_matte`, `vid_composite`, `vid_trim`,
`vid_concat`. See each `.json` manifest for the exact input/output
schema, or `.agent/prompts/orchestrator.md` for the "when to use which"
guide and the mandatory `vid_track` → `vid_poll_job` submit/poll loop
(tracking is async — poll patiently, don't busy-spin, never double-submit).

## Tests

```bash
cd integrations/fabri
uv run --group dev pytest
```

Runs with **zero network access and zero fabri package installed** —
tests invoke each tool script the same way the fabri runner does
(`subprocess.run` with the manifest's `command`, stdin/stdout JSON) and
assert IoU against golden mask fixtures in `tools/goldens/fixtures/video/`.
`vid_ground_text`'s stub path always runs; its real-Gemini path is
`skipif`'d without `GEMINI_API_KEY` so CI stays offline by default.

## License note

fabri (the orchestration framework, `~/gba/fabri`) is BUSL-licensed and
is **not** part of this repository — nothing under `integrations/fabri/`
imports or vendors fabri source. This directory is exactly the surface
fabri expects a project to provide for itself: an `agent.yaml`, prompt
files, and a `tools/` directory of manifest+script pairs, all of which
are this project's own MIT-licensed code.
