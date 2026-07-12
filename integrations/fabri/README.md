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
| `OPENAI_API_KEY` | EITHER this OR `GEMINI_API_KEY`. When set (and `FABRI_LLM_PROVIDER` unset) the agent runs on provider `openai`, model `gpt-4o`, narrator `gpt-4o-mini`. |
| `GEMINI_API_KEY` | Required for the real `vid_ground_text` path; also the orchestrator key when `gemini` is the chosen provider. |
| `FABRI_LLM_PROVIDER` | Optional: force the orchestrator provider instead of auto-pick. One of `openai \| gemini \| anthropic \| openrouter \| bedrock`. |
| `FABRI_LLM_MODEL` | Optional for openai/gemini (defaults `gpt-4o` / `gemini-2.5-pro`); required for any other provider. |
| `FABRI_LLM_NARRATOR_MODEL` | Optional: cheap model for the live narration chat lines. Defaults `gpt-4o-mini` (openai), `gemini-2.5-flash-lite` (gemini), else the main model. |
| `FABRI_GATEWAY_CORS_ORIGIN` | Comma-separated exact browser origins allowed to call the gateway (also enforced as an Origin-header check on `POST /runs`). Default covers Vite dev on `localhost` and `127.0.0.1`. |
| `FABRI_GATEWAY_PORT` | Gateway listen port. Keep in sync with `apps/studio/.env`'s `VITE_FABRI_GATEWAY_URL`. Default `8787`. |
| `FABRI_RUNS_DIR` | Per-run sandboxes (uploaded videos + artifacts), fabri session homes, and sqlite memory. Relative paths resolve against `integrations/fabri/`. Default `./runs`, git-ignored. |
| `WEBSAM_GROUND_TEXT_STUB` | Forces `vid_ground_text` onto a deterministic offline stub instead of calling Gemini — any non-empty value enables the stub; the value must be inline JSON (box/point payload) or a fixture path — never a bare `1`/`0`. |
| `FABRI_SANDBOX_ROOT` | Set by the fabri runner/service at launch; every `vid_*` script independently re-checks it and refuses paths that escape it. Not something you normally set by hand. |

## Run it

```bash
cd integrations/fabri
fabri run --config .agent/fabri_agent.yaml \
           "cut out the red car from clip.mp4 and export it as a transparent mp4"
```

Current fabri has **no** `--system-prompt-file` flag. The gateway
(below) injects `.agent/prompts/orchestrator.md` as a literal string via
the `agent.system_prompt` override on each run. A direct-CLI user who
wants that same prompt should temporarily set `agent.system_prompt` in a
copy of the yaml.

You can also invoke any `vid_*` tool directly, exactly as the fabri
runner would (stdin JSON in, stdout JSON out) — useful for debugging a
single step without spinning up the full agent loop:

```bash
echo '{"video": "clip.mp4", "timeSec": 0.3}' | uv run python tools/video_editing/vid_extract_frame.py
```

## Gateway service (conversational API for Studio)

`service/app.py` is a small FastAPI app that wraps fabri's Python
`FabriService` (submit / stream / result), turning the one-shot CLI
agent into a conversational, browser-reachable video-editing feature for
`apps/studio`.

| Endpoint | What it does |
|---|---|
| `POST /runs` | Multipart fields `video` + `task` → `{"sessionId"}`. The request `Origin` is checked against `FABRI_GATEWAY_CORS_ORIGIN`. |
| `GET /runs/{sessionId}/events` | SSE stream of live progress: unnamed `data:` events for each narrator/tool/step line, then a final named `event: end` sentinel. |
| `GET /runs/{sessionId}/result` | fabri's result envelope verbatim: `session_id`, `success`, `outcome`, `final_text`, `structured_output`, `usage`, `cost`, `error`. |
| `GET /runs/{sessionId}/artifact?path=…` | Streams a file back out of the run's sandbox dir; the path is jailed to that directory (any path resolving outside it is rejected). |
| `GET /health` | Liveness probe. |

### Run locally

```bash
cd integrations/fabri && uv sync && uv pip install -e /path/to/your/fabri  # BYO BUSL checkout — re-run after every uv sync (sync prunes undeclared packages)
cp .env.example .env  # set OPENAI_API_KEY or GEMINI_API_KEY
uv run uvicorn --factory service.app:create_app --host 127.0.0.1 --port 8787
```

`uv run python -m service.app` is equivalent and honors
`FABRI_GATEWAY_PORT`; both bind `127.0.0.1` — the gateway has no auth, so
exposing it beyond localhost requires a reverse proxy that adds auth.

### Provider selection

`FABRI_LLM_PROVIDER` wins if set; else `openai` when `OPENAI_API_KEY` is
set, else `gemini` when `GEMINI_API_KEY` is set, else the gateway refuses
to start, naming the missing vars. The per-run overrides always send an
explicit `provider`, `model`, `api_key_env` **and** `llm.narrator.model`
(the template's narrator pins a Gemini model that would otherwise
silently kill all narration on non-Gemini providers).
`anthropic`/`openrouter`/`bedrock` additionally require `FABRI_LLM_MODEL`
and their own key env. Without `GEMINI_API_KEY`, `vid_ground_text`
auto-falls back to its offline stub (the gateway sets
`WEBSAM_GROUND_TEXT_STUB` to an inline-JSON point payload; reduced
grounding accuracy, warned once at startup).

### Docker (self-host, bring your own fabri checkout)

From the **repo root**:

```bash
docker build -f integrations/fabri/service/Dockerfile --build-context fabri-src=/path/to/your/fabri -t websam-fabri-gateway .   # requires BuildKit — Docker >= 23 or DOCKER_BUILDKIT=1
docker run --env-file integrations/fabri/.env -e FABRI_GATEWAY_PORT=8787 -p 8787:8787 -v fabri-runs:/data/runs websam-fabri-gateway
```

The explicit `-e FABRI_GATEWAY_PORT=8787` pins the container port so an
`.env` override can't desync the `-p` mapping. Or, from `service/`:

```bash
FABRI_CHECKOUT=/path/to/fabri docker compose up --build
```

fabri is BUSL-licensed and is **never vendored** in this repo or its
image — the deployer supplies their own checkout at build time via the
named build context. `service/Dockerfile.dockerignore` guarantees
`.env` / `runs/` / `.venv` are never baked into the image. No cloud
provider or hosting account is assumed — the image runs on any container
host.

### Bring your own everything

All secrets and provider choice live only in the git-ignored `.env`.
EITHER an OpenAI key OR a Gemini key works out of the box: OpenAI-only
runs the orchestrator on `gpt-4o` + narrator `gpt-4o-mini` and
auto-stubs text-grounding with a logged reduced-accuracy warning; a
Gemini key also unlocks real vision grounding. Point Studio users at
`apps/studio/.env.example` (`VITE_FABRI_GATEWAY_URL`).

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
