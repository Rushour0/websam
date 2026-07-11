# fabri × websam — Phase B contracts (video_editing agent)

*Status: DESIGN ONLY. No fabri core code is touched — fabri is BUSL; this integration is
config + a tools dir, exactly the `ludexel-gba` pattern. Written against
`docs/plans/studio-and-fabri-product.md` (authoritative scope) and verified against real fabri
source (`~/gba/fabri/src/fabri/tools/{manifest_schema,runner,result}.py`,
`~/gba/fabri/src/fabri/sandbox/__init__.py`, `~/gba/fabri/docs/creating-an-agent.md`) and the
working `~/gba/ludexel-gba/.agent/` + `tools/agent_tools/*` integration.*

## 0. Non-negotiables carried from the plan

- Segmentation tools run the EdgeTAM ONNX graphs **directly in Python via onnxruntime**, reusing
  `tools/export/spikes/m2-edgetam/e2e_loop.py` (`OrtEngine` + `MemoryBank`) — no browser, no Node,
  no `OrtNodeBackend`. Same exported graphs the Studio uses (`tools/goldens/models-cache/edgetam/`);
  one source of truth.
- fabri's LLM never receives image bytes. Text→prompt grounding is a **tool**
  (`vid_ground_text`): extract a frame → call a vision LLM → return box/point JSON. The
  orchestrator's tool-calling loop stays text-only end to end.
- All artifacts are file paths jailed under `$FABRI_SANDBOX_ROOT`, passed between tool calls by
  convention (tool N's output path is tool N+1's input path) — the same discipline as
  `read_file.py`/`write_file.py`'s `target.is_relative_to(root)` check.
- Every tool name is prefixed `vid_`. Manifests live beside their executables under
  `integrations/fabri/tools/video_editing/`.
- fabri is synchronous per tool call (the runner blocks on `proc.communicate(timeout=manifest.timeout_s)`
  and SIGKILLs the process group on timeout — see `runner.py`). `vid_track` on more than a couple of
  seconds of video will not fit inside one call; it uses submit→job_id→poll (§3.4).

## 1. Layout (mirrors ludexel-gba exactly)

```
integrations/fabri/
├── pyproject.toml                      uv project; onnxruntime, numpy, pillow, imageio[ffmpeg],
│                                        google-genai (vision LLM client), pytest
├── README.md                           how to install fabri, set env vars, run the tools directly
│                                        and (optionally) via the full agent
├── .agent/
│   ├── fabri_agent.yaml                orchestrator config (§2)
│   └── prompts/
│       ├── orchestrator.md             lists the video_editing domain + tool-call order
│       └── domains/
│           └── video_editing.md        the ONLY domain for Phase B (single-domain agent —
│                                        no spawn_subagent fan-out needed; see §2 note)
├── tools/
│   └── video_editing/
│       ├── _websam_ort.py              shared lib: ORT session cache + thin wrapper around
│       │                               e2e_loop.OrtEngine/MemoryBank (§4) — NOT a manifest'd tool
│       ├── vid_extract_frame.json / .py
│       ├── vid_ground_text.json / .py
│       ├── vid_segment.json / .py
│       ├── vid_track.json / .py
│       ├── vid_poll_job.json / .py
│       ├── vid_export_matte.json / .py
│       ├── vid_composite.json / .py
│       ├── vid_trim.json / .py
│       └── vid_concat.json / .py
├── tests/
│   ├── conftest.py                     fixtures: golden clip path, sandbox tmp dir, IoU helper
│   ├── test_vid_track.py               core ORT correctness gate (§5.1)
│   ├── test_vid_segment.py
│   ├── test_vid_export_matte.py
│   ├── test_vid_ground_text.py         deterministic-stub path; real-LLM path skipped w/o key
│   ├── test_vid_extract_frame.py
│   ├── test_vid_composite_trim_concat.py
│   └── test_e2e_agent.py               OPTIONAL full-agent run, skippable (§5.2)
└── .gitignore                          .fabri/, *.onnx (symlinked, not vendored), __pycache__
```

`tools/goldens/models-cache/edgetam/*.onnx` and `tools/goldens/fixtures/video/{clip-256.mp4,
golden-*.json}` are **not copied** into `integrations/fabri/` — `_websam_ort.py` and the test
fixtures resolve them via a relative path up to the repo's `tools/goldens/` (see `_repo_root()` in
§4). This keeps one weights/fixtures source of truth for Studio, core, and fabri.

## 2. `.agent/fabri_agent.yaml`

Single-domain agent — Phase B's tool surface (9 tools) is small enough that the orchestrator calls
`vid_*` tools directly; there is no `spawn_subagent` fan-out like ludexel's per-content-domain
children. `orchestrator.md` *is* the (only) system prompt; `domains/video_editing.md` is included
into it by reference (kept as a separate file so a future multi-domain fabri video agent — e.g. an
`audio_editing` domain — can reuse the split, matching ludexel's convention).

```yaml
agent:
  name: websam-video-editing
  max_steps: 20                 # extract → ground → segment/track → export is a 4-6 step happy
                                 # path; 20 gives headroom for a retry/refine loop without inviting
                                 # runaway iteration on a synchronous-per-call, poll-heavy tool set.
  max_cost_usd: 3.0
  response_schema:
    type: object
    required: [outputPath]
    properties:
      outputPath: {type: string}            # sandbox-relative path to the final artifact
      masks:      {type: array, items: {type: string}}   # sandbox-relative mask/RLE paths, if any
      objects:    {type: array, items: {type: object,
                    properties: {label: {type: string}, maskPath: {type: string}}}}
  response_retries: 1
  error_strategy: strict
  system_prompt: ""              # service passes --system-prompt-file .agent/prompts/orchestrator.md

llm:
  provider: gemini                # matches ludexel-gba's default-provider decision; free-tier-friendly
  model: gemini-2.5-flash
  max_tokens: 8192
  api_key_env: GEMINI_API_KEY
  cache_messages: false

tools:
  manifest_dir:
    - builtin                     # read_file/write_file/list_dir for path bookkeeping + ask_user
    - tools/video_editing
  enabled:
    - read_file
    - write_file
    - list_dir
    - vid_extract_frame
    - vid_ground_text
    - vid_segment
    - vid_track
    - vid_poll_job
    - vid_export_matte
    - vid_composite
    - vid_trim
    - vid_concat
  sandbox_root: project           # service cd's into the per-run materialized work dir first;
                                   # every vid_* script independently re-checks FABRI_SANDBOX_ROOT
  result_format: toon
  decompose:
    enabled: false                # the domain prompt already encodes the tool order; a research-
                                   # style decompose pass would just add a step for no benefit

memory:
  collection: fabri_websam_video
  qdrant_url: http://localhost:6333
  top_k: 8
  similarity_threshold: 0.8
  promotion_threshold_sessions: 3
  guideline_max_tokens: 60
```

Deltas from ludexel-gba worth flagging: no `subagent:` budget block (no fan-out), no
`retrieval_strategy`/`rrf_k` hybrid-memory tuning (Phase B ships without it — revisit once there's
a real guideline corpus), Gemini `flash` not `pro` (same free-tier-headroom reasoning as ludexel).

## 3. Tools

Every script follows the exact fabri contract verified in `runner.py`/`result.py`: read one JSON
object from stdin, print one JSON object to stdout, exit 0/nonzero. The runner wraps that in
`{ok, result, error?, stderr?}` — **scripts must NOT self-wrap** in `{ok: ...}`; they print their
raw result payload (mirrors `fetch_url.py`/`write_file.py`, which print `{"path": ..., ...}` and
`{"error": ...}` directly). All scripts import a shared `_sandbox.py`-style helper (inlined per
script, matching `write_file.py`'s ~8-line pattern — small enough not to warrant a shared module
per fabri's own convention) that resolves `$FABRI_SANDBOX_ROOT`, rejects paths that escape it, and
exits 1 with `{"error": "..."}` on violation.

### 3.1 `vid_extract_frame`

```json
{
  "name": "vid_extract_frame",
  "description": "Extract one video frame as a PNG. args: {videoPath, timeSec} (both sandbox-relative). Returns {framePath, width, height, timeSec}. Use this before vid_ground_text or vid_segment — they need a frame image, not a video.",
  "command": ["python3", "vid_extract_frame.py"],
  "input_schema": {"type": "object", "properties": {
    "videoPath": {"type": "string"}, "timeSec": {"type": "number"}},
    "required": ["videoPath", "timeSec"]},
  "output_schema": {"type": "object"},
  "timeout_s": 20
}
```
Implementation: `imageio.v3` (already a project-friendly ffmpeg wrapper, avoids shelling raw
`ffmpeg` args out to a string) seeks the nearest frame ≤ `timeSec` and writes
`<sandbox>/artifacts/frames/<stem>-<timeSec>.png`. Sandbox-jails both `videoPath` (read) and the
output dir (write).

### 3.2 `vid_ground_text`

```json
{
  "name": "vid_ground_text",
  "description": "Locate an object described in plain English within a video frame. args: {framePath, phrase} (e.g. phrase='the red ball'). Returns {points: [{x,y,label}], boxes: [{x0,y0,x1,y1}]} in the frame's ORIGINAL pixel coordinates. Call this to turn a text description into a prompt for vid_segment/vid_track — never guess coordinates yourself.",
  "command": ["python3", "vid_ground_text.py"],
  "input_schema": {"type": "object", "properties": {
    "framePath": {"type": "string"}, "phrase": {"type": "string"}},
    "required": ["framePath", "phrase"]},
  "output_schema": {"type": "object"},
  "timeout_s": 30
}
```
**Vision-LLM provider choice: Gemini (`gemini-2.5-flash`), via `google-genai`**, same provider as
the orchestrator's own `llm.api_key_env` (`GEMINI_API_KEY`) — one API key covers both roles, and
Gemini's structured-output mode (`response_mime_type: application/json` + a schema) gives box
coordinates without a fragile free-text-parse step. Falls back to a **deterministic stub** path
that MUST exist independent of any live key, gated by an env var read inside the script (not the
manifest):

```python
if os.environ.get("WEBSAM_GROUND_TEXT_STUB"):
    # loads tests/fixtures/ground_text_stub.json keyed by phrase substring;
    # lets vid_track/e2e tests run with zero network + zero API key.
    ...
else:
    if not os.environ.get("GEMINI_API_KEY"):
        print(json.dumps({"error": "GEMINI_API_KEY not set and WEBSAM_GROUND_TEXT_STUB not set"}))
        return 1
    # real google-genai vision call, image bytes read from framePath, JSON-schema response
```
Output box/point coords are in the frame's native pixel space (NOT the 1024×1024 model space) —
`_websam_ort.py` does the rescale (mirrors `e2e_loop.py`'s `PROMPT_POINT_XY * 1024 / WIDTH`
convention), so `vid_ground_text` and a human clicking in the Studio produce interchangeable
prompts.

### 3.3 `vid_segment`

```json
{
  "name": "vid_segment",
  "description": "Segment a single video frame given a point or box prompt. args: {framePath, point?: {x,y}, box?: {x0,y0,x1,y1}} (original pixel coords; exactly one of point/box). Returns {maskPath} (a PNG, white=object). Use for a one-off still-frame cutout; use vid_track to follow an object across the whole video.",
  "command": ["python3", "vid_segment.py"],
  "input_schema": {"type": "object", "properties": {
    "framePath": {"type": "string"},
    "point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
    "box": {"type": "object", "properties": {"x0": {"type": "number"}, "y0": {"type": "number"},
             "x1": {"type": "number"}, "y1": {"type": "number"}}}},
    "required": ["framePath"]},
  "output_schema": {"type": "object"},
  "timeout_s": 30
}
```
Single-frame path through `_websam_ort.py`: `encode_frame` → `condition_no_memory` (no memory bank
needed for a single still) → `decode` with the given prompt → threshold `low_res_masks` (>0,
mirrors `Sam2VideoProcessor.post_process_masks(binarize=True)`) → upsample to original resolution
→ write PNG.

### 3.4 `vid_track` + `vid_poll_job`

`vid_track` is the multi-frame case and does NOT fit fabri's synchronous per-call model for
anything beyond a couple of frames on CPU (each frame does one `vision_encoder` +
`memory_attention` + `mask_decoder_video` + `memory_encoder` ORT run — the 10-frame golden clip
takes low-single-digit seconds on CPU, but a real Studio-length clip won't). Submit→poll:

```json
{
  "name": "vid_track",
  "description": "Track an object across a whole video, starting from a prompt on one frame. args: {videoPath, promptFrameSec, point?: {x,y}, box?: {x0,y0,x1,y1}} (exactly one of point/box, original pixel coords at the prompt frame). Returns immediately with {jobId, status:'running'} — the job runs in the background; call vid_poll_job with the jobId to check progress and get the final result. Do not call vid_track again for the same object; poll instead.",
  "command": ["python3", "vid_track.py"],
  "input_schema": {"type": "object", "properties": {
    "videoPath": {"type": "string"}, "promptFrameSec": {"type": "number"},
    "point": {"type": "object"}, "box": {"type": "object"}},
    "required": ["videoPath", "promptFrameSec"]},
  "output_schema": {"type": "object"},
  "timeout_s": 15
}
```
```json
{
  "name": "vid_poll_job",
  "description": "Check the status of a background vid_track job. args: {jobId}. Returns {status: 'running'|'done'|'error', progress?: number, maskDir?: string, maskRlePath?: string, error?: string} once done. Poll every few seconds until status is 'done' or 'error'.",
  "command": ["python3", "vid_poll_job.py"],
  "input_schema": {"type": "object", "properties": {"jobId": {"type": "string"}}, "required": ["jobId"]},
  "output_schema": {"type": "object"},
  "timeout_s": 10
}
```
Implementation: `vid_track.py` writes a job record (`<sandbox>/artifacts/jobs/<jobId>/status.json`)
and `subprocess.Popen`'s a **detached** worker (`python3 -m video_editing._track_worker <jobId>
<args-json-path>`, `start_new_session=True` so it survives the parent tool's own process exit —
`vid_track.py` itself must return within its 15s `timeout_s`, well before tracking finishes) that
runs the real per-frame loop from `_websam_ort.py` and updates `status.json` after every frame
(`{status:"running", progress: t/numFrames}`) so `vid_poll_job` never blocks — it just reads the
current JSON off disk. On completion the worker writes per-frame mask PNGs to
`artifacts/jobs/<jobId>/masks/frame-%04d.png` plus one `masks.rle.json` (list of per-frame COCO-RLE
dicts, same `{width,height,counts}` shape as `tools/goldens/fixtures/video/golden-mask-f*.rle.json`)
and flips `status.json` to `{status:"done", maskDir, maskRlePath}`. This is a **file-based job
queue**, deliberately not a real task queue (Celery/RQ) — Phase B is single-tenant/dev-scoped
(`LocalSandbox`); revisit only if/when `DockerSandbox` multi-tenancy is wired.

### 3.5 `vid_export_matte`

```json
{
  "name": "vid_export_matte",
  "description": "Package a tracked object's per-frame masks into a downloadable artifact. args: {maskDir or maskRlePath, videoPath, format: 'matte_zip'|'mp4_cutout'}. matte_zip: a zip of PNG masks (cross-platform default). mp4_cutout: the source video masked to an alpha-cutout MP4 (may fall back to matte_zip if alpha-MP4 isn't available in this environment). Returns {outputPath}.",
  "command": ["python3", "vid_export_matte.py"],
  "input_schema": {"type": "object", "properties": {
    "maskDir": {"type": "string"}, "maskRlePath": {"type": "string"},
    "videoPath": {"type": "string"}, "format": {"type": "string", "enum": ["matte_zip", "mp4_cutout"]}},
    "required": ["videoPath", "format"]},
  "output_schema": {"type": "object"},
  "timeout_s": 60
}
```
`matte_zip`: `zipfile.ZipFile` over the mask PNGs (or, if only `maskRlePath` is given, decode RLE
→ PNG first — same COCO-RLE decode `tests/` needs for the IoU check, factored into
`_websam_ort.py::rle_decode`). `mp4_cutout`: shells `ffmpeg` (via `subprocess`, args list not a
shell string) to alpha-composite masks onto the source frames and mux a VP9/WebM-alpha or
ProRes4444 output — mirrors the Studio's documented VP9-alpha caveat (`AlphaMatteExporter` is
still `NotImplementedError` per the plan); Phase B's `mp4_cutout` is allowed to be best-effort and
fall back to `matte_zip` with a `{"warning": "..."}` field rather than block the whole tool on
codec availability.

### 3.6 `vid_composite` / `vid_trim` / `vid_concat`

Straightforward ffmpeg wrappers (`subprocess.run(["ffmpeg", ...], capture_output=True)`, argv list
form — never a shell string, matching `fetch_url.py`'s SSRF-hardening spirit of not trusting
model-controlled input into a shell). Manifests:

```json
{"name": "vid_composite", "description": "Overlay a cutout/matte onto a background video or image. args: {foregroundPath, backgroundPath, outPath, position?: {x,y}}. Returns {outputPath}.", "command": ["python3", "vid_composite.py"], "timeout_s": 60, ...}
{"name": "vid_trim", "description": "Cut a video to [startSec, endSec]. args: {videoPath, startSec, endSec, outPath}. Returns {outputPath}.", "command": ["python3", "vid_trim.py"], "timeout_s": 30, ...}
{"name": "vid_concat", "description": "Concatenate multiple video clips in order. args: {videoPaths: [...], outPath}. Returns {outputPath}.", "command": ["python3", "vid_concat.py"], "timeout_s": 60, ...}
```

## 4. `_websam_ort.py` — shared ORT core (not a manifest'd tool)

This is the single file every ORT-touching tool imports; it is the thin wrapper the task calls
for, over `e2e_loop.py` — **not a reimplementation**.

```python
# integrations/fabri/tools/video_editing/_websam_ort.py
import sys, pathlib

def _repo_root() -> pathlib.Path:
    # integrations/fabri/tools/video_editing/_websam_ort.py -> repo root is 3 parents up
    return pathlib.Path(__file__).resolve().parents[3]

_M2_DIR = _repo_root() / "tools" / "export" / "spikes" / "m2-edgetam"
sys.path.insert(0, str(_M2_DIR))
from e2e_loop import OrtEngine, MemoryBank, iou            # noqa: E402  (reused, not duplicated)

MODELS_DIR = _repo_root() / "tools" / "goldens" / "models-cache" / "edgetam"

_engine_cache: OrtEngine | None = None
def get_engine() -> OrtEngine:
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = OrtEngine(MODELS_DIR)     # fp32 *.onnx for CPU determinism (see friction note)
    return _engine_cache

def preprocess_frame(pil_image, size=1024): ...   # square-stretch resize + ImageNet normalize,
                                                    # ported from the HF video_processor call in
                                                    # e2e_loop.main() so a single still frame
                                                    # (vid_segment) doesn't need the full
                                                    # AutoProcessor/transformers dependency
def rescale_prompt(x, y, orig_w, orig_h, size=1024): ...   # mirrors PROMPT_POINT_XY * 1024/W
def rle_encode(mask) / rle_decode(rle) -> np.ndarray: ...  # COCO-style, matches golden-mask-f*.rle.json
def run_track(video_frames, prompt_frame_idx, point_or_box, progress_cb=None): ...
    # the per-frame loop body of e2e_loop.main()'s `for t in range(NUM_FRAMES)` block,
    # lifted out so vid_track's worker can call it directly instead of duplicating the
    # MemoryBank/OrtEngine choreography. progress_cb(t, num_frames) drives status.json updates.
```

`get_engine()` loads the **fp32** graphs (`vision_encoder.onnx` etc., not the `_fp16` variants) —
Phase B runs on CPU (`CPUExecutionProvider`, same as `e2e_loop.py`), where fp16 buys nothing and
the fp32 graphs are what the IoU golden was captured against, keeping the pytest gate tight without
a fp16-vs-fp32 tolerance fudge. Studio's browser path uses fp16 for download-size/WebGPU reasons
that don't apply here.

**`run_track` and `preprocess_frame`/`rescale_prompt` do not exist in `e2e_loop.py` today** — see
Friction §7.1. They are new thin functions in `_websam_ort.py`, built by literally moving the
existing loop-body code (no math changes), not new tracking logic.

## 5. Test plan (uv pytest, no fabri install, no LLM required)

`integrations/fabri/pyproject.toml` declares a `uv` project; `uv run --extra test pytest` from
`integrations/fabri/` runs everything in this section with **zero** network access and **zero**
`fabri` package installed.

### 5.1 Core correctness (mandatory, always runs)

Each test invokes the tool script exactly as the fabri runner would — `subprocess.run([*manifest
command], input=json.dumps(args), capture_output=True, text=True)`, reading the manifest's
`command` from its `.json` file via a tiny local re-implementation of `ToolManifest.from_file`'s
JSON-load-and-parse (not importing `fabri`) — plus a helper that sets `FABRI_SANDBOX_ROOT` to a
pytest `tmp_path` and copies `tools/goldens/fixtures/video/clip-256.mp4` into it first.

- **`test_vid_track.py`**: submit `vid_track` on `clip-256.mp4` with the golden prompt
  (`point={x:60,y:128}` at `promptFrameSec=0`, per `golden-video-meta.json`), poll `vid_poll_job`
  until `done`, decode each `masks.rle.json[i]` and `golden-mask-f{i}.rle.json`, assert
  `iou(ours, golden) >= 0.90` per frame (same bar as `docs/plans/...` §Phase-B-gate; note this is
  the **Python fp32 CPU** path so it should track closer to `e2e_loop.py`'s own 0.95 IOU_GATE, but
  0.90 is the contract-level floor to match the plan).
- **`test_vid_segment.py`**: single-frame `vid_segment` on frame 0 of the golden clip with the same
  point prompt, decode the returned mask PNG, IoU vs `golden-mask-f0.rle.json` >= 0.90.
- **`test_vid_export_matte.py`**: feed `vid_track`'s output `maskDir` into
  `vid_export_matte(format='matte_zip')`, assert `zipfile.is_zipfile(outputPath)` and that it
  contains 10 PNG entries; separately assert `format='mp4_cutout'` either produces a playable file
  (`ffprobe` exit 0) or returns the documented `warning` fallback — never a hard failure.
- **`test_vid_extract_frame.py`**: extract frame at `timeSec=0.3` from the golden clip (10 fps, so
  frame index 3), assert PNG dimensions == 256×256.
- **`test_vid_composite_trim_concat.py`**: smoke tests only (produce a non-empty output file of the
  expected container/duration via `ffprobe`) — these are thin ffmpeg wrappers, not model-bearing.

### 5.2 `vid_ground_text` (mandatory stub path; real path gated)

- **`test_vid_ground_text.py::test_stub`**: set `WEBSAM_GROUND_TEXT_STUB=1`, call with
  `phrase="the red ball"` against a fixture image, assert the returned point falls inside the
  fixture's known ball bounding box. Always runs.
- **`test_vid_ground_text.py::test_real_gemini`**: `@pytest.mark.skipif(not
  os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY")` — real network call, only runs
  when a key is present (CI leaves it skipped by default).

### 5.3 OPTIONAL full-agent e2e (documented, skippable, not part of the default gate)

```python
# tests/test_e2e_agent.py
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fabri") is None or not os.environ.get("GEMINI_API_KEY"),
    reason="requires `pip install -e ~/gba/fabri` + GEMINI_API_KEY (see README.md)",
)

def test_track_and_export_matte(tmp_project):
    from fabri import run_agent, build_llm, build_tools, build_tool_defs
    from fabri.config import load_config
    config = load_config(".agent/fabri_agent.yaml")
    ...
    result = run_agent(
        "track the object in clip-256.mp4 and export a matte",
        llm, tools, store, max_steps=config["agent"]["max_steps"],
    )
    assert result["outcome"] == "success"
    assert Path(tmp_project, result["response"]["outputPath"]).exists()
```
README.md documents the two extra setup steps (`pip install -e /Users/rushour0/gba/fabri`,
`export GEMINI_API_KEY=...`) and that this test is **excluded from the default `pytest` run** via
`pyproject.toml`'s `[tool.pytest.ini_options] markers` + `-m "not e2e_agent"` default addopts, run
explicitly with `pytest -m e2e_agent`.

## 6. Ownership table + wave plan

One agent per file (or per tight file-group where a manifest+script pair is inseparable); no two
agents touch the same file.

| File | Wave | Owner scope |
|---|---|---|
| `pyproject.toml` | 1 | deps only |
| `tools/video_editing/_websam_ort.py` | 1 | ORT core wrapper (§4) |
| `tools/video_editing/vid_track.{json,py}` | 1 | + `_track_worker.py` (job worker) |
| `tools/video_editing/vid_poll_job.{json,py}` | 1 | job-status reader |
| `tools/video_editing/vid_segment.{json,py}` | 1 | single-frame path |
| `tools/video_editing/vid_extract_frame.{json,py}` | 2 | ffmpeg/imageio frame grab |
| `tools/video_editing/vid_ground_text.{json,py}` | 2 | + stub fixture json |
| `tools/video_editing/vid_export_matte.{json,py}` | 2 | zip + mp4-cutout |
| `tools/video_editing/vid_composite.{json,py}` | 2 | |
| `tools/video_editing/vid_trim.{json,py}` | 2 | |
| `tools/video_editing/vid_concat.{json,py}` | 2 | |
| `.agent/fabri_agent.yaml` | 3 | |
| `.agent/prompts/orchestrator.md` | 3 | |
| `.agent/prompts/domains/video_editing.md` | 3 | |
| `tests/*.py` (all) | 3 | one agent, after tool scripts exist and are stable |
| `README.md` | 3 | |

**Wave 1** (ORT core, can build/test standalone against the golden clip without any fabri manifest
plumbing): `_websam_ort.py`, `vid_track` + `_track_worker.py`, `vid_poll_job`, `vid_segment`,
`pyproject.toml`. Gate: a throwaway script (not the final pytest suite) confirming `run_track`
reproduces `e2e_loop.py`'s own IoU numbers.

**Wave 2** (independent ffmpeg/LLM tools, each touches disjoint files, all depend only on
`_websam_ort.py`'s `rle_decode`/sandbox convention, not on each other): `vid_extract_frame`,
`vid_ground_text`, `vid_export_matte`, `vid_composite`, `vid_trim`, `vid_concat`.

**Wave 3** (glue + verification, depends on every tool script existing): `fabri_agent.yaml`,
`orchestrator.md`, `domains/video_editing.md`, the full `tests/` suite, `README.md`. This wave is
where the mandatory §5.1/§5.2 gate actually runs and is where I'd expect the wave to report red at
least once (IoU threshold tuning, RLE encode/decode edge cases) before going green.

## 7. Friction flags

### 7.1 `e2e_loop.py` is a script, not a library — moderate refactor needed

`e2e_loop.py`'s `OrtEngine` and `MemoryBank` classes are already import-friendly (plain classes,
`from clip_util import ...` is the only script-shaped coupling, and `_websam_ort.py` sidesteps that
by inserting `m2-edgetam/` onto `sys.path` rather than requiring an install). But the actual
per-frame tracking **loop lives inline inside `main()`** (`for t in range(NUM_FRAMES): ...`,
lines ~225–258), interleaved with golden-loading, `AutoProcessor.from_pretrained(...)` (a
`transformers` dependency Phase B does not want as a hard runtime dep — the graphs already bake in
preprocessing-adjacent shapes; only the resize+normalize math is needed, not the tokenizer/HF-hub
network fetch), and the IoU-gate print/raise. Wave-1 work must:
1. Extract the loop body into `run_track(frames, prompt_frame_idx, prompt, progress_cb)` (either
   as a **new function added to `e2e_loop.py`** so the spike file itself gains a reusable entry
   point, or purely inside `_websam_ort.py` by copying the ~30 lines — decide in Wave 1 based on
   whether `tools/export` maintainers want `e2e_loop.py` touched at all; if not, copy + comment
   pointing back at the line range, never fork the *math*).
2. Reimplement `AutoProcessor`'s square-stretch resize + ImageNet normalize as ~15 lines of
   PIL/numpy in `_websam_ort.py::preprocess_frame` (documented in `FINDINGS.md`'s "Preprocessing
   contract" already, per the module docstring) instead of pulling in `transformers` +
   network-fetching the HF processor config at tool-run time — a hard requirement for the sandboxed
   tool to work offline.
3. `tpos_table.npy` (`activations/tpos_table.npy`, produced by `dump_constants.py`) is a **required
   input** to `MemoryBank.__init__` — confirm it's checked into `tools/export/spikes/m2-edgetam/activations/`
   or regenerate/vendor it under `integrations/fabri/` at Wave-1 time; the design doc assumes it's
   already committed (verify in Wave 1, first task, before writing `run_track`).

### 7.2 Vision-LLM provider choice is a real decision, not a default

Picked **Gemini** to reuse the orchestrator's own `api_key_env` and get native JSON-schema
structured output (fewer coordinate-parsing failures than free-text). Claude vision is a documented
fallback (`ANTHROPIC_API_KEY`, swap the client in `vid_ground_text.py`) if Gemini's grounding
accuracy on small/occluded objects proves worse in practice — flag for a Wave-2 spike-check against
a handful of real phrases on `clip-256.mp4`, not just the deterministic stub.

### 7.3 CPU inference latency for `vid_track` is unmeasured beyond the 10-frame golden

`e2e_loop.py` runs 10 frames on `CPUExecutionProvider` in low single-digit seconds locally, but
Phase B's `timeout_s: 15` on `vid_track` (the submit call, not the worker) assumes the **worker**
detaches cleanly before that budget — untested. A Studio-length clip (hundreds of frames) run
end-to-end via `vid_poll_job` could take minutes; that's fine (poll pattern absorbs it) but the
agent's own `max_steps: 20` budget must not be exhausted by polling — `domains/video_editing.md`
should instruct the model to poll with a bounded number of calls (e.g. back off, or accept "still
running" as a valid final answer with a resumable `jobId` in `response_schema`) rather than
poll-spinning until `max_steps` runs out. Flagging for Wave 3 prompt-writing, not a code fix.

### 7.4 BUSL boundary

Confirmed nothing here needs fabri core changes: `spawn_subagent` isn't used (single-domain agent),
`response_schema`/`error_strategy`/`response_retries` are all existing config knobs
(`creating-an-agent.md` §2), and the submit→poll pattern is pure userland (a tool script managing
its own background subprocess + status file) — no framework primitive for "long-running tool" is
needed. If a future domain wants native async tool support or image-input to the orchestrator LLM,
that's a fabri-core ask and the user (licensor, `pataderushikesh@gmail.com`) decides — out of scope
for Phase B, which is designed specifically to avoid needing it.
