"""websam fabri gateway — FastAPI app (factory pattern) over fabri's FabriService.

Turns the one-shot CLI fabri video_editing agent into a conversational feature
reachable from apps/studio's UI. Multipart video upload -> per-run sandbox dir ->
FabriService.submit(overrides=...) -> live SSE trace -> final envelope -> artifact
download. Provider/secret config is driven entirely by a git-ignored .env via the
sibling `settings` module; nothing here is hardcoded.

Run:  uvicorn --factory service.app:create_app
Tests set env (notably FABRI_GATEWAY_DOTENV -> a nonexistent path so a developer's
real .env can't leak into assertions) then call create_app() directly.

fabri itself (BUSL, installed separately — see integrations/fabri/README.md) is
imported LAZILY inside _get_service so /health and the test suite work without a
fabri checkout present.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .settings import (
    DEFAULT_GROUND_TEXT_STUB,
    FABRI_INTEGRATION_DIR,
    SettingsError,
    TEMPLATE_CONFIG,
    build_overrides,
    load_settings,
)

logger = logging.getLogger("websam.fabri.gateway")


def _check_origin(request: Request) -> None:
    """Origin guard for state-changing / credit-burning requests.

    A multipart POST is a CORS "simple request" — no preflight — and
    CORSMiddleware only gates whether the browser lets JS *read* the response,
    not whether the request reaches us. So any web page could fire a drive-by
    POST that burns the user's LLM credits. If an Origin header is present and
    not in the configured allow-list, reject it. No-Origin requests (curl,
    same-machine tools) pass untouched.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return
    if origin not in request.app.state.settings.cors_origins:
        raise HTTPException(status_code=403, detail="origin not allowed")


def _get_service(app: FastAPI):
    """Lazily construct + cache the single FabriService instance.

    fabri is imported here (not at module top) so the module imports and /health
    serves without a fabri checkout installed.
    """
    if app.state.fabri_service is None:
        try:
            from fabri.service.service import FabriService
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "fabri is not installed — pip install -e /path/to/your/fabri "
                    "checkout (see integrations/fabri/README.md)"
                ),
            )
        settings = app.state.settings
        app.state.fabri_service = FabriService(
            template_config=str(TEMPLATE_CONFIG),
            home_root=settings.runs_dir / "homes",
        )
    return app.state.fabri_service


def create_app() -> FastAPI:
    # (1) Load .env into the process env. override=False so real process env
    # always wins. FABRI_GATEWAY_DOTENV is an internal test seam: tests point it
    # at a nonexistent file so a developer's real .env can't leak into assertions.
    load_dotenv(
        os.environ.get(
            "FABRI_GATEWAY_DOTENV", str(FABRI_INTEGRATION_DIR / ".env")
        )
    )

    # (2) Resolve settings. Let SettingsError propagate and kill startup loudly:
    # uvicorn prints the traceback naming the missing env vars. Do NOT catch it.
    settings = load_settings()

    # (3) If no GEMINI_API_KEY, force vid_ground_text onto its offline stub.
    # Setting os.environ here suffices: launcher.py:150 copies os.environ into
    # every agent child, and tools/runner.py layers tool envs on top — so this
    # var reaches every tool subprocess before any submit() runs.
    if settings.ground_text_stub_forced:
        os.environ["WEBSAM_GROUND_TEXT_STUB"] = DEFAULT_GROUND_TEXT_STUB
        logger.warning(
            "GEMINI_API_KEY not set — vid_ground_text will run in offline stub "
            "mode (reduced accuracy: text-grounding returns a fixed point, not a "
            "real vision call). Set GEMINI_API_KEY in .env to enable real grounding."
        )

    # (4) Ensure the per-run scratch layout exists.
    (settings.runs_dir / "sandboxes").mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / "homes").mkdir(parents=True, exist_ok=True)

    # (5) Build the app.
    app = FastAPI(title="websam fabri gateway")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    # session_id -> that session's sandbox dir. In-memory only: a gateway
    # restart orphans live sessions' lookups, the same property fabri's own
    # _runs dict has (service.py:70). Acceptable for v1.
    app.state.run_dirs = {}  # type: dict[str, Path]
    app.state.fabri_service = None

    # ------------------------------------------------------------------ [A]
    @app.post("/runs")
    def create_run(
        request: Request,
        video: UploadFile = File(...),
        task: str = Form(...),
    ):
        # PLAIN SYNC def on purpose: shutil.copyfileobj of a multi-GB spooled
        # upload + svc.submit's subprocess spawn would stall the event loop and
        # freeze every open SSE stream. Sync def runs in Starlette's threadpool
        # like [B]/[C]; UploadFile.file and svc.submit are both sync APIs.
        _check_origin(request)
        if not task.strip():
            raise HTTPException(status_code=422, detail="task must not be empty")

        svc = _get_service(app)

        run_dir = settings.runs_dir / "sandboxes" / uuid.uuid4().hex
        run_dir.mkdir(parents=True)

        fname = (
            re.sub(r"[^A-Za-z0-9._-]", "_", Path(video.filename or "input.mp4").name)
            or "input.mp4"
        )
        dest = run_dir / fname
        with dest.open("wb") as out:
            shutil.copyfileobj(video.file, out)

        full_task = (
            f"{task.strip()}\n\n"
            f'The input video is at "{fname}" (a sandbox-relative path).'
        )
        session_id = svc.submit(
            full_task, overrides=build_overrides(settings, run_dir)
        )
        app.state.run_dirs[session_id] = run_dir
        # Key casing is EXACTLY 'sessionId' — ChatPanel contract.
        return {"sessionId": session_id}

    # ------------------------------------------------------------------ [B]
    @app.get("/runs/{session_id}/events")
    def stream_events(session_id: str):
        # SYNC def: svc.stream is a blocking poll loop (tailer.py:97-119); each
        # open SSE holds one threadpool thread (default pool of 40 bounds
        # concurrency).
        if session_id not in app.state.run_dirs:
            raise HTTPException(status_code=404, detail="unknown session")
        svc = _get_service(app)

        def event_gen():
            # Trace events are UNNAMED (no 'event:' field) so the browser's
            # EventSource.onmessage fires — ChatPanel contract. Events pass
            # through untransformed: fabri vocabulary {type: narration|thought|
            # tool_started|tool_call|step_finished|final|failed|incomplete|
            # error|usage, text?, name?, reason?, outcome?}.
            for event in svc.stream(session_id, timeout=3600.0):
                yield f"data: {json.dumps(event)}\n\n"
            # ONE final NAMED sentinel: EventSource would otherwise auto-reconnect
            # forever; ChatPanel listens for it via addEventListener('end').
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ [C]
    @app.get("/runs/{session_id}/result")
    def get_result(session_id: str):
        # SYNC def. Returns fabri's envelope VERBATIM — all eight keys
        # {session_id, success, outcome, final_text, structured_output, usage,
        # cost, error} (service.py:109-131). Deliberately NOT wrapped in a strict
        # response model that would drop usage/cost.
        if session_id not in app.state.run_dirs:
            raise HTTPException(status_code=404, detail="unknown session")
        svc = _get_service(app)
        try:
            return svc.result(session_id, timeout=600.0)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="run still in progress")

    # ------------------------------------------------------------------ [D]
    @app.get("/runs/{session_id}/artifact")
    def get_artifact(session_id: str, path: str):
        # Deliberately never calls _get_service so it's testable without fabri.
        run_dir = app.state.run_dirs.get(session_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="unknown session")

        # PATH JAIL (mandatory): resolve against the run dir and refuse escapes.
        base = run_dir.resolve()
        candidate = Path(path)
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (base / candidate).resolve()
        )
        if not target.is_relative_to(base):
            raise HTTPException(
                status_code=403, detail="path escapes the run sandbox"
            )
        if not target.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")

        return FileResponse(
            target,
            media_type=mimetypes.guess_type(target.name)[0]
            or "application/octet-stream",
            filename=target.name,
        )

    # ------------------------------------------------------------------ [E]
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "provider": settings.provider,
            "model": settings.model,
            "groundTextStub": settings.ground_text_stub_forced,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1, NOT 0.0.0.0: an unauthenticated gateway must not be LAN-exposed
    # by default. 0.0.0.0 lives only in the Docker CMD, where the container
    # boundary is the exposure control.
    uvicorn.run(create_app(), host="127.0.0.1", port=load_settings().port)
