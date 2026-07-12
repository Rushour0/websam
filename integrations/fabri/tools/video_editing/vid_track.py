"""vid_track — submit half of the submit->poll long-job pattern (fabri
contract §3.4). Fabri's runner blocks synchronously on this script and
SIGKILLs it at `timeout_s` (15s, see vid_track.json), so this script must
NEVER run the actual tracking loop itself — it only:

  1. validates + sandbox-jails the input video path,
  2. allocates a job dir under `$FABRI_SANDBOX_ROOT/artifacts/jobs/<jobId>/`,
  3. writes an initial `status.json` ({"status": "running", "progress": 0.0}),
  4. spawns `_track_worker.py` as a fully detached subprocess
     (`start_new_session=True`, stdin/stdout/stderr redirected to files —
     never inherited pipes, which would keep this process's fds open and
     block fabri's `proc.communicate()` from returning), and
  5. returns `{jobId, status: "running"}` immediately.

The worker (not this script) does the actual `_websam_ort.run_track` call
and owns all subsequent `status.json` writes. Poll via `vid_poll_job.py`.

Sandbox jailing follows the exact pattern in fabri's own
`tools/examples/write_file.py` (root_env -> resolve -> is_relative_to
check), per docs/fabri-contracts.md §3.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

SANDBOX_ROOT_ENV = "FABRI_SANDBOX_ROOT"


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def _resolve_sandboxed(root: Path, rel_path: str) -> Path | None:
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root):
        return None
    return target


def main() -> int:
    try:
        args = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        return _fail(f"invalid JSON on stdin: {e}")

    root_env = os.environ.get(SANDBOX_ROOT_ENV)
    if not root_env:
        return _fail(f"{SANDBOX_ROOT_ENV} is not set; refusing to run unsandboxed")
    root = Path(root_env).resolve()

    video_path = args.get("videoPath")
    if not video_path:
        return _fail("missing required field: videoPath")
    prompt_frame_sec = args.get("promptFrameSec")
    if prompt_frame_sec is None:
        return _fail("missing required field: promptFrameSec")

    point = args.get("point")
    box = args.get("box")
    if bool(point) == bool(box):
        return _fail("exactly one of point/box must be given")

    video_target = _resolve_sandboxed(root, video_path)
    if video_target is None:
        return _fail(f"path escapes sandbox root: {video_path}")
    if not video_target.is_file():
        return _fail(f"videoPath does not exist: {video_path}")

    job_id = uuid.uuid4().hex[:16]
    job_dir = root / "artifacts" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job_args = {
        "videoPath": str(video_target),
        "promptFrameSec": prompt_frame_sec,
        "point": point,
        "box": box,
    }
    args_path = job_dir / "args.json"
    args_path.write_text(json.dumps(job_args))

    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running", "progress": 0.0}))

    worker_script = Path(__file__).resolve().parent / "_track_worker.py"
    log_path = job_dir / "worker.log"

    import subprocess

    with open(log_path, "ab", buffering=0) as log_fh:
        subprocess.Popen(
            [sys.executable, str(worker_script), str(root), job_id],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,  # detach fully: survives this process's exit
            close_fds=True,
        )

    print(json.dumps({"jobId": job_id, "status": "running"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
