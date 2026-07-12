"""vid_poll_job — poll half of the submit->poll long-job pattern
(fabri contract §3.4). Never blocks: reads whatever `status.json`
`_track_worker.py` last wrote for the given jobId and returns it verbatim
(plus the sandbox jailing / jobId validation below). Does not touch the
video model or import _websam_ort at all.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SANDBOX_ROOT_ENV = "FABRI_SANDBOX_ROOT"
_JOB_ID_RE = re.compile(r"^[a-f0-9]+$")  # vid_track.py mints uuid4().hex[:16]


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def main() -> int:
    try:
        args = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        return _fail(f"invalid JSON on stdin: {e}")

    root_env = os.environ.get(SANDBOX_ROOT_ENV)
    if not root_env:
        return _fail(f"{SANDBOX_ROOT_ENV} is not set; refusing to run unsandboxed")
    root = Path(root_env).resolve()

    job_id = args.get("jobId")
    if not job_id:
        return _fail("missing required field: jobId")
    if not _JOB_ID_RE.match(job_id):
        # Defense in depth: jobId is model-controlled input threaded straight
        # into a path join below; reject anything that isn't the exact shape
        # vid_track.py mints rather than trust it not to contain `../`.
        return _fail(f"invalid jobId: {job_id}")

    job_dir = (root / "artifacts" / "jobs" / job_id).resolve()
    if not job_dir.is_relative_to(root):
        return _fail(f"invalid jobId: {job_id}")

    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return _fail(f"unknown jobId: {job_id}")

    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        # A concurrent write from _track_worker.py could in principle race
        # here; _track_worker.py writes via a tmp-file + os.replace so this
        # should be rare-to-never, but degrade to "still running" rather than
        # surface a parse error to the caller.
        status = {"status": "running"}

    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
