"""fabri tool: vid_trim -- cut a video to [startSec, endSec).

Reads one JSON object from stdin, writes one JSON object to stdout, exit 0 =
ok (mirrors fabri's fetch_url.py/write_file.py contract -- no self-wrapping
{ok: ...}, the runner does that). Sandbox jail pattern copied from
write_file.py (fabri's own example tool): resolve against
$FABRI_SANDBOX_ROOT, refuse escapes.

Uses imageio.v3 (imageio-ffmpeg backend) per integrations/fabri/pyproject.toml
-- no system ffmpeg binary required.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SANDBOX_ROOT_ENV = "FABRI_SANDBOX_ROOT"


def _sandbox_root() -> Path:
    root_env = os.environ.get(SANDBOX_ROOT_ENV)
    if not root_env:
        raise ValueError(f"{SANDBOX_ROOT_ENV} is not set; refusing to run unsandboxed")
    return Path(root_env).resolve()


def _resolve_in(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes sandbox root: {rel}")
    return target


def main() -> int:
    args = json.loads(sys.stdin.read())
    try:
        root = _sandbox_root()
        video_path = _resolve_in(root, args["video"])
        if not video_path.is_file():
            raise ValueError(f"video not found: {args['video']}")
        start_sec = float(args["startSec"])
        end_sec = float(args["endSec"])
        if end_sec <= start_sec:
            raise ValueError(f"endSec ({end_sec}) must be > startSec ({start_sec})")
    except (KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return 1

    import imageio.v3 as iio

    meta = iio.immeta(video_path, plugin="FFMPEG")
    fps = float(meta.get("fps", 10.0))

    frames = []
    for i, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        t = i / fps
        if t < start_sec:
            continue
        if t >= end_sec:
            break
        frames.append(frame)

    if not frames:
        print(json.dumps({"error": "trim range selects zero frames"}))
        return 1

    out_dir = root / "artifacts" / "trim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}-trim-{start_sec:g}-{end_sec:g}.mp4"

    iio.imwrite(out_path, frames, fps=fps, plugin="FFMPEG", codec="libx264")

    duration_sec = len(frames) / fps
    print(json.dumps({
        "outputPath": str(out_path.relative_to(root)),
        "durationSec": duration_sec,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
