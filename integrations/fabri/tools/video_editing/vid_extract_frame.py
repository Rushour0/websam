"""Extract one video frame as a PNG under $FABRI_SANDBOX_ROOT.

args (stdin JSON): {video, timeSec?, frameIndex?} — exactly one of timeSec /
frameIndex. `video` is a sandbox-relative (or sandbox-absolute) path to the
source video. Uses imageio/imageio-ffmpeg only — no system ffmpeg required
beyond the imageio-ffmpeg-vendored binary already a project dependency.

stdout (JSON): {framePath, width, height, frameIndex} on success, exit 0.
{"error": "..."} on failure, exit 1. Mirrors fetch_url.py's contract: the
script prints its raw result payload, it does not self-wrap.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def _sandbox_root() -> pathlib.Path:
    root = os.environ.get("FABRI_SANDBOX_ROOT")
    return pathlib.Path(root).resolve() if root else pathlib.Path.cwd().resolve()


def _resolve_in_sandbox(root: pathlib.Path, rel: str) -> pathlib.Path:
    """Resolve `rel` against `root`, refusing any path that escapes it."""
    candidate = (root / rel).resolve() if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel).resolve()
    if not (candidate == root or root in candidate.parents):
        raise ValueError(f"refused: path {rel!r} escapes sandbox root {root}")
    return candidate


def main() -> int:
    args = json.loads(sys.stdin.read())
    root = _sandbox_root()

    video_arg = args.get("video") or args.get("videoPath")
    if not video_arg:
        print(json.dumps({"error": "missing required field 'video'"}))
        return 1

    time_sec = args.get("timeSec")
    frame_index = args.get("frameIndex")
    if time_sec is None and frame_index is None:
        print(json.dumps({"error": "must provide either timeSec or frameIndex"}))
        return 1

    try:
        video_path = _resolve_in_sandbox(root, video_arg)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        return 1

    if not video_path.is_file():
        print(json.dumps({"error": f"video not found: {video_arg}"}))
        return 1

    import imageio.v3 as iio

    try:
        if frame_index is None:
            meta = iio.immeta(video_path)
            fps = float(meta.get("fps") or 0.0)
            if fps <= 0:
                print(json.dumps({"error": "could not determine video fps to resolve timeSec"}))
                return 1
            frame_index = max(0, round(float(time_sec) * fps))

        frame = None
        for i, f in enumerate(iio.imiter(video_path)):
            if i == frame_index:
                frame = f
                break
        if frame is None:
            print(json.dumps({"error": f"frameIndex {frame_index} out of range"}))
            return 1
    except Exception as e:  # noqa: BLE001 — surface as tool error, not a crash
        print(json.dumps({"error": f"failed to decode video: {e}"}))
        return 1

    import numpy as np
    from PIL import Image

    arr = np.asarray(frame)
    img = Image.fromarray(arr).convert("RGB")

    out_dir = _resolve_in_sandbox(root, "artifacts/frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_path = out_dir / f"{stem}-{frame_index}.png"
    img.save(out_path)

    try:
        frame_path_out = str(out_path.relative_to(root))
    except ValueError:
        frame_path_out = str(out_path)

    print(json.dumps({
        "framePath": frame_path_out,
        "width": img.width,
        "height": img.height,
        "frameIndex": int(frame_index),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
