"""fabri tool: vid_concat -- concatenate multiple video clips in order.

Contract + sandbox-jail pattern match vid_trim.py / write_file.py (see those
docstrings). Reads all frames of every input clip, resamples each clip to
the FIRST clip's fps (simple nearest-frame resample -- clips are expected to
be short/low-fps segmentation-tool outputs, not long-form footage), and
writes one mp4 via imageio-ffmpeg. No system ffmpeg binary required.
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


def _resample(frames: list, src_fps: float, dst_fps: float) -> list:
    if not frames or src_fps == dst_fps:
        return frames
    src_duration = len(frames) / src_fps
    n_out = max(1, round(src_duration * dst_fps))
    out = []
    for i in range(n_out):
        src_idx = min(len(frames) - 1, round(i * src_fps / dst_fps))
        out.append(frames[src_idx])
    return out


def main() -> int:
    args = json.loads(sys.stdin.read())
    try:
        root = _sandbox_root()
        video_rels = args["videos"]
        if not isinstance(video_rels, list) or len(video_rels) < 2:
            raise ValueError("videos must be a list of at least 2 sandbox-relative paths")
        video_paths = [_resolve_in(root, rel) for rel in video_rels]
        for p, rel in zip(video_paths, video_rels):
            if not p.is_file():
                raise ValueError(f"video not found: {rel}")
    except (KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return 1

    import imageio.v3 as iio

    metas = [iio.immeta(p, plugin="FFMPEG") for p in video_paths]
    target_fps = float(metas[0].get("fps", 10.0))

    all_frames = []
    for p, meta in zip(video_paths, metas):
        clip_fps = float(meta.get("fps", target_fps))
        clip_frames = [f for f in iio.imiter(p, plugin="FFMPEG")]
        all_frames.extend(_resample(clip_frames, clip_fps, target_fps))

    if not all_frames:
        print(json.dumps({"error": "no frames decoded from input videos"}))
        return 1

    out_dir = root / "artifacts" / "concat"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = "-".join(p.stem for p in video_paths)[:80]
    out_path = out_dir / f"concat-{stems}.mp4"

    iio.imwrite(out_path, all_frames, fps=target_fps, plugin="FFMPEG", codec="libx264")

    duration_sec = len(all_frames) / target_fps
    print(json.dumps({
        "outputPath": str(out_path.relative_to(root)),
        "durationSec": duration_sec,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
