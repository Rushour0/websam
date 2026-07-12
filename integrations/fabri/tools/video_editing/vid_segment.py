"""fabri tool: vid_segment — single-frame EdgeTAM segmentation.

Reads one JSON object from stdin, writes one JSON object to stdout, exit 0 =
ok (per docs/fabri-contracts.md §3; mirrors fabri's own examples/fetch_url.py
— this script does NOT self-wrap in {"ok": ...}, it prints the raw result
payload and lets the fabri runner do that).

Contract: EXACTLY ONE of {video, frameIndex} or {frame} selects the source
frame; {prompt} is one of {point}, {points}, or {box} in original pixel
coords (same shape _websam_ort.segment_frame's _prompt_tensors_1024 expects).
Writes a 0/255 PNG mask under $FABRI_SANDBOX_ROOT and returns
{maskPath, area, width, height}.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _websam_ort import decode_video, load_engine, segment_frame  # noqa: E402

# NOTE: _websam_ort.py's own MODELS_DIR_DEFAULT (via its _repo_root(), parents[3])
# resolves one level too shallow from this file's location and points at
# integrations/tools/goldens/... instead of <repo>/tools/goldens/... (confirmed by
# running this tool: NO_SUCHFILE against that wrong path). tests/test_websam_ort.py
# sidesteps the same bug by passing an explicit models_dir; do the same here rather
# than editing _websam_ort.py (owned by another agent/wave). Flag for a follow-up fix
# to _websam_ort.py's _repo_root() (should be parents[4], not parents[3]).
_MODELS_DIR = pathlib.Path(__file__).resolve().parents[4] / "tools" / "goldens" / "models-cache" / "edgetam"


def _sandbox_root() -> pathlib.Path:
    root = os.environ.get("FABRI_SANDBOX_ROOT")
    if not root:
        raise ValueError("FABRI_SANDBOX_ROOT is not set")
    return pathlib.Path(root).resolve()


def _resolve_in(root: pathlib.Path, rel: str) -> pathlib.Path:
    """Resolve `rel` against `root`, refusing anything that escapes it
    (mirrors write_file.py's target.is_relative_to(root) pattern per
    docs/fabri-contracts.md §3)."""
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"refused: path {rel!r} escapes sandbox root")
    return p


def _load_frame(root: pathlib.Path, args: dict) -> np.ndarray:
    has_video = "video" in args
    has_frame = "frame" in args
    if has_video == has_frame:
        raise ValueError("exactly one of {video, frameIndex} or {frame} is required")

    if has_frame:
        frame_path = _resolve_in(root, args["frame"])
        if not frame_path.is_file():
            raise ValueError(f"frame not found: {args['frame']!r}")
        return np.asarray(Image.open(frame_path).convert("RGB"))

    if "frameIndex" not in args:
        raise ValueError("frameIndex is required when video is given")
    video_path = _resolve_in(root, args["video"])
    if not video_path.is_file():
        raise ValueError(f"video not found: {args['video']!r}")
    frames = decode_video(video_path)
    frame_index = int(args["frameIndex"])
    if not (0 <= frame_index < len(frames)):
        raise ValueError(
            f"frameIndex {frame_index} out of range for {len(frames)}-frame video"
        )
    return frames[frame_index]


def main() -> int:
    args = json.loads(sys.stdin.read())

    try:
        root = _sandbox_root()
        frame = _load_frame(root, args)
        prompt = args.get("prompt")
        if not prompt or not any(k in prompt for k in ("point", "points", "box")):
            raise ValueError("prompt must have one of 'point', 'points', or 'box'")
    except (ValueError, KeyError) as e:
        print(json.dumps({"error": str(e)}))
        return 1

    try:
        engine = load_engine(_MODELS_DIR)
        mask = segment_frame(engine, frame, prompt)  # (H, W) uint8 {0,1}
    except Exception as e:  # noqa: BLE001 - surface any inference failure to the caller
        print(json.dumps({"error": f"segmentation failed: {e}"}))
        return 1

    height, width = mask.shape
    area = int(mask.sum())

    out_dir = root / "artifacts" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.get("frame") or args.get("video") or "frame").stem
    frame_index = args.get("frameIndex")
    suffix = f"-f{frame_index}" if frame_index is not None else ""
    out_path = out_dir / f"{stem}{suffix}-mask.png"
    # Disambiguate if a mask for this stem already exists (repeat calls in one run).
    n = 1
    while out_path.exists():
        out_path = out_dir / f"{stem}{suffix}-mask-{n}.png"
        n += 1

    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_path)

    mask_rel = out_path.relative_to(root).as_posix()
    print(json.dumps({"maskPath": mask_rel, "area": area, "width": width, "height": height}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
