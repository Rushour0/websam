"""fabri tool: vid_composite -- mask-driven cutout / highlight / background-swap.

Contract + sandbox-jail pattern match vid_trim.py / write_file.py (see those
docstrings). Reuses _websam_ort.decode_video / rle_decode (do NOT
reimplement -- see integrations/fabri/docs/fabri-contracts.md §3 "ORT
reuse") for video decode and COCO-RLE mask decode; needs no ONNX model
inference itself (masks are supplied, not computed here).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SANDBOX_ROOT_ENV = "FABRI_SANDBOX_ROOT"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _websam_ort import decode_video, rle_decode  # noqa: E402  (reused, not duplicated)

DEFAULT_HIGHLIGHT_COLOR = (255, 215, 0)  # gold
DEFAULT_BACKGROUND_COLOR = (0, 200, 0)   # green-screen swap default
HIGHLIGHT_ALPHA = 0.45


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


def _load_masks_dir(masks_dir: Path) -> list:
    import numpy as np
    from PIL import Image

    files = sorted(p for p in masks_dir.iterdir() if p.suffix.lower() == ".png")
    if not files:
        raise ValueError(f"masksDir contains no PNG masks: {masks_dir}")
    masks = []
    for f in files:
        arr = np.asarray(Image.open(f).convert("L"))
        masks.append((arr > 127).astype("uint8"))
    return masks


def _load_masks_rle(rle_path: Path) -> list:
    data = json.loads(rle_path.read_text())
    if isinstance(data, list):
        return [rle_decode(rle) for rle in data]
    return [rle_decode(data)]


def _align_masks(masks: list, num_frames: int) -> tuple[list, str | None]:
    warning = None
    if len(masks) < num_frames:
        warning = (
            f"only {len(masks)} mask(s) for {num_frames} frame(s); "
            "repeating the last mask for remaining frames"
        )
        masks = masks + [masks[-1]] * (num_frames - len(masks))
    elif len(masks) > num_frames:
        warning = f"{len(masks)} masks but only {num_frames} frame(s); extra masks dropped"
        masks = masks[:num_frames]
    return masks, warning


def _composite_frame(frame, mask, mode: str, color: tuple[int, int, int]):
    import numpy as np

    frame = frame.astype("float32")
    mask_bool = mask.astype(bool)
    if mask_bool.shape != frame.shape[:2]:
        from PIL import Image
        mask_img = Image.fromarray((mask_bool.astype("uint8")) * 255).resize(
            (frame.shape[1], frame.shape[0]), resample=Image.NEAREST
        )
        mask_bool = np.asarray(mask_img) > 127

    out = frame.copy()
    color_arr = np.array(color, dtype="float32")

    if mode == "cutout":
        out[~mask_bool] = 0.0
    elif mode == "background":
        out[~mask_bool] = color_arr
    elif mode == "highlight":
        blended = frame * (1 - HIGHLIGHT_ALPHA) + color_arr * HIGHLIGHT_ALPHA
        out[mask_bool] = blended[mask_bool]
    else:
        raise ValueError(f"unknown mode: {mode}")

    return out.clip(0, 255).astype("uint8")


def main() -> int:
    args = json.loads(sys.stdin.read())
    try:
        root = _sandbox_root()
        video_path = _resolve_in(root, args["video"])
        if not video_path.is_file():
            raise ValueError(f"video not found: {args['video']}")
        mode = args["mode"]
        if mode not in ("cutout", "highlight", "background"):
            raise ValueError(f"unknown mode: {mode}")
        has_masks_dir = bool(args.get("masksDir"))
        has_rle_json = bool(args.get("rleJson"))
        if has_masks_dir == has_rle_json:
            raise ValueError("exactly one of masksDir or rleJson is required")

        if has_masks_dir:
            masks_dir = _resolve_in(root, args["masksDir"])
            if not masks_dir.is_dir():
                raise ValueError(f"masksDir not found: {args['masksDir']}")
            masks = _load_masks_dir(masks_dir)
        else:
            rle_path = _resolve_in(root, args["rleJson"])
            if not rle_path.is_file():
                raise ValueError(f"rleJson not found: {args['rleJson']}")
            masks = _load_masks_rle(rle_path)

        color = args.get("color")
        if color is not None:
            color = tuple(int(c) for c in color)
    except (KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return 1

    import imageio.v3 as iio

    frames = decode_video(video_path)
    if not frames:
        print(json.dumps({"error": "no frames decoded from video"}))
        return 1

    masks, warning = _align_masks(masks, len(frames))

    if color is None:
        color = DEFAULT_HIGHLIGHT_COLOR if mode == "highlight" else DEFAULT_BACKGROUND_COLOR

    out_frames = [
        _composite_frame(frame, mask, mode, color) for frame, mask in zip(frames, masks)
    ]

    meta = iio.immeta(video_path, plugin="FFMPEG")
    fps = float(meta.get("fps", 10.0))

    out_dir = root / "artifacts" / "composite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}-{mode}.mp4"

    iio.imwrite(out_path, out_frames, fps=fps, plugin="FFMPEG", codec="libx264")

    result = {"outputPath": str(out_path.relative_to(root))}
    if warning:
        result["warning"] = warning
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
