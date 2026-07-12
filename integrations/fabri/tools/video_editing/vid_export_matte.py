"""Package a tracked object's per-frame masks into a downloadable artifact.

args (stdin JSON): {masksDir?, rleJson?, format, video?} — exactly one of
masksDir (a sandbox-relative dir of frame-*.png mask files, white=object) or
rleJson (a sandbox-relative path to a COCO-RLE JSON: either a single
{width,height,counts} object or a list of them, same shape as
tools/goldens/fixtures/video/golden-mask-f*.rle.json).

format:
  - "png-sequence" (default/primary, most portable): zips the masks into
    matte.zip.
  - "mp4-cutout" (best-effort): needs `video` (the source clip); writes an
    MP4 with the background composited to black outside the mask (true RGBA
    alpha isn't broadly supported by MP4 containers/imageio's writer, so this
    is a documented best-effort, not a hard alpha channel). On any failure it
    falls back to the png-sequence path and returns a `warning` field instead
    of a hard error, per docs/fabri-contracts.md §3.5.

Also accepts the docs/fabri-contracts.md §3.5 field names (`maskDir`,
`maskRlePath`, `videoPath`, `format: 'matte_zip'|'mp4_cutout'`) as aliases so
either convention works against this tool.

stdout (JSON): {outputPath, frames, format, warning?} on success, exit 0.
{"error": "..."} on failure, exit 1.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import sys
import zipfile


def _sandbox_root() -> pathlib.Path:
    root = os.environ.get("FABRI_SANDBOX_ROOT")
    return pathlib.Path(root).resolve() if root else pathlib.Path.cwd().resolve()


def _resolve_in_sandbox(root: pathlib.Path, rel: str) -> pathlib.Path:
    candidate = (root / rel).resolve() if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel).resolve()
    if not (candidate == root or root in candidate.parents):
        raise ValueError(f"refused: path {rel!r} escapes sandbox root {root}")
    return candidate


def _to_sandbox_relative(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


_FRAME_IDX_RE = re.compile(r"(\d+)")


def _load_masks_from_dir(mask_dir: pathlib.Path):
    """Returns a sorted (by trailing integer in filename) list of
    (index, np.ndarray uint8 HxW 0/1)."""
    from PIL import Image
    import numpy as np

    entries = []
    for p in sorted(mask_dir.glob("*.png")):
        m = _FRAME_IDX_RE.findall(p.stem)
        idx = int(m[-1]) if m else len(entries)
        arr = np.asarray(Image.open(p).convert("L"))
        entries.append((idx, (arr > 127).astype(np.uint8)))
    entries.sort(key=lambda t: t[0])
    return entries


def _load_masks_from_rle(rle_path: pathlib.Path):
    # Reuse _websam_ort.rle_decode rather than reimplementing RLE decode
    # (matches the shared-core rule in docs/fabri-contracts.md §3.5).
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _websam_ort import rle_decode  # noqa: E402

    data = json.loads(rle_path.read_text())
    rles = data if isinstance(data, list) else [data]
    return [(i, rle_decode(rle)) for i, rle in enumerate(rles)]


def _mask_to_png_bytes(mask) -> bytes:
    from PIL import Image

    img = Image.fromarray((mask * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_matte_zip(root: pathlib.Path, masks) -> pathlib.Path:
    out_dir = _resolve_in_sandbox(root, "artifacts/mattes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matte.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, mask in masks:
            zf.writestr(f"frame-{idx:04d}.png", _mask_to_png_bytes(mask))
    return out_path


def _write_mp4_cutout(root: pathlib.Path, masks, video_path: pathlib.Path) -> pathlib.Path:
    import imageio.v3 as iio
    import numpy as np

    meta = iio.immeta(video_path)
    fps = float(meta.get("fps") or 10.0)

    mask_by_idx = dict(masks)
    n = min(len(masks), 1_000_000)

    out_dir = _resolve_in_sandbox(root, "artifacts/mattes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cutout.mp4"

    composited = []
    for i, frame in enumerate(iio.imiter(video_path)):
        if i >= n or i not in mask_by_idx:
            continue
        mask = mask_by_idx[i]
        arr = np.asarray(frame)[..., :3]
        m = mask.astype(np.uint8)[..., None]
        composited.append((arr * m).astype(np.uint8))
    if not composited:
        raise RuntimeError("no overlapping frames between masks and video")

    iio.imwrite(out_path, composited, fps=fps, codec="libx264")
    return out_path


def main() -> int:
    args = json.loads(sys.stdin.read())
    root = _sandbox_root()

    masks_dir_arg = args.get("masksDir") or args.get("maskDir")
    rle_json_arg = args.get("rleJson") or args.get("maskRlePath")
    video_arg = args.get("video") or args.get("videoPath")
    fmt = args.get("format")

    fmt_map = {
        "png-sequence": "png-sequence",
        "matte_zip": "png-sequence",
        "mp4-cutout": "mp4-cutout",
        "mp4_cutout": "mp4-cutout",
    }
    fmt_norm = fmt_map.get(fmt)
    if fmt_norm is None:
        print(json.dumps({"error": f"unknown format {fmt!r}; expected 'png-sequence' or 'mp4-cutout'"}))
        return 1

    if not masks_dir_arg and not rle_json_arg:
        print(json.dumps({"error": "must provide exactly one of masksDir or rleJson"}))
        return 1
    if masks_dir_arg and rle_json_arg:
        print(json.dumps({"error": "provide only one of masksDir or rleJson, not both"}))
        return 1

    try:
        if masks_dir_arg:
            mask_dir = _resolve_in_sandbox(root, masks_dir_arg)
            if not mask_dir.is_dir():
                print(json.dumps({"error": f"masksDir not found: {masks_dir_arg}"}))
                return 1
            masks = _load_masks_from_dir(mask_dir)
        else:
            rle_path = _resolve_in_sandbox(root, rle_json_arg)
            if not rle_path.is_file():
                print(json.dumps({"error": f"rleJson not found: {rle_json_arg}"}))
                return 1
            masks = _load_masks_from_rle(rle_path)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        return 1

    if not masks:
        print(json.dumps({"error": "no masks found"}))
        return 1

    warning = None
    result_format = fmt_norm

    if fmt_norm == "mp4-cutout":
        if not video_arg:
            warning = "mp4-cutout requires 'video'; falling back to png-sequence"
        else:
            try:
                video_path = _resolve_in_sandbox(root, video_arg)
                if not video_path.is_file():
                    raise ValueError(f"video not found: {video_arg}")
                out_path = _write_mp4_cutout(root, masks, video_path)
                print(json.dumps({
                    "outputPath": _to_sandbox_relative(root, out_path),
                    "frames": len(masks),
                    "format": "mp4-cutout",
                    "warning": "mp4-cutout composites the mask onto a black background; "
                               "true RGBA alpha is not preserved (best-effort, see "
                               "docs/fabri-contracts.md §3.5).",
                }))
                return 0
            except Exception as e:  # noqa: BLE001 — best-effort fallback, not a hard failure
                warning = f"mp4-cutout failed ({e}); fell back to png-sequence"
        result_format = "png-sequence"

    try:
        out_path = _write_matte_zip(root, masks)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"failed to write matte.zip: {e}"}))
        return 1

    out = {
        "outputPath": _to_sandbox_relative(root, out_path),
        "frames": len(masks),
        "format": result_format,
    }
    if warning:
        out["warning"] = warning
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
