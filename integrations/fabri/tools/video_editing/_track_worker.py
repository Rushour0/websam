"""_track_worker — the detached background worker for vid_track (fabri
contract §3.4). NOT a manifest'd tool; invoked directly by `vid_track.py` as
a fully detached subprocess (`start_new_session=True`), never by fabri's
runner. Owns the entire lifecycle of one job's `status.json` after the
initial "running" write in `vid_track.py`.

Usage: python3 _track_worker.py <sandbox_root> <job_id>

Reads `<sandbox_root>/artifacts/jobs/<job_id>/args.json` (written by
vid_track.py), runs the real per-frame tracking loop via
`_websam_ort.run_track` (imported, not reimplemented — see repo-level
instructions), and on completion writes:
  - `<job_dir>/masks/frame-%04d.png` (one per frame, white=object, mode "L")
  - `<job_dir>/masks.rle.json` (list of per-frame COCO-RLE dicts, same
    {width,height,counts} shape as tools/goldens/fixtures/video/golden-mask-f*.rle.json)
  - `<job_dir>/status.json` flipped to
    {"status": "done", "progress": 1.0, "maskDir": <sandbox-relative>,
     "maskRlePath": <sandbox-relative>, "frameCount": N}
    or, on any failure, {"status": "error", "error": "..."}.

`status.json` is updated after every frame while running
({"status": "running", "progress": t/numFrames}) so vid_poll_job.py never
blocks — it only ever reads whatever is currently on disk.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _websam_ort as ort_lib  # noqa: E402

# WORKAROUND (flagged, not fixed here — _websam_ort.py is not owned by this
# tool per repo-level instructions): _websam_ort._repo_root() is off by one
# parent (`parents[3]` resolves to `.../integrations`, not the repo root),
# so its module-level `MODELS_DIR_DEFAULT` points at a nonexistent
# `integrations/tools/goldens/...`. Recompute the correct repo root from
# this file's own location (same directory as _websam_ort.py) and pass it
# explicitly to load_engine() rather than relying on the broken default.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODELS_DIR = _REPO_ROOT / "tools" / "goldens" / "models-cache" / "edgetam"


def _write_status(status_path: Path, payload: dict) -> None:
    # Write-then-rename for atomicity: a concurrent vid_poll_job.py read must
    # never observe a partially-written JSON file.
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(status_path)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _track_worker.py <sandbox_root> <job_id>", file=sys.stderr)
        return 1

    root = Path(sys.argv[1]).resolve()
    job_id = sys.argv[2]
    job_dir = root / "artifacts" / "jobs" / job_id
    status_path = job_dir / "status.json"

    try:
        job_args = json.loads((job_dir / "args.json").read_text())
        video_path = Path(job_args["videoPath"])
        prompt_frame_sec = float(job_args["promptFrameSec"])
        point = job_args.get("point")
        box = job_args.get("box")

        import imageio.v3 as iio

        meta = iio.immeta(str(video_path))
        fps = float(meta.get("fps") or 10.0)
        frame_index = round(prompt_frame_sec * fps)

        frames = ort_lib.decode_video(video_path)
        if not frames:
            raise RuntimeError(f"decoded zero frames from {video_path}")
        frame_index = max(0, min(frame_index, len(frames) - 1))

        prompt: dict = {"frameIndex": frame_index}
        if point is not None:
            prompt["point"] = {"x": point["x"], "y": point["y"], "label": point.get("label", 1)}
        else:
            prompt["box"] = box

        engine = ort_lib.load_engine(_MODELS_DIR)

        num_frames = len(frames)

        def progress_cb(t: int, total: int) -> None:
            _write_status(status_path, {"status": "running", "progress": (t + 1) / total})

        masks = ort_lib.run_track(engine, frames, prompt, progress_cb=progress_cb)

        masks_dir = job_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        rle_list = []
        from PIL import Image

        for i, mask in enumerate(masks):
            png_path = masks_dir / f"frame-{i:04d}.png"
            Image.fromarray((mask * 255).astype("uint8"), mode="L").save(png_path)
            rle_list.append(ort_lib.rle_encode(mask))

        rle_path = job_dir / "masks.rle.json"
        rle_path.write_text(json.dumps(rle_list))

        _write_status(
            status_path,
            {
                "status": "done",
                "progress": 1.0,
                "maskDir": str(masks_dir.relative_to(root)),
                "maskRlePath": str(rle_path.relative_to(root)),
                "frameCount": num_frames,
            },
        )
        return 0
    except Exception as e:  # noqa: BLE001 — a worker crash must still flip status.json
        _write_status(
            status_path,
            {"status": "error", "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"},
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
