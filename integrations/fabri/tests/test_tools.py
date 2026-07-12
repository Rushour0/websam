"""fabri video_editing tool-script test suite (docs/fabri-contracts.md §6).

Runs WITHOUT installing fabri and WITHOUT hitting a live LLM. Every test
invokes a tool's manifest'd `command` via subprocess exactly the way fabri's
runner does: one JSON object on stdin, one JSON object off stdout, exit code
checked. `FABRI_SANDBOX_ROOT` is set to a fresh pytest tmp_path sandbox for
every call (see conftest.py::sandbox), and the golden clip is copied into
that sandbox first since all tool paths are sandbox-jailed.

Run: `cd integrations/fabri && uv sync && uv run --group dev pytest -q`
"""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest

from conftest import GOLDEN_CLIP, IOU_GATE, ort_core, run_tool

# ---------------------------------------------------------------------------
# vid_extract_frame
# ---------------------------------------------------------------------------


def test_vid_extract_frame_by_time(sandbox):
    rc, out, err = run_tool(
        "vid_extract_frame", {"video": "clip-256.mp4", "timeSec": 0.3}, sandbox
    )
    assert rc == 0, err
    assert out["width"] == 256
    assert out["height"] == 256
    # clip is 10fps -> timeSec=0.3 rounds to frame index 3
    assert out["frameIndex"] == 3
    frame_path = sandbox / out["framePath"]
    assert frame_path.is_file()

    from PIL import Image

    with Image.open(frame_path) as im:
        assert im.size == (256, 256)


def test_vid_extract_frame_by_index(sandbox):
    rc, out, err = run_tool(
        "vid_extract_frame", {"video": "clip-256.mp4", "frameIndex": 0}, sandbox
    )
    assert rc == 0, err
    assert out["frameIndex"] == 0
    assert (sandbox / out["framePath"]).is_file()


# ---------------------------------------------------------------------------
# vid_ground_text — deterministic stub path only (no key / no network)
# ---------------------------------------------------------------------------


def test_vid_ground_text_stub_deterministic(sandbox):
    fixed_box = [10, 20, 110, 220]
    extra_env = {"WEBSAM_GROUND_TEXT_STUB": json.dumps({"box": fixed_box})}
    for _ in range(2):  # two calls must return the exact same box -> deterministic
        rc, out, err = run_tool(
            "vid_ground_text",
            {"frame": "clip-256.mp4", "phrase": "the red ball"},
            sandbox,
            extra_env=extra_env,
        )
        assert rc == 0, err
        assert out["chosen"] == {"box": [float(v) for v in fixed_box]}
        assert out["boxes"] == [[float(v) for v in fixed_box]]


def test_vid_ground_text_real_gemini_requires_key():
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("no GEMINI_API_KEY")
    pytest.skip("real-Gemini path not exercised in the default gate")


# ---------------------------------------------------------------------------
# vid_segment — single-frame IoU gate
# ---------------------------------------------------------------------------


def test_vid_segment_frame0_iou(sandbox, golden_meta, golden_masks):
    prompt = golden_meta["prompt"]
    assert prompt["frameIndex"] == 0

    rc, out, err = run_tool(
        "vid_segment",
        {
            "video": "clip-256.mp4",
            "frameIndex": 0,
            "prompt": {"point": {"x": prompt["x"], "y": prompt["y"], "label": prompt["label"]}},
        },
        sandbox,
        timeout=120,
    )
    assert rc == 0, err
    mask_path = sandbox / out["maskPath"]
    assert mask_path.is_file()
    assert out["width"] == golden_meta["clip"]["width"]
    assert out["height"] == golden_meta["clip"]["height"]

    from PIL import Image

    ours = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(bool)
    golden0 = golden_masks[0].astype(bool)
    v = ort_core.iou(ours, golden0)
    print(f"vid_segment frame0 IoU = {v:.4f}")
    assert v >= IOU_GATE, f"vid_segment frame0 IoU {v:.4f} < gate {IOU_GATE}"


# ---------------------------------------------------------------------------
# vid_track + vid_poll_job — submit -> poll -> per-frame IoU gate
# ---------------------------------------------------------------------------


def _poll_until_done(sandbox, job_id: str, timeout_s: float = 180.0, interval_s: float = 1.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        rc, out, err = run_tool("vid_poll_job", {"jobId": job_id}, sandbox)
        assert rc == 0, err
        last = out
        if out.get("status") in ("done", "error"):
            return out
        time.sleep(interval_s)
    raise AssertionError(f"vid_track job {job_id} did not finish within {timeout_s}s; last status={last}")


@pytest.fixture(scope="module")
def tracked_job(tmp_path_factory, golden_meta):
    """Runs vid_track -> polls to completion once per test module (the ORT
    tracking loop is the expensive part of this suite; reused by the export
    test below rather than re-run)."""
    import shutil

    root = tmp_path_factory.mktemp("track_sandbox") / "sandbox"
    root.mkdir()
    shutil.copy(GOLDEN_CLIP, root / "clip-256.mp4")

    prompt = golden_meta["prompt"]
    rc, out, err = run_tool(
        "vid_track",
        {
            "videoPath": "clip-256.mp4",
            "promptFrameSec": prompt["frameIndex"] / golden_meta["clip"]["fps"],
            "point": {"x": prompt["x"], "y": prompt["y"], "label": prompt["label"]},
        },
        root,
    )
    assert rc == 0, err
    assert out["status"] == "running"
    job_id = out["jobId"]

    final = _poll_until_done(root, job_id)
    assert final["status"] == "done", final
    return root, final


def test_vid_track_iou_per_frame(tracked_job, golden_masks):
    root, final = tracked_job
    assert final["frameCount"] == len(golden_masks)

    rle_path = root / final["maskRlePath"]
    rle_list = json.loads(rle_path.read_text())
    assert len(rle_list) == len(golden_masks)

    per_frame_iou = []
    for t, (rle, golden) in enumerate(zip(rle_list, golden_masks)):
        ours = ort_core.rle_decode(rle).astype(bool)
        v = ort_core.iou(ours, golden.astype(bool))
        per_frame_iou.append(v)
        print(f"vid_track frame {t}: IoU = {v:.4f}")

    worst = min(per_frame_iou)
    print(f"vid_track worst-frame IoU = {worst:.4f} (gate >= {IOU_GATE})")
    for t, v in enumerate(per_frame_iou):
        assert v >= IOU_GATE, f"vid_track frame {t} IoU {v:.4f} < gate {IOU_GATE}"


# ---------------------------------------------------------------------------
# vid_export_matte — package the tracked job's masks into matte.zip
# ---------------------------------------------------------------------------


def test_vid_export_matte_png_sequence(tracked_job):
    root, final = tracked_job
    rc, out, err = run_tool(
        "vid_export_matte",
        {"masksDir": final["maskDir"], "format": "png-sequence"},
        root,
    )
    assert rc == 0, err
    out_path = root / out["outputPath"]
    assert out_path.is_file()
    assert zipfile.is_zipfile(out_path)

    with zipfile.ZipFile(out_path) as zf:
        names = sorted(zf.namelist())
        assert len(names) == final["frameCount"]
        for name in names:
            assert name.lower().endswith(".png")
            data = zf.read(name)
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            assert img.format == "PNG"
    assert out["frames"] == final["frameCount"]
    assert out["format"] == "png-sequence"


def test_vid_export_matte_mp4_cutout_best_effort_or_fallback(tracked_job):
    root, final = tracked_job
    rc, out, err = run_tool(
        "vid_export_matte",
        {
            "masksDir": final["maskDir"],
            "video": "clip-256.mp4",
            "format": "mp4-cutout",
        },
        root,
        timeout=60,
    )
    assert rc == 0, err
    out_path = root / out["outputPath"]
    assert out_path.is_file()
    # Never a hard failure: either a real mp4-cutout (with its documented
    # best-effort warning) or a matte_zip fallback (with its own warning).
    assert out["format"] in ("mp4-cutout", "png-sequence")
    if out["format"] == "mp4-cutout":
        import imageio.v3 as iio

        frames = list(iio.imiter(out_path))
        assert len(frames) > 0
    else:
        assert zipfile.is_zipfile(out_path)


# ---------------------------------------------------------------------------
# vid_trim / vid_concat / vid_composite — ffmpeg-wrapper smoke tests
# ---------------------------------------------------------------------------


def test_vid_trim_smoke(sandbox):
    rc, out, err = run_tool(
        "vid_trim", {"video": "clip-256.mp4", "startSec": 0.0, "endSec": 0.5}, sandbox
    )
    assert rc == 0, err
    out_path = sandbox / out["outputPath"]
    assert out_path.is_file()

    import imageio.v3 as iio

    frames = list(iio.imiter(out_path))
    assert len(frames) > 0
    assert out["durationSec"] < 1.0


def test_vid_concat_longer_than_trim(sandbox):
    rc_trim, trim_out, err = run_tool(
        "vid_trim", {"video": "clip-256.mp4", "startSec": 0.0, "endSec": 0.5}, sandbox
    )
    assert rc_trim == 0, err

    rc, out, err = run_tool(
        "vid_concat", {"videos": ["clip-256.mp4", trim_out["outputPath"]]}, sandbox
    )
    assert rc == 0, err
    out_path = sandbox / out["outputPath"]
    assert out_path.is_file()

    import imageio.v3 as iio

    frames = list(iio.imiter(out_path))
    assert len(frames) > 0
    assert out["durationSec"] > trim_out["durationSec"]


def test_vid_composite_cutout_smoke(tracked_job):
    root, final = tracked_job
    rc, out, err = run_tool(
        "vid_composite",
        {"video": "clip-256.mp4", "masksDir": final["maskDir"], "mode": "cutout"},
        root,
    )
    assert rc == 0, err
    out_path = root / out["outputPath"]
    assert out_path.is_file()

    import imageio.v3 as iio

    frames = list(iio.imiter(out_path))
    assert len(frames) > 0
