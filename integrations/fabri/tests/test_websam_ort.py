"""Wave-1 gate: the pure-ORT EdgeTAM tracking loop reproduces the golden
per-frame masks (>= 0.90 IoU on every frame) against
tools/goldens/models-cache/edgetam's production graphs.

Also spot-checks preprocess_frame against the golden fixture metadata's
documented preprocess block (square-stretch, 1024, ImageNet mean/std) via a
shape/range sanity check (no transformers/torch dependency is available in
this project's venv, so the numeric cross-check against the real
AutoProcessor was done once, out-of-band, in tools/export's venv — see
_websam_ort.py's module docstring for the recorded max-abs-diff result).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tools" / "video_editing"))

import _websam_ort as ort_core  # noqa: E402

_REPO_ROOT = _HERE.parents[2]
_FIXTURES = _REPO_ROOT / "tools" / "goldens" / "fixtures" / "video"
_MODELS_DIR = _REPO_ROOT / "tools" / "goldens" / "models-cache" / "edgetam"

IOU_GATE = 0.90


def _load_meta() -> dict:
    return json.loads((_FIXTURES / "golden-video-meta.json").read_text())


@pytest.fixture(scope="module")
def engine():
    return ort_core.load_engine(_MODELS_DIR)


@pytest.fixture(scope="module")
def frames():
    return ort_core.decode_video(_FIXTURES / "clip-256.mp4")


def _golden_masks(meta: dict) -> list[np.ndarray]:
    out = []
    for name in meta["masks"]:
        rle = json.loads((_FIXTURES / name).read_text())
        out.append(ort_core.rle_decode(rle))
    return out


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = (rng.random((32, 40)) > 0.5).astype(np.uint8)
    rle = ort_core.rle_encode(mask)
    back = ort_core.rle_decode(rle)
    np.testing.assert_array_equal(mask, back)


def test_rle_decode_matches_golden_fixture_shape():
    meta = _load_meta()
    masks = _golden_masks(meta)
    assert len(masks) == meta["clip"]["numFrames"]
    for m in masks:
        assert m.shape == (meta["clip"]["height"], meta["clip"]["width"])


def test_preprocess_frame_shape_and_range():
    meta = _load_meta()
    frame = np.zeros((meta["clip"]["height"], meta["clip"]["width"], 3), dtype=np.uint8)
    out = ort_core.preprocess_frame(frame)
    assert out.shape == (1, 3, 1024, 1024)
    assert out.dtype == np.float32
    # all-black input after ImageNet normalize should sit at -mean/std per channel
    expected = (-ort_core.IMAGENET_MEAN / ort_core.IMAGENET_STD).astype(np.float32)
    np.testing.assert_allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_run_track_matches_golden_iou(engine, frames):
    meta = _load_meta()
    prompt = meta["prompt"]
    assert prompt["type"] == "point"

    golden = _golden_masks(meta)
    assert len(frames) == len(golden) == meta["clip"]["numFrames"]

    track_prompt = {
        "frameIndex": prompt["frameIndex"],
        "point": {"x": prompt["x"], "y": prompt["y"], "label": prompt["label"]},
    }
    ours = ort_core.run_track(engine, frames, track_prompt)

    assert len(ours) == len(golden)
    per_frame_iou = []
    for t, (m_ours, m_gold) in enumerate(zip(ours, golden)):
        v = ort_core.iou(m_ours.astype(bool), m_gold.astype(bool))
        per_frame_iou.append(v)
        print(f"frame {t}: IoU = {v:.4f}")

    worst = min(per_frame_iou)
    print(f"worst-frame IoU = {worst:.4f} (gate >= {IOU_GATE})")
    for t, v in enumerate(per_frame_iou):
        assert v >= IOU_GATE, f"frame {t} IoU {v:.4f} < gate {IOU_GATE}"


def test_segment_frame_matches_golden_frame0(engine, frames):
    meta = _load_meta()
    prompt = meta["prompt"]
    golden0 = ort_core.rle_decode(
        json.loads((_FIXTURES / meta["masks"][0]).read_text())
    )
    seg_prompt = {"point": {"x": prompt["x"], "y": prompt["y"], "label": prompt["label"]}}
    mask0 = ort_core.segment_frame(engine, frames[0], seg_prompt)
    v = ort_core.iou(mask0.astype(bool), golden0.astype(bool))
    print(f"segment_frame frame0 IoU = {v:.4f}")
    assert v >= IOU_GATE
