"""Golden fixture generator for the websam M2 video browser gate.

Mirrors `tools/goldens/generate.mjs` (M1 image gate) but for the EdgeTAM
video path: a deterministic short synthetic clip + one point prompt on
frame 0, run end-to-end through HF `EdgeTamVideoModel` (the same authority
`tools/export/spikes/m2-edgetam/e2e_loop.py` / the production parity gate
compare against), producing per-frame golden mask RLEs the browser pipeline
must match at IoU >= 0.90 per frame.

Deterministic: no randomness anywhere (a fixed synthetic clip drawn by PIL,
a fixed prompt, a fixed HF checkpoint revision). Re-running this script
byte-for-byte reproduces `fixtures/video/golden-video-meta.json`'s
`masks_sha256` list.

Usage (from `tools/goldens/`):
    ../export/.venv/bin/python make-video-golden.py

Requires the `tools/export` venv (`cd tools/export && uv sync --extra export`)
and `ffmpeg` on PATH (clip muxing only — no ffmpeg-python binding).

Outputs (committed, small):
    fixtures/video/clip-256.mp4                 10 frames, 256x256, H.264
    fixtures/video/golden-mask-f{t}.rle.json     per-frame RLE (t = 0..9)
    fixtures/video/golden-video-meta.json        model rev, prompt, preprocessing

Weights (gitignored, produced by `tools/export/src/websam_export/export_edgetam.py`):
    models-cache/edgetam/*.onnx                  fp16, self-contained (no external data)
    models-cache/edgetam/manifest.json           tier 'edgetam' (websam_export.manifest_edgetam)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, EdgeTamVideoModel

HERE = pathlib.Path(__file__).parent
FIXTURES = HERE / "fixtures" / "video"
MODELS_CACHE = HERE / "models-cache" / "edgetam"
EXPORT_SRC = HERE.parent / "export" / "src"

sys.path.insert(0, str(EXPORT_SRC))
from websam_export.export_edgetam import export_all  # noqa: E402
from websam_export import manifest_edgetam as me  # noqa: E402

MODEL_ID = "yonigozlan/EdgeTAM-hf"

# ---------------------------------------------------------------------------
# Deterministic synthetic clip: a red disc sliding right over a static
# gradient background with a gray distractor square. 256x256 (square, unlike
# the export spike's 640x480 clip — deliberately exercises the square-stretch
# preprocessing's DEGENERATE case, width==height, so any residual anisotropic-
# scale bug in the browser's `sourceToModel`/`computeTransform` path would
# have to show up on the M1 non-square golden instead; kept here for a
# smaller, faster-loading browser-gate fixture).
# ---------------------------------------------------------------------------

NUM_FRAMES = 10
SIZE = 256
BALL_R = 28
BALL_Y = 128
BALL_X0 = 60
BALL_DX = 14  # ball center x at frame t: BALL_X0 + BALL_DX * t
PROMPT_FRAME = 0
PROMPT_POINT_XY = (BALL_X0, BALL_Y)
IOU_GATE = 0.90  # browser-gate threshold (looser than the Python e2e gate: WebGPU fp16 + resampling add drift)


def make_frame(t: int) -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(img)
    for y in range(0, SIZE, 4):
        g = 40 + int(120 * y / SIZE)
        d.rectangle([0, y, SIZE, y + 4], fill=(g // 2, g, 160))
    d.rectangle([190, 20, 230, 60], fill=(120, 120, 120))  # static distractor
    cx = BALL_X0 + BALL_DX * t
    d.ellipse([cx - BALL_R, BALL_Y - BALL_R, cx + BALL_R, BALL_Y + BALL_R],
              fill=(220, 60, 50), outline=(255, 200, 180), width=3)
    return img


def make_clip() -> list[Image.Image]:
    return [make_frame(t) for t in range(NUM_FRAMES)]


# ---------------------------------------------------------------------------
# RLE — identical format to generate.mjs's encodeRLE/decodeRLE: row-major
# scan (y*width + x), counts[] alternate run lengths starting with a run of
# ZEROS (counts[0] may be 0), sum(counts) == width * height.
# ---------------------------------------------------------------------------

def encode_rle(bin_mask: np.ndarray) -> dict:
    height, width = bin_mask.shape
    flat = bin_mask.reshape(-1).astype(np.uint8)
    counts: list[int] = []
    current = 0
    run = 0
    for v in flat:
        v = int(v)
        if v == current:
            run += 1
        else:
            counts.append(run)
            current = v
            run = 1
    counts.append(run)
    return {"width": int(width), "height": int(height), "counts": counts}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mux_mp4(frames: list[Image.Image], out_path: pathlib.Path) -> None:
    """PNG-sequence -> H.264 mp4 via ffmpeg (deterministic: `-fps_mode cfr`,
    fixed `-g`, no b-frames, yuv420p for universal browser `<video>` decode)."""
    frame_dir = out_path.parent / "_frames_tmp"
    frame_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        f.save(frame_dir / f"f{i:03d}.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", "10",
        "-i", str(frame_dir / "f%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1", "-bf", "0",
        "-fps_mode", "cfr", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    for f in frame_dir.glob("*.png"):
        f.unlink()
    frame_dir.rmdir()


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    frames = make_clip()

    print("1/3 muxing clip.mp4 ...")
    clip_path = FIXTURES / "clip-256.mp4"
    mux_mp4(frames, clip_path)

    print("2/3 running HF EdgeTamVideoModel end-to-end (golden reference) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = EdgeTamVideoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    session = processor.init_video_session(video=frames, inference_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        session, frame_idx=PROMPT_FRAME, obj_ids=1,
        input_points=[[[list(PROMPT_POINT_XY)]]], input_labels=[[[1]]],
    )

    def to_orig(pred_masks):
        m = processor.post_process_masks([pred_masks], original_sizes=[[SIZE, SIZE]], binarize=True)[0]
        return m[0, 0].cpu().numpy().astype(np.uint8)

    masks: list[np.ndarray] = []
    with torch.no_grad():
        out0 = model(session, frame_idx=PROMPT_FRAME)
        masks.append(to_orig(out0.pred_masks))
        for t in range(PROMPT_FRAME + 1, session.num_frames):
            out = model(session, frame_idx=t)
            masks.append(to_orig(out.pred_masks))

    rle_files = []
    masks_sha256 = []
    for t, m in enumerate(masks):
        rle = encode_rle(m)
        rle_bytes = json.dumps(rle).encode()
        name = f"golden-mask-f{t}.rle.json"
        (FIXTURES / name).write_bytes(rle_bytes)
        rle_files.append(name)
        masks_sha256.append(sha256_bytes(rle_bytes))
        print(f"  frame {t}: area={int(m.sum())} sha256={masks_sha256[-1][:12]}...")

    print("3/3 exporting production ONNX graphs + manifest ...")
    export_all(MODELS_CACHE)
    manifest = me.build_manifest(
        MODELS_CACHE,
        toolchain={"pytorch": torch.__version__, "onnx": "see files.*.sha256"},
    )
    (MODELS_CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    meta = {
        "model_id": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", None) or "unpinned (see toolchain versions)",
        "dtype": "float32 (golden reference); manifest ships fp16",
        "clip": {
            "path": clip_path.name,
            "numFrames": NUM_FRAMES,
            "width": SIZE,
            "height": SIZE,
            "fps": 10,
        },
        "prompt": {
            "frameIndex": PROMPT_FRAME,
            "type": "point",
            "x": PROMPT_POINT_XY[0],
            "y": PROMPT_POINT_XY[1],
            "label": 1,
            "note": "source-pixel coords (256-space); the browser rescales via computeTransform, mode 'square-stretch'",
        },
        "preprocess": {
            "mode": "square-stretch",
            "inputSize": 1024,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "iouGate": IOU_GATE,
        "masks": rle_files,
        "masksSha256": masks_sha256,
        "toolchain": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
        },
        "note": (
            "Golden masks are HF EdgeTamVideoModel fp32 PyTorch output (the "
            "SAME authority tools/export's parity tests compare the exported "
            "ONNX graphs against). The browser pipeline (fp16, WebGPU/wasm) "
            "must match each frame at IoU >= iouGate."
        ),
    }
    (FIXTURES / "golden-video-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {len(masks)} golden masks + clip + meta under {FIXTURES}")
    print(f"wrote weights + manifest under {MODELS_CACHE}")


if __name__ == "__main__":
    main()
