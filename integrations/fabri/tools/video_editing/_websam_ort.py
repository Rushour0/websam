"""fabri video_editing shared ORT core: EdgeTAM video tracking, pure onnxruntime.

NOT a manifest'd tool — every vid_* script that touches the model imports this
module. Deliberately has **no** dependency on ``torch`` or ``transformers`` at
runtime (friction #1 in ``docs/fabri-contracts.md`` §7.1): preprocessing is
re-implemented in plain PIL/numpy and verified numerically against the HF
video processor (see ``tests/test_websam_ort.py::test_preprocess_matches_hf``
docstring below for the comparison methodology).

Reuse note vs. ``tools/export/spikes/m2-edgetam/e2e_loop.py``
---------------------------------------------------------------
``e2e_loop.py``'s ``iou()`` is copied verbatim below (not imported) —
``e2e_loop.py`` has HARD module-level ``import torch`` /
``from transformers import AutoProcessor`` (line ~73), so even
``from e2e_loop import iou`` transitively requires torch+transformers to be
importable, which fabri's runtime explicitly must not depend on (confirmed:
``uv run python3 -c "import e2e_loop"`` in this project's venv raises
``ModuleNotFoundError: No module named 'transformers'``). Per
``docs/fabri-contracts.md`` §7.1's approved fallback ("copy + comment
pointing back at the line range, never fork the math"), the 4-line pure-numpy
``iou()`` is copied here unchanged rather than imported. Everything else in
this module that touches the model graphs is genuinely new (see below), not
copied from ``e2e_loop.py``.

``e2e_loop.py``'s ``OrtEngine``/``MemoryBank`` are **not** reused as-is. They
were written against the graphs checked into
``tools/export/spikes/m2-edgetam/onnx/`` (the M2-spike export), which take a
single PRE-ASSEMBLED ``(1, 3648, 64)`` memory buffer with tpos already added
and an externally-computed additive attention bias. The graphs this task was
told to load — ``tools/goldens/models-cache/edgetam/*.onnx`` — are a *later*,
different export ("M2 wave-3 production", per that dir's ``manifest.json``
``toolchain.exporter``) with a reconciled, richer ``memory_attention`` I/O
contract that does the KV assembly IN-GRAPH instead: separate per-map
``memory_spatial``/``memory_spatial_pos`` tensors, ``tpos_indices`` (gathered
against a graph-internal tpos table — the checked-in
``activations/tpos_table.npy`` is NOT needed for this graph, only for the
spike's), a bool ``memory_mask`` instead of a float bias, and a raw
``(1, 16, 256)`` pointer bank instead of pre-split 64-d tokens. Confirmed by
introspecting both ``.onnx`` files' ``get_inputs()``/``get_outputs()`` (they
have different byte sizes and different input names/shapes) and by reading
the *actual* wrapper source that produced the production graphs,
``tools/export/src/websam_export/wrappers/edgetam.py``
(``EdgeTamMemoryAttentionWrapper``/``EdgeTamMemoryEncoderWrapper`` docstrings,
"INTERFACE RECONCILIATION" sections) — that file is the authoritative spec
for what ``tools/goldens/models-cache/edgetam/memory_attention.onnx`` and
``memory_encoder.onnx`` actually expect. ``memory_encoder.onnx`` likewise
takes the decoder's LOW-res ``mask_logits`` (256x256), not the spike's
upsampled ``high_res_masks`` (1024x1024) — the 256->1024 upsample now happens
in-graph inside ``memory_encoder.onnx`` itself.

The MemoryBank *bookkeeping rules* below (which frames go in `cond` vs
`recent`, insertion order, the tpos-row convention cond->row 6 / recent
offset k->row k-1, the ring eviction at depth 6, the object-pointer ordering
cond-then-recent-offsets-1..15) are IDENTICAL to ``e2e_loop.py``'s — that
choreography is reused, just re-expressed against the new tensor layout
required by the production graphs (contiguous-pack + boolean mask instead of
manual padding + float bias). See ``docs/fabri-contracts.md`` §7.1 for the
friction this was flagged under.
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort
from PIL import Image


def _repo_root() -> pathlib.Path:
    # integrations/fabri/tools/video_editing/_websam_ort.py -> repo root is 3 parents up
    return pathlib.Path(__file__).resolve().parents[3]


MODELS_DIR_DEFAULT = _repo_root() / "tools" / "goldens" / "models-cache" / "edgetam"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Copied verbatim from tools/export/spikes/m2-edgetam/e2e_loop.py's
    ``iou()`` (see module docstring for why this is a copy, not an import)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0

# ---------------------------------------------------------------------------
# Preprocessing constants (manifest.json's "preprocess" block / spec.py's
# EDGETAM_1024 tier: square-stretch resize, ImageNet mean/std).
# ---------------------------------------------------------------------------
INPUT_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Frozen memory-bank / KV constants, matching
# tools/export/src/websam_export/wrappers/edgetam.py and
# tools/goldens/models-cache/edgetam/manifest.json's "video" block.
# ---------------------------------------------------------------------------
NUM_RECENT = 6                 # num_maskmem - 1
TOKENS_PER_MAP = 512
MAX_MEMORY_MAPS = 7            # 1 cond + 6 recent (single-prompt session)
MEM_DIM = 64
MAX_POINTERS = 16
PTR_DIM = 256
KV_LEN = MAX_MEMORY_MAPS * TOKENS_PER_MAP + MAX_POINTERS * (PTR_DIM // MEM_DIM)  # 3648


class OrtEngine:
    """The five ONNX sessions for the production ("M2 wave-3") EdgeTAM export
    at ``tools/goldens/models-cache/edgetam``. fp32 graphs, CPUExecutionProvider
    — this is the same set of graphs the golden masks were captured against
    (``golden-video-meta.json``'s note), so no fp16 tolerance fudge is needed.
    """

    def __init__(self, models_dir: pathlib.Path | str = MODELS_DIR_DEFAULT):
        models_dir = pathlib.Path(models_dir)

        def load(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(models_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"]
            )

        self.models_dir = models_dir
        self.vision = load("vision_encoder")
        self.no_mem = load("no_mem_embed")
        self.mem_attn = load("memory_attention")
        self.decoder = load("mask_decoder_video")
        self.mem_enc = load("memory_encoder")

    # -- per-graph thin wrappers ------------------------------------------------

    def encode_frame(self, pixel_values: np.ndarray):
        feats, pos, hr0, hr1 = self.vision.run(None, {"pixel_values": pixel_values})
        return feats, pos, hr0, hr1

    def condition_no_memory(self, feats: np.ndarray) -> np.ndarray:
        return self.no_mem.run(None, {"vision_features": feats})[0]

    def condition_with_memory(
        self,
        feats: np.ndarray,
        feats_pos: np.ndarray,
        memory_spatial: np.ndarray,
        memory_spatial_pos: np.ndarray,
        tpos_indices: np.ndarray,
        memory_mask: np.ndarray,
        object_pointers: np.ndarray,
        pointer_deltas: np.ndarray,
        pointer_mask: np.ndarray,
    ) -> np.ndarray:
        return self.mem_attn.run(
            None,
            {
                "current_vision_features": feats,
                "current_vision_pos_embed": feats_pos,
                "memory_spatial": memory_spatial,
                "memory_spatial_pos": memory_spatial_pos,
                "tpos_indices": tpos_indices,
                "memory_mask": memory_mask,
                "object_pointers": object_pointers,
                "pointer_deltas": pointer_deltas,
                "pointer_mask": pointer_mask,
            },
        )[0]

    def decode(self, conditioned, hr0, hr1, coords, labels) -> dict[str, np.ndarray]:
        names = [
            "low_res_masks",
            "high_res_masks",
            "object_pointer",
            "object_score_logits",
            "iou_scores",
        ]
        out = self.decoder.run(
            None,
            {
                "conditioned_features": conditioned,
                "high_res_features_0": hr0,
                "high_res_features_1": hr1,
                "point_coords": coords,
                "point_labels": labels,
            },
        )
        return dict(zip(names, out))

    def encode_memory(self, feats: np.ndarray, low_res_masks: np.ndarray, is_prompted: bool):
        mem, mem_pos = self.mem_enc.run(
            None,
            {
                "vision_features": feats,
                "mask_logits": low_res_masks,
                "is_prompted": np.array([1.0 if is_prompted else 0.0], dtype=np.float32),
            },
        )
        return mem, mem_pos


class MemoryBank:
    """Slot bookkeeping for the production memory_attention/memory_encoder
    graphs. Choreography (which frame goes where, insertion order, tpos-row
    convention, ring eviction, pointer ordering) mirrors
    ``e2e_loop.py::MemoryBank`` exactly; tensor packing is adapted to the
    production graphs' separated, boolean-masked, contiguous-pack contract
    (see module docstring)."""

    def __init__(self) -> None:
        self.cond: dict[int, dict] = {}     # insertion-ordered (dict semantics)
        self.recent: dict[int, dict] = {}

    def store(
        self,
        frame_idx: int,
        memory_features: np.ndarray,   # (1, 512, 64)
        memory_pos_embed: np.ndarray,  # (1, 512, 64)
        object_pointer: np.ndarray,    # (1, 1, 256) or (256,)
        *,
        is_cond: bool,
    ) -> None:
        entry = {
            "mem": memory_features[0],          # (512, 64)
            "mem_pos": memory_pos_embed[0],      # (512, 64)
            "ptr": np.asarray(object_pointer).reshape(PTR_DIM),  # (256,)
        }
        if is_cond:
            self.cond[frame_idx] = entry
        else:
            self.recent[frame_idx] = entry
            # Eviction: offsets beyond NUM_RECENT are unreachable (mirrors
            # e2e_loop.py's MemoryBank.store).
            for old in [f for f in self.recent if f < frame_idx - NUM_RECENT]:
                del self.recent[old]

    def assemble(self, frame_idx: int):
        """Returns the 7 production-graph tensors for
        ``OrtEngine.condition_with_memory`` at ``frame_idx``."""
        # --- spatial maps: cond (tpos row = NUM_RECENT) then recent 6..1 ------
        maps: list[tuple[np.ndarray, np.ndarray, int]] = [
            (e["mem"], e["mem_pos"], NUM_RECENT) for e in self.cond.values()
        ]
        for offset in range(NUM_RECENT, 0, -1):  # oldest first
            prev = frame_idx - offset
            if prev in self.recent:
                maps.append((self.recent[prev]["mem"], self.recent[prev]["mem_pos"], offset - 1))
        assert len(maps) <= MAX_MEMORY_MAPS
        n_maps = len(maps)

        memory_spatial = np.zeros((1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM), dtype=np.float32)
        memory_spatial_pos = np.zeros((1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM), dtype=np.float32)
        tpos_indices = np.zeros((1, MAX_MEMORY_MAPS), dtype=np.int64)
        for i, (mem_map, mem_pos, tpos_row) in enumerate(maps):
            memory_spatial[0, i] = mem_map
            memory_spatial_pos[0, i] = mem_pos
            tpos_indices[0, i] = tpos_row
        # Padding slots: tpos row is irrelevant (masked out below); leave at 0.

        # --- object pointers: cond (past only) then tracked offsets 1..15 ---
        pointers = [e["ptr"] for f, e in self.cond.items() if f <= frame_idx]
        for d in range(1, MAX_POINTERS):
            ref = frame_idx - d
            if ref in self.recent:
                pointers.append(self.recent[ref]["ptr"])
        pointers = pointers[:MAX_POINTERS]
        n_ptrs = len(pointers)

        object_pointers = np.zeros((1, MAX_POINTERS, PTR_DIM), dtype=np.float32)
        for i, ptr in enumerate(pointers):
            object_pointers[0, i] = ptr
        pointer_deltas = np.zeros((1, MAX_POINTERS), dtype=np.int64)  # unused by this checkpoint
        pointer_mask = np.zeros((1, MAX_POINTERS), dtype=bool)
        pointer_mask[0, :n_ptrs] = True

        # --- boolean KV validity mask (contiguous-pack: maps/pointers are
        # packed front-first above, so validity is just a prefix per group) ---
        memory_mask = np.zeros((1, KV_LEN), dtype=bool)
        memory_mask[0, : n_maps * TOKENS_PER_MAP] = True
        ptr_base = MAX_MEMORY_MAPS * TOKENS_PER_MAP
        memory_mask[0, ptr_base : ptr_base + n_ptrs * (PTR_DIM // MEM_DIM)] = True

        return memory_spatial, memory_spatial_pos, tpos_indices, memory_mask, object_pointers, pointer_deltas, pointer_mask


# ---------------------------------------------------------------------------
# Preprocessing — plain PIL/numpy, no transformers. Verified numerically
# against ``AutoProcessor(...).video_processor`` (square-stretch resize +
# ImageNet normalize, resample=BILINEAR): max abs diff ~2e-7 (float32 noise)
# on the golden clip's frame 0. See tests/test_websam_ort.py.
# ---------------------------------------------------------------------------


def _to_pil(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def preprocess_frame(frame, size: int = INPUT_SIZE) -> np.ndarray:
    """Square-stretch resize to ``size`` x ``size`` + ImageNet mean/std
    normalize. Returns ``(1, 3, size, size)`` float32, NCHW — the exact
    ``pixel_values`` shape ``vision_encoder.onnx`` expects.

    ``frame`` may be a PIL Image or an (H, W, 3) uint8 ndarray (RGB).
    """
    img = _to_pil(frame).resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = arr.transpose(2, 0, 1).astype(np.float32)
    return chw[None, ...]


def rescale_prompt_xy(x: float, y: float, orig_w: int, orig_h: int, size: int = INPUT_SIZE):
    """Anisotropic square-stretch rescale of an original-pixel-space point
    into the model's ``size`` x ``size`` input space (mirrors
    ``e2e_loop.py``'s ``PROMPT_POINT_XY * 1024 / WIDTH`` convention)."""
    return x * size / orig_w, y * size / orig_h


# ---------------------------------------------------------------------------
# RLE — identical format to tools/goldens/make-video-golden.py's encode_rle:
# row-major scan (y*width + x), counts[] alternate run lengths starting with
# a run of ZEROS (counts[0] may be 0), sum(counts) == width * height.
# ---------------------------------------------------------------------------


def rle_encode(bin_mask: np.ndarray) -> dict:
    height, width = bin_mask.shape
    flat = bin_mask.reshape(-1).astype(np.uint8)
    # Vectorized run-length encode.
    change = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    run_lengths = np.diff(boundaries).tolist()
    counts: list[int] = []
    current = int(flat[0]) if flat.size else 0
    if current == 1:
        counts.append(0)  # must start with a (possibly zero-length) run of zeros
    counts.extend(run_lengths)
    return {"width": int(width), "height": int(height), "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    width, height = rle["width"], rle["height"]
    counts = rle["counts"]
    flat = np.zeros(width * height, dtype=np.uint8)
    pos = 0
    value = 0
    for c in counts:
        if value:
            flat[pos : pos + c] = 1
        pos += c
        value ^= 1
    return flat.reshape(height, width)


# ---------------------------------------------------------------------------
# Video decode.
# ---------------------------------------------------------------------------


def decode_video(path) -> list[np.ndarray]:
    """Decode an mp4 (or any imageio-ffmpeg-supported container) into a list
    of RGB (H, W, 3) uint8 ndarrays, in presentation order."""
    import imageio.v3 as iio

    frames = []
    for frame in iio.imiter(str(path)):
        frames.append(np.asarray(frame))
    return frames


# ---------------------------------------------------------------------------
# The tracking loop (reused choreography from e2e_loop.py's main(), lifted
# out into a callable per docs/fabri-contracts.md §4/§7.1 — the per-frame
# loop body there is what this function's structure is based on, adapted to
# the production graph's tensor contract, see module docstring).
# ---------------------------------------------------------------------------


def _mask_to_original(high_res_masks: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """high_res_masks: (1, 1, 1024, 1024) float logits (already upsampled to
    1024x1024 in-graph by mask_decoder_video.onnx). Downsample/upsample to
    the frame's original resolution via bilinear, then threshold > 0
    (mirrors Sam2VideoProcessor.post_process_masks(binarize=True); the
    square-stretch preprocessing has no aspect-preserving pad to crop, so
    that step of post_process_masks is a no-op here)."""
    logits = high_res_masks[0, 0].astype(np.float32)
    if (orig_h, orig_w) != logits.shape:
        img = Image.fromarray(logits, mode="F").resize((orig_w, orig_h), resample=Image.BILINEAR)
        logits = np.asarray(img, dtype=np.float32)
    return (logits > 0.0).astype(np.uint8)


def _prompt_tensors_1024(prompt: dict, orig_w: int, orig_h: int):
    """Build (point_coords, point_labels) at the prompt frame in 1024-space,
    from a {"points": [{x,y,label}]} or {"box": {x0,y0,x1,y1}} prompt dict."""
    if "points" in prompt and prompt["points"]:
        pts = prompt["points"]
        coords = np.zeros((1, 1, len(pts), 2), dtype=np.float32)
        labels = np.zeros((1, 1, len(pts)), dtype=np.int64)
        for i, p in enumerate(pts):
            px, py = rescale_prompt_xy(p["x"], p["y"], orig_w, orig_h)
            coords[0, 0, i] = (px, py)
            labels[0, 0, i] = int(p["label"])
        return coords, labels
    if "point" in prompt:
        p = prompt["point"]
        px, py = rescale_prompt_xy(p["x"], p["y"], orig_w, orig_h)
        coords = np.array([[[[px, py]]]], dtype=np.float32)
        labels = np.array([[[int(p.get("label", 1))]]], dtype=np.int64)
        return coords, labels
    if "box" in prompt:
        b = prompt["box"]
        x0, y0 = rescale_prompt_xy(b["x0"], b["y0"], orig_w, orig_h)
        x1, y1 = rescale_prompt_xy(b["x1"], b["y1"], orig_w, orig_h)
        coords = np.array([[[[x0, y0], [x1, y1]]]], dtype=np.float32)
        labels = np.array([[[2, 3]]], dtype=np.int64)  # SAM box-corner labels
        return coords, labels
    raise ValueError("prompt must have 'points', 'point', or 'box'")


_TRACK_COORDS = np.zeros((1, 1, 1, 2), dtype=np.float32)
_TRACK_LABELS = -np.ones((1, 1, 1), dtype=np.int64)


def _frame_hw(frame) -> tuple[int, int]:
    if isinstance(frame, np.ndarray):
        return frame.shape[0], frame.shape[1]
    w, h = frame.size  # PIL Image
    return h, w


def run_track(engine: OrtEngine, frames: list, prompt: dict, progress_cb=None) -> list[np.ndarray]:
    """Track an object across ``frames`` starting from ``prompt`` at
    ``prompt["frameIndex"]`` (default 0). Returns one ``(H, W)`` uint8 {0,1}
    mask per frame, at each frame's original resolution.

    ``prompt``: ``{"frameIndex": int, "points": [{x,y,label}]}`` or
    ``{"frameIndex": int, "point": {x,y,label}}`` or
    ``{"frameIndex": int, "box": {x0,y0,x1,y1}}`` — original pixel coords.
    """
    prompt_frame_idx = prompt.get("frameIndex", 0)
    num_frames = len(frames)
    orig_h, orig_w = _frame_hw(frames[prompt_frame_idx])

    click_coords, click_labels = _prompt_tensors_1024(prompt, orig_w, orig_h)

    bank = MemoryBank()
    masks: list[np.ndarray] = []

    for t in range(num_frames):
        frame_h, frame_w = _frame_hw(frames[t])
        pixel_values = preprocess_frame(frames[t])
        feats, feats_pos, hr0, hr1 = engine.encode_frame(pixel_values)

        if t == prompt_frame_idx:
            conditioned = engine.condition_no_memory(feats)
            dec = engine.decode(conditioned, hr0, hr1, click_coords, click_labels)
            mem, mem_pos = engine.encode_memory(feats, dec["low_res_masks"], is_prompted=True)
            bank.store(t, mem, mem_pos, dec["object_pointer"], is_cond=True)
        else:
            (memory_spatial, memory_spatial_pos, tpos_indices, memory_mask,
             object_pointers, pointer_deltas, pointer_mask) = bank.assemble(t)
            conditioned = engine.condition_with_memory(
                feats, feats_pos, memory_spatial, memory_spatial_pos, tpos_indices,
                memory_mask, object_pointers, pointer_deltas, pointer_mask,
            )
            dec = engine.decode(conditioned, hr0, hr1, _TRACK_COORDS, _TRACK_LABELS)
            mem, mem_pos = engine.encode_memory(feats, dec["low_res_masks"], is_prompted=False)
            bank.store(t, mem, mem_pos, dec["object_pointer"], is_cond=False)

        masks.append(_mask_to_original(dec["high_res_masks"], frame_h, frame_w))
        if progress_cb is not None:
            progress_cb(t, num_frames)

    return masks


def segment_frame(engine: OrtEngine, frame, prompt: dict) -> np.ndarray:
    """Single-frame segmentation: no memory bank needed (mirrors
    ``docs/fabri-contracts.md`` §3.3's ``vid_segment`` path)."""
    arr = frame if isinstance(frame, np.ndarray) else np.asarray(_to_pil(frame))
    orig_h, orig_w = arr.shape[:2]
    coords, labels = _prompt_tensors_1024(prompt, orig_w, orig_h)

    pixel_values = preprocess_frame(frame)
    feats, feats_pos, hr0, hr1 = engine.encode_frame(pixel_values)
    conditioned = engine.condition_no_memory(feats)
    dec = engine.decode(conditioned, hr0, hr1, coords, labels)
    return _mask_to_original(dec["high_res_masks"], orig_h, orig_w)


def load_engine(models_dir: pathlib.Path | str = MODELS_DIR_DEFAULT) -> OrtEngine:
    return OrtEngine(models_dir)
