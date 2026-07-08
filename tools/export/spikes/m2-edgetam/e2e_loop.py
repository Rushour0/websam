"""Pure-ONNX-Runtime EdgeTAM video tracking loop — the EXECUTABLE SPEC for the
websam JS video engine.

Every inference op runs in ORT on the five exported graphs; PyTorch is not
imported. The HF video processor is used ONLY for preprocessing (frame resize
+ normalize + prompt rescale), which the JS engine replicates separately
(bilinear-antialias square-stretch to 1024x1024, ImageNet mean/std — see
FINDINGS.md "Preprocessing contract").

Success gate: per-frame mask IoU >= 0.95 vs the HF PyTorch end-to-end golden
(activations/golden.npz, produced by capture_golden.py) at the original video
resolution.

--------------------------------------------------------------------------
Memory-bank bookkeeping (mirrors modeling_edgetam_video.py, transformers
v5.13.0; line numbers cited). This is the contract `memory-bank.ts` must
replicate:

STATE per tracked object:
  cond:    {frame_idx: BankEntry}   conditioning frames (user-prompted),
                                    insertion-ordered, never evicted
  recent:  {frame_idx: BankEntry}   tracked frames; only the last
                                    NUM_RECENT(=6) offsets are ever read, so a
                                    ring of 6 suffices
  BankEntry = {memory_features (1,512,64), object_pointer (1,1,256)}
  (maskmem_pos_enc is frame-independent — cache one copy.)

PER FRAME t (tracked, i.e. not the first prompted frame):
  1. spatial memory maps, ordered  [L2589-2634]:
       a. all cond entries, insertion order, each at temporal_offset 0
          (max_cond_frame_num = -1 -> no cond selection/capping)  [L2610-2615]
       b. for offset k = 6,5,...,1 (oldest first): entry for frame t-k from
          `recent` — SKIPPED if that frame is a cond frame or missing
          [L2620-2632]
  2. per-map positional embedding = maskmem_pos_enc + tpos[k-1] where k is
     the temporal offset (cond k=0 wraps to tpos[-1] = tpos[6])  [L2663-2667]
     -> cond maps get tpos row 6; recent offset k gets row k-1.
  3. object pointers, ordered  [L2672-2731]:
       a. pointers of cond frames with frame_idx <= t, insertion order
       b. for offset d = 1..15: pointer of tracked frame t-d if present
     each pointer (256,) is split into 4 consecutive 64-d tokens  [L2771-2778]
     pointer positional embedding = ZEROS (this checkpoint has
     enable_temporal_pos_encoding_for_object_pointers = False)  [L2766-2769]
  4. KV assembly into the frozen (1, 3648, 64) buffers:
       [0 : n_maps*512)  map tokens in order (1);  bias 0
       [n_maps*512 : 3584)  padding;                bias -1e4
       [3584 : 3584 + 4*n_ptrs)  pointer tokens;    bias 0
       [.. : 3648)  padding;                        bias -1e4
     Padding placement is free ONLY at whole-map granularity (512-token
     slots): the graph RoPEs keys in 7 groups of 512.
  5. run memory_attention -> conditioned features -> mask_decoder_video with
     the no-prompt point ((0,0), label -1)  [L2414-2419]
  6. run memory_encoder on (raw vision features, high_res_masks,
     is_prompted=0) -> store BankEntry in `recent[t]`; drop entries older
     than t-6 (they can never be read again).

FIRST PROMPTED FRAME (t=0 here):
  no memory attention: conditioned = no_mem_embed(vision_features)
  [L2843-2852]; decode with the user's click (label 1); memory-encode with
  is_prompted=1 (mask BINARIZED before encoding, L3046-3048); store in
  `cond[0]`. Tracked frames use sigmoid instead (L3050-3051).
--------------------------------------------------------------------------

Run:  uv run --extra export python e2e_loop.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

# torch is used ONLY inside post_process (mask upsample to display resolution,
# a compositor concern in the browser); all model inference is ORT.
import torch

from clip_util import HEIGHT, NUM_FRAMES, PROMPT_FRAME, PROMPT_POINT_XY, WIDTH, make_clip

HERE = pathlib.Path(__file__).parent
MODEL_ID = "yonigozlan/EdgeTAM-hf"

# Frozen graph constants (must match websam_export.wrappers.edgetam).
NUM_RECENT = 6            # num_maskmem - 1
TOKENS_PER_MAP = 512
MAX_MEMORY_MAPS = 7
PTR_SPLITS = 4            # 256-d pointer -> 4 x 64-d tokens
MAX_POINTERS = 16
KV_LEN = MAX_MEMORY_MAPS * TOKENS_PER_MAP + MAX_POINTERS * PTR_SPLITS  # 3648
BIAS_NEG = np.float32(-1e4)
IOU_GATE = 0.95


class OrtEngine:
    """The five ONNX sessions. JS analog: five ort-web InferenceSessions."""

    def __init__(self, onnx_dir: pathlib.Path):
        def load(name):
            return ort.InferenceSession(str(onnx_dir / f"{name}.onnx"),
                                        providers=["CPUExecutionProvider"])

        self.vision = load("vision_encoder")
        self.no_mem = load("no_mem_embed")
        self.mem_attn = load("memory_attention")
        self.decoder = load("mask_decoder_video")
        self.mem_enc = load("memory_encoder")

    def encode_frame(self, pixel_values):
        feats, pos, hr0, hr1 = self.vision.run(None, {"pixel_values": pixel_values})
        return feats, pos, hr0, hr1

    def condition_no_memory(self, feats):
        return self.no_mem.run(None, {"vision_features": feats})[0]

    def condition_with_memory(self, feats, feats_pos, memory, memory_pos, bias):
        return self.mem_attn.run(None, {
            "current_vision_features": feats,
            "current_vision_pos_embed": feats_pos,
            "memory": memory,
            "memory_pos_embed": memory_pos,
            "attn_bias": bias,
        })[0]

    def decode(self, conditioned, hr0, hr1, coords, labels):
        names = ["low_res_masks", "high_res_masks", "object_pointer",
                 "object_score_logits", "iou_scores"]
        out = self.decoder.run(None, {
            "conditioned_features": conditioned,
            "high_res_features_0": hr0,
            "high_res_features_1": hr1,
            "point_coords": coords,
            "point_labels": labels,
        })
        return dict(zip(names, out))

    def encode_memory(self, feats, high_res_masks, is_prompted: bool):
        mem, mem_pos = self.mem_enc.run(None, {
            "vision_features": feats,
            "high_res_masks": high_res_masks,
            "is_prompted": np.array([1.0 if is_prompted else 0.0], dtype=np.float32),
        })
        return mem, mem_pos


class MemoryBank:
    """Slot bookkeeping. JS analog: memory-bank.ts."""

    def __init__(self, tpos: np.ndarray, mem_pos_const: np.ndarray):
        self.cond: dict[int, dict] = {}     # insertion-ordered (dict semantics)
        self.recent: dict[int, dict] = {}
        self.tpos = tpos                    # (7, 64) temporal embeddings
        self.mem_pos_const = mem_pos_const  # (512, 64) per-map positional enc

    def store(self, frame_idx: int, memory_features: np.ndarray,
              object_pointer: np.ndarray, *, is_cond: bool) -> None:
        entry = {"mem": memory_features[0], "ptr": object_pointer.reshape(256)}
        if is_cond:
            self.cond[frame_idx] = entry
        else:
            self.recent[frame_idx] = entry
            # Eviction: offsets beyond NUM_RECENT are unreachable (L2620).
            for old in [f for f in self.recent if f < frame_idx - NUM_RECENT]:
                del self.recent[old]

    def assemble(self, frame_idx: int):
        """Returns (memory (1,KV_LEN,64), memory_pos (1,KV_LEN,64), bias)."""
        memory = np.zeros((1, KV_LEN, 64), dtype=np.float32)
        mem_pos = np.zeros((1, KV_LEN, 64), dtype=np.float32)
        bias = np.full((1, 1, 1, KV_LEN), BIAS_NEG, dtype=np.float32)

        # --- spatial maps: cond (offset 0 -> tpos[-1]) then recent 6..1 ------
        maps: list[tuple[np.ndarray, int]] = [
            (e["mem"], NUM_RECENT) for e in self.cond.values()          # tpos row 6
        ]
        for offset in range(NUM_RECENT, 0, -1):                          # oldest first
            prev = frame_idx - offset
            if prev in self.recent:                                      # cond frames
                maps.append((self.recent[prev]["mem"], offset - 1))      # never here
        pos_in = 0
        for mem_map, tpos_row in maps:
            memory[0, pos_in:pos_in + TOKENS_PER_MAP] = mem_map
            mem_pos[0, pos_in:pos_in + TOKENS_PER_MAP] = self.mem_pos_const + self.tpos[tpos_row]
            bias[0, 0, 0, pos_in:pos_in + TOKENS_PER_MAP] = 0.0
            pos_in += TOKENS_PER_MAP
        assert len(maps) <= MAX_MEMORY_MAPS

        # --- object pointers: cond (past only) then tracked offsets 1..15 ---
        pointers = [e["ptr"] for f, e in self.cond.items() if f <= frame_idx]
        for d in range(1, MAX_POINTERS):
            ref = frame_idx - d
            if ref in self.recent:
                pointers.append(self.recent[ref]["ptr"])
        pointers = pointers[:MAX_POINTERS]
        ptr_base = MAX_MEMORY_MAPS * TOKENS_PER_MAP                      # 3584
        for i, ptr in enumerate(pointers):
            tok = ptr_base + i * PTR_SPLITS
            memory[0, tok:tok + PTR_SPLITS] = ptr.reshape(PTR_SPLITS, 64)
            # pointer positional embedding stays ZERO (temporal PE disabled)
            bias[0, 0, 0, tok:tok + PTR_SPLITS] = 0.0
        return memory, mem_pos, bias


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def main() -> None:
    engine = OrtEngine(HERE / "onnx")
    golden = np.load(HERE / "activations" / "golden.npz")
    tpos_table = np.load(HERE / "activations" / "tpos_table.npy")  # (7, 64)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ---- preprocessing (the only non-ORT model-adjacent step) ---------------
    frames = make_clip()
    vp_out = processor.video_processor(videos=[frames], return_tensors="np")
    pixel_values = np.asarray(vp_out["pixel_values_videos"][0], dtype=np.float32)  # (T,3,1024,1024)
    # Prompt rescale: original -> 1024-space, per-axis (anisotropic stretch).
    px = PROMPT_POINT_XY[0] * 1024.0 / WIDTH
    py = PROMPT_POINT_XY[1] * 1024.0 / HEIGHT
    click_coords = np.array([[[[px, py]]]], dtype=np.float32)   # (1,1,1,2)
    click_labels = np.array([[[1]]], dtype=np.int64)            # (1,1,1)
    # Tracked frames: HF's "no prompt" point (L2414-2419).
    track_coords = np.zeros((1, 1, 1, 2), dtype=np.float32)
    track_labels = -np.ones((1, 1, 1), dtype=np.int64)

    bank: MemoryBank | None = None
    masks_orig: list[np.ndarray] = []

    for t in range(NUM_FRAMES):
        feats, feats_pos, hr0, hr1 = engine.encode_frame(pixel_values[t:t + 1])

        if t == PROMPT_FRAME:
            conditioned = engine.condition_no_memory(feats)
            dec = engine.decode(conditioned, hr0, hr1, click_coords, click_labels)
            mem, mem_pos = engine.encode_memory(feats, dec["high_res_masks"], is_prompted=True)
            # First memory write also fixes the constant per-map positional
            # encoding and the tpos table (loaded once from the checkpoint by
            # the JS engine; here we read tpos from golden capture metadata).
            bank = MemoryBank(tpos=tpos_table, mem_pos_const=mem_pos[0])
            bank.store(t, mem, dec["object_pointer"], is_cond=True)
        else:
            memory, memory_pos, bias = bank.assemble(t)
            conditioned = engine.condition_with_memory(feats, feats_pos, memory, memory_pos, bias)
            dec = engine.decode(conditioned, hr0, hr1, track_coords, track_labels)
            mem, _ = engine.encode_memory(feats, dec["high_res_masks"], is_prompted=False)
            bank.store(t, mem, dec["object_pointer"], is_cond=False)

        # Post-process to original resolution (mirrors
        # Sam2VideoProcessor.post_process_masks: bilinear to (H,W), > 0.0) —
        # identical call used for the golden, so the comparison is symmetric.
        m = processor.post_process_masks(
            [torch.from_numpy(dec["low_res_masks"])], original_sizes=[[HEIGHT, WIDTH]],
            binarize=True,
        )[0][0, 0]
        masks_orig.append(m.numpy().astype(np.uint8))

    # ---- gate: per-frame IoU vs the HF PyTorch golden ------------------------
    hf_masks = golden["masks_orig_res"]
    print(f"{'frame':>5} {'IoU':>8} {'area_ort':>9} {'area_hf':>8}")
    worst = 1.0
    for t, (ours, ref) in enumerate(zip(masks_orig, hf_masks)):
        v = iou(ours > 0, ref > 0)
        worst = min(worst, v)
        print(f"{t:>5} {v:>8.4f} {int((ours > 0).sum()):>9} {int((ref > 0).sum()):>8}")
    print(f"\nworst-frame IoU = {worst:.4f}  (gate: >= {IOU_GATE})")
    if worst < IOU_GATE:
        raise SystemExit("E2E FAILED")
    print("E2E PASSED: pure-ORT loop matches HF PyTorch end-to-end")


if __name__ == "__main__":
    if not (HERE / "activations" / "tpos_table.npy").exists():
        raise SystemExit("run dump_constants.py first (writes activations/tpos_table.npy)")
    main()
