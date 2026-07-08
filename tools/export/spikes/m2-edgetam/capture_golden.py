"""Run the HF EdgeTAM video pipeline end-to-end on the synthetic clip and
capture every graph-boundary activation as golden fixtures for parity.

Outputs (all under ./activations/, gitignored):
  golden.npz  — per-frame tensors keyed  f{t}/<name>  plus clip-level keys.

Captured per frame t:
  f{t}/pixel_values            (1,3,1024,1024)  processed frame
  f{t}/dec_pix_feat            (1,256,64,64)    decoder input (no-mem or mem-conditioned)
  f{t}/dec_high_res_0          (1,32,256,256)
  f{t}/dec_high_res_1          (1,64,128,128)
  f{t}/dec_point_coords        (1,1,P,2)        1024-space (HF-normalized)
  f{t}/dec_point_labels        (1,1,P)
  f{t}/dec_low_res_masks       (1,1,256,256)
  f{t}/dec_high_res_masks      (1,1,1024,1024)
  f{t}/dec_object_pointer      (1,1,256)
  f{t}/dec_object_score_logits (1,1,1)
  f{t}/dec_iou_scores          (1,1,3)
  f{t}/enc_vision_features     (1,256,64,64)    raw feats into memory encoder
  f{t}/enc_high_res_masks      (1,1,1024,1024)
  f{t}/enc_is_prompted         (1,)
  f{t}/enc_memory_features     (1,512,64)
  f{t}/enc_memory_pos          (1,512,64)
  and for t >= 1 (memory attention ran):
  f{t}/ma_current_vision_features (4096,1,256)  seq-first, as HF passes them
  f{t}/ma_current_vision_pos      (4096,1,256)
  f{t}/ma_memory                  (S,1,64)      unpadded, real tokens only
  f{t}/ma_memory_pos              (S,1,64)
  f{t}/ma_num_spatial_maps        ()            int (number of 512-token maps)
  f{t}/ma_num_ptr_tokens          ()            int (pointer tokens, = 4*n_ptrs)
  f{t}/ma_output                  (1,256,64,64) conditioned features (BCHW)

Clip-level:
  masks_orig_res  (T,480,640)  post-processed binary masks (the user-facing IoU target)
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from transformers import AutoProcessor, EdgeTamVideoModel

from clip_util import PROMPT_FRAME, PROMPT_POINT_XY, HEIGHT, WIDTH, make_clip

HERE = pathlib.Path(__file__).parent
OUT = HERE / "activations"
OUT.mkdir(exist_ok=True)

MODEL_ID = "yonigozlan/EdgeTAM-hf"


def main() -> None:
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = EdgeTamVideoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    frames = make_clip()
    session = processor.init_video_session(video=frames, inference_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        session,
        frame_idx=PROMPT_FRAME,
        obj_ids=1,
        input_points=[[[list(PROMPT_POINT_XY)]]],
        input_labels=[[[1]]],
    )

    store: dict[str, np.ndarray] = {}

    def put(key: str, value) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        store[key] = np.asarray(value)

    frame_ptr = {"t": None}  # current frame index, set inside the loop

    # ---- capture memory attention IO (kwargs call, L2885-2892) -------------
    def ma_hook(module, args, kwargs, output):
        t = frame_ptr["t"]
        put(f"f{t}/ma_current_vision_features", kwargs["current_vision_features"])
        put(f"f{t}/ma_current_vision_pos", kwargs["current_vision_position_embeddings"])
        put(f"f{t}/ma_memory", kwargs["memory"])
        put(f"f{t}/ma_memory_pos", kwargs["memory_posision_embeddings"])
        put(f"f{t}/ma_num_ptr_tokens", kwargs["num_object_pointer_tokens"])
        put(f"f{t}/ma_num_spatial_maps", kwargs["num_spatial_memory_tokens"])
        # HF returns (1,1,4096,256); store as BCHW like the exported graph.
        cond = output.squeeze(1).transpose(1, 2).reshape(1, 256, 64, 64)
        put(f"f{t}/ma_output", cond)

    model.memory_attention.register_forward_hook(ma_hook, with_kwargs=True)

    # ---- capture fused-decoder IO by wrapping _single_frame_forward --------
    orig_sff = model._single_frame_forward

    def sff_wrap(**kwargs):
        t = frame_ptr["t"]
        out = orig_sff(**kwargs)
        emb = kwargs["image_embeddings"]
        put(f"f{t}/dec_high_res_0", emb[0])
        put(f"f{t}/dec_high_res_1", emb[1])
        put(f"f{t}/dec_pix_feat", emb[2])
        pts = kwargs.get("input_points")
        lbl = kwargs.get("input_labels")
        if pts is None:  # tracked frame: HF synthesizes (0,0)/-1 (L2414-2419)
            pts = torch.zeros(1, 1, 1, 2)
            lbl = -torch.ones(1, 1, 1, dtype=torch.int64)
        put(f"f{t}/dec_point_coords", pts)
        put(f"f{t}/dec_point_labels", lbl.to(torch.int64))
        put(f"f{t}/dec_low_res_masks", out.pred_masks)
        put(f"f{t}/dec_high_res_masks", out.high_res_masks)
        put(f"f{t}/dec_object_pointer", out.object_pointer)
        put(f"f{t}/dec_object_score_logits", out.object_score_logits)
        put(f"f{t}/dec_iou_scores", out.iou_scores)
        return out

    model._single_frame_forward = sff_wrap

    # ---- capture memory-encoder IO by wrapping _encode_new_memory ----------
    orig_enc = model._encode_new_memory

    def enc_wrap(current_vision_feats, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        t = frame_ptr["t"]
        feats, pos = orig_enc(current_vision_feats, pred_masks_high_res, object_score_logits, is_mask_from_pts)
        pix = current_vision_feats.permute(1, 2, 0).reshape(1, 256, 64, 64)
        put(f"f{t}/enc_vision_features", pix)
        put(f"f{t}/enc_high_res_masks", pred_masks_high_res)
        put(f"f{t}/enc_is_prompted", np.array([1.0 if is_mask_from_pts else 0.0], dtype=np.float32))
        put(f"f{t}/enc_memory_features", feats)
        put(f"f{t}/enc_memory_pos", pos)
        return feats, pos

    model._encode_new_memory = enc_wrap

    # ---- run the reference loop --------------------------------------------
    masks_orig = []
    frame_ptr["t"] = PROMPT_FRAME
    out0 = model(session, frame_idx=PROMPT_FRAME)

    def to_orig(pred_masks):
        m = processor.post_process_masks([pred_masks], original_sizes=[[HEIGHT, WIDTH]],
                                         binarize=True)[0]
        return m[0, 0].cpu().numpy().astype(np.uint8)

    masks_orig.append(to_orig(out0.pred_masks))
    # Drive frames one by one (equivalent to propagate_in_video_iterator,
    # L3082-3134, which just calls model(session, frame_idx=t) in order) so the
    # hooks see the right frame index in frame_ptr BEFORE each forward.
    for t in range(PROMPT_FRAME + 1, session.num_frames):
        frame_ptr["t"] = t
        out = model(session, frame_idx=t)
        masks_orig.append(to_orig(out.pred_masks))

    put("masks_orig_res", np.stack(masks_orig))

    for t in range(session.num_frames):
        put(f"f{t}/pixel_values", session.get_frame(t).unsqueeze(0))

    np.savez_compressed(OUT / "golden.npz", **store)
    sizes = {k: v.shape for k, v in store.items() if k.startswith("f1/")}
    print("captured", len(store), "tensors ->", OUT / "golden.npz")
    for k, v in sorted(sizes.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
