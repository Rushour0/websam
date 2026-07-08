"""EdgeTAM (transformers v5 ``EdgeTamVideoModel``) export wrappers.

Five graphs, mirroring the plan's six-graph partition adapted to EdgeTAM
(preprocess stays in JS, so five ``.onnx`` artifacts):

* :class:`EdgeTamVisionEncoderWrapper`   -> ``vision_encoder.onnx``
* :class:`EdgeTamNoMemEmbedWrapper`      -> ``no_mem_embed.onnx``
* :class:`EdgeTamMemoryAttentionWrapper` -> ``memory_attention.onnx``
* :class:`EdgeTamMaskDecoderVideoWrapper`-> ``mask_decoder_video.onnx``
* :class:`EdgeTamMemoryEncoderWrapper`   -> ``memory_encoder.onnx``

Every wrapper reuses the checkpoint's submodules; only the *control flow*
that HF keeps in Python (memory-bank assembly, mask selection, occlusion
blending) is re-expressed tensor-functionally so it can be traced. All
line-number citations below refer to
``transformers/models/edgetam_video/modeling_edgetam_video.py`` at
transformers v5.13.0.

Layout convention (deliberate divergence from ``spec.py``'s seq-first
``(tokens, 1, C)``): all spatial tensors cross graph boundaries as BCHW and
all memory-token tensors as batch-first ``(1, tokens, C)``, so the JS engine
never has to transpose between graphs. The seq-first permutes HF does
internally are folded into the wrappers.

Memory-attention KV freeze (from the real EdgeTAM config):

* the 2D Spatial Perceiver (``spatial_perceiver``) compresses each encoded
  memory frame to ``256 (1D latents) + 256 (2D latents) = 512`` tokens of
  ``mem_dim=64`` channels — NOT 256 as ``spec.py``'s EDGETAM_1024 currently
  assumes;
* the bank holds up to ``num_maskmem = 7`` memory maps (1 initial
  conditioning frame + 6 recent, for single-prompt sessions);
* object pointers: up to 16 pointers x 4 splits (256/64) = 64 KV tokens.

Frozen KV length = ``7 * 512 + 64 = 3648``. Validity is driven by an
additive float attention bias (0 for valid, large negative for padding).
Padding must respect the *group* structure: the RoPE application reshapes
spatial-memory keys to ``(B, H, 7, 512, D)`` groups (L419-448), so padded
slots must be whole 512-token maps, and pointer tokens always occupy the
final 64 positions (they are excluded from RoPE by a trailing-token count,
L411-413).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Frozen EdgeTAM export constants (derived from yonigozlan/EdgeTAM-hf config).
# ---------------------------------------------------------------------------

IMAGE_SIZE = 1024
GRID = 64                        # backbone_feature_sizes[-1]
HIDDEN = 256                     # fpn_hidden_size == memory_attention_hidden_size
MEM_DIM = 64                     # memory_encoder_output_channels
TOKENS_PER_MAP = 512             # 256 perceiver 1D latents + 256 2D latents
MAX_MEMORY_MAPS = 7              # num_maskmem (1 cond + 6 recent, single prompt)
MAX_OBJECT_POINTERS = 16         # max_object_pointers_in_encoder
PTR_SPLITS = HIDDEN // MEM_DIM   # 4 tokens per pointer (L2771-2778)
PTR_TOKENS = MAX_OBJECT_POINTERS * PTR_SPLITS  # 64
KV_LEN = MAX_MEMORY_MAPS * TOKENS_PER_MAP + PTR_TOKENS  # 3648
ATTN_BIAS_NEG = -1e4             # fp16-safe additive mask value (plan spec)
NO_OBJ_SCORE = -1024.0           # modeling_edgetam_video.py L1992
SIGMOID_SCALE_FOR_MEM_ENC = 20.0
SIGMOID_BIAS_FOR_MEM_ENC = -10.0


def rotate_pairwise(x: Tensor) -> Tensor:
    """Pairwise rotation used by EdgeTAM's 2D axial RoPE (L272-287)."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


class EdgeTamVisionEncoderWrapper(nn.Module):
    """``vision_encoder.onnx`` — RepViT backbone + FPN neck + decoder skip convs.

    Mirrors ``EdgeTamVideoModel.get_image_features`` (L2244-2273): the
    stride-4/stride-8 FPN levels are pre-projected with the mask decoder's
    ``conv_s0``/``conv_s1`` exactly as HF does once per frame, so the decoder
    graph never re-runs them per click.

    Outputs are the *raw* features — ``no_memory_embedding`` is NOT added here
    (in the video path it is only applied on initial conditioning frames, see
    L2843-2852); frame-0 handling uses :class:`EdgeTamNoMemEmbedWrapper`.
    """

    def __init__(self, model):
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.conv_s0 = model.mask_decoder.conv_s0
        self.conv_s1 = model.mask_decoder.conv_s1

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = self.vision_encoder(pixel_values, return_dict=True)
        feats = list(out.fpn_hidden_states)          # [(1,256,256,256),(1,256,128,128),(1,256,64,64)]
        pos = list(out.fpn_position_encoding)
        high_res_0 = self.conv_s0(feats[0])          # (1, 32, 256, 256)
        high_res_1 = self.conv_s1(feats[1])          # (1, 64, 128, 128)
        return (
            feats[2],        # vision_features     (1, 256, 64, 64)
            pos[2],          # vision_pos_embed    (1, 256, 64, 64)
            high_res_0,      # high_res_features_0 (1, 32, 256, 256)
            high_res_1,      # high_res_features_1 (1, 64, 128, 128)
        )


class EdgeTamNoMemEmbedWrapper(nn.Module):
    """``no_mem_embed.onnx`` — initial-conditioning-frame feature conditioning.

    On frames with first user input there is no memory to attend to; HF adds
    the learned ``no_memory_embedding`` (1,1,256) to the flattened top-level
    features instead (L2843-2852). Seq-first broadcast == per-channel add in
    BCHW, which is what this graph does.
    """

    def __init__(self, model):
        super().__init__()
        self.no_memory_embedding = model.no_memory_embedding  # (1, 1, 256)

    def forward(self, vision_features: Tensor) -> Tensor:
        return vision_features + self.no_memory_embedding.view(1, -1, 1, 1)


class EdgeTamMemoryAttentionWrapper(nn.Module):
    """``memory_attention.onnx`` — fixed-max padded KV + additive attention bias.

    Functionally re-implements ``EdgeTamVideoMemoryAttention.forward``
    (L1229-1277) + ``EdgeTamVideoMemoryAttentionLayer.forward`` (L1183-1214)
    + the RoPE self/cross attention (L338-377, L476-528, L380-452), reusing
    the checkpoint's weights, with two deliberate changes:

    1. KV length frozen at :data:`KV_LEN` (= 7 maps x 512 + 64 ptr tokens);
       ``num_k_exclude_rope`` frozen at 64 and ``rope_k_repeat`` at 7.
    2. ``attn_bias`` (1, 1, 1, KV_LEN) float32 is ADDED to the cross-attention
       scores (HF passes ``attention_mask=None`` here; eager attention's
       additive-mask hook, L189-191, is the sanctioned insertion point).
       0 = valid token, :data:`ATTN_BIAS_NEG` = padding. Self-attention over
       the 4096 queries is never masked, matching HF.

    IO is batch-first: queries in/out as BCHW (1,256,64,64), memory KV as
    (1, KV_LEN, 64). The wrapper folds HF's seq-first permutes.
    """

    def __init__(self, model):
        super().__init__()
        ma = model.memory_attention
        self.layers = ma.layers
        self.layer_norm = ma.layer_norm
        # Precomputed RoPE tables (constant: fixed 64x64 query grid, 16x16 key grid).
        cos_q, sin_q = ma.rotary_emb()
        cos_k, sin_k = ma.rotary_emb_k()
        self.register_buffer("rope_cos_q", cos_q, persistent=False)  # (4096, 256)
        self.register_buffer("rope_sin_q", sin_q, persistent=False)
        self.register_buffer("rope_cos_k", cos_k, persistent=False)  # (256, 256)
        self.register_buffer("rope_sin_k", sin_k, persistent=False)

    def _rope_q(self, q: Tensor) -> Tensor:
        # apply_rotary_pos_emb_2d_* query path (L308-310 / L406-408).
        q = q.float()
        return (q * self.rope_cos_q) + (rotate_pairwise(q) * self.rope_sin_q)

    def _self_attn(self, attn, hidden: Tensor) -> Tensor:
        # EdgeTamVideoRoPESelfAttention.forward (L338-377), eval-mode eager.
        batch_size, point_batch_size = hidden.shape[:2]
        new_shape = (batch_size * point_batch_size, -1, attn.num_attention_heads, attn.head_dim)
        query = attn.q_proj(hidden).view(*new_shape).transpose(1, 2)
        key = attn.k_proj(hidden).view(*new_shape).transpose(1, 2)
        value = attn.v_proj(hidden).view(*new_shape).transpose(1, 2)
        query = self._rope_q(query).type_as(value)
        key = self._rope_q(key).type_as(value)
        weights = torch.matmul(query, key.transpose(2, 3)) * attn.scaling
        weights = torch.softmax(weights, dim=-1)
        out = torch.matmul(weights, value).transpose(1, 2)
        out = out.reshape(batch_size, point_batch_size, -1, attn.num_attention_heads * attn.head_dim)
        return attn.o_proj(out)

    def _cross_attn(self, attn, hidden: Tensor, key_in: Tensor, value_in: Tensor,
                    attn_bias: Tensor) -> Tensor:
        # EdgeTamVideoRoPECrossAttention.forward (L476-528) with frozen
        # num_k_exclude_rope=PTR_TOKENS, rope_k_repeat=MAX_MEMORY_MAPS, plus
        # the additive bias threaded into the eager score matrix (L189-191).
        batch_size, point_batch_size = hidden.shape[:2]
        new_shape = (batch_size * point_batch_size, -1, attn.num_attention_heads, attn.head_dim)
        query = attn.q_proj(hidden).view(*new_shape).transpose(1, 2)
        key = attn.k_proj(key_in).view(*new_shape).transpose(1, 2)
        value = attn.v_proj(value_in).view(*new_shape).transpose(1, 2)

        query = self._rope_q(query).type_as(value)

        # apply_rotary_pos_emb_2d_cross_attn key path (L410-451).
        k_for_rope = key[..., : KV_LEN - PTR_TOKENS, :]
        k_excluded = key[..., KV_LEN - PTR_TOKENS:, :]
        bsz, num_heads, _, ch = k_for_rope.shape
        # (B, H, 7, 512, D): groups of one memory map each.
        k_grouped = k_for_rope.view(bsz, num_heads, MAX_MEMORY_MAPS, TOKENS_PER_MAP, ch)
        # First 256 tokens per map = perceiver 1D latents ("temporal", no RoPE);
        # last 256 = 2D latents on the 16x16 grid (RoPE'd). L421-433.
        spatial_tokens = self.rope_cos_k.shape[-2]                      # 256
        temporal_tokens = TOKENS_PER_MAP - spatial_tokens               # 256
        k_temporal = k_grouped[..., :temporal_tokens, :].reshape(bsz, num_heads, -1, ch)
        k_spatial = k_grouped[..., temporal_tokens:, :].reshape(bsz, num_heads, -1, ch)
        cos_k = self.rope_cos_k.repeat(1, 1, MAX_MEMORY_MAPS, 1)        # L436-438
        sin_k = self.rope_sin_k.repeat(1, 1, MAX_MEMORY_MAPS, 1)
        k_spatial_embed = k_spatial.float()
        k_spatial_embed = (k_spatial_embed * cos_k) + (rotate_pairwise(k_spatial_embed) * sin_k)
        k_spatial_reshaped = k_spatial_embed.view(bsz, num_heads, MAX_MEMORY_MAPS, -1, ch)
        k_temporal_reshaped = k_temporal.view(bsz, num_heads, MAX_MEMORY_MAPS, -1, ch)
        k_final = torch.cat([k_temporal_reshaped, k_spatial_reshaped], dim=3)
        k_final = k_final.view(bsz, num_heads, MAX_MEMORY_MAPS * TOKENS_PER_MAP, ch)
        key = torch.cat([k_final.type_as(key), k_excluded], dim=-2)

        weights = torch.matmul(query, key.transpose(2, 3)) * attn.scaling
        weights = weights + attn_bias                                   # padding mask
        weights = torch.softmax(weights, dim=-1)
        out = torch.matmul(weights, value).transpose(1, 2)
        out = out.reshape(batch_size, point_batch_size, -1, attn.num_attention_heads * attn.head_dim)
        return attn.o_proj(out)

    def forward(
        self,
        current_vision_features: Tensor,   # (1, 256, 64, 64) raw frame features
        current_vision_pos_embed: Tensor,  # (1, 256, 64, 64) sine pos encoding
        memory: Tensor,                    # (1, KV_LEN, 64) padded memory bank
        memory_pos_embed: Tensor,          # (1, KV_LEN, 64) spatial+tpos (+zeros for ptrs)
        attn_bias: Tensor,                 # (1, 1, 1, KV_LEN) additive float mask
    ) -> Tensor:
        batch = current_vision_features.shape[0]
        # BCHW -> (B, HW, C); HF works seq-first then transposes (L1251-1258).
        feats = current_vision_features.flatten(2).transpose(1, 2)
        pos = current_vision_pos_embed.flatten(2).transpose(1, 2)
        output = feats + 0.1 * pos                                      # L1252-1253
        output = output.unsqueeze(1)                                    # (B, 1, 4096, 256)
        keys = memory.unsqueeze(1)                                      # (B, 1, KV_LEN, 64)
        key_pos = memory_pos_embed.unsqueeze(1)

        for layer in self.layers:                                       # L1183-1214
            query = layer.layer_norm1(output)
            query = self._self_attn(layer.self_attn, query)
            output = output + query                                     # dropout inert in eval
            query = layer.layer_norm2(output)
            query = self._cross_attn(
                layer.cross_attn_image, query, keys + key_pos, keys, attn_bias
            )                                                           # L1200-1208
            output = output + query
            query = layer.layer_norm3(output)
            output = output + layer.mlp(query)

        output = self.layer_norm(output)                                # L1272
        # (B, 1, 4096, 256) -> BCHW, as _prepare_memory_conditioned_features
        # L2894-2897 does before the decoder.
        conditioned = output.squeeze(1).transpose(1, 2).view(batch, HIDDEN, GRID, GRID)
        return conditioned


class EdgeTamMaskDecoderVideoWrapper(nn.Module):
    """``mask_decoder_video.onnx`` — fused prompt encoder + mask decoder +
    object-pointer/occlusion heads + best-mask selection.

    Mirrors ``_single_frame_forward`` (L2305-2501) for the video loop's
    point-prompt path with ``multimask_output=True`` frozen. That freeze is
    exact for this checkpoint: ``_use_multimask`` (L2900-2908) is True
    whenever ``num_pts <= 1`` (``multimask_output_in_sam`` and
    ``multimask_output_for_tracking`` are both True, ``multimask_min/max_pt_num``
    = 0/1), i.e. for single-click conditioning frames AND all tracked frames.
    Multi-point refinement prompts (>1 point) would need a second export with
    ``multimask_output=False`` (which routes through
    ``_dynamic_multimask_via_stability``, L1929-1931) — out of M2 scope.

    ``num_points`` is a dynamic axis: HF appends its own (0,0)/-1 padding
    point inside ``_embed_points`` (L1654-1656), and padded ``-10`` tokens are
    NOT numerically inert in the two-way attention, so the JS engine must feed
    the exact prompt count instead of a frozen P:

    * conditioning frame, one click: coords (1,1,1,2) in 1024-space, labels
      (1,1,1) = 1;
    * tracked frames (no prompts): coords (1,1,1,2) = (0,0), labels = -1,
      matching L2414-2419.
    """

    def __init__(self, model):
        super().__init__()
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder
        self.object_pointer_proj = model.object_pointer_proj
        self.no_object_pointer = model.no_object_pointer
        # Constant image-wide positional embedding (L2388): buffer-derived,
        # baked into the graph as a constant.
        image_pe = model.get_image_wide_positional_embeddings()
        self.register_buffer("image_positional_embeddings", image_pe, persistent=False)

    def forward(
        self,
        conditioned_features: Tensor,  # (1, 256, 64, 64) memory-conditioned (or no-mem) feats
        high_res_features_0: Tensor,   # (1, 32, 256, 256)
        high_res_features_1: Tensor,   # (1, 64, 128, 128)
        point_coords: Tensor,          # (1, 1, num_points, 2) float32, 1024-space pixels
        point_labels: Tensor,          # (1, 1, num_points) int64 (1 pos, 0 neg, -1 pad/none)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=point_coords,
            input_labels=point_labels,
            input_boxes=None,
            input_masks=None,
        )
        low_res_multimasks, iou_scores, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=conditioned_features,
            image_positional_embeddings=self.image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            high_resolution_features=[high_res_features_0, high_res_features_1],
            attention_similarity=None,
            target_embedding=None,
        )
        # Occlusion clamp (L2451-2458): all-mask NO_OBJ_SCORE when object absent.
        is_obj_appearing = object_score_logits > 0                       # (1, 1, 1) bool
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            torch.full_like(low_res_multimasks, NO_OBJ_SCORE),
        )
        high_res_multimasks = nn.functional.interpolate(
            low_res_multimasks.squeeze(1).float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(1)                                                   # L2462-2471
        # Best-of-3 selection by predicted IoU (L2472-2481).
        best_iou_inds = torch.argmax(iou_scores, dim=-1)                 # (1, 1)
        batch_inds = torch.arange(1, device=iou_scores.device)
        low_res_masks = low_res_multimasks[batch_inds, batch_inds, best_iou_inds]
        high_res_masks = high_res_multimasks[batch_inds, batch_inds, best_iou_inds]
        sam_output_token = sam_output_tokens[batch_inds, batch_inds, best_iou_inds]
        # Object pointer with occlusion blend (L2486-2490):
        # is_obj_appearing (B,P,1) broadcasts against object_pointer (B,P,256).
        object_pointer = self.object_pointer_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.to(object_pointer.dtype)
        object_pointer = lambda_is_obj_appearing * object_pointer
        object_pointer = object_pointer + (1 - lambda_is_obj_appearing) * self.no_object_pointer
        return (
            low_res_masks,        # (1, 1, 256, 256) logits
            high_res_masks,       # (1, 1, 1024, 1024) logits
            object_pointer,       # (1, 1, 256)
            object_score_logits,  # (1, 1, 1)
            iou_scores,           # (1, 1, 3)
        )


class EdgeTamMemoryEncoderWrapper(nn.Module):
    """``memory_encoder.onnx`` — mask prep + memory encoder + 2D Spatial Perceiver.

    Mirrors ``_encode_new_memory`` (L3033-3073). The Perceiver lives HERE (it
    compresses the just-encoded 64x64 memory map to 512 latent tokens before
    storage), not in the attention graph. The prompted-vs-tracked branch
    (binarize vs sigmoid, L3046-3051) is folded in via a float ``is_prompted``
    input so the graph is branch-free.

    ``object_score_logits`` is NOT an input: this checkpoint has
    ``enable_occlusion_spatial_embedding=False`` so the occlusion-embedding add
    (L3062-3066) is dead code.

    Note: ``memory_pos_embed`` output is a frame-independent constant (sine
    encodings of fixed grids; the 1D-latent half is exactly zero, L1512-1514).
    Exported as an output anyway for contract clarity; JS may cache it.
    """

    def __init__(self, model):
        super().__init__()
        self.memory_encoder = model.memory_encoder
        self.spatial_perceiver = model.spatial_perceiver

    def forward(
        self,
        vision_features: Tensor,  # (1, 256, 64, 64) RAW frame features (no no_mem_embed)
        high_res_masks: Tensor,   # (1, 1, 1024, 1024) decoder mask logits
        is_prompted: Tensor,      # (1,) float32: 1.0 if this frame had point/mask inputs
    ) -> tuple[Tensor, Tensor]:
        binarized = (high_res_masks > 0).to(high_res_masks.dtype)
        soft = torch.sigmoid(high_res_masks)
        mask_for_mem = torch.where(is_prompted.view(1, 1, 1, 1) > 0.5, binarized, soft)
        mask_for_mem = mask_for_mem * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC
        features, pos = self.memory_encoder(vision_features, mask_for_mem)  # (1,64,64,64) x2
        features, pos = self.spatial_perceiver(features, pos)               # (1,512,64) x2
        return features, pos
