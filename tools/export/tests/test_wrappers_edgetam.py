"""Wrapper-vs-HF equivalence tests for the EdgeTAM export wrappers.

Network-free: builds a randomly initialized ``EdgeTamVideoModel`` from the
default config with the two flags pinned to the shipped checkpoint's values
(``yonigozlan/EdgeTAM-hf``: pointer temporal PE and occlusion spatial
embedding both DISABLED — the wrappers encode that assumption).

Requires the ``export`` extra (torch + transformers + timm); skipped otherwise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("timm")

from transformers import EdgeTamVideoConfig, EdgeTamVideoModel  # noqa: E402

from websam_export.wrappers.edgetam import (  # noqa: E402
    ATTN_BIAS_NEG,
    GRID,
    HIDDEN,
    IMAGE_SIZE,
    KV_LEN,
    MAX_MEMORY_MAPS,
    MAX_OBJECT_POINTERS,
    MEM_DIM,
    PTR_TOKENS,
    TOKENS_PER_MAP,
    EdgeTamMaskDecoderVideoWrapper,
    EdgeTamMemoryAttentionWrapper,
    EdgeTamMemoryEncoderWrapper,
    EdgeTamNoMemEmbedWrapper,
    EdgeTamVisionEncoderWrapper,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = EdgeTamVideoConfig(
        enable_temporal_pos_encoding_for_object_pointers=False,
        enable_occlusion_spatial_embedding=False,
    )
    config._attn_implementation = "eager"
    with torch.no_grad():
        m = EdgeTamVideoModel(config).eval()
    return m


def build_separated_inputs(model, n_maps: int, n_ptr_vecs: int, *, fill: float = 0.0):
    """Build the SEPARATED (JS-fed) inputs for `n_maps` valid spatial maps and
    `n_ptr_vecs` valid pointer vectors, plus the equivalent seq-first HF
    `memory`/`memory_pos_embed` tensors (built via the same tpos + split rule)
    so both sides can be compared against the real `model.memory_attention`.
    `fill` seeds the padding regions (0.0 clean / large poisoning value).
    """
    tpos = model.memory_temporal_positional_encoding.detach().reshape(MAX_MEMORY_MAPS, MEM_DIM)

    memory_spatial = torch.full((1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM), fill)
    memory_spatial_pos = torch.full((1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM), fill)
    tpos_indices = torch.full((1, MAX_MEMORY_MAPS), -1, dtype=torch.int64)
    for s in range(n_maps):
        memory_spatial[0, s] = torch.randn(TOKENS_PER_MAP, MEM_DIM)
        memory_spatial_pos[0, s] = torch.randn(TOKENS_PER_MAP, MEM_DIM)
        row = s % MAX_MEMORY_MAPS  # arbitrary valid rows, exercises the gather
        tpos_indices[0, s] = row

    object_pointers = torch.full((1, MAX_OBJECT_POINTERS, 256), fill)
    for p in range(n_ptr_vecs):
        object_pointers[0, p] = torch.randn(256)

    memory_mask = torch.zeros(1, KV_LEN, dtype=torch.bool)
    memory_mask[0, : n_maps * TOKENS_PER_MAP] = True
    ptr_base = MAX_MEMORY_MAPS * TOKENS_PER_MAP
    memory_mask[0, ptr_base : ptr_base + n_ptr_vecs * 4] = True

    pointer_deltas = torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.int64)
    pointer_mask = torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.bool)
    pointer_mask[0, :n_ptr_vecs] = True

    # --- equivalent seq-first HF tensors (ground truth path) ---
    n_ptr_tok = n_ptr_vecs * 4
    seq = n_maps * TOKENS_PER_MAP + n_ptr_tok
    mem_seq = torch.zeros(seq, 1, MEM_DIM)
    mem_pos_seq = torch.zeros(seq, 1, MEM_DIM)
    for s in range(n_maps):
        row = s % MAX_MEMORY_MAPS
        lo, hi = s * TOKENS_PER_MAP, (s + 1) * TOKENS_PER_MAP
        mem_seq[lo:hi, 0] = memory_spatial[0, s]
        mem_pos_seq[lo:hi, 0] = memory_spatial_pos[0, s] + tpos[row]
    for p in range(n_ptr_vecs):
        lo, hi = n_maps * TOKENS_PER_MAP + p * 4, n_maps * TOKENS_PER_MAP + (p + 1) * 4
        mem_seq[lo:hi, 0] = object_pointers[0, p].reshape(4, MEM_DIM)
        # mem_pos_seq stays zero for pointer tokens (temporal PE disabled).

    separated = (memory_spatial, memory_spatial_pos, tpos_indices, memory_mask,
                 object_pointers, pointer_deltas, pointer_mask)
    return separated, mem_seq, mem_pos_seq, n_maps, n_ptr_tok


@torch.no_grad()
def test_memory_attention_wrapper_matches_hf(model):
    torch.manual_seed(1)
    n_maps, n_ptr_vecs = 3, 2  # 3 real memory maps, 2 pointer vectors (8 tokens)
    feats_seq = torch.randn(GRID * GRID, 1, HIDDEN)
    pos_seq = torch.randn(GRID * GRID, 1, HIDDEN)

    separated, mem_seq, mem_pos_seq, n_maps, n_ptr_tok = build_separated_inputs(
        model, n_maps, n_ptr_vecs
    )

    hf_out = model.memory_attention(
        current_vision_features=feats_seq,
        current_vision_position_embeddings=pos_seq,
        memory=mem_seq,
        memory_posision_embeddings=mem_pos_seq,
        num_object_pointer_tokens=n_ptr_tok,
        num_spatial_memory_tokens=n_maps,
    )
    hf_bchw = hf_out.squeeze(1).transpose(1, 2).reshape(1, HIDDEN, GRID, GRID)

    w = EdgeTamMemoryAttentionWrapper(model).eval()
    got = w(
        feats_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous(),
        pos_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous(),
        *separated,
    )
    assert torch.allclose(got, hf_bchw, atol=1e-5), (got - hf_bchw).abs().max()


@torch.no_grad()
def test_memory_attention_padding_is_inert(model):
    torch.manual_seed(2)
    feats = torch.randn(1, HIDDEN, GRID, GRID)
    pos = torch.randn(1, HIDDEN, GRID, GRID)
    w = EdgeTamMemoryAttentionWrapper(model).eval()

    torch.manual_seed(20)
    clean_inputs, *_ = build_separated_inputs(model, 2, 1, fill=0.0)
    torch.manual_seed(20)
    poisoned_inputs, *_ = build_separated_inputs(model, 2, 1, fill=1e3)

    clean = w(feats, pos, *clean_inputs)
    poisoned = w(feats, pos, *poisoned_inputs)
    assert torch.equal(clean, poisoned)


@torch.no_grad()
def test_memory_attention_tpos_gather_matches_direct_add(model):
    """The in-graph `tpos_table[idx]` gather must equal a direct row lookup —
    guards against an off-by-one in the clamp/gather that padding-inertness
    alone would not catch (padding is masked, but valid rows are not)."""
    torch.manual_seed(3)
    tpos = model.memory_temporal_positional_encoding.detach().reshape(MAX_MEMORY_MAPS, MEM_DIM)
    feats = torch.randn(1, HIDDEN, GRID, GRID)
    pos = torch.randn(1, HIDDEN, GRID, GRID)
    w = EdgeTamMemoryAttentionWrapper(model).eval()

    memory_spatial = torch.zeros(1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM)
    memory_spatial[0, 0] = torch.randn(TOKENS_PER_MAP, MEM_DIM)
    memory_spatial_pos = torch.zeros(1, MAX_MEMORY_MAPS, TOKENS_PER_MAP, MEM_DIM)
    object_pointers = torch.zeros(1, MAX_OBJECT_POINTERS, 256)
    memory_mask = torch.zeros(1, KV_LEN, dtype=torch.bool)
    memory_mask[0, :TOKENS_PER_MAP] = True
    pointer_deltas = torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.int64)
    pointer_mask = torch.zeros(1, MAX_OBJECT_POINTERS, dtype=torch.bool)

    for row in range(MAX_MEMORY_MAPS):
        tpos_indices = torch.full((1, MAX_MEMORY_MAPS), -1, dtype=torch.int64)
        tpos_indices[0, 0] = row
        got = w(feats, pos, memory_spatial, memory_spatial_pos, tpos_indices,
                 memory_mask, object_pointers, pointer_deltas, pointer_mask)

        # Reference: assemble the seq-first KV with the row added directly.
        mem_seq = memory_spatial[0, 0].unsqueeze(1)
        mem_pos_seq = tpos[row].expand(TOKENS_PER_MAP, MEM_DIM).unsqueeze(1)
        hf_out = model.memory_attention(
            current_vision_features=feats.flatten(2).permute(2, 0, 1),
            current_vision_position_embeddings=pos.flatten(2).permute(2, 0, 1),
            memory=mem_seq,
            memory_posision_embeddings=mem_pos_seq,
            num_object_pointer_tokens=0,
            num_spatial_memory_tokens=1,
        )
        want = hf_out.squeeze(1).transpose(1, 2).reshape(1, HIDDEN, GRID, GRID)
        assert torch.allclose(got, want, atol=1e-5), f"row {row}: {(got - want).abs().max()}"


@torch.no_grad()
def test_no_mem_embed_matches_hf(model):
    torch.manual_seed(3)
    feats_bchw = torch.randn(1, HIDDEN, GRID, GRID)
    w = EdgeTamNoMemEmbedWrapper(model)
    got = w(feats_bchw)
    # HF: seq-first add of (1,1,256) then reshape back (L2843-2852).
    seq = feats_bchw.flatten(2).permute(2, 0, 1)
    want = (seq + model.no_memory_embedding).permute(1, 2, 0).reshape(1, HIDDEN, GRID, GRID)
    assert torch.allclose(got, want, atol=0)


@torch.no_grad()
def test_mask_decoder_wrapper_matches_single_frame_forward(model):
    torch.manual_seed(4)
    pix_feat = torch.randn(1, HIDDEN, GRID, GRID)
    hr0 = torch.randn(1, 32, GRID * 4, GRID * 4)
    hr1 = torch.randn(1, 64, GRID * 2, GRID * 2)
    coords = torch.tensor([[[[512.0, 384.0]]]])
    labels = torch.ones(1, 1, 1, dtype=torch.int64)

    hf = model._single_frame_forward(
        input_points=coords,
        input_labels=labels,
        image_embeddings=[hr0, hr1, pix_feat],
        multimask_output=True,
    )
    w = EdgeTamMaskDecoderVideoWrapper(model).eval()
    low, high, ptr, score, iou = w(pix_feat, hr0, hr1, coords, labels)
    assert torch.allclose(low, hf.pred_masks, atol=1e-6)
    assert torch.allclose(high, hf.high_res_masks, atol=1e-6)
    assert torch.allclose(ptr, hf.object_pointer, atol=1e-6)
    assert torch.allclose(score, hf.object_score_logits, atol=1e-6)
    assert torch.allclose(iou, hf.iou_scores, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("prompted", [True, False])
def test_memory_encoder_wrapper_matches_encode_new_memory(model, prompted):
    torch.manual_seed(5)
    feats_seq = torch.randn(GRID * GRID, 1, HIDDEN)
    # LOW-res decoder output (== mask_decoder_video's `low_res_masks` /
    # maskLogits semantic key) — PIN-7 reconciliation: the wrapper now
    # upsamples in-graph, so the HF reference upsamples identically here.
    low_res_masks = torch.randn(1, 1, GRID * 4, GRID * 4) * 8
    high_res_masks = torch.nn.functional.interpolate(
        low_res_masks.float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    hf_feats, hf_pos = model._encode_new_memory(
        current_vision_feats=feats_seq,
        pred_masks_high_res=high_res_masks,
        object_score_logits=torch.tensor([[10.0]]),
        is_mask_from_pts=prompted,
    )
    w = EdgeTamMemoryEncoderWrapper(model).eval()
    feats_bchw = feats_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous()
    got_feats, got_pos = w(feats_bchw, low_res_masks,
                           torch.tensor([1.0 if prompted else 0.0]))
    assert torch.allclose(got_feats, hf_feats, atol=1e-6)
    assert torch.allclose(got_pos, hf_pos, atol=1e-6)


@torch.no_grad()
def test_vision_encoder_wrapper_matches_get_image_features(model):
    torch.manual_seed(6)
    px = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model.get_image_features(px, return_dict=True)
    # get_image_features returns seq-first lists with conv_s0/s1 pre-applied.
    want_hr0 = out.fpn_hidden_states[0][:, 0].T.reshape(1, 32, GRID * 4, GRID * 4)
    want_hr1 = out.fpn_hidden_states[1][:, 0].T.reshape(1, 64, GRID * 2, GRID * 2)
    want_top = out.fpn_hidden_states[2][:, 0].T.reshape(1, HIDDEN, GRID, GRID)
    want_pos = out.fpn_position_encoding[2][:, 0].T.reshape(1, HIDDEN, GRID, GRID)
    w = EdgeTamVisionEncoderWrapper(model).eval()
    feats, pos, hr0, hr1 = w(px)
    assert torch.allclose(feats, want_top, atol=1e-6)
    assert torch.allclose(pos, want_pos, atol=1e-6)
    assert torch.allclose(hr0, want_hr0, atol=1e-6)
    assert torch.allclose(hr1, want_hr1, atol=1e-6)


def test_frozen_constants_consistent():
    assert KV_LEN == MAX_MEMORY_MAPS * TOKENS_PER_MAP + PTR_TOKENS == 3648
