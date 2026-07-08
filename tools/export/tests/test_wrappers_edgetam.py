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


def pad_kv(mem_seq: torch.Tensor, pos_seq: torch.Tensor, n_maps: int, n_ptr: int,
           fill: float = 0.0):
    """(S,1,64) seq-first HF tensors -> frozen (1,KV_LEN,64) + bias."""
    memory = torch.full((1, KV_LEN, MEM_DIM), fill)
    mem_pos = torch.full((1, KV_LEN, MEM_DIM), fill)
    bias = torch.full((1, 1, 1, KV_LEN), ATTN_BIAS_NEG)
    spatial = n_maps * TOKENS_PER_MAP
    memory[0, :spatial] = mem_seq[:spatial, 0]
    mem_pos[0, :spatial] = pos_seq[:spatial, 0]
    bias[..., :spatial] = 0.0
    base = MAX_MEMORY_MAPS * TOKENS_PER_MAP
    memory[0, base:base + n_ptr] = mem_seq[spatial:, 0]
    mem_pos[0, base:base + n_ptr] = pos_seq[spatial:, 0]
    bias[..., base:base + n_ptr] = 0.0
    return memory, mem_pos, bias


@torch.no_grad()
def test_memory_attention_wrapper_matches_hf(model):
    torch.manual_seed(1)
    n_maps, n_ptr = 3, 8  # 3 real memory maps, 2 pointers x 4 splits
    seq = n_maps * TOKENS_PER_MAP + n_ptr
    feats_seq = torch.randn(GRID * GRID, 1, HIDDEN)
    pos_seq = torch.randn(GRID * GRID, 1, HIDDEN)
    mem_seq = torch.randn(seq, 1, MEM_DIM)
    mem_pos_seq = torch.randn(seq, 1, MEM_DIM)

    hf_out = model.memory_attention(
        current_vision_features=feats_seq,
        current_vision_position_embeddings=pos_seq,
        memory=mem_seq,
        memory_posision_embeddings=mem_pos_seq,
        num_object_pointer_tokens=n_ptr,
        num_spatial_memory_tokens=n_maps,
    )
    hf_bchw = hf_out.squeeze(1).transpose(1, 2).reshape(1, HIDDEN, GRID, GRID)

    w = EdgeTamMemoryAttentionWrapper(model).eval()
    memory, mem_pos, bias = pad_kv(mem_seq, mem_pos_seq, n_maps, n_ptr)
    got = w(
        feats_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous(),
        pos_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous(),
        memory, mem_pos, bias,
    )
    assert torch.allclose(got, hf_bchw, atol=1e-5), (got - hf_bchw).abs().max()


@torch.no_grad()
def test_memory_attention_padding_is_inert(model):
    torch.manual_seed(2)
    n_maps, n_ptr = 2, 4
    seq = n_maps * TOKENS_PER_MAP + n_ptr
    feats = torch.randn(1, HIDDEN, GRID, GRID)
    pos = torch.randn(1, HIDDEN, GRID, GRID)
    mem_seq = torch.randn(seq, 1, MEM_DIM)
    mem_pos_seq = torch.randn(seq, 1, MEM_DIM)
    w = EdgeTamMemoryAttentionWrapper(model).eval()
    clean = w(feats, pos, *pad_kv(mem_seq, mem_pos_seq, n_maps, n_ptr, fill=0.0))
    poisoned = w(feats, pos, *pad_kv(mem_seq, mem_pos_seq, n_maps, n_ptr, fill=1e3))
    assert torch.equal(clean, poisoned)


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
    high_res_masks = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE) * 8
    hf_feats, hf_pos = model._encode_new_memory(
        current_vision_feats=feats_seq,
        pred_masks_high_res=high_res_masks,
        object_score_logits=torch.tensor([[10.0]]),
        is_mask_from_pts=prompted,
    )
    w = EdgeTamMemoryEncoderWrapper(model).eval()
    feats_bchw = feats_seq[:, 0].T.reshape(1, HIDDEN, GRID, GRID).contiguous()
    got_feats, got_pos = w(feats_bchw, high_res_masks,
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
