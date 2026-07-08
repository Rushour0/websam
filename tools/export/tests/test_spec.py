"""Tests for the executable export contract in websam_export.spec."""

import dataclasses

import pytest

from websam_export.spec import (
    COND_TPOS_INDEX,
    EDGETAM_1024,
    MAX_COND_FRAMES,
    MAX_MEMORY_MAPS,
    NUM_RECENT,
    PTR_TOKENS,
    SAM3_560,
    SAM3_1008,
    TIERS,
    ExportSpec,
    GraphSpec,
    TensorSpec,
    tpos_index,
)


class TestKvLenArithmetic:
    def test_sam3_1008_kv_len(self):
        assert SAM3_1008.tokens_per_memory_map == 72 * 72 == 5184
        assert SAM3_1008.kv_len == 10 * 5184 + 64 == 51904

    def test_sam3_560_kv_len(self):
        assert SAM3_560.tokens_per_memory_map == 40 * 40 == 1600
        assert SAM3_560.kv_len == 10 * 1600 + 64 == 16064

    def test_edgetam_kv_len(self):
        # Perceiver-compressed: 7 memory frames x 256 latents + 64 ptr tokens.
        assert EDGETAM_1024.kv_len == 7 * 512 + 64 == 3648

    def test_memory_bank_constants(self):
        assert MAX_MEMORY_MAPS == MAX_COND_FRAMES + NUM_RECENT == 10
        assert PTR_TOKENS == 64
        for spec in (SAM3_1008, SAM3_560, EDGETAM_1024):
            assert spec.kv_len == (
                spec.max_memory_maps * spec.tokens_per_memory_map + spec.ptr_tokens
            )
            assert spec.max_memory_maps == spec.max_cond_frames + spec.num_recent
            assert spec.streaming is True

    def test_inconsistent_kv_len_rejected(self):
        with pytest.raises(ValueError, match="kv_len"):
            dataclasses.replace(SAM3_560, kv_len=SAM3_560.kv_len + 1)

    def test_inconsistent_memory_map_count_rejected(self):
        with pytest.raises(ValueError, match="max_memory_maps"):
            dataclasses.replace(
                SAM3_560,
                max_cond_frames=5,
                kv_len=SAM3_560.kv_len,  # maps total unchanged -> 5+6 != 10
            )


class TestTposRule:
    def test_conditioning_maps_to_last_slot(self):
        assert tpos_index(is_conditioning=True) == NUM_RECENT == COND_TPOS_INDEX == 6
        # recent_offset is ignored for conditioning frames.
        assert tpos_index(is_conditioning=True, recent_offset=3) == 6

    @pytest.mark.parametrize("offset,expected", [(1, 0), (2, 1), (5, 4), (6, 5)])
    def test_recent_offset_k_maps_to_k_minus_1(self, offset, expected):
        assert tpos_index(is_conditioning=False, recent_offset=offset) == expected

    @pytest.mark.parametrize("bad", [0, -1, 7, 100])
    def test_out_of_range_offset_rejected(self, bad):
        with pytest.raises(ValueError):
            tpos_index(is_conditioning=False, recent_offset=bad)

    def test_missing_offset_rejected(self):
        with pytest.raises(ValueError):
            tpos_index(is_conditioning=False)

    def test_all_slots_covered_exactly_once(self):
        # 6 recent offsets fill slots 0..5; conditioning fills slot 6.
        slots = {tpos_index(is_conditioning=False, recent_offset=k) for k in range(1, 7)}
        slots.add(tpos_index(is_conditioning=True))
        assert slots == set(range(NUM_RECENT + 1))


class TestSpecStructure:
    def test_tiers_registry(self):
        assert set(TIERS) == {"SAM3_1008", "SAM3_560", "EDGETAM_1024"}
        for key, spec in TIERS.items():
            assert isinstance(spec, ExportSpec)
            assert spec.tier == key

    def test_every_tier_ships_the_four_graphs(self):
        for spec in TIERS.values():
            assert {g.name for g in spec.graphs} == {
                "image_encoder", "memory_attention", "mask_decoder", "memory_encoder",
            }

    def test_memory_attention_kv_dims_match_spec(self):
        for spec in TIERS.values():
            mem_kv = next(
                t for t in spec.graph("memory_attention").inputs if t.name == "memory_kv"
            )
            assert mem_kv.dims[0] == spec.kv_len

    def test_specs_are_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SAM3_1008.kv_len = 0  # type: ignore[misc]

    def test_tensor_spec_rejects_bad_dtype_and_dims(self):
        with pytest.raises(ValueError):
            TensorSpec("x", "float64", (1,))
        with pytest.raises(ValueError):
            TensorSpec("x", "float32", (0,))

    def test_graph_spec_rejects_duplicate_tensor_names(self):
        t = TensorSpec("same", "float32", (1,))
        with pytest.raises(ValueError):
            GraphSpec("g", inputs=(t,), outputs=(t,))
