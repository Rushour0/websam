"""Tests for manifest building, validation, and file hashing."""

import copy
import hashlib
import json

import pytest

from websam_export.manifest import (
    SCHEMA_VERSION,
    build_manifest,
    sha256_file,
    validate_manifest,
)
from websam_export.spec import SAM3_560, SAM3_1008


@pytest.fixture
def artifact(tmp_path):
    """A fake exported artifact with known bytes."""
    payload = b"onnx-bytes-" * 5000  # multi-chunk-ish, deterministic
    path = tmp_path / "image_encoder.onnx"
    path.write_bytes(payload)
    return path, payload


class TestSha256:
    def test_matches_hashlib_on_same_bytes(self, artifact):
        path, payload = artifact
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_streams_across_chunks(self, artifact):
        path, payload = artifact
        # Tiny chunk size forces many read() iterations through the same data.
        assert sha256_file(path, chunk_size=7) == hashlib.sha256(payload).hexdigest()

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


class TestBuildManifest:
    def test_round_trip_through_json_validates(self, artifact):
        path, payload = artifact
        manifest = build_manifest(SAM3_1008, {"image_encoder": path})
        restored = json.loads(json.dumps(manifest))
        validate_manifest(restored)  # must not raise
        assert restored == manifest  # nothing lossy in the JSON round trip

    def test_core_fields(self, artifact):
        path, payload = artifact
        manifest = build_manifest(SAM3_560, {"image_encoder": path})
        assert manifest["schemaVersion"] == SCHEMA_VERSION == 1
        assert manifest["tier"] == "SAM3_560"
        assert manifest["opset"] == SAM3_560.opset
        assert manifest["constants"]["kvLen"] == 16064
        assert manifest["constants"]["gridSize"] == 40

    def test_file_entry_has_streamed_hash_and_size(self, artifact):
        path, payload = artifact
        manifest = build_manifest(SAM3_1008, {"image_encoder": path})
        entry = manifest["files"]["image_encoder"]
        assert entry["path"] == "image_encoder.onnx"
        assert entry["bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()

    def test_graph_signatures_carry_dims_and_dtype(self, artifact):
        path, _ = artifact
        manifest = build_manifest(SAM3_1008, {"image_encoder": path})
        mem_inputs = manifest["graphs"]["memory_attention"]["inputs"]
        kv = next(t for t in mem_inputs if t["name"] == "memory_kv")
        assert kv["dtype"] == "float32"
        assert kv["dims"][0] == 51904

    def test_toolchain_versions_are_strings(self, artifact):
        path, _ = artifact
        # torch/transformers are not installed in the light test env, so the
        # graceful "not-installed" fallback must kick in rather than raising.
        manifest = build_manifest(SAM3_1008, {"image_encoder": path})
        toolchain = manifest["toolchain"]
        assert set(toolchain) >= {"python", "torch", "transformers", "onnx"}
        assert all(isinstance(v, str) and v for v in toolchain.values())

    def test_missing_artifact_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_manifest(SAM3_1008, {"image_encoder": tmp_path / "nope.onnx"})


class TestValidateManifest:
    @pytest.fixture
    def manifest(self, artifact):
        path, _ = artifact
        return build_manifest(SAM3_1008, {"image_encoder": path})

    def test_accepts_built_manifest(self, manifest):
        validate_manifest(manifest)

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda m: m.__setitem__("schemaVersion", 2), "schemaVersion"),
            (lambda m: m.__setitem__("tier", ""), "tier"),
            (lambda m: m.__setitem__("opset", "20"), "opset"),
            (lambda m: m.__setitem__("toolchain", {}), "toolchain"),
            (lambda m: m["files"]["image_encoder"].__setitem__("sha256", "abc"), "sha256"),
            (lambda m: m["files"]["image_encoder"].__setitem__("bytes", -1), "bytes"),
            (
                lambda m: m["graphs"]["mask_decoder"]["inputs"][0].__setitem__(
                    "dtype", "float64"
                ),
                "dtype",
            ),
            (lambda m: m["graphs"].__setitem__("bad", {"inputs": [], "outputs": []}), "inputs"),
        ],
    )
    def test_rejects_tampered_manifest(self, manifest, mutate, match):
        broken = copy.deepcopy(manifest)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            validate_manifest(broken)

    def test_symbolic_dims_survive_validation(self, manifest):
        # mask_decoder's point_coords has a dynamic "num_points" axis.
        decoder_inputs = manifest["graphs"]["mask_decoder"]["inputs"]
        coords = next(t for t in decoder_inputs if t["name"] == "point_coords")
        assert "num_points" in coords["dims"]
        validate_manifest(manifest)
