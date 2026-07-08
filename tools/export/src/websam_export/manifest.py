"""Build and validate the artifact manifest the browser runtime downloads.

The manifest is plain JSON (schemaVersion 1). It records the tier, the ONNX
opset, the exact toolchain versions the graphs were exported with, every
graph's typed input/output signature, and a streamed SHA-256 per artifact
file so the runtime can verify integrity before creating a session.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Mapping

from .spec import ExportSpec, TensorSpec

SCHEMA_VERSION: int = 1
"""Current manifest schema version; bump on breaking layout changes."""

_TOOLCHAIN_PACKAGES: tuple[str, ...] = ("torch", "transformers", "onnx", "onnxslim", "onnxruntime")

_VALID_DTYPES = {"float32", "float16", "int64", "bool"}


def _dist_version(package: str) -> str:
    """Installed version of ``package``, or ``"not-installed"``.

    Read at runtime via :mod:`importlib.metadata` so the manifest always
    reflects the environment that actually produced the artifacts, and so
    this module stays importable in the light test environment where the
    heavy export extra is absent.
    """
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streamed SHA-256 hex digest of a file (constant memory, 1 MiB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_entry(t: TensorSpec) -> dict[str, Any]:
    return {"name": t.name, "dtype": t.dtype, "dims": list(t.dims)}


def build_manifest(spec: ExportSpec, files: Mapping[str, str | Path]) -> dict[str, Any]:
    """Assemble the schemaVersion-1 manifest for one exported tier.

    :param spec: the tier's :class:`~websam_export.spec.ExportSpec`.
    :param files: artifact key -> path of the produced file on disk
        (e.g. ``{"image_encoder": "out/image_encoder.onnx"}``). Each file is
        hashed with streamed SHA-256.
    :returns: a JSON-serialisable dict; feed it to :func:`validate_manifest`
        before writing.
    :raises FileNotFoundError: if any listed artifact path does not exist.
    """
    file_entries: dict[str, Any] = {}
    for key, raw_path in files.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact {key!r}: no such file {path}")
        file_entries[key] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "schemaVersion": SCHEMA_VERSION,
        "tier": spec.tier,
        "opset": spec.opset,
        "toolchain": {
            "python": platform.python_version(),
            **{pkg: _dist_version(pkg) for pkg in _TOOLCHAIN_PACKAGES},
        },
        "constants": {
            "imageSize": spec.image_size,
            "gridSize": spec.grid_size,
            "tokensPerMemoryMap": spec.tokens_per_memory_map,
            "maxCondFrames": spec.max_cond_frames,
            "numRecent": spec.num_recent,
            "maxMemoryMaps": spec.max_memory_maps,
            "ptrTokens": spec.ptr_tokens,
            "maxObjectPointers": spec.max_object_pointers,
            "kvLen": spec.kv_len,
            "streaming": spec.streaming,
        },
        "graphs": {
            g.name: {
                "inputs": [_tensor_entry(t) for t in g.inputs],
                "outputs": [_tensor_entry(t) for t in g.outputs],
            }
            for g in spec.graphs
        },
        "files": file_entries,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid manifest: {message}")


def _validate_tensor_list(tensors: Any, where: str) -> None:
    _require(isinstance(tensors, list) and tensors, f"{where} must be a non-empty list")
    for t in tensors:
        _require(isinstance(t, dict), f"{where} entries must be objects")
        _require(isinstance(t.get("name"), str) and t["name"], f"{where}: missing tensor name")
        _require(t.get("dtype") in _VALID_DTYPES,
                 f"{where}.{t.get('name')}: bad dtype {t.get('dtype')!r}")
        dims = t.get("dims")
        _require(isinstance(dims, list), f"{where}.{t['name']}: dims must be a list")
        for d in dims:
            _require(
                (isinstance(d, int) and not isinstance(d, bool) and d > 0)
                or (isinstance(d, str) and d != ""),
                f"{where}.{t['name']}: bad dim {d!r}",
            )


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Structurally validate a manifest dict; raises ``ValueError`` on defects.

    Checks schema version, tier/opset presence, toolchain map, per-graph
    typed signatures, and per-file byte counts + 64-hex-char SHA-256 digests.
    Accepts exactly what :func:`build_manifest` produces (including after a
    JSON round-trip).
    """
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schemaVersion") == SCHEMA_VERSION,
             f"schemaVersion must be {SCHEMA_VERSION}")
    _require(isinstance(manifest.get("tier"), str) and bool(manifest["tier"]),
             "tier must be a non-empty string")
    opset = manifest.get("opset")
    _require(isinstance(opset, int) and not isinstance(opset, bool) and opset > 0,
             "opset must be a positive integer")

    toolchain = manifest.get("toolchain")
    _require(isinstance(toolchain, dict) and toolchain, "toolchain must be a non-empty object")
    for name, version in toolchain.items():
        _require(isinstance(version, str) and bool(version),
                 f"toolchain.{name} must be a non-empty string")

    graphs = manifest.get("graphs")
    _require(isinstance(graphs, dict) and graphs, "graphs must be a non-empty object")
    for graph_name, graph in graphs.items():
        _require(isinstance(graph, dict), f"graphs.{graph_name} must be an object")
        _validate_tensor_list(graph.get("inputs"), f"graphs.{graph_name}.inputs")
        _validate_tensor_list(graph.get("outputs"), f"graphs.{graph_name}.outputs")

    files = manifest.get("files")
    _require(isinstance(files, dict), "files must be an object")
    for key, entry in files.items():
        _require(isinstance(entry, dict), f"files.{key} must be an object")
        _require(isinstance(entry.get("path"), str) and bool(entry["path"]),
                 f"files.{key}.path must be a non-empty string")
        size = entry.get("bytes")
        _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                 f"files.{key}.bytes must be a non-negative integer")
        digest = entry.get("sha256")
        _require(
            isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"files.{key}.sha256 must be 64 lowercase hex chars",
        )
