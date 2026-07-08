"""Executable export contract for the WebSAM ONNX pipeline.

This module is the single source of truth for the shapes the browser runtime
compiles against. Every constant here is derived from the reference HF
implementation (``transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py``)
and from the EdgeTAM release, and every derived quantity is re-checked in
``__post_init__`` so a typo cannot silently ship a wrong graph.

Memory-bank model (SAM3 tracker video, streaming semantics)
-----------------------------------------------------------
``modeling_sam3_tracker_video.py`` maintains, per tracked object:

* up to ``MAX_COND_FRAMES`` (4) *conditioning* frames — frames where the user
  provided prompts; their memory maps are always kept;
* the ``NUM_RECENT`` (6) most recent non-conditioning frames — a sliding
  window updated every frame (streaming semantics: frame N's memory attention
  reads only memories produced at frames < N, then frame N's own encoded
  memory is pushed and the oldest recent entry evicted);
* up to ``MAX_OBJECT_POINTERS`` (16) object-pointer vectors, each projected to
  ``mem_dim``-sized tokens; the pointer bank contributes a fixed
  ``PTR_TOKENS`` (64) key/value tokens after projection.

Memory attention therefore sees a key/value sequence of
``MAX_MEMORY_MAPS * tokens_per_memory_map + PTR_TOKENS`` tokens, where
``MAX_MEMORY_MAPS = MAX_COND_FRAMES + NUM_RECENT = 10`` and
``tokens_per_memory_map = grid**2`` (spatial memory features flattened).

For ONNX export we freeze this to the *maximum* KV length and drive validity
with an attention mask, so the graph shape is static per tier:

* ``SAM3_1008``: 72x72 grid  -> ``10 * 5184 + 64 = 51904``
* ``SAM3_560``:  40x40 grid  -> ``10 * 1600 + 64 = 16064``
* ``EDGETAM_1024``: the perceiver resampler compresses each memory frame to
  256 latents and keeps 7 memory frames (1 conditioning + 6 recent), so
  ``7 * 256 + 64 = 1856``.

Temporal position (tpos) rule
-----------------------------
The memory attention adds one of ``NUM_RECENT + 1`` learned temporal
embeddings to each memory map. In ``modeling_sam3_tracker_video.py`` the index
is computed from the frame-age of the memory:

* a **conditioning** frame always uses the *last* slot, index ``NUM_RECENT``
  (i.e. 6 for both SAM3 tiers);
* a **recent** frame at offset ``k`` (``k = 1`` is the immediately previous
  frame, ``k = NUM_RECENT`` the oldest kept) uses index ``k - 1``.

That mapping is encoded in :func:`tpos_index` and unit-tested, because getting
it wrong produces masks that look plausible on frame 1 and drift afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Corrected memory-bank constants (see module docstring for the derivation).
# ---------------------------------------------------------------------------

MAX_COND_FRAMES: int = 4
"""Maximum prompted (conditioning) frames whose memories are always retained."""

NUM_RECENT: int = 6
"""Sliding window of most-recent non-conditioning frame memories."""

MAX_MEMORY_MAPS: int = MAX_COND_FRAMES + NUM_RECENT
"""Total spatial memory maps visible to memory attention (4 + 6 = 10)."""

PTR_TOKENS: int = 64
"""Key/value tokens contributed by the projected object-pointer bank."""

MAX_OBJECT_POINTERS: int = 16
"""Maximum object-pointer vectors kept before projection to PTR_TOKENS tokens."""

COND_TPOS_INDEX: int = NUM_RECENT
"""Temporal-position embedding slot used by conditioning-frame memories (6)."""


def tpos_index(*, is_conditioning: bool, recent_offset: int | None = None,
               num_recent: int = NUM_RECENT) -> int:
    """Map a memory entry to its temporal-position embedding index.

    Implements the rule from ``modeling_sam3_tracker_video.py``:
    conditioning-frame memories use index ``num_recent`` (the dedicated last
    slot); a recent-frame memory at offset ``k`` (1 = previous frame) uses
    index ``k - 1``.

    :param is_conditioning: True for a prompted (conditioning) frame memory.
    :param recent_offset: 1-based age of a recent memory; required when
        ``is_conditioning`` is False, must satisfy ``1 <= k <= num_recent``.
    :param num_recent: size of the recent window (6 for all shipped tiers).
    :raises ValueError: if ``recent_offset`` is missing or out of range.
    """
    if is_conditioning:
        return num_recent
    if recent_offset is None:
        raise ValueError("recent_offset is required for non-conditioning memories")
    if not 1 <= recent_offset <= num_recent:
        raise ValueError(
            f"recent_offset must be in [1, {num_recent}], got {recent_offset}"
        )
    return recent_offset - 1


# ---------------------------------------------------------------------------
# Spec dataclasses.
# ---------------------------------------------------------------------------

Dim = int | str
"""A tensor dimension: a fixed int, or a symbolic name for a dynamic axis."""


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One named input or output tensor of an exported ONNX graph."""

    name: str
    """ONNX value name, stable across exports (the runtime binds by name)."""

    dtype: str
    """Element type: ``"float32"``, ``"float16"``, ``"int64"``, or ``"bool"``."""

    dims: tuple[Dim, ...]
    """Shape; ints are frozen at export time, strings are dynamic axes."""

    doc: str = ""
    """Human-readable description carried into the manifest."""

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float16", "int64", "bool"}:
            raise ValueError(f"unsupported dtype {self.dtype!r} for {self.name!r}")
        for d in self.dims:
            if isinstance(d, int) and d <= 0:
                raise ValueError(f"non-positive fixed dim {d} in {self.name!r}")


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """One exported ONNX graph (a single ``.onnx`` file) and its signature."""

    name: str
    """Artifact key, e.g. ``"image_encoder"`` or ``"memory_attention"``."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    doc: str = ""

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError(f"graph {self.name!r} must declare inputs and outputs")
        names = [t.name for t in (*self.inputs, *self.outputs)]
        if len(names) != len(set(names)):
            raise ValueError(f"graph {self.name!r} has duplicate tensor names")


@dataclass(frozen=True, slots=True)
class ExportSpec:
    """Complete export contract for one model tier.

    Frozen and self-validating: ``kv_len`` must equal
    ``max_memory_maps * tokens_per_memory_map + ptr_tokens`` exactly, so the
    dataclass cannot be constructed with inconsistent memory-bank arithmetic.
    """

    tier: str
    """Tier key, e.g. ``"SAM3_1008"``; also the manifest ``tier`` field."""

    image_size: int
    """Square input resolution fed to the image encoder."""

    grid_size: int
    """Spatial memory grid edge (``image_size / patch_stride``); 0 tokens-per-
    map tiers (EdgeTAM) instead set :attr:`tokens_per_memory_map` directly."""

    tokens_per_memory_map: int
    """KV tokens contributed per memory map: ``grid_size**2`` for SAM3 tiers,
    the perceiver latent count (256) for EdgeTAM."""

    max_cond_frames: int
    num_recent: int
    max_memory_maps: int
    ptr_tokens: int
    max_object_pointers: int

    kv_len: int
    """Frozen memory-attention KV length the ONNX graph is exported with."""

    streaming: bool = True
    """Streaming semantics: frame N attends only to memories from frames < N;
    its own encoded memory is appended after decoding (see module docstring)."""

    opset: int = 20
    """ONNX opset the tier's graphs target."""

    graphs: tuple[GraphSpec, ...] = field(default=())
    """Signatures of every ``.onnx`` artifact this tier ships."""

    def __post_init__(self) -> None:
        expected = self.max_memory_maps * self.tokens_per_memory_map + self.ptr_tokens
        if self.kv_len != expected:
            raise ValueError(
                f"{self.tier}: kv_len {self.kv_len} != "
                f"{self.max_memory_maps} * {self.tokens_per_memory_map} "
                f"+ {self.ptr_tokens} = {expected}"
            )
        if self.max_memory_maps != self.max_cond_frames + self.num_recent:
            raise ValueError(
                f"{self.tier}: max_memory_maps {self.max_memory_maps} != "
                f"max_cond_frames + num_recent"
            )

    def graph(self, name: str) -> GraphSpec:
        """Look up a graph by artifact key; raises ``KeyError`` if absent."""
        for g in self.graphs:
            if g.name == name:
                return g
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Shared graph signatures (dims parameterised per tier).
# ---------------------------------------------------------------------------

def _sam3_graphs(image_size: int, grid: int, kv_len: int) -> tuple[GraphSpec, ...]:
    """Standard four-graph split used by both SAM3 tiers.

    Mirrors the module boundaries of ``modeling_sam3_tracker_video.py``:
    vision encoder / prompt+mask decoder / memory encoder / memory attention.
    Hidden sizes (256 embed, 64 mem_dim) follow the HF config defaults.
    """
    tokens = grid * grid
    return (
        GraphSpec(
            name="image_encoder",
            inputs=(TensorSpec("pixel_values", "float32", (1, 3, image_size, image_size),
                               "Normalised RGB frame."),),
            outputs=(
                TensorSpec("vision_features", "float32", (1, 256, grid, grid),
                           "FPN top-level features consumed by memory attention."),
                TensorSpec("vision_pos_embed", "float32", (1, 256, grid, grid),
                           "Positional encoding for the vision features."),
                TensorSpec("high_res_features_0", "float32", (1, 32, grid * 4, grid * 4),
                           "Stride-4 skip features for the mask decoder."),
                TensorSpec("high_res_features_1", "float32", (1, 64, grid * 2, grid * 2),
                           "Stride-8 skip features for the mask decoder."),
            ),
            doc="Per-frame vision backbone + FPN neck; run once per frame.",
        ),
        GraphSpec(
            name="memory_attention",
            inputs=(
                TensorSpec("current_vision_features", "float32", (tokens, 1, 256),
                           "Flattened current-frame features (queries)."),
                TensorSpec("current_vision_pos_embed", "float32", (tokens, 1, 256)),
                TensorSpec("memory_kv", "float32", (kv_len, 1, 64),
                           "Concatenated memory maps + projected object pointers, "
                           "padded to the frozen maximum."),
                TensorSpec("memory_pos_embed", "float32", (kv_len, 1, 64),
                           "Spatial + tpos embeddings (tpos rule: cond->6, "
                           "recent offset k -> k-1)."),
                TensorSpec("memory_mask", "bool", (1, kv_len),
                           "True for valid KV positions; padding is masked out."),
            ),
            outputs=(TensorSpec("conditioned_features", "float32", (tokens, 1, 256),
                                "Memory-conditioned features for the decoder."),),
            doc="Cross-attention over the frozen-length memory bank.",
        ),
        GraphSpec(
            name="mask_decoder",
            inputs=(
                TensorSpec("conditioned_features", "float32", (1, 256, grid, grid)),
                TensorSpec("high_res_features_0", "float32", (1, 32, grid * 4, grid * 4)),
                TensorSpec("high_res_features_1", "float32", (1, 64, grid * 2, grid * 2)),
                TensorSpec("point_coords", "float32", (1, "num_points", 2),
                           "Prompt points in pixel space (dynamic axis)."),
                TensorSpec("point_labels", "int64", (1, "num_points"),
                           "1=positive, 0=negative, 2/3=box corners, -1=pad."),
            ),
            outputs=(
                TensorSpec("low_res_masks", "float32", (1, 4, grid * 4, grid * 4)),
                TensorSpec("iou_scores", "float32", (1, 4)),
                TensorSpec("object_pointer", "float32", (1, 256),
                           "Pointer vector pushed into the object-pointer bank."),
                TensorSpec("object_score_logits", "float32", (1, 1)),
            ),
            doc="Prompt encoder + two-way transformer mask decoder.",
        ),
        GraphSpec(
            name="memory_encoder",
            inputs=(
                TensorSpec("vision_features", "float32", (1, 256, grid, grid)),
                TensorSpec("mask_for_memory", "float32", (1, 1, image_size, image_size),
                           "Predicted mask (sigmoid-scaled) fused into memory."),
            ),
            outputs=(
                TensorSpec("memory_features", "float32", (1, 64, grid, grid),
                           "One memory map (grid**2 tokens after flattening)."),
                TensorSpec("memory_pos_embed", "float32", (1, 64, grid, grid)),
            ),
            doc="Encodes the decoded frame into one memory-bank entry.",
        ),
    )


# ---------------------------------------------------------------------------
# Tiers.
# ---------------------------------------------------------------------------

SAM3_1008 = ExportSpec(
    tier="SAM3_1008",
    image_size=1008,
    grid_size=72,
    tokens_per_memory_map=72 * 72,          # 5184
    max_cond_frames=MAX_COND_FRAMES,        # 4
    num_recent=NUM_RECENT,                  # 6
    max_memory_maps=MAX_MEMORY_MAPS,        # 10
    ptr_tokens=PTR_TOKENS,                  # 64
    max_object_pointers=MAX_OBJECT_POINTERS,  # 16
    kv_len=10 * 5184 + 64,                  # 51904
    streaming=True,
    graphs=_sam3_graphs(1008, 72, 10 * 5184 + 64),
)
"""Full-quality SAM3 tier: 1008px input, 14px patch stride -> 72x72 memory grid.

Derivation (``modeling_sam3_tracker_video.py`` memory-bank assembly): 4
conditioning + 6 recent memory maps of 5184 tokens each, plus 64 projected
object-pointer tokens -> KV length 51904. tpos rule: conditioning memories use
embedding index 6, a recent memory at offset k uses index k-1.
"""

SAM3_560 = ExportSpec(
    tier="SAM3_560",
    image_size=560,
    grid_size=40,
    tokens_per_memory_map=40 * 40,          # 1600
    max_cond_frames=MAX_COND_FRAMES,
    num_recent=NUM_RECENT,
    max_memory_maps=MAX_MEMORY_MAPS,
    ptr_tokens=PTR_TOKENS,
    max_object_pointers=MAX_OBJECT_POINTERS,
    kv_len=10 * 1600 + 64,                  # 16064
    streaming=True,
    graphs=_sam3_graphs(560, 40, 10 * 1600 + 64),
)
"""Reduced-resolution SAM3 tier: 560px input -> 40x40 grid, KV length 16064.

Same memory-bank structure and tpos rule as :data:`SAM3_1008`; only the
spatial token count per memory map shrinks (1600 vs 5184).
"""

EDGETAM_1024 = ExportSpec(
    tier="EDGETAM_1024",
    image_size=1024,
    grid_size=64,
    tokens_per_memory_map=256,              # perceiver latents per memory frame
    max_cond_frames=1,
    num_recent=NUM_RECENT,
    max_memory_maps=7,                      # 1 cond + 6 recent
    ptr_tokens=PTR_TOKENS,
    max_object_pointers=MAX_OBJECT_POINTERS,
    kv_len=7 * 256 + 64,                    # 1856
    streaming=True,
    graphs=_sam3_graphs(1024, 64, 7 * 256 + 64),
)
"""EdgeTAM tier: 1024px input; the 2D perceiver resampler compresses each
64x64 memory map to 256 latent tokens, and the bank keeps 1 conditioning + 6
recent frames -> KV length ~= 7*256 + 64 = 1856. Same tpos rule as SAM3
(conditioning -> last slot, recent offset k -> k-1)."""


TIERS: dict[str, ExportSpec] = {
    "SAM3_1008": SAM3_1008,
    "SAM3_560": SAM3_560,
    "EDGETAM_1024": EDGETAM_1024,
}
"""All shipped export tiers, keyed by tier name (also the manifest ``tier``)."""
