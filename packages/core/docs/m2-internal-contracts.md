# M2 Internal Contracts — EdgeTAM Video Path

Normative design for the M2 implementation waves. Eight agents implement disjoint file sets
against the signatures below; any deviation is integration drift and must come back to this doc
first. Public API (`segmenter.ts` — `VideoSession`/`FramePropagationResult` are already frozen —
`errors.ts`, `coords.ts`, `masks/*`, `backend/backend.ts` existing method signatures) stays
frozen; M2 adds bodies, internal modules, and the explicitly sanctioned additions in §1.1, §2.1,
§4.1 and §7.1. All paths relative to `packages/core/` unless rooted.

**Model-agnostic rule (inherited from M1):** every shape, tensor name, and semantic constant
comes from the manifest (§2) or a per-arch strategy object (§3.1) — never a TS constant. The
EdgeTAM export spike (`tools/export/spikes/m2-edgetam/FINDINGS.md`, in flight, may not exist yet)
pins the values marked **⚠ PIN-n**; §10 is the exhaustive pin list. No `src/` design changes when
FINDINGS.md lands — only manifest values emitted by `tools/export` — that is the design invariant.

---

## 0. File ownership (no two agents share a file) + waves

| Agent | Wave | Creates | Modifies |
|---|---|---|---|
| **backend-video** | 1 | `src/backend/memory-primitives.test.ts`, `src/backend/memory-primitives.browser.test.ts` | `src/backend/backend.ts` (§1.1 sanctioned doc/optional-member additions only), `src/backend/webgpu-backend.ts`, `src/backend/wasm-backend.ts` (real `allocTensor('device')` + `copyRegion` + `debugStats`) |
| **memory-bank** | 1 | `src/worker/video/arch-strategy.ts`, `src/worker/video/memory-bank.ts`, `src/worker/video/*.test.ts` (bank+strategy only) | `src/weights/manifest.ts` (additive `video` section, §2.1) + `src/weights/manifest.test.ts` |
| **frame-source** | 1 | `src/video/frame-source.ts`, `src/video/webcodecs-source.ts`, `src/video/webcodecs-source.browser.test.ts`, `src/video/fixtures/clip-320x180-10f.mp4` | `packages/core/package.json` (adds `mp4box` dep ONLY — no other field) |
| **editing** | 1 | `packages/video-editing/src/exporter.test.ts`, `exporter.browser.test.ts` | `packages/video-editing/src/exporter.ts`, `timeline.ts` (adds `collect`), `index.ts` (exports), `package.json` (adds `fflate`) |
| **worker-video** | 2 | `src/worker/video/video-engine.ts`, `src/worker/video/propagation-port.ts`, `src/worker/video/video-engine.test.ts`, `src/worker/video/propagation-port.test.ts` | `src/worker/protocol.ts` (additive, §5 verbatim), `src/worker/engine.ts` (video methods), `src/worker/preprocess.ts` (`bitmapToTensor` accepts `VideoFrame`) |
| **video-session** | 2 | `src/sessions/video-session.ts`, `src/sessions/propagation-iterator.ts`, `src/sessions/video-session.test.ts`, `src/sessions/propagation-iterator.test.ts` | `src/sessions/segmenter-impl.ts` (`createVideoSession` body; `DEFAULT_MODEL_ID` flips to `'edgetam'`), `src/sessions/spawn-worker.ts` (additive `RemoteEngine` methods) |
| **demo** | 3 | `apps/demo/src/VideoTab.tsx` | `apps/demo/src/App.tsx` (tab), `apps/demo/src/App.css` |
| **e2e** | 3 | `src/e2e/video-golden.browser.test.ts`, `tools/goldens/make-video-golden.py`, `tools/goldens/fixtures/video/*` | `packages/core/vitest.config.ts` (only if fs.allow needs widening), `tools/goldens/package.json` (if needed) |

Ownership decisions (explicit):
- `src/index.ts`, `tsdown.config.ts` — **untouched at M2**. No new package entries; no new public
  exports (the video surface already exists on `Segmenter`/`VideoSession`).
- `src/backend/*` → backend-video only. `src/weights/*` → memory-bank only (one file).
- `packages/core/package.json` → frame-source only (single-line dep add; conflicts impossible).
- Wave-2 cross-import (video-session → `src/worker/protocol.ts` types) is pinned **verbatim** by
  §5, same pattern as M1 §3.1; the wave-2 gate (typecheck+build) catches drift.
- Waves gate: after each wave `pnpm -F @websam/core build && pnpm -F @websam/core test` (+
  `-F @websam/video-editing` after wave 1); wave 3 adds `test:browser` + demo build.

Dependency order: wave 1 items are mutually disjoint. worker-video depends on backend-video +
memory-bank + frame-source; video-session depends on protocol §5 (types only). demo + e2e depend
on wave 2.

---

## 1. Backend: the two M2 primitives become real (backend-video)

### 1.1 Sanctioned `src/backend/backend.ts` changes (doc + one optional member; no signature edits)

1. **`copyRegion` slot-shape rule relaxed (TSDoc only):** `src` must have exactly
   `dst.shape.slice(1)`'s element count and the same dtype; the copy is a contiguous byte copy
   (reshape-free). Rationale: the engine copies a whole per-object KV ring (`[S, T, D]`) into one
   batch slot of a `[B, S*T, D]` graph input (§4.4); requiring literal shape equality would force
   pointless reshapes on an opaque handle.
2. **Optional debug census (new optional member, additive):**

```ts
/** Live-resource census for leak gates. Optional; browser backends implement it in M2. */
debugStats?(): { liveTensors: number; liveBytes: number };
```

### 1.2 Implementations

- `WebGpuBackend.allocTensor(shape, dtype, 'device')`: real — allocate a zeroed `GPUBuffer`
  through ort's device (read out post-init per ⚠REV ORT#26107), wrap via
  `ort.Tensor.fromGpuBuffer`, register in the backend's tensor census. Throws
  `OutOfMemoryError` on allocation failure.
- `WebGpuBackend.copyRegion(src, dst, slotIndex)`: `commandEncoder.copyBufferToBuffer` of
  `slotBytes = dst.bytes / dst.shape[0]` at offset `slotIndex * slotBytes`; validates dtype match,
  slot bounds, and the §1.1 byte-count rule (`InvalidStateError` otherwise). Device↔device only;
  a `'cpu'`-located operand on the webgpu backend is `InvalidStateError` (upload first).
- `WasmBackend`: both real on CPU — `allocTensor('device')` degrades to a zeroed cpu tensor
  (documented: on wasm, `'device'` === `'cpu'`), `copyRegion` is a `TypedArray.prototype.set`.
- `debugStats()`: both backends count every non-disposed tensor they created (alloc/upload/run
  outputs) — this is the browser-e2e flatness witness (§9.4).

---

## 2. Manifest: the `video` section + video graph roles (memory-bank)

### 2.1 `src/weights/manifest.ts` additions (additive; `schemaVersion` stays 1)

```ts
export type KnownGraphRole =
  | 'visionEncoder' | 'promptDecoder'                                   // M1 image path
  | 'videoEncoder' | 'memoryAttention' | 'maskDecoderVideo' | 'memoryEncoder'; // M2

/** Memory-loop constants pinned by tools/export (mirrors websam_export.spec.ExportSpec). */
export interface VideoManifestSection {
  maxCondFrames: number;        // EdgeTAM: 1
  numRecent: number;            // 6
  tokensPerMemoryMap: number;   // 256 (perceiver latents)      ⚠ PIN-1
  ptrTokens: number;            // 64
  maxObjectPointers: number;    // 16
  kvLen: number;                // 1856; parse-validated == maps*tokens + ptrTokens
  memDim: number;               // 64
  embedDim: number;             // 256 (object-pointer width)
  gridSize: number;             // 64 (imageSize / patch stride)
  /** Memory graphs exported with a symbolic leading batch dim (objects = batch). ⚠ PIN-4 */
  multiObjectBatch: boolean;
  /** How the init (no-memory) frame is conditioned. ⚠ PIN-2 */
  initPath: 'noMemFlag' | 'noMemGraph';
  /** How temporal-position embeddings reach the graph. ⚠ PIN-3 */
  tposDelivery: 'indices' | 'precombined';
  /** object_score_logits threshold below which the object counts as occluded. ⚠ PIN-8 */
  occlusionThreshold: number;
}

export interface ModelManifest {
  // ...existing fields unchanged...
  /** Present iff the tier ships video graphs. */
  video?: VideoManifestSection;
}
```

`parseModelManifest` validates the section when present (all counts positive ints, the `kvLen`
identity, enum fields) and cross-checks that all four video roles exist in `graphs` when `video`
is set — else `WeightVerifyError`.

### 2.2 Required semantic keys (video roles; ONNX names ⚠ PIN-1, keys normative)

| Role | inputs (semantic key) | outputs |
|---|---|---|
| `videoEncoder` | `pixels` `[B,3,S,S]` f32 | `visionFeatures` `[B,256,g,g]`, `visionPos` `[B,256,g,g]`, `highRes0` `[B,32,4g,4g]`, `highRes1` `[B,64,2g,2g]` |
| `memoryAttention` | `queries` `[B,g*g,256]`, `queriesPos` `[B,g*g,256]`, `memorySpatial` `[B,M,T,64]`, `memorySpatialPos` `[B,M,T,64]`, `tposIndices` i64 `[B,M]` (iff `tposDelivery:'indices'`), `memoryMask` bool `[B,kvLen]`, `objectPointers` f32 `[B,P,256]`, `pointerDeltas` i64 `[B,P]`, `pointerMask` bool `[B,P]`, `noMem` bool `[B]` (iff `initPath:'noMemFlag'`) | `conditionedFeatures` `[B,g*g,256]` |
| `maskDecoderVideo` | `conditionedFeatures`, `highRes0`, `highRes1`, `points` f32 `[B,1,P8,2]`, `labels` i64 `[B,1,P8]` (P8 = 8 padded points, label −1 = pad) | `maskLogits` `[B,C,4g,4g]`, `iouScores` `[B,C]`, `objectPointer` `[B,256]`, `objectScoreLogits` `[B,1]` |
| `memoryEncoder` | `visionFeatures` `[B,256,g,g]`, `maskLogits` `[B,1,4g,4g]` (low-res; sigmoid+resize IN-graph ⚠ PIN-7) | `memoryFeatures` `[B,T,64]`, `memoryPos` `[B,T,64]` (post-perceiver, flattened to T tokens ⚠ PIN-1) |

`M = maxMemoryMaps`, `T = tokensPerMemoryMap`, `P = maxObjectPointers`, `g = gridSize`,
`B = num_objects` when `multiObjectBatch`, else 1. The attention mask enters as **bool**; the
additive `-1e4` fp16-safe bias is built in-graph (plan ⚠REV) — JS never assembles bias values,
only validity bits ("attn-bias assembly" == mask-bit assembly here).

---

## 3. `src/worker/video/` — arch strategy + memory bank (memory-bank agent)

### 3.1 `src/worker/video/arch-strategy.ts`

Per-arch semantics live in ONE strategy object so SAM3 lands at M3 by adding a branch, not
touching the bank/engine. Mirrors the executable spec (`tools/export/src/websam_export/spec.py`).

```ts
import type { ModelSpec } from '../../registry.js';
import type { VideoManifestSection } from '../../weights/manifest.js';

export interface MemoryEntryRef { frameIdx: number; isCond: boolean }

export interface VideoArchStrategy {
  readonly arch: ModelSpec['arch'];
  /**
   * Temporal-position embedding index for one memory entry (spec.py tpos_index):
   * cond → numRecent; recent at offset k (1 = most recent valid entry) → k-1.
   * `recentOffset` is the 1-based RECENCY RANK among valid recent slots
   * (descending frameIdx), not the raw frame distance.            ⚠ PIN-5
   */
  tposIndex(entry: { isCond: boolean; recentOffset?: number }): number;
  /**
   * Which cond frames stay when the region overflows / at assembly, replicating
   * HF `_select_closest_cond_frames` tie-breaking exactly (EdgeTAM max=1 → the
   * single closest to `currentFrame`; ties break toward the LOWER frameIdx). ⚠ PIN-6
   */
  selectCondFrames(condFrames: readonly number[], currentFrame: number, max: number): number[];
  /**
   * Streaming pointer deltas (`currentFrame - ptrFrame`), most-recent-first,
   * zero-padded to maxObjectPointers; int64 because it is a graph input.  ⚠ PIN-9
   */
  pointerTimeDeltas(ptrFrames: readonly number[], currentFrame: number): BigInt64Array;
  /** Whether an occluded frame's memory is still committed to the bank. ⚠ PIN-8 */
  readonly commitOccludedMemory: boolean;
}

export function strategyFor(arch: ModelSpec['arch'], video: VideoManifestSection): VideoArchStrategy;
```

Unit tests assert `tposIndex` against the exact case table of `spec.py::tpos_index` (both valid
and throwing inputs) so JS and the export pipeline share one executable spec.

### 3.2 `src/worker/video/memory-bank.ts`

Generic over arch; PER-OBJECT instance (one bank per tracked object). Storage goes through the
`Backend` interface exclusively: on webgpu the rings are `allocTensor(..., 'device')` GPUBuffers
written with `copyRegion`; on wasm they are cpu tensors — the bank cannot tell (`location` is
`backend.kind === 'webgpu' ? 'device' : 'cpu'`, decided by the engine).

```ts
import type { Backend, DeviceTensor } from '../../backend/backend.js';
import type { VideoManifestSection } from '../../weights/manifest.js';
import type { VideoArchStrategy } from './arch-strategy.js';

export interface MemorySlotMeta {
  frameIdx: number;   // -1 when invalid
  isCond: boolean;
  valid: boolean;
}

export interface MemoryAssembly {
  /** Persistent rings, BORROWED (never dispose): [M, T, memDim]. */
  memorySpatial: DeviceTensor;
  memorySpatialPos: DeviceTensor;
  /** [M] per-slot tpos indices (strategy.tposIndex), −1 for invalid slots. */
  tposIndices: BigInt64Array;
  /** [kvLen] validity bits (spatial region per slot, then pointer region). */
  memoryMask: Uint8Array;
  /** [P, embedDim] zero-padded pointer bank (cpu-side, uploaded per frame). */
  objectPointers: Float32Array;
  pointerDeltas: BigInt64Array;   // [P]
  pointerMask: Uint8Array;        // [P]
  /** Count of valid spatial maps (0 → engine takes the init/no-mem path). */
  validMaps: number;
}

export class MemoryBank {
  constructor(init: {
    backend: Backend;
    video: VideoManifestSection;
    strategy: VideoArchStrategy;
    location: 'device' | 'cpu';
  });
  /** Slot layout: [0, maxCondFrames) = cond region; [maxCondFrames, M) = recent ring. */
  readonly slots: readonly MemorySlotMeta[];
  /** True iff any slot is valid (assemble would yield validMaps > 0). */
  hasMemory(currentFrameIdx: number): boolean;
  /**
   * Commit one frame's encoded memory. `memoryFeatures`/`memoryPos` are the
   * memoryEncoder outputs for THIS object ([T, memDim] after batch-slicing);
   * the bank copyRegions them into the chosen slot and does NOT take ownership
   * (caller disposes per §4.5). Slot choice:
   *   cond   → first invalid cond slot, else evict per strategy.selectCondFrames
   *            (keep the winners; the loser slot is overwritten);
   *   recent → first invalid recent slot, else the recent slot with the
   *            smallest frameIdx (oldest).
   */
  commit(frameIdx: number, isCond: boolean, memoryFeatures: DeviceTensor, memoryPos: DeviceTensor): void;
  /** Push an object pointer (cpu Float32Array[embedDim]); ring of maxObjectPointers, oldest evicted. */
  commitPointer(frameIdx: number, pointer: Float32Array): void;
  /**
   * Build the frame-N attention feeds. STREAMING RULE: only slots with
   * frameIdx < currentFrameIdx count as valid (frame N never sees its own
   * memory); assembly happens BEFORE commit each frame. Physical slot order is
   * NOT temporal order — memory attention is permutation-invariant given
   * correct per-slot pos/tpos and mask bits, so slots are bound in place and a
   * frame step performs ZERO spatial copies beyond the one commit.
   */
  assemble(currentFrameIdx: number): MemoryAssembly;
  /** Refine support: invalidate every NON-cond slot and pointer with frameIdx > frameIdx. */
  invalidateAfter(frameIdx: number): void;
  reset(): void;   // all slots/pointers invalid; rings retained
  dispose(): void; // rings disposed; further use → InvalidStateError
}
```

Memory cost check (EdgeTAM, fp32): 2 rings × 7 × 256 × 64 × 4 B ≈ 917 KB per object — flat for
the whole session. Pointer bank stays cpu-side (16×256 f32 = 16 KB; uploaded per frame, negligible)
— deliberate: pointers are graph inputs anyway (`pointerDeltas` is data-dependent) so a device
ring buys nothing.

`tposDelivery: 'precombined'` fallback: `assemble` instead returns a cpu `Float32Array` of
spatial-pos + tpos-embedding sums for upload (requires the manifest to carry the tpos table as a
small weight file). The `'indices'` path is strongly preferred (per-frame upload cost at SAM3 1008
would be ~13 MB/frame otherwise) — the bank implements `'indices'`; `'precombined'` throws
`NotImplementedError` at M2 unless the spike forces it. ⚠ PIN-3

---

## 4. `src/worker/video/video-engine.ts` + per-frame loop (worker-video)

### 4.1 Construction

```ts
export interface VideoEngineInit {
  backend: Backend;
  manifest: ModelManifest;              // manifest.video is required (InvalidStateError otherwise)
  spec: ModelSpec;
  graphs: Record<'videoEncoder' | 'memoryAttention' | 'maskDecoderVideo' | 'memoryEncoder', BackendSession>;
}

export class VideoEngine {
  constructor(init: VideoEngineInit);
  attach(source: FrameSource, info: VideoSourceInfo): void;   // computes the ONE CoordinateTransform
  addObject(req: { frameIndex: number; prompts: Prompt[]; objectId?: number }): Promise<VideoObjectResult>;
  refineObject(req: { objectId: number; frameIndex: number; prompts: Prompt[] }): Promise<VideoObjectResult>;
  removeObject(objectId: number): void;
  /**
   * Run the propagation loop; `emit` resolves when the frame may be sent
   * (credit available — see §5.2), `isCancelled` is polled between frames.
   */
  propagate(
    req: { startFrame: number; endFrame: number; epoch: number },
    emit: (frame: PropagationFrame) => Promise<void>,
    isCancelled: () => boolean,
  ): Promise<void>;
  reset(): void;
  dispose(): Promise<void>;
}
```

The engine holds: the shared `FrameSource`, one `CoordinateTransform` (video frames never change
dims — computed once at `attach`, stamped on every payload), an insertion-ordered
`Map<objectId, { bank: MemoryBank; insertionIndex: number }>`, and the fixed batched graph-input
tensors (§4.4). Object ids auto-assign from 1 when omitted; duplicate explicit id →
`InvalidStateError`.

### 4.2 Interaction step (`addObject` / `refineObject`)

1. `refineObject` only: `bank.invalidateAfter(frameIndex)` (cond memories persist, downstream
   recent memories + pointers drop — mirrors `MaskTimeline.invalidateAfter`).
2. `source.frameAt(frameIndex)` → preprocess (`bitmapToTensor` now accepting `VideoFrame`;
   worker closes the frame) → `videoEncoder` (B=1, graphCapture-eligible: static shapes).
3. Conditioning: `bank.hasMemory(frameIndex)` → memory path (§4.3 step 3); else init path:
   `initPath:'noMemFlag'` → run `memoryAttention` with `noMem=true` and all-invalid mask;
   `'noMemGraph'` → run the dedicated `noMemCondition` graph role instead. ⚠ PIN-2
4. `maskDecoderVideo` with real prompt tensors (source→model via the session transform, padded
   to P8=8 points, label −1 padding; box prompts encoded as 2 corner points labels 2/3; mask
   prompts → `NotImplementedError` in M2, same as M1).
5. `memoryEncoder` on the winning mask → `bank.commit(frameIndex, /*isCond*/ true, …)`,
   `bank.commitPointer`.
6. Postprocess winning logits → `MaskPayload` (reuses `logitsToSourceMask`).

### 4.3 Propagation step (per frame N, forward only at M2)

```
frame = source.read(...) next        → decode (worker-owned VideoFrame)
pixels = preprocess(frame)           → videoEncoder.run (B=1, shared by all objects)
for objects (one batched run when video.multiObjectBatch, else sequential):
  asm = bank.assemble(N)             → feeds per §2.2 (assemble BEFORE commit: streaming)
  conditioned = memoryAttention.run  (or init path if validMaps === 0)
  dec = maskDecoderVideo.run         (P8 all-padding points on propagation frames)
  best = argmax(iouScores)           → occluded = objectScoreLogits < video.occlusionThreshold
  mem = memoryEncoder.run(visionFeatures, dec.maskLogits[best])
  if (!occluded || strategy.commitOccludedMemory) bank.commit(N, false, mem…); bank.commitPointer(N, …)
  payload = logitsToSourceMask(...)  → MaskPayload (binaryMask transferred)
await emit({frameIndex, timestampUs, epoch, masks})   // blocks on pull credits
```

**Multi-object: DECIDED = batch dim** (objects stack on the leading `B` of the three memory
graphs; the video encoder always runs B=1 once per frame), per the plan's "objects = batch dim"
costing. **Sequential fallback flag**: when `video.multiObjectBatch === false` (⚠ PIN-4 — spike
may find dynamic-batch export shaky) the engine runs the same three graphs once per object with
B=1 feeds; the code path is the same loop over a `batchSlices` array of length 1 or numObjects,
so the flag is data, not a second engine.

### 4.4 Batched KV binding

Per-object rings cannot be bound directly when batching. The engine owns persistent batched
input tensors `memorySpatial[B, M*T, memDim]`, `memorySpatialPos[B, M*T, memDim]` (B = current
object count, reallocated only when an object is added/removed — not per frame) and, per frame
per object, issues exactly **two** `copyRegion` calls: object ring → its batch slot (sanctioned
by the §1.1 byte-count rule), after the bank's own commit copy. Cost: 2 × 458 KB device-device
per object per frame (EdgeTAM) — noise. Scalar/mask feeds (`tposIndices`, `memoryMask`,
pointers) are rebuilt cpu-side per frame (< 4 KB/object) and go through `uploadTensor`.

### 4.5 Per-frame tensor lifecycle (normative dispose schedule)

| Tensor | Producer | Disposed by | When |
|---|---|---|---|
| `pixels` upload | engine | engine | after videoEncoder.run returns |
| videoEncoder outputs (4) | run | engine | end of frame N (after memoryEncoder consumed visionFeatures) |
| memoryAttention output | run | engine | after maskDecoderVideo.run |
| maskDecoderVideo outputs (4) | run | engine | end of frame N (`maskLogits` after memoryEncoder + readback; `objectPointer` after readback→commitPointer) |
| memoryEncoder outputs (2) | run | engine | after `bank.commit` copies them into the ring |
| per-frame cpu uploads (mask/tpos/pointers) | engine | engine | after the run that consumed them |
| rings + batched KV inputs | bank/engine | dispose()/removeObject | session lifetime |
| decoded `VideoFrame` | FrameSource | engine (`frame.close()`) | after preprocess |

**GPU-memory flatness requirement:** after frame 3 (steady state), `backend.debugStats()` must
report identical `liveTensors`/`liveBytes` at every subsequent frame boundary. `video-engine.test.ts`
asserts this against a mock backend; the browser e2e (§9.3) asserts it against the real one.
IOBindingPlan: all persistent-loop outputs (`visionFeatures`, `visionPos`, `conditionedFeatures`,
`memoryFeatures`, `memoryPos`) stay `'device'` on webgpu; `maskLogits`, `iouScores`,
`objectPointer`, `objectScoreLogits` come back `'cpu'`. wasm: all `'cpu'`.

---

## 5. Worker protocol extension (worker-video writes; video-session imports VERBATIM)

### 5.1 `src/worker/protocol.ts` additions (structured-clone-safe)

```ts
export interface VideoSourceInfo {
  frameCount: number;        // from the mp4 sample table (exact at M2 — mp4box counts samples)
  fps: number;               // frameCount / durationSeconds (VFR flattened; documented)
  width: number; height: number;
  durationUs: number;
  codec: string;             // e.g. 'avc1.640028'
}

export interface VideoObjectResult {
  objectId: number;
  epoch: number;             // the session epoch this result belongs to
  mask: MaskPayload;         // binaryMask TRANSFERRED
}

export interface PropagateRequest {
  startFrame: number;
  endFrame: number;          // exclusive
  epoch: number;             // stamped on every emitted frame
  prefetch: number;          // initial credit grant the iterator will post (4)
}

/** Messages the MAIN thread posts on the propagation MessagePort. */
export type PropagationPullMessage =
  | { type: 'pull'; credits: number }
  | { type: 'cancel' };

/** Messages the WORKER posts on the propagation MessagePort. */
export type PropagationPushMessage =
  | { type: 'frame'; frameIndex: number; timestampUs: number; epoch: number; masks: MaskPayload[] }
  | { type: 'done'; framesEmitted: number; cancelled: boolean }
  | { type: 'error'; error: ErrorEnvelope };

export interface WorkerEngineApi {
  // ...M1 members unchanged...
  createVideoSession(): Promise<number>;
  attachVideoSource(sessionId: number, source: Blob): Promise<VideoSourceInfo>;
  addVideoObject(sessionId: number, req: { frameIndex: number; prompts: Prompt[]; objectId?: number; epoch: number }): Promise<VideoObjectResult>;
  refineVideoObject(sessionId: number, req: { objectId: number; frameIndex: number; prompts: Prompt[]; epoch: number }): Promise<VideoObjectResult>;
  removeVideoObject(sessionId: number, objectId: number): Promise<void>;
  /** Starts the loop; the port (transferred) carries everything else. Resolves when the loop is scheduled. */
  propagateVideo(sessionId: number, req: PropagateRequest, port: MessagePort): Promise<void>;
  resetVideoSession(sessionId: number): Promise<void>;
  closeVideoSession(sessionId: number): Promise<void>;
}
```

Blob crosses the boundary by structured clone — Blobs clone by reference (no byte copy); the
main thread never touches demux/decode. Epoch ownership: the **main thread is authoritative**
(bumps before dispatching mutators); the worker only stamps it onto results/frames so
`MaskTimeline.set(..., epoch)` staleness rejection composes end-to-end.

### 5.2 Pull-credit stream (DECIDED: raw MessagePort, prefetch = 4 — NOT a Comlink async iterator)

Comlink async-iterator proxying round-trips every `next()` through the RPC layer with no
backpressure and no transferable batching; a dedicated port with credits gives both. Protocol:

- **Worker side** (`src/worker/video/propagation-port.ts`): holds `credits = 0`; `onmessage`
  `'pull'` adds credits and resolves the engine's pending `emit`; each `'frame'` posted
  decrements one credit and **transfers** every `masks[i].binaryMask`. `credits === 0` → `emit`
  blocks → the engine stalls **before decoding further frames**, so the FrameSource stops feeding
  the VideoDecoder (decode pauses, buffers held — the documented stall semantics). `'cancel'` (or
  the engine's `isCancelled()`) → finish the in-flight frame WITHOUT emitting, post
  `{type:'done', cancelled: true}`, close the port. Loop completion → `'done'`; any thrown error →
  `'error'` with the M1 `ErrorEnvelope` (reusing `error-envelope.ts` serialization), then close.
- **Main side** (§6): posts `{type:'pull', credits: prefetch}` (4) immediately, then
  `{type:'pull', credits: 1}` after each `next()` the consumer actually awaits — so at most 4
  frames of decoded masks are ever in flight.

```ts
// src/worker/video/propagation-port.ts
export function runPropagationPort(
  port: MessagePort,
  run: (emit: (f: PropagationFrame) => Promise<void>, isCancelled: () => boolean) => Promise<void>,
): void;
export interface PropagationFrame {
  frameIndex: number; timestampUs: number; epoch: number; masks: MaskPayload[];
}
```

---

## 6. `src/sessions/video-session.ts` — the state machine (video-session)

### 6.1 States and epoch (normative)

```
states: 'idle' | 'propagating' | 'disposed'
epoch:  number, starts 0; bumped by refineObject, removeObject, reset (NOT by addObject — adding
        an object does not stale already-yielded frames of other objects; the plan contract only
        binds refine)
```

| Call | idle | propagating | disposed |
|---|---|---|---|
| `attachSource(Blob)` | ok once; 2nd → `InvalidStateError('one source per session')` | `InvalidStateError` | `InvalidStateError` |
| `attachSource(HTMLVideoElement)` | **DECIDED deferred**: `NotImplementedError('attachSource(HTMLVideoElement), lands in M4 — pass a Blob/File at M2')`; no `element-source.ts` file is created | same | same |
| `addObject` | ok (needs attach) | `InvalidStateError('finish or cancel propagate() first')` | `InvalidStateError` |
| `refineObject` | ok | **ok**: epoch++ → post `'cancel'` on the active port → await its `'done'` → state='idle' → dispatch RPC | `InvalidStateError` |
| `removeObject` | ok; epoch++ | like refineObject (cancel first) | `InvalidStateError` |
| `propagate` | returns iterator; state→'propagating' on first `next()` | returns a POISONED iterator whose first `next()` rejects `InvalidStateError` (the plan's "second propagate rejects"; the active iterator is undisturbed) | poisoned: `InvalidStateError` |
| `reset` | ok; epoch++; `resetVideoSession` RPC | cancel-first, then as idle | no-op |
| `dispose` | `closeVideoSession` RPC (fire-and-forget, like ImageSession); state→'disposed'; idempotent | cancel-first, then dispose | no-op |

### 6.2 The iterator (`src/sessions/propagation-iterator.ts`)

```ts
export function createPropagationIterator(init: {
  port: MessagePort;                       // port1; port2 was transferred to the worker
  capturedEpoch: number;
  currentEpoch: () => number;              // reads the session's live counter
  transform: CoordinateTransform;          // stamped on every MaskResultImpl (per-frame reuse)
  prefetch: number;                        // 4
  signal?: AbortSignal;
  onSettled: (reason: 'done' | 'cancelled' | 'error') => void;  // session flips to 'idle'
}): AsyncIterableIterator<FramePropagationResult>;
```

Behavior (normative):
- First `next()`: post `{pull, credits: prefetch}`. Arriving `'frame'` messages queue (≤ prefetch
  deep by construction); each satisfied `next()` posts `{pull, credits: 1}` and wraps payloads:
  `masks = frame.masks.map(p => new MaskResultImpl({...p, binaryMask: new Uint8Array(p.binaryMask), transform}))`
  — masks arrive in **object insertion order** (the worker's insertion-ordered map, stable across
  frames — the contract's "keyed by objectId, stable order").
- Every `next()` FIRST checks `currentEpoch() !== capturedEpoch` → run the cancel path once
  (post `'cancel'`, drain to `'done'`, close port, `onSettled('cancelled')`) then **throw
  `EpochInvalidatedError`** — never a silent stop; queued stale frames are discarded.
- `return()` / `throw()` / `signal` abort: post `'cancel'`, close after `'done'`, release queued
  buffers, resolve `{done: true}` (or rethrow / reject with `signal.reason`). Credits die with
  the port — the worker's stalled `emit` observes cancel via `isCancelled`.
- `'error'` message → rehydrate via the error-envelope map → reject the pending/next `next()`.
- `'done'` → `{done: true}`, `onSettled('done')`.

### 6.3 `VideoSessionImpl` + `createVideoSession` body (segmenter-impl)

`SegmenterImpl.createVideoSession()`: gate `spec.supportsVideo && spec.devices[this.device]`
(else `UnsupportedDeviceError` per the public TSDoc; a manifest without `video` graphs fails
worker-side with `InvalidStateError` at attach) → `engine.createVideoSession()` →
`new VideoSessionImpl(handle.engine, sessionId)`. `DEFAULT_MODEL_ID` flips to `'edgetam'`
(EdgeTAM is now the only tier whose manifest really exists with video graphs; sam3 tiers keep
image-only manifests until M3). `propagate()` options: `direction: 'backward'` →
`NotImplementedError('propagate({direction:"backward"}), lands in M3')` — a poisoned iterator,
consistent with §6.1. `startFrame` defaults to 0, `endFrame` to `frameCount`.

---

## 7. `@websam/video-editing` additions (editing agent)

### 7.1 `exporter.ts` — PNG-zip becomes real

**New dep: `fflate` — the SOLE new video-editing dependency.** Justification: MIT, ~8 KB gzipped
for the tree-shaken `Zip`/`ZipPassThrough` streaming path, zero transitive deps, works in window
+ worker; PNGs are already DEFLATE-compressed so entries use **store** (`ZipPassThrough`), which
means fflate does no compression work — it is purely the container writer, and the streaming API
avoids ever holding all frames + the zip in memory simultaneously.

Behavior matrix (public signatures unchanged; `ExportResult` gains one additive field
`framesExported: number`):

| mode/format | M2 |
|---|---|
| `'matte'` + `'png-sequence'` | **real** |
| `'matte'` + `'auto'` | resolves to `'png-sequence'` (VP9-alpha detection lands M4) |
| `'cutout'` (any) | `NotImplementedError('cutout export, lands in M4')` |
| `'webm-vp9-alpha'` | `NotImplementedError('webm-vp9-alpha export, lands in M4')` |
| `MaskCompositor.composite` | stays `NotImplementedError('MaskCompositor, lands in M4')` — the demo overlays with 2D canvas (§8) |

PNG-sequence layout: per frame, `decodeRLE` → white-on-black opaque RGBA `ImageData` →
`OffscreenCanvas.convertToBlob({type:'image/png'})` (browser-only path; no canvas →
`InvalidStateError`). Single tracked object → `frame-%06d.png` at zip root; multiple → one folder
per object `obj-<id>/frame-%06d.png`; plus `timeline.json` (fps, frameCount, width, height,
per-object frame index lists). **Holes are skipped, not failed** — sparse timelines are the normal
result of a cancelled propagation; `framesExported` and `onProgress(framesDone, frameCount)`
report reality. `suggestedFileName: 'matte.zip'`.

### 7.2 `timeline.ts` — `MaskTimeline.collect` (additive static)

```ts
static async collect(
  frames: AsyncIterable<FramePropagationResult>,      // e.g. session.propagate()
  init: MaskTimelineInit,
  options?: {
    /** Called after each frame is stored — lets the demo render live from the same single consumer. */
    onFrame?: (frame: FramePropagationResult) => void;
    /** Epoch stamped into set(); pairs with invalidateAfter for the refine flow. */
    epoch?: number;
  },
): Promise<MaskTimeline>
```

Consumes the iterator (one consumer — the iterator contract allows no second), storing
`timeline.set(String(mask.objectId), frame.frameIndex, mask.toRLE(), options?.epoch)`.
`EpochInvalidatedError` propagates to the caller (who refines, calls
`timeline.invalidateAfter`, and re-collects into the SAME timeline with the new epoch).
`index.ts` re-exports nothing new beyond the types already exported.

---

## 8. `apps/demo` video tab (demo agent) — decided scope

**M2 demo = honest scrub-less rotobrush:** pick/drop an MP4 → `createSegmenter({model:
'edgetam', modelBaseUrl})` (no license gate — Apache-2.0) → `createVideoSession` →
`attachSource(file)`. A paused `<video src=objectURL>` element (display only — the worker owns
the real decode) shows the current frame with a 2D-canvas mask overlay
(`mask.toImageData()` tinted per object, 50% alpha — NOT MaskCompositor). Flow:

- **Prompt:** click = positive point, shift-click = negative, on the displayed frame (frame 0
  initially); "add object" button starts the next objectId; each click round-trips
  `addObject`/`refineObject` and paints the returned mask immediately.
- **Track:** starts `MaskTimeline.collect(session.propagate({startFrame})), {onFrame})`; `onFrame`
  seeks the video element to `frameIndex / fps` and repaints the overlay (live, as fast as
  tracking runs — no playback clock, that is the scrub-less cut); progress = `frameIndex /
  frameCount` %, plus fps counter.
- **Cancel:** `iterator.return()` via an AbortController passed to `propagate`.
- **Refine:** only while paused (idle) — click on the current frame calls
  `refineObject(selectedObjectId, currentFrame, [point])`, then
  `timeline.invalidateAfter(...)`, then Track resumes with `startFrame = currentFrame`
  (exercises the epoch path end-to-end).
- **Export:** `new AlphaMatteExporter(timeline).export({mode:'matte'})` → download `matte.zip`.

Out of scope at M2 (stated in the UI): timeline scrubbing, backward tracking, cutout preview,
VP9 export, HTMLVideoElement sources.

---

## 9. Testing plan

### 9.1 Unit (node, per module, colocated `*.test.ts`)

- **arch-strategy / memory-bank** (memory-bank agent): `tposIndex` against the `spec.py`
  case table; cond pinning + `selectCondFrames` tie-breaks; recent-ring eviction order incl.
  post-`invalidateAfter` slot reuse; streaming rule (assemble at N excludes frame N); mask-bit
  layout vs `kvLen`; pointer ring + delta padding. Backend = a `FakeBackend` (cpu tensors,
  recorded `copyRegion` calls) so slot arithmetic is asserted on real bytes.
- **video-engine** (worker-video): mock `Backend` + mock `BackendSession`s returning
  deterministic tensors + scripted `FrameSource`; asserts the §4.3 call sequence per frame, the
  §4.5 dispose schedule (mock `debugStats` flat from frame 3), batch vs `multiObjectBatch:false`
  sequential equivalence, occlusion gating, and init-path selection (`validMaps === 0`).
- **propagation-port** (worker-video): real `MessageChannel` (node ≥ 15 has it): credits gate
  `emit` (no credit → emit pending), cancel mid-stream → `done{cancelled:true}`, error → envelope.
- **propagation-iterator / video-session** (video-session): mocked `RemoteEngine` + real
  `MessageChannel`; the full §6.1 table — refine mid-flight → `EpochInvalidatedError` on next
  `next()` (and never a silent stop), second `propagate` poisoned, `return()` posts cancel and
  releases credits, backpressure (worker side never >4 unconsumed frames), abort signal.
- **exporter / collect** (editing): timeline → zip → parse with fflate's `unzip` in the node
  test (store entries), name layout + holes skipped + `framesExported`; browser test decodes an
  entry PNG via `createImageBitmap` and round-trips pixels; `collect` staleness (late `set` with
  old epoch rejected).

### 9.2 Browser (existing `*.browser.test.ts` lane, SwiftShader)

- **memory-primitives** (backend-video): `allocTensor('device')` + `copyRegion` round-trip via
  `readback` on webgpu (soft-pass on no adapter, same pattern as `ort.browser.test.ts`) and wasm;
  slot-offset correctness against a cpu reference; `debugStats` counts.
- **webcodecs-source** (frame-source): committed 10-frame 320×180 H.264 fixture
  (`src/video/fixtures/clip-320x180-10f.mp4`, ~40 KB); asserts `VideoSourceInfo`
  (frameCount=10, dims), sequential read order + monotone timestamps, `frameAt(7)` lands on the
  right frame via a per-frame solid-color pattern, mid-stream `return()` leaves no dangling
  VideoFrames, non-MP4 blob → `InvalidStateError`.

### 9.3 Browser e2e gate placeholder (e2e agent) — wired, red-until-export

`src/e2e/video-golden.browser.test.ts` lands FULLY WRITTEN but marked
`describe.skipIf(!hasVideoModels)` with a loud failure message naming the fetch script (never a
silent skip once models exist — same M1 rule). Wiring, pinned now:

- `tools/goldens/make-video-golden.py`: runs the **Python pure-ORT e2e loop** from
  `tools/export` (the executable spec: slot selection, tpos, streaming pointers) on the golden
  clip with 1 prompted object + 1 refine at mid-clip, and commits per-frame RLEs to
  `tools/goldens/fixtures/video/golden-clip.rle.json` + prompt metadata. ⚠ PIN-10
- Golden clip: `tools/goldens/fixtures/video/golden-clip-*.mp4` — ~10 frames, non-square,
  committed (small). The test: register a local-manifest tier → attach → addObject →
  `for await` propagate → per-frame `IoU(mask.toBinary(), decodeRLE(golden[frame])) >= 0.90`,
  on wasm (int8 — the "WASM completes propagation" gate) and webgpu-if-available.
- **Memory-flatness gate, honestly scoped:** true GPU memory is NOT measurable from content JS in
  Chromium (`performance.memory` is JS-heap only; `measureUserAgentSpecificMemory` needs COI and
  still excludes GPU buffers). The gate therefore asserts (a) `backend.debugStats()` (via a
  debug-only `WorkerEngineApi.debugStats()` passthrough, worker-video owns it) identical at every
  frame boundary from frame 3 on — this catches every leak our code CAN cause — and (b) JS-heap
  slope < 5 MB across the clip as a smoke check where `performance.memory` exists. Real-GPU
  trend artifacts remain the M3 S5 lane's job.

### 9.4 Wave gates

Wave 1/2: `pnpm -F @websam/core -F @websam/video-editing build && test` (+ publint/attw stay
green — no export-map changes expected). Wave 3: + `test:browser` + demo build + bundler-matrix
(mp4box must not leak into the index chunk — it is imported only from worker-reachable modules).

---

## 10. ⚠ Spike-dependent pins (`tools/export/spikes/m2-edgetam/FINDINGS.md`, in flight)

Every pin lands as a **manifest value or golden fixture** — no `src/` signature changes.

| Pin | What FINDINGS.md must settle | Where it lands |
|---|---|---|
| PIN-1 | Exact ONNX tensor names/shapes for all four video roles; perceiver output really is `[B,T,64]` flat, `T=256`; whether pointer tokens enter as separate inputs (assumed §2.2) or fused into a `memory_kv` | manifest `graphs.*` + `video.*` emitted by `tools/export` |
| PIN-2 | Init/no-mem path: `noMem` flag input exportable? Else the `noMemCondition` micro-graph role | `video.initPath` (+ extra graph role) |
| PIN-3 | In-graph tpos gather from `tposIndices` (preferred) vs precombined pos upload | `video.tposDelivery` |
| PIN-4 | Dynamic leading batch dim on the three memory graphs survives export + ort-web | `video.multiObjectBatch` |
| PIN-5 | Recent-offset semantics: recency RANK vs raw frame distance (matters after refine gaps) | `strategyFor` branch + bank tests re-pinned to the Python executable spec |
| PIN-6 | EdgeTAM cond-selection/tie-break (maxCond=1 makes it near-trivial; confirm replace-on-new-prompt) | strategy + tests |
| PIN-7 | memoryEncoder input: low-res logits with in-graph sigmoid+resize (preferred) vs image-size mask (would need a decoder-side full-res output) | manifest `memoryEncoder` entry |
| PIN-8 | Occlusion: `object_score_logits` threshold value; commit-memory-when-occluded behavior | `video.occlusionThreshold`, `strategy.commitOccludedMemory` |
| PIN-9 | EdgeTAM pointer_time_deltas rule + whether deltas/pointer inputs exist at all in the EdgeTAM graph (EdgeTAM may fold pointers differently than SAM3) | strategy + manifest keys |
| PIN-10 | Golden clip + per-frame RLEs + refine-step goldens regenerated by the pinned toolchain; EdgeTAM preprocess constants (mean/std — SAM2-family ImageNet stats expected, NOT 0.5 — mode, maskSize) | `tools/goldens/fixtures/video/*` + manifest `preprocess` |

Contingency: if the spike downgrades batch (PIN-4) AND separate-pointer-inputs (PIN-1/9)
simultaneously, §2.2's semantic-key table is re-cut in THIS doc before wave 2 starts — wave 1
files (bank slot logic, strategy, sources, backend primitives, exporter) are unaffected by
construction.
