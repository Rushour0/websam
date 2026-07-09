# M1 Internal Contracts — Image Path

Normative design for the M1 implementation waves. Five agents implement disjoint file sets
against the signatures below; any deviation from a signature in this doc is integration drift
and must come back to this doc first. Public API (`segmenter.ts`, `index.ts` exported types,
`backend/backend.ts` existing method signatures, `errors.ts`, `coords.ts`, `masks/rle.ts`)
is **frozen** — M1 adds bodies and internal modules, plus the two explicitly sanctioned
surface additions called out in §2.4 and §4.1.

All paths below are relative to `packages/core/` unless rooted.

---

## 0. File ownership (no two agents share a file)

| Agent | Wave | Creates | Modifies |
|---|---|---|---|
| **weights** | 1 | `src/weights/manifest.ts`, `src/weights/sha256.ts`, `src/weights/weight-store.ts`, `src/weights/load-model-assets.ts`, `src/weights/*.test.ts`, `src/weights/weight-store.browser.test.ts` | — |
| **runtime** | 1 | `src/runtime/ort-env.ts`, `src/runtime/ort-tensor.ts`, `src/runtime/ort-session.ts`, `src/runtime/resolve-device.ts`, `src/runtime/*.test.ts`, `src/runtime/backend-impl.browser.test.ts` | `src/backend/backend.ts` (adds `uploadTensor` only), `src/backend/webgpu-backend.ts`, `src/backend/wasm-backend.ts` |
| **mask-result** | 1 | `src/masks/mask-result.ts`, `src/masks/mask-result.test.ts` | — |
| **worker** | 2 | `src/worker/protocol.ts`, `src/worker/error-envelope.ts`, `src/worker/preprocess.ts`, `src/worker/postprocess.ts`, `src/worker/engine.ts`, `src/worker/worker-entry.ts`, `src/worker/*.test.ts` | — |
| **sessions** | 2 | `src/sessions/segmenter-impl.ts`, `src/sessions/image-session.ts`, `src/sessions/spawn-worker.ts`, `src/sessions/*.test.ts` | `src/index.ts` (createSegmenter body + additive `SegmenterConfig` fields), `package.json` (exports + entry), `tsdown.config.ts` (multi-entry) |
| **demo** | 3 | `apps/demo/src/ImageTab.tsx` | `apps/demo/src/App.tsx`, `apps/demo/src/App.css`, `apps/demo/vite.config.ts` (if fs.allow needed) |
| **e2e** | 3 | `src/e2e/image-path.browser.test.ts`, `tools/goldens/fixtures/*` (scene png, golden json), `tools/goldens/make-image-golden.py`, `tools/goldens/fetch-models.mjs` | `packages/core/vitest.config.ts` (`server.fs.allow`), `tools/goldens/package.json` |

Ownership decisions (explicit, per shared-file risk):
- **`src/index.ts`, `package.json`, `tsdown.config.ts` → sessions agent only** (wave 2). Wave-1 agents
  export nothing from the package root; all their symbols are internal and imported by relative path.
- **`src/backend/*` → runtime agent only.**
- **`vitest.config.ts` (core) and everything under `tools/goldens/` → e2e agent only.**
- Worker and sessions run in the same wave and cross-import: sessions imports `src/worker/protocol.ts`
  and `src/worker/error-envelope.ts` **exactly as specified in §3** — both agents implement to this doc,
  and the wave-2 gate (typecheck + build) catches any drift.

---

## 1. `src/weights/` — manifest, store, loader

### 1.1 `src/weights/manifest.ts`

```ts
import type { DType } from '../backend/backend.js';
import type { TransformMode } from '../coords.js';

/** Weight quantization variants a manifest may carry. */
export type Quant = 'fp32' | 'fp16' | 'int8' | 'q4f16';

/** Graph roles. Open union: video roles ('memoryAttention', …) arrive in M2/M3 without a schema bump. */
export type KnownGraphRole = 'visionEncoder' | 'promptDecoder';
export type GraphRole = KnownGraphRole | (string & {});

/** One tensor of a graph's IO contract. `name` is the literal ONNX tensor name. */
export interface TensorSpec {
  name: string;
  dtype: DType;
  /** number = static dim, string = symbolic dim (e.g. 'batch_size'). */
  shape: readonly (number | string)[];
}

export interface WeightFileRef {
  /** Relative to the manifest URL (or `modelBaseUrl` override). Immutable, revisioned. */
  path: string;
  /** Lowercase hex SHA-256 of the file bytes. Also the cache identity (content-addressed). */
  sha256: string;
  bytes: number;
}

export interface GraphManifestEntry {
  /** Available quantizations of this graph. */
  files: Partial<Record<Quant, WeightFileRef>>;
  /**
   * IO contract keyed by SEMANTIC name — runtime code binds tensors via
   * `entry.inputs.points.name`, NEVER a hardcoded ONNX name. Required semantic
   * keys per role are listed below (§1.1.1).
   */
  inputs: Record<string, TensorSpec>;
  outputs: Record<string, TensorSpec>;
}

export interface ModelManifest {
  schemaVersion: 1;
  /** Must equal the registry ModelSpec.id this manifest serves. */
  tier: string;
  opset: number;
  graphs: Partial<Record<GraphRole, GraphManifestEntry>>;
  toolchain: { exporter: string; pytorch?: string; onnx?: string; transformers?: string };
  /** ⚠ S0 — values pinned by the export pipeline, consumed by the worker (never hardcoded in TS). */
  preprocess: {
    mode: TransformMode;                 // ⚠ S0: tentative 'square-stretch' (preprocessor_config: size 1008×1008, do_pad null)
    inputSize: number;                   // 1008 for sam3-tracker
    mean: [number, number, number];      // [0.5, 0.5, 0.5]
    std: [number, number, number];       // [0.5, 0.5, 0.5]
    /** Decoder low-res logit grid side (mask_size), e.g. 288. */
    maskSize: number;
  };
}

/** Validate untrusted JSON. Throws WeightVerifyError (bad/missing fields, wrong schemaVersion). */
export function parseModelManifest(json: unknown, sourceUrl: string): ModelManifest;
```

#### 1.1.1 Required semantic keys (M1, from the community sam3-tracker ONNX — ⚠ S0 tentative)

| Role | inputs | outputs |
|---|---|---|
| `visionEncoder` | `pixels` → `pixel_values` f32 `[B,3,1008,1008]` | `embed0` → `image_embeddings.0` f32 `[B,32,288,288]`; `embed1` → `image_embeddings.1` `[B,64,144,144]`; `embed2` → `image_embeddings.2` `[B,256,72,72]` |
| `promptDecoder` | `points` → `input_points` f32 `[B,1,P,2]`; `labels` → `input_labels` i64 `[B,1,P]`; `boxes` → `input_boxes` f32 `[B,Nb,4]`; `embed0/1/2` (same names as encoder outputs) | `iouScores` → `iou_scores` f32 `[B,N,3]`; `maskLogits` → `pred_masks` f32 `[B,N,3,288,288]`; `objectScoreLogits` → `object_score_logits` f32 `[B,N,1]` |

The worker feeds `boxes` as an empty `[1,0,4]` tensor when no box prompt exists (dynamic dim = 0).
If S0 finds zero-dim inputs unsupported, the padding convention becomes a manifest field — not code.

### 1.2 `src/weights/sha256.ts`

```ts
/** Incremental SHA-256 (pure TS, ~streaming 50–100 MB/s — overlapped with network, acceptable). */
export class Sha256Stream {
  update(chunk: Uint8Array): void;
  /** Lowercase hex digest; the instance is dead afterwards. */
  digestHex(): string;
}
```
Pure TS (not `crypto.subtle`) because WebCrypto has no incremental API and the encoder is ~300 MB —
buffering the whole file just to hash defeats streaming. Unit-tested against known vectors.

### 1.3 `src/weights/weight-store.ts`

```ts
import type { WeightFileRef } from './manifest.js';

export interface WeightStore {
  /** True iff a fully committed, previously verified copy exists. */
  has(ref: WeightFileRef): Promise<boolean>;
  /** Verified content, or undefined on miss. Never returns a partially written file. */
  get(ref: WeightFileRef): Promise<Blob | undefined>;
  /**
   * Stream `data` to storage, hashing as it writes. On digest mismatch with
   * `ref.sha256`: discard everything, throw WeightVerifyError (nothing cached).
   * On success: atomically commit and return the stored content.
   */
  put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob>;
  delete(ref: WeightFileRef): Promise<void>;
}

/** Routes per file: ref.bytes > OPFS_THRESHOLD_BYTES → OPFS, else Cache API; degrades (OPFS
 * missing → Cache API; both missing → in-memory passthrough, i.e. no persistence, never an error). */
export const OPFS_THRESHOLD_BYTES: number; // 64 * 1024 * 1024
export function createWeightStore(): WeightStore;

export class OpfsWeightStore implements WeightStore {}
export class CacheApiWeightStore implements WeightStore {}
export class MemoryWeightStore implements WeightStore {} // tests + cache:false
```

Storage identity is **content-addressed by sha256** (dedupes shared graphs across tiers).
**Atomic commit = marker-file pattern** (decided; portable — `FileSystemFileHandle.move()` is not
universal): OPFS writes `<sha256>.bin` via `createWritable`, verifies while streaming, then creates
zero-byte `<sha256>.ok`; `has`/`get` require both files. Cache API impl stores under synthetic key
`https://websam.invalid/weights/<sha256>` (put buffers — only ever ≤64 MB files).
OPFS directory: `websam-weights/`. Runs in window and worker contexts (M1 uses it in the worker).

### 1.4 `src/weights/load-model-assets.ts`

```ts
import type { LoadProgressEvent } from '../index.js';   // type-only import: no cycle at runtime
import type { ModelSpec } from '../registry.js';
import type { GraphRole, ModelManifest, Quant } from './manifest.js';
import type { WeightStore } from './weight-store.js';

export interface LoadAssetsConfig {
  /** Rebase for weight file paths AND the manifest itself when set. */
  modelBaseUrl?: string;
  cache: boolean;                       // false → MemoryWeightStore
  quantPreference: readonly Quant[];    // from resolveDevice (§2.3)
  roles: readonly GraphRole[];          // M1: ['visionEncoder', 'promptDecoder']
  fetchImpl?: typeof fetch;             // test seam
  store?: WeightStore;                  // test seam
}

export interface LoadedModelAssets {
  manifest: ModelManifest;
  /** First quantPreference entry available for ALL requested roles. */
  quant: Quant;
  /** ORT-ready bytes per role. Uint8Array (not Blob/ArrayBuffer): the sole consumer is
   *  InferenceSession.create(Uint8Array); Blob would force an extra async hop per graph. */
  graphs: Map<GraphRole, Uint8Array>;
  totalBytes: number;
}

export function loadModelAssets(
  spec: ModelSpec,
  config: LoadAssetsConfig,
  onProgress?: (e: LoadProgressEvent) => void,
): Promise<LoadedModelAssets>;
```

Errors: manifest invalid / digest mismatch → `WeightVerifyError`; explicit quant absent from
manifest → `InvalidStateError` naming the available quants; network failure → the fetch `TypeError`
propagates untouched (callers see the real cause).

#### 1.4.1 LoadPhase mapping (who emits what — exhaustive)

| `LoadPhase` | Emitted by | When | Fields |
|---|---|---|---|
| `manifest` | weights | before/while fetching+parsing the manifest | — |
| `download` | weights | streaming a file from network | `loaded`, `total`, `file` (=`ref.path`) |
| `verify` | weights | digest finalization per file | `file` |
| `offline-cache` | weights | file served from WeightStore instead of network | `file`, `loaded=total=ref.bytes` |
| `compile` | worker engine (§3.4) | per `createOrtSession` call | `file` (= role name) |
| `ready` | sessions (§4.2) | after `WorkerEngine.init` resolves | — |

---

## 2. `src/runtime/` — ORT bootstrap, sessions, device resolution, backend bodies

### 2.1 `src/runtime/ort-env.ts`

```ts
import type { OrtModule } from '../backend/webgpu-backend.js';

export interface OrtEnvOptions {
  /** ort .wasm/.mjs asset base; default: ort's own import.meta.url resolution. */
  wasmPaths?: string;
  /** Default: undefined (ort default) when crossOriginIsolated, else forced to 1. */
  numThreads?: number;
}

/** Dynamic-imports onnxruntime-web exactly once (memoized), applies env flags BEFORE any
 *  session exists. Never a static top-level import — keeps ort out of non-worker bundles. */
export function loadOrt(options?: OrtEnvOptions): Promise<OrtModule>;
```
Applied flags: `env.wasm.wasmPaths` (iff provided), `env.wasm.numThreads` (rule above),
`env.wasm.proxy = false` (we already are in a worker). Second call with different options →
`InvalidStateError`.

### 2.2 `src/runtime/ort-session.ts` and `src/runtime/ort-tensor.ts`

```ts
// ort-tensor.ts — the one DeviceTensor implementation for both browser backends.
export class OrtDeviceTensor implements DeviceTensor {
  /** Wraps an ort.Tensor; location 'device' iff tensor.location === 'gpu-buffer'. */
  static wrap(t: import('onnxruntime-web').Tensor): OrtDeviceTensor;
  readonly ortTensor: import('onnxruntime-web').Tensor;   // internal-only escape hatch
  // shape / dtype / location / dispose() per the Backend contract (dispose → ort dispose;
  // second dispose throws InvalidStateError).
}

// ort-session.ts
export interface CreateOrtSessionOptions { ioPlan?: IOBindingPlan }
export function createOrtSession(
  ort: OrtModule,
  kind: 'webgpu' | 'wasm',
  bytes: Uint8Array,
  options?: CreateOrtSessionOptions,
): Promise<import('onnxruntime-web').InferenceSession>;
```
webgpu: `executionProviders: ['webgpu']`, `preferredOutputLocation` built from
`ioPlan.outputLocations` (`'device'`→`'gpu-buffer'`, `'cpu'`→`'cpu'`); wasm: `['wasm']`, ioPlan ignored
(everything is cpu). Per plan ⚠REV ORT#26107: never inject a custom GPUDevice — ORT creates it.

### 2.3 `src/runtime/resolve-device.ts`

```ts
import type { WebGpuProbeResult } from '../backend/webgpu-backend.js';
import type { WasmProbeResult } from '../backend/wasm-backend.js';
import type { Quant } from '../weights/manifest.js';

export interface DeviceRequest { device: 'webgpu' | 'wasm' | 'auto'; quant: 'auto' | 'fp16' | 'int8' | 'q4f16' }
export interface DeviceResolution { device: 'webgpu' | 'wasm'; quantPreference: readonly Quant[] }

/** Pure + synchronous (probes injected) — unit-testable without a browser. */
export function resolveDevice(
  request: DeviceRequest,
  probes: { webgpu: WebGpuProbeResult; wasm: WasmProbeResult },
): DeviceResolution;
```

Resolution table (normative):

| device req | probe state | → device |
|---|---|---|
| `auto` | webgpu granted | `webgpu` |
| `auto` | no webgpu, wasm ok | `wasm` |
| `auto` | neither | throw `UnsupportedDeviceError` |
| `webgpu` | not granted | throw `UnsupportedDeviceError` |
| `wasm` | wasm missing | throw `UnsupportedDeviceError` |

| quant req | device caps | → quantPreference (ordered; final pick = first available for all roles, §1.4) |
|---|---|---|
| `auto` | webgpu + f16 | `['q4f16', 'fp16', 'fp32', 'int8']` |
| `auto` | webgpu, no f16 | `['fp32', 'int8']` |
| `auto` | wasm | `['int8', 'fp32']` |
| `fp16`/`q4f16` | webgpu + f16 | `[req]` (no fallback) |
| `fp16`/`q4f16` | no f16 or wasm | throw `UnsupportedDeviceError` |
| `int8` | any | `['int8']` |

### 2.4 Backend real implementations (runtime agent)

**One sanctioned interface addition** to `src/backend/backend.ts` (existing five method
signatures unchanged): the M0 contract had no way to create an *initialized* tensor, which the
image path needs for pixels/points and the video loop will need per frame:

```ts
// Added to interface Backend, with full TSDoc:
/** Create a tensor initialized from host data ('cpu' location; int64 takes BigInt64Array). */
uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor;
```

Bodies gained in M1 (identical split for `WebGpuBackend` and `WasmBackend`; all methods keep
their exact existing signatures; all throw `InvalidStateError` before `init()`):

| Method | M1 status |
|---|---|
| `createSession(graph, plan)` | **real** — requires `graph.bytes` (`url` variant → `NotImplementedError`, M2); wraps `createOrtSession(this.ort, kind, bytes, {ioPlan: plan})` in an `OrtBackendSession` whose `run()` maps `DeviceTensor`⇄`ort.Tensor` via `OrtDeviceTensor` and honors `fetches` |
| `uploadTensor(data, shape, dtype)` | **real** (new) |
| `allocTensor(shape, dtype, loc)` | **real for `'cpu'`** (zeroed typed array); `'device'` stays `NotImplementedError` (', lands in M2' — video ring) |
| `copyRegion(src, dst, slot)` | stays `NotImplementedError` (', lands in M2') — memory-bank primitive |
| `readback(tensor)` | **real** — cpu: view over data; device: `ortTensor.getData()` (f16 → `Uint16Array` raw bits per contract) |
| `dispose()` | **real** — disposes sessions/tensors the backend tracked, resets `initialized` |

The worker engine (§3.4) drives inference **through the `Backend` interface** (createSession /
uploadTensor / readback), proving the M0 abstraction on the image path. M1 IOBindingPlan:
webgpu keeps `embed0/1/2` at `'device'` (fed back to the decoder copy-free), decoder outputs `'cpu'`;
wasm: all `'cpu'`.

---

## 3. `src/worker/` — worker entry, engine, protocol, errors

### 3.1 `src/worker/protocol.ts` (structured-clone-safe; implemented **verbatim** — sessions imports it)

```ts
import type { LoadProgressEvent } from '../index.js';
import type { ModelSpec } from '../registry.js';
import type { Prompt } from '../segmenter.js';
import type { CoordinateTransform } from '../coords.js';
import type { Quant } from '../weights/manifest.js';

export interface WorkerInitRequest {
  spec: ModelSpec;                       // plain object — clones fine
  device: 'webgpu' | 'wasm';
  quantPreference: readonly Quant[];
  modelBaseUrl?: string;
  cache: boolean;
  wasmPaths?: string;
}
export interface WorkerInitResult { device: 'webgpu' | 'wasm'; quant: Quant; totalBytes: number }

export interface EncodeResponse {
  width: number; height: number; encodeMs: number;
  /** Computed IN the worker from bitmap dims + manifest.preprocess (mode, inputSize). */
  transform: CoordinateTransform;
}

export interface DecodeRequest { prompts: Prompt[]; multimask?: boolean; objectId?: number }

export interface MaskPayload {
  objectId: number;
  score: number;                         // best iou_scores entry for the chosen mask
  width: number; height: number;         // source-pixel dims
  /** Row-major 0/1 bytes, width*height — TRANSFERRED (zero-copy) to the main thread. */
  binaryMask: ArrayBuffer;
  // lowResLogits: intentionally absent in M1 — mask-prompt feedback lands with video (M2).
}

export interface WorkerEngineApi {
  init(req: WorkerInitRequest, onProgress?: (e: LoadProgressEvent) => void): Promise<WorkerInitResult>;
  createSession(): Promise<number>;                              // → sessionId
  encodeImage(sessionId: number, bitmap: ImageBitmap): Promise<EncodeResponse>;
  decode(sessionId: number, req: DecodeRequest): Promise<MaskPayload[]>;
  closeSession(sessionId: number): void;
  dispose(): Promise<void>;
}
```

### 3.2 Coordinate decision (normative)

**The worker receives PROMPTS IN SOURCE-PIXEL space and converts internally.** The worker — not the
main thread — computes the `CoordinateTransform` at `encodeImage` time (it has the bitmap dims, and
`manifest.preprocess.mode`/`inputSize` live worker-side), stores it per session, applies
`sourceToModel` from `src/coords.ts` to every point/box at decode, and returns the transform in
`EncodeResponse` so the main thread can stamp it onto `MaskResultImpl`. Rationale: exactly one
process ever runs the transform math (no main/worker drift), and `coords.ts` is already in the
worker bundle. Mask prompts (`{type:'mask'}`) are decoder-logit space per the contract and are
**rejected with `NotImplementedError` in M1** (no logits round-trip yet).

### 3.3 `src/worker/error-envelope.ts`

Structured clone drops custom own-properties of `Error` (Comlink's built-in `'throw'` handler keeps
only name/message/stack), so `WebsamError.code` would not survive. Fix: **override Comlink's
`'throw'` transfer handler on both sides**.

```ts
export interface ErrorEnvelope {
  name: string; message: string; stack?: string;
  websamCode?: WebsamErrorCode;          // present iff the thrown value was a WebsamError
}
/** Call once per realm (worker-entry.ts AND segmenter-impl.ts):
 *  comlink.transferHandlers.set('throw', …). Serialize: WebsamError → envelope with websamCode.
 *  Deserialize: websamCode → rehydrate via a code→constructor map (covers every WebsamErrorCode;
 *  message/stack preserved); no code → plain Error. */
export function installErrorTransferHandler(comlink: typeof import('comlink')): void;
```
DOMException `AbortError` never crosses the boundary (abort is handled main-side, §4.3).

### 3.4 `src/worker/engine.ts`, `preprocess.ts`, `postprocess.ts`, `worker-entry.ts`

`WorkerEngine implements WorkerEngineApi`. `init`: `loadOrt` (§2.1) → construct + `init()` the
backend for `req.device` → `loadModelAssets` (§1.4; the **whole weight pipeline runs in the worker**
— OPFS/Cache/fetch all work here, and bytes never cross a thread) → `backend.createSession` per role
(emit `compile` per §1.4.1) → return `{device, quant, totalBytes}`. `onProgress` arrives as a Comlink
proxy; the engine may call it fire-and-forget.

`encodeImage(sessionId, bitmap)`: `computeTransform(bitmap.width, bitmap.height,
preprocess.inputSize, preprocess.mode)` → `preprocess.ts: bitmapToTensor(bitmap, preprocess)`
(OffscreenCanvas 2D `drawImage` performs the resize — square-stretch: `drawImage(b,0,0,S,S)`;
letterbox: draw at `(padX,padY,srcW*scale,srcH*scale)` — → `getImageData` → Float32 CHW
`(v/255 − mean)/std`; letterbox pad regions are **explicitly zeroed post-normalization** for
exactness) → **worker closes the bitmap** (`bitmap.close()`; it owns it post-transfer) →
`backend.uploadTensor` → run visionEncoder → keep the three embedding `DeviceTensor`s in the
session slot (≈21 MB fp32/session; a re-`encodeImage` on the same slot disposes the old ones) →
`{width, height, encodeMs, transform}`.

`decode(sessionId, req)`: no encoded slot → `InvalidStateError`. Points/boxes →
`sourceToModel` → f32/i64 tensors (labels as `BigInt64Array`; empty `[1,0,4]` boxes when none) →
run promptDecoder feeding the cached embeddings → pick mask: `multimask:true` → all 3, else argmax
`iouScores` → `postprocess.ts: logitsToSourceMask(logits, gridSize, transform)` — for each source
pixel, map through the transform into the model square, scale by `maskSize/inputSize`, bilinear-
sample the logit grid, threshold `> 0` → `Uint8Array` — return `MaskPayload[]` with every
`binaryMask` **transferred** back. `objectId` = `req.objectId ?? 0`.

`worker-entry.ts` (the `"./worker"` entry): `installErrorTransferHandler(Comlink);
Comlink.expose(new WorkerEngine())`. No side effects beyond that; module worker.

---

## 4. `src/sessions/` + real `createSegmenter`

### 4.1 `src/index.ts` changes (sessions agent)

`createSegmenter` body becomes `return createSegmenterImpl(config)` (impl in
`src/sessions/segmenter-impl.ts`). Two **additive** optional fields on `SegmenterConfig`:

```ts
/** Override the worker script URL (bundler escape hatch; see §4.2). */
workerUrl?: string | URL;
/** Forwarded to onnxruntime-web env.wasm.wasmPaths inside the worker. */
wasmPaths?: string;
```
No other export changes; `MaskResultImpl` and all M1 modules stay internal.

### 4.2 `src/sessions/segmenter-impl.ts` + `spawn-worker.ts`

`createSegmenterImpl(config)` sequence:
1. Resolve spec: `getModel(config.model ?? 'sam3-tracker')` (M1 default — the only tier with image
   graphs; flips to `'edgetam'` at M2). Unknown id → `InvalidStateError`.
2. License gate: `spec.requiresLicenseAcceptance && config.acceptLicense !== 'sam'` → reject
   `InvalidStateError` with a message naming the config key.
3. **Probe on the main thread, before spawning**: `Promise.all([WebGpuBackend.probe(),
   WasmBackend.probe()])` → `resolveDevice({device: config.device ?? 'auto', quant: config.quant ?? 'auto'}, probes)`.
   Device-support cross-check against `spec.devices` (unsupported → `UnsupportedDeviceError`).
4. Spawn (`spawn-worker.ts`): `new Worker(config.workerUrl ?? new URL('./worker.js',
   import.meta.url), { type: 'module' })`. **M1 implements only the URL path** — the plan's inlined
   single-file fallback is deferred (M2/M4, bundler-matrix driven); `workerUrl` is the documented
   escape hatch for bundlers that break sibling-URL resolution. In dist, `index.js` and `worker.js`
   are siblings (§6.1), so the relative URL resolves for ESM consumers.
5. `installErrorTransferHandler(Comlink)`; `engine = Comlink.wrap<WorkerEngineApi>(worker)`;
   `await engine.init(req, Comlink.proxy(onProgress))`; emit `{phase:'ready'}`.
6. Return `Segmenter`: `device`/`model` (`{spec, quant, totalBytes}` from `WorkerInitResult`);
   `createImageSession()` → `engine.createSession()` → `new ImageSessionImpl(engine, sessionId)`;
   `createVideoSession()` → rejects `NotImplementedError('createVideoSession, lands in M2')`
   (likewise any live-session surface — video stays NIE in M1); `dispose()` → `await engine.dispose()`
   then `worker.terminate()`; double-dispose is a no-op, use-after-dispose → `InvalidStateError`.

### 4.3 `src/sessions/image-session.ts`

`ImageSessionImpl implements ImageSession`. `encode(image, {signal})`: signal already aborted →
reject `AbortError` pre-dispatch; normalize input to `ImageBitmap` **on the main thread**
(`createImageBitmap(ImageData | canvas)`; an `ImageBitmap` input is consumed as-is — documented) and
send with `Comlink.transfer(bitmap, [bitmap])` — **main thread transfers, worker closes** (§3.4);
abort during flight cannot cancel the ORT run (M1 semantics): the result is discarded and the call
rejects `AbortError`, `isEncoded` stays false. On success store the returned `transform`, set
`isEncoded`. `decode(prompts, opts)`: not encoded → `InvalidStateError`; call `engine.decode`; wrap
each `MaskPayload` in `new MaskResultImpl({...payload, binaryMask: new Uint8Array(payload.binaryMask),
transform})`. `dispose()`: `engine.closeSession(id)`; idempotent; later calls → `InvalidStateError`.
Sessions are cheap views — no M1 hard cap (each encoded slot holds ≈21 MB device memory; documented).

---

## 5. `src/masks/mask-result.ts`

```ts
import type { MaskResult } from '../segmenter.js';
import type { CoordinateTransform } from '../coords.js';
import { encodeRLE, toCocoRLE, type RLEMask, type CocoRLE } from './rle.js';

export interface MaskResultInit {
  objectId: number; score: number; width: number; height: number;
  /** Row-major 0/1 bytes, length width*height (validated). */
  binaryMask: Uint8Array;
  transform: CoordinateTransform;
}

export class MaskResultImpl implements MaskResult {
  constructor(init: MaskResultInit);      // ALWAYS copies binaryMask — see below
  readonly objectId: number; readonly score: number;
  readonly width: number; readonly height: number;
  /** Coordinate-contract rule 4: every result carries its transform (extra member beyond the
   *  frozen public interface — structurally compatible; surfaced on the interface post-M1). */
  readonly transform: CoordinateTransform;
  toImageData(): ImageData;               // RGBA: mask=1 → [255,255,255,255], else [0,0,0,0]
  toBitmap(): Promise<ImageBitmap>;       // createImageBitmap(this.toImageData())
  toRLE(): RLEMask;                       // encodeRLE; memoized (legal: instance is immutable)
  toCocoRLE(): CocoRLE;                   // masks/rle.ts toCocoRLE(this.toRLE())
  toBinary(): Uint8Array;                 // defensive copy every call
}
```
**Immutability = copy on construction, NOT pooling** (M1 decision; pooling arrives with video).
The constructor unconditionally copies `binaryMask` and validates its length; no reference to
caller memory is retained, no websam code ever mutates the copy. Cost: one ≈`w*h` copy per mask
per click — irrelevant at interactive rates. Pure main-thread module: no worker/ORT imports, fully
unit-testable in node (ImageData/createImageBitmap paths covered in the wave-3 browser e2e).

---

## 6. Cross-cutting

### 6.1 `package.json` + `tsdown.config.ts` (sessions agent, wave 2)

```ts
// tsdown.config.ts
entry: { index: 'src/index.ts', worker: 'src/worker/worker-entry.ts' },  // rest unchanged
```
→ `dist/index.js` + `dist/worker.js` as siblings (satisfies §4.2's relative URL). onnxruntime-web
stays a dynamic import inside the worker chunk (external dependency — not bundled). Exports map gains:

```json
"./worker": { "types": "./dist/worker.d.ts", "default": "./dist/worker.js" }
```
No new dependencies (comlink + onnxruntime-web already declared). publint/attw stay green.

### 6.2 Browser e2e gate (e2e agent, wave 3)

Fixtures — `tools/goldens/fixtures/` (committed, small):
- `scene-1280x720.png` — NON-square, per the coordinate contract's golden requirement.
- `image-golden.json` — `{ prompt: {x, y, label}, mode: TransformMode, rle: {width, height,
  counts: number[]}, minIoU: 0.9 }`, generated by `tools/goldens/make-image-golden.py` (HF reference
  pipeline via the existing `tools/export` venv). **⚠ S0 spot** (§7).

Model weights are NOT committed: `tools/goldens/fetch-models.mjs` pulls the community ONNX from HF
into `tools/goldens/.cache/models/` (gitignored, CI-cached) and emits a local
`manifest.json` conforming to §1.1 (sha256s computed at fetch time). The test:

```ts
// src/e2e/image-path.browser.test.ts (matches the existing *.browser.test.ts include glob)
import sceneUrl from '../../../../tools/goldens/fixtures/scene-1280x720.png?url';
import golden from '../../../../tools/goldens/fixtures/image-golden.json';
```
requires `server.fs.allow: [resolve(__dirname, '../..')]` in `packages/core/vitest.config.ts` (e2e
agent's only config edit). Flow: `registerModel` a test tier whose `manifestUrl` points at the
served `.cache/models/manifest.json` → `createSegmenter({model: testId, device})` →
`createImageSession` → `encode(await createImageBitmap(fetched png))` → `decode([point])` →
IoU(`toBinary()`, `decodeRLE(golden.rle)`) ≥ 0.9. Runs on **wasm** (deterministic in CI) and
attempts **webgpu** with the same soft-pass-on-no-adapter pattern as `ort.browser.test.ts`
(SwiftShader lane). Models absent → the suite fails with a message naming `fetch-models.mjs`
(the gate must never silently skip). Plus: coordinate round-trip assertion — the prompt point maps
into the golden mask's model-space footprint under `EncodeResponse.transform`.

### 6.3 Demo image tab (demo agent, wave 3)

`apps/demo/src/ImageTab.tsx`: sample image (or file input) → `createSegmenter({model:
'sam3-tracker', acceptLicense: 'sam', modelBaseUrl: import.meta.env.VITE_WEBSAM_MODELS ??
'/models/'})` (weights in `apps/demo/public/models/`, gitignored, populated by
`fetch-models.mjs`) → click = positive point, shift-click = negative → overlay
`mask.toImageData()` at 50% alpha on a canvas → show `score`, `encodeMs`, load-progress bar from
`onProgress`. `App.tsx` gains the tab switch only.

### 6.4 Dependency-ordered apply waves (gate after every wave: `pnpm -F @websam3/core build && test`; wave 3 adds `test:browser` + demo build)

| Wave | Agent | Files (see §0 for the full lists) | Depends on |
|---|---|---|---|
| **1** | weights | `src/weights/*` (new only) | — |
| **1** | runtime | `src/runtime/*` (new) + `src/backend/*` (bodies) | — |
| **1** | mask-result | `src/masks/mask-result.ts` (+test) | — |
| **2** | worker | `src/worker/*` (new only) | weights, runtime |
| **2** | sessions | `src/sessions/*` + `src/index.ts` + `package.json` + `tsdown.config.ts` | all wave 1; worker **types per §3.1 only** (compiled at the wave-2 gate) |
| **3** | demo | `apps/demo/src/*` | wave 2 |
| **3** | e2e | `src/e2e/*`, `vitest.config.ts`, `tools/goldens/*` | wave 2 |

Wave-1 items are mutually disjoint (weights imports only M0 types; runtime owns backend/;
mask-result touches one new file). The single intra-wave coupling (sessions → worker protocol) is
pinned verbatim by §3.1/§3.3.

---

## 7. ⚠ S0 open items (in-flight spike; `tools/export/spikes/s0/FINDINGS.md` does not exist yet)

Everything S0-dependent is a **manifest value, never a TS constant**: the transform mode lives in
`manifest.preprocess.mode` (worker reads it; `coords.ts` implements both modes), tensor names/shapes
live in `manifest.graphs.*.inputs/outputs` (worker binds via semantic keys, §1.1.1). Current
tentative values come from direct inspection of the community graphs + `preprocessor_config.json`
(square 1008×1008 resize, `do_pad: null` ⇒ tentatively `'square-stretch'`; mean/std 0.5;
mask grid 288; names as tabled in §1.1.1).

**The two spots that MUST be updated when FINDINGS.md lands:**
1. **`tools/goldens/fetch-models.mjs` manifest emission** (and the demo's copy of it) — pin
   `preprocess.mode` and confirm/correct every `TensorSpec` name, dtype, and shape against S0's
   verified IO dump (including whether zero-dim `input_boxes` feeds are accepted, §1.1.1).
2. **`tools/goldens/fixtures/image-golden.json` (+ `scene-1280x720.png` reference mask)** —
   regenerate via `make-image-golden.py` under the pinned mode so the IoU gate asserts against the
   true reference; the committed golden records the `mode` it was generated with, and the e2e fails
   loudly on a manifest/golden mode mismatch rather than comparing across modes.

No `packages/core/src` file changes when S0 lands — that is the design invariant.
