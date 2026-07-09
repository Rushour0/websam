/**
 * WorkerEngine — the worker-side implementation of {@link WorkerEngineApi}.
 *
 * Runs the ENTIRE weight pipeline in the worker (OPFS/Cache/fetch all work
 * here, and bytes never cross a thread), boots ort via {@link loadOrt},
 * drives inference exclusively through the {@link Backend} interface
 * (createSession / uploadTensor / readback — proving the M0 abstraction on
 * the image path), and binds every tensor by the manifest's SEMANTIC keys
 * (`entry.inputs.points.name`), never a hardcoded ONNX name.
 *
 * M1 IOBindingPlan (docs/m1-internal-contracts.md §2.4): on webgpu the three
 * encoder embeddings stay at `'device'` (fed back to the decoder copy-free)
 * and decoder outputs come back `'cpu'`; on wasm everything is `'cpu'`.
 */

import * as Comlink from 'comlink';
import type {
  Backend,
  BackendSession,
  DeviceTensor,
  DType,
  IOBindingPlan,
} from '../backend/backend.js';
import { WasmBackend } from '../backend/wasm-backend.js';
import { WebGpuBackend } from '../backend/webgpu-backend.js';
import { computeTransform, type CoordinateTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';
import type { LoadProgressEvent } from '../index.js';
import type { ModelSpec } from '../registry.js';
import { loadOrt } from '../runtime/ort-env.js';
import type { Prompt } from '../segmenter.js';
import { createWebCodecsFrameSource } from '../video/webcodecs-source.js';
import { loadModelAssets } from '../weights/load-model-assets.js';
import type {
  GraphManifestEntry,
  GraphRole,
  ModelManifest,
  Quant,
  TensorSpec,
} from '../weights/manifest.js';
import { logitsToSourceMask } from './postprocess.js';
import { bitmapToTensor, buildPromptTensors } from './preprocess.js';
import type {
  DecodeRequest,
  EncodeResponse,
  MaskPayload,
  PropagateRequest,
  VideoObjectResult,
  VideoSourceInfo,
  WorkerEngineApi,
  WorkerInitRequest,
  WorkerInitResult,
} from './protocol.js';
import { runPropagationPort } from './video/propagation-port.js';
import { VideoEngine, type VideoEngineGraphs } from './video/video-engine.js';

/** Graph roles the M1 image path loads. */
const M1_ROLES: readonly GraphRole[] = ['visionEncoder', 'promptDecoder'];

/** Semantic keys of the three FPN embedding tensors (encoder outputs = decoder inputs). */
const EMBED_KEYS = ['embed0', 'embed1', 'embed2'] as const;

/** Everything held per encoded image (per session slot). */
interface EncodedImage {
  transform: CoordinateTransform;
  width: number;
  height: number;
  /** Semantic key (`embed0/1/2`) → cached embedding tensor (device-resident on webgpu). */
  embeddings: Map<string, DeviceTensor>;
}

interface SessionSlot {
  encoded: EncodedImage | undefined;
}

/** Engine state established by a successful {@link WorkerEngine.init}. */
interface EngineState {
  device: 'webgpu' | 'wasm';
  backend: Backend;
  manifest: ModelManifest;
  encoderEntry: GraphManifestEntry;
  decoderEntry: GraphManifestEntry;
  encoder: BackendSession;
  decoder: BackendSession;
  /** Retained so `createVideoSession`'s lazy graph load (§4/§5) can re-run `loadModelAssets`. */
  spec: ModelSpec;
  modelBaseUrl: string | undefined;
  cache: boolean;
  quantPreference: readonly Quant[];
}

/** Graph roles the M2 video path loads (§4.1); `noMemCondition` is conditional (§10 PIN-2). */
const VIDEO_CORE_ROLES: readonly GraphRole[] = [
  'videoEncoder',
  'memoryAttention',
  'maskDecoderVideo',
  'memoryEncoder',
];

/** Compiled once per worker (shared by every video session) — mirrors the M1 encoder/decoder pattern. */
interface VideoAssets {
  manifest: ModelManifest;
  graphs: VideoEngineGraphs;
}

interface VideoSessionSlot {
  /** `undefined` until `attachVideoSource` succeeds (§6.1: source attaches once, worker-side mirror). */
  videoEngine: VideoEngine | undefined;
}

/** Look up a semantic tensor key, failing loudly when the manifest lacks it. */
function requireTensorSpec(
  map: Record<string, TensorSpec>,
  key: string,
  where: string,
): TensorSpec {
  const spec = map[key];
  if (!spec) {
    throw new InvalidStateError(
      `Model manifest is missing semantic tensor key '${key}' in ${where} ` +
        `(has: ${Object.keys(map).join(', ') || 'none'})`,
    );
  }
  return spec;
}

/** M1 generates host data in fixed dtypes; a manifest disagreeing is a hard error. */
function assertDtype(spec: TensorSpec, expected: DType, where: string): void {
  if (spec.dtype !== expected) {
    throw new InvalidStateError(
      `${where}: manifest declares dtype '${spec.dtype}' for '${spec.name}' but the M1 ` +
        `worker feeds '${expected}'`,
    );
  }
}

function disposeEmbeddings(slot: SessionSlot): void {
  if (!slot.encoded) return;
  for (const tensor of slot.encoded.embeddings.values()) {
    tensor.dispose();
  }
  slot.encoded = undefined;
}

/**
 * The worker-side engine exposed over Comlink (see `worker-entry.ts`).
 * Construct once per worker; {@link init} may run once.
 */
export class WorkerEngine implements WorkerEngineApi {
  #initCalled = false;
  #disposed = false;
  #state: EngineState | undefined;
  #nextSessionId = 1;
  readonly #sessions = new Map<number, SessionSlot>();

  // -- M2 video path (§4/§5/§6) ---------------------------------------------
  #videoAssets: VideoAssets | undefined;
  #videoAssetsLoading: Promise<VideoAssets> | undefined;
  #nextVideoSessionId = 1;
  readonly #videoSessions = new Map<number, VideoSessionSlot>();

  #requireState(method: string): EngineState {
    if (this.#disposed) {
      throw new InvalidStateError(`WorkerEngine.${method} called after dispose()`);
    }
    if (!this.#state) {
      throw new InvalidStateError(`WorkerEngine.${method} called before init() completed`);
    }
    return this.#state;
  }

  #requireSlot(sessionId: number): SessionSlot {
    const slot = this.#sessions.get(sessionId);
    if (!slot) {
      throw new InvalidStateError(`Unknown or closed worker session id ${sessionId}`);
    }
    return slot;
  }

  /**
   * Boot ort, init the backend for `req.device`, run the whole weight
   * pipeline in the worker, and compile encoder + decoder sessions.
   *
   * `onProgress` typically arrives as a Comlink proxy; it is invoked
   * fire-and-forget (per-call rejections are swallowed). Emits `compile`
   * per createSession (§1.4.1); `ready` is the sessions layer's to emit.
   */
  async init(
    req: WorkerInitRequest,
    onProgress?: (e: LoadProgressEvent) => void,
  ): Promise<WorkerInitResult> {
    if (this.#disposed) {
      throw new InvalidStateError('WorkerEngine.init called after dispose()');
    }
    if (this.#initCalled) {
      throw new InvalidStateError('WorkerEngine.init called twice');
    }
    this.#initCalled = true;

    // A Comlink-proxied callback returns a promise; fire-and-forget per §3.4.
    const progress = onProgress
      ? (e: LoadProgressEvent): void => {
          void Promise.resolve(onProgress(e)).catch(() => undefined);
        }
      : undefined;

    const ort = await loadOrt(req.wasmPaths !== undefined ? { wasmPaths: req.wasmPaths } : {});
    const backend: Backend =
      req.device === 'webgpu' ? new WebGpuBackend(ort) : new WasmBackend(ort);
    await backend.init();

    try {
      const assets = await loadModelAssets(
        req.spec,
        {
          modelBaseUrl: req.modelBaseUrl,
          cache: req.cache,
          quantPreference: req.quantPreference,
          roles: M1_ROLES,
        },
        progress,
      );

      const encoderEntry = assets.manifest.graphs['visionEncoder'];
      const decoderEntry = assets.manifest.graphs['promptDecoder'];
      const encoderBytes = assets.graphs.get('visionEncoder');
      const decoderBytes = assets.graphs.get('promptDecoder');
      if (!encoderEntry || !decoderEntry || !encoderBytes || !decoderBytes) {
        // Unreachable in practice: loadModelAssets validates the roles.
        throw new InvalidStateError(
          `Model '${req.spec.id}' assets are missing a required M1 graph role`,
        );
      }

      // §2.4: webgpu keeps the embeddings device-resident for copy-free
      // decoder feeds; wasm is all-cpu (its sessions ignore the plan anyway).
      const embedLocation = req.device === 'webgpu' ? 'device' : 'cpu';
      const encoderPlan: IOBindingPlan = { outputLocations: {} };
      for (const key of EMBED_KEYS) {
        const spec = requireTensorSpec(encoderEntry.outputs, key, 'visionEncoder.outputs');
        encoderPlan.outputLocations[spec.name] = embedLocation;
      }
      const decoderPlan: IOBindingPlan = { outputLocations: {} };
      for (const key of ['iouScores', 'maskLogits', 'objectScoreLogits']) {
        // objectScoreLogits present-or-not is manifest business; only pin
        // locations for outputs the manifest declares.
        const spec = decoderEntry.outputs[key];
        if (spec) decoderPlan.outputLocations[spec.name] = 'cpu';
      }

      progress?.({ phase: 'compile', file: 'visionEncoder' });
      const encoder = await backend.createSession(
        { name: 'visionEncoder', bytes: encoderBytes },
        encoderPlan,
      );
      progress?.({ phase: 'compile', file: 'promptDecoder' });
      const decoder = await backend.createSession(
        { name: 'promptDecoder', bytes: decoderBytes },
        decoderPlan,
      );

      this.#state = {
        device: req.device,
        backend,
        manifest: assets.manifest,
        encoderEntry,
        decoderEntry,
        encoder,
        decoder,
        spec: req.spec,
        modelBaseUrl: req.modelBaseUrl,
        cache: req.cache,
        quantPreference: req.quantPreference,
      };
      return { device: req.device, quant: assets.quant, totalBytes: assets.totalBytes };
    } catch (err) {
      // Leave nothing device-resident behind a failed init; the engine stays
      // unusable (init is once-only) and the main thread terminates the worker.
      await backend.dispose().catch(() => undefined);
      throw err;
    }
  }

  /** Open a session slot (cheap: state is per-encoded-image). */
  async createSession(): Promise<number> {
    this.#requireState('createSession');
    const id = this.#nextSessionId++;
    this.#sessions.set(id, { encoded: undefined });
    return id;
  }

  /**
   * Preprocess + run the vision encoder and cache the three embedding
   * tensors in the session slot (≈21 MB fp32/session; a re-encode on the
   * same slot disposes the old ones). The worker CLOSES the bitmap — it
   * owns it post-transfer (§3.4).
   */
  async encodeImage(sessionId: number, bitmap: ImageBitmap): Promise<EncodeResponse> {
    const state = this.#requireState('encodeImage');
    const slot = this.#requireSlot(sessionId);
    const preprocess = state.manifest.preprocess;
    const width = bitmap.width;
    const height = bitmap.height;

    const started = performance.now();
    const transform = computeTransform(width, height, preprocess.inputSize, preprocess.mode);
    let chw: Float32Array;
    try {
      chw = bitmapToTensor(bitmap, transform, preprocess);
    } finally {
      bitmap.close();
    }

    const pixelsSpec = requireTensorSpec(state.encoderEntry.inputs, 'pixels', 'visionEncoder.inputs');
    assertDtype(pixelsSpec, 'float32', 'encodeImage');
    const pixels = state.backend.uploadTensor(
      chw,
      [1, 3, preprocess.inputSize, preprocess.inputSize],
      'float32',
    );
    let outputs: Record<string, DeviceTensor>;
    try {
      outputs = await state.encoder.run({ [pixelsSpec.name]: pixels });
    } finally {
      pixels.dispose();
    }

    const embeddings = new Map<string, DeviceTensor>();
    const kept = new Set<string>();
    for (const key of EMBED_KEYS) {
      const spec = requireTensorSpec(state.encoderEntry.outputs, key, 'visionEncoder.outputs');
      const tensor = outputs[spec.name];
      if (!tensor) {
        throw new InvalidStateError(
          `Vision encoder run produced no output named '${spec.name}' (semantic key '${key}')`,
        );
      }
      embeddings.set(key, tensor);
      kept.add(spec.name);
    }
    for (const [name, tensor] of Object.entries(outputs)) {
      if (!kept.has(name)) tensor.dispose();
    }

    disposeEmbeddings(slot); // re-encode on the same slot replaces the old image
    slot.encoded = { transform, width, height, embeddings };
    return { width, height, encodeMs: performance.now() - started, transform };
  }

  /**
   * Decode masks for prompts (SOURCE pixels — converted internally per
   * §3.2) against the session's cached embeddings. Returns one payload per
   * selected candidate (`multimask: true` → all; else the best-IoU one),
   * every `binaryMask` TRANSFERRED back zero-copy.
   */
  async decode(sessionId: number, req: DecodeRequest): Promise<MaskPayload[]> {
    const state = this.#requireState('decode');
    const slot = this.#requireSlot(sessionId);
    const encoded = slot.encoded;
    if (!encoded) {
      throw new InvalidStateError(
        `decode called before encodeImage on worker session ${sessionId}`,
      );
    }
    if (req.prompts.length === 0) {
      throw new InvalidStateError('decode requires at least one prompt');
    }

    const prompts = buildPromptTensors(req.prompts, encoded.transform);
    const inputs = state.decoderEntry.inputs;
    const feeds: Record<string, DeviceTensor> = {};
    const scratch: DeviceTensor[] = [];
    try {
      const pointsSpec = requireTensorSpec(inputs, 'points', 'promptDecoder.inputs');
      assertDtype(pointsSpec, 'float32', 'decode');
      const points = state.backend.uploadTensor(
        prompts.points,
        [1, 1, prompts.pointCount, 2],
        'float32',
      );
      scratch.push(points);
      feeds[pointsSpec.name] = points;

      const labelsSpec = requireTensorSpec(inputs, 'labels', 'promptDecoder.inputs');
      assertDtype(labelsSpec, 'int64', 'decode');
      const labels = state.backend.uploadTensor(
        prompts.labels,
        [1, 1, prompts.pointCount],
        'int64',
      );
      scratch.push(labels);
      feeds[labelsSpec.name] = labels;

      // Boxes are fed iff the manifest declares the input (empty [1,0,4]
      // when no box prompt exists, §1.1.1). If S0-final finds zero-dim feeds
      // unsupported, the convention changes in the MANIFEST, not here.
      const boxesSpec = inputs['boxes'];
      if (boxesSpec) {
        assertDtype(boxesSpec, 'float32', 'decode');
        const boxes = state.backend.uploadTensor(
          prompts.boxes,
          [1, prompts.boxCount, 4],
          'float32',
        );
        scratch.push(boxes);
        feeds[boxesSpec.name] = boxes;
      } else if (prompts.boxCount > 0) {
        throw new InvalidStateError(
          "box prompts were given but the model manifest declares no 'boxes' decoder input",
        );
      }

      for (const key of EMBED_KEYS) {
        const spec = requireTensorSpec(inputs, key, 'promptDecoder.inputs');
        const tensor = encoded.embeddings.get(key);
        if (!tensor) {
          throw new InvalidStateError(`Session ${sessionId} has no cached '${key}' embedding`);
        }
        feeds[spec.name] = tensor;
      }

      const outputs = await state.decoder.run(feeds);
      try {
        return await this.#selectMasks(state, encoded, req, outputs);
      } finally {
        for (const tensor of Object.values(outputs)) {
          tensor.dispose();
        }
      }
    } finally {
      for (const tensor of scratch) {
        tensor.dispose();
      }
    }
  }

  /** Pick candidate mask(s) from decoder outputs and build the payloads. */
  async #selectMasks(
    state: EngineState,
    encoded: EncodedImage,
    req: DecodeRequest,
    outputs: Record<string, DeviceTensor>,
  ): Promise<MaskPayload[]> {
    const iouSpec = requireTensorSpec(state.decoderEntry.outputs, 'iouScores', 'promptDecoder.outputs');
    const maskSpec = requireTensorSpec(state.decoderEntry.outputs, 'maskLogits', 'promptDecoder.outputs');
    const iouTensor = outputs[iouSpec.name];
    const maskTensor = outputs[maskSpec.name];
    if (!iouTensor || !maskTensor) {
      throw new InvalidStateError(
        `Prompt decoder run is missing '${iouSpec.name}' or '${maskSpec.name}' outputs`,
      );
    }

    // pred_masks is [B, numObjects, numCandidates, H, W]; M1 decodes one
    // object per call, so everything reads from object slot 0.
    const shape = maskTensor.shape;
    if (shape.length !== 5) {
      throw new InvalidStateError(
        `Expected 5-D mask logits [B,N,C,H,W], got shape [${shape.join(', ')}]`,
      );
    }
    const candidates = shape[2] as number;
    const gridH = shape[3] as number;
    const gridW = shape[4] as number;
    if (gridH !== gridW) {
      throw new InvalidStateError(`Expected a square logit grid, got ${gridH}x${gridW}`);
    }

    const iou = (await state.backend.readback(iouTensor)) as Float32Array;
    const logits = (await state.backend.readback(maskTensor)) as Float32Array;
    if (!(iou instanceof Float32Array) || !(logits instanceof Float32Array)) {
      throw new InvalidStateError('Decoder outputs read back as unexpected typed-array kinds');
    }

    // iou_scores is [B, numObjects, numCandidates]; object 0 occupies the
    // first `candidates` entries.
    let selected: number[];
    if (req.multimask === true) {
      selected = Array.from({ length: candidates }, (_, i) => i);
    } else {
      let best = 0;
      for (let c = 1; c < candidates; c++) {
        if ((iou[c] as number) > (iou[best] as number)) best = c;
      }
      selected = [best];
    }

    const objectId = req.objectId ?? 0;
    const per = gridH * gridW;
    const payloads: MaskPayload[] = [];
    for (const c of selected) {
      const mask = logitsToSourceMask(
        logits.subarray(c * per, (c + 1) * per),
        gridH,
        encoded.transform,
      );
      payloads.push({
        objectId,
        score: iou[c] as number,
        width: encoded.width,
        height: encoded.height,
        binaryMask: mask.buffer as ArrayBuffer,
      });
    }
    return Comlink.transfer(
      payloads,
      payloads.map((p) => p.binaryMask),
    );
  }

  /** Release the slot's cached embeddings and forget the id. */
  closeSession(sessionId: number): void {
    this.#requireState('closeSession');
    const slot = this.#requireSlot(sessionId);
    disposeEmbeddings(slot);
    this.#sessions.delete(sessionId);
  }

  // ---------------------------------------------------------------------
  // M2 video path (§4/§5/§6). CROSS-FILE NOTE (flagged for the
  // orchestrator): `init()` above unconditionally requires M1_ROLES
  // (`visionEncoder`/`promptDecoder`). If a video-only tier's manifest
  // ships ONLY the four video roles (no M1-compatible image graphs),
  // `init()` fails before any video RPC is ever reached — a tools/export
  // manifest-shape concern (not a `src/` one) that pre-dates this file.
  // ---------------------------------------------------------------------

  #requireVideoSlot(sessionId: number): VideoSessionSlot {
    const slot = this.#videoSessions.get(sessionId);
    if (!slot) {
      throw new InvalidStateError(`Unknown or closed worker video session id ${sessionId}`);
    }
    return slot;
  }

  #requireVideoEngine(sessionId: number, method: string): VideoEngine {
    const slot = this.#requireVideoSlot(sessionId);
    if (!slot.videoEngine) {
      throw new InvalidStateError(`WorkerEngine.${method} called before attachVideoSource on session ${sessionId}`);
    }
    return slot.videoEngine;
  }

  /**
   * Compile the four video graphs (+ `noMemCondition` when
   * `video.initPath === 'noMemGraph'`, §10 PIN-2) ONCE per worker, shared by
   * every video session — mirrors the M1 encoder/decoder compile-once
   * pattern. Concurrent callers share the same in-flight load.
   */
  async #ensureVideoAssets(): Promise<VideoAssets> {
    if (this.#videoAssets) return this.#videoAssets;
    if (this.#videoAssetsLoading) return this.#videoAssetsLoading;

    const state = this.#requireState('attachVideoSource');
    const loading = (async (): Promise<VideoAssets> => {
      const loadConfig = { modelBaseUrl: state.modelBaseUrl, cache: state.cache, quantPreference: state.quantPreference };
      const core = await loadModelAssets(state.spec, { ...loadConfig, roles: VIDEO_CORE_ROLES });
      const video = core.manifest.video;
      if (!video) {
        throw new InvalidStateError(`Model '${state.spec.id}' manifest ships no video graphs`);
      }

      let noMemBytes: Uint8Array | undefined;
      if (video.initPath === 'noMemGraph') {
        const extra = await loadModelAssets(state.spec, { ...loadConfig, roles: ['noMemCondition'] });
        noMemBytes = extra.graphs.get('noMemCondition');
        if (!noMemBytes) {
          throw new InvalidStateError(`Model '${state.spec.id}' manifest.video.initPath is 'noMemGraph' but ships no 'noMemCondition' graph`);
        }
      }

      // §4.5 IOBindingPlan: persistent-loop outputs stay 'device' on webgpu;
      // maskLogits/iouScores/objectPointer/objectScoreLogits come back 'cpu'.
      const hotLocation = state.device === 'webgpu' ? 'device' : 'cpu';
      const deviceOutputs = (entry: GraphManifestEntry, keys: readonly string[]): IOBindingPlan => {
        const plan: IOBindingPlan = { outputLocations: {} };
        for (const key of keys) {
          const spec = entry.outputs[key];
          if (spec) plan.outputLocations[spec.name] = hotLocation;
        }
        return plan;
      };
      const cpuOutputs = (entry: GraphManifestEntry, keys: readonly string[]): IOBindingPlan => {
        const plan: IOBindingPlan = { outputLocations: {} };
        for (const key of keys) {
          const spec = entry.outputs[key];
          if (spec) plan.outputLocations[spec.name] = 'cpu';
        }
        return plan;
      };

      const bytesFor = (role: GraphRole): Uint8Array => {
        const bytes = core.graphs.get(role);
        if (!bytes) throw new InvalidStateError(`loadModelAssets did not return bytes for '${role}'`);
        return bytes;
      };
      const entryFor = (role: GraphRole): GraphManifestEntry => {
        const entry = core.manifest.graphs[role];
        if (!entry) throw new InvalidStateError(`Video manifest is missing graph role '${role}'`);
        return entry;
      };

      const videoEncoderEntry = entryFor('videoEncoder');
      const memoryAttentionEntry = entryFor('memoryAttention');
      const maskDecoderEntry = entryFor('maskDecoderVideo');
      const memoryEncoderEntry = entryFor('memoryEncoder');

      const videoEncoder = await state.backend.createSession(
        { name: 'videoEncoder', bytes: bytesFor('videoEncoder') },
        deviceOutputs(videoEncoderEntry, ['visionFeatures', 'visionPos', 'highRes0', 'highRes1']),
      );
      const memoryAttention = await state.backend.createSession(
        { name: 'memoryAttention', bytes: bytesFor('memoryAttention') },
        deviceOutputs(memoryAttentionEntry, ['conditionedFeatures']),
      );
      const maskDecoderVideo = await state.backend.createSession(
        { name: 'maskDecoderVideo', bytes: bytesFor('maskDecoderVideo') },
        cpuOutputs(maskDecoderEntry, ['maskLogits', 'iouScores', 'objectPointer', 'objectScoreLogits']),
      );
      const memoryEncoder = await state.backend.createSession(
        { name: 'memoryEncoder', bytes: bytesFor('memoryEncoder') },
        deviceOutputs(memoryEncoderEntry, ['memoryFeatures', 'memoryPos']),
      );
      const noMemCondition = noMemBytes
        ? await state.backend.createSession(
            { name: 'noMemCondition', bytes: noMemBytes },
            deviceOutputs(entryFor('noMemCondition'), ['conditionedFeatures']),
          )
        : undefined;

      const graphs: VideoEngineGraphs = { videoEncoder, memoryAttention, maskDecoderVideo, memoryEncoder, noMemCondition };
      this.#videoAssets = { manifest: core.manifest, graphs };
      return this.#videoAssets;
    })();
    this.#videoAssetsLoading = loading;
    try {
      return await loading;
    } finally {
      this.#videoAssetsLoading = undefined;
    }
  }

  /** Open a video session slot (cheap: video graphs load lazily, on `attachVideoSource`). */
  async createVideoSession(): Promise<number> {
    this.#requireState('createVideoSession');
    const id = this.#nextVideoSessionId++;
    this.#videoSessions.set(id, { videoEngine: undefined });
    return id;
  }

  /**
   * Demux `source` (a structured-clone-by-reference Blob) and attach it to
   * the session's {@link VideoEngine}, lazily compiling the shared video
   * graphs on first use. Per §6.1 this is where an unsupported/video-less
   * manifest surfaces as `InvalidStateError` — NOT at `createVideoSession`.
   */
  async attachVideoSource(sessionId: number, source: Blob): Promise<VideoSourceInfo> {
    const state = this.#requireState('attachVideoSource');
    const slot = this.#requireVideoSlot(sessionId);
    if (slot.videoEngine) {
      throw new InvalidStateError(`VideoEngine on session ${sessionId} already has a source attached`);
    }
    const assets = await this.#ensureVideoAssets();
    const frameSource = await createWebCodecsFrameSource(source);
    const videoEngine = new VideoEngine({
      backend: state.backend,
      manifest: assets.manifest,
      spec: state.spec,
      graphs: assets.graphs,
    });
    videoEngine.attach(frameSource);
    slot.videoEngine = videoEngine;
    return frameSource.info;
  }

  async addVideoObject(
    sessionId: number,
    req: { frameIndex: number; prompts: Prompt[]; objectId?: number; epoch: number },
  ): Promise<VideoObjectResult> {
    const engine = this.#requireVideoEngine(sessionId, 'addVideoObject');
    const { objectId, mask } = await engine.addObject(req);
    return Comlink.transfer({ objectId, epoch: req.epoch, mask }, [mask.binaryMask]);
  }

  async refineVideoObject(
    sessionId: number,
    req: { objectId: number; frameIndex: number; prompts: Prompt[]; epoch: number },
  ): Promise<VideoObjectResult> {
    const engine = this.#requireVideoEngine(sessionId, 'refineVideoObject');
    const mask = await engine.refineObject(req);
    return Comlink.transfer({ objectId: req.objectId, epoch: req.epoch, mask }, [mask.binaryMask]);
  }

  async removeVideoObject(sessionId: number, objectId: number): Promise<void> {
    const engine = this.#requireVideoEngine(sessionId, 'removeVideoObject');
    engine.removeObject(objectId);
  }

  /**
   * Starts the pull-credit propagation loop on `port` (§5.2) and resolves
   * once it is SCHEDULED — `runPropagationPort` drives the loop
   * fire-and-forget from here on, reporting `frame`/`done`/`error` over the
   * port until it closes.
   */
  async propagateVideo(sessionId: number, req: PropagateRequest, port: MessagePort): Promise<void> {
    const engine = this.#requireVideoEngine(sessionId, 'propagateVideo');
    runPropagationPort(port, (emit, isCancelled) =>
      engine.propagate({ startFrame: req.startFrame, endFrame: req.endFrame, epoch: req.epoch }, emit, isCancelled),
    );
  }

  async resetVideoSession(sessionId: number): Promise<void> {
    const slot = this.#requireVideoSlot(sessionId);
    slot.videoEngine?.reset();
  }

  async closeVideoSession(sessionId: number): Promise<void> {
    const slot = this.#videoSessions.get(sessionId);
    if (!slot) return; // idempotent, mirrors closeSession
    if (slot.videoEngine) await slot.videoEngine.dispose();
    this.#videoSessions.delete(sessionId);
  }

  /** Debug-only passthrough to `Backend.debugStats()` for the browser e2e flatness gate (§9.3). */
  debugStats(sessionId: number): { liveTensors: number; liveBytes: number } | undefined {
    this.#requireVideoSlot(sessionId);
    return this.#state?.backend.debugStats?.();
  }

  /**
   * Release every session slot and the backend (which sweeps its compiled
   * sessions and tracked tensors). Idempotent — the main thread terminates
   * the worker right after, so a second call must not throw.
   */
  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const slot of this.#sessions.values()) {
      disposeEmbeddings(slot);
    }
    this.#sessions.clear();
    for (const slot of this.#videoSessions.values()) {
      if (slot.videoEngine) await slot.videoEngine.dispose().catch(() => undefined);
    }
    this.#videoSessions.clear();
    this.#videoAssets = undefined;
    const state = this.#state;
    this.#state = undefined;
    if (state) {
      await state.backend.dispose();
    }
  }
}
