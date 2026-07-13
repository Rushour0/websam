/**
 * VideoEngine — per-frame step orchestration for the EdgeTAM memory-bank
 * video path (docs/m2-internal-contracts.md §4).
 *
 * Drives everything through the {@link Backend} + {@link MemoryBank} +
 * {@link FrameSource} abstractions established in wave 1 — never touches
 * onnxruntime-web or WebCodecs specifics directly. Object ids are
 * insertion-ordered (`Map` iteration order); results and propagated frames
 * key masks in that same order (the "stable order across frames" contract).
 *
 * MULTI-OBJECT EXECUTION (§4.3-§4.4 scoping decision, flagged for the
 * orchestrator): the doc's DECIDED batch-dim design requires splitting a
 * BATCHED device-resident output per object (`memoryEncoder`'s
 * `memoryFeatures`/`memoryPos` stay `'device'` per the §4.5 IOBindingPlan),
 * and `Backend` exposes no device-side slice-READ primitive — only the
 * ring-WRITE primitive `copyRegion` (src → one slot of dst). Splitting a
 * batched output per object is therefore impossible without an extra graph
 * neither `tools/export` nor this manifest shape provides. This engine
 * instead runs the ALWAYS-CORRECT "sequential fallback" path (§4.3's
 * `video.multiObjectBatch === false` branch) UNCONDITIONALLY for every
 * object, on every frame, regardless of the manifest flag: each object gets
 * its own B=1 `memoryAttention`/`maskDecoderVideo`/`memoryEncoder` call.
 * This is correct (batching is a perf optimization, not a numerics change —
 * B=1 per-object graphs are the same math) but leaves the `B = objects.length`
 * batched-call optimization as a follow-up once a batched-output-splitting
 * primitive exists.
 *
 * CROSS-FILE ASSUMPTION (flagged per the task brief): §2.2's semantic-key
 * table (`queries`/`objectPointers`/`tposIndices`/`noMem` as separate
 * memoryAttention inputs) is the one implemented here, matching wave 1's
 * `MemoryBank.assemble()` output shape. The EdgeTAM export spike
 * (`FINDINGS.md`) shows the REAL onnx graph instead takes one pre-assembled
 * `(1, kvLen, 64)` KV buffer + an additive bias — §10's PIN-1/PIN-9
 * contingency ("re-cut §2.2 before wave 2 starts") was never actioned in the
 * doc. Per the model-agnostic rule ("no src/ design changes when FINDINGS.md
 * lands — only manifest values"), this file binds every tensor by SEMANTIC
 * KEY read from the manifest, never a hardcoded ONNX name — reconciling the
 * spike's raw graph IO with the §2.2 key names is `tools/export`'s job, not
 * this file's. `tposDelivery: 'precombined'` (the spike's actual value) is
 * therefore NOT implemented here — {@link VideoEngine.propagate} and the
 * interaction methods throw `NotImplementedError` for it, exactly as §3.2
 * originally scoped ("'precombined' throws NotImplementedError at M2"); it
 * needs the manifest to carry a tpos-table weight, which `VideoManifestSection`
 * does not yet have a field for.
 */

import type { Backend, DeviceTensor } from '../../backend/backend.js';
import type { CoordinateTransform } from '../../coords.js';
import { computeTransform, sourceToModel } from '../../coords.js';
import { InvalidStateError, NotImplementedError } from '../../errors.js';
import type { ModelSpec } from '../../registry.js';
import { float16BitsToFloat32, float32ToFloat16Bits } from '../../runtime/float16.js';
import type { Prompt } from '../../segmenter.js';
import type { FrameSource } from '../../video/frame-source.js';
import type { GraphManifestEntry, ModelManifest, TensorSpec } from '../../weights/manifest.js';
import { logitsToSourceMask } from '../postprocess.js';
import { bitmapToTensor } from '../preprocess.js';
import type { MaskPayload } from '../protocol.js';
import type { VideoArchStrategy } from './arch-strategy.js';
import { strategyFor } from './arch-strategy.js';
import { MemoryBank } from './memory-bank.js';
import type { PropagationFrame } from './propagation-port.js';

/** The four graph roles every EdgeTAM-family video engine needs. */
export interface VideoEngineGraphs {
  videoEncoder: import('../../backend/backend.js').BackendSession;
  memoryAttention: import('../../backend/backend.js').BackendSession;
  maskDecoderVideo: import('../../backend/backend.js').BackendSession;
  memoryEncoder: import('../../backend/backend.js').BackendSession;
  /** Present iff `manifest.video.initPath === 'noMemGraph'` (§10 PIN-2). */
  noMemCondition?: import('../../backend/backend.js').BackendSession;
}

export interface VideoEngineInit {
  backend: Backend;
  /** `manifest.video` is required; throws `InvalidStateError` otherwise. */
  manifest: ModelManifest;
  spec: ModelSpec;
  graphs: VideoEngineGraphs;
}

interface ObjectState {
  bank: MemoryBank;
}

/** Look up a semantic tensor key, failing loudly when the manifest lacks it. */
function requireTensorSpec(map: Record<string, TensorSpec>, key: string, where: string): TensorSpec {
  const spec = map[key];
  if (!spec) {
    throw new InvalidStateError(
      `Model manifest is missing semantic tensor key '${key}' in ${where} (has: ${
        Object.keys(map).join(', ') || 'none'
      })`,
    );
  }
  return spec;
}

function requireEntry(
  graphs: ModelManifest['graphs'],
  role: string,
  where: string,
): GraphManifestEntry {
  const entry = graphs[role];
  if (!entry) throw new InvalidStateError(`${where}: manifest has no graph role '${role}'`);
  return entry;
}

/** Flat model-input-space prompt tensors for `maskDecoderVideo` (exact count, no padding — FINDINGS.md gotcha 6: the real graph exports `num_points` as a dynamic axis). */
function buildVideoPrompts(
  prompts: readonly Prompt[],
  transform: CoordinateTransform,
): { points: Float32Array; labels: BigInt64Array; count: number } {
  const points: number[] = [];
  const labels: bigint[] = [];
  for (const prompt of prompts) {
    if (prompt.type === 'point') {
      const p = sourceToModel({ x: prompt.x, y: prompt.y }, transform);
      points.push(p.x, p.y);
      labels.push(BigInt(prompt.label));
    } else if (prompt.type === 'box') {
      const a = sourceToModel({ x: prompt.x1, y: prompt.y1 }, transform);
      const b = sourceToModel({ x: prompt.x2, y: prompt.y2 }, transform);
      points.push(a.x, a.y, b.x, b.y);
      labels.push(2n, 3n);
    } else {
      throw new NotImplementedError('mask prompts, lands in M2');
    }
  }
  if (points.length === 0) {
    // The no-prompt placeholder point every tracked/propagation frame uses.
    points.push(0, 0);
    labels.push(-1n);
  }
  return { points: Float32Array.from(points), labels: BigInt64Array.from(labels), count: labels.length };
}

/** One object's decode result, still holding its device-resident scratch for the caller to dispose. */
interface DecodedObject {
  objectId: number;
  score: number;
  occluded: boolean;
  maskLogits: DeviceTensor; // 'cpu'-located per IOBindingPlan
  objectPointer: Float32Array;
}

export class VideoEngine {
  readonly #backend: Backend;
  readonly #manifest: ModelManifest;
  readonly #video: NonNullable<ModelManifest['video']>;
  readonly #strategy: VideoArchStrategy;
  readonly #graphs: VideoEngineGraphs;
  readonly #location: 'device' | 'cpu';

  readonly #videoEncoderEntry: GraphManifestEntry;
  readonly #memoryAttentionEntry: GraphManifestEntry;
  readonly #maskDecoderEntry: GraphManifestEntry;
  readonly #memoryEncoderEntry: GraphManifestEntry;

  #source: FrameSource | undefined;
  #transform: CoordinateTransform | undefined;
  readonly #objects = new Map<number, ObjectState>();
  #nextObjectId = 1;
  #disposed = false;

  constructor(init: VideoEngineInit) {
    if (!init.manifest.video) {
      throw new InvalidStateError('VideoEngine requires manifest.video (this tier ships no video graphs)');
    }
    this.#backend = init.backend;
    this.#manifest = init.manifest;
    this.#video = init.manifest.video;
    this.#strategy = strategyFor(init.spec.arch, this.#video);
    this.#graphs = init.graphs;
    this.#location = init.backend.kind === 'webgpu' ? 'device' : 'cpu';

    this.#videoEncoderEntry = requireEntry(init.manifest.graphs, 'videoEncoder', 'VideoEngine');
    this.#memoryAttentionEntry = requireEntry(init.manifest.graphs, 'memoryAttention', 'VideoEngine');
    this.#maskDecoderEntry = requireEntry(init.manifest.graphs, 'maskDecoderVideo', 'VideoEngine');
    this.#memoryEncoderEntry = requireEntry(init.manifest.graphs, 'memoryEncoder', 'VideoEngine');

    if (this.#video.initPath === 'noMemGraph' && !init.graphs.noMemCondition) {
      throw new InvalidStateError(
        "VideoEngine: manifest.video.initPath is 'noMemGraph' but no noMemCondition graph was provided",
      );
    }
  }

  #assertLive(method: string): void {
    if (this.#disposed) throw new InvalidStateError(`VideoEngine.${method} called after dispose()`);
  }

  #requireSource(method: string): FrameSource {
    this.#assertLive(method);
    if (!this.#source || !this.#transform) {
      throw new InvalidStateError(`VideoEngine.${method} called before attach()`);
    }
    return this.#source;
  }

  #requireObject(objectId: number): ObjectState {
    const obj = this.#objects.get(objectId);
    if (!obj) throw new InvalidStateError(`VideoEngine: unknown objectId ${objectId}`);
    return obj;
  }

  /** Compute the ONE {@link CoordinateTransform} for the attached source; video frames never change dims. */
  attach(source: FrameSource): void {
    this.#assertLive('attach');
    if (this.#source) throw new InvalidStateError('VideoEngine.attach called twice');
    this.#source = source;
    this.#transform = computeTransform(
      source.info.width,
      source.info.height,
      this.#manifest.preprocess.inputSize,
      this.#manifest.preprocess.mode,
    );
  }

  #newBank(): MemoryBank {
    return new MemoryBank({
      backend: this.#backend,
      video: this.#video,
      strategy: this.#strategy,
      location: this.#location,
    });
  }

  /**
   * NOTE: unlike the doc's §4.1 sketch, this returns `{objectId, mask}` —
   * NOT the wire-level `VideoObjectResult` (which also carries `epoch`).
   * Epoch is a session/protocol concept the engine has no state for (the
   * main thread is authoritative, §5.1); `WorkerEngine.addVideoObject`
   * (`engine.ts`) stamps `req.epoch` onto the result it hands back over
   * Comlink.
   */
  async addObject(req: {
    frameIndex: number;
    prompts: Prompt[];
    objectId?: number;
  }): Promise<{ objectId: number; mask: MaskPayload }> {
    this.#requireSource('addObject');
    const objectId = req.objectId ?? this.#nextObjectId;
    if (this.#objects.has(objectId)) {
      throw new InvalidStateError(`VideoEngine.addObject: duplicate objectId ${objectId}`);
    }
    this.#nextObjectId = Math.max(this.#nextObjectId, objectId + 1);
    const obj: ObjectState = { bank: this.#newBank() };
    this.#objects.set(objectId, obj);
    const mask = await this.#interact(objectId, obj, req.frameIndex, req.prompts, /* isCond */ true);
    return { objectId, mask };
  }

  /** See {@link addObject}'s note: returns the mask directly, not a `VideoObjectResult`. */
  async refineObject(req: { objectId: number; frameIndex: number; prompts: Prompt[] }): Promise<MaskPayload> {
    this.#requireSource('refineObject');
    const obj = this.#requireObject(req.objectId);
    obj.bank.invalidateAfter(req.frameIndex);
    return this.#interact(req.objectId, obj, req.frameIndex, req.prompts, /* isCond */ true);
  }

  removeObject(objectId: number): void {
    this.#assertLive('removeObject');
    const obj = this.#objects.get(objectId);
    if (!obj) return;
    obj.bank.dispose();
    this.#objects.delete(objectId);
  }

  /** §4.2 interaction step: encode → condition (memory or init path) → decode (real prompts) → memory-encode → commit. */
  async #interact(
    objectId: number,
    obj: ObjectState,
    frameIndex: number,
    prompts: Prompt[],
    isCond: boolean,
  ): Promise<MaskPayload> {
    const source = this.#requireSource('#interact');
    const transform = this.#transform!;
    const frame = await source.frameAt(frameIndex);
    let visionOut: { visionFeatures: DeviceTensor; visionPos: DeviceTensor; highRes0: DeviceTensor; highRes1: DeviceTensor };
    try {
      visionOut = await this.#runVideoEncoder(frame);
    } finally {
      frame.close();
    }
    const gridSize = this.#video.gridSize * 4; // low-res logit grid side, mirrors M1's maskSize convention.
    try {
      const conditioned = await this.#conditionOne(obj, frameIndex, visionOut.visionFeatures, visionOut.visionPos);
      let decoded: DecodedObject;
      try {
        decoded = await this.#decodeOne(objectId, conditioned, visionOut.highRes0, visionOut.highRes1, prompts);
      } finally {
        conditioned.dispose();
      }
      // maskLogits is 'float16' per the manifest; readback hands back raw
      // half bits (Backend contract) that must be unpacked before postprocess math.
      const logitsData = float16BitsToFloat32(
        (await this.#backend.readback(decoded.maskLogits)) as Uint16Array,
      );
      try {
        await this.#encodeAndCommit(obj, frameIndex, isCond, visionOut.visionFeatures, decoded);
      } finally {
        decoded.maskLogits.dispose();
      }
      return this.#maskPayloadFromDecoded(objectId, decoded, logitsData, gridSize, transform);
    } finally {
      visionOut.visionFeatures.dispose();
      visionOut.visionPos.dispose();
      visionOut.highRes0.dispose();
      visionOut.highRes1.dispose();
    }
  }

  /**
   * Run the video encoder for one frame: preprocess (`bitmapToTensor` with a
   * `VideoFrame`) → upload → `videoEncoder.run`. Disposes the pixel upload
   * immediately after the run returns (§4.5); the caller owns and disposes
   * the four outputs.
   */
  async #runVideoEncoder(
    frame: VideoFrame,
  ): Promise<{ visionFeatures: DeviceTensor; visionPos: DeviceTensor; highRes0: DeviceTensor; highRes1: DeviceTensor }> {
    const transform = this.#transform!;
    const chw = bitmapToTensor(frame, transform, this.#manifest.preprocess);
    const pixelsSpec = requireTensorSpec(this.#videoEncoderEntry.inputs, 'pixels', 'videoEncoder.inputs');
    const pixels = this.#backend.uploadTensor(
      float32ToFloat16Bits(chw),
      [1, 3, this.#manifest.preprocess.inputSize, this.#manifest.preprocess.inputSize],
      'float16',
    );
    let out: Record<string, DeviceTensor>;
    try {
      out = await this.#graphs.videoEncoder.run({ [pixelsSpec.name]: pixels });
    } finally {
      pixels.dispose();
    }
    const pick = (key: string): DeviceTensor => {
      const spec = requireTensorSpec(this.#videoEncoderEntry.outputs, key, 'videoEncoder.outputs');
      const tensor = out[spec.name];
      if (!tensor) throw new InvalidStateError(`videoEncoder.run produced no output '${spec.name}' ('${key}')`);
      return tensor;
    };
    return {
      visionFeatures: pick('visionFeatures'),
      visionPos: pick('visionPos'),
      highRes0: pick('highRes0'),
      highRes1: pick('highRes1'),
    };
  }

  /** §4.2 step 3 / §4.3 init branch: single-object (B=1) conditioning, memory path or init path. */
  async #conditionOne(
    obj: ObjectState,
    frameIndex: number,
    visionFeatures: DeviceTensor,
    visionPos: DeviceTensor,
  ): Promise<DeviceTensor> {
    if (this.#video.tposDelivery === 'precombined') {
      throw new NotImplementedError(
        "VideoEngine: video.tposDelivery 'precombined', lands post-M2 (manifest has no tpos-table weight yet)",
      );
    }
    if (!obj.bank.hasMemory(frameIndex)) {
      return this.#conditionNoMemory(visionFeatures, visionPos);
    }
    const asm = obj.bank.assemble(frameIndex);
    const scratch: DeviceTensor[] = [];
    try {
      const inputs = this.#memoryAttentionEntry.inputs;
      const feeds: Record<string, DeviceTensor> = {};

      const queriesSpec = requireTensorSpec(inputs, 'queries', 'memoryAttention.inputs');
      feeds[queriesSpec.name] = visionFeatures;
      const queriesPosSpec = requireTensorSpec(inputs, 'queriesPos', 'memoryAttention.inputs');
      feeds[queriesPosSpec.name] = visionPos;

      // The bank ring is `[M,T,C]`; the graph declares `memory_spatial`
      // batch-first `[1,M,T,C]`. Feed a zero-copy `[1,...]` view (disposed via
      // scratch; the bank still owns the underlying ring storage).
      const spatialSpec = requireTensorSpec(inputs, 'memorySpatial', 'memoryAttention.inputs');
      const spatialView = this.#backend.reshape(asm.memorySpatial, [1, ...asm.memorySpatial.shape]);
      scratch.push(spatialView);
      feeds[spatialSpec.name] = spatialView;
      const spatialPosSpec = requireTensorSpec(inputs, 'memorySpatialPos', 'memoryAttention.inputs');
      const spatialPosView = this.#backend.reshape(asm.memorySpatialPos, [
        1,
        ...asm.memorySpatialPos.shape,
      ]);
      scratch.push(spatialPosView);
      feeds[spatialPosSpec.name] = spatialPosView;

      if (this.#video.tposDelivery === 'indices') {
        const spec = requireTensorSpec(inputs, 'tposIndices', 'memoryAttention.inputs');
        const t = this.#backend.uploadTensor(asm.tposIndices, [1, asm.tposIndices.length], 'int64');
        scratch.push(t);
        feeds[spec.name] = t;
      }

      const maskSpec = requireTensorSpec(inputs, 'memoryMask', 'memoryAttention.inputs');
      const maskTensor = this.#backend.uploadTensor(asm.memoryMask, [1, asm.memoryMask.length], 'bool');
      scratch.push(maskTensor);
      feeds[maskSpec.name] = maskTensor;

      const ptrSpec = requireTensorSpec(inputs, 'objectPointers', 'memoryAttention.inputs');
      const p = this.#video.maxObjectPointers;
      const ptrTensor = this.#backend.uploadTensor(
        float32ToFloat16Bits(asm.objectPointers),
        [1, p, this.#video.embedDim],
        'float16',
      );
      scratch.push(ptrTensor);
      feeds[ptrSpec.name] = ptrTensor;

      const deltaSpec = requireTensorSpec(inputs, 'pointerDeltas', 'memoryAttention.inputs');
      const deltaTensor = this.#backend.uploadTensor(asm.pointerDeltas, [1, p], 'int64');
      scratch.push(deltaTensor);
      feeds[deltaSpec.name] = deltaTensor;

      const ptrMaskSpec = requireTensorSpec(inputs, 'pointerMask', 'memoryAttention.inputs');
      const ptrMaskTensor = this.#backend.uploadTensor(asm.pointerMask, [1, p], 'bool');
      scratch.push(ptrMaskTensor);
      feeds[ptrMaskSpec.name] = ptrMaskTensor;

      const out = await this.#graphs.memoryAttention.run(feeds);
      const outSpec = requireTensorSpec(this.#memoryAttentionEntry.outputs, 'conditionedFeatures', 'memoryAttention.outputs');
      const conditioned = out[outSpec.name];
      if (!conditioned) throw new InvalidStateError("memoryAttention.run produced no 'conditionedFeatures' output");
      for (const t of Object.values(out)) if (t !== conditioned) t.dispose();
      return conditioned;
    } finally {
      for (const t of scratch) t.dispose();
    }
  }

  async #conditionNoMemory(visionFeatures: DeviceTensor, visionPos: DeviceTensor): Promise<DeviceTensor> {
    if (this.#video.initPath === 'noMemGraph') {
      const graph = this.#graphs.noMemCondition!;
      const entry = requireEntry(this.#manifest.graphs, 'noMemCondition', 'VideoEngine');
      const inSpec = requireTensorSpec(entry.inputs, 'visionFeatures', 'noMemCondition.inputs');
      const out = await graph.run({ [inSpec.name]: visionFeatures });
      const outSpec = requireTensorSpec(entry.outputs, 'conditionedFeatures', 'noMemCondition.outputs');
      const conditioned = out[outSpec.name];
      if (!conditioned) throw new InvalidStateError("noMemCondition.run produced no 'conditionedFeatures' output");
      for (const [, t] of Object.entries(out)) if (t !== conditioned) t.dispose();
      return conditioned;
    }
    // 'noMemFlag': run memoryAttention with noMem=true and an all-invalid mask/pointer bank.
    const inputs = this.#memoryAttentionEntry.inputs;
    const scratch: DeviceTensor[] = [];
    try {
      const feeds: Record<string, DeviceTensor> = {};
      const queriesSpec = requireTensorSpec(inputs, 'queries', 'memoryAttention.inputs');
      feeds[queriesSpec.name] = visionFeatures;
      const queriesPosSpec = requireTensorSpec(inputs, 'queriesPos', 'memoryAttention.inputs');
      feeds[queriesPosSpec.name] = visionPos;

      const { maxCondFrames, numRecent, tokensPerMemoryMap, memDim, maxObjectPointers, embedDim, kvLen } = this.#video;
      const M = maxCondFrames + numRecent;
      // Batch-first `[1,M,T,C]` to match the graph input (no copyRegion here,
      // so the leading batch dim can be baked into the zero allocation).
      const zeroSpatial = this.#backend.allocTensor(
        [1, M, tokensPerMemoryMap, memDim],
        'float16',
        'cpu',
      );
      scratch.push(zeroSpatial);
      const zeroSpatialPos = this.#backend.allocTensor(
        [1, M, tokensPerMemoryMap, memDim],
        'float16',
        'cpu',
      );
      scratch.push(zeroSpatialPos);
      const spatialSpec = requireTensorSpec(inputs, 'memorySpatial', 'memoryAttention.inputs');
      feeds[spatialSpec.name] = zeroSpatial;
      const spatialPosSpec = requireTensorSpec(inputs, 'memorySpatialPos', 'memoryAttention.inputs');
      feeds[spatialPosSpec.name] = zeroSpatialPos;

      if (this.#video.tposDelivery === 'indices') {
        const spec = requireTensorSpec(inputs, 'tposIndices', 'memoryAttention.inputs');
        const t = this.#backend.uploadTensor(new BigInt64Array(M).fill(-1n), [1, M], 'int64');
        scratch.push(t);
        feeds[spec.name] = t;
      }

      const maskSpec = requireTensorSpec(inputs, 'memoryMask', 'memoryAttention.inputs');
      const maskTensor = this.#backend.uploadTensor(new Uint8Array(kvLen), [1, kvLen], 'bool');
      scratch.push(maskTensor);
      feeds[maskSpec.name] = maskTensor;

      const ptrSpec = requireTensorSpec(inputs, 'objectPointers', 'memoryAttention.inputs');
      const ptrTensor = this.#backend.uploadTensor(
        new Uint16Array(maxObjectPointers * embedDim),
        [1, maxObjectPointers, embedDim],
        'float16',
      );
      scratch.push(ptrTensor);
      feeds[ptrSpec.name] = ptrTensor;

      const deltaSpec = requireTensorSpec(inputs, 'pointerDeltas', 'memoryAttention.inputs');
      const deltaTensor = this.#backend.uploadTensor(new BigInt64Array(maxObjectPointers), [1, maxObjectPointers], 'int64');
      scratch.push(deltaTensor);
      feeds[deltaSpec.name] = deltaTensor;

      const ptrMaskSpec = requireTensorSpec(inputs, 'pointerMask', 'memoryAttention.inputs');
      const ptrMaskTensor = this.#backend.uploadTensor(new Uint8Array(maxObjectPointers), [1, maxObjectPointers], 'bool');
      scratch.push(ptrMaskTensor);
      feeds[ptrMaskSpec.name] = ptrMaskTensor;

      const noMemSpec = requireTensorSpec(inputs, 'noMem', 'memoryAttention.inputs');
      const noMemTensor = this.#backend.uploadTensor(new Uint8Array([1]), [1], 'bool');
      scratch.push(noMemTensor);
      feeds[noMemSpec.name] = noMemTensor;

      const out = await this.#graphs.memoryAttention.run(feeds);
      const outSpec = requireTensorSpec(this.#memoryAttentionEntry.outputs, 'conditionedFeatures', 'memoryAttention.outputs');
      const conditioned = out[outSpec.name];
      if (!conditioned) throw new InvalidStateError("memoryAttention.run produced no 'conditionedFeatures' output");
      for (const [, t] of Object.entries(out)) if (t !== conditioned) t.dispose();
      return conditioned;
    } finally {
      for (const t of scratch) t.dispose();
    }
  }

  /** `maskDecoderVideo` for one object (B=1), real prompts. */
  async #decodeOne(
    objectId: number,
    conditioned: DeviceTensor,
    highRes0: DeviceTensor,
    highRes1: DeviceTensor,
    prompts: Prompt[],
  ): Promise<DecodedObject> {
    const transform = this.#transform!;
    const built = buildVideoPrompts(prompts, transform);
    const inputs = this.#maskDecoderEntry.inputs;
    const scratch: DeviceTensor[] = [];
    try {
      const feeds: Record<string, DeviceTensor> = {};
      const condSpec = requireTensorSpec(inputs, 'conditionedFeatures', 'maskDecoderVideo.inputs');
      feeds[condSpec.name] = conditioned;
      const hr0Spec = requireTensorSpec(inputs, 'highRes0', 'maskDecoderVideo.inputs');
      feeds[hr0Spec.name] = highRes0;
      const hr1Spec = requireTensorSpec(inputs, 'highRes1', 'maskDecoderVideo.inputs');
      feeds[hr1Spec.name] = highRes1;

      const pointsSpec = requireTensorSpec(inputs, 'points', 'maskDecoderVideo.inputs');
      const points = this.#backend.uploadTensor(
        float32ToFloat16Bits(built.points),
        [1, 1, built.count, 2],
        'float16',
      );
      scratch.push(points);
      feeds[pointsSpec.name] = points;

      const labelsSpec = requireTensorSpec(inputs, 'labels', 'maskDecoderVideo.inputs');
      const labels = this.#backend.uploadTensor(built.labels, [1, 1, built.count], 'int64');
      scratch.push(labels);
      feeds[labelsSpec.name] = labels;

      const out = await this.#graphs.maskDecoderVideo.run(feeds);
      const maskSpec = requireTensorSpec(this.#maskDecoderEntry.outputs, 'maskLogits', 'maskDecoderVideo.outputs');
      const iouSpec = requireTensorSpec(this.#maskDecoderEntry.outputs, 'iouScores', 'maskDecoderVideo.outputs');
      const ptrSpec = requireTensorSpec(this.#maskDecoderEntry.outputs, 'objectPointer', 'maskDecoderVideo.outputs');
      const scoreSpec = requireTensorSpec(this.#maskDecoderEntry.outputs, 'objectScoreLogits', 'maskDecoderVideo.outputs');
      const maskLogits = out[maskSpec.name];
      const iouScores = out[iouSpec.name];
      const objectPointerT = out[ptrSpec.name];
      const objectScoreLogits = out[scoreSpec.name];
      if (!maskLogits || !iouScores || !objectPointerT || !objectScoreLogits) {
        throw new InvalidStateError('maskDecoderVideo.run is missing a required output');
      }
      const kept = new Set([maskLogits, iouScores, objectPointerT, objectScoreLogits]);
      for (const t of Object.values(out)) if (!kept.has(t)) t.dispose();

      // All maskDecoderVideo outputs are 'float16' per the manifest — unpack
      // the raw half bits Backend.readback hands back before any numeric use.
      const iou = float16BitsToFloat32((await this.#backend.readback(iouScores)) as Uint16Array);
      const score = iou.reduce((m, v) => Math.max(m, v), -Infinity);
      const scoreLogits = float16BitsToFloat32(
        (await this.#backend.readback(objectScoreLogits)) as Uint16Array,
      );
      const occluded = (scoreLogits[0] as number) < this.#video.occlusionThreshold;
      const pointer = float16BitsToFloat32((await this.#backend.readback(objectPointerT)) as Uint16Array);
      iouScores.dispose();
      objectScoreLogits.dispose();
      objectPointerT.dispose();

      return { objectId, score, occluded, maskLogits, objectPointer: Float32Array.from(pointer) };
    } finally {
      for (const t of scratch) t.dispose();
    }
  }

  /** `memoryEncoder` on the winning mask (B=1) → `bank.commit` / `bank.commitPointer`. */
  async #encodeAndCommit(
    obj: ObjectState,
    frameIndex: number,
    isCond: boolean,
    visionFeatures: DeviceTensor,
    decoded: DecodedObject,
  ): Promise<void> {
    const inputs = this.#memoryEncoderEntry.inputs;
    const feeds: Record<string, DeviceTensor> = {};
    const visSpec = requireTensorSpec(inputs, 'visionFeatures', 'memoryEncoder.inputs');
    feeds[visSpec.name] = visionFeatures;
    const maskSpec = requireTensorSpec(inputs, 'maskLogits', 'memoryEncoder.inputs');
    feeds[maskSpec.name] = decoded.maskLogits;
    // Prompted (conditioning) frames binarize the mask for memory; tracked
    // frames sigmoid it — the graph selects on this flag (FINDINGS.md §memory
    // encoder). The manifest declares this fp16, like every other video-graph
    // tensor (§4.3), so the host float is packed to half bits before upload.
    const promptedSpec = requireTensorSpec(inputs, 'isPrompted', 'memoryEncoder.inputs');
    const promptedTensor = this.#backend.uploadTensor(
      float32ToFloat16Bits(Float32Array.of(isCond ? 1 : 0)),
      [1],
      'float16',
    );
    feeds[promptedSpec.name] = promptedTensor;

    const shouldCommit = !decoded.occluded || this.#strategy.commitOccludedMemory || isCond;
    let out: Record<string, DeviceTensor>;
    try {
      out = await this.#graphs.memoryEncoder.run(feeds);
    } finally {
      promptedTensor.dispose();
    }
    const featSpec = requireTensorSpec(this.#memoryEncoderEntry.outputs, 'memoryFeatures', 'memoryEncoder.outputs');
    const posSpec = requireTensorSpec(this.#memoryEncoderEntry.outputs, 'memoryPos', 'memoryEncoder.outputs');
    const memoryFeatures = out[featSpec.name];
    const memoryPos = out[posSpec.name];
    if (!memoryFeatures || !memoryPos) {
      throw new InvalidStateError('memoryEncoder.run is missing memoryFeatures/memoryPos');
    }
    for (const t of Object.values(out)) if (t !== memoryFeatures && t !== memoryPos) t.dispose();
    try {
      if (shouldCommit) {
        obj.bank.commit(frameIndex, isCond, memoryFeatures, memoryPos);
        obj.bank.commitPointer(frameIndex, decoded.objectPointer);
      }
    } finally {
      memoryFeatures.dispose();
      memoryPos.dispose();
    }
  }

  #maskPayloadFromDecoded(
    objectId: number,
    decoded: DecodedObject,
    logitsData: Float32Array,
    gridSize: number,
    transform: CoordinateTransform,
  ): MaskPayload {
    const mask = logitsToSourceMask(logitsData, gridSize, transform);
    return {
      objectId,
      score: decoded.score,
      width: transform.srcW,
      height: transform.srcH,
      binaryMask: mask.buffer as ArrayBuffer,
    };
  }

  /**
   * Run the propagation loop over `[req.startFrame, req.endFrame)`. `emit`
   * resolves once a pull credit is available (§5.2); `isCancelled` is polled
   * between frames.
   */
  async propagate(
    req: { startFrame: number; endFrame: number; epoch: number },
    emit: (frame: PropagationFrame) => Promise<void>,
    isCancelled: () => boolean,
  ): Promise<void> {
    const source = this.#requireSource('propagate');
    const transform = this.#transform!;
    if (this.#video.tposDelivery === 'precombined') {
      throw new NotImplementedError(
        "VideoEngine: video.tposDelivery 'precombined', lands post-M2 (manifest has no tpos-table weight yet)",
      );
    }
    const gridSize = this.#video.gridSize * 4; // low-res logit grid side, mirrors M1's maskSize convention.

    // `for await...of` auto-invokes `reader.return()` on `break`/throw, which
    // per the FrameSource contract closes any frames decoded but not yet
    // yielded — no manual cleanup needed here.
    for await (const decoded of source.read({ startFrame: req.startFrame, endFrame: req.endFrame })) {
      if (isCancelled()) {
        decoded.frame.close();
        break;
      }
      let visionOut: { visionFeatures: DeviceTensor; visionPos: DeviceTensor; highRes0: DeviceTensor; highRes1: DeviceTensor };
      try {
        visionOut = await this.#runVideoEncoder(decoded.frame);
      } finally {
        decoded.frame.close();
      }
      try {
        const masks: MaskPayload[] = [];
        for (const [objectId, obj] of this.#objects) {
          const conditioned = await this.#conditionOne(obj, decoded.frameIndex, visionOut.visionFeatures, visionOut.visionPos);
          let result: DecodedObject;
          try {
            result = await this.#decodeOne(objectId, conditioned, visionOut.highRes0, visionOut.highRes1, []);
          } finally {
            conditioned.dispose();
          }
          const logitsData = float16BitsToFloat32(
            (await this.#backend.readback(result.maskLogits)) as Uint16Array,
          );
          try {
            await this.#encodeAndCommit(obj, decoded.frameIndex, /* isCond */ false, visionOut.visionFeatures, result);
          } finally {
            result.maskLogits.dispose();
          }
          masks.push(this.#maskPayloadFromDecoded(objectId, result, logitsData, gridSize, transform));
        }
        await emit({
          frameIndex: decoded.frameIndex,
          timestampUs: decoded.timestampUs,
          epoch: req.epoch,
          masks,
        });
      } finally {
        visionOut.visionFeatures.dispose();
        visionOut.visionPos.dispose();
        visionOut.highRes0.dispose();
        visionOut.highRes1.dispose();
      }
      if (isCancelled()) break;
    }
  }

  reset(): void {
    this.#assertLive('reset');
    for (const obj of this.#objects.values()) obj.bank.reset();
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const obj of this.#objects.values()) obj.bank.dispose();
    this.#objects.clear();
    if (this.#source) await this.#source.close();
  }
}
