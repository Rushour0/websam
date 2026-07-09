/**
 * @websam3/core — SAM-family interactive image & video segmentation in the
 * browser (WebGPU-first, WASM fallback).
 *
 * M1 surface: the interactive image path is real — {@link createSegmenter}
 * loads a model into a module worker and {@link ImageSession} runs
 * encode-once / decode-per-click inference. Video sessions land in M2 and
 * still reject with `NotImplementedError`.
 */

export {
  WebsamError,
  NotImplementedError,
  UnsupportedDeviceError,
  CrossOriginIsolationRequiredError,
  WeightVerifyError,
  OutOfMemoryError,
  EpochInvalidatedError,
  InvalidStateError,
  type WebsamErrorCode,
} from './errors.js';

export type {
  TensorLocation,
  DType,
  DeviceTensor,
  IOBindingPlan,
  GraphAsset,
  BackendSession,
  Backend,
} from './backend/backend.js';

export {
  WebGpuBackend,
  type WebGpuProbeResult,
  type OrtModule,
} from './backend/webgpu-backend.js';
export { WasmBackend, type WasmProbeResult } from './backend/wasm-backend.js';

export {
  computeTransform,
  sourceToModel,
  modelToSource,
  type TransformMode,
  type CoordinateTransform,
  type Point,
} from './coords.js';

export {
  encodeRLE,
  decodeRLE,
  toCocoRLE,
  type RLEMask,
  type CocoRLE,
} from './masks/rle.js';

export {
  registerModel,
  getModel,
  listModels,
  type ModelSpec,
  type ModelDeviceSupport,
} from './registry.js';

export type {
  Segmenter,
  ImageSession,
  VideoSession,
  MaskResult,
  Prompt,
  EncodeResult,
  FramePropagationResult,
  ResolvedModelInfo,
} from './segmenter.js';

import { createSegmenterImpl } from './sessions/segmenter-impl.js';
import type { Segmenter } from './segmenter.js';

/** Phases of model load, in the order they normally occur. */
export type LoadPhase =
  | 'manifest'
  | 'download'
  | 'verify'
  | 'compile'
  | 'ready'
  | 'offline-cache';

/** Progress event delivered to {@link SegmenterConfig.onProgress} during load. */
export interface LoadProgressEvent {
  /**
   * Load phase: `'manifest'` (fetching the model manifest), `'download'`
   * (weight files), `'verify'` (digest checks), `'compile'` (backend session
   * creation), `'ready'` (usable), `'offline-cache'` (files served from the
   * local cache instead of the network).
   */
  phase: LoadPhase;
  /** Bytes completed within the current phase, when measurable. */
  loaded?: number;
  /** Total bytes for the current phase, when known. */
  total?: number;
  /** File currently being processed (download/verify phases). */
  file?: string;
}

/** Configuration for {@link createSegmenter}. */
export interface SegmenterConfig {
  /** Registered model id (see registry); defaults to the best tier for the device. */
  model?: string;
  /** Execution device; `'auto'` (default) probes WebGPU and falls back to WASM. */
  device?: 'webgpu' | 'wasm' | 'auto';
  /** Weight quantization; `'auto'` (default) picks per device capability (e.g. f16 support). */
  quant?: 'auto' | 'fp16' | 'int8' | 'q4f16';
  /** Override the base URL weights are fetched from (self-hosting; defaults to the manifest's host). */
  modelBaseUrl?: string;
  /** Cache verified weights locally (OPFS) for offline reload. Default true. */
  cache?: boolean;
  /**
   * Explicit license acceptance for SAM-licensed models
   * (`requiresLicenseAcceptance` in the registry). Loading such a model
   * without `acceptLicense: 'sam'` rejects.
   */
  acceptLicense?: 'sam';
  /** Progress callback for the load pipeline. */
  onProgress?: (event: LoadProgressEvent) => void;
  /**
   * Override the worker script URL (bundler escape hatch). By default the
   * worker is spawned from the `worker.js` sibling of the library bundle via
   * `new URL('./worker.js', import.meta.url)`; bundlers that break that
   * relative resolution can point this at wherever they serve
   * `@websam3/core/worker`.
   */
  workerUrl?: string | URL;
  /**
   * Forwarded to onnxruntime-web `env.wasm.wasmPaths` inside the worker: the
   * base URL its `.wasm`/`.mjs` assets are served from. Defaults to ort's own
   * `import.meta.url` resolution.
   */
  wasmPaths?: string;
}

/**
 * Create a segmenter: validates the config, resolves device and quantization
 * from main-thread capability probes, spawns the inference module worker,
 * loads + compiles the model inside it (progress via
 * {@link SegmenterConfig.onProgress}), and returns the interactive
 * image/video segmentation API.
 *
 * The returned promise rejects (never throws synchronously) with:
 * - `InvalidStateError` — unknown model id, invalid config values, or a
 *   license-gated model without `acceptLicense: 'sam'`.
 * - `UnsupportedDeviceError` — no usable execution device, or the resolved
 *   device is not in the model tier's support matrix.
 * - `WeightVerifyError` / other load failures — propagated from the worker
 *   with their `code` intact; the worker is terminated before rejecting.
 */
export async function createSegmenter(config: SegmenterConfig = {}): Promise<Segmenter> {
  return createSegmenterImpl(config);
}
