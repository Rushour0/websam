/**
 * @websam/core — SAM-family interactive image & video segmentation in the
 * browser (WebGPU-first, WASM fallback).
 *
 * M0 surface: contracts, errors, coordinate math, RLE, model registry, and
 * backend capability probing are real; heavy runtime entry points throw
 * {@link NotImplementedError} until their milestone lands.
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

import { NotImplementedError } from './errors.js';
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
}

/**
 * Create a segmenter: loads the model, picks device/quant, and returns the
 * interactive image/video segmentation API.
 *
 * @throws NotImplementedError — createSegmenter lands in M1. The config,
 * progress, and {@link Segmenter} types are the stable contract it will
 * implement; the returned promise rejects (never throws synchronously).
 */
export async function createSegmenter(_config: SegmenterConfig = {}): Promise<Segmenter> {
  throw new NotImplementedError('createSegmenter, lands in M1');
}
