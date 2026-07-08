/**
 * The worker protocol — the structured-clone-safe types that cross the
 * main-thread ⇄ worker boundary via Comlink.
 *
 * Implemented VERBATIM from docs/m1-internal-contracts.md §3.1: the sessions
 * layer (main thread) imports these types and codes against them, so any
 * drift here is integration drift and must go back to the contracts doc.
 *
 * Coordinate decision (§3.2, normative): the worker receives prompts in
 * SOURCE-pixel space and converts internally — the worker computes the
 * {@link CoordinateTransform} at `encodeImage` time and returns it so the
 * main thread can stamp it onto `MaskResultImpl`.
 */

import type { CoordinateTransform } from '../coords.js';
import type { LoadProgressEvent } from '../index.js';
import type { ModelSpec } from '../registry.js';
import type { Prompt } from '../segmenter.js';
import type { Quant } from '../weights/manifest.js';

/** Everything `WorkerEngine.init` needs; plain objects only — clones fine. */
export interface WorkerInitRequest {
  /** Resolved registry spec for the model tier (plain object — clones fine). */
  spec: ModelSpec;
  /** Device resolved ON THE MAIN THREAD by resolveDevice (§2.3), before spawning. */
  device: 'webgpu' | 'wasm';
  /** Ordered quant preference from resolveDevice; the loader picks the first available for all roles. */
  quantPreference: readonly Quant[];
  /** Rebase for weight file paths AND the manifest itself when set. */
  modelBaseUrl?: string;
  /** false → weights are verified but never persisted (MemoryWeightStore). */
  cache: boolean;
  /** Forwarded to onnxruntime-web `env.wasm.wasmPaths` inside the worker. */
  wasmPaths?: string;
}

/** What `WorkerEngine.init` resolves with (feeds `ResolvedModelInfo`). */
export interface WorkerInitResult {
  device: 'webgpu' | 'wasm';
  /** Quantization variant actually loaded. */
  quant: Quant;
  totalBytes: number;
}

/** Result of `WorkerEngine.encodeImage`. */
export interface EncodeResponse {
  width: number;
  height: number;
  /** Wall-clock encoder time (preprocess + vision-encoder run), ms. */
  encodeMs: number;
  /** Computed IN the worker from bitmap dims + manifest.preprocess (mode, inputSize). */
  transform: CoordinateTransform;
}

/** Prompts for one decode; prompts are in SOURCE-pixel space (§3.2). */
export interface DecodeRequest {
  prompts: Prompt[];
  /** true → return all mask candidates; default: only the best-IoU candidate. */
  multimask?: boolean;
  /** Stamped onto every returned {@link MaskPayload}; defaults to 0. */
  objectId?: number;
}

/** One decoded mask, shipped back with its buffer TRANSFERRED (zero-copy). */
export interface MaskPayload {
  objectId: number;
  /** Best `iou_scores` entry for the chosen mask. */
  score: number;
  /** Source-pixel dims. */
  width: number;
  height: number;
  /** Row-major 0/1 bytes, width*height — TRANSFERRED (zero-copy) to the main thread. */
  binaryMask: ArrayBuffer;
  // lowResLogits: intentionally absent in M1 — mask-prompt feedback lands with video (M2).
}

/**
 * The API the worker exposes via `Comlink.expose` and the main thread wraps
 * via `Comlink.wrap`. `onProgress` crosses as a Comlink proxy callback; the
 * engine may call it fire-and-forget.
 */
export interface WorkerEngineApi {
  init(
    req: WorkerInitRequest,
    onProgress?: (e: LoadProgressEvent) => void,
  ): Promise<WorkerInitResult>;
  /** Open a session slot; returns its id. */
  createSession(): Promise<number>;
  /** Preprocess + run the vision encoder; the worker CLOSES the bitmap (it owns it post-transfer). */
  encodeImage(sessionId: number, bitmap: ImageBitmap): Promise<EncodeResponse>;
  /** Decode masks for prompts (source pixels) against the session's cached embeddings. */
  decode(sessionId: number, req: DecodeRequest): Promise<MaskPayload[]>;
  closeSession(sessionId: number): void;
  dispose(): Promise<void>;
}
