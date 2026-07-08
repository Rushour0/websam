import type { RLEMask } from './masks/rle.js';
import type { ModelSpec } from './registry.js';

/**
 * A visual prompt, in **source-pixel coordinates** (see
 * `docs/coordinate-contract.md`). Mask prompts are the exception: they live in
 * decoder-logit space (a low-res logit map from a previous decode), not
 * source-pixel space.
 */
export type Prompt =
  | { type: 'point'; x: number; y: number; label: 0 | 1 }
  | { type: 'box'; x1: number; y1: number; x2: number; y2: number }
  | { type: 'mask'; mask: RLEMask };

/**
 * One segmentation result. Instances handed to callers are **immutable
 * snapshots** — internal scratch buffers are pooled, but a `MaskResult` never
 * mutates after it is yielded (see plan review finding on pooled-buffer
 * use-after-free).
 */
export interface MaskResult {
  readonly objectId: number;
  /** Predicted IoU / confidence score for this mask. */
  readonly score: number;
  /** Source-image dimensions the mask maps onto. */
  readonly width: number;
  readonly height: number;
  toImageData(): ImageData;
  toBitmap(): Promise<ImageBitmap>;
  /** Our documented row-major RLE (NOT COCO — use {@link toCocoRLE}). */
  toRLE(): RLEMask;
  /** Real COCO RLE: column-major, compressed-string counts. */
  toCocoRLE(): { size: [number, number]; counts: string };
  /** Row-major 0/1 bytes, `width * height`. */
  toBinary(): Uint8Array;
}

/** Result of {@link ImageSession.encode}. */
export interface EncodeResult {
  width: number;
  height: number;
  /** Wall-clock encoder time, for perf surfaces. */
  encodeMs: number;
}

/**
 * Encode-once / decode-per-click interactive image segmentation.
 * Lands in M1; the shape below is the stable contract.
 */
export interface ImageSession {
  /** Run the vision encoder once; embeddings stay device-resident. */
  encode(
    image: ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas,
    options?: { signal?: AbortSignal },
  ): Promise<EncodeResult>;
  /** Decode masks for prompts against the cached embedding (fast, per click). */
  decode(prompts: Prompt[], options?: { multimask?: boolean; objectId?: number }): Promise<MaskResult[]>;
  readonly isEncoded: boolean;
  dispose(): void;
}

/** Per-frame result yielded by {@link VideoSession.propagate}. */
export interface FramePropagationResult {
  frameIndex: number;
  timestampUs: number;
  /** One mask per tracked object, keyed order stable across frames. */
  masks: MaskResult[];
}

/**
 * Interactive video object tracking (memory-attention loop). Lands in M2
 * (EdgeTAM tier) / M3 (SAM3 tiers).
 *
 * Iterator contract: each `propagate()` iterator captures the session epoch at
 * creation. `refineObject()` bumps the epoch; an in-flight iterator then
 * throws `EpochInvalidatedError` on its next `next()` (never a silent stop).
 * `break`/`return()` cancels worker-side work and releases buffers. One
 * active iterator per session; a second `propagate()` rejects with
 * `InvalidStateError`.
 */
export interface VideoSession {
  attachSource(
    source: Blob | HTMLVideoElement,
  ): Promise<{ frameCount?: number; fps: number; width: number; height: number }>;
  addObject(options: {
    frameIndex: number;
    prompts: Prompt[];
    objectId?: number;
  }): Promise<{ objectId: number; mask: MaskResult }>;
  refineObject(objectId: number, frameIndex: number, prompts: Prompt[]): Promise<MaskResult>;
  removeObject(objectId: number): void;
  propagate(options?: {
    startFrame?: number;
    endFrame?: number;
    direction?: 'forward' | 'backward';
    signal?: AbortSignal;
  }): AsyncIterableIterator<FramePropagationResult>;
  reset(): void;
  dispose(): void;
}

/** The resolved model a {@link Segmenter} loaded, for UI/telemetry surfaces. */
export interface ResolvedModelInfo {
  spec: ModelSpec;
  /** Quantization variant actually loaded (after `'auto'` resolution). */
  quant: 'fp32' | 'fp16' | 'int8' | 'q4f16';
  totalBytes: number;
}

/**
 * The loaded segmentation engine: owns the weights and the worker; sessions
 * are cheap views over it. Returned by `createSegmenter` (M1).
 */
export interface Segmenter {
  /** Device that was actually selected after `'auto'` probing. */
  readonly device: 'webgpu' | 'wasm';
  readonly model: ResolvedModelInfo;
  createImageSession(): Promise<ImageSession>;
  /** Rejects with `UnsupportedDeviceError` if the model tier has no video support on this device. */
  createVideoSession(): Promise<VideoSession>;
  dispose(): Promise<void>;
}
