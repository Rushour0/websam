import {
  InvalidStateError,
  NotImplementedError,
  OutOfMemoryError,
  UnsupportedDeviceError,
} from '../errors.js';
import { createOrtSession, OrtBackendSession } from '../runtime/ort-session.js';
import {
  allocCpuTensor,
  createCpuTensor,
  OrtDeviceTensor,
  readbackTensor,
} from '../runtime/ort-tensor.js';
import type {
  Backend,
  BackendSession,
  DeviceTensor,
  DType,
  GraphAsset,
  IOBindingPlan,
  TensorLocation,
} from './backend.js';

/**
 * The onnxruntime-web module namespace, injected rather than imported so the
 * backend never forces ort into a bundle (and so tests can stub it).
 */
export type OrtModule = typeof import('onnxruntime-web');

/** Bytes per element for each {@link DType} (used by the census + slot math). */
const DTYPE_BYTES = {
  float32: 4,
  float16: 2,
  int64: 8,
  uint8: 1,
  int32: 4,
  bool: 1,
} as const satisfies Record<DType, number>;

/** Total byte size of a tensor of `shape` and `dtype`. */
function tensorByteLength(shape: readonly number[], dtype: DType): number {
  let elements = 1;
  for (const dim of shape) elements *= dim;
  return elements * DTYPE_BYTES[dtype];
}

/** Element count of one slot (`prod(shape.slice(1))`) of a ring tensor. */
function slotElementCount(shape: readonly number[]): number {
  let elements = 1;
  for (let i = 1; i < shape.length; i += 1) elements *= shape[i] ?? 0;
  return elements;
}

/**
 * Shared {@link Backend.copyRegion} precondition check (the §1.1 byte-count
 * rule): `dst` has a leading slot axis, `slotIndex` is in range, dtypes
 * match, and `src` holds exactly one slot's worth of elements. Throws
 * {@link InvalidStateError} on any violation; location checks are the
 * caller's (device vs cpu differ per backend).
 */
function validateCopyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
  if (dst.shape.length === 0) {
    throw new InvalidStateError('copyRegion: dst must have a leading slot axis');
  }
  const slots = dst.shape[0] ?? 0;
  if (!Number.isInteger(slotIndex) || slotIndex < 0 || slotIndex >= slots) {
    throw new InvalidStateError(`copyRegion: slotIndex ${slotIndex} out of bounds [0, ${slots})`);
  }
  if (src.dtype !== dst.dtype) {
    throw new InvalidStateError(
      `copyRegion: dtype mismatch (src '${src.dtype}', dst '${dst.dtype}')`,
    );
  }
  const slotElems = slotElementCount(dst.shape);
  let srcElems = 1;
  for (const dim of src.shape) srcElems *= dim;
  if (srcElems !== slotElems) {
    throw new InvalidStateError(
      `copyRegion: src has ${srcElems} elements but one dst slot holds ${slotElems}`,
    );
  }
}

/**
 * Structural subset of the WebGPU device API this backend touches, declared
 * locally so @websam/core need not depend on `@webgpu/types`. Per ⚠REV
 * ORT#26107 the device is READ from ort (`ort.env.webgpu.device`), never
 * created by websam.
 */
interface GpuBufferLike {
  destroy?(): void;
}
interface GpuCommandEncoderLike {
  copyBufferToBuffer(
    source: GpuBufferLike,
    sourceOffset: number,
    destination: GpuBufferLike,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}
interface GpuQueueLike {
  submit(commandBuffers: readonly unknown[]): void;
}
interface GpuDeviceLike {
  createBuffer(descriptor: { size: number; usage: number }): GpuBufferLike;
  createCommandEncoder(): GpuCommandEncoderLike;
  readonly queue: GpuQueueLike;
}

/**
 * WebGPU `GPUBufferUsage` bit flags (spec-stable numeric constants; declared
 * locally to avoid depending on `@webgpu/types`). A ring buffer must be
 * bindable as a storage input (`STORAGE`) and both a copy source and
 * destination for {@link WebGpuBackend.copyRegion}.
 */
const GPU_BUFFER_USAGE = 0x0080 /* STORAGE */ | 0x0004 /* COPY_SRC */ | 0x0008; /* COPY_DST */

/** The GPUBuffer backing a device-located {@link OrtDeviceTensor}. */
function gpuBufferOf(tensor: DeviceTensor): GpuBufferLike {
  if (!(tensor instanceof OrtDeviceTensor)) {
    throw new InvalidStateError('copyRegion: tensor was not created by this backend');
  }
  return (tensor.ortTensor as unknown as { gpuBuffer: GpuBufferLike }).gpuBuffer;
}

/**
 * Structural subset of the WebGPU API that probing needs. Declared locally
 * so @websam/core does not depend on `@webgpu/types` — probing only touches
 * `navigator.gpu.requestAdapter()` and `adapter.features`.
 */
interface GpuAdapterLike {
  readonly features: ReadonlySet<string>;
}
interface GpuLike {
  requestAdapter(): Promise<GpuAdapterLike | null>;
}

/** Result of {@link WebGpuBackend.probe}: pure capability facts, no ort involved. */
export interface WebGpuProbeResult {
  /** True iff `navigator.gpu` exists AND an adapter was actually granted. */
  webgpu: boolean;
  /** True iff the granted adapter advertises the `'shader-f16'` feature. */
  f16: boolean;
  /** Whether the page is cross-origin isolated (enables multi-threaded WASM fallback). */
  crossOriginIsolated: boolean;
  /** Best device for this environment: `'webgpu'` when available, else `'wasm'`. */
  recommendedDevice: 'webgpu' | 'wasm';
}

/**
 * WebGPU implementation of {@link Backend}, driving onnxruntime-web's webgpu
 * execution provider.
 *
 * M2 status: probing, session creation, tensor upload/alloc (cpu AND
 * device), {@link copyRegion}, readback, {@link debugStats} and dispose are
 * all real. Device-located {@link allocTensor} backs the video memory-bank
 * ring with a zeroed `GPUBuffer`; {@link copyRegion} is a command-encoder
 * buffer byte-range copy.
 *
 * ⚠REV ORT#26107: this backend NEVER injects a GPUDevice into ort — ort
 * creates and owns the WebGPU device at session creation; anything websam
 * needs about the device is READ from ort post-init.
 */
export class WebGpuBackend implements Backend {
  readonly kind = 'webgpu' as const;

  /**
   * Device capabilities discovered by {@link init}. `f16` gates whether
   * fp16-quantized graphs may be selected for this device.
   */
  features: { f16: boolean } = { f16: false };

  readonly #ort: OrtModule;
  #initialized = false;
  readonly #sessions = new Set<OrtBackendSession>();
  readonly #tensors = new Set<OrtDeviceTensor>();

  /**
   * @param ort - The onnxruntime-web module (e.g. `import * as ort from 'onnxruntime-web'`).
   * Injected so callers control which ort build (and wasm asset paths) load.
   */
  constructor(ort: OrtModule) {
    this.#ort = ort;
  }

  /** The injected onnxruntime-web module (exposed for session wiring). */
  protected get ort(): OrtModule {
    return this.#ort;
  }

  /**
   * Probe the environment and acquire adapter capabilities.
   *
   * Verifies `navigator.gpu` exists and an adapter is granted, detects
   * `'shader-f16'`, and exposes the result on {@link features}. Throws
   * {@link UnsupportedDeviceError} when WebGPU is unavailable — callers
   * wanting graceful degradation should call {@link probe} first.
   */
  async init(): Promise<void> {
    const gpu = (globalThis as { navigator?: { gpu?: GpuLike } }).navigator?.gpu;
    if (!gpu) {
      throw new UnsupportedDeviceError(
        'WebGPU is not available in this environment (navigator.gpu is undefined)',
      );
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new UnsupportedDeviceError('WebGPU adapter request was denied (no adapter available)');
    }
    this.features = { f16: adapter.features.has('shader-f16') };
    this.#initialized = true;
  }

  /** Whether {@link init} has completed successfully (and {@link dispose} has not run since). */
  get initialized(): boolean {
    return this.#initialized;
  }

  #assertInitialized(method: string): void {
    if (!this.#initialized) {
      throw new InvalidStateError(`WebGpuBackend.${method} called before init()`);
    }
  }

  /**
   * Compile `graph.bytes` on the webgpu execution provider, honoring `plan`
   * via ort's `preferredOutputLocation` (`'device'` → `'gpu-buffer'`).
   * Streaming (`url`) graphs land in M2.
   */
  async createSession(graph: GraphAsset, plan?: IOBindingPlan): Promise<BackendSession> {
    this.#assertInitialized('createSession');
    if (graph.bytes === undefined) {
      throw new NotImplementedError(
        `WebGpuBackend.createSession(url graph '${graph.name}'), lands in M2`,
      );
    }
    const inner = await createOrtSession(this.#ort, 'webgpu', graph.bytes, { ioPlan: plan });
    const session = new OrtBackendSession(inner, (s) => this.#sessions.delete(s));
    this.#sessions.add(session);
    return session;
  }

  /** Create a tensor initialized from host data (`'cpu'` location; int64 takes BigInt64Array). */
  uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor {
    this.#assertInitialized('uploadTensor');
    const tensor = createCpuTensor(this.#ort, data, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * Allocate a zeroed tensor. `'cpu'` returns a zeroed typed array;
   * `'device'` allocates a zeroed `GPUBuffer` through ort's WebGPU device
   * (which the WebGPU spec zero-initializes) and wraps it via
   * `ort.Tensor.fromGpuBuffer` — the video memory-bank ring primitive.
   * Throws {@link OutOfMemoryError} when the GPU cannot satisfy the
   * allocation.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    if (location === 'cpu') {
      const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
      this.#tensors.add(tensor);
      return tensor;
    }
    const device = this.#gpuDevice('allocTensor');
    // WebGPU requires buffer sizes to be a multiple of 4; the tensor's own
    // logical byte length may be smaller (e.g. an odd count of uint8), so pad.
    const byteLength = tensorByteLength(shape, dtype);
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    let buffer: GpuBufferLike;
    try {
      buffer = device.createBuffer({ size, usage: GPU_BUFFER_USAGE });
    } catch (err) {
      throw new OutOfMemoryError(
        `WebGpuBackend.allocTensor('device'): GPU buffer allocation of ${size} bytes failed`,
        { cause: err },
      );
    }
    const fromGpuBuffer = (
      this.#ort.Tensor as unknown as {
        fromGpuBuffer(b: GpuBufferLike, opts: { dataType: DType; dims: readonly number[] }): unknown;
      }
    ).fromGpuBuffer(buffer, { dataType: dtype, dims: shape });
    const tensor = OrtDeviceTensor.wrap(
      fromGpuBuffer as import('onnxruntime-web').Tensor,
      (t) => {
        this.#tensors.delete(t);
        // ort.Tensor.fromGpuBuffer does not own the external buffer, so the
        // backend destroys it when the wrapper is disposed.
        buffer.destroy?.();
      },
    );
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * Copy `src` into slot `slotIndex` of the ring `dst` via a command-encoder
   * buffer byte-range copy (`copyBufferToBuffer`), entirely on-device. Both
   * operands must be device-located (a `'cpu'` operand → upload first,
   * {@link InvalidStateError}); the §1.1 byte-count rule is validated before
   * the copy is issued.
   */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    this.#assertInitialized('copyRegion');
    validateCopyRegion(src, dst, slotIndex);
    if (src.location !== 'device' || dst.location !== 'device') {
      throw new InvalidStateError(
        'WebGpuBackend.copyRegion requires device-located operands (upload to device first)',
      );
    }
    const slotBytes = tensorByteLength(dst.shape, dst.dtype) / (dst.shape[0] ?? 1);
    const device = this.#gpuDevice('copyRegion');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      gpuBufferOf(src),
      0,
      gpuBufferOf(dst),
      slotIndex * slotBytes,
      slotBytes,
    );
    device.queue.submit([encoder.finish()]);
  }

  /** Read ort's WebGPU device (created lazily by ort during session creation). */
  #gpuDevice(method: string): GpuDeviceLike {
    const device = (this.#ort.env?.webgpu as unknown as { device?: GpuDeviceLike } | undefined)
      ?.device;
    if (!device) {
      throw new InvalidStateError(
        `WebGpuBackend.${method}: ort has not created a WebGPU device yet (create a session first)`,
      );
    }
    return device;
  }

  /**
   * Live-resource census (§1.1): every tensor this backend uploaded or
   * allocated and has not yet disposed, plus their aggregate byte size. The
   * video loop's steady-state flatness gate reads this each frame boundary.
   */
  debugStats(): { liveTensors: number; liveBytes: number } {
    let liveBytes = 0;
    for (const tensor of this.#tensors) {
      liveBytes += tensorByteLength(tensor.shape, tensor.dtype);
    }
    return { liveTensors: this.#tensors.size, liveBytes };
  }

  /**
   * Explicit device→CPU crossing: cpu tensors return a view over their
   * data; device tensors read back via ort (`float16` → `Uint16Array` raw
   * half bits).
   */
  async readback(tensor: DeviceTensor): Promise<ArrayBufferView> {
    this.#assertInitialized('readback');
    return readbackTensor(tensor);
  }

  /**
   * Dispose every tensor and session this backend still tracks and reset to
   * the uninitialized state. Further calls (including a second dispose)
   * throw {@link InvalidStateError} until {@link init} runs again.
   */
  async dispose(): Promise<void> {
    this.#assertInitialized('dispose');
    this.#initialized = false;
    for (const tensor of [...this.#tensors]) {
      tensor.dispose();
    }
    this.#tensors.clear();
    for (const session of [...this.#sessions]) {
      await session.dispose();
    }
    this.#sessions.clear();
  }

  /**
   * Pure capability detection — needs no ort module and no backend
   * instance. Degrades, never throws: any probing failure (blocked adapter,
   * throwing `requestAdapter`, exotic embedders) reports `webgpu: false`
   * and recommends `'wasm'`.
   */
  static async probe(): Promise<WebGpuProbeResult> {
    const crossOriginIsolated =
      (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    let webgpu = false;
    let f16 = false;
    try {
      const gpu = (globalThis as { navigator?: { gpu?: GpuLike } }).navigator?.gpu;
      if (gpu) {
        const adapter = await gpu.requestAdapter();
        if (adapter) {
          webgpu = true;
          f16 = adapter.features.has('shader-f16');
        }
      }
    } catch {
      // Degrade, never throw: report WebGPU as unavailable.
      webgpu = false;
      f16 = false;
    }
    return {
      webgpu,
      f16,
      crossOriginIsolated,
      recommendedDevice: webgpu ? 'webgpu' : 'wasm',
    };
  }
}
