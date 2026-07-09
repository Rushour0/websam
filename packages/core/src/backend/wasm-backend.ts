import { InvalidStateError, NotImplementedError, UnsupportedDeviceError } from '../errors.js';
import { createOrtSession, OrtBackendSession } from '../runtime/ort-session.js';
import {
  allocCpuTensor,
  createCpuTensor,
  reshapeOrtView,
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
import type { OrtModule } from './webgpu-backend.js';

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
 * {@link InvalidStateError} on any violation.
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

/** The backing typed array of a cpu-located {@link OrtDeviceTensor}. */
function typedDataOf(tensor: DeviceTensor): { set(source: unknown, offset: number): void } {
  if (!(tensor instanceof OrtDeviceTensor)) {
    throw new InvalidStateError('copyRegion: tensor was not created by this backend');
  }
  return tensor.ortTensor.data as unknown as { set(source: unknown, offset: number): void };
}

/** Result of {@link WasmBackend.probe}: pure capability facts, no ort involved. */
export interface WasmProbeResult {
  /** True iff a WebAssembly runtime is present (it is in every supported environment). */
  wasm: boolean;
  /**
   * True iff multi-threaded WASM can run: requires `crossOriginIsolated`
   * (COOP/COEP headers) so `SharedArrayBuffer` is available. When false the
   * backend still works — it silently falls back to a single thread, it just
   * runs slower. It never throws over missing isolation.
   */
  threads: boolean;
  /** Whether the page is cross-origin isolated. */
  crossOriginIsolated: boolean;
  /** Always `'wasm'`: this probe only judges the CPU path (see WebGpuBackend.probe for GPU). */
  recommendedDevice: 'wasm';
}

/**
 * CPU (WebAssembly) implementation of {@link Backend}, driving
 * onnxruntime-web's wasm execution provider. This is the universal fallback:
 * it must work on every supported browser, isolated or not.
 *
 * M2 status: probing, session creation, tensor upload/alloc,
 * {@link copyRegion}, readback, {@link debugStats} and dispose are all real.
 * On this backend everything is cpu-located, so device-located
 * {@link allocTensor} degrades to a cpu tensor and {@link copyRegion} is a
 * plain typed-array region copy.
 */
export class WasmBackend implements Backend {
  readonly kind = 'wasm' as const;

  /**
   * Capabilities discovered by {@link init}. `threads: false` means the
   * single-thread fallback is active (page not cross-origin isolated).
   */
  features: { threads: boolean } = { threads: false };

  readonly #ort: OrtModule;
  #initialized = false;
  readonly #sessions = new Set<OrtBackendSession>();
  readonly #tensors = new Set<OrtDeviceTensor>();

  /**
   * @param ort - The onnxruntime-web module. Injected so callers control
   * which ort build (and wasm asset paths / thread count) load.
   */
  constructor(ort: OrtModule) {
    this.#ort = ort;
  }

  /** The injected onnxruntime-web module (exposed for session wiring). */
  protected get ort(): OrtModule {
    return this.#ort;
  }

  /**
   * Probe the environment. Verifies WebAssembly exists and records whether
   * threads are available; a non-isolated page is NOT an error
   * (single-thread fallback), so this only throws
   * {@link UnsupportedDeviceError} when WebAssembly itself is missing.
   */
  async init(): Promise<void> {
    const probed = await WasmBackend.probe();
    if (!probed.wasm) {
      throw new UnsupportedDeviceError('WebAssembly is not available in this environment');
    }
    this.features = { threads: probed.threads };
    this.#initialized = true;
  }

  /** Whether {@link init} has completed successfully (and {@link dispose} has not run since). */
  get initialized(): boolean {
    return this.#initialized;
  }

  #assertInitialized(method: string): void {
    if (!this.#initialized) {
      throw new InvalidStateError(`WasmBackend.${method} called before init()`);
    }
  }

  /**
   * Compile `graph.bytes` on the wasm execution provider. Everything is
   * cpu-located on this backend, so any {@link IOBindingPlan} is ignored.
   * Streaming (`url`) graphs land in M2.
   */
  async createSession(graph: GraphAsset, plan?: IOBindingPlan): Promise<BackendSession> {
    this.#assertInitialized('createSession');
    if (graph.bytes === undefined) {
      throw new NotImplementedError(
        `WasmBackend.createSession(url graph '${graph.name}'), lands in M2`,
      );
    }
    const inner = await createOrtSession(this.#ort, 'wasm', graph.bytes, { ioPlan: plan });
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
   * Allocate a zeroed tensor. On wasm there is no accelerator, so `'device'`
   * degrades to a `'cpu'`-located tensor (documented: on this backend
   * `'device'` === `'cpu'`) — identical to the `'cpu'` path. This lets the
   * video memory-bank ring allocate uniformly through the backend without
   * caring which environment it runs in.
   */
  allocTensor(shape: readonly number[], dtype: DType, _location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * Copy `src` into slot `slotIndex` of the ring `dst`. Everything is cpu on
   * this backend, so this is a `TypedArray.prototype.set` of one slot's
   * worth of elements at the slot's element offset. The §1.1 byte-count rule
   * is validated first ({@link InvalidStateError} on violation).
   */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    this.#assertInitialized('copyRegion');
    validateCopyRegion(src, dst, slotIndex);
    const slotElems = slotElementCount(dst.shape);
    typedDataOf(dst).set(typedDataOf(src) as unknown, slotIndex * slotElems);
  }

  reshape(tensor: DeviceTensor, shape: readonly number[]): DeviceTensor {
    this.#assertInitialized('reshape');
    return reshapeOrtView(this.#ort, tensor, shape);
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
   * Explicit device→CPU crossing. On this backend every tensor is
   * cpu-located, so this is a view over the existing data — no copy.
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
   * Pure capability detection — needs no ort module. Degrades, never
   * throws. `threads` requires cross-origin isolation AND
   * `SharedArrayBuffer`; when unavailable, callers should expect the
   * single-thread fallback rather than an error.
   */
  static async probe(): Promise<WasmProbeResult> {
    const crossOriginIsolated =
      (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    let wasm = false;
    try {
      wasm = typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function';
    } catch {
      wasm = false;
    }
    const threads = wasm && crossOriginIsolated && typeof SharedArrayBuffer === 'function';
    return { wasm, threads, crossOriginIsolated, recommendedDevice: 'wasm' };
  }
}
