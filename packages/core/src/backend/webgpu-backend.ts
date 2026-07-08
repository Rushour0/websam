import {
  InvalidStateError,
  NotImplementedError,
  OutOfMemoryError,
  UnsupportedDeviceError,
} from '../errors.js';
import { createOrtSession, OrtBackendSession } from '../runtime/ort-session.js';
import { allocCpuTensor, createCpuTensor, OrtDeviceTensor, readbackTensor } from '../runtime/ort-tensor.js';
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

type OrtInferenceSession = import('onnxruntime-web').InferenceSession;

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

/* -------------------------------------------------------------------------
 * Shared M2 memory-primitive helpers.
 *
 * Used by BOTH browser backends' `copyRegion`/`allocTensor`/`debugStats`
 * bodies. They live in this module (not a new file) because `src/backend/*`
 * is one ownership unit and `wasm-backend.ts` already imports from here —
 * same dependency direction as {@link OrtModule}.
 * ------------------------------------------------------------------------- */

/**
 * Bytes per element for each {@link DType} (`float16` = raw half bits in a
 * `Uint16Array`, `bool` = one 0/1 byte — mirrors ort's own data map).
 */
const BYTES_PER_ELEMENT = {
  float32: 4,
  float16: 2,
  int64: 8,
  uint8: 1,
  int32: 4,
  bool: 1,
} as const satisfies Record<DType, number>;

/** Typed-array view constructor per {@link DType} (device readback views). */
const DTYPE_VIEWS = {
  float32: Float32Array,
  float16: Uint16Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  int32: Int32Array,
  bool: Uint8Array,
} as const satisfies Record<DType, new (buffer: ArrayBuffer) => ArrayBufferView>;

/** Total element count of a shape (empty shape = scalar = 1 element). */
function elementCountOf(shape: readonly number[]): number {
  let count = 1;
  for (const dim of shape) {
    count *= dim;
  }
  return count;
}

/** Logical byte length of a tensor: shape product × element width. */
export function tensorByteLength(shape: readonly number[], dtype: DType): number {
  return elementCountOf(shape) * BYTES_PER_ELEMENT[dtype];
}

/**
 * The shared `Backend.debugStats` body: counts every tracked, non-disposed
 * tensor and its logical bytes. `run()` outputs carry no dispose hook (they
 * are wrapped inside `BackendSession.run`), so entries the caller already
 * disposed are pruned here by their `disposed` flag instead of eagerly.
 */
export function censusStats(census: Set<OrtDeviceTensor>): {
  liveTensors: number;
  liveBytes: number;
} {
  let liveTensors = 0;
  let liveBytes = 0;
  for (const tensor of census) {
    if (tensor.disposed) {
      census.delete(tensor);
      continue;
    }
    liveTensors += 1;
    liveBytes += tensorByteLength(tensor.shape, tensor.dtype);
  }
  return { liveTensors, liveBytes };
}

/** Validated geometry of one `copyRegion` call, in elements and bytes. */
export interface CopyRegionGeometry {
  src: OrtDeviceTensor;
  dst: OrtDeviceTensor;
  /** Elements in one `dst` slot (== `src`'s total element count). */
  slotElements: number;
  /** Bytes in one `dst` slot. */
  slotBytes: number;
  /** Start of slot `slotIndex` inside `dst`, in elements. */
  elementOffset: number;
  /** Start of slot `slotIndex` inside `dst`, in bytes. */
  byteOffset: number;
}

/**
 * Shared `copyRegion` argument validation for both browser backends:
 * operands must be live tensors created by this backend, dtypes must match,
 * `slotIndex` must address a real slot, and `src` must have exactly
 * `dst.shape.slice(1)`'s element count (the Backend contract's
 * byte-count-equal rule — the copy is contiguous and reshape-free).
 *
 * @throws InvalidStateError on any violation.
 */
export function checkCopyRegionArgs(
  method: string,
  src: DeviceTensor,
  dst: DeviceTensor,
  slotIndex: number,
): CopyRegionGeometry {
  const operands: readonly ['src' | 'dst', DeviceTensor][] = [
    ['src', src],
    ['dst', dst],
  ];
  for (const [name, tensor] of operands) {
    if (!(tensor instanceof OrtDeviceTensor)) {
      throw new InvalidStateError(
        `${method}: ${name} was not created by this backend (expected an OrtDeviceTensor)`,
      );
    }
    if (tensor.disposed) {
      throw new InvalidStateError(`${method}: ${name} is disposed`);
    }
  }
  const srcTensor = src as OrtDeviceTensor;
  const dstTensor = dst as OrtDeviceTensor;
  if (srcTensor.dtype !== dstTensor.dtype) {
    throw new InvalidStateError(
      `${method}: dtype mismatch (src '${srcTensor.dtype}' vs dst '${dstTensor.dtype}')`,
    );
  }
  const slotCount = dstTensor.shape[0];
  if (slotCount === undefined || slotCount <= 0) {
    throw new InvalidStateError(
      `${method}: dst has no slot axis (shape [${dstTensor.shape.join(', ')}])`,
    );
  }
  if (!Number.isInteger(slotIndex) || slotIndex < 0 || slotIndex >= slotCount) {
    throw new InvalidStateError(`${method}: slotIndex ${slotIndex} out of range [0, ${slotCount})`);
  }
  const slotElements = elementCountOf(dstTensor.shape) / slotCount;
  const srcElements = elementCountOf(srcTensor.shape);
  if (srcElements !== slotElements) {
    throw new InvalidStateError(
      `${method}: src element count ${srcElements} != slot element count ${slotElements} ` +
        `(src must have exactly dst.shape.slice(1)'s element count and the same dtype)`,
    );
  }
  const slotBytes = slotElements * BYTES_PER_ELEMENT[dstTensor.dtype];
  return {
    src: srcTensor,
    dst: dstTensor,
    slotElements,
    slotBytes,
    elementOffset: slotIndex * slotElements,
    byteOffset: slotIndex * slotBytes,
  };
}

/**
 * {@link OrtBackendSession} that registers every `run()` output in the
 * owning backend's tensor census (the `Backend.debugStats` leak gate).
 * Outputs are wrapped without a dispose hook inside `run`, so the census
 * prunes them lazily by their `disposed` flag (see {@link censusStats}).
 */
export class CensusOrtBackendSession extends OrtBackendSession {
  readonly #census: Set<OrtDeviceTensor>;

  constructor(
    session: OrtInferenceSession,
    census: Set<OrtDeviceTensor>,
    onDispose?: (session: OrtBackendSession) => void,
  ) {
    super(session, onDispose);
    this.#census = census;
  }

  override async run(
    feeds: Record<string, DeviceTensor>,
    fetches?: readonly string[],
  ): Promise<Record<string, DeviceTensor>> {
    const outputs = await super.run(feeds, fetches);
    for (const tensor of Object.values(outputs)) {
      if (tensor instanceof OrtDeviceTensor) {
        this.#census.add(tensor);
      }
    }
    return outputs;
  }
}

/* -------------------------------------------------------------------------
 * Structural WebGPU surface for the M2 device primitives. Declared locally
 * for the same reason as {@link GpuLike}: @websam/core does not depend on
 * `@webgpu/types`, and the primitives only touch buffers + one copy encoder.
 * ------------------------------------------------------------------------- */

/** Structural subset of `GPUBuffer` the device primitives touch. */
export interface GpuBufferLike {
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}
interface GpuCommandEncoderLike {
  copyBufferToBuffer(
    src: GpuBufferLike,
    srcOffset: number,
    dst: GpuBufferLike,
    dstOffset: number,
    size: number,
  ): void;
  finish(): object;
}
/** Structural subset of `GPUDevice` the device primitives drive. */
export interface GpuDeviceLike {
  createBuffer(descriptor: { size: number; usage: number; label?: string }): GpuBufferLike;
  createCommandEncoder(): GpuCommandEncoderLike;
  readonly queue: { submit(commandBuffers: readonly object[]): void };
}

/** WebGPU spec-pinned bit flags (avoids `GPUBufferUsage`/`GPUMapMode` globals in node tests). */
const USAGE_MAP_READ = 0x0001;
const USAGE_COPY_SRC = 0x0004;
const USAGE_COPY_DST = 0x0008;
const USAGE_STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;

/**
 * WebGPU implementation of {@link Backend}, driving onnxruntime-web's webgpu
 * execution provider.
 *
 * M2 status: everything on the Backend interface is real except streaming
 * (`url`) graph compilation. {@link copyRegion} and device-located
 * {@link allocTensor} became real with the video memory bank, and
 * {@link debugStats} provides the leak-gate tensor census.
 *
 * ⚠REV ORT#26107: this backend NEVER injects a GPUDevice into ort — ort
 * creates and owns the WebGPU device at session creation; anything websam
 * needs about the device is READ from ort post-init (the device is captured
 * from `ort.env.webgpu` after the first session compiles).
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
  /** ort's own GPUDevice, read out after the first session compiles (⚠REV ORT#26107). */
  #device: GpuDeviceLike | undefined;

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
    // ⚠REV ORT#26107: ort creates and owns the GPUDevice at session creation;
    // websam READS it out post-init, never injects one. The first successful
    // session is the earliest moment it exists — captured here for the
    // device-memory primitives (allocTensor('device') / copyRegion).
    if (this.#device === undefined) {
      const webgpuEnv = (
        this.#ort.env as unknown as
          | { webgpu?: { device?: GpuDeviceLike | Promise<GpuDeviceLike> } }
          | undefined
      )?.webgpu;
      const device = webgpuEnv ? await webgpuEnv.device : undefined;
      if (device !== undefined) {
        this.#device = device;
      }
    }
    const session = new CensusOrtBackendSession(inner, this.#tensors, (s) =>
      this.#sessions.delete(s),
    );
    this.#sessions.add(session);
    return session;
  }

  /** ort's GPUDevice, or `InvalidStateError` when no session has created it yet. */
  #requireDevice(method: string): GpuDeviceLike {
    if (this.#device === undefined) {
      throw new InvalidStateError(
        `WebGpuBackend.${method}: ort's GPUDevice is not available yet — ` +
          'it exists only after the first session is created (⚠REV ORT#26107: ort owns the device)',
      );
    }
    return this.#device;
  }

  /** Create a tensor initialized from host data (`'cpu'` location; int64 takes BigInt64Array). */
  uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor {
    this.#assertInitialized('uploadTensor');
    const tensor = createCpuTensor(this.#ort, data, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * Allocate a zeroed tensor. `'cpu'` → typed-array-backed ort tensor.
   * `'device'` (M2, the video ring's primitive) → a zero-initialized
   * `GPUBuffer` allocated on ort's own device (read out post-init per
   * ⚠REV ORT#26107) and wrapped via `ort.Tensor.fromGpuBuffer`.
   *
   * Note ort's download semantics: reading a `'device'`-alloc'd tensor back
   * (`Backend.readback`) downloads once and pins the HANDLE to `'cpu'`
   * afterwards — fine for debugging, but ring tensors must not be read back
   * mid-session (the memory bank never does).
   *
   * @throws InvalidStateError for `'device'` before any session exists (ort
   * owns the GPUDevice; there is none to allocate on yet).
   * @throws OutOfMemoryError when the device cannot satisfy the allocation.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    if (location === 'device') {
      return this.#allocDeviceTensor(shape, dtype);
    }
    const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  #allocDeviceTensor(shape: readonly number[], dtype: DType): DeviceTensor {
    const device = this.#requireDevice('allocTensor');
    for (const dim of shape) {
      if (!Number.isInteger(dim) || dim < 0) {
        throw new RangeError(
          `WebGpuBackend.allocTensor: invalid dimension ${dim} in shape [${shape.join(', ')}]`,
        );
      }
    }
    const byteLength = tensorByteLength(shape, dtype);
    // GPUBuffers are zero-initialized per the WebGPU spec. Pad to 16 bytes so
    // every dtype satisfies WebGPU's 4-byte copy alignment and ort's binding.
    const size = Math.max(16, Math.ceil(byteLength / 16) * 16);
    let buffer: GpuBufferLike;
    try {
      buffer = device.createBuffer({
        size,
        usage: USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST,
        label: `websam-alloc-${dtype}[${shape.join('x')}]`,
      });
    } catch (err) {
      throw new OutOfMemoryError(
        `WebGpuBackend.allocTensor: device allocation of ${size} bytes failed ` +
          `for ${dtype} shape [${shape.join(', ')}]`,
        { cause: err },
      );
    }
    const TensorCtor = this.#ort.Tensor as unknown as {
      fromGpuBuffer(
        buffer: GpuBufferLike,
        options: {
          dataType: DType;
          dims: readonly number[];
          download?: () => Promise<ArrayBufferView>;
          dispose?: () => void;
        },
      ): import('onnxruntime-web').Tensor;
    };
    const ortTensor = TensorCtor.fromGpuBuffer(buffer, {
      dataType: dtype,
      dims: [...shape],
      download: () => this.#downloadBuffer(buffer, byteLength, dtype),
      dispose: () => buffer.destroy(),
    });
    const tensor = OrtDeviceTensor.wrap(ortTensor, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * `download` callback for device-alloc'd tensors (the `Backend.readback`
   * path): copy into a staging buffer, map it, and view per dtype.
   */
  async #downloadBuffer(
    buffer: GpuBufferLike,
    byteLength: number,
    dtype: DType,
  ): Promise<ArrayBufferView> {
    const device = this.#requireDevice('readback');
    const alignedBytes = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const staging = device.createBuffer({
      size: alignedBytes,
      usage: USAGE_MAP_READ | USAGE_COPY_DST,
      label: 'websam-readback-staging',
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, alignedBytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(MAP_MODE_READ);
      const bytes = staging.getMappedRange().slice(0, byteLength);
      staging.unmap();
      return new DTYPE_VIEWS[dtype](bytes);
    } finally {
      staging.destroy();
    }
  }

  /**
   * The memory-bank ring primitive, real in M2: one
   * `commandEncoder.copyBufferToBuffer` of a slot's bytes at
   * `slotIndex * slotBytes`, entirely device-side. dtype, slot bounds, and
   * the byte-count-equal rule are validated per the Backend contract.
   * Device↔device only — a `'cpu'`-located operand on this backend is
   * `InvalidStateError` (upload/alloc on the device first).
   */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    this.#assertInitialized('copyRegion');
    const geometry = checkCopyRegionArgs('WebGpuBackend.copyRegion', src, dst, slotIndex);
    if (geometry.src.location !== 'device' || geometry.dst.location !== 'device') {
      throw new InvalidStateError(
        "WebGpuBackend.copyRegion: both operands must be 'device'-located on webgpu " +
          '(upload/alloc on the device first)',
      );
    }
    if (geometry.slotBytes % 4 !== 0) {
      throw new InvalidStateError(
        `WebGpuBackend.copyRegion: slot byte size ${geometry.slotBytes} is not ` +
          '4-byte aligned (WebGPU copy constraint)',
      );
    }
    const device = this.#requireDevice('copyRegion');
    const srcBuffer = geometry.src.ortTensor.gpuBuffer as unknown as GpuBufferLike;
    const dstBuffer = geometry.dst.ortTensor.gpuBuffer as unknown as GpuBufferLike;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, geometry.byteOffset, geometry.slotBytes);
    device.queue.submit([encoder.finish()]);
  }

  /**
   * Live-resource census for leak gates (M2): every non-disposed tensor
   * this backend created — alloc, upload, and `run()` outputs. Callable in
   * any state; after {@link dispose} it reports zeros.
   */
  debugStats(): { liveTensors: number; liveBytes: number } {
    return censusStats(this.#tensors);
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
      // run() outputs stay in the census after the caller disposes them
      // (pruned lazily, see censusStats) — never double-dispose those.
      if (!tensor.disposed) {
        tensor.dispose();
      }
    }
    this.#tensors.clear();
    for (const session of [...this.#sessions]) {
      await session.dispose();
    }
    this.#sessions.clear();
    this.#device = undefined;
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
