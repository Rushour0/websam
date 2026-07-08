import { InvalidStateError, NotImplementedError, UnsupportedDeviceError } from '../errors.js';
import { createOrtSession, type OrtBackendSession } from '../runtime/ort-session.js';
import {
  allocCpuTensor,
  createCpuTensor,
  readbackTensor,
  type OrtDeviceTensor,
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
import {
  censusStats,
  CensusOrtBackendSession,
  checkCopyRegionArgs,
  type OrtModule,
} from './webgpu-backend.js';

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
 * M2 status: everything on the Backend interface is real except streaming
 * (`url`) graph compilation. On this backend everything is cpu-located, so
 * {@link copyRegion} is a typed-array region copy and `'device'` allocation
 * degrades to `'cpu'` by design; {@link debugStats} provides the leak-gate
 * tensor census.
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
    const session = new CensusOrtBackendSession(inner, this.#tensors, (s) =>
      this.#sessions.delete(s),
    );
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
   * Allocate a zeroed tensor. On wasm everything is host memory, so
   * `'device'` degrades to `'cpu'` by design (documented: on this backend
   * `'device'` === `'cpu'`) — the video memory bank's rings are plain
   * typed-array tensors here and the returned tensor reports
   * `location: 'cpu'`.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    void location; // 'device' === 'cpu' on wasm — every allocation is host memory.
    const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /**
   * The memory-bank ring primitive, real in M2: a single
   * `TypedArray.prototype.set` of `src`'s elements into slot `slotIndex`
   * of `dst`. dtype, slot bounds, and the byte-count-equal rule are
   * validated per the Backend contract (`InvalidStateError` otherwise).
   */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void {
    this.#assertInitialized('copyRegion');
    const geometry = checkCopyRegionArgs('WasmBackend.copyRegion', src, dst, slotIndex);
    if (geometry.src.location !== 'cpu' || geometry.dst.location !== 'cpu') {
      throw new InvalidStateError(
        'WasmBackend.copyRegion: operands must be cpu-located (everything is cpu on wasm)',
      );
    }
    // Same dtype ⇒ same typed-array kind; the casts only narrow for TS —
    // set() dispatches on the operands' real typed-array types at runtime.
    const srcData = geometry.src.ortTensor.data as Uint8Array;
    const dstData = geometry.dst.ortTensor.data as Uint8Array;
    dstData.set(srcData, geometry.elementOffset);
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
