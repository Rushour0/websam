import { InvalidStateError, NotImplementedError, UnsupportedDeviceError } from '../errors.js';
import { createOrtSession, OrtBackendSession } from '../runtime/ort-session.js';
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
import type { OrtModule } from './webgpu-backend.js';

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
 * M1 status: probing, session creation, cpu tensor upload/alloc, readback
 * and dispose are real. {@link copyRegion} and device-located
 * {@link allocTensor} land in M2 (video memory bank) and still throw
 * {@link NotImplementedError} — on this backend everything is cpu-located
 * anyway.
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
   * Allocate a zeroed `'cpu'` tensor. `'device'` allocation is the video
   * ring's primitive and lands in M2.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    if (location === 'device') {
      throw new NotImplementedError("WasmBackend.allocTensor('device'), lands in M2");
    }
    const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /** @throws NotImplementedError — memory-bank primitive, lands in M2. */
  copyRegion(_src: DeviceTensor, _dst: DeviceTensor, _slotIndex: number): void {
    throw new NotImplementedError('WasmBackend.copyRegion, lands in M2');
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
