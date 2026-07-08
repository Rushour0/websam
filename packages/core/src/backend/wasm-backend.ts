import { NotImplementedError, UnsupportedDeviceError } from '../errors.js';
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
 * M0 status: {@link WasmBackend.probe} and {@link WasmBackend.init} are
 * real; session/tensor methods land in M1 and currently throw
 * {@link NotImplementedError}.
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

  /**
   * @param ort - The onnxruntime-web module. Injected so callers control
   * which ort build (and wasm asset paths / thread count) load.
   */
  constructor(ort: OrtModule) {
    this.#ort = ort;
  }

  /** The injected onnxruntime-web module (exposed for M1 session wiring). */
  protected get ort(): OrtModule {
    return this.#ort;
  }

  /**
   * Probe the environment. Real at M0: verifies WebAssembly exists and
   * records whether threads are available; a non-isolated page is NOT an
   * error (single-thread fallback), so this only throws
   * {@link UnsupportedDeviceError} when WebAssembly itself is missing.
   */
  async init(): Promise<void> {
    const probed = await WasmBackend.probe();
    if (!probed.wasm) {
      throw new UnsupportedDeviceError(
        'WebAssembly is not available in this environment',
      );
    }
    this.features = { threads: probed.threads };
    this.#initialized = true;
  }

  /** Whether {@link init} has completed successfully. */
  get initialized(): boolean {
    return this.#initialized;
  }

  /** @throws NotImplementedError — lands in M1. */
  createSession(_graph: GraphAsset, _plan?: IOBindingPlan): Promise<BackendSession> {
    throw new NotImplementedError('WasmBackend.createSession, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  allocTensor(_shape: readonly number[], _dtype: DType, _location: TensorLocation): DeviceTensor {
    throw new NotImplementedError('WasmBackend.allocTensor, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  copyRegion(_src: DeviceTensor, _dst: DeviceTensor, _slotIndex: number): void {
    throw new NotImplementedError('WasmBackend.copyRegion, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  readback(_tensor: DeviceTensor): Promise<ArrayBufferView> {
    throw new NotImplementedError('WasmBackend.readback, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  dispose(): Promise<void> {
    throw new NotImplementedError('WasmBackend.dispose, lands in M1');
  }

  /**
   * Pure capability detection — REAL at M0, needs no ort module. Degrades,
   * never throws. `threads` requires cross-origin isolation AND
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
    const threads =
      wasm && crossOriginIsolated && typeof SharedArrayBuffer === 'function';
    return { wasm, threads, crossOriginIsolated, recommendedDevice: 'wasm' };
  }
}
