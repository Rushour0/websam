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

/**
 * The onnxruntime-web module namespace, injected rather than imported so the
 * backend never forces ort into a bundle (and so tests can stub it).
 */
export type OrtModule = typeof import('onnxruntime-web');

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
 * M0 status: {@link WebGpuBackend.probe} and {@link WebGpuBackend.init}
 * (capability probing) are real; session/tensor methods land in M1 and
 * currently throw {@link NotImplementedError}.
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

  /**
   * @param ort - The onnxruntime-web module (e.g. `import * as ort from 'onnxruntime-web'`).
   * Injected so callers control which ort build (and wasm asset paths) load.
   */
  constructor(ort: OrtModule) {
    this.#ort = ort;
  }

  /** The injected onnxruntime-web module (exposed for M1 session wiring). */
  protected get ort(): OrtModule {
    return this.#ort;
  }

  /**
   * Probe the environment and acquire adapter capabilities.
   *
   * Real at M0: verifies `navigator.gpu` exists and an adapter is granted,
   * detects `'shader-f16'`, and exposes the result on {@link features}.
   * Throws {@link UnsupportedDeviceError} when WebGPU is unavailable —
   * callers wanting graceful degradation should call {@link probe} first.
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

  /** Whether {@link init} has completed successfully. */
  get initialized(): boolean {
    return this.#initialized;
  }

  /** @throws NotImplementedError — lands in M1. */
  createSession(_graph: GraphAsset, _plan?: IOBindingPlan): Promise<BackendSession> {
    throw new NotImplementedError('WebGpuBackend.createSession, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  allocTensor(_shape: readonly number[], _dtype: DType, _location: TensorLocation): DeviceTensor {
    throw new NotImplementedError('WebGpuBackend.allocTensor, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  copyRegion(_src: DeviceTensor, _dst: DeviceTensor, _slotIndex: number): void {
    throw new NotImplementedError('WebGpuBackend.copyRegion, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  readback(_tensor: DeviceTensor): Promise<ArrayBufferView> {
    throw new NotImplementedError('WebGpuBackend.readback, lands in M1');
  }

  /** @throws NotImplementedError — lands in M1. */
  dispose(): Promise<void> {
    throw new NotImplementedError('WebGpuBackend.dispose, lands in M1');
  }

  /**
   * Pure capability detection — REAL at M0, needs no ort module and no
   * backend instance. Degrades, never throws: any probing failure (blocked
   * adapter, throwing `requestAdapter`, exotic embedders) reports
   * `webgpu: false` and recommends `'wasm'`.
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
