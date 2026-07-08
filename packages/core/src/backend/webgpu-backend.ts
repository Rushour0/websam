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
 * M1 status: probing, session creation, cpu tensor upload/alloc, readback
 * and dispose are real. {@link copyRegion} and device-located
 * {@link allocTensor} land in M2 (video memory bank) and still throw
 * {@link NotImplementedError}.
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
   * Allocate a zeroed `'cpu'` tensor. `'device'` allocation is the video
   * ring's primitive and lands in M2.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor {
    this.#assertInitialized('allocTensor');
    if (location === 'device') {
      throw new NotImplementedError("WebGpuBackend.allocTensor('device'), lands in M2");
    }
    const tensor = allocCpuTensor(this.#ort, shape, dtype, (t) => this.#tensors.delete(t));
    this.#tensors.add(tensor);
    return tensor;
  }

  /** @throws NotImplementedError — memory-bank primitive, lands in M2. */
  copyRegion(_src: DeviceTensor, _dst: DeviceTensor, _slotIndex: number): void {
    throw new NotImplementedError('WebGpuBackend.copyRegion, lands in M2');
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
