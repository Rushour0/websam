/**
 * Real `createSegmenter` body (docs/m1-internal-contracts.md §4.2):
 * validate config → registry lookup → license gate → main-thread capability
 * probes → device/quant resolution → spawn the module worker → init the
 * engine (weights load + compile happen worker-side) → return the
 * {@link Segmenter} facade.
 *
 * The worker spawn is behind an injectable factory seam
 * ({@link CreateSegmenterInternals.spawnWorker}) and the real
 * `./spawn-worker.js` is imported DYNAMICALLY, so unit tests exercise every
 * path without a DOM `Worker` (and without `src/worker/**` at runtime).
 */

import * as Comlink from 'comlink';
import { WasmBackend } from '../backend/wasm-backend.js';
import { WebGpuBackend } from '../backend/webgpu-backend.js';
import { InvalidStateError, NotImplementedError, UnsupportedDeviceError } from '../errors.js';
import type { SegmenterConfig } from '../index.js';
import { getModel, type ModelSpec } from '../registry.js';
import { resolveDevice } from '../runtime/resolve-device.js';
import type { ImageSession, ResolvedModelInfo, Segmenter, VideoSession } from '../segmenter.js';
import type { WorkerInitRequest, WorkerInitResult } from '../worker/protocol.js';
import { ImageSessionImpl } from './image-session.js';
import type { WorkerHandle } from './spawn-worker.js';

/** M1 default tier: the only registered tier with image graphs (flips to 'edgetam' at M2). */
const DEFAULT_MODEL_ID = 'sam3-tracker';

const DEVICE_VALUES = ['webgpu', 'wasm', 'auto'] as const;
const QUANT_VALUES = ['auto', 'fp16', 'int8', 'q4f16'] as const;

/**
 * Internal test seams for {@link createSegmenterImpl}. NOT public API — the
 * public entry (`createSegmenter`) never passes this.
 */
export interface CreateSegmenterInternals {
  /** Replaces the real module-worker spawn (unit tests run without `Worker`). */
  spawnWorker?: (workerUrl?: string | URL) => WorkerHandle;
}

/** Reject config values that only unchecked-JS callers can produce. */
function validateConfig(config: SegmenterConfig): void {
  const device = config.device ?? 'auto';
  if (!(DEVICE_VALUES as readonly string[]).includes(device)) {
    throw new InvalidStateError(
      `createSegmenter: invalid config.device '${String(device)}' — expected one of ${DEVICE_VALUES.join(', ')}`,
    );
  }
  const quant = config.quant ?? 'auto';
  if (!(QUANT_VALUES as readonly string[]).includes(quant)) {
    throw new InvalidStateError(
      `createSegmenter: invalid config.quant '${String(quant)}' — expected one of ${QUANT_VALUES.join(', ')}`,
    );
  }
  if (config.acceptLicense !== undefined && config.acceptLicense !== 'sam') {
    throw new InvalidStateError(
      `createSegmenter: invalid config.acceptLicense '${String(config.acceptLicense)}' — the only accepted value is 'sam'`,
    );
  }
}

/**
 * The real {@link Segmenter} factory behind `createSegmenter`. See the module
 * doc for the sequence; error behavior:
 *
 * - unknown model id, license not accepted, invalid config → {@link InvalidStateError}
 * - no usable device / device unsupported by the tier → {@link UnsupportedDeviceError}
 * - worker init failures (weight verify, OOM, …) propagate with their
 *   {@link WebsamErrorCode} intact (error transfer handler) and the worker is
 *   terminated before rejecting.
 */
export async function createSegmenterImpl(
  config: SegmenterConfig = {},
  internals: CreateSegmenterInternals = {},
): Promise<Segmenter> {
  validateConfig(config);

  // 1. Registry lookup.
  const modelId = config.model ?? DEFAULT_MODEL_ID;
  const spec = getModel(modelId);
  if (!spec) {
    throw new InvalidStateError(
      `createSegmenter: unknown model id '${modelId}' — register it with registerModel() first`,
    );
  }

  // 2. License gate (before any probing or download).
  if (spec.requiresLicenseAcceptance && config.acceptLicense !== 'sam') {
    throw new InvalidStateError(
      `createSegmenter: model '${spec.id}' ships under the '${spec.license}' license, which must be ` +
        `explicitly accepted — pass acceptLicense: 'sam' in the config`,
    );
  }

  // 3. Probe on the main thread, before spawning; then resolve device+quant
  //    and cross-check against the tier's support matrix.
  const [webgpu, wasm] = await Promise.all([WebGpuBackend.probe(), WasmBackend.probe()]);
  const resolution = resolveDevice(
    { device: config.device ?? 'auto', quant: config.quant ?? 'auto' },
    { webgpu, wasm },
  );
  if (!spec.devices[resolution.device]) {
    throw new UnsupportedDeviceError(
      `model '${spec.id}' does not support the resolved device '${resolution.device}' ` +
        `(supported: ${(Object.keys(spec.devices) as ('webgpu' | 'wasm')[])
          .filter((d) => spec.devices[d])
          .join(', ')})`,
    );
  }

  // 4. Spawn the module worker (dynamic import keeps Worker/Comlink wiring
  //    and src/worker/** out of unit-test runtime; tests inject the seam).
  const spawn = internals.spawnWorker ?? (await import('./spawn-worker.js')).spawnWorker;
  const handle = spawn(config.workerUrl);

  // 5. Init the engine (weights + compile run worker-side), progress proxied.
  const onProgress = config.onProgress;
  const initRequest: WorkerInitRequest = {
    spec,
    device: resolution.device,
    quantPreference: resolution.quantPreference,
    modelBaseUrl: config.modelBaseUrl,
    cache: config.cache ?? true,
    wasmPaths: config.wasmPaths,
  };
  let initResult: WorkerInitResult;
  try {
    initResult = await handle.engine.init(
      initRequest,
      onProgress ? Comlink.proxy(onProgress) : undefined,
    );
  } catch (err) {
    handle.terminate();
    throw err;
  }
  onProgress?.({ phase: 'ready' });

  // 6. The Segmenter facade.
  return new SegmenterImpl(handle, spec, initResult);
}

/** The loaded engine facade: owns the worker; sessions are cheap views over it. */
class SegmenterImpl implements Segmenter {
  readonly device: 'webgpu' | 'wasm';
  readonly model: ResolvedModelInfo;

  readonly #handle: WorkerHandle;
  #disposed = false;

  constructor(handle: WorkerHandle, spec: ModelSpec, init: WorkerInitResult) {
    this.#handle = handle;
    this.device = init.device;
    this.model = { spec, quant: init.quant, totalBytes: init.totalBytes };
  }

  #assertLive(method: string): void {
    if (this.#disposed) {
      throw new InvalidStateError(`Segmenter.${method} called on a disposed segmenter`);
    }
  }

  /** Open a new encode-once/decode-per-click image session (worker-side slot). */
  async createImageSession(): Promise<ImageSession> {
    this.#assertLive('createImageSession');
    const sessionId = await this.#handle.engine.createSession();
    return new ImageSessionImpl(this.#handle.engine, sessionId);
  }

  /** @throws NotImplementedError — the video path lands in M2. */
  async createVideoSession(): Promise<VideoSession> {
    this.#assertLive('createVideoSession');
    throw new NotImplementedError('createVideoSession, lands in M2');
  }

  /**
   * Dispose the worker-side engine, then hard-terminate the worker.
   * Double-dispose is a no-op; any other member after dispose throws
   * {@link InvalidStateError}.
   */
  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    try {
      await this.#handle.engine.dispose();
    } finally {
      this.#handle.terminate();
    }
  }
}
