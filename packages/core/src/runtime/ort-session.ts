/**
 * ort InferenceSession creation + the {@link BackendSession} wrapper both
 * browser backends share.
 *
 * ⚠REV ORT#26107: websam NEVER injects a custom GPUDevice into ort — ort
 * creates (and owns) the WebGPU device itself during session creation; any
 * device websam needs to observe is READ from ort post-init.
 */
import type { BackendSession, DeviceTensor, IOBindingPlan } from '../backend/backend.js';
import type { OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError } from '../errors.js';
import { OrtDeviceTensor } from './ort-tensor.js';

type OrtInferenceSession = import('onnxruntime-web').InferenceSession;
type OrtSessionOptions = import('onnxruntime-web').InferenceSession.SessionOptions;

/** Options for {@link createOrtSession}. */
export interface CreateOrtSessionOptions {
  /**
   * Where each graph output should materialize. Honored on webgpu only
   * (`'device'` → `'gpu-buffer'`, `'cpu'` → `'cpu'`); wasm ignores it —
   * everything is cpu there.
   */
  ioPlan?: IOBindingPlan;
}

/**
 * Compile `bytes` into an ort InferenceSession on the requested execution
 * provider. webgpu builds `preferredOutputLocation` from
 * `options.ioPlan.outputLocations`; wasm uses the `['wasm']` provider and
 * ignores any plan.
 */
export function createOrtSession(
  ort: OrtModule,
  kind: 'webgpu' | 'wasm',
  bytes: Uint8Array,
  options?: CreateOrtSessionOptions,
): Promise<OrtInferenceSession> {
  if (kind === 'wasm') {
    return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  }
  const sessionOptions: OrtSessionOptions = { executionProviders: ['webgpu'] };
  const plan = options?.ioPlan;
  if (plan) {
    const preferred: Record<string, 'cpu' | 'gpu-buffer'> = {};
    for (const [output, location] of Object.entries(plan.outputLocations)) {
      preferred[output] = location === 'device' ? 'gpu-buffer' : 'cpu';
    }
    sessionOptions.preferredOutputLocation = preferred;
  }
  return ort.InferenceSession.create(bytes, sessionOptions);
}

/**
 * {@link BackendSession} over an ort InferenceSession: `run()` unwraps
 * {@link OrtDeviceTensor} feeds to raw ort tensors (device tensors bind in
 * place — no CPU round trip), honors `fetches`, and wraps every output back
 * into an {@link OrtDeviceTensor} whose location follows the session's
 * `preferredOutputLocation`.
 */
export class OrtBackendSession implements BackendSession {
  readonly #session: OrtInferenceSession;
  readonly #onDispose: ((session: OrtBackendSession) => void) | undefined;
  #disposed = false;

  /**
   * @param session - The compiled ort session to wrap.
   * @param onDispose - Internal bookkeeping hook (backends untrack the
   * wrapper when it is disposed).
   */
  constructor(session: OrtInferenceSession, onDispose?: (session: OrtBackendSession) => void) {
    this.#session = session;
    this.#onDispose = onDispose;
  }

  /** Whether {@link dispose} has run (internal; used by backend sweeps). */
  get disposed(): boolean {
    return this.#disposed;
  }

  async run(
    feeds: Record<string, DeviceTensor>,
    fetches?: readonly string[],
  ): Promise<Record<string, DeviceTensor>> {
    if (this.#disposed) {
      throw new InvalidStateError('OrtBackendSession.run called on a disposed session');
    }
    const ortFeeds: Record<string, import('onnxruntime-web').Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      if (!(tensor instanceof OrtDeviceTensor)) {
        throw new InvalidStateError(
          `OrtBackendSession.run: feed '${name}' was not created by this backend`,
        );
      }
      ortFeeds[name] = tensor.ortTensor;
    }
    const results =
      fetches === undefined
        ? await this.#session.run(ortFeeds)
        : await this.#session.run(ortFeeds, fetches);
    const outputs: Record<string, DeviceTensor> = {};
    for (const [name, value] of Object.entries(results)) {
      outputs[name] = OrtDeviceTensor.wrap(value);
    }
    return outputs;
  }

  /**
   * Release the compiled session. Idempotent (unlike tensor dispose):
   * `Backend.dispose()` sweeps every tracked session, including ones the
   * caller already released.
   */
  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#onDispose?.(this);
    await this.#session.release();
  }
}
