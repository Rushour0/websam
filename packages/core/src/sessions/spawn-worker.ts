/**
 * Main-thread side of the worker boundary: spawns the module worker hosting
 * the inference engine and wraps it in a Comlink proxy.
 *
 * This module is the ONLY sessions file with a runtime dependency on
 * `src/worker/**` (the error transfer handler) and on `Worker`/Comlink
 * wiring; `segmenter-impl.ts` imports it DYNAMICALLY behind an injectable
 * factory seam so unit tests never touch it (see
 * docs/m1-internal-contracts.md §4.2).
 */

import * as Comlink from 'comlink';
import type { LoadProgressEvent } from '../index.js';
import { installErrorTransferHandler } from '../worker/error-envelope.js';
import type {
  DecodeRequest,
  EncodeResponse,
  MaskPayload,
  WorkerEngineApi,
  WorkerInitRequest,
  WorkerInitResult,
} from '../worker/protocol.js';

/**
 * Promise-typed view of {@link WorkerEngineApi} as seen through
 * `Comlink.wrap` on the main thread: every method returns a promise
 * (including `closeSession`, which is fire-and-forget `void` worker-side).
 *
 * Declared structurally (instead of `Comlink.Remote<WorkerEngineApi>`) so
 * unit tests can satisfy it with a plain object — the Comlink `Remote` type
 * drags in proxy-release symbol members that mocks should not need.
 */
export interface RemoteEngine {
  init(
    req: WorkerInitRequest,
    onProgress?: (e: LoadProgressEvent) => void,
  ): Promise<WorkerInitResult>;
  createSession(): Promise<number>;
  encodeImage(sessionId: number, bitmap: ImageBitmap): Promise<EncodeResponse>;
  decode(sessionId: number, req: DecodeRequest): Promise<MaskPayload[]>;
  closeSession(sessionId: number): Promise<void>;
  dispose(): Promise<void>;
}

/** A spawned engine worker: the Comlink-wrapped API plus its kill switch. */
export interface WorkerHandle {
  engine: RemoteEngine;
  /** Hard-terminate the underlying worker (call after `engine.dispose()`). */
  terminate(): void;
}

/**
 * Spawn the module worker and Comlink-wrap {@link WorkerEngineApi}.
 *
 * Default script URL is `new URL('./worker.js', import.meta.url)`: in dist,
 * `index.js` and `worker.js` are sibling entry points (tsdown multi-entry,
 * docs/m1-internal-contracts.md §6.1), so the relative URL resolves for ESM
 * consumers. `workerUrl` (from `SegmenterConfig.workerUrl`) is the documented
 * escape hatch for bundlers that break sibling-URL resolution — M1 implements
 * only the URL path (the inlined single-file fallback is deferred to M2/M4).
 *
 * Also installs the WebsamError-preserving Comlink transfer handler in THIS
 * realm (the worker entry installs its own side).
 */
export function spawnWorker(workerUrl?: string | URL): WorkerHandle {
  installErrorTransferHandler(Comlink);
  const worker = new Worker(workerUrl ?? new URL('./worker.js', import.meta.url), {
    type: 'module',
  });
  const remote = Comlink.wrap<WorkerEngineApi>(worker);
  return {
    engine: remote as unknown as RemoteEngine,
    terminate() {
      remote[Comlink.releaseProxy]();
      worker.terminate();
    },
  };
}
