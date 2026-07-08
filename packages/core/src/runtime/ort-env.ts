/**
 * onnxruntime-web bootstrap — the ONE place ort enters the process.
 *
 * ort is loaded via a dynamic `import('onnxruntime-web')` (never a static
 * top-level import) so it stays out of non-worker bundles, and its global
 * `env` flags are applied exactly once, BEFORE any session exists — ort
 * reads them lazily at first session creation, so late mutation is a
 * silent misconfiguration, not an error. {@link loadOrt} memoizes the
 * import + flag application to enforce that ordering.
 */
import type { OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError } from '../errors.js';

/** Options applied to onnxruntime-web's global `env` on first load. */
export interface OrtEnvOptions {
  /** ort .wasm/.mjs asset base; default: ort's own `import.meta.url` resolution. */
  wasmPaths?: string;
  /**
   * WASM thread count. Default: undefined (ort default) when
   * `crossOriginIsolated`, else forced to 1 — without COOP/COEP there is no
   * `SharedArrayBuffer`, so pinning a single thread avoids ort's probe.
   */
  numThreads?: number;
}

interface NormalizedOptions {
  wasmPaths: string | undefined;
  numThreads: number | undefined;
}

let memo: { options: NormalizedOptions; promise: Promise<OrtModule> } | undefined;

/**
 * Dynamic-imports onnxruntime-web exactly once (memoized), applying env
 * flags BEFORE any session exists: `env.wasm.wasmPaths` (iff provided),
 * `env.wasm.numThreads` (see {@link OrtEnvOptions.numThreads}), and
 * `env.wasm.proxy = false` (we already run inside a worker).
 *
 * Subsequent calls with the same options return the memoized module; a
 * second call with DIFFERENT options throws {@link InvalidStateError},
 * because the flags it asks for can no longer take effect.
 */
export function loadOrt(options?: OrtEnvOptions): Promise<OrtModule> {
  const normalized: NormalizedOptions = {
    wasmPaths: options?.wasmPaths,
    numThreads: options?.numThreads,
  };
  if (memo) {
    if (
      memo.options.wasmPaths !== normalized.wasmPaths ||
      memo.options.numThreads !== normalized.numThreads
    ) {
      throw new InvalidStateError(
        'loadOrt already ran with different options — ort env flags apply once, before any session exists',
      );
    }
    return memo.promise;
  }
  memo = { options: normalized, promise: importAndConfigure(normalized) };
  return memo.promise;
}

async function importAndConfigure(options: NormalizedOptions): Promise<OrtModule> {
  const ort = await import('onnxruntime-web');
  if (options.wasmPaths !== undefined) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.numThreads !== undefined) {
    ort.env.wasm.numThreads = options.numThreads;
  } else if ((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated !== true) {
    ort.env.wasm.numThreads = 1;
  }
  // websam always drives ort from inside its own worker; ort's proxy worker
  // would only add a second thread hop.
  ort.env.wasm.proxy = false;
  return ort;
}
