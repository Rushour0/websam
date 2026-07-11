/**
 * Owns the single {@link Segmenter} instance for the app session (one
 * `createSegmenter()` call, not one per clip — see studio-contracts.md §4.1).
 *
 * Mirrors `apps/demo/src/VideoTab.tsx`'s `loadModel` config exactly:
 * EdgeTAM tier, `device: 'auto'`, weights served from `/models/`, worker
 * spawned via the `@websam3/core/worker?worker&url` Vite escape hatch.
 */
import { createSegmenter } from '@websam3/core';
import type { LoadProgressEvent, Segmenter } from '@websam3/core';
// Vite bundles the core worker entry as a dedicated worker chunk and hands us
// its URL — same pattern the demo uses.
import segmenterWorkerUrl from '@websam3/core/worker?worker&url';

/** The tier the studio drives at M2/A4: EdgeTAM (Apache-2.0, no license gate). */
export const MODEL_ID = 'edgetam';

/**
 * Where the EdgeTAM manifest + weights are served from. `setup-weights.mjs`
 * stages them under `public/models/edgetam/`, and `resolveManifestUrl` rebases
 * the manifest's *filename* (`manifest.json`) + each graph's relative file path
 * onto this base — so it must point at the tier dir, not just `/models/`.
 * Deployments may override with `VITE_WEBSAM_MODELS`.
 */
export const MODEL_BASE_URL =
  (import.meta.env?.VITE_WEBSAM_MODELS as string | undefined) ?? '/models/edgetam/';

/** Optional override for onnxruntime-web's .wasm/.mjs asset base inside the worker. */
const ORT_WASM_PATHS = import.meta.env?.VITE_ORT_WASM_PATHS as string | undefined;

let segmenterPromise: Promise<Segmenter> | null = null;
let segmenterInstance: Segmenter | null = null;

/**
 * Load (or return the already-loading/loaded) app-session {@link Segmenter}.
 *
 * Memoized module-level so repeated calls (e.g. `store.loadModel()` invoked
 * from multiple components) are idempotent and share one worker + one set of
 * loaded weights. `onProgress` is forwarded to every concurrent caller for
 * the duration of an in-flight load; callers that arrive after load finishes
 * get the resolved segmenter immediately without a fresh progress stream.
 *
 * A failed load clears the memo so the caller can retry via `loadSegmenter`
 * again (mirrors the demo's error → idle → retry flow).
 */
export function loadSegmenter(onProgress?: (event: LoadProgressEvent) => void): Promise<Segmenter> {
  if (segmenterInstance) return Promise.resolve(segmenterInstance);
  if (segmenterPromise) return segmenterPromise;

  segmenterPromise = createSegmenter({
    model: MODEL_ID,
    device: 'auto',
    modelBaseUrl: MODEL_BASE_URL,
    workerUrl: segmenterWorkerUrl,
    wasmPaths: ORT_WASM_PATHS,
    onProgress,
  })
    .then((segmenter) => {
      segmenterInstance = segmenter;
      return segmenter;
    })
    .catch((err: unknown) => {
      // Allow a subsequent loadSegmenter() call to retry from scratch.
      segmenterPromise = null;
      throw err;
    });

  return segmenterPromise;
}

/** The loaded segmenter, or `null` if `loadSegmenter()` hasn't resolved yet. */
export function getSegmenter(): Segmenter | null {
  return segmenterInstance;
}

/** True once `loadSegmenter()` has resolved successfully. */
export function isSegmenterReady(): boolean {
  return segmenterInstance !== null;
}

/**
 * Dispose the app-session segmenter and its worker. Callers must dispose all
 * per-clip `VideoSession`s first (see `session-manager.ts`'s `disposeAllClips`)
 * — this only tears down the shared segmenter.
 */
export async function disposeSegmenter(): Promise<void> {
  const segmenter = segmenterInstance;
  segmenterInstance = null;
  segmenterPromise = null;
  await segmenter?.dispose();
}
