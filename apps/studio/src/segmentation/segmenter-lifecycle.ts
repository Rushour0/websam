/**
 * Owns the single active {@link Segmenter} instance for the app session (one
 * `createSegmenter()` call at a time, not one per clip — see
 * studio-contracts.md §4.1), but unlike the original single-model version
 * this module can SWITCH which model tier is loaded: `loadSegmenter(modelId)`
 * disposes whatever tier is currently loaded (if different) and loads the
 * requested one.
 *
 * Mirrors `apps/demo/src/VideoTab.tsx`'s `loadModel` config: `device: 'auto'`,
 * weights served from `/models/<modelId>/`, worker spawned via the
 * `@websam3/core/worker?worker&url` Vite escape hatch. The model id is now a
 * PARAMETER — EdgeTAM is only `studio-store.ts`'s default `selectedModelId`,
 * not a hardcoded constant here.
 */
import { createSegmenter, getModel } from '@websam3/core';
import type { LoadProgressEvent, Segmenter } from '@websam3/core';
// Vite bundles the core worker entry as a dedicated worker chunk and hands us
// its URL — same pattern the demo uses.
import segmenterWorkerUrl from '@websam3/core/worker?worker&url';

/**
 * Per-model weights base URL. `setup-weights.mjs` stages EdgeTAM under
 * `public/models/edgetam/`; other tiers (SAM3) have no real manifest/CDN yet
 * (M3, tracked separately) but still resolve to a same-shaped local path so
 * a missing manifest fails with a clear 404 at download time, not a crash.
 * `VITE_WEBSAM_MODELS` (if set) overrides ONLY the default 'edgetam' base,
 * preserving the exact prior single-model override behavior.
 */
function resolveModelBaseUrl(modelId: string): string {
  if (modelId === 'edgetam') {
    return (import.meta.env?.VITE_WEBSAM_MODELS as string | undefined) ?? '/models/edgetam/';
  }
  return `/models/${modelId}/`;
}

/** Kept for any existing external reference; prefer `resolveModelBaseUrl(modelId)`. */
export const MODEL_BASE_URL = resolveModelBaseUrl('edgetam');

/** Optional override for onnxruntime-web's .wasm/.mjs asset base inside the worker. */
const ORT_WASM_PATHS = import.meta.env?.VITE_ORT_WASM_PATHS as string | undefined;

let generation = 0;
let loaded: { modelId: string; segmenter: Segmenter } | null = null;
let loading: { modelId: string; promise: Promise<Segmenter> } | null = null;

/**
 * Load (or return the already-loading/loaded) segmenter for `modelId`. Memoized per model id —
 * repeated calls for the SAME id are idempotent and share one worker. Calling with a DIFFERENT id
 * than what's currently loaded disposes the old segmenter (freeing its worker/GPU resources) and
 * starts a fresh load; a `generation` counter guards against a stale in-flight load clobbering
 * `loaded` if the caller switches models again before the first load settles.
 *
 * `acceptLicense: 'sam'` is passed automatically whenever the requested tier's registry spec sets
 * `requiresLicenseAcceptance` — Studio's own consent-dialog flow (studio-store.ts) is what decides
 * WHETHER to call this in the first place; this function does not re-implement that gate, it just
 * satisfies createSegmenter's own API-level gate (segmenter-impl.ts) once Studio has already gated it.
 */
export function loadSegmenter(
  modelId: string,
  onProgress?: (event: LoadProgressEvent) => void,
): Promise<Segmenter> {
  if (loaded && loaded.modelId === modelId) return Promise.resolve(loaded.segmenter);
  if (loading && loading.modelId === modelId) return loading.promise;

  const myGeneration = ++generation;
  const stale = loaded;
  loaded = null;
  if (stale) void stale.segmenter.dispose();

  const spec = getModel(modelId);
  const promise = createSegmenter({
    model: modelId,
    device: 'auto',
    modelBaseUrl: resolveModelBaseUrl(modelId),
    workerUrl: segmenterWorkerUrl,
    wasmPaths: ORT_WASM_PATHS,
    acceptLicense: spec?.requiresLicenseAcceptance ? 'sam' : undefined,
    onProgress,
  }).then((segmenter) => {
    if (generation !== myGeneration) {
      // Superseded by a newer model switch while this load was in flight — discard it.
      void segmenter.dispose();
      return segmenter;
    }
    loaded = { modelId, segmenter };
    return segmenter;
  });

  loading = { modelId, promise };
  promise.finally(() => {
    if (loading?.promise === promise) loading = null;
  });
  return promise;
}

/** The model id currently loaded or in flight, or null before any load starts. */
export function currentModelId(): string | null {
  return loaded?.modelId ?? loading?.modelId ?? null;
}

/**
 * The currently loaded segmenter, or null if none is loaded yet (still loading, never started,
 * or disposed). This is the sole read path session-manager.ts's `requireSegmenter()` uses to
 * obtain the active segmenter for activateClip/addPromptObject/refineObject/startTracking —
 * it must stay in sync with `loaded`, not `loading`.
 */
export function getSegmenter(): Segmenter | null {
  return loaded?.segmenter ?? null;
}

/** True once a segmenter has finished loading and is ready to use (mirrors `getSegmenter() !== null`). */
export function isSegmenterReady(): boolean {
  return loaded !== null;
}

/**
 * Dispose the currently loaded segmenter (if any) and cancel any in-flight load by bumping
 * `generation`, so a load that settles after this call discards its result instead of
 * repopulating `loaded`. Called from App.tsx's `pagehide`/unmount cleanup. Safe to call when
 * nothing is loaded/loading (no-op).
 */
export async function disposeSegmenter(): Promise<void> {
  generation++;
  loading = null;
  const current = loaded;
  loaded = null;
  if (current) await current.segmenter.dispose();
}
