import { useEffect, useState } from 'react';
import type { Segmenter, SegmenterConfig } from '@websam/core';

/**
 * Lifecycle status of a segmenter load driven by {@link useSegmenter}.
 *
 * - `'idle'` — the hook has rendered but the load effect has not run yet.
 * - `'loading'` — the loader is in flight.
 * - `'ready'` — the segmenter resolved and is usable.
 * - `'error'` — the loader rejected (at M0 core's `createSegmenter` rejects
 *   with a `NotImplementedError`).
 */
export type UseSegmenterStatus = 'idle' | 'loading' | 'ready' | 'error';

/**
 * Context passed to a {@link SegmenterLoader} by {@link useSegmenter}.
 */
export interface SegmenterLoaderContext {
  /**
   * Aborted (one microtask after) the last mounted hook sharing this load
   * unmounts. Loaders should stop downloading model weights when it fires.
   */
  signal: AbortSignal;
  /**
   * Report load progress as a fraction in `[0, 1]`. Forwarded to every
   * mounted hook sharing this load as the `progress` result field.
   */
  onProgress: (progress: number) => void;
}

/**
 * The injectable loading seam used by {@link useSegmenter}. The default
 * loader calls `createSegmenter` from `@websam/core`; tests (and advanced
 * consumers) may supply their own.
 */
export type SegmenterLoader = (
  config: SegmenterConfig | undefined,
  context: SegmenterLoaderContext,
) => Promise<Segmenter>;

/**
 * Options accepted by {@link useSegmenter}.
 */
export interface UseSegmenterOptions {
  /**
   * Replaces the default `@websam/core` loader. The loader's function
   * identity participates in the cache key, so pass a stable reference.
   */
  loader?: SegmenterLoader;
}

/**
 * Snapshot returned by {@link useSegmenter}.
 */
export interface UseSegmenterResult {
  /** The loaded segmenter, or `null` until `status` is `'ready'`. */
  segmenter: Segmenter | null;
  /** Current lifecycle status of the load. */
  status: UseSegmenterStatus;
  /** Latest loader-reported progress in `[0, 1]`, if any was reported. */
  progress?: number;
  /** The rejection reason when `status` is `'error'`. */
  error?: Error;
}

/**
 * Internal shared-load record. One entry exists per (loader, config-key)
 * pair; every mounted hook with that pair subscribes to the same entry.
 */
interface CacheEntry {
  promise: Promise<Segmenter>;
  controller: AbortController;
  refCount: number;
  status: Exclude<UseSegmenterStatus, 'idle'>;
  segmenter: Segmenter | null;
  progress?: number;
  error?: Error;
  subscribers: Set<() => void>;
}

/**
 * Module-level cache of in-flight (and settled) loads, keyed first by loader
 * identity, then by the stable-serialized config. Keeping it at module level
 * is what makes the hook StrictMode-safe: React 18 dev's synchronous
 * mount → cleanup → remount cycle re-acquires the same entry instead of
 * starting a second load.
 */
const caches = new WeakMap<SegmenterLoader, Map<string, CacheEntry>>();

/**
 * Default loader: defers to `createSegmenter` from `@websam/core`. Imported
 * dynamically so the core runtime is only touched when this path actually
 * runs (at M0 it rejects with core's `NotImplementedError`).
 */
const defaultLoader: SegmenterLoader = async (config) => {
  const core = await import('@websam/core');
  return core.createSegmenter(config);
};

/**
 * Deterministically serialize a config so structurally-equal configs map to
 * the same cache entry regardless of key insertion order.
 */
function stableStringify(value: unknown): string {
  if (value === undefined) return 'undefined';
  return JSON.stringify(value, (_key, val: unknown) =>
    val !== null && typeof val === 'object' && !Array.isArray(val)
      ? Object.fromEntries(
          Object.entries(val).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
        )
      : val,
  );
}

function notify(entry: CacheEntry): void {
  for (const subscriber of entry.subscribers) subscriber();
}

function acquire(
  loader: SegmenterLoader,
  key: string,
  config: SegmenterConfig | undefined,
): CacheEntry {
  let byKey = caches.get(loader);
  if (!byKey) {
    byKey = new Map();
    caches.set(loader, byKey);
  }
  let entry = byKey.get(key);
  if (!entry) {
    const controller = new AbortController();
    const created: CacheEntry = {
      controller,
      refCount: 0,
      status: 'loading',
      segmenter: null,
      subscribers: new Set(),
      // Assigned immediately below; the async wrapper needs `created` in scope.
      promise: undefined as unknown as Promise<Segmenter>,
    };
    created.promise = (async () =>
      loader(config, {
        signal: controller.signal,
        onProgress: (progress) => {
          created.progress = progress;
          notify(created);
        },
      }))();
    created.promise.then(
      (segmenter) => {
        created.segmenter = segmenter;
        created.status = 'ready';
        notify(created);
      },
      (cause: unknown) => {
        created.status = 'error';
        created.error = cause instanceof Error ? cause : new Error(String(cause));
        notify(created);
      },
    );
    byKey.set(key, created);
    entry = created;
  }
  entry.refCount += 1;
  return entry;
}

function release(loader: SegmenterLoader, key: string, entry: CacheEntry): void {
  entry.refCount -= 1;
  if (entry.refCount > 0) return;
  // Defer disposal one microtask: StrictMode's synchronous remount will have
  // re-acquired the entry by then, so a dev double-mount never aborts or
  // restarts the load.
  queueMicrotask(() => {
    if (entry.refCount > 0) return;
    entry.controller.abort();
    const byKey = caches.get(loader);
    if (byKey?.get(key) === entry) byKey.delete(key);
  });
}

/**
 * Load (or join an in-flight load of) a shared {@link Segmenter} for the
 * given config.
 *
 * Loads are deduplicated across components and across React 18 StrictMode's
 * dev double-mount via a config-keyed module-level cache, and aborted via
 * `AbortController` once the last subscribed component unmounts.
 *
 * At M0, core's `createSegmenter` rejects with a `NotImplementedError`, which
 * this hook surfaces as `status: 'error'` with the error attached.
 *
 * @param config - Segmenter configuration; structurally-equal configs share
 *   one load. Omit for core defaults.
 * @param options - Advanced options, e.g. an injected {@link SegmenterLoader}.
 * @returns The current {@link UseSegmenterResult} snapshot.
 */
export function useSegmenter(
  config?: SegmenterConfig,
  options?: UseSegmenterOptions,
): UseSegmenterResult {
  const loader = options?.loader ?? defaultLoader;
  const key = stableStringify(config);
  const [snapshot, setSnapshot] = useState<UseSegmenterResult>({
    segmenter: null,
    status: 'idle',
  });

  useEffect(() => {
    // `config` is intentionally read here without being an effect dependency:
    // `key` is its stable serialization, so a same-key rerun would pass an
    // equivalent value anyway.
    const entry = acquire(loader, key, config);
    const update = () => {
      setSnapshot({
        segmenter: entry.segmenter,
        status: entry.status,
        progress: entry.progress,
        error: entry.error,
      });
    };
    entry.subscribers.add(update);
    update();
    return () => {
      entry.subscribers.delete(update);
      release(loader, key, entry);
    };
  }, [loader, key]);

  return snapshot;
}
