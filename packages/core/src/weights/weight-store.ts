/**
 * Content-addressed local storage for verified weight files.
 *
 * Storage identity is the file's sha256 (content-addressed — shared graphs
 * dedupe across tiers). Two persistent backends, routed per file size by
 * {@link createWeightStore}:
 *
 * - OPFS for large files (> {@link OPFS_THRESHOLD_BYTES}) — true streaming
 *   writes, atomic commit via the marker-file pattern (decided; portable —
 *   `FileSystemFileHandle.move()` is not universal): write `<sha256>.bin`,
 *   verify while streaming, then create zero-byte `<sha256>.ok`. `has`/`get`
 *   require BOTH files, so a crashed partial write is never served.
 * - Cache API for small files — `put` buffers (only ever ≤64 MiB files).
 *
 * Both degrade gracefully: OPFS missing → Cache API; both missing →
 * in-memory passthrough (no persistence, never an error). Runs in window and
 * worker contexts (M1 uses it in the worker).
 */

import { WeightVerifyError } from '../errors.js';
import type { WeightFileRef } from './manifest.js';
import { Sha256Stream } from './sha256.js';

/** Local store of verified weight files, keyed by content (sha256). */
export interface WeightStore {
  /** True iff a fully committed, previously verified copy exists. */
  has(ref: WeightFileRef): Promise<boolean>;
  /** Verified content, or undefined on miss. Never returns a partially written file. */
  get(ref: WeightFileRef): Promise<Blob | undefined>;
  /**
   * Stream `data` to storage, hashing as it writes. On digest mismatch with
   * `ref.sha256`: discard everything, throw WeightVerifyError (nothing cached).
   * On success: atomically commit and return the stored content.
   */
  put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob>;
  delete(ref: WeightFileRef): Promise<void>;
}

/** Files larger than this go to OPFS (streaming writes); smaller to the Cache API. */
export const OPFS_THRESHOLD_BYTES = 64 * 1024 * 1024;

/** OPFS directory all weight files live under. */
const OPFS_DIR = 'websam-weights';
/** Cache API bucket name. */
const CACHE_NAME = 'websam-weights';

function digestMismatch(ref: WeightFileRef, actual: string): WeightVerifyError {
  return new WeightVerifyError(
    `Weight file '${ref.path}' failed integrity verification: expected sha256 ${ref.sha256}, got ${actual}`,
  );
}

/**
 * Drain `data`, hashing every chunk and forwarding it to `sink`. Throws
 * {@link WeightVerifyError} when the final digest differs from `ref.sha256`.
 */
async function pumpVerified(
  ref: WeightFileRef,
  data: ReadableStream<Uint8Array>,
  sink: (chunk: Uint8Array) => void | Promise<void>,
): Promise<void> {
  const hash = new Sha256Stream();
  const reader = data.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    hash.update(value);
    await sink(value);
  }
  const actual = hash.digestHex();
  if (actual !== ref.sha256) {
    throw digestMismatch(ref, actual);
  }
}

/** Drain + verify `data` into one contiguous (plain-ArrayBuffer-backed) Uint8Array. */
async function readAllVerified(
  ref: WeightFileRef,
  data: ReadableStream<Uint8Array>,
): Promise<Uint8Array<ArrayBuffer>> {
  const chunks: Uint8Array[] = [];
  let total = 0;
  await pumpVerified(ref, data, (chunk) => {
    chunks.push(chunk);
    total += chunk.byteLength;
  });
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/**
 * Non-persistent {@link WeightStore}: verifies and holds content in a Map.
 * Used for tests, `cache: false`, and as the everything-missing fallback.
 */
export class MemoryWeightStore implements WeightStore {
  readonly #files = new Map<string, Uint8Array<ArrayBuffer>>();

  async has(ref: WeightFileRef): Promise<boolean> {
    return this.#files.has(ref.sha256);
  }

  async get(ref: WeightFileRef): Promise<Blob | undefined> {
    const bytes = this.#files.get(ref.sha256);
    return bytes ? new Blob([bytes]) : undefined;
  }

  async put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob> {
    const bytes = await readAllVerified(ref, data); // throws before anything is stored
    this.#files.set(ref.sha256, bytes);
    return new Blob([bytes]);
  }

  async delete(ref: WeightFileRef): Promise<void> {
    this.#files.delete(ref.sha256);
  }
}

/**
 * OPFS-backed {@link WeightStore} for large files. Atomic commit = marker
 * file: `<sha256>.bin` is only trusted when zero-byte `<sha256>.ok` exists
 * (created strictly after the verified write completes).
 */
export class OpfsWeightStore implements WeightStore {
  async #dir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(OPFS_DIR, { create: true });
  }

  async has(ref: WeightFileRef): Promise<boolean> {
    try {
      const dir = await this.#dir();
      await dir.getFileHandle(`${ref.sha256}.ok`);
      await dir.getFileHandle(`${ref.sha256}.bin`);
      return true;
    } catch {
      return false;
    }
  }

  async get(ref: WeightFileRef): Promise<Blob | undefined> {
    try {
      const dir = await this.#dir();
      await dir.getFileHandle(`${ref.sha256}.ok`); // both required — no partial reads
      const bin = await dir.getFileHandle(`${ref.sha256}.bin`);
      return await bin.getFile();
    } catch {
      return undefined;
    }
  }

  async put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob> {
    const dir = await this.#dir();
    // Invalidate any previous commit of this content before rewriting.
    await removeIfPresent(dir, `${ref.sha256}.ok`);
    const bin = await dir.getFileHandle(`${ref.sha256}.bin`, { create: true });
    const writable = await bin.createWritable();
    try {
      // Fetch/network chunks are always plain-ArrayBuffer-backed; the cast
      // bridges TS 5.9's `ArrayBufferLike` default on ReadableStream chunks.
      await pumpVerified(ref, data, (chunk) => writable.write(chunk as Uint8Array<ArrayBuffer>));
    } catch (err) {
      await writable.abort().catch(() => undefined);
      await removeIfPresent(dir, `${ref.sha256}.bin`);
      throw err;
    }
    await writable.close();
    // Commit: the zero-byte marker makes the .bin visible to has()/get().
    await dir.getFileHandle(`${ref.sha256}.ok`, { create: true });
    return bin.getFile();
  }

  async delete(ref: WeightFileRef): Promise<void> {
    const dir = await this.#dir();
    await removeIfPresent(dir, `${ref.sha256}.ok`); // marker first — never orphan a trusted .bin
    await removeIfPresent(dir, `${ref.sha256}.bin`);
  }
}

async function removeIfPresent(dir: FileSystemDirectoryHandle, name: string): Promise<void> {
  try {
    await dir.removeEntry(name);
  } catch {
    // Missing entry — nothing to remove.
  }
}

/**
 * Cache API-backed {@link WeightStore} for small files, under the synthetic
 * key `https://websam.invalid/weights/<sha256>`. `put` buffers the whole
 * file — this store only ever receives files ≤ {@link OPFS_THRESHOLD_BYTES}.
 */
export class CacheApiWeightStore implements WeightStore {
  #key(ref: WeightFileRef): string {
    return `https://websam.invalid/weights/${ref.sha256}`;
  }

  async has(ref: WeightFileRef): Promise<boolean> {
    const cache = await caches.open(CACHE_NAME);
    return (await cache.match(this.#key(ref))) !== undefined;
  }

  async get(ref: WeightFileRef): Promise<Blob | undefined> {
    const cache = await caches.open(CACHE_NAME);
    const res = await cache.match(this.#key(ref));
    return res ? res.blob() : undefined;
  }

  async put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob> {
    const bytes = await readAllVerified(ref, data); // throws before anything is stored
    const cache = await caches.open(CACHE_NAME);
    const blob = new Blob([bytes]);
    await cache.put(this.#key(ref), new Response(blob));
    return blob;
  }

  async delete(ref: WeightFileRef): Promise<void> {
    const cache = await caches.open(CACHE_NAME);
    await cache.delete(this.#key(ref));
  }
}

function opfsAvailable(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.storage?.getDirectory === 'function'
  );
}

function cacheApiAvailable(): boolean {
  return typeof caches !== 'undefined' && typeof caches.open === 'function';
}

/** Routes each ref to one backing store, deterministically by `ref.bytes`. */
class RoutingWeightStore implements WeightStore {
  readonly #large: WeightStore;
  readonly #small: WeightStore;

  constructor(large: WeightStore, small: WeightStore) {
    this.#large = large;
    this.#small = small;
  }

  #pick(ref: WeightFileRef): WeightStore {
    return ref.bytes > OPFS_THRESHOLD_BYTES ? this.#large : this.#small;
  }

  has(ref: WeightFileRef): Promise<boolean> {
    return this.#pick(ref).has(ref);
  }

  get(ref: WeightFileRef): Promise<Blob | undefined> {
    return this.#pick(ref).get(ref);
  }

  put(ref: WeightFileRef, data: ReadableStream<Uint8Array>): Promise<Blob> {
    return this.#pick(ref).put(ref, data);
  }

  delete(ref: WeightFileRef): Promise<void> {
    return this.#pick(ref).delete(ref);
  }
}

/**
 * Create the environment's best {@link WeightStore}.
 *
 * Routing per file: `ref.bytes > OPFS_THRESHOLD_BYTES` → OPFS, else Cache
 * API. Degrades feature-by-feature (OPFS missing → Cache API; both missing →
 * in-memory passthrough — no persistence, never an error), so it also works
 * in node unit tests.
 */
export function createWeightStore(): WeightStore {
  const opfs = opfsAvailable() ? new OpfsWeightStore() : undefined;
  const cacheApi = cacheApiAvailable() ? new CacheApiWeightStore() : undefined;
  const memory = opfs && cacheApi ? undefined : new MemoryWeightStore();
  const large = opfs ?? cacheApi ?? (memory as WeightStore);
  const small = cacheApi ?? opfs ?? (memory as WeightStore);
  return new RoutingWeightStore(large, small);
}
