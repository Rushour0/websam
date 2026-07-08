/**
 * Browser-only WeightStore coverage: the OPFS marker-file atomic commit and
 * the Cache API path need a real browser (node has neither); the node unit
 * suite covers verification logic and the in-memory degrade.
 */
import { describe, expect, it } from 'vitest';
import { WeightVerifyError } from '../errors.js';
import type { WeightFileRef } from './manifest.js';
import {
  CacheApiWeightStore,
  createWeightStore,
  MemoryWeightStore,
  OpfsWeightStore,
  type WeightStore,
} from './weight-store.js';

/** Unique random content per test run so persisted state never collides across runs. */
function randomContent(length: number): Uint8Array {
  const out = new Uint8Array(length);
  // crypto.getRandomValues caps each call at 65,536 bytes in browsers.
  const STEP = 65_536;
  for (let offset = 0; offset < length; offset += STEP) {
    crypto.getRandomValues(out.subarray(offset, Math.min(offset + STEP, length)));
  }
  return out;
}

async function refFor(bytes: Uint8Array, path = 'graph.onnx'): Promise<WeightFileRef> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as BufferSource);
  const sha256 = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  return { path, sha256, bytes: bytes.byteLength };
}

function streamOf(...chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller): void {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

function roundTripSuite(name: string, makeStore: () => WeightStore): void {
  describe(name, () => {
    it('round-trips put → has/get, then deletes', async () => {
      const store = makeStore();
      const bytes = randomContent(70_000);
      const ref = await refFor(bytes);

      expect(await store.has(ref)).toBe(false);
      expect(await store.get(ref)).toBeUndefined();

      const blob = await store.put(ref, streamOf(bytes.subarray(0, 30_000), bytes.subarray(30_000)));
      expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);

      expect(await store.has(ref)).toBe(true);
      const got = await store.get(ref);
      expect(got).toBeDefined();
      expect(new Uint8Array(await (got as Blob).arrayBuffer())).toEqual(bytes);

      await store.delete(ref);
      expect(await store.has(ref)).toBe(false);
      expect(await store.get(ref)).toBeUndefined();
    });

    it('discards everything on digest mismatch (nothing cached, WeightVerifyError)', async () => {
      const store = makeStore();
      const bytes = randomContent(1024);
      const ref: WeightFileRef = { ...(await refFor(bytes)), sha256: 'f'.repeat(64) };

      await expect(store.put(ref, streamOf(bytes))).rejects.toThrow(WeightVerifyError);
      expect(await store.has(ref)).toBe(false);
      expect(await store.get(ref)).toBeUndefined();
    });

    it('discards everything when the source stream errors mid-flight', async () => {
      const store = makeStore();
      const bytes = randomContent(512);
      const ref = await refFor(bytes);
      const broken = new ReadableStream<Uint8Array>({
        start(controller): void {
          controller.enqueue(bytes.subarray(0, 100));
          controller.error(new TypeError('connection reset'));
        },
      });

      await expect(store.put(ref, broken)).rejects.toThrow('connection reset');
      expect(await store.has(ref)).toBe(false);
      expect(await store.get(ref)).toBeUndefined();
    });
  });
}

roundTripSuite('OpfsWeightStore (browser)', () => new OpfsWeightStore());
roundTripSuite('CacheApiWeightStore (browser)', () => new CacheApiWeightStore());
roundTripSuite('MemoryWeightStore (browser)', () => new MemoryWeightStore());
roundTripSuite('createWeightStore (browser routing)', () => createWeightStore());

describe('OPFS marker-file atomicity', () => {
  it('ignores a .bin without its .ok marker (crashed partial write is never served)', async () => {
    const bytes = randomContent(256);
    const ref = await refFor(bytes);

    // Simulate a crash between the .bin write and the .ok commit by writing
    // the payload file directly, bypassing the store.
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('websam-weights', { create: true });
    const bin = await dir.getFileHandle(`${ref.sha256}.bin`, { create: true });
    const writable = await bin.createWritable();
    await writable.write(bytes as BufferSource);
    await writable.close();

    const store = new OpfsWeightStore();
    expect(await store.has(ref)).toBe(false);
    expect(await store.get(ref)).toBeUndefined();

    // A subsequent put over the stale partial commits properly.
    const blob = await store.put(ref, streamOf(bytes));
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);
    expect(await store.has(ref)).toBe(true);
    await store.delete(ref);
  });

  it('re-putting existing content overwrites and stays committed', async () => {
    const bytes = randomContent(128);
    const ref = await refFor(bytes);
    const store = new OpfsWeightStore();

    await store.put(ref, streamOf(bytes));
    await store.put(ref, streamOf(bytes)); // idempotent re-put
    expect(await store.has(ref)).toBe(true);
    expect(new Uint8Array(await ((await store.get(ref)) as Blob).arrayBuffer())).toEqual(bytes);
    await store.delete(ref);
  });
});
