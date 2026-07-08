import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { WeightVerifyError } from '../errors.js';
import type { WeightFileRef } from './manifest.js';
import { createWeightStore, MemoryWeightStore, OPFS_THRESHOLD_BYTES } from './weight-store.js';

function refFor(bytes: Uint8Array, path = 'graph.onnx'): WeightFileRef {
  return {
    path,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    bytes: bytes.byteLength,
  };
}

function streamOf(...chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller): void {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

/** Deterministic pseudo-random content so digests are stable per test. */
function content(length: number, seed: number): Uint8Array {
  const out = new Uint8Array(length);
  let state = seed;
  for (let i = 0; i < length; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out[i] = state & 0xff;
  }
  return out;
}

describe('MemoryWeightStore', () => {
  it('round-trips a chunked put through get/has and deletes', async () => {
    const store = new MemoryWeightStore();
    const bytes = content(1000, 1);
    const ref = refFor(bytes);

    expect(await store.has(ref)).toBe(false);
    expect(await store.get(ref)).toBeUndefined();

    const blob = await store.put(ref, streamOf(bytes.subarray(0, 300), bytes.subarray(300)));
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);

    expect(await store.has(ref)).toBe(true);
    const got = await store.get(ref);
    expect(got).toBeDefined();
    expect(new Uint8Array(await (got as Blob).arrayBuffer())).toEqual(bytes);

    await store.delete(ref);
    expect(await store.has(ref)).toBe(false);
    expect(await store.get(ref)).toBeUndefined();
  });

  it('rejects a digest mismatch with WeightVerifyError and caches nothing', async () => {
    const store = new MemoryWeightStore();
    const bytes = content(500, 2);
    const ref: WeightFileRef = { ...refFor(bytes), sha256: 'f'.repeat(64) };

    await expect(store.put(ref, streamOf(bytes))).rejects.toThrow(WeightVerifyError);
    await expect(store.put(ref, streamOf(bytes))).rejects.toThrow(/graph\.onnx/);
    expect(await store.has(ref)).toBe(false);
    expect(await store.get(ref)).toBeUndefined();
  });

  it('propagates an errored source stream and caches nothing', async () => {
    const store = new MemoryWeightStore();
    const bytes = content(100, 3);
    const ref = refFor(bytes);
    const broken = new ReadableStream<Uint8Array>({
      start(controller): void {
        controller.enqueue(bytes.subarray(0, 10));
        controller.error(new TypeError('network dropped'));
      },
    });

    await expect(store.put(ref, broken)).rejects.toThrow('network dropped');
    expect(await store.has(ref)).toBe(false);
  });

  it('is content-addressed: the same bytes under two paths share one entry', async () => {
    const store = new MemoryWeightStore();
    const bytes = content(64, 4);
    const refA = refFor(bytes, 'a.onnx');
    const refB = refFor(bytes, 'b.onnx');

    await store.put(refA, streamOf(bytes));
    expect(await store.has(refB)).toBe(true); // same sha256 → same entry
  });
});

describe('createWeightStore (node: no OPFS, no Cache API)', () => {
  it('degrades to a working in-memory store instead of throwing', async () => {
    const store = createWeightStore();
    const bytes = content(2048, 5);
    const ref = refFor(bytes);

    expect(await store.has(ref)).toBe(false);
    const blob = await store.put(ref, streamOf(bytes));
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);
    expect(await store.has(ref)).toBe(true);

    await store.delete(ref);
    expect(await store.has(ref)).toBe(false);
  });

  it('routes large and small refs consistently (put then get hits)', async () => {
    const store = createWeightStore();
    const small = content(128, 6);
    // A ref CLAIMING to be huge routes to the large-file store; content size is irrelevant here.
    const largeRef: WeightFileRef = { ...refFor(small, 'big.onnx'), bytes: OPFS_THRESHOLD_BYTES + 1 };

    // Digest still verifies (routing must not affect verification)…
    const blob = await store.put(largeRef, streamOf(small));
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(small);
    // …and the same ref (same routing decision) finds it again.
    expect(await store.has(largeRef)).toBe(true);
    expect(await store.get(largeRef)).toBeDefined();
  });

  it('still enforces verification when degraded', async () => {
    const store = createWeightStore();
    const bytes = content(32, 7);
    const ref: WeightFileRef = { ...refFor(bytes), sha256: '0'.repeat(64) };
    await expect(store.put(ref, streamOf(bytes))).rejects.toThrow(WeightVerifyError);
    expect(await store.has(ref)).toBe(false);
  });
});
