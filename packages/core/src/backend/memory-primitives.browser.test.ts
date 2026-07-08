/**
 * Browser integration for the M2 memory primitives: real onnxruntime-web
 * driving `allocTensor('device')` + `copyRegion` round-trips (verified via
 * `readback` against a cpu reference) and the `debugStats()` census, on both
 * browser backends. webgpu uses the same soft-pass-on-no-adapter pattern as
 * ort.browser.test.ts (SwiftShader lane).
 */
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import addModelUrl from '../__fixtures__/add.onnx?url';
import type { DeviceTensor } from './backend.js';
import { WasmBackend } from './wasm-backend.js';
import { WebGpuBackend } from './webgpu-backend.js';
import { InvalidStateError } from '../errors.js';

// Deterministic single-thread WASM: no COOP/COEP requirement in the test server.
ort.env.wasm.numThreads = 1;

// Structural view of WebGPU (lib.dom has no navigator.gpu typing).
const gpu = (
  globalThis.navigator as unknown as
    | { gpu?: { requestAdapter(): Promise<object | null> } }
    | undefined
)?.gpu;

async function fetchFixture(): Promise<Uint8Array> {
  const res = await fetch(addModelUrl);
  if (!res.ok) throw new Error(`failed to fetch fixture: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

describe('M2 memory primitives against real onnxruntime-web', () => {
  it('WasmBackend: copyRegion slot offsets are correct on real cpu tensors', async () => {
    const backend = new WasmBackend(ort);
    await backend.init();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });

    // 'device' degrades to cpu on wasm — the documented equivalence.
    const ring = backend.allocTensor([3, 2, 2], 'float32', 'device');
    expect(ring.location).toBe('cpu');

    // Byte-count-equal rule: a flat [4] src fills a [2, 2] slot.
    const src = backend.uploadTensor(Float32Array.from([1, 2, 3, 4]), [4], 'float32');
    backend.copyRegion(src, ring, 1);
    const overwrite = backend.uploadTensor(Float32Array.from([9, 8, 7, 6]), [2, 2], 'float32');
    backend.copyRegion(overwrite, ring, 2);

    const view = (await backend.readback(ring)) as Float32Array;
    expect(Array.from(view)).toEqual([0, 0, 0, 0, 1, 2, 3, 4, 9, 8, 7, 6]);

    // Census: ring 48 B + two [4]-element f32 srcs at 16 B each.
    expect(backend.debugStats()).toEqual({ liveTensors: 3, liveBytes: 80 });
    src.dispose();
    overwrite.dispose();
    ring.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });

    await backend.dispose();
    console.log('[memory-primitives.browser.test] EP ran: wasm');
  });

  it('WasmBackend: debugStats counts real run() outputs until disposed', async () => {
    const backend = new WasmBackend(ort);
    await backend.init();
    const session = await backend.createSession({ name: 'add', bytes: await fetchFixture() });
    const a = backend.uploadTensor(Float32Array.from([1]), [1], 'float32');
    const b = backend.uploadTensor(Float32Array.from([2]), [1], 'float32');

    const outputs = await session.run({ a, b });
    const c = outputs['c'] as DeviceTensor;
    expect(((await backend.readback(c)) as Float32Array)[0]).toBe(3);
    // a + b uploads (4 B each) + the run output c (4 B).
    expect(backend.debugStats()).toEqual({ liveTensors: 3, liveBytes: 12 });

    c.dispose();
    a.dispose();
    b.dispose();
    expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });

    await backend.dispose();
  });

  // Never fails on a missing adapter: skipped without navigator.gpu, and a
  // denied adapter / EP init failure downgrades to a logged wasm-only pass
  // (same soft-pass pattern as ort.browser.test.ts).
  it.skipIf(!gpu)(
    'WebGpuBackend: device alloc + copyRegion round-trip when an adapter exists',
    async () => {
      const adapter = await gpu!.requestAdapter().catch(() => null);
      if (!adapter) {
        console.log(
          '[memory-primitives.browser.test] EP ran: none (navigator.gpu present but no adapter)',
        );
        return;
      }
      const backend = new WebGpuBackend(ort);
      await backend.init();

      // ort owns the GPUDevice: it exists only after the first session, so
      // device allocation before that is an InvalidStateError by contract.
      expect(() => backend.allocTensor([4, 1], 'float32', 'device')).toThrow(InvalidStateError);

      let session;
      try {
        session = await backend.createSession(
          { name: 'add', bytes: await fetchFixture() },
          { outputLocations: { c: 'device' } },
        );
      } catch (err) {
        // Adapter exists but the EP could not initialize (e.g. headless
        // driver quirks) — log and soft-pass rather than flake CI.
        console.log('[memory-primitives.browser.test] EP ran: none (webgpu init failed)', err);
        return;
      }

      const baseline = backend.debugStats();
      const ring = backend.allocTensor([4, 1], 'float32', 'device');
      expect(ring.location).toBe('device');

      // Freshly alloc'd device memory reads back zeroed (separate tensor —
      // ort pins a handle to cpu after its one download).
      const zeroed = backend.allocTensor([2, 2], 'float32', 'device');
      expect(Array.from((await backend.readback(zeroed)) as Float32Array)).toEqual([0, 0, 0, 0]);
      zeroed.dispose();

      // Produce device-located sources with KNOWN values via real runs.
      const run = async (x: number, y: number): Promise<DeviceTensor> => {
        const a = backend.uploadTensor(Float32Array.from([x]), [1], 'float32');
        const b = backend.uploadTensor(Float32Array.from([y]), [1], 'float32');
        const outputs = await session.run({ a, b });
        a.dispose();
        b.dispose();
        return outputs['c'] as DeviceTensor;
      };
      const three = await run(1, 2); // device-located [1] = 3
      const eleven = await run(5, 6); // device-located [1] = 11
      expect(three.location).toBe('device');

      backend.copyRegion(three, ring, 2);
      backend.copyRegion(eleven, ring, 0);

      // A cpu-located operand on webgpu must be rejected (upload first).
      const cpuSrc = backend.uploadTensor(Float32Array.from([7]), [1], 'float32');
      expect(() => backend.copyRegion(cpuSrc, ring, 1)).toThrow(InvalidStateError);
      cpuSrc.dispose();

      // Slot-offset correctness against the cpu reference.
      const view = (await backend.readback(ring)) as Float32Array;
      expect(Array.from(view)).toEqual([11, 0, 3, 0]);

      three.dispose();
      eleven.dispose();
      ring.dispose();
      expect(backend.debugStats()).toEqual(baseline);

      await backend.dispose();
      expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
      console.log('[memory-primitives.browser.test] EP ran: webgpu');
    },
  );
});
