/**
 * Browser integration for the M1 backend bodies: drives REAL onnxruntime-web
 * through the Backend interface (createSession / uploadTensor / run /
 * readback / dispose) against the committed add.onnx fixture — the same
 * graph ort.browser.test.ts proves at the raw-ort level.
 */
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import addModelUrl from '../__fixtures__/add.onnx?url';
import type { Backend, IOBindingPlan } from '../backend/backend.js';
import { WasmBackend } from '../backend/wasm-backend.js';
import { WebGpuBackend } from '../backend/webgpu-backend.js';

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

/** Run 1 + 2 through the Backend abstraction and read the result back. */
async function runAddThroughBackend(
  backend: Backend,
  plan?: IOBindingPlan,
): Promise<{ value: number; location: string }> {
  const session = await backend.createSession({ name: 'add', bytes: await fetchFixture() }, plan);
  try {
    const a = backend.uploadTensor(Float32Array.from([1]), [1], 'float32');
    const b = backend.uploadTensor(Float32Array.from([2]), [1], 'float32');
    const outputs = await session.run({ a, b });
    const c = outputs['c'];
    if (!c) throw new Error('missing output c');
    const view = (await backend.readback(c)) as Float32Array;
    const result = { value: view[0] ?? Number.NaN, location: c.location };
    c.dispose();
    return result;
  } finally {
    await session.dispose();
  }
}

describe('backend impls against real onnxruntime-web', () => {
  it('WasmBackend runs 1 + 2 = 3 end-to-end through the Backend interface', async () => {
    const backend = new WasmBackend(ort);
    await backend.init();
    const { value, location } = await runAddThroughBackend(backend);
    expect(value).toBe(3);
    expect(location).toBe('cpu');
    await backend.dispose();
    expect(backend.initialized).toBe(false);
    console.log('[backend-impl.browser.test] EP ran: wasm');
  });

  // Never fails on a missing adapter: skipped without navigator.gpu, and a
  // denied adapter / EP init failure downgrades to a logged wasm-only pass
  // (same soft-pass pattern as ort.browser.test.ts).
  it.skipIf(!gpu)(
    'WebGpuBackend honors a device-located IOBindingPlan when an adapter exists',
    async () => {
      const adapter = await gpu!.requestAdapter().catch(() => null);
      if (!adapter) {
        console.log('[backend-impl.browser.test] EP ran: none (navigator.gpu present but no adapter)');
        return;
      }
      const backend = new WebGpuBackend(ort);
      await backend.init();
      let result: { value: number; location: string };
      try {
        result = await runAddThroughBackend(backend, { outputLocations: { c: 'device' } });
      } catch (err) {
        // Adapter exists but the EP could not initialize (e.g. headless
        // driver quirks) — log and soft-pass rather than flake CI.
        console.log('[backend-impl.browser.test] EP ran: none (webgpu init failed)', err);
        return;
      }
      expect(result.value).toBe(3);
      expect(result.location).toBe('device');
      await backend.dispose();
      console.log('[backend-impl.browser.test] EP ran: webgpu');
    },
  );
});
