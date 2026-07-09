/**
 * Browser integration for the M2 memory-bank backend primitives
 * (backend-video §1.1): drives REAL onnxruntime-web through the Backend
 * interface for `allocTensor('device')` + `copyRegion` + `readback` +
 * `debugStats`.
 *
 * The WASM path always runs (deterministic, cpu). The WebGPU path is
 * soft-passed on a missing adapter / failed EP init (the same pattern as
 * backend-impl.browser.test.ts), and — because ort creates its WebGPU device
 * lazily during the first webgpu session — first compiles the committed
 * add.onnx fixture to materialize `ort.env.webgpu.device`, then round-trips a
 * ring copy and reads it back to assert slot-offset correctness against a cpu
 * reference.
 */
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import addModelUrl from '../__fixtures__/add.onnx?url';
import { OrtDeviceTensor } from '../runtime/ort-tensor.js';
import { WasmBackend } from './wasm-backend.js';
import { WebGpuBackend } from './webgpu-backend.js';

// Deterministic single-thread WASM: no COOP/COEP requirement in the test server.
ort.env.wasm.numThreads = 1;

// Structural view of WebGPU (lib.dom typings for navigator.gpu are not enabled).
const gpu = (
  globalThis.navigator as unknown as
    | { gpu?: { requestAdapter(): Promise<object | null> } }
    | undefined
)?.gpu;

/** Minimal structural view of the ort-owned GPUDevice queue we write through. */
interface GpuQueueLike {
  writeBuffer(buffer: unknown, offset: number, data: BufferSource): void;
}

async function fetchAddFixture(): Promise<Uint8Array> {
  const res = await fetch(addModelUrl);
  if (!res.ok) throw new Error(`failed to fetch fixture: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

describe('memory primitives against real onnxruntime-web — WASM', () => {
  it('allocTensor(device→cpu) + copyRegion round-trips a slot, and debugStats counts', async () => {
    const backend = new WasmBackend(ort);
    await backend.init();
    try {
      // Ring of 3 slots x 4 floats; on wasm this degrades to a cpu tensor.
      const ring = backend.allocTensor([3, 4], 'float32', 'device');
      expect(ring.location).toBe('cpu');
      const slot = backend.uploadTensor(Float32Array.from([10, 20, 30, 40]), [4], 'float32');
      expect(backend.debugStats()).toEqual({ liveTensors: 2, liveBytes: 3 * 4 * 4 + 4 * 4 });

      backend.copyRegion(slot, ring, 1);
      const view = (await backend.readback(ring)) as Float32Array;
      expect(Array.from(view)).toEqual([0, 0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0]);

      ring.dispose();
      slot.dispose();
      expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
    } finally {
      await backend.dispose();
    }
    console.log('[memory-primitives.browser.test] EP ran: wasm');
  });
});

describe('memory primitives against real onnxruntime-web — WebGPU', () => {
  // Never fails on a missing adapter: skipped without navigator.gpu, and a
  // denied adapter / EP init failure downgrades to a logged soft-pass.
  it.skipIf(!gpu)('allocTensor(device) + copyRegion is a real GPU-buffer ring copy', async () => {
    const adapter = await gpu!.requestAdapter().catch(() => null);
    if (!adapter) {
      console.log('[memory-primitives.browser.test] EP ran: none (no adapter)');
      return;
    }
    const backend = new WebGpuBackend(ort);
    await backend.init();
    let session;
    try {
      // Force ort to create its WebGPU device (exposed on ort.env.webgpu.device).
      session = await backend.createSession({ name: 'add', bytes: await fetchAddFixture() });
    } catch (err) {
      console.log('[memory-primitives.browser.test] EP ran: none (webgpu init failed)', err);
      await backend.dispose().catch(() => undefined);
      return;
    }

    try {
      const device = (ort.env.webgpu as unknown as { device: { queue: GpuQueueLike } }).device;

      const slotFloats = 4;
      const slots = 3;
      const ring = backend.allocTensor([slots, slotFloats], 'float32', 'device');
      expect(ring.location).toBe('device');

      // Fill a device-located source slot with known data via the ort device queue.
      const known = Float32Array.from([11, 22, 33, 44]);
      const src = backend.allocTensor([slotFloats], 'float32', 'device');
      const srcBuffer = (src as OrtDeviceTensor).ortTensor.gpuBuffer;
      device.queue.writeBuffer(srcBuffer, 0, known);

      expect(backend.debugStats()).toEqual({
        liveTensors: 2,
        liveBytes: (slots * slotFloats + slotFloats) * 4,
      });

      const targetSlot = 2;
      backend.copyRegion(src, ring, targetSlot);

      const view = (await backend.readback(ring)) as Float32Array;
      const expected = new Float32Array(slots * slotFloats);
      expected.set(known, targetSlot * slotFloats);
      expect(Array.from(view)).toEqual(Array.from(expected));

      ring.dispose();
      src.dispose();
      expect(backend.debugStats()).toEqual({ liveTensors: 0, liveBytes: 0 });
      console.log('[memory-primitives.browser.test] EP ran: webgpu');
    } finally {
      await session.dispose();
      await backend.dispose();
    }
  });
});
