/**
 * Device + quantization resolution: turns the user's request plus the two
 * capability probes into the concrete device and the ordered quantization
 * preference list handed to the weight loader (which picks the first entry
 * available for ALL requested roles).
 *
 * Pure and synchronous — probes are injected, so the normative table below
 * is unit-testable without a browser.
 */
import type { WasmProbeResult } from '../backend/wasm-backend.js';
import type { WebGpuProbeResult } from '../backend/webgpu-backend.js';
import { UnsupportedDeviceError } from '../errors.js';
import type { Quant } from '../weights/manifest.js';

/** What the user asked for (from `SegmenterConfig.device` / `.quant`). */
export interface DeviceRequest {
  device: 'webgpu' | 'wasm' | 'auto';
  quant: 'auto' | 'fp16' | 'int8' | 'q4f16';
}

/** The resolved device plus the ordered quantization preference for it. */
export interface DeviceResolution {
  device: 'webgpu' | 'wasm';
  /** Ordered best-first; the weight loader picks the first entry available for all roles. */
  quantPreference: readonly Quant[];
}

/**
 * Resolve the execution device and quantization preference (normative table
 * in docs/m1-internal-contracts.md §2.3).
 *
 * Device: `'auto'` prefers webgpu when granted, falls back to wasm, and
 * throws when neither exists; an explicit device that its probe cannot
 * satisfy throws. Quant: `'auto'` on webgpu+f16 →
 * `['q4f16','fp16','fp32','int8']`; webgpu without f16 → `['fp32','int8']`;
 * wasm → `['int8','fp32']`. Explicit `'fp16'`/`'q4f16'` require webgpu+f16
 * (no fallback); explicit `'int8'` runs anywhere.
 *
 * @throws UnsupportedDeviceError per the table (unavailable device, or an
 * explicit quant the resolved device cannot run).
 */
export function resolveDevice(
  request: DeviceRequest,
  probes: { webgpu: WebGpuProbeResult; wasm: WasmProbeResult },
): DeviceResolution {
  let device: 'webgpu' | 'wasm';
  if (request.device === 'webgpu') {
    if (!probes.webgpu.webgpu) {
      throw new UnsupportedDeviceError(
        "device 'webgpu' was requested but WebGPU is unavailable (no adapter granted)",
      );
    }
    device = 'webgpu';
  } else if (request.device === 'wasm') {
    if (!probes.wasm.wasm) {
      throw new UnsupportedDeviceError(
        "device 'wasm' was requested but WebAssembly is unavailable",
      );
    }
    device = 'wasm';
  } else if (probes.webgpu.webgpu) {
    device = 'webgpu';
  } else if (probes.wasm.wasm) {
    device = 'wasm';
  } else {
    throw new UnsupportedDeviceError(
      'no usable execution device: WebGPU is unavailable and WebAssembly is missing',
    );
  }

  const f16 = device === 'webgpu' && probes.webgpu.f16;

  switch (request.quant) {
    case 'auto': {
      if (device === 'webgpu') {
        return {
          device,
          quantPreference: f16 ? ['q4f16', 'fp16', 'fp32', 'int8'] : ['fp32', 'int8'],
        };
      }
      return { device, quantPreference: ['int8', 'fp32'] };
    }
    case 'int8':
      return { device, quantPreference: ['int8'] };
    case 'fp16':
    case 'q4f16': {
      if (!f16) {
        throw new UnsupportedDeviceError(
          `quant '${request.quant}' requires WebGPU with the 'shader-f16' feature; resolved device '${device}'${
            device === 'webgpu' ? ' has no f16 support' : ' cannot run f16 graphs'
          }`,
        );
      }
      return { device, quantPreference: [request.quant] };
    }
  }
}
