import { describe, expect, it } from 'vitest';
import type { WasmProbeResult } from '../backend/wasm-backend.js';
import type { WebGpuProbeResult } from '../backend/webgpu-backend.js';
import { UnsupportedDeviceError } from '../errors.js';
import { resolveDevice, type DeviceRequest } from './resolve-device.js';

function gpuProbe(overrides: Partial<WebGpuProbeResult> = {}): WebGpuProbeResult {
  return {
    webgpu: true,
    f16: true,
    crossOriginIsolated: false,
    recommendedDevice: 'webgpu',
    ...overrides,
  };
}

function wasmProbe(overrides: Partial<WasmProbeResult> = {}): WasmProbeResult {
  return {
    wasm: true,
    threads: false,
    crossOriginIsolated: false,
    recommendedDevice: 'wasm',
    ...overrides,
  };
}

/** Full-capability environment: webgpu granted with f16, wasm present. */
const full = { webgpu: gpuProbe(), wasm: wasmProbe() };
/** webgpu granted but no shader-f16. */
const noF16 = { webgpu: gpuProbe({ f16: false }), wasm: wasmProbe() };
/** No webgpu at all, wasm present. */
const wasmOnly = {
  webgpu: gpuProbe({ webgpu: false, f16: false, recommendedDevice: 'wasm' }),
  wasm: wasmProbe(),
};
/** Nothing usable. */
const nothing = {
  webgpu: gpuProbe({ webgpu: false, f16: false, recommendedDevice: 'wasm' }),
  wasm: wasmProbe({ wasm: false }),
};

const req = (device: DeviceRequest['device'], quant: DeviceRequest['quant']): DeviceRequest => ({
  device,
  quant,
});

describe('resolveDevice — device table', () => {
  it("auto → 'webgpu' when the adapter is granted", () => {
    expect(resolveDevice(req('auto', 'auto'), full).device).toBe('webgpu');
  });

  it("auto → 'wasm' when webgpu is unavailable but wasm is ok", () => {
    expect(resolveDevice(req('auto', 'auto'), wasmOnly).device).toBe('wasm');
  });

  it('auto → UnsupportedDeviceError when neither device exists', () => {
    expect(() => resolveDevice(req('auto', 'auto'), nothing)).toThrow(UnsupportedDeviceError);
  });

  it("explicit 'webgpu' → UnsupportedDeviceError when not granted", () => {
    expect(() => resolveDevice(req('webgpu', 'auto'), wasmOnly)).toThrow(UnsupportedDeviceError);
  });

  it("explicit 'webgpu' resolves when granted", () => {
    expect(resolveDevice(req('webgpu', 'auto'), full).device).toBe('webgpu');
  });

  it("explicit 'wasm' resolves even when webgpu is also available", () => {
    expect(resolveDevice(req('wasm', 'auto'), full).device).toBe('wasm');
  });

  it("explicit 'wasm' → UnsupportedDeviceError when WebAssembly is missing", () => {
    expect(() => resolveDevice(req('wasm', 'auto'), nothing)).toThrow(UnsupportedDeviceError);
  });
});

describe('resolveDevice — quant preference table', () => {
  it('auto quant on webgpu + f16 → q4f16 > fp16 > fp32 > int8', () => {
    expect(resolveDevice(req('auto', 'auto'), full).quantPreference).toEqual([
      'q4f16',
      'fp16',
      'fp32',
      'int8',
    ]);
  });

  it('auto quant on webgpu without f16 → fp32 > int8', () => {
    expect(resolveDevice(req('auto', 'auto'), noF16).quantPreference).toEqual(['fp32', 'int8']);
  });

  it('auto quant on wasm → int8 > fp32', () => {
    expect(resolveDevice(req('wasm', 'auto'), full).quantPreference).toEqual(['int8', 'fp32']);
    expect(resolveDevice(req('auto', 'auto'), wasmOnly).quantPreference).toEqual(['int8', 'fp32']);
  });

  it("explicit 'fp16' on webgpu + f16 → exactly ['fp16'] (no fallback)", () => {
    expect(resolveDevice(req('auto', 'fp16'), full)).toEqual({
      device: 'webgpu',
      quantPreference: ['fp16'],
    });
  });

  it("explicit 'q4f16' on webgpu + f16 → exactly ['q4f16'] (no fallback)", () => {
    expect(resolveDevice(req('webgpu', 'q4f16'), full)).toEqual({
      device: 'webgpu',
      quantPreference: ['q4f16'],
    });
  });

  it("explicit 'fp16'/'q4f16' on webgpu without f16 → UnsupportedDeviceError", () => {
    expect(() => resolveDevice(req('auto', 'fp16'), noF16)).toThrow(UnsupportedDeviceError);
    expect(() => resolveDevice(req('webgpu', 'q4f16'), noF16)).toThrow(UnsupportedDeviceError);
  });

  it("explicit 'fp16'/'q4f16' on a wasm resolution → UnsupportedDeviceError", () => {
    // Forced wasm even though the gpu could do f16 — the resolved device decides.
    expect(() => resolveDevice(req('wasm', 'fp16'), full)).toThrow(UnsupportedDeviceError);
    expect(() => resolveDevice(req('wasm', 'q4f16'), full)).toThrow(UnsupportedDeviceError);
    // auto device that degrades to wasm cannot satisfy f16 quants either.
    expect(() => resolveDevice(req('auto', 'fp16'), wasmOnly)).toThrow(UnsupportedDeviceError);
    expect(() => resolveDevice(req('auto', 'q4f16'), wasmOnly)).toThrow(UnsupportedDeviceError);
  });

  it("explicit 'int8' works on any device → exactly ['int8']", () => {
    expect(resolveDevice(req('webgpu', 'int8'), full)).toEqual({
      device: 'webgpu',
      quantPreference: ['int8'],
    });
    expect(resolveDevice(req('wasm', 'int8'), full)).toEqual({
      device: 'wasm',
      quantPreference: ['int8'],
    });
    expect(resolveDevice(req('auto', 'int8'), wasmOnly)).toEqual({
      device: 'wasm',
      quantPreference: ['int8'],
    });
  });

  it('incompatible explicit quant beats device availability: throws before returning', () => {
    // Device row would resolve fine; the quant row is what rejects.
    expect(() => resolveDevice(req('auto', 'q4f16'), noF16)).toThrow(/shader-f16/);
  });
});
