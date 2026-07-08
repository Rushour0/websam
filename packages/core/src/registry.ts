/**
 * Model registry: the catalog of SAM-family models websam knows how to load.
 *
 * Model ids name a CAPABILITY TIER (e.g. `'sam3-560'` = the 560px SAM3
 * tier), never a quantization — quant (`fp16`/`int8`/`q4f16`) is selected at
 * load time per device from the model's manifest, so the same id serves
 * every device.
 */

import { InvalidStateError } from './errors.js';

/** Which backends a model tier can realistically run on. */
export interface ModelDeviceSupport {
  /** Runs on the WebGPU backend. */
  webgpu: boolean;
  /** Runs on the WASM (CPU) backend at usable speed. */
  wasm: boolean;
}

/** A registered model tier. */
export interface ModelSpec {
  /** Capability-tier id, e.g. `'edgetam'`, `'sam3-tracker'`. NO quant suffix — quant is chosen at load time. */
  id: string;
  /** Human-readable name for pickers and logs. */
  displayName: string;
  /** Model architecture family, which determines the graph set and pre/post pipeline. */
  arch: 'sam3-tracker' | 'edgetam';
  /** Square model input side length in pixels. */
  inputSize: number;
  /** Whether the tier ships video (memory-bank tracking) graphs. */
  supportsVideo: boolean;
  /** License the weights ship under. */
  license: 'apache-2.0' | 'sam-license';
  /**
   * True when the user must explicitly accept the license before weights
   * download (SAM-licensed models: pass `acceptLicense: 'sam'` in
   * SegmenterConfig). Omitted/false for permissive licenses.
   */
  requiresLicenseAcceptance?: boolean;
  /** URL of the model manifest (files, digests, quants per device). */
  manifestUrl: string;
  /** Per-backend support matrix. */
  devices: ModelDeviceSupport;
}

const registry = new Map<string, ModelSpec>();

/** Deep-enough copy so callers can never mutate registry state. */
function cloneSpec(spec: ModelSpec): ModelSpec {
  return { ...spec, devices: { ...spec.devices } };
}

/**
 * Register a model tier. Throws {@link InvalidStateError} if the id is
 * already registered — ids are stable public identifiers and must not be
 * silently redefined.
 */
export function registerModel(spec: ModelSpec): void {
  if (registry.has(spec.id)) {
    throw new InvalidStateError(`registerModel: model id '${spec.id}' is already registered`);
  }
  registry.set(spec.id, cloneSpec(spec));
}

/** Look up a registered model tier by id; `undefined` when unknown. */
export function getModel(id: string): ModelSpec | undefined {
  const spec = registry.get(id);
  return spec ? cloneSpec(spec) : undefined;
}

/** All registered model tiers, in registration order. */
export function listModels(): ModelSpec[] {
  return [...registry.values()].map(cloneSpec);
}

// ---------------------------------------------------------------------------
// Built-in tiers. manifestUrl values are placeholders until the model CDN
// lands (M1); the shapes and license facts are real.
// ---------------------------------------------------------------------------

registerModel({
  id: 'edgetam',
  displayName: 'EdgeTAM',
  arch: 'edgetam',
  inputSize: 1024,
  supportsVideo: true,
  license: 'apache-2.0',
  manifestUrl: 'https://models.websam.dev/edgetam/manifest.json',
  devices: { webgpu: true, wasm: true },
});

registerModel({
  id: 'sam3-tracker',
  displayName: 'SAM 3 Tracker',
  arch: 'sam3-tracker',
  inputSize: 1008,
  supportsVideo: true,
  license: 'sam-license',
  requiresLicenseAcceptance: true,
  manifestUrl: 'https://models.websam.dev/sam3-tracker/manifest.json',
  devices: { webgpu: true, wasm: false },
});

registerModel({
  id: 'sam3-560',
  displayName: 'SAM 3 (560px)',
  arch: 'sam3-tracker',
  inputSize: 560,
  supportsVideo: true,
  license: 'sam-license',
  requiresLicenseAcceptance: true,
  manifestUrl: 'https://models.websam.dev/sam3-560/manifest.json',
  devices: { webgpu: true, wasm: true },
});
