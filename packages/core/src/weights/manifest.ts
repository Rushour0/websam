/**
 * Model manifest: the schema describing a model tier's weight files, graph
 * IO contracts, and preprocessing constants.
 *
 * Everything S0-dependent is a MANIFEST VALUE, never a TS constant: the
 * transform mode lives in `preprocess.mode`, tensor names/shapes live in
 * `graphs.*.inputs/outputs` keyed by SEMANTIC name — runtime code binds
 * tensors via `entry.inputs.points.name`, never a hardcoded ONNX name (see
 * docs/m1-internal-contracts.md §1.1 and tools/export/spikes/s0/FINDINGS.md).
 */

import type { DType } from '../backend/backend.js';
import { WeightVerifyError } from '../errors.js';
import type { TransformMode } from '../coords.js';

/** Weight quantization variants a manifest may carry. */
export type Quant = 'fp32' | 'fp16' | 'int8' | 'q4f16';

/** Graph roles the M1 image path knows about. */
export type KnownGraphRole = 'visionEncoder' | 'promptDecoder';

/**
 * Graph roles. Open union: video roles (`'memoryAttention'`, …) arrive in
 * M2/M3 without a schema bump.
 */
export type GraphRole = KnownGraphRole | (string & {});

/** One tensor of a graph's IO contract. `name` is the literal ONNX tensor name. */
export interface TensorSpec {
  name: string;
  dtype: DType;
  /** number = static dim, string = symbolic dim (e.g. 'batch_size'). */
  shape: readonly (number | string)[];
}

/** One downloadable weight file of a graph. */
export interface WeightFileRef {
  /** Relative to the manifest URL (or `modelBaseUrl` override). Immutable, revisioned. */
  path: string;
  /** Lowercase hex SHA-256 of the file bytes. Also the cache identity (content-addressed). */
  sha256: string;
  bytes: number;
}

/** Files and IO contract of one graph role. */
export interface GraphManifestEntry {
  /** Available quantizations of this graph. */
  files: Partial<Record<Quant, WeightFileRef>>;
  /**
   * IO contract keyed by SEMANTIC name — runtime code binds tensors via
   * `entry.inputs.points.name`, NEVER a hardcoded ONNX name. Required
   * semantic keys per role are listed in docs/m1-internal-contracts.md §1.1.1.
   */
  inputs: Record<string, TensorSpec>;
  outputs: Record<string, TensorSpec>;
}

/** The full model manifest (`schemaVersion: 1`). */
export interface ModelManifest {
  schemaVersion: 1;
  /** Must equal the registry ModelSpec.id this manifest serves. */
  tier: string;
  opset: number;
  graphs: Partial<Record<GraphRole, GraphManifestEntry>>;
  toolchain: { exporter: string; pytorch?: string; onnx?: string; transformers?: string };
  /** Values pinned by the export pipeline, consumed by the worker (never hardcoded in TS). */
  preprocess: {
    /** S0-pinned: 'square-stretch' for sam3-tracker (preprocessor_config: size 1008×1008, do_pad null). */
    mode: TransformMode;
    /** 1008 for sam3-tracker. */
    inputSize: number;
    mean: [number, number, number];
    std: [number, number, number];
    /** Decoder low-res logit grid side (mask_size), e.g. 288. */
    maskSize: number;
  };
}

const QUANTS: readonly Quant[] = ['fp32', 'fp16', 'int8', 'q4f16'];
const DTYPES: readonly DType[] = ['float32', 'float16', 'int64', 'uint8', 'int32', 'bool'];
const TRANSFORM_MODES: readonly TransformMode[] = ['square-stretch', 'letterbox'];
const SHA256_HEX = /^[0-9a-f]{64}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

/** Validation failure carrying a JSON-path-ish location for the error message. */
class ManifestFieldError extends Error {}

function fail(where: string, detail: string): never {
  throw new ManifestFieldError(`${where}: ${detail}`);
}

function parseFileRef(value: unknown, where: string): WeightFileRef {
  if (!isRecord(value)) fail(where, 'must be an object');
  if (!isNonEmptyString(value['path'])) fail(`${where}.path`, 'must be a non-empty string');
  const sha256 = value['sha256'];
  if (typeof sha256 !== 'string' || !SHA256_HEX.test(sha256)) {
    fail(`${where}.sha256`, 'must be 64 lowercase hex characters');
  }
  if (!isPositiveInteger(value['bytes'])) fail(`${where}.bytes`, 'must be a positive integer');
  return { path: value['path'], sha256, bytes: value['bytes'] };
}

function parseTensorSpec(value: unknown, where: string): TensorSpec {
  if (!isRecord(value)) fail(where, 'must be an object');
  if (!isNonEmptyString(value['name'])) fail(`${where}.name`, 'must be a non-empty string');
  const dtype = value['dtype'];
  if (typeof dtype !== 'string' || !(DTYPES as readonly string[]).includes(dtype)) {
    fail(`${where}.dtype`, `must be one of ${DTYPES.join(', ')}`);
  }
  const shape = value['shape'];
  if (!Array.isArray(shape)) fail(`${where}.shape`, 'must be an array');
  for (const [i, dim] of shape.entries()) {
    const staticDim = typeof dim === 'number' && Number.isInteger(dim) && dim >= 0;
    const symbolicDim = isNonEmptyString(dim);
    if (!staticDim && !symbolicDim) {
      fail(`${where}.shape[${i}]`, 'must be a non-negative integer or a symbolic dim name');
    }
  }
  return { name: value['name'], dtype: dtype as DType, shape: [...shape] as (number | string)[] };
}

function parseTensorMap(value: unknown, where: string): Record<string, TensorSpec> {
  if (!isRecord(value)) fail(where, 'must be an object');
  const out: Record<string, TensorSpec> = {};
  for (const [key, spec] of Object.entries(value)) {
    out[key] = parseTensorSpec(spec, `${where}.${key}`);
  }
  return out;
}

function parseGraphEntry(value: unknown, where: string): GraphManifestEntry {
  if (!isRecord(value)) fail(where, 'must be an object');
  const files = value['files'];
  if (!isRecord(files)) fail(`${where}.files`, 'must be an object');
  const parsedFiles: Partial<Record<Quant, WeightFileRef>> = {};
  for (const [quant, ref] of Object.entries(files)) {
    if (!(QUANTS as readonly string[]).includes(quant)) {
      fail(`${where}.files.${quant}`, `unknown quant (expected one of ${QUANTS.join(', ')})`);
    }
    parsedFiles[quant as Quant] = parseFileRef(ref, `${where}.files.${quant}`);
  }
  if (Object.keys(parsedFiles).length === 0) {
    fail(`${where}.files`, 'must list at least one quantization');
  }
  return {
    files: parsedFiles,
    inputs: parseTensorMap(value['inputs'], `${where}.inputs`),
    outputs: parseTensorMap(value['outputs'], `${where}.outputs`),
  };
}

function parseTriple(value: unknown, where: string): [number, number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((v) => typeof v === 'number' && Number.isFinite(v))
  ) {
    fail(where, 'must be an array of exactly 3 finite numbers');
  }
  return [value[0] as number, value[1] as number, value[2] as number];
}

function parseManifestFields(json: unknown): ModelManifest {
  if (!isRecord(json)) fail('manifest', 'must be a JSON object');
  if (json['schemaVersion'] !== 1) {
    fail('schemaVersion', `must be 1, got ${JSON.stringify(json['schemaVersion'])}`);
  }
  if (!isNonEmptyString(json['tier'])) fail('tier', 'must be a non-empty string');
  if (!isPositiveInteger(json['opset'])) fail('opset', 'must be a positive integer');

  const graphsJson = json['graphs'];
  if (!isRecord(graphsJson)) fail('graphs', 'must be an object');
  const graphs: Partial<Record<GraphRole, GraphManifestEntry>> = {};
  for (const [role, entry] of Object.entries(graphsJson)) {
    graphs[role] = parseGraphEntry(entry, `graphs.${role}`);
  }

  const toolchainJson = json['toolchain'];
  if (!isRecord(toolchainJson)) fail('toolchain', 'must be an object');
  if (!isNonEmptyString(toolchainJson['exporter'])) {
    fail('toolchain.exporter', 'must be a non-empty string');
  }
  const toolchain: ModelManifest['toolchain'] = { exporter: toolchainJson['exporter'] };
  for (const key of ['pytorch', 'onnx', 'transformers'] as const) {
    const v = toolchainJson[key];
    if (v !== undefined) {
      if (!isNonEmptyString(v)) fail(`toolchain.${key}`, 'must be a non-empty string when present');
      toolchain[key] = v;
    }
  }

  const pre = json['preprocess'];
  if (!isRecord(pre)) fail('preprocess', 'must be an object');
  const mode = pre['mode'];
  if (typeof mode !== 'string' || !(TRANSFORM_MODES as readonly string[]).includes(mode)) {
    fail('preprocess.mode', `must be one of ${TRANSFORM_MODES.join(', ')}`);
  }
  if (!isPositiveInteger(pre['inputSize'])) fail('preprocess.inputSize', 'must be a positive integer');
  if (!isPositiveInteger(pre['maskSize'])) fail('preprocess.maskSize', 'must be a positive integer');

  return {
    schemaVersion: 1,
    tier: json['tier'],
    opset: json['opset'],
    graphs,
    toolchain,
    preprocess: {
      mode: mode as TransformMode,
      inputSize: pre['inputSize'],
      mean: parseTriple(pre['mean'], 'preprocess.mean'),
      std: parseTriple(pre['std'], 'preprocess.std'),
      maskSize: pre['maskSize'],
    },
  };
}

/**
 * Validate untrusted manifest JSON and return a typed, defensive copy
 * (unknown top-level fields are dropped; nothing references the input).
 *
 * @param json - Parsed JSON of unknown shape (e.g. `await res.json()`).
 * @param sourceUrl - Where the JSON came from, for error messages.
 * @throws WeightVerifyError — bad/missing fields or wrong `schemaVersion`.
 */
export function parseModelManifest(json: unknown, sourceUrl: string): ModelManifest {
  try {
    return parseManifestFields(json);
  } catch (err) {
    if (err instanceof ManifestFieldError) {
      throw new WeightVerifyError(`Invalid model manifest at ${sourceUrl} — ${err.message}`);
    }
    throw err;
  }
}
