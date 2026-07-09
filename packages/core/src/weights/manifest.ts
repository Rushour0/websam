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

/**
 * Graph roles websam knows about.
 *
 * - `'visionEncoder' | 'promptDecoder'` — the M1 image path.
 * - `'videoEncoder' | 'memoryAttention' | 'maskDecoderVideo' | 'memoryEncoder'`
 *   — the M2 EdgeTAM video (memory-bank tracking) path.
 */
export type KnownGraphRole =
  | 'visionEncoder'
  | 'promptDecoder'
  | 'videoEncoder'
  | 'memoryAttention'
  | 'maskDecoderVideo'
  | 'memoryEncoder';

/**
 * Graph roles. Open union: future roles arrive without a schema bump; the
 * four video roles above are the concrete M2 additions.
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

/**
 * Memory-loop constants for the video (EdgeTAM) path — pinned by
 * `tools/export` (mirrors `websam_export.spec.ExportSpec`) and consumed by the
 * worker memory bank / arch strategy. Every value is a MANIFEST VALUE, never a
 * TS constant. EdgeTAM values below are pinned by the M2 export spike
 * (`tools/export/spikes/m2-edgetam/FINDINGS.md`): the 2D Spatial Perceiver
 * compresses each memory frame to 512 latent tokens (256 1D + 256 2D), keeping
 * 1 conditioning + 6 recent maps, so `kvLen = 7 * 512 + 64 = 3648`.
 */
export interface VideoManifestSection {
  /** Conditioning frames whose maps are always retained. EdgeTAM: 1. */
  maxCondFrames: number;
  /** Sliding window of most-recent tracked-frame maps. EdgeTAM: 6. */
  numRecent: number;
  /** KV tokens contributed per memory map (perceiver latents). EdgeTAM: 512. */
  tokensPerMemoryMap: number;
  /** KV tokens contributed by the projected object-pointer bank. EdgeTAM: 64. */
  ptrTokens: number;
  /** Object-pointer vectors kept before projection to `ptrTokens` tokens. EdgeTAM: 16. */
  maxObjectPointers: number;
  /**
   * Frozen memory-attention KV length the ONNX graph is exported with.
   * Parse-validated: `== (maxCondFrames + numRecent) * tokensPerMemoryMap + ptrTokens`.
   * EdgeTAM: 3648.
   */
  kvLen: number;
  /** Memory-feature channel width (per token). EdgeTAM: 64. */
  memDim: number;
  /** Object-pointer vector width. EdgeTAM: 256. */
  embedDim: number;
  /** Spatial memory grid edge (`imageSize / patch stride`). EdgeTAM: 64. */
  gridSize: number;
  /** Memory graphs exported with a symbolic leading batch dim (objects = batch). */
  multiObjectBatch: boolean;
  /** How the init (no-memory) frame is conditioned. */
  initPath: 'noMemFlag' | 'noMemGraph';
  /** How temporal-position embeddings reach the graph. */
  tposDelivery: 'indices' | 'precombined';
  /** `object_score_logits` threshold below which the object counts as occluded. */
  occlusionThreshold: number;
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
  /** Present iff the tier ships video graphs (the four video roles). */
  video?: VideoManifestSection;
}

const QUANTS: readonly Quant[] = ['fp32', 'fp16', 'int8', 'q4f16'];
const DTYPES: readonly DType[] = ['float32', 'float16', 'int64', 'uint8', 'int32', 'bool'];
const TRANSFORM_MODES: readonly TransformMode[] = ['square-stretch', 'letterbox'];
const SHA256_HEX = /^[0-9a-f]{64}$/;
const INIT_PATHS: readonly VideoManifestSection['initPath'][] = ['noMemFlag', 'noMemGraph'];
const TPOS_DELIVERIES: readonly VideoManifestSection['tposDelivery'][] = ['indices', 'precombined'];
/** Roles a `video` manifest section requires to be present in `graphs`. */
const VIDEO_ROLES: readonly KnownGraphRole[] = [
  'videoEncoder',
  'memoryAttention',
  'maskDecoderVideo',
  'memoryEncoder',
];

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

function parseVideoSection(
  value: unknown,
  graphs: Partial<Record<GraphRole, GraphManifestEntry>>,
  where: string,
): VideoManifestSection {
  if (!isRecord(value)) fail(where, 'must be an object');
  const int = (key: string): number => {
    if (!isPositiveInteger(value[key])) fail(`${where}.${key}`, 'must be a positive integer');
    return value[key] as number;
  };
  const maxCondFrames = int('maxCondFrames');
  const numRecent = int('numRecent');
  const tokensPerMemoryMap = int('tokensPerMemoryMap');
  const ptrTokens = int('ptrTokens');
  const maxObjectPointers = int('maxObjectPointers');
  const kvLen = int('kvLen');
  const memDim = int('memDim');
  const embedDim = int('embedDim');
  const gridSize = int('gridSize');

  const expectedKvLen = (maxCondFrames + numRecent) * tokensPerMemoryMap + ptrTokens;
  if (kvLen !== expectedKvLen) {
    fail(
      `${where}.kvLen`,
      `must equal (maxCondFrames + numRecent) * tokensPerMemoryMap + ptrTokens = ${expectedKvLen}, got ${kvLen}`,
    );
  }

  const multiObjectBatch = value['multiObjectBatch'];
  if (typeof multiObjectBatch !== 'boolean') fail(`${where}.multiObjectBatch`, 'must be a boolean');

  const initPath = value['initPath'];
  if (typeof initPath !== 'string' || !(INIT_PATHS as readonly string[]).includes(initPath)) {
    fail(`${where}.initPath`, `must be one of ${INIT_PATHS.join(', ')}`);
  }

  const tposDelivery = value['tposDelivery'];
  if (
    typeof tposDelivery !== 'string' ||
    !(TPOS_DELIVERIES as readonly string[]).includes(tposDelivery)
  ) {
    fail(`${where}.tposDelivery`, `must be one of ${TPOS_DELIVERIES.join(', ')}`);
  }

  const occlusionThreshold = value['occlusionThreshold'];
  if (typeof occlusionThreshold !== 'number' || !Number.isFinite(occlusionThreshold)) {
    fail(`${where}.occlusionThreshold`, 'must be a finite number');
  }

  for (const role of VIDEO_ROLES) {
    if (!graphs[role]) {
      fail(where, `requires graph role '${role}' to be present in graphs`);
    }
  }

  return {
    maxCondFrames,
    numRecent,
    tokensPerMemoryMap,
    ptrTokens,
    maxObjectPointers,
    kvLen,
    memDim,
    embedDim,
    gridSize,
    multiObjectBatch,
    initPath: initPath as VideoManifestSection['initPath'],
    tposDelivery: tposDelivery as VideoManifestSection['tposDelivery'],
    occlusionThreshold,
  };
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

  const manifest: ModelManifest = {
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

  if (json['video'] !== undefined) {
    manifest.video = parseVideoSection(json['video'], graphs, 'video');
  }

  return manifest;
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
