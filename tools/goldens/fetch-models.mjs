/**
 * fetch-models.mjs — populate the M1 browser-gate model cache.
 *
 * Downloads the community SAM3-tracker ONNX graphs (q4f16) at the EXACT
 * revision the committed goldens were generated from
 * (`fixtures/golden-meta.json`), merges each graph's external `.onnx_data`
 * into a single self-contained `.onnx` file (websam's weight pipeline loads
 * one verified byte blob per graph — `InferenceSession.create(bytes)` cannot
 * chase external-data references), and emits schemaVersion-1 model manifests
 * (packages/core/src/weights/manifest.ts) with REAL sha256 + byte counts:
 *
 *   models-cache/vision_encoder_q4f16.onnx              (self-contained, ~297 MB)
 *   models-cache/prompt_encoder_mask_decoder_q4f16.onnx (self-contained, ~5 MB)
 *   models-cache/manifest.json          tier 'sam3-tracker'      (demo / self-hosting)
 *   models-cache/manifest-e2e.json      tier 'sam3-tracker-e2e'  (browser gate —
 *       packages/core/src/e2e/image-golden.browser.test.ts registers this tier)
 *
 * Everything under models-cache/ is gitignored and CI-cacheable.
 *
 * Requirements:
 *   - Node 22 (`. ~/.nvm/nvm.sh && nvm use 22`)
 *   - the tools/export Python venv with `onnx` installed (used ONLY for the
 *     external-data merge); override the interpreter with
 *     `WEBSAM_EXPORT_PYTHON=/path/to/python`.
 *
 * Usage:
 *   node tools/goldens/fetch-models.mjs          # idempotent; reuses verified files
 *   FORCE=1 node tools/goldens/fetch-models.mjs  # redo the merge + manifests
 *
 * Raw downloads are sha256-verified against the Hugging Face LFS oids for the
 * pinned revision; an existing transformers.js cache from `generate.mjs`
 * (models/<model_id>/onnx/) is reused when its digests match, so a machine
 * that already ran the golden generator downloads nothing.
 *
 * ⚠ S0 note (docs/m1-internal-contracts.md §7): tensor names/shapes and the
 * preprocess block below are pinned by tools/export/spikes/s0/FINDINGS.md
 * (square-stretch, 1008, mean=std=0.5, mask grid 288, opset 18). If the
 * export pipeline re-pins any of these, update THIS emitter — never the
 * packages/core sources.
 */

import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { access, copyFile, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const metaPath = join(here, 'fixtures', 'golden-meta.json');
const cacheDir = join(here, 'models-cache');
const rawDir = join(cacheDir, 'raw');
const transformersCacheDir = join(here, 'models'); // generate.mjs (transformers.js) cache
const defaultPython = resolve(here, '..', 'export', '.venv', 'bin', 'python');
const python = process.env.WEBSAM_EXPORT_PYTHON ?? defaultPython;
const force = process.env.FORCE === '1';

/** Tier ids the two emitted manifests serve (must equal the registry ModelSpec.id). */
const CANONICAL_TIER = 'sam3-tracker';
const E2E_TIER = 'sam3-tracker-e2e';

// ---------------------------------------------------------------------------
// Pinned source (from the committed golden metadata — single source of truth).
// ---------------------------------------------------------------------------

const meta = JSON.parse(await readFile(metaPath, 'utf8'));
const modelId = meta.model_id; // onnx-community/sam3-tracker-ONNX
const revision = meta.model_revision; // 429305c8a5b3de597243d919a07e4e6bdcd00ef7
const dtype = meta.dtype; // q4f16
if (!modelId || !revision || !dtype) {
  throw new Error(`golden-meta.json at ${metaPath} is missing model_id/model_revision/dtype`);
}

/** The two graphs of the M1 image path, plus their external-data companions. */
const GRAPHS = [
  { role: 'visionEncoder', base: `vision_encoder_${dtype}` },
  { role: 'promptDecoder', base: `prompt_encoder_mask_decoder_${dtype}` },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function sha256File(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function fmtMB(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** HF tree listing at the pinned revision → { name → { size, sha256 } }. */
async function fetchExpectedDigests() {
  const url = `https://huggingface.co/api/models/${modelId}/tree/${revision}/onnx`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Hugging Face tree listing failed: HTTP ${res.status} (${url})`);
  }
  const entries = await res.json();
  const out = new Map();
  for (const entry of entries) {
    const name = entry.path.replace(/^onnx\//, '');
    out.set(name, { size: entry.size, sha256: entry.lfs?.oid });
  }
  return out;
}

/** Ensure models-cache/raw/<name> exists with the expected sha256. */
async function ensureRawFile(name, expected) {
  const target = join(rawDir, name);
  if (await exists(target)) {
    const digest = await sha256File(target);
    if (digest === expected.sha256) {
      console.log(`  raw ${name}: cached (${fmtMB(expected.size)}, sha256 ok)`);
      return;
    }
    console.log(`  raw ${name}: cached copy has wrong digest, refetching`);
    await rm(target);
  }

  // Reuse the transformers.js cache from generate.mjs when digests match.
  const seeded = join(transformersCacheDir, modelId, 'onnx', name);
  if (await exists(seeded)) {
    const digest = await sha256File(seeded);
    if (digest === expected.sha256) {
      await copyFile(seeded, target);
      console.log(`  raw ${name}: copied from generate.mjs cache (${fmtMB(expected.size)}, sha256 ok)`);
      return;
    }
    console.log(`  raw ${name}: generate.mjs cache digest mismatch (different revision?), downloading`);
  }

  const url = `https://huggingface.co/${modelId}/resolve/${revision}/onnx/${name}`;
  console.log(`  raw ${name}: downloading ${fmtMB(expected.size)} from ${url}`);
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`download failed: HTTP ${res.status} (${url})`);
  }
  const partial = `${target}.partial`;
  const hash = createHash('sha256');
  await pipeline(
    Readable.fromWeb(res.body),
    async function* (source) {
      for await (const chunk of source) {
        hash.update(chunk);
        yield chunk;
      }
    },
    createWriteStream(partial),
  );
  const digest = hash.digest('hex');
  if (expected.sha256 && digest !== expected.sha256) {
    await rm(partial, { force: true });
    throw new Error(
      `sha256 mismatch for ${name}: expected ${expected.sha256}, got ${digest} — refusing to cache`,
    );
  }
  await rename(partial, target);
  console.log(`  raw ${name}: downloaded (sha256 ok)`);
}

/**
 * Merge `<base>.onnx` + `<base>.onnx_data` into one self-contained file via
 * the export venv's `onnx` package (onnx.load resolves external data into
 * raw_data and clears the external references; save_model then writes a
 * single protobuf — both graphs are far below the 2 GB protobuf limit).
 */
function mergeExternalData(src, dst) {
  const script = `
import sys
import onnx
from onnx.external_data_helper import _get_all_tensors

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)  # load_external_data=True by default
external = [t.name for t in _get_all_tensors(model) if t.data_location == onnx.TensorProto.EXTERNAL]
if external:
    raise SystemExit(f"external data survived the load for tensors: {external[:5]} ...")
onnx.save_model(model, dst)
`;
  const result = spawnSync(python, ['-c', script, src, dst], { stdio: 'inherit' });
  if (result.error || result.status !== 0) {
    throw new Error(
      `external-data merge failed for ${src} (python: ${python}). ` +
        `Ensure the tools/export venv exists with 'onnx' installed, or set WEBSAM_EXPORT_PYTHON.`,
    );
  }
}

// ---------------------------------------------------------------------------
// Manifest emission (schemaVersion 1 — packages/core/src/weights/manifest.ts)
// ---------------------------------------------------------------------------

/** Tensor IO contracts pinned by tools/export/spikes/s0/FINDINGS.md. */
const EMBED_OUTPUTS = {
  embed0: { name: 'image_embeddings.0', dtype: 'float32', shape: ['batch_size', 32, 288, 288] },
  embed1: { name: 'image_embeddings.1', dtype: 'float32', shape: ['batch_size', 64, 144, 144] },
  embed2: { name: 'image_embeddings.2', dtype: 'float32', shape: ['batch_size', 256, 72, 72] },
};

function buildManifest(tier, refs) {
  // Each graph's q4f16 file is ALSO listed under 'fp32' (same bytes): the
  // repo only ships this quant, and resolveDevice's auto preference on
  // wasm (['int8','fp32']) / webgpu-without-f16 (['fp32','int8']) would
  // otherwise find nothing. Local test/demo manifests only — a production
  // CDN manifest must never alias quants.
  const files = (ref) => ({ q4f16: ref, fp32: ref });
  return {
    schemaVersion: 1,
    tier,
    opset: 18,
    _source: `${modelId}@${revision} (external data merged by tools/goldens/fetch-models.mjs)`,
    _quantAlias: "fp32 aliases the q4f16 bytes so quant:'auto' resolves on every device; test/demo only",
    graphs: {
      visionEncoder: {
        files: files(refs.visionEncoder),
        inputs: {
          pixels: { name: 'pixel_values', dtype: 'float32', shape: ['batch_size', 3, 1008, 1008] },
        },
        outputs: EMBED_OUTPUTS,
      },
      promptDecoder: {
        files: files(refs.promptDecoder),
        inputs: {
          points: {
            name: 'input_points',
            dtype: 'float32',
            shape: ['batch_size', 1, 'num_points_per_image', 2],
          },
          labels: {
            name: 'input_labels',
            dtype: 'int64',
            shape: ['batch_size', 1, 'num_points_per_image'],
          },
          boxes: {
            name: 'input_boxes',
            dtype: 'float32',
            shape: ['batch_size', 'num_boxes_per_image', 4],
          },
          ...EMBED_OUTPUTS,
        },
        outputs: {
          iouScores: {
            name: 'iou_scores',
            dtype: 'float32',
            shape: ['batch_size', 'num_boxes_or_points', 3],
          },
          maskLogits: {
            name: 'pred_masks',
            dtype: 'float32',
            shape: ['batch_size', 'num_boxes_or_points', 3, 288, 288],
          },
          objectScoreLogits: {
            name: 'object_score_logits',
            dtype: 'float32',
            shape: ['batch_size', 'num_boxes_or_points', 1],
          },
        },
      },
    },
    toolchain: {
      exporter: `onnx-community (community export; merged single-file by fetch-models.mjs)`,
      pytorch: '2.9.0',
      transformers: '5.0.0.dev0',
    },
    preprocess: {
      mode: 'square-stretch',
      inputSize: 1008,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      maskSize: 288,
    },
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log(`fetch-models: ${modelId}@${revision} dtype=${dtype}`);
await mkdir(rawDir, { recursive: true });

const expected = await fetchExpectedDigests();

const refs = {};
for (const graph of GRAPHS) {
  const onnxName = `${graph.base}.onnx`;
  const dataName = `${graph.base}.onnx_data`;
  const onnxExpected = expected.get(onnxName);
  if (!onnxExpected) {
    throw new Error(`${onnxName} not found in the HF listing at revision ${revision}`);
  }

  console.log(`${graph.role}:`);
  await ensureRawFile(onnxName, onnxExpected);
  const dataExpected = expected.get(dataName);
  if (dataExpected) {
    await ensureRawFile(dataName, dataExpected);
  }

  const merged = join(cacheDir, onnxName);
  if (force || !(await exists(merged))) {
    console.log(`  merging external data → ${onnxName}`);
    mergeExternalData(join(rawDir, onnxName), merged);
  } else {
    console.log(`  merged ${onnxName}: cached (FORCE=1 to redo)`);
  }

  const bytes = (await stat(merged)).size;
  const sha256 = await sha256File(merged);
  refs[graph.role] = { path: onnxName, sha256, bytes };
  console.log(`  merged ${onnxName}: ${bytes} bytes (${fmtMB(bytes)}) sha256=${sha256}`);
}

for (const [file, tier] of [
  ['manifest.json', CANONICAL_TIER],
  ['manifest-e2e.json', E2E_TIER],
]) {
  const path = join(cacheDir, file);
  await writeFile(path, `${JSON.stringify(buildManifest(tier, refs), null, 2)}\n`);
  console.log(`wrote ${file} (tier '${tier}')`);
}

console.log(`done — model cache at ${cacheDir}`);
