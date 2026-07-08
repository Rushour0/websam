import { describe, expect, it } from 'vitest';
import { WeightVerifyError } from '../errors.js';
import { parseModelManifest, type ModelManifest } from './manifest.js';

const SOURCE_URL = 'https://models.example.test/sam3-tracker/manifest.json';
const SHA = 'a'.repeat(64);

/** A fresh, fully valid manifest JSON object (mutate per test). */
function validManifest(): Record<string, unknown> {
  return {
    schemaVersion: 1,
    tier: 'sam3-tracker',
    opset: 18,
    graphs: {
      visionEncoder: {
        files: {
          fp32: { path: 'vision_encoder.onnx', sha256: SHA, bytes: 1024 },
          q4f16: { path: 'vision_encoder_q4f16.onnx', sha256: 'b'.repeat(64), bytes: 512 },
        },
        inputs: {
          pixels: { name: 'pixel_values', dtype: 'float32', shape: ['batch_size', 3, 1008, 1008] },
        },
        outputs: {
          embed0: { name: 'image_embeddings.0', dtype: 'float32', shape: ['batch_size', 32, 288, 288] },
          embed1: { name: 'image_embeddings.1', dtype: 'float32', shape: ['batch_size', 64, 144, 144] },
          embed2: { name: 'image_embeddings.2', dtype: 'float32', shape: ['batch_size', 256, 72, 72] },
        },
      },
      promptDecoder: {
        files: {
          fp32: { path: 'prompt_encoder_mask_decoder.onnx', sha256: 'c'.repeat(64), bytes: 2048 },
        },
        inputs: {
          points: { name: 'input_points', dtype: 'float32', shape: ['batch_size', 1, 'num_points', 2] },
          labels: { name: 'input_labels', dtype: 'int64', shape: ['batch_size', 1, 'num_points'] },
          boxes: { name: 'input_boxes', dtype: 'float32', shape: ['batch_size', 'num_boxes', 4] },
        },
        outputs: {
          iouScores: { name: 'iou_scores', dtype: 'float32', shape: ['batch_size', 'num', 3] },
          maskLogits: { name: 'pred_masks', dtype: 'float32', shape: ['batch_size', 'num', 3, 288, 288] },
          objectScoreLogits: { name: 'object_score_logits', dtype: 'float32', shape: ['batch_size', 'num', 1] },
        },
      },
    },
    toolchain: { exporter: 'onnx-community', pytorch: '2.9.0', transformers: '5.0.0.dev0' },
    preprocess: {
      mode: 'square-stretch',
      inputSize: 1008,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      maskSize: 288,
    },
  };
}

/** Minimal graph entry for a given ONNX file name (video-role stubs). */
function graphEntry(file: string): Record<string, unknown> {
  return {
    files: { fp32: { path: file, sha256: 'e'.repeat(64), bytes: 128 } },
    inputs: {},
    outputs: {},
  };
}

/** The EdgeTAM `video` section (tools/export spec.py EDGETAM_1024 constants). */
function edgetamVideoSection(): Record<string, unknown> {
  return {
    maxCondFrames: 1,
    numRecent: 6,
    tokensPerMemoryMap: 256,
    ptrTokens: 64,
    maxObjectPointers: 16,
    kvLen: 1856, // 7 * 256 + 64
    memDim: 64,
    embedDim: 256,
    gridSize: 64,
    multiObjectBatch: true,
    initPath: 'noMemFlag',
    tposDelivery: 'indices',
    occlusionThreshold: 0,
  };
}

/** A valid manifest carrying the video section + all four video graph roles. */
function validVideoManifest(): Record<string, unknown> {
  const json = validManifest();
  const graphs = json['graphs'] as Record<string, unknown>;
  graphs['videoEncoder'] = graphEntry('video_encoder.onnx');
  graphs['memoryAttention'] = graphEntry('memory_attention.onnx');
  graphs['maskDecoderVideo'] = graphEntry('mask_decoder_video.onnx');
  graphs['memoryEncoder'] = graphEntry('memory_encoder.onnx');
  json['video'] = edgetamVideoSection();
  return json;
}

function expectRejects(json: unknown, messagePart: string | RegExp): void {
  let caught: unknown;
  try {
    parseModelManifest(json, SOURCE_URL);
  } catch (err) {
    caught = err;
  }
  expect(caught).toBeInstanceOf(WeightVerifyError);
  const error = caught as WeightVerifyError;
  expect(error.code).toBe('WEIGHT_VERIFY_FAILED');
  expect(error.message).toContain(SOURCE_URL);
  expect(error.message).toMatch(messagePart);
}

describe('parseModelManifest', () => {
  it('accepts a valid manifest and returns typed data', () => {
    const manifest: ModelManifest = parseModelManifest(validManifest(), SOURCE_URL);
    expect(manifest.schemaVersion).toBe(1);
    expect(manifest.tier).toBe('sam3-tracker');
    expect(manifest.opset).toBe(18);
    expect(manifest.preprocess).toEqual({
      mode: 'square-stretch',
      inputSize: 1008,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      maskSize: 288,
    });
    // Semantic-name binding: runtime code reads the ONNX name off the spec.
    expect(manifest.graphs.visionEncoder?.inputs['pixels']?.name).toBe('pixel_values');
    expect(manifest.graphs.promptDecoder?.outputs['maskLogits']?.name).toBe('pred_masks');
    expect(manifest.graphs.visionEncoder?.files.fp32).toEqual({
      path: 'vision_encoder.onnx',
      sha256: SHA,
      bytes: 1024,
    });
  });

  it('returns a defensive copy — mutating the input does not affect the result', () => {
    const json = validManifest();
    const manifest = parseModelManifest(json, SOURCE_URL);
    (json['preprocess'] as Record<string, unknown>)['inputSize'] = 1;
    expect(manifest.preprocess.inputSize).toBe(1008);
  });

  it('keeps the GraphRole union open: unknown roles parse fine (no schema bump for video)', () => {
    const json = validManifest();
    (json['graphs'] as Record<string, unknown>)['memoryAttention'] = {
      files: { fp32: { path: 'memory_attention.onnx', sha256: 'd'.repeat(64), bytes: 64 } },
      inputs: {},
      outputs: {},
    };
    const manifest = parseModelManifest(json, SOURCE_URL);
    expect(manifest.graphs['memoryAttention']?.files.fp32?.bytes).toBe(64);
  });

  it('rejects non-object json', () => {
    expectRejects(null, /must be a JSON object/);
    expectRejects([], /must be a JSON object/);
    expectRejects('{}', /must be a JSON object/);
  });

  it('rejects a wrong or missing schemaVersion', () => {
    expectRejects({ ...validManifest(), schemaVersion: 2 }, /schemaVersion/);
    expectRejects({ ...validManifest(), schemaVersion: '1' }, /schemaVersion/);
    const missing = validManifest();
    delete missing['schemaVersion'];
    expectRejects(missing, /schemaVersion/);
  });

  it('rejects a missing or empty tier and a bad opset', () => {
    expectRejects({ ...validManifest(), tier: '' }, /tier/);
    expectRejects({ ...validManifest(), tier: 42 }, /tier/);
    expectRejects({ ...validManifest(), opset: 0 }, /opset/);
    expectRejects({ ...validManifest(), opset: 18.5 }, /opset/);
  });

  it('rejects malformed weight file refs', () => {
    const upperSha = validManifest();
    const encoder = (upperSha['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (encoder?.['files'] as Record<string, Record<string, unknown>>)['fp32'] = {
      path: 'x.onnx',
      sha256: 'A'.repeat(64), // uppercase → invalid
      bytes: 10,
    };
    expectRejects(upperSha, /sha256/);

    const shortSha = validManifest();
    const enc2 = (shortSha['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc2?.['files'] as Record<string, Record<string, unknown>>)['fp32'] = {
      path: 'x.onnx',
      sha256: 'ab12',
      bytes: 10,
    };
    expectRejects(shortSha, /sha256/);

    const badBytes = validManifest();
    const enc3 = (badBytes['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc3?.['files'] as Record<string, Record<string, unknown>>)['fp32'] = {
      path: 'x.onnx',
      sha256: SHA,
      bytes: 0,
    };
    expectRejects(badBytes, /bytes/);
  });

  it('rejects an unknown quant key and an empty files map', () => {
    const unknownQuant = validManifest();
    const enc = (unknownQuant['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc?.['files'] as Record<string, unknown>)['fp8'] = { path: 'x.onnx', sha256: SHA, bytes: 1 };
    expectRejects(unknownQuant, /unknown quant/);

    const emptyFiles = validManifest();
    const enc2 = (emptyFiles['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    if (enc2) enc2['files'] = {};
    expectRejects(emptyFiles, /at least one quantization/);
  });

  it('rejects bad tensor specs (dtype, shape, name)', () => {
    const badDtype = validManifest();
    const enc = (badDtype['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc?.['inputs'] as Record<string, unknown>)['pixels'] = {
      name: 'pixel_values',
      dtype: 'float64',
      shape: [1],
    };
    expectRejects(badDtype, /dtype/);

    const badShape = validManifest();
    const enc2 = (badShape['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc2?.['inputs'] as Record<string, unknown>)['pixels'] = {
      name: 'pixel_values',
      dtype: 'float32',
      shape: [1, null],
    };
    expectRejects(badShape, /shape\[1\]/);

    const badName = validManifest();
    const enc3 = (badName['graphs'] as Record<string, Record<string, unknown>>)['visionEncoder'];
    (enc3?.['inputs'] as Record<string, unknown>)['pixels'] = { name: '', dtype: 'float32', shape: [] };
    expectRejects(badName, /name/);
  });

  it('rejects a missing toolchain exporter', () => {
    expectRejects({ ...validManifest(), toolchain: {} }, /exporter/);
    expectRejects({ ...validManifest(), toolchain: { exporter: 'x', pytorch: 2.9 } }, /pytorch/);
  });

  it('rejects bad preprocess blocks', () => {
    const badMode = validManifest();
    (badMode['preprocess'] as Record<string, unknown>)['mode'] = 'crop';
    expectRejects(badMode, /preprocess\.mode/);

    const badMean = validManifest();
    (badMean['preprocess'] as Record<string, unknown>)['mean'] = [0.5, 0.5];
    expectRejects(badMean, /preprocess\.mean/);

    const badStd = validManifest();
    (badStd['preprocess'] as Record<string, unknown>)['std'] = [0.5, 0.5, 'x'];
    expectRejects(badStd, /preprocess\.std/);

    const badSize = validManifest();
    (badSize['preprocess'] as Record<string, unknown>)['inputSize'] = -1008;
    expectRejects(badSize, /inputSize/);

    const badMask = validManifest();
    delete (badMask['preprocess'] as Record<string, unknown>)['maskSize'];
    expectRejects(badMask, /maskSize/);
  });
});

describe('parseModelManifest — video section (M2)', () => {
  it('parses a valid EdgeTAM video section', () => {
    const manifest = parseModelManifest(validVideoManifest(), SOURCE_URL);
    expect(manifest.video).toEqual({
      maxCondFrames: 1,
      numRecent: 6,
      tokensPerMemoryMap: 256,
      ptrTokens: 64,
      maxObjectPointers: 16,
      kvLen: 1856,
      memDim: 64,
      embedDim: 256,
      gridSize: 64,
      multiObjectBatch: true,
      initPath: 'noMemFlag',
      tposDelivery: 'indices',
      occlusionThreshold: 0,
    });
  });

  it('omits `video` when absent (M1 image-only manifests keep parsing)', () => {
    const manifest = parseModelManifest(validManifest(), SOURCE_URL);
    expect(manifest.video).toBeUndefined();
  });

  it('returns a defensive copy of the video section', () => {
    const json = validVideoManifest();
    const manifest = parseModelManifest(json, SOURCE_URL);
    (json['video'] as Record<string, unknown>)['kvLen'] = 1;
    expect(manifest.video?.kvLen).toBe(1856);
  });

  it('accepts the alternate enum values and a negative occlusion threshold', () => {
    const json = validVideoManifest();
    const video = json['video'] as Record<string, unknown>;
    video['initPath'] = 'noMemGraph';
    video['tposDelivery'] = 'precombined';
    video['multiObjectBatch'] = false;
    video['occlusionThreshold'] = -2.5;
    const manifest = parseModelManifest(json, SOURCE_URL);
    expect(manifest.video?.initPath).toBe('noMemGraph');
    expect(manifest.video?.tposDelivery).toBe('precombined');
    expect(manifest.video?.multiObjectBatch).toBe(false);
    expect(manifest.video?.occlusionThreshold).toBe(-2.5);
  });

  it('enforces the kvLen identity: maps × tokensPerMemoryMap + ptrTokens', () => {
    const json = validVideoManifest();
    (json['video'] as Record<string, unknown>)['kvLen'] = 1857;
    expectRejects(json, /video\.kvLen.*1856/);
  });

  it('rejects non-positive or non-integer counts', () => {
    for (const key of [
      'maxCondFrames',
      'numRecent',
      'tokensPerMemoryMap',
      'ptrTokens',
      'maxObjectPointers',
      'kvLen',
      'memDim',
      'embedDim',
      'gridSize',
    ]) {
      const zero = validVideoManifest();
      (zero['video'] as Record<string, unknown>)[key] = 0;
      expectRejects(zero, new RegExp(`video\\.${key}`));

      const fractional = validVideoManifest();
      (fractional['video'] as Record<string, unknown>)[key] = 1.5;
      expectRejects(fractional, new RegExp(`video\\.${key}`));
    }
  });

  it('rejects bad enum and flag fields', () => {
    const badInit = validVideoManifest();
    (badInit['video'] as Record<string, unknown>)['initPath'] = 'always';
    expectRejects(badInit, /video\.initPath/);

    const badTpos = validVideoManifest();
    (badTpos['video'] as Record<string, unknown>)['tposDelivery'] = 'gather';
    expectRejects(badTpos, /video\.tposDelivery/);

    const badBatch = validVideoManifest();
    (badBatch['video'] as Record<string, unknown>)['multiObjectBatch'] = 'yes';
    expectRejects(badBatch, /video\.multiObjectBatch/);

    const badThreshold = validVideoManifest();
    (badThreshold['video'] as Record<string, unknown>)['occlusionThreshold'] = 'low';
    expectRejects(badThreshold, /video\.occlusionThreshold/);

    const nanThreshold = validVideoManifest();
    (nanThreshold['video'] as Record<string, unknown>)['occlusionThreshold'] = Number.NaN;
    expectRejects(nanThreshold, /video\.occlusionThreshold/);

    const notObject = validVideoManifest();
    notObject['video'] = 'edgetam';
    expectRejects(notObject, /video: must be an object/);
  });

  it('requires all four video graph roles when the video section is present', () => {
    for (const role of ['videoEncoder', 'memoryAttention', 'maskDecoderVideo', 'memoryEncoder']) {
      const json = validVideoManifest();
      delete (json['graphs'] as Record<string, unknown>)[role];
      expectRejects(json, new RegExp(`'${role}' is required`));
    }
  });

  it('does not require video roles when the video section is absent', () => {
    // validManifest ships only the two image graphs and stays valid.
    expect(() => parseModelManifest(validManifest(), SOURCE_URL)).not.toThrow();
  });
});
