// Golden fixture generator for the websam M1 gate.
//
// Runs onnx-community/sam3-tracker-ONNX via @huggingface/transformers (Node,
// CPU) on the deterministic non-square scene (fixtures/scene-640x427.png) and
// writes reference masks + metadata that the browser pipeline must match at
// IoU >= 0.9.
//
// Usage:  . ~/.nvm/nvm.sh && nvm use 22 && node generate.mjs
//
// Per the model card (https://huggingface.co/onnx-community/sam3-tracker-ONNX):
//   Sam3TrackerModel + AutoProcessor + RawImage; input_points are given in
//   SOURCE-image pixel coords (the processor rescales them to the reshaped
//   input size); processor.post_process_masks() upsamples the 256x256 logits
//   back to source resolution and binarizes at mask_threshold (default 0.0).
import {
  Sam3TrackerModel,
  AutoProcessor,
  RawImage,
  env,
} from "@huggingface/transformers";
import sharp from "sharp";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "fixtures");
mkdirSync(fixtures, { recursive: true });

// Keep the model cache inside this dir (gitignored) so it never pollutes the
// pnpm workspace or the user's global HF cache.
env.cacheDir = join(here, "models");

const MODEL_ID = "onnx-community/sam3-tracker-ONNX";
// Smallest variant pair on the model card. Overridable: DTYPE=fp16 node generate.mjs
const DTYPE = process.env.DTYPE ?? "q4f16";
const DEVICE = "cpu"; // Node: onnxruntime-node CPU EP. Determinism > speed.
const MASK_THRESHOLD = 0.0; // transformers.js post_process_masks default

const TRANSFORMERS_VERSION = JSON.parse(
  readFileSync(join(here, "node_modules/@huggingface/transformers/package.json"), "utf8"),
).version;

// Scene geometry — MUST match make-scene.mjs.
const SCENE = { width: 640, height: 427 };
const SHAPES = {
  bigCircle: { bbox: [90, 120, 270, 300], center: [180, 210], desc: "orange circle c=(180,210) r=90" },
  rectangle: { bbox: [380, 120, 550, 320], center: [465, 220], desc: "green rect x 380..550, y 120..320" },
  smallCircle: { bbox: [55, 15, 145, 105], center: [100, 60], desc: "blue circle c=(100,60) r=45" },
};

// The two golden prompts (source-pixel coords, [x, y]).
const PROMPTS = [
  {
    name: "point1",
    description: "single positive point at the center of the big orange circle",
    input_points: [[[SHAPES.bigCircle.center]]],
    input_labels: [[[1]]],
    expectShape: "bigCircle",
  },
  {
    name: "point2",
    description:
      "two-point prompt: positive at the center of the green rectangle, negative at the center of the big orange circle",
    input_points: [[[SHAPES.rectangle.center, SHAPES.bigCircle.center]]],
    input_labels: [[[1, 0]]],
    expectShape: "rectangle",
  },
];

// ---------------------------------------------------------------------------
// RLE helpers. Format: row-major scan of the binary mask (y*width + x),
// counts[] = alternating run lengths starting with a run of ZEROS
// (counts[0] may be 0 if the mask starts with a foreground pixel).
// sum(counts) === width * height.
function encodeRLE(bin, width, height) {
  const counts = [];
  let current = 0; // we always start by counting zeros
  let run = 0;
  for (let i = 0; i < bin.length; ++i) {
    const v = bin[i] ? 1 : 0;
    if (v === current) {
      ++run;
    } else {
      counts.push(run);
      current = v;
      run = 1;
    }
  }
  counts.push(run);
  return { width, height, counts };
}

function decodeRLE({ width, height, counts }) {
  const out = new Uint8Array(width * height);
  let i = 0;
  let value = 0;
  for (const run of counts) {
    if (value) out.fill(1, i, i + run);
    i += run;
    value ^= 1;
  }
  if (i !== width * height) throw new Error(`RLE decode length mismatch: ${i} != ${width * height}`);
  return out;
}

function maskStats(bin, width, height) {
  let area = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      if (bin[y * width + x]) {
        ++area;
        sx += x;
        sy += y;
      }
    }
  }
  return {
    area,
    areaFraction: area / (width * height),
    centroid: area ? [sx / area, sy / area] : null,
  };
}

// ---------------------------------------------------------------------------
async function main() {
  // Resolve the exact model revision for the meta files.
  let revision = "main";
  try {
    const res = await fetch(`https://huggingface.co/api/models/${MODEL_ID}`);
    revision = (await res.json()).sha ?? "main";
  } catch {
    console.warn("warning: could not resolve model revision sha; recording 'main'");
  }

  console.log(`loading ${MODEL_ID} (dtype=${DTYPE}, device=${DEVICE})...`);
  const model = await Sam3TrackerModel.from_pretrained(MODEL_ID, { dtype: DTYPE, device: DEVICE });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);

  const scenePath = join(fixtures, "scene-640x427.png");
  const raw_image = await RawImage.read(scenePath);
  if (raw_image.width !== SCENE.width || raw_image.height !== SCENE.height) {
    throw new Error(`scene is ${raw_image.width}x${raw_image.height}, expected ${SCENE.width}x${SCENE.height}`);
  }

  for (const prompt of PROMPTS) {
    console.log(`\n=== ${prompt.name}: ${prompt.description}`);
    const inputs = await processor(raw_image, {
      input_points: prompt.input_points,
      input_labels: prompt.input_labels,
    });

    const original_sizes = inputs.original_sizes.map((s) => [...s]);
    const reshaped_input_sizes = inputs.reshaped_input_sizes.map((s) => [...s]);
    console.log("original_sizes:", original_sizes, "reshaped_input_sizes:", reshaped_input_sizes);
    console.log("pixel_values dims:", inputs.pixel_values.dims);

    const t0 = Date.now();
    const outputs = await model(inputs);
    console.log(`inference: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    console.log("pred_masks dims:", outputs.pred_masks.dims, "iou_scores dims:", outputs.iou_scores.dims);

    // Upscale to source resolution + binarize at MASK_THRESHOLD.
    const masks = await processor.post_process_masks(
      outputs.pred_masks,
      inputs.original_sizes,
      inputs.reshaped_input_sizes,
      { mask_threshold: MASK_THRESHOLD },
    );
    const maskTensor = masks[0]; // dims [point_batch=1, num_multimask=3, H, W]
    const [pointBatch, numMasks, H, W] = maskTensor.dims;
    if (H !== SCENE.height || W !== SCENE.width) {
      throw new Error(`post-processed mask is ${W}x${H}, expected source resolution ${SCENE.width}x${SCENE.height}`);
    }

    // Pick the multimask candidate with the highest predicted IoU.
    const iouScores = Array.from(outputs.iou_scores.data, Number);
    let best = 0;
    for (let i = 1; i < iouScores.length; ++i) if (iouScores[i] > iouScores[best]) best = i;
    console.log("iou_scores:", iouScores.map((s) => s.toFixed(4)), "-> selected mask", best);

    const maskData = maskTensor.data; // Uint8Array, [1, 3, H, W] flattened, values 0/1
    const bin = new Uint8Array(W * H);
    bin.set(maskData.subarray(best * W * H, (best + 1) * W * H));

    const stats = maskStats(bin, W, H);
    console.log(
      `area: ${stats.area} px (${(stats.areaFraction * 100).toFixed(2)}%), ` +
        `centroid: (${stats.centroid?.[0].toFixed(1)}, ${stats.centroid?.[1].toFixed(1)})`,
    );

    // --- Sanity checks ------------------------------------------------------
    if (!(stats.areaFraction > 0.01 && stats.areaFraction < 0.9)) {
      throw new Error(`SANITY FAIL (${prompt.name}): mask area fraction ${stats.areaFraction} outside (0.01, 0.9)`);
    }
    const [bx1, by1, bx2, by2] = SHAPES[prompt.expectShape].bbox;
    const [cx, cy] = stats.centroid;
    if (!(cx >= bx1 && cx <= bx2 && cy >= by1 && cy <= by2)) {
      throw new Error(
        `SANITY FAIL (${prompt.name}): centroid (${cx.toFixed(1)}, ${cy.toFixed(1)}) not inside ` +
          `${prompt.expectShape} bbox [${SHAPES[prompt.expectShape].bbox}]`,
      );
    }

    // --- Write PNG (binary 0/255, source resolution) ------------------------
    const png255 = Buffer.alloc(W * H);
    for (let i = 0; i < bin.length; ++i) png255[i] = bin[i] ? 255 : 0;
    const pngPath = join(fixtures, `golden-mask-${prompt.name}.png`);
    await sharp(png255, { raw: { width: W, height: H, channels: 1 } })
      .png({ compressionLevel: 9 })
      .toFile(pngPath);

    // --- Write RLE JSON ------------------------------------------------------
    const rle = encodeRLE(bin, W, H);
    const rlePath = join(fixtures, `golden-mask-${prompt.name}.rle.json`);
    writeFileSync(rlePath, JSON.stringify(rle));

    // --- Verify: RLE decode must match the PNG pixel-for-pixel --------------
    const decoded = decodeRLE(JSON.parse(readFileSync(rlePath, "utf8")));
    const pngBack = await sharp(pngPath).raw().toBuffer({ resolveWithObject: true });
    if (pngBack.info.width !== W || pngBack.info.height !== H) {
      throw new Error(`SANITY FAIL (${prompt.name}): PNG roundtrip size mismatch`);
    }
    const ch = pngBack.info.channels;
    for (let i = 0; i < W * H; ++i) {
      const pngVal = pngBack.data[i * ch] === 255 ? 1 : pngBack.data[i * ch] === 0 ? 0 : -1;
      if (pngVal !== decoded[i]) {
        throw new Error(`SANITY FAIL (${prompt.name}): RLE/PNG mismatch at pixel ${i} (png=${pngBack.data[i * ch]}, rle=${decoded[i]})`);
      }
    }
    console.log("sanity checks passed: RLE == PNG pixel-for-pixel, non-trivial area, centroid inside prompted shape");

    // --- Write meta ----------------------------------------------------------
    const meta = {
      generated: new Date().toISOString(),
      model_id: MODEL_ID,
      model_revision: revision,
      dtype: DTYPE,
      device: DEVICE,
      transformers_js_version: TRANSFORMERS_VERSION,
      source_image: "scene-640x427.png",
      source_size: { width: SCENE.width, height: SCENE.height },
      prompt: {
        description: prompt.description,
        // Source-pixel coordinates, [x, y]; labels: 1 = positive, 0 = negative.
        input_points: prompt.input_points,
        input_labels: prompt.input_labels,
        coordinate_space: "source image pixels (processor rescales to reshaped input size)",
      },
      expected_shape: { name: prompt.expectShape, ...SHAPES[prompt.expectShape] },
      preprocessing: {
        original_sizes,
        reshaped_input_sizes,
        pixel_values_dims: [...inputs.pixel_values.dims],
      },
      mask_threshold: MASK_THRESHOLD,
      multimask: {
        iou_scores: iouScores,
        selected_index: best,
        num_candidates: numMasks,
      },
      mask_stats: {
        area_px: stats.area,
        area_fraction: stats.areaFraction,
        centroid_xy: stats.centroid,
      },
      rle_format:
        "row-major scan (index = y*width + x); counts = alternating run lengths starting with a run of ZEROS; sum(counts) = width*height",
      files: {
        png: `golden-mask-${prompt.name}.png`,
        rle: `golden-mask-${prompt.name}.rle.json`,
      },
    };
    writeFileSync(join(fixtures, `golden-meta-${prompt.name}.json`), JSON.stringify(meta, null, 2) + "\n");
    console.log(`wrote ${pngPath}, ${rlePath}, golden-meta-${prompt.name}.json`);
  }

  // Combined meta for convenience (golden-meta.json = both prompts).
  const combined = {
    note: "See golden-meta-point1.json / golden-meta-point2.json for full per-prompt metadata.",
    model_id: MODEL_ID,
    model_revision: revision,
    dtype: DTYPE,
    device: DEVICE,
    transformers_js_version: TRANSFORMERS_VERSION,
    mask_threshold: MASK_THRESHOLD,
    prompts: PROMPTS.map((p) => ({
      name: p.name,
      description: p.description,
      input_points: p.input_points,
      input_labels: p.input_labels,
    })),
  };
  writeFileSync(join(fixtures, "golden-meta.json"), JSON.stringify(combined, null, 2) + "\n");
  console.log("\nall golden fixtures written to", fixtures);
}

await main();
