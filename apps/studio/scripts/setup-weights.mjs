#!/usr/bin/env node
/**
 * Copies the EdgeTAM ONNX graphs + manifest.json from the shared golden
 * models cache (`tools/goldens/models-cache/edgetam/`) into
 * `apps/studio/public/models/edgetam/` so the dev server / build / browser
 * test suite can fetch them from `/models/edgetam/*` at runtime.
 *
 * `apps/studio/public/models/` is gitignored — this script is a required
 * step before `dev`, `build`, and `test:browser` (see apps/studio/README.md).
 *
 * Mirrors the never-silently-skip precondition used by
 * `packages/core/src/e2e/video-golden.browser.test.ts`'s `requireModels()`:
 * if the source cache isn't present, fail loudly with the regen command
 * instead of silently producing an empty/missing models dir.
 */
import { cp, readFile, stat, readdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, '..', '..', '..');

const SRC = join(repoRoot, 'tools', 'goldens', 'models-cache', 'edgetam');
const DEST = join(__dirname, '..', 'public', 'models', 'edgetam');

/**
 * @param {string} path
 * @returns {Promise<boolean>}
 */
async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * @param {string} dir
 * @returns {Promise<number>} total bytes of all files under `dir` (recursive)
 */
async function totalBytes(dir) {
  let total = 0;
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      total += await totalBytes(full);
    } else if (entry.isFile()) {
      total += (await stat(full)).size;
    }
  }
  return total;
}

async function main() {
  const manifestPath = join(SRC, 'manifest.json');
  if (!(await exists(SRC)) || !(await exists(manifestPath))) {
    console.error(
      `[setup-weights] Source models cache not found at:\n  ${SRC}\n` +
        `(or its manifest.json is missing).\n\n` +
        `Run:\n  cd tools/goldens && ../export/.venv/bin/python make-video-golden.py\n` +
        `then re-run this script.`,
    );
    process.exit(1);
  }

  await cp(SRC, DEST, { recursive: true, force: true });

  const manifest = JSON.parse(await readFile(join(DEST, 'manifest.json'), 'utf8'));
  if (manifest.tier !== 'edgetam') {
    console.error(
      `[setup-weights] Copied manifest.json has unexpected tier "${manifest.tier}" ` +
        `(expected "edgetam"). Refusing to proceed silently.`,
    );
    process.exit(1);
  }

  const bytes = await totalBytes(DEST);
  const mb = (bytes / (1024 * 1024)).toFixed(1);
  const entries = await readdir(DEST, { withFileTypes: true });
  const files = entries.filter((e) => e.isFile()).map((e) => e.name).sort();

  console.log(`[setup-weights] Copied ${files.length} files from:\n  ${SRC}\nto:\n  ${DEST}`);
  for (const f of files) console.log(`  - ${f}`);
  console.log(`[setup-weights] Total: ${bytes} bytes (${mb} MB)`);
  console.log(`[setup-weights] Manifest tier: ${manifest.tier}, opset: ${manifest.opset}`);
}

main().catch((err) => {
  console.error('[setup-weights] Failed:', err);
  process.exit(1);
});
