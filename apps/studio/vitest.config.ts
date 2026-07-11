import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

/**
 * Repo root — the browser project serves `tools/goldens/fixtures/video/**`
 * (the committed golden clip + RLE fixtures) via `/@fs/` URLs, which Vite
 * only allows for paths inside `server.fs.allow`. Same relative depth from
 * `apps/studio/` as `packages/core/vitest.config.ts` is from `packages/core/`.
 */
const repoRoot = fileURLToPath(new URL('../..', import.meta.url));

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.browser.test.ts'],
        },
      },
      {
        // onnxruntime-web resolves its .wasm/.mjs assets via import.meta.url;
        // pre-bundling would break those relative lookups (mirrors
        // packages/core/vitest.config.ts and apps/studio/vite.config.ts).
        optimizeDeps: {
          exclude: ['onnxruntime-web'],
        },
        // Vitest does NOT merge apps/studio/vite.config.ts when a sibling
        // vitest.config.ts exists, so the app config's worker/COOP settings
        // must be duplicated here. `format: 'es'` is required: the
        // `@websam3/core/worker?worker&url` entry (segmenter-lifecycle.ts)
        // uses a top-level `await import('onnxruntime-web')`, which the
        // default `iife` worker bundle cannot represent.
        worker: {
          format: 'es',
        },
        server: {
          fs: {
            allow: [repoRoot],
          },
        },
        test: {
          name: 'browser',
          include: ['src/**/*.browser.test.ts'],
          browser: {
            enabled: true,
            provider: 'playwright',
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
