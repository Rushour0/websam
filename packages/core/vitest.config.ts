import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Repo root: the browser project serves files from tools/goldens/** (golden
// fixtures + the fetch-models.mjs model cache) via /@fs/ URLs, which Vite
// only allows for paths inside server.fs.allow.
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
        // pre-bundling would break those relative lookups.
        optimizeDeps: {
          exclude: ['onnxruntime-web'],
        },
        server: {
          fs: {
            // The e2e golden gate (src/e2e/*.browser.test.ts) fetches the
            // committed fixtures and the gitignored model cache from
            // tools/goldens/ — out of the package root, inside the repo.
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
