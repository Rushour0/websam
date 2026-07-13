import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * COOP/COEP headers required for `crossOriginIsolated === true`, which in turn
 * unlocks SharedArrayBuffer and therefore multithreaded WASM inference.
 * Mirrors `apps/demo/vite.config.ts`.
 */
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
} as const;

// Repo root, for serving out-of-workspace fixtures (tools/goldens) to the
// browser test project — same relative depth as packages/core/vitest.config.ts.
const repoRoot = fileURLToPath(new URL('../..', import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // The @websam3/core inference worker (imported via `?worker&url`) uses
  // dynamic import (onnxruntime-web), which the default iife worker build
  // cannot code-split — module workers require the es format.
  worker: {
    format: 'es',
  },
  server: {
    // Pinned: OPFS/Cache API model-weight storage is origin-scoped, so a
    // silent port hop (Vite's default when 5173 is taken) turns into a fresh,
    // empty origin and re-downloads the model on every reload.
    port: 5173,
    strictPort: true,
    headers: crossOriginIsolationHeaders,
    fs: {
      allow: [repoRoot],
    },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
