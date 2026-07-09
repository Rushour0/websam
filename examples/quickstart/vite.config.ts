import { defineConfig } from 'vite';

/**
 * COOP/COEP headers required for `crossOriginIsolated === true`, which in
 * turn unlocks SharedArrayBuffer for multithreaded WASM inference. The
 * WebGPU path does not strictly need this, but the worker always sets up the
 * WASM fallback, so keep it on in dev. Mirror this in production hosting
 * (e.g. a `_headers` file) — see the README.
 */
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
} as const;

export default defineConfig({
  // The @websam3/core inference worker (imported via `?worker&url`) uses
  // dynamic import (onnxruntime-web), which the default iife worker build
  // cannot code-split — module workers require the es format.
  worker: {
    format: 'es',
  },
  server: {
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
});
