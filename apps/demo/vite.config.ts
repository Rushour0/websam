import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * COOP/COEP headers required for `crossOriginIsolated === true`, which in turn
 * unlocks SharedArrayBuffer and therefore multithreaded WASM inference.
 * Mirrored for production hosting in `public/_headers` (Cloudflare Pages).
 */
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
} as const;

export default defineConfig({
  plugins: [react()],
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
