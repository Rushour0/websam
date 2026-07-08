import { defineConfig } from 'tsdown';

export default defineConfig({
  // Two sibling entry points (docs/m1-internal-contracts.md §6.1):
  // dist/index.js (public API) + dist/worker.js (module-worker entry, spawned
  // via `new URL('./worker.js', import.meta.url)` from index).
  entry: { index: 'src/index.ts', worker: 'src/worker/worker-entry.ts' },
  dts: true,
  format: 'esm',
  target: 'es2022',
  platform: 'browser',
});
