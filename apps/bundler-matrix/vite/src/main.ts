/**
 * Bundler-matrix canary entry (Vite).
 *
 * Imports real runtime values from `@websam/core` so that any breakage in the
 * package's exports map, ESM output, or tree-shaking metadata fails this
 * app's `vite build` in CI rather than surfacing in downstream consumers.
 */
import { WebGpuBackend, listModels } from '@websam/core';

const models = listModels();

const app = document.querySelector('#app');
if (app === null) {
  throw new Error('missing #app mount point');
}

app.textContent = `models registered: ${models.length} | backend: ${WebGpuBackend.name}`;
