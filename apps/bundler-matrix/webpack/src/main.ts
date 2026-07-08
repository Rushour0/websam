/**
 * Bundler-matrix canary entry (Webpack).
 *
 * Imports real runtime values from `@websam/core` so that any breakage in the
 * package's exports map, ESM output, or tree-shaking metadata fails this
 * app's `webpack` build in CI rather than surfacing in downstream consumers.
 */
import { WebGpuBackend, listModels } from '@websam/core';

const models = listModels();

const el = document.createElement('pre');
el.id = 'app';
el.textContent = `models registered: ${models.length} | backend: ${WebGpuBackend.name}`;
document.body.append(el);
