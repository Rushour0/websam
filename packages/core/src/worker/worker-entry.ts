/**
 * The `"./worker"` module-worker entry (dist sibling `worker.js`, spawned by
 * the sessions layer as `new Worker(url, { type: 'module' })`).
 *
 * Side effects are exactly two, per docs/m1-internal-contracts.md §3.4:
 * install the typed-error `'throw'` transfer handler for this realm, then
 * expose one {@link WorkerEngine} over Comlink.
 */

import * as Comlink from 'comlink';
import { WorkerEngine } from './engine.js';
import { installErrorTransferHandler } from './error-envelope.js';

installErrorTransferHandler(Comlink);
Comlink.expose(new WorkerEngine());
