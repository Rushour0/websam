/**
 * Worker bootstrap for the e2e browser gate — test infrastructure, never
 * shipped.
 *
 * Two vitest-browser quirks make spawning the REAL worker entry directly
 * impossible in browser-mode tests:
 *
 * 1. Vitest's mocker plugin rewrites every dynamic import served by the test
 *    server into `globalThis[accessor].wrapDynamicImport(...)` — the
 *    accessor is `"__vitest_browser_runner__"` in browser mode
 *    (@vitest/browser dist/index.js `globalThisAccessor`) — but that global
 *    only exists in vitest's own frame, not in a raw module worker spawned
 *    by the code under test. Without a stub the engine's
 *    `import('onnxruntime-web')` (src/runtime/ort-env.ts) explodes inside
 *    the worker. The passthrough below is installed during module
 *    evaluation, before any `init()` message can trigger that import (both
 *    accessor spellings are stubbed in case the default changes back).
 *
 * 2. The import of the real entry must be STATIC: a top-level `await
 *    import(...)` suspends the worker's initial evaluation, the browser
 *    starts dispatching queued messages while Comlink's `expose` listener
 *    does not exist yet, and the engine's `init` call is silently dropped —
 *    an infinite hang, not an error.
 */
import '../worker/worker-entry.js';

for (const accessor of ['__vitest_browser_runner__', '__vitest_mocker__']) {
  (globalThis as Record<string, unknown>)[accessor] ??= {
    wrapDynamicImport: (moduleFactory: () => unknown): unknown => moduleFactory(),
  };
}
