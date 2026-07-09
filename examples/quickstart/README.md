# websam quickstart

A minimal, vanilla-TypeScript (no framework) example: pick an image, click on
it, get a mask back — all running in the browser via `@websam3/core`. ~150
lines in `src/main.ts`; read it top to bottom, it's the whole app.

## Install

In your own project:

```sh
npm install @websam3/core
```

(This directory itself is part of the websam pnpm workspace and depends on
`@websam3/core` via `workspace:^` so it always builds against the in-repo
source — that's a monorepo convenience, not something external consumers do.)

## The minimal code

```ts
import { createSegmenter } from '@websam3/core';
import segmenterWorkerUrl from '@websam3/core/worker?worker&url'; // bundler escape hatch

const segmenter = await createSegmenter({
  model: 'edgetam', // Apache-2.0, no license gate, smallest download
  modelBaseUrl: '/models/', // see "Providing weights" below
  workerUrl: segmenterWorkerUrl,
  onProgress: (event) => console.log(event.phase, event.loaded, event.total),
});

const session = await segmenter.createImageSession();
await session.encode(imageBitmapOrCanvas); // run the vision encoder once

const [mask] = await session.decode([{ type: 'point', x: 180, y: 210, label: 1 }]);
const imageData = mask.toImageData(); // or mask.toBinary() / mask.toRLE()
```

That's the whole interactive-image API surface: `createSegmenter` →
`createImageSession` → `encode` once → `decode` per prompt.

## Why COOP/COEP

`vite.config.ts` sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` in dev. This makes the page
`crossOriginIsolated`, which unlocks `SharedArrayBuffer` — required for
multithreaded WASM inference (the fallback path). If you skip these headers,
websam still runs, but the WASM backend falls back to single-threaded and
will throw `CrossOriginIsolationRequiredError` in some configurations.

The **WebGPU path does not strictly need this** — if your target browsers
have WebGPU (recent desktop Chrome/Edge), you can relax COOP/COEP, but you
lose the WASM fallback's multithreading. Setting the headers costs nothing
and covers both paths, so we recommend keeping them in production too (mirror
`vite.config.ts` in your host's headers config, e.g. a `_headers` file on
Cloudflare Pages or Netlify).

## Providing weights

**Model weight hosting is not live yet** (tracked for milestone M4) — there
is no public URL you can point `modelBaseUrl` at today. Until then, pick one:

### Option A — generate EdgeTAM weights locally

From the repo root:

```sh
cd tools/goldens
../export/.venv/bin/python make-video-golden.py
```

This produces `tools/goldens/models-cache/edgetam/` (manifest + ONNX weights).
Copy it into this example's `public/models/edgetam/`:

```sh
mkdir -p examples/quickstart/public/models
cp -r tools/goldens/models-cache/edgetam examples/quickstart/public/models/
```

### Option B — reuse the M1 image (SAM 3 Tracker) weights

```sh
cd tools/goldens
node fetch-models.mjs
```

Then point at those weights and switch `model: 'edgetam'` to
`model: 'sam3-tracker'` in `src/main.ts` (SAM 3 Tracker is SAM-licensed —
you'll also need `acceptLicense: 'sam'` in the `createSegmenter` call).

### Option C — wait for hosted weights (M4)

Once weights are hosted, drop the `modelBaseUrl` override entirely (or point
it at the published host) and this example works with zero local setup.

### If weights aren't there

The app **builds and typechecks with no weights present** — `npm run build`
/ `npm run dev` both work. At runtime, without reachable weights,
`createSegmenter` rejects with `WeightVerifyError` and the page shows a
plain-language notice pointing back here, instead of a silent failure.

## Running it

```sh
# from the repo root (workspace build), or from this directory after
# `npm install @websam3/core` in a standalone copy
npm run dev
```

Open the printed URL, pick an image, click on it. Shift-click adds a negative
point to the same prompt call (this example keeps it to a single point per
click for readability — see `apps/demo/src/ImageTab.tsx` for a fuller example
that accumulates multi-point prompts and shows load-progress/cache UI).

## Browser support

- **WebGPU** (best): recent desktop Chrome/Edge. `device: 'auto'` (the
  default) picks this when available.
- **WASM fallback**: works broadly, slower, needs COOP/COEP for
  multithreading (see above).
- **iOS Safari**: experimental — EdgeTAM only, WASM only, expect it to be
  slow and possibly memory-constrained.
