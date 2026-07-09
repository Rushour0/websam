# websam

**SAM-family interactive image & video segmentation in the browser (WebGPU-first, WASM fallback).**

> **Status: pre-alpha.** The full interactive image path (M1) and in-browser EdgeTAM video
> tracking (M2) are real and gated (real-weights parity IoU 0.989–0.995 across 10 frames on the
> video path). Packages are published to npm under `@websam3/*@0.0.1`. What's still missing:
> **hosted model weights** — there is no public URL to fetch weights from yet, so the published
> packages run but can't load a model out of the box (see [Providing weights](#providing-weights)
> below). Tracked for milestone M4.

websam brings the Segment Anything Model family to the web platform:

- **EdgeTAM** (Apache-2.0) is the **default model tier** — a lightweight, on-device track-anything
  model suited to browser deployment, with real-time video tracking.
- **SAM 3 Tracker** is an **opt-in tier** for users who accept the SAM License terms and the
  larger download; it currently backs the interactive-image path in the demo.
- To our knowledge, **nobody has shipped in-browser SAM-family video tracking before** — this
  project is an attempt to be the first, and the engineering risk that implies is real.

## Packages

| Package | npm | Description |
| --- | --- | --- |
| [`@websam3/core`](packages/core) | [`@websam3/core`](https://www.npmjs.com/package/@websam3/core) | Model runtime: `createSegmenter`, `ImageSession`/`VideoSession`, backends (WebGPU / WASM), mask decoding, the inference worker. |
| [`@websam3/video-editing`](packages/video-editing) | [`@websam3/video-editing`](https://www.npmjs.com/package/@websam3/video-editing) | `MaskTimeline` + `AlphaMatteExporter`: per-frame mask storage and alpha-matte export. |
| [`@websam3/react`](packages/react) | [`@websam3/react`](https://www.npmjs.com/package/@websam3/react) | React bindings: `useSegmenter`, `useImageSession`, `useVideoSession`. |

## Getting started

```sh
npm install @websam3/core
```

See [`examples/quickstart`](examples/quickstart) for a minimal, standalone Vite + TypeScript app
(no framework) that runs this end to end — the fastest way to see the real API in ~150 lines. It
also has the full walkthrough for **providing model weights locally** (required until M4).

### Providing weights

`createSegmenter`'s `modelBaseUrl` must point at a served model manifest + weight files — there is
no public host yet. Until M4 ships hosted weights, either:

- generate EdgeTAM weights locally: `cd tools/goldens && ../export/.venv/bin/python make-video-golden.py`,
  then copy `tools/goldens/models-cache/edgetam/` into your app's static assets, or
- reuse the M1 image (SAM 3 Tracker) weights via `tools/goldens/fetch-models.mjs`.

Full instructions: [`examples/quickstart/README.md`](examples/quickstart/README.md#providing-weights).

## Usage

### Interactive image segmentation

```ts
import { createSegmenter } from '@websam3/core';
import segmenterWorkerUrl from '@websam3/core/worker?worker&url'; // bundler escape hatch

const segmenter = await createSegmenter({
  model: 'edgetam',
  modelBaseUrl: '/models/', // see "Providing weights"
  workerUrl: segmenterWorkerUrl,
});

const session = await segmenter.createImageSession();
await session.encode(imageBitmapOrCanvas); // run the vision encoder once

const [mask] = await session.decode([{ type: 'point', x: 180, y: 210, label: 1 }]);
const imageData = mask.toImageData(); // or mask.toBinary() / mask.toRLE() / mask.toCocoRLE()
```

### Video object tracking

```ts
const segmenter = await createSegmenter({ model: 'edgetam', modelBaseUrl: '/models/' });
const session = await segmenter.createVideoSession();

await session.attachSource(videoBlob); // returns { frameCount, fps, width, height }
await session.addObject({ frameIndex: 0, prompts: [{ type: 'point', x: 180, y: 210, label: 1 }] });

for await (const { frameIndex, masks } of session.propagate()) {
  for (const mask of masks) {
    // draw mask.toImageData() for this frame, store mask.toRLE(), etc.
  }
}
```

`refineObject(objectId, frameIndex, prompts)` re-prompts an already-tracked object at a given
frame; `propagate()`'s iterator throws `EpochInvalidatedError` on its next tick if a refinement
lands mid-iteration (never a silent stop).

### React

```tsx
import { useSegmenter } from '@websam3/react';
import { useEffect, useState } from 'react';
import type { ImageSession } from '@websam3/core';

function Segmenter() {
  const { segmenter, status } = useSegmenter({ model: 'edgetam', modelBaseUrl: '/models/' });
  // status: 'idle' | 'loading' | 'ready' | 'error'
  const [session, setSession] = useState<ImageSession | null>(null);

  useEffect(() => {
    if (status !== 'ready' || !segmenter) return;
    let cancelled = false;
    void segmenter.createImageSession().then((s) => !cancelled && setSession(s));
    return () => {
      cancelled = true;
    };
  }, [segmenter, status]);
  // then session.encode(...) / session.decode(...) as in the vanilla example above
}
```

`useSegmenter` is the real, implemented hook (with request de-duplication and abort-on-unmount).
`useImageSession`/`useVideoSession` are still stubs in `@websam3/react` — they throw
`NotImplementedError` today; call `segmenter.createImageSession()` / `createVideoSession()`
directly as shown, per `apps/demo/src/{ImageTab,VideoTab}.tsx`.

### Mask timeline + alpha-matte export

```ts
import { MaskTimeline, AlphaMatteExporter } from '@websam3/video-editing';

const timeline = new MaskTimeline({ frameCount, fps, width, height });
for await (const { frameIndex, masks } of session.propagate()) {
  for (const mask of masks) timeline.set(String(mask.objectId), frameIndex, mask.toRLE());
}

const exporter = new AlphaMatteExporter(timeline);
const { blob } = await exporter.export({ mode: 'matte', format: 'png-sequence' });
// mode: 'cutout' and format: 'webm-vp9-alpha' land in M4; 'matte' + 'png-sequence' work today.
```

## Monorepo layout

```
.
├── packages/
│   ├── core/            # @websam3/core
│   ├── video-editing/   # @websam3/video-editing
│   └── react/            # @websam3/react
├── examples/
│   └── quickstart/       # minimal standalone Vite + TS example (not published)
├── apps/                 # demo + bundler-matrix apps (not published)
├── tools/                 # model export tooling (Python, ONNX) + golden fixtures
├── tsconfig.base.json
└── pnpm-workspace.yaml
```

- **Package manager:** pnpm 10 workspaces, Node >= 20 (repo development uses Node 22).
- **Build:** tsdown (ESM-only, `dist/` with bundled `.d.ts`).
- **Tests:** vitest (unit + browser projects), plus e2e golden gates against real model weights.

```sh
pnpm install
pnpm build
pnpm test
```

## License matrix

Three distinct licenses apply depending on what you use:

| Layer | License | Notes |
| --- | --- | --- |
| All code in this repo | **MIT** | See [LICENSE](LICENSE). |
| Default model weights (EdgeTAM) | **Apache-2.0** | Downloaded at runtime; permissive. |
| Opt-in model weights (SAM3) | **SAM License** | You must explicitly opt in (`acceptLicense: 'sam'`) and accept Meta's SAM License terms. Not fetched by default. |

Weights are never bundled into the npm packages; only MIT-licensed code is published.

## Roadmap

- **M4** — hosted model weights (make the published packages runnable out of the box), WebGPU
  compositor, VP9-alpha + cutout export, `0.1.0`.
- **M3** — SAM3-tracker video tier.
- **M5** — [fabri](https://github.com/Rushour0) integration via `OrtNodeBackend` + VLM text→prompt.

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the detailed milestone log.
