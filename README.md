# websam

**SAM-family interactive image & video segmentation in the browser (WebGPU-first, WASM fallback).**

> **Status: pre-alpha scaffold.** The public API surface exists and is typed, but the heavy
> runtime paths (model loading, inference, video tracking) are not implemented yet — they throw
> `NotImplementedError` with the milestone they land in. Do not depend on this for anything real yet.

websam brings the Segment Anything Model family to the web platform:

- **EdgeTAM** (Apache-2.0) is the **default model tier** — a lightweight, on-device track-anything
  model suited to browser deployment.
- **SAM3** is an **opt-in tier** for users who accept the SAM License terms and the larger download.
- To our knowledge, **nobody has shipped in-browser SAM-family video tracking before** — this
  project is an attempt to be the first, and the engineering risk that implies is real.

## Packages

| Package | Description |
| --- | --- |
| [`@websam3/core`](packages/core) | Model runtime: session management, image/video predictors, backends (WebGPU / WASM), mask decoding. |
| [`@websam3/video-editing`](packages/video-editing) | Video-editing utilities on top of core: tracklets, matting-oriented mask post-processing, export helpers. |
| [`@websam3/react`](packages/react) | React bindings: hooks and components for interactive segmentation UIs. |

## Monorepo layout

```
.
├── packages/
│   ├── core/            # @websam3/core
│   ├── video-editing/   # @websam3/video-editing
│   └── react/           # @websam3/react
├── apps/                # demo + bundler-matrix apps (not published)
├── tools/               # model export tooling (Python, ONNX)
├── tsconfig.base.json
└── pnpm-workspace.yaml
```

- **Package manager:** pnpm 10 workspaces, Node >= 20.
- **Build:** tsdown (ESM-only, `dist/` with bundled `.d.ts`).
- **Tests:** vitest.

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
| Opt-in model weights (SAM3) | **SAM License** | You must explicitly opt in and accept Meta's SAM License terms. Not fetched by default. |

Weights are never bundled into the npm packages; only MIT-licensed code is published.

## Publishing note

The npm scope **`@websam` must be claimed before the first publish**. Until then, package names
in this repo are provisional.
