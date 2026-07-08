# bundler-matrix

Tiny consumer apps that import `@websam/core` and **build in CI**, so that
worker/asset packaging regressions (broken `exports` map, non-ESM output,
tree-shaking metadata, and later worker-entry/wasm asset resolution) fail a
build here instead of in downstream users' projects.

Each app is a private workspace package with a single `build` script. None of
them are meant to be deployed; a green build **is** the test.

## Matrix

| App        | Bundler          | Entry             | What it exercises                                         |
| ---------- | ---------------- | ----------------- | --------------------------------------------------------- |
| `vite/`    | Vite 8           | `src/main.ts`     | `exports` map resolution, ESM output, esbuild transform   |
| `webpack/` | Webpack 5        | `src/main.ts`     | `exports` map resolution via enhanced-resolve, ts-loader  |

Both entries do a **real value import** (`WebGpuBackend`, `listModels`) and
render the result to the DOM, so the imports cannot be elided as type-only and
a broken package surface fails the bundle step.

At **M1**, when the `@websam/core` worker entry lands, a Next.js app joins the
matrix (worker URL resolution + `new Worker(new URL(...))` packaging is the
main thing bundlers disagree on).

## Running locally

```sh
pnpm --filter './apps/bundler-matrix/*' build
```

## CI wiring

The root `pnpm build` script only builds `packages/*`. CI must additionally run
the line above in the `build-and-test` job of `.github/workflows/ci.yml`
(after the `Build` step, so `@websam/core`'s `dist/` exists):

```yaml
- name: Bundler matrix
  run: pnpm --filter './apps/bundler-matrix/*' build
```

If that step is missing from `ci.yml`, these apps are not being exercised.
