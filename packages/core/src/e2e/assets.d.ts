/**
 * Vite `?url` asset imports used by the e2e browser gate: the committed
 * golden scene PNG lives outside the package root (tools/goldens/fixtures)
 * and is imported as a served URL, which also anchors the URLs of its
 * sibling RLE fixtures and the model cache. `src/assets.d.ts` already
 * declares `*.onnx?url`; this covers the PNG.
 */
declare module '*.png?url' {
  const url: string;
  export default url;
}
