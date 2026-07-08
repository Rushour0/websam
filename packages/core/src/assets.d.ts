/**
 * Vite `?url` asset imports used by browser tests (e.g. the committed
 * `src/__fixtures__/add.onnx` fixture). The import resolves to the served
 * URL of the asset.
 */
declare module '*.onnx?url' {
  const url: string;
  export default url;
}
