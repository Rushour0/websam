/**
 * Typed error taxonomy for @websam3/core.
 *
 * Every error thrown by websam extends {@link WebsamError} and carries a
 * stable string {@link WebsamError.code} discriminant, so callers can switch
 * on `err.code` without relying on `instanceof` across bundle boundaries.
 */

/** Discriminant codes for every error class in the websam taxonomy. */
export type WebsamErrorCode =
  | 'NOT_IMPLEMENTED'
  | 'UNSUPPORTED_DEVICE'
  | 'CROSS_ORIGIN_ISOLATION_REQUIRED'
  | 'WEIGHT_VERIFY_FAILED'
  | 'OUT_OF_MEMORY'
  | 'EPOCH_INVALIDATED'
  | 'INVALID_STATE';

/**
 * Base class for all websam errors.
 *
 * Prefer matching on {@link WebsamError.code} (survives multiple copies of
 * the library in one page) over `instanceof` where robustness matters.
 */
export class WebsamError extends Error {
  /** Stable machine-readable discriminant for this error class. */
  readonly code: WebsamErrorCode;

  constructor(code: WebsamErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
    this.code = code;
  }
}

/**
 * Thrown when calling API surface that is typed and contracted but whose
 * runtime lands in a later milestone. The message always names the member
 * and the milestone, e.g. `'createSegmenter, lands in M1'`.
 */
export class NotImplementedError extends WebsamError {
  constructor(what: string, options?: ErrorOptions) {
    super('NOT_IMPLEMENTED', `Not implemented: ${what}`, options);
  }
}

/**
 * Thrown when the requested execution device cannot run at all in the
 * current environment — e.g. `device: 'webgpu'` was forced but
 * `navigator.gpu` is absent or no adapter is available, or WebAssembly is
 * disabled. With `device: 'auto'` websam degrades instead of throwing this.
 */
export class UnsupportedDeviceError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('UNSUPPORTED_DEVICE', message, options);
  }
}

/**
 * Thrown when a feature requires `crossOriginIsolated === true` (COOP/COEP
 * headers) and the page is not isolated — e.g. multi-threaded WASM with
 * SharedArrayBuffer was explicitly requested. The WASM backend's automatic
 * single-thread fallback never throws this; only explicit thread requests do.
 */
export class CrossOriginIsolationRequiredError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('CROSS_ORIGIN_ISOLATION_REQUIRED', message, options);
  }
}

/**
 * Thrown when a downloaded weight/graph file fails integrity verification
 * against the digest pinned in the model manifest (corrupt CDN response,
 * truncated download, or tampering). The file is discarded, never cached.
 */
export class WeightVerifyError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('WEIGHT_VERIFY_FAILED', message, options);
  }
}

/**
 * Thrown when a device or host allocation fails — GPU buffer allocation is
 * rejected, the WASM heap cannot grow, or the memory-bank ring cannot fit
 * the requested number of slots. Callers may retry with a smaller model,
 * lower resolution, or fewer tracked objects.
 */
export class OutOfMemoryError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('OUT_OF_MEMORY', message, options);
  }
}

/**
 * Thrown when an async result is delivered for a stale interaction epoch —
 * the user reset the session, seeked the video, or changed prompts while an
 * inference was in flight, so its result must not be applied. Callers should
 * silently drop the result and await the current epoch instead.
 */
export class EpochInvalidatedError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('EPOCH_INVALIDATED', message, options);
  }
}

/**
 * Thrown when an object is used in a state that cannot service the call —
 * running inference on a disposed session, initializing a backend twice,
 * registering a duplicate model id, or reading back a disposed tensor.
 */
export class InvalidStateError extends WebsamError {
  constructor(message: string, options?: ErrorOptions) {
    super('INVALID_STATE', message, options);
  }
}
