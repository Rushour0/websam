/**
 * Typed-error transport across the Comlink boundary.
 *
 * Structured clone drops custom own-properties of `Error`, and Comlink's
 * built-in `'throw'` transfer handler keeps only name/message/stack — so
 * {@link WebsamError.code} would not survive the worker → main hop. The fix
 * (docs/m1-internal-contracts.md §3.3) is to OVERRIDE Comlink's `'throw'`
 * transfer handler on BOTH sides ({@link installErrorTransferHandler} is
 * called once per realm: `worker-entry.ts` and `segmenter-impl.ts`).
 *
 * Serialize: a thrown `WebsamError` becomes an {@link ErrorEnvelope} carrying
 * `websamCode`. Deserialize: `websamCode` rehydrates through a
 * code → constructor map covering every {@link WebsamErrorCode} (message and
 * stack preserved); an envelope without a code becomes a plain `Error`.
 *
 * DOMException `AbortError` never crosses the boundary — abort is handled
 * main-side (§4.3).
 */

import {
  CrossOriginIsolationRequiredError,
  EpochInvalidatedError,
  InvalidStateError,
  NotImplementedError,
  OutOfMemoryError,
  UnsupportedDeviceError,
  WebsamError,
  WeightVerifyError,
  type WebsamErrorCode,
} from '../errors.js';

/** Wire form of a thrown value: structured-clone-safe, code-preserving. */
export interface ErrorEnvelope {
  name: string;
  message: string;
  stack?: string;
  /** Present iff the thrown value was a {@link WebsamError}. */
  websamCode?: WebsamErrorCode;
}

/**
 * The prefix {@link NotImplementedError}'s constructor prepends. Stripped
 * before rehydration so the constructor re-adding it keeps the message an
 * exact round-trip.
 */
const NOT_IMPLEMENTED_PREFIX = 'Not implemented: ';

/**
 * code → constructor map covering every {@link WebsamErrorCode}. Each entry
 * rebuilds the concrete class so both `instanceof` and `.code` matching work
 * on the receiving side, with the message preserved verbatim.
 */
const REHYDRATORS: Record<WebsamErrorCode, (message: string) => WebsamError> = {
  NOT_IMPLEMENTED: (message) =>
    new NotImplementedError(
      message.startsWith(NOT_IMPLEMENTED_PREFIX)
        ? message.slice(NOT_IMPLEMENTED_PREFIX.length)
        : message,
    ),
  UNSUPPORTED_DEVICE: (message) => new UnsupportedDeviceError(message),
  CROSS_ORIGIN_ISOLATION_REQUIRED: (message) => new CrossOriginIsolationRequiredError(message),
  WEIGHT_VERIFY_FAILED: (message) => new WeightVerifyError(message),
  OUT_OF_MEMORY: (message) => new OutOfMemoryError(message),
  EPOCH_INVALIDATED: (message) => new EpochInvalidatedError(message),
  INVALID_STATE: (message) => new InvalidStateError(message),
};

/**
 * Serialize any thrown value into a structured-clone-safe envelope.
 * Non-`Error` throws (strings, objects) become a plain-`Error` envelope with
 * a stringified message.
 */
export function errorToEnvelope(value: unknown): ErrorEnvelope {
  if (value instanceof Error) {
    const envelope: ErrorEnvelope = { name: value.name, message: value.message };
    if (value.stack !== undefined) envelope.stack = value.stack;
    if (value instanceof WebsamError) envelope.websamCode = value.code;
    return envelope;
  }
  return { name: 'Error', message: String(value) };
}

/**
 * Rehydrate an envelope into a throwable error: known `websamCode` → the
 * concrete {@link WebsamError} subclass; no code → plain `Error`. Name and
 * stack from the envelope always win (they describe the ORIGINAL throw site
 * in the other realm).
 */
export function envelopeToError(envelope: ErrorEnvelope): Error {
  const error =
    envelope.websamCode !== undefined && Object.hasOwn(REHYDRATORS, envelope.websamCode)
      ? REHYDRATORS[envelope.websamCode](envelope.message)
      : new Error(envelope.message);
  error.name = envelope.name;
  if (envelope.stack !== undefined) error.stack = envelope.stack;
  return error;
}

/**
 * Comlink wraps every thrown value in `{ value, [Symbol('Comlink.thrown')]: 0 }`
 * before consulting the `'throw'` transfer handler. The marker symbol is
 * module-private to comlink, so the replacement handler detects the wrapper
 * by the symbol's description.
 */
function isThrowWrapper(value: unknown): value is { value: unknown } {
  return (
    typeof value === 'object' &&
    value !== null &&
    Object.getOwnPropertySymbols(value).some((s) => s.description === 'Comlink.thrown')
  );
}

/**
 * Replace Comlink's built-in `'throw'` transfer handler with the
 * code-preserving envelope codec. Call ONCE PER REALM, before any
 * `Comlink.expose` / `Comlink.wrap` traffic: `worker-entry.ts` (worker side)
 * AND `segmenter-impl.ts` (main-thread side). Idempotent — repeat calls just
 * re-set the same handler.
 *
 * @param comlink - The realm's comlink module, injected so both sides
 * install onto the exact module instance their endpoints use.
 */
export function installErrorTransferHandler(comlink: typeof import('comlink')): void {
  const handler: import('comlink').TransferHandler<{ value: unknown }, ErrorEnvelope> = {
    canHandle: isThrowWrapper,
    serialize(wrapper) {
      return [errorToEnvelope(wrapper.value), []];
    },
    deserialize(envelope) {
      throw envelopeToError(envelope);
    },
  };
  comlink.transferHandlers.set(
    'throw',
    handler as unknown as import('comlink').TransferHandler<unknown, unknown>,
  );
}
