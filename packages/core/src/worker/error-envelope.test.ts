import * as Comlink from 'comlink';
import { describe, expect, it } from 'vitest';
import {
  CrossOriginIsolationRequiredError,
  EpochInvalidatedError,
  InvalidStateError,
  NotImplementedError,
  OutOfMemoryError,
  UnsupportedDeviceError,
  WeightVerifyError,
  type WebsamErrorCode,
} from '../errors.js';
import {
  envelopeToError,
  errorToEnvelope,
  installErrorTransferHandler,
  type ErrorEnvelope,
} from './error-envelope.js';

/** One representative instance per WebsamErrorCode (exhaustive rehydration coverage). */
const SAMPLES: { error: Error; code: WebsamErrorCode; ctor: new (m: string) => Error }[] = [
  { error: new NotImplementedError('foo, lands in M2'), code: 'NOT_IMPLEMENTED', ctor: NotImplementedError },
  { error: new UnsupportedDeviceError('no adapter'), code: 'UNSUPPORTED_DEVICE', ctor: UnsupportedDeviceError },
  {
    error: new CrossOriginIsolationRequiredError('needs COOP/COEP'),
    code: 'CROSS_ORIGIN_ISOLATION_REQUIRED',
    ctor: CrossOriginIsolationRequiredError,
  },
  { error: new WeightVerifyError('digest mismatch'), code: 'WEIGHT_VERIFY_FAILED', ctor: WeightVerifyError },
  { error: new OutOfMemoryError('heap grow failed'), code: 'OUT_OF_MEMORY', ctor: OutOfMemoryError },
  { error: new EpochInvalidatedError('stale epoch'), code: 'EPOCH_INVALIDATED', ctor: EpochInvalidatedError },
  { error: new InvalidStateError('used after dispose'), code: 'INVALID_STATE', ctor: InvalidStateError },
];

describe('errorToEnvelope / envelopeToError round-trip', () => {
  it.each(SAMPLES)('preserves class, code, name, message and stack for $code', (sample) => {
    const envelope = errorToEnvelope(sample.error);
    expect(envelope).toMatchObject({
      name: sample.error.name,
      message: sample.error.message,
      websamCode: sample.code,
    });
    expect(envelope.stack).toBe(sample.error.stack);

    const back = envelopeToError(envelope);
    expect(back).toBeInstanceOf(sample.ctor);
    expect(back.name).toBe(sample.error.name);
    expect(back.message).toBe(sample.error.message);
    expect(back.stack).toBe(sample.error.stack);
    expect((back as { code?: string }).code).toBe(sample.code);
  });

  it('round-trips the NotImplementedError message exactly (no double prefix)', () => {
    const original = new NotImplementedError('mask prompts, lands in M2');
    const back = envelopeToError(errorToEnvelope(original));
    expect(back.message).toBe('Not implemented: mask prompts, lands in M2');
    const twice = envelopeToError(errorToEnvelope(back));
    expect(twice.message).toBe(back.message);
  });

  it('serializes a plain Error without a websamCode and rehydrates a plain Error', () => {
    const original = new TypeError('failed to fetch');
    const envelope = errorToEnvelope(original);
    expect(envelope.websamCode).toBeUndefined();
    expect(envelope).toMatchObject({ name: 'TypeError', message: 'failed to fetch' });

    const back = envelopeToError(envelope);
    expect(back).toBeInstanceOf(Error);
    expect(back.name).toBe('TypeError');
    expect(back.message).toBe('failed to fetch');
    expect((back as { code?: string }).code).toBeUndefined();
  });

  it('stringifies non-Error thrown values', () => {
    expect(errorToEnvelope('boom')).toEqual({ name: 'Error', message: 'boom' });
    expect(errorToEnvelope(42)).toEqual({ name: 'Error', message: '42' });
    const back = envelopeToError(errorToEnvelope('boom'));
    expect(back).toBeInstanceOf(Error);
    expect(back.message).toBe('boom');
  });

  it('tolerates an envelope without a stack', () => {
    const envelope: ErrorEnvelope = {
      name: 'WeightVerifyError',
      message: 'bad digest',
      websamCode: 'WEIGHT_VERIFY_FAILED',
    };
    const back = envelopeToError(envelope);
    expect(back).toBeInstanceOf(WeightVerifyError);
    // The freshly constructed error keeps its own capture-site stack.
    expect(typeof back.stack).toBe('string');
  });
});

describe('installErrorTransferHandler', () => {
  /** Comlink's internal wrapper shape: `{ value, [Symbol('Comlink.thrown')]: 0 }`. */
  function throwWrapper(value: unknown): { value: unknown } {
    return Object.assign(Object.create(null) as object, {
      value,
      [Symbol('Comlink.thrown')]: 0,
    }) as { value: unknown };
  }

  it("replaces comlink's 'throw' handler in place", () => {
    const before = Comlink.transferHandlers.get('throw');
    expect(before).toBeDefined();
    installErrorTransferHandler(Comlink);
    const after = Comlink.transferHandlers.get('throw');
    expect(after).toBeDefined();
    expect(after).not.toBe(before);
    // Idempotent: a second install just re-sets the handler.
    installErrorTransferHandler(Comlink);
    expect(Comlink.transferHandlers.get('throw')).toBeDefined();
  });

  it('canHandle recognizes comlink throw-wrappers and nothing else', () => {
    installErrorTransferHandler(Comlink);
    const handler = Comlink.transferHandlers.get('throw')!;
    expect(handler.canHandle(throwWrapper(new Error('x')))).toBe(true);
    expect(handler.canHandle({ value: new Error('x') })).toBe(false);
    expect(handler.canHandle(new Error('x'))).toBe(false);
    expect(handler.canHandle(null)).toBe(false);
    expect(handler.canHandle('boom')).toBe(false);
  });

  it('serialize → deserialize rethrows the typed error with its code intact', () => {
    installErrorTransferHandler(Comlink);
    const handler = Comlink.transferHandlers.get('throw')!;
    const original = new WeightVerifyError('sha256 mismatch for encoder.onnx');

    const [wire, transfers] = handler.serialize(throwWrapper(original));
    expect(transfers).toEqual([]);
    // The wire value must be a structured-clone-safe plain envelope.
    expect(wire).toMatchObject({
      name: 'WeightVerifyError',
      message: 'sha256 mismatch for encoder.onnx',
      websamCode: 'WEIGHT_VERIFY_FAILED',
    });

    let caught: unknown;
    try {
      handler.deserialize(wire);
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(WeightVerifyError);
    expect(caught).toMatchObject({
      name: 'WeightVerifyError',
      code: 'WEIGHT_VERIFY_FAILED',
      message: 'sha256 mismatch for encoder.onnx',
    });
    expect((caught as Error).stack).toBe(original.stack);
  });

  it('deserializes an envelope without a code into a plain Error', () => {
    installErrorTransferHandler(Comlink);
    const handler = Comlink.transferHandlers.get('throw')!;
    const [wire] = handler.serialize(throwWrapper(new RangeError('bad dims')));
    let caught: unknown;
    try {
      handler.deserialize(wire);
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(caught).toMatchObject({ name: 'RangeError', message: 'bad dims' });
    expect((caught as { code?: string }).code).toBeUndefined();
  });
});
