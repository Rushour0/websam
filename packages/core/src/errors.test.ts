import { describe, expect, it } from 'vitest';
import {
  CrossOriginIsolationRequiredError,
  EpochInvalidatedError,
  InvalidStateError,
  NotImplementedError,
  OutOfMemoryError,
  UnsupportedDeviceError,
  WebsamError,
  WeightVerifyError,
} from './errors.js';

describe('error taxonomy', () => {
  const cases = [
    { make: () => new NotImplementedError('thing, lands in M1'), code: 'NOT_IMPLEMENTED', name: 'NotImplementedError' },
    { make: () => new UnsupportedDeviceError('no webgpu'), code: 'UNSUPPORTED_DEVICE', name: 'UnsupportedDeviceError' },
    { make: () => new CrossOriginIsolationRequiredError('need COOP/COEP'), code: 'CROSS_ORIGIN_ISOLATION_REQUIRED', name: 'CrossOriginIsolationRequiredError' },
    { make: () => new WeightVerifyError('digest mismatch'), code: 'WEIGHT_VERIFY_FAILED', name: 'WeightVerifyError' },
    { make: () => new OutOfMemoryError('alloc failed'), code: 'OUT_OF_MEMORY', name: 'OutOfMemoryError' },
    { make: () => new EpochInvalidatedError('stale epoch'), code: 'EPOCH_INVALIDATED', name: 'EpochInvalidatedError' },
    { make: () => new InvalidStateError('disposed'), code: 'INVALID_STATE', name: 'InvalidStateError' },
  ] as const;

  it.each(cases)('$name extends WebsamError with code $code', ({ make, code, name }) => {
    const err = make();
    expect(err).toBeInstanceOf(WebsamError);
    expect(err).toBeInstanceOf(Error);
    expect(err.code).toBe(code);
    expect(err.name).toBe(name);
    expect(err.message.length).toBeGreaterThan(0);
  });

  it('codes are unique across the taxonomy', () => {
    const codes = cases.map((c) => c.make().code);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it('NotImplementedError formats the "what" into the message', () => {
    const err = new NotImplementedError('createSegmenter, lands in M1');
    expect(err.message).toBe('Not implemented: createSegmenter, lands in M1');
  });

  it('supports error cause chaining', () => {
    const cause = new Error('boom');
    const err = new WeightVerifyError('digest mismatch', { cause });
    expect(err.cause).toBe(cause);
  });

  it('code can discriminate without instanceof', () => {
    const err: unknown = new UnsupportedDeviceError('x');
    // Simulates matching across duplicated bundles: only structural checks.
    const code = (err as { code?: string }).code;
    expect(code).toBe('UNSUPPORTED_DEVICE');
  });
});
