import { describe, expect, it } from 'vitest';
import { float16BitsToFloat32, float32ToFloat16Bits } from './float16.js';

describe('float32ToFloat16Bits / float16BitsToFloat32 round trip', () => {
  it('round-trips exactly-representable values', () => {
    const values = Float32Array.from([0, 1, -1, 0.5, -0.5, 2, 100, -100, 1234]);
    const back = float16BitsToFloat32(float32ToFloat16Bits(values));
    expect(Array.from(back)).toEqual(Array.from(values));
  });

  it('encodes signed zero, +/-Infinity, and NaN', () => {
    const values = Float32Array.from([0, -0, Infinity, -Infinity, NaN]);
    const bits = float32ToFloat16Bits(values);
    expect(bits[0]).toBe(0x0000);
    expect(bits[1]).toBe(0x8000);
    expect(bits[2]).toBe(0x7c00);
    expect(bits[3]).toBe(0xfc00);
    expect(bits[4]! & 0x7c00).toBe(0x7c00); // NaN: exponent all-ones

    const back = float16BitsToFloat32(bits);
    expect(back[0]).toBe(0);
    expect(Object.is(back[1], -0)).toBe(true);
    expect(back[2]).toBe(Infinity);
    expect(back[3]).toBe(-Infinity);
    expect(Number.isNaN(back[4])).toBe(true);
  });

  it('rounds values beyond half precision to the nearest representable half', () => {
    const back = float16BitsToFloat32(float32ToFloat16Bits(Float32Array.of(1.0001)));
    expect(back[0]).toBeCloseTo(1.0001, 3);
    expect(back[0]).not.toBe(1.0001);
  });

  it('flushes sub-half-precision magnitudes toward zero/subnormal without throwing', () => {
    const back = float16BitsToFloat32(float32ToFloat16Bits(Float32Array.of(1e-8, -1e-8)));
    expect(back[0]).toBeCloseTo(0, 5);
    expect(back[1]).toBeCloseTo(0, 5);
  });

  it('saturates magnitudes beyond half range to +/-Infinity', () => {
    const back = float16BitsToFloat32(float32ToFloat16Bits(Float32Array.of(1e10, -1e10)));
    expect(back[0]).toBe(Infinity);
    expect(back[1]).toBe(-Infinity);
  });
});
