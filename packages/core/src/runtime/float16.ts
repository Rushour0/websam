/**
 * fp32 <-> fp16 packing for host-generated data feeding `'float16'`-dtype
 * graph inputs. onnxruntime-web's `float16` tensors carry raw half-precision
 * bits in a `Uint16Array` (see `ort-tensor.ts`'s `TYPED_ARRAY_CTORS`) — this
 * is the missing conversion between that contract and the `Float32Array`
 * host code naturally produces (pixel preprocessing, prompt coordinates,
 * zero-filled scratch tensors, ...).
 */

/** Round-to-nearest-even fp32 -> IEEE 754 half-precision bit pattern. */
function float32BitsToFloat16Bits(bits: number): number {
  const sign = (bits >>> 16) & 0x8000;
  let exp = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;

  if (exp === 0xff) {
    // Inf / NaN: preserve.
    return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }

  // Rebase the exponent from fp32 bias (127) to fp16 bias (15).
  let halfExp = exp - 127 + 15;

  if (halfExp >= 0x1f) {
    // Overflow -> infinity.
    return sign | 0x7c00;
  }

  if (halfExp <= 0) {
    if (halfExp < -10) {
      // Too small even for a subnormal half -> signed zero.
      return sign;
    }
    // Subnormal half: shift the implicit-1 mantissa right, rounding to nearest-even.
    mantissa |= 0x800000;
    const shift = 14 - halfExp;
    let half = mantissa >>> shift;
    const remainder = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (half & 1) === 1)) {
      half += 1;
    }
    return sign | half;
  }

  // Normal half: round the 23-bit mantissa down to 10 bits, nearest-even.
  let half = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  const halfway = 0x1000;
  if (remainder > halfway || (remainder === halfway && (half & 1) === 1)) {
    half += 1;
    if (half === 0x400) {
      // Mantissa rounded up to the next power of two: carry into the exponent.
      half = 0;
      halfExp += 1;
      if (halfExp >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (halfExp << 10) | half;
}

const f32Scratch = new Float32Array(1);
const f32ScratchBits = new Uint32Array(f32Scratch.buffer);

/**
 * Convert a `Float32Array` to raw half-precision bits (`Uint16Array`), ready
 * to hand `Backend.uploadTensor`/`allocTensor` as `'float16'` data.
 */
export function float32ToFloat16Bits(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    f32Scratch[0] = src[i]!;
    out[i] = float32BitsToFloat16Bits(f32ScratchBits[0]!);
  }
  return out;
}

/** IEEE 754 half-precision bit pattern -> fp32 bit pattern. */
function float16BitsToFloat32Bits(half: number): number {
  const sign = (half & 0x8000) << 16;
  const exp = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;

  if (exp === 0) {
    if (mantissa === 0) return sign; // signed zero
    // Subnormal half -> normalize into a normal fp32.
    let m = mantissa;
    let e = -1;
    do {
      e += 1;
      m <<= 1;
    } while ((m & 0x400) === 0);
    m &= 0x3ff;
    return sign | (((-14 - e + 127) & 0xff) << 23) | (m << 13);
  }
  if (exp === 0x1f) {
    // Inf / NaN.
    return sign | 0x7f800000 | (mantissa << 13);
  }
  return sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
}

/**
 * Convert raw half-precision bits (`Uint16Array`, the shape
 * `Backend.readback` hands back for a `'float16'` tensor) to a real
 * `Float32Array` of values, ready for numeric use (thresholds, sigmoids,
 * postprocessing math — anything beyond re-uploading the bits verbatim).
 */
export function float16BitsToFloat32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  const bits = new Uint32Array(out.buffer);
  for (let i = 0; i < src.length; i++) {
    bits[i] = float16BitsToFloat32Bits(src[i]!);
  }
  return out;
}
