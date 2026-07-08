/**
 * Incremental (streaming) SHA-256 in pure TypeScript.
 *
 * WHY NOT WebCrypto: `crypto.subtle.digest` has no incremental API — hashing
 * a ~300 MB encoder file with it would mean buffering the entire download
 * just to verify it, defeating streaming. This implementation hashes chunk
 * by chunk as bytes arrive (throughput is overlapped with the network, so
 * pure-TS speed is acceptable), per FIPS 180-4.
 *
 * Unit-tested against the FIPS known-answer vectors and cross-checked
 * against `node:crypto` on random inputs.
 */

import { InvalidStateError } from '../errors.js';

/** Per-round constants: first 32 bits of the fractional parts of the cube roots of the first 64 primes. */
// prettier-ignore
const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Initial hash state: first 32 bits of the fractional parts of the square roots of the first 8 primes. */
// prettier-ignore
const H_INIT = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

const BLOCK_BYTES = 64;
const HEX = '0123456789abcdef';

/**
 * Incremental SHA-256 (pure TS, ~streaming 50–100 MB/s — overlapped with
 * network, acceptable). Feed chunks with {@link update}, finish with
 * {@link digestHex}; the instance is single-use.
 */
export class Sha256Stream {
  /** Current hash state (a…h). */
  readonly #h = new Uint32Array(H_INIT);
  /** Message-schedule scratch, reused across blocks. */
  readonly #w = new Uint32Array(64);
  /** Holds a partial block between update() calls. */
  readonly #buffer = new Uint8Array(BLOCK_BYTES);
  #bufferLen = 0;
  /** Total bytes hashed so far (exact up to 2^53 — far beyond any weight file). */
  #totalBytes = 0;
  #dead = false;

  /** Absorb the next chunk of the message. Throws {@link InvalidStateError} after {@link digestHex}. */
  update(chunk: Uint8Array): void {
    if (this.#dead) {
      throw new InvalidStateError('Sha256Stream: update() after digestHex()');
    }
    this.#totalBytes += chunk.byteLength;
    let offset = 0;

    // Top up a pending partial block first.
    if (this.#bufferLen > 0) {
      const take = Math.min(BLOCK_BYTES - this.#bufferLen, chunk.byteLength);
      this.#buffer.set(chunk.subarray(0, take), this.#bufferLen);
      this.#bufferLen += take;
      offset = take;
      if (this.#bufferLen === BLOCK_BYTES) {
        this.#processBlock(this.#buffer, 0);
        this.#bufferLen = 0;
      }
    }

    // Whole blocks straight from the chunk, no copy.
    while (offset + BLOCK_BYTES <= chunk.byteLength) {
      this.#processBlock(chunk, offset);
      offset += BLOCK_BYTES;
    }

    // Stash the tail.
    if (offset < chunk.byteLength) {
      this.#buffer.set(chunk.subarray(offset), 0);
      this.#bufferLen = chunk.byteLength - offset;
    }
  }

  /**
   * Finalize and return the lowercase hex digest; the instance is dead
   * afterwards (any further call throws {@link InvalidStateError}).
   */
  digestHex(): string {
    if (this.#dead) {
      throw new InvalidStateError('Sha256Stream: digestHex() called twice');
    }
    this.#dead = true;

    // Padding: 0x80, zeros, then the 64-bit big-endian bit length.
    const bitLenHi = Math.floor(this.#totalBytes / 0x20000000); // totalBytes / 2^29 = (totalBytes*8) / 2^32
    const bitLenLo = (this.#totalBytes << 3) >>> 0;
    const block = this.#buffer;
    let len = this.#bufferLen;
    block[len++] = 0x80;
    if (len > BLOCK_BYTES - 8) {
      block.fill(0, len);
      this.#processBlock(block, 0);
      len = 0;
    }
    block.fill(0, len, BLOCK_BYTES - 8);
    block[56] = (bitLenHi >>> 24) & 0xff;
    block[57] = (bitLenHi >>> 16) & 0xff;
    block[58] = (bitLenHi >>> 8) & 0xff;
    block[59] = bitLenHi & 0xff;
    block[60] = (bitLenLo >>> 24) & 0xff;
    block[61] = (bitLenLo >>> 16) & 0xff;
    block[62] = (bitLenLo >>> 8) & 0xff;
    block[63] = bitLenLo & 0xff;
    this.#processBlock(block, 0);

    let hex = '';
    for (let i = 0; i < 8; i++) {
      const word = this.#h[i] ?? 0;
      for (let shift = 28; shift >= 0; shift -= 4) {
        hex += HEX[(word >>> shift) & 0xf];
      }
    }
    return hex;
  }

  /** FIPS 180-4 §6.2.2 compression of one 64-byte block at `offset` in `data`. */
  #processBlock(data: Uint8Array, offset: number): void {
    const w = this.#w;
    const h = this.#h;

    for (let i = 0; i < 16; i++) {
      const j = offset + i * 4;
      w[i] =
        ((data[j] ?? 0) << 24) |
        ((data[j + 1] ?? 0) << 16) |
        ((data[j + 2] ?? 0) << 8) |
        (data[j + 3] ?? 0);
    }
    for (let i = 16; i < 64; i++) {
      const w15 = w[i - 15] ?? 0;
      const w2 = w[i - 2] ?? 0;
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      w[i] = ((w[i - 16] ?? 0) + s0 + (w[i - 7] ?? 0) + s1) | 0;
    }

    let a = h[0] ?? 0;
    let b = h[1] ?? 0;
    let c = h[2] ?? 0;
    let d = h[3] ?? 0;
    let e = h[4] ?? 0;
    let f = h[5] ?? 0;
    let g = h[6] ?? 0;
    let hh = h[7] ?? 0;

    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + (K[i] ?? 0) + (w[i] ?? 0)) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) | 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }

    h[0] = (a + (h[0] ?? 0)) | 0;
    h[1] = (b + (h[1] ?? 0)) | 0;
    h[2] = (c + (h[2] ?? 0)) | 0;
    h[3] = (d + (h[3] ?? 0)) | 0;
    h[4] = (e + (h[4] ?? 0)) | 0;
    h[5] = (f + (h[5] ?? 0)) | 0;
    h[6] = (g + (h[6] ?? 0)) | 0;
    h[7] = (hh + (h[7] ?? 0)) | 0;
  }
}
