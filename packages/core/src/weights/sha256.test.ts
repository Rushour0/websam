import { createHash, randomBytes } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { InvalidStateError } from '../errors.js';
import { Sha256Stream } from './sha256.js';

const encoder = new TextEncoder();

function hashOf(...chunks: Uint8Array[]): string {
  const stream = new Sha256Stream();
  for (const chunk of chunks) stream.update(chunk);
  return stream.digestHex();
}

function nodeHash(data: Uint8Array): string {
  return createHash('sha256').update(data).digest('hex');
}

describe('Sha256Stream', () => {
  it('matches the FIPS 180-4 known-answer vectors', () => {
    // Empty message.
    expect(hashOf()).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    // "abc".
    expect(hashOf(encoder.encode('abc'))).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
    // Two-block message.
    expect(hashOf(encoder.encode('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'))).toBe(
      '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
    );
  });

  it('hashes one million "a" fed in uneven chunks (FIPS long vector)', () => {
    const stream = new Sha256Stream();
    const chunk = new Uint8Array(4096 + 7).fill(0x61); // deliberately not block-aligned
    let remaining = 1_000_000;
    while (remaining > 0) {
      const take = Math.min(remaining, chunk.byteLength);
      stream.update(chunk.subarray(0, take));
      remaining -= take;
    }
    expect(stream.digestHex()).toBe(
      'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0',
    );
  });

  it('cross-checks against node:crypto across block-boundary lengths', () => {
    for (const length of [0, 1, 3, 31, 55, 56, 57, 63, 64, 65, 127, 128, 129, 1000, 65_537]) {
      const data = randomBytes(length);
      expect(hashOf(data), `length ${length}`).toBe(nodeHash(data));
    }
  });

  it('is chunking-invariant: arbitrary splits equal the one-shot digest', () => {
    const data = randomBytes(10_000);
    const whole = hashOf(data);
    expect(whole).toBe(nodeHash(data));

    for (const sizes of [[1], [7], [63], [64], [65], [1, 55, 64, 1000], [9999, 1]]) {
      const stream = new Sha256Stream();
      let offset = 0;
      let i = 0;
      while (offset < data.length) {
        const size = sizes[i % sizes.length] ?? 1;
        stream.update(data.subarray(offset, Math.min(offset + size, data.length)));
        offset += size;
        i++;
      }
      expect(stream.digestHex(), `chunk sizes ${sizes.join(',')}`).toBe(whole);
    }
  });

  it('handles chunks that are views into a larger buffer (non-zero byteOffset)', () => {
    const backing = randomBytes(256);
    const view = new Uint8Array(backing.buffer, backing.byteOffset + 13, 200);
    expect(hashOf(view)).toBe(nodeHash(Uint8Array.from(view)));
  });

  it('is single-use: update or digestHex after digestHex throws InvalidStateError', () => {
    const stream = new Sha256Stream();
    stream.update(encoder.encode('abc'));
    stream.digestHex();
    expect(() => stream.update(encoder.encode('x'))).toThrow(InvalidStateError);
    expect(() => stream.digestHex()).toThrow(InvalidStateError);
  });
});
