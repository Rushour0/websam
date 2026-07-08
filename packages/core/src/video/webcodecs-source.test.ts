/**
 * Node unit tests for the pure parts of the WebCodecs frame source:
 * {@link SampleTable} math (presentation⇄decode mapping, GOP-aware sync
 * lookup, timestamp bookkeeping) and {@link demuxMp4} against the committed
 * fixture — mp4box is environment-agnostic, so demux runs in node; only the
 * VideoDecoder path needs the browser lane
 * (`webcodecs-source.browser.test.ts`).
 */
import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { InvalidStateError } from '../errors.js';
import {
  SampleTable,
  createWebCodecsFrameSource,
  demuxMp4,
  type SampleTiming,
} from './webcodecs-source.js';

/** CFR all-sync track: `count` samples, `duration` ticks each. */
function cfrSamples(count: number, duration: number, syncEvery = 1): SampleTiming[] {
  return Array.from({ length: count }, (_, i) => ({
    dts: i * duration,
    cts: i * duration,
    duration,
    isSync: i % syncEvery === 0,
  }));
}

async function fixtureBytes(): Promise<ArrayBuffer> {
  const buf = await readFile(new URL('./fixtures/clip-320x180-10f.mp4', import.meta.url));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe('SampleTable', () => {
  it('maps a CFR all-sync track: identity order, exact timestamps, flat fps', () => {
    const table = new SampleTable(cfrSamples(10, 1000), 30000);
    expect(table.frameCount).toBe(10);
    expect(table.durationUs).toBe(333333);
    expect(table.fps).toBeCloseTo(30, 2);
    expect(table.timestampUs(0)).toBe(0);
    expect(table.timestampUs(1)).toBe(33333);
    expect(table.timestampUs(9)).toBe(300000);
    for (let i = 0; i < 10; i++) {
      expect(table.decodeIndexOf(i)).toBe(i);
      expect(table.syncDecodeIndexFor(i)).toBe(i);
    }
    // Monotone presentation timestamps.
    for (let i = 1; i < 10; i++) {
      expect(table.timestampUs(i)).toBeGreaterThan(table.timestampUs(i - 1));
    }
  });

  it('finds the GOP start: sync every 5 → frames 5..9 seek to decode index 5', () => {
    const table = new SampleTable(cfrSamples(10, 1000, 5), 30000);
    expect(table.syncDecodeIndexFor(0)).toBe(0);
    expect(table.syncDecodeIndexFor(4)).toBe(0);
    expect(table.syncDecodeIndexFor(5)).toBe(5);
    expect(table.syncDecodeIndexFor(7)).toBe(5);
    expect(table.syncDecodeIndexFor(9)).toBe(5);
  });

  it('handles B-frame reordering: presentation order sorts by cts, feeds by decode order', () => {
    // Decode order I P B B (dts 0..3), presentation order I B B P:
    //   decode 0: cts 0 (sync), decode 1: cts 3, decode 2: cts 1, decode 3: cts 2.
    const samples: SampleTiming[] = [
      { dts: 0, cts: 0, duration: 1, isSync: true },
      { dts: 1, cts: 3, duration: 1, isSync: false },
      { dts: 2, cts: 1, duration: 1, isSync: false },
      { dts: 3, cts: 2, duration: 1, isSync: false },
    ];
    const table = new SampleTable(samples, 1000);
    expect(table.decodeIndexOf(0)).toBe(0);
    expect(table.decodeIndexOf(1)).toBe(2);
    expect(table.decodeIndexOf(2)).toBe(3);
    expect(table.decodeIndexOf(3)).toBe(1);
    // Presentation timestamps stay monotone even though decode cts jump around.
    for (let i = 1; i < 4; i++) {
      expect(table.timestampUs(i)).toBeGreaterThan(table.timestampUs(i - 1));
    }
    // Emitting presentation frame 1 requires decoding through decode index 2.
    expect(table.lastDecodeIndexFor(0, 2)).toBe(2);
    // Frames [2, 4) need decode indices {3, 1} → 3.
    expect(table.lastDecodeIndexFor(2, 4)).toBe(3);
    // Frame 3 sits at decode index 1; its GOP starts at the sync sample 0.
    expect(table.syncDecodeIndexFor(3)).toBe(0);
  });

  it('maps decoder-output timestamps back to frame indices (exact, then nearest)', () => {
    const table = new SampleTable(cfrSamples(10, 1000), 30000);
    for (let i = 0; i < 10; i++) {
      expect(table.frameIndexForTimestampUs(table.timestampUs(i))).toBe(i);
    }
    // Nearest fallback: 1µs off still lands on the right frame.
    expect(table.frameIndexForTimestampUs(table.timestampUs(3) + 1)).toBe(3);
    expect(table.frameIndexForTimestampUs(table.timestampUs(7) - 1)).toBe(7);
    // Beyond both ends clamps to the nearest end.
    expect(table.frameIndexForTimestampUs(-5)).toBe(0);
    expect(table.frameIndexForTimestampUs(10_000_000)).toBe(9);
  });

  it('flattens VFR to one rate: fps = frameCount / durationSeconds', () => {
    const samples: SampleTiming[] = [
      { dts: 0, cts: 0, duration: 1000, isSync: true },
      { dts: 1000, cts: 1000, duration: 2000, isSync: false },
      { dts: 3000, cts: 3000, duration: 1000, isSync: false },
    ];
    const table = new SampleTable(samples, 1000);
    expect(table.durationUs).toBe(4_000_000);
    expect(table.fps).toBeCloseTo(0.75, 10);
  });

  it('rejects invalid construction and out-of-range queries', () => {
    expect(() => new SampleTable([], 1000)).toThrow(InvalidStateError);
    expect(() => new SampleTable(cfrSamples(3, 100), 0)).toThrow(InvalidStateError);
    expect(() => new SampleTable(cfrSamples(3, 100), 1.5)).toThrow(InvalidStateError);
    expect(() => new SampleTable(cfrSamples(3, 0), 1000)).toThrow(InvalidStateError);
    // dts must be non-decreasing (decode order).
    expect(
      () =>
        new SampleTable(
          [
            { dts: 100, cts: 100, duration: 100, isSync: true },
            { dts: 0, cts: 0, duration: 100, isSync: false },
          ],
          1000,
        ),
    ).toThrow(InvalidStateError);

    const table = new SampleTable(cfrSamples(3, 100), 1000);
    for (const bad of [-1, 3, 1.5, Number.NaN]) {
      expect(() => table.timestampUs(bad)).toThrow(InvalidStateError);
      expect(() => table.decodeIndexOf(bad)).toThrow(InvalidStateError);
    }

    // No sync sample at all → unseekable.
    const noSync = new SampleTable(
      [
        { dts: 0, cts: 0, duration: 100, isSync: false },
        { dts: 100, cts: 100, duration: 100, isSync: false },
      ],
      1000,
    );
    const err = (() => {
      try {
        noSync.syncDecodeIndexFor(1);
        return undefined;
      } catch (e) {
        return e as InvalidStateError;
      }
    })();
    expect(err).toBeInstanceOf(InvalidStateError);
    expect(err?.code).toBe('INVALID_STATE');
  });
});

describe('demuxMp4 (real mp4box, node)', () => {
  it('extracts info, sample table, and decoder config from the committed fixture', async () => {
    const { info, table, samples, config } = demuxMp4(await fixtureBytes());

    expect(info.frameCount).toBe(10);
    expect(info.width).toBe(320);
    expect(info.height).toBe(180);
    expect(info.codec.startsWith('avc1')).toBe(true);
    expect(info.durationUs).toBe(333333);
    expect(info.fps).toBeCloseTo(30, 2);

    // GOP structure pinned by the fixture generator: keyframes at 0 and 5.
    expect(table.syncDecodeIndexFor(4)).toBe(0);
    expect(table.syncDecodeIndexFor(7)).toBe(5);

    expect(samples).toHaveLength(10);
    expect(samples[0]?.isSync).toBe(true);
    expect(samples[1]?.isSync).toBe(false);
    expect(samples[5]?.isSync).toBe(true);
    // Decode order timestamps are monotone (baseline profile: no reordering).
    for (let i = 1; i < 10; i++) {
      expect(samples[i]!.timestampUs).toBeGreaterThan(samples[i - 1]!.timestampUs);
    }

    expect(config.codec).toBe(info.codec);
    expect(config.codedWidth).toBe(320);
    expect(config.codedHeight).toBe(180);
    // avcC description, box header stripped → starts with configurationVersion 1.
    const description = config.description as Uint8Array;
    expect(description).toBeInstanceOf(Uint8Array);
    expect(description[0]).toBe(1);
  });

  it('rejects a non-MP4 buffer with InvalidStateError', () => {
    const garbage = new TextEncoder().encode('this is definitely not an mp4 container').buffer;
    const err = (() => {
      try {
        demuxMp4(garbage as ArrayBuffer);
        return undefined;
      } catch (e) {
        return e as InvalidStateError;
      }
    })();
    expect(err).toBeInstanceOf(InvalidStateError);
    expect(err?.code).toBe('INVALID_STATE');
  });
});

describe('createWebCodecsFrameSource (node environment guard)', () => {
  it('rejects with InvalidStateError where WebCodecs is unavailable', async () => {
    await expect(createWebCodecsFrameSource(new Blob(['x']))).rejects.toMatchObject({
      code: 'INVALID_STATE',
    });
  });
});
