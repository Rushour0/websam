/**
 * Browser test for {@link WebCodecsFrameSource}: exercises real mp4box demux +
 * WebCodecs `VideoDecoder` against the committed 10-frame 320x180 H.264 fixture
 * (`fixtures/clip-320x180-10f.mp4`). Each fixture frame is a distinct solid
 * gray level `16 + 24*i` (i in 0..9), so a decoded center pixel identifies the
 * frame it came from.
 *
 * Demux and error assertions are hard (they need no decoder). The decode
 * assertions are gated on `VideoDecoder.isConfigSupported`, soft-passing where
 * the browser build ships no H.264 decoder — the same soft-pass convention as
 * `ort.browser.test.ts`'s webgpu lane.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { WebCodecsFrameSource } from './webcodecs-source.js';

const fixtureUrl = new URL('./fixtures/clip-320x180-10f.mp4', import.meta.url).href;

/** Expected solid-gray level of fixture frame `i`. */
const expectedGray = (i: number): number => 16 + 24 * i;

// Resolved once: does this browser's VideoDecoder actually decode the fixture's
// H.264? Playwright's open-source Chromium may lack a proprietary codec.
const decodeSupported: boolean = await (async () => {
  if (typeof VideoDecoder === 'undefined') return false;
  try {
    const { supported } = await VideoDecoder.isConfigSupported({
      codec: 'avc1.42c01e',
      codedWidth: 320,
      codedHeight: 180,
    });
    return supported === true;
  } catch {
    return false;
  }
})();

if (!decodeSupported) {
  // eslint-disable-next-line no-console
  console.log('[webcodecs-source.browser.test] H.264 decode unsupported — decode cases soft-pass');
}

async function fetchFixture(): Promise<Blob> {
  const res = await fetch(fixtureUrl);
  if (!res.ok) throw new Error(`failed to fetch fixture: ${res.status}`);
  return await res.blob();
}

/** Center-pixel gray value of a decoded frame, via an OffscreenCanvas readback. */
function centerGray(frame: VideoFrame): number {
  const canvas = new OffscreenCanvas(frame.displayWidth, frame.displayHeight);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no 2d context');
  ctx.drawImage(frame, 0, 0);
  const x = Math.floor(frame.displayWidth / 2);
  const y = Math.floor(frame.displayHeight / 2);
  const [r] = ctx.getImageData(x, y, 1, 1).data;
  return r ?? Number.NaN;
}

describe('WebCodecsFrameSource', () => {
  let blob: Blob;
  beforeAll(async () => {
    blob = await fetchFixture();
  });

  it('demuxes VideoSourceInfo from the sample table', async () => {
    const source = await WebCodecsFrameSource.create(blob);
    try {
      const { info } = source;
      expect(info.frameCount).toBe(10);
      expect(info.width).toBe(320);
      expect(info.height).toBe(180);
      expect(info.codec.startsWith('avc1')).toBe(true);
      expect(info.durationUs).toBe(1_000_000);
      expect(info.fps).toBeCloseTo(10, 5);
    } finally {
      await source.close();
    }
  });

  it('rejects a non-MP4 blob with InvalidStateError', async () => {
    const junk = new Blob([new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])], {
      type: 'application/octet-stream',
    });
    await expect(WebCodecsFrameSource.create(junk)).rejects.toMatchObject({
      code: 'INVALID_STATE',
    });
  });

  it('throws InvalidStateError after close()', async () => {
    const source = await WebCodecsFrameSource.create(blob);
    await source.close();
    await expect(source.frameAt(0)).rejects.toMatchObject({ code: 'INVALID_STATE' });
  });

  it.skipIf(!decodeSupported)(
    'reads frames sequentially in presentation order with monotone timestamps',
    async () => {
      const source = await WebCodecsFrameSource.create(blob);
      try {
        const indices: number[] = [];
        const timestamps: number[] = [];
        for await (const { frame, frameIndex, timestampUs } of source.read()) {
          indices.push(frameIndex);
          timestamps.push(timestampUs);
          expect(timestampUs).toBe(frame.timestamp);
          expect(centerGray(frame)).toBeCloseTo(expectedGray(frameIndex), -1);
          frame.close();
        }
        expect(indices).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        for (let i = 1; i < timestamps.length; i++) {
          expect(timestamps[i]!).toBeGreaterThan(timestamps[i - 1]!);
        }
      } finally {
        await source.close();
      }
    },
  );

  it.skipIf(!decodeSupported)('reads a sub-range [3, 7)', async () => {
    const source = await WebCodecsFrameSource.create(blob);
    try {
      const indices: number[] = [];
      for await (const { frame, frameIndex } of source.read({ startFrame: 3, endFrame: 7 })) {
        indices.push(frameIndex);
        expect(centerGray(frame)).toBeCloseTo(expectedGray(frameIndex), -1);
        frame.close();
      }
      expect(indices).toEqual([3, 4, 5, 6]);
    } finally {
      await source.close();
    }
  });

  it.skipIf(!decodeSupported)('frameAt returns the addressed frame', async () => {
    const source = await WebCodecsFrameSource.create(blob);
    try {
      for (const target of [0, 2, 7, 9]) {
        const frame = await source.frameAt(target);
        try {
          expect(frame.timestamp).toBe(target * 100_000);
          expect(centerGray(frame)).toBeCloseTo(expectedGray(target), -1);
        } finally {
          frame.close();
        }
      }
    } finally {
      await source.close();
    }
  });

  it.skipIf(!decodeSupported)('frameAt rejects an out-of-range index', async () => {
    const source = await WebCodecsFrameSource.create(blob);
    try {
      await expect(source.frameAt(10)).rejects.toMatchObject({ code: 'INVALID_STATE' });
      await expect(source.frameAt(-1)).rejects.toMatchObject({ code: 'INVALID_STATE' });
    } finally {
      await source.close();
    }
  });

  it.skipIf(!decodeSupported)(
    'return() mid-stream releases undelivered frames and the source recovers',
    async () => {
      const source = await WebCodecsFrameSource.create(blob);
      try {
        // Pull three frames, then break — the generator's finally must close
        // any decoded-but-unyielded frames and the decoder without throwing.
        const seen: number[] = [];
        for await (const { frame, frameIndex } of source.read()) {
          seen.push(frameIndex);
          frame.close();
          if (seen.length === 3) break;
        }
        expect(seen).toEqual([0, 1, 2]);

        // A fresh full read must still complete — proves no leaked/broken decoder.
        const all: number[] = [];
        for await (const { frame, frameIndex } of source.read()) {
          all.push(frameIndex);
          frame.close();
        }
        expect(all).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      } finally {
        await source.close();
      }
    },
  );
});
