/**
 * Browser tests for the WebCodecs frame source against the committed fixture
 * (docs/m2-internal-contracts.md §9.2): `src/video/fixtures/
 * clip-320x180-10f.mp4` — 10 frames, 320×180, H.264 constrained-baseline,
 * GOP of 5 (sync samples at frames 0 and 5), one distinct solid color per
 * frame so frame identity is assertable from pixels.
 *
 * Fixture palette (frame index → RGB), pinned by the ffmpeg generator:
 * red, green, blue, yellow, magenta, cyan, white, black, orange, purple.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { InvalidStateError } from '../errors.js';
import type { DecodedFrame } from './frame-source.js';
import { createWebCodecsFrameSource, type WebCodecsFrameSource } from './webcodecs-source.js';

const clipUrl = new URL('./fixtures/clip-320x180-10f.mp4', import.meta.url).href;

/** Frame index → encoded solid color (see the module docstring). */
const PALETTE: readonly (readonly [number, number, number])[] = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [255, 255, 0],
  [255, 0, 255],
  [0, 255, 255],
  [255, 255, 255],
  [0, 0, 0],
  [255, 128, 0],
  [128, 0, 255],
];

const hasWebCodecs = typeof VideoDecoder === 'function';

async function fetchClip(): Promise<Blob> {
  const res = await fetch(clipUrl);
  if (!res.ok) throw new Error(`failed to fetch fixture: ${res.status}`);
  return res.blob();
}

/** Average an 8×8 patch at the frame center and return [r, g, b]. */
function centerColor(frame: VideoFrame): [number, number, number] {
  const canvas = new OffscreenCanvas(frame.displayWidth, frame.displayHeight);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
  ctx.drawImage(frame, 0, 0);
  const patch = ctx.getImageData(frame.displayWidth / 2 - 4, frame.displayHeight / 2 - 4, 8, 8);
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < patch.data.length; i += 4) {
    r += patch.data[i] as number;
    g += patch.data[i + 1] as number;
    b += patch.data[i + 2] as number;
  }
  const n = patch.data.length / 4;
  return [r / n, g / n, b / n];
}

/** Nearest-palette classification — robust to yuv420 + x264 quantization. */
function classify(color: readonly [number, number, number]): number {
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < PALETTE.length; i++) {
    const p = PALETTE[i] as readonly [number, number, number];
    const dist = (p[0] - color[0]) ** 2 + (p[1] - color[1]) ** 2 + (p[2] - color[2]) ** 2;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

describe.skipIf(!hasWebCodecs)('WebCodecsFrameSource (real decode)', () => {
  let source: WebCodecsFrameSource | undefined;

  afterEach(() => {
    source?.dispose();
    source = undefined;
  });

  it('reports VideoSourceInfo from the sample table', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    expect(source.info.frameCount).toBe(10);
    expect(source.info.width).toBe(320);
    expect(source.info.height).toBe(180);
    expect(source.info.durationUs).toBe(333333);
    expect(source.info.fps).toBeCloseTo(30, 2);
    expect(source.info.codec.startsWith('avc1')).toBe(true);
  });

  it('reads sequentially: ascending indices, monotone timestamps, right pixels', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    const seen: { frameIndex: number; timestampUs: number; color: number }[] = [];
    for await (const decoded of source.read(0, 10)) {
      expect(decoded.timestampUs).toBe(decoded.frame.timestamp);
      expect(decoded.frame.displayWidth).toBe(320);
      expect(decoded.frame.displayHeight).toBe(180);
      seen.push({
        frameIndex: decoded.frameIndex,
        timestampUs: decoded.timestampUs,
        color: classify(centerColor(decoded.frame)),
      });
      decoded.frame.close();
    }
    expect(seen.map((s) => s.frameIndex)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(seen[0]?.timestampUs).toBe(0);
    for (let i = 1; i < seen.length; i++) {
      expect(seen[i]!.timestampUs).toBeGreaterThan(seen[i - 1]!.timestampUs);
    }
    // Every frame decodes to its own palette color.
    expect(seen.map((s) => s.color)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(source.debugStats().liveFrames).toBe(0);
  });

  it('reads a mid-clip subrange starting on a non-sync frame', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    const indices: number[] = [];
    const colors: number[] = [];
    // Frame 7 is a P-frame — decode must warm up from the sync sample at 5.
    for await (const decoded of source.read(7, 10)) {
      indices.push(decoded.frameIndex);
      colors.push(classify(centerColor(decoded.frame)));
      decoded.frame.close();
    }
    expect(indices).toEqual([7, 8, 9]);
    expect(colors).toEqual([7, 8, 9]);
  });

  it('frameAt lands on the exact frame across GOP boundaries', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    // Frame 7 → GOP starting at sync frame 5; frame 2 → GOP at frame 0.
    const frame7 = await source.frameAt(7);
    expect(classify(centerColor(frame7))).toBe(7);
    expect(frame7.timestamp).toBeGreaterThan(0);
    frame7.close();
    const frame2 = await source.frameAt(2);
    expect(classify(centerColor(frame2))).toBe(2);
    frame2.close();
    expect(source.debugStats().liveFrames).toBe(0);
  });

  it('cancels mid-stream via return() without dangling frames, then reads again', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    const iterator = source.read(0, 10);
    const yielded: DecodedFrame[] = [];
    for (let i = 0; i < 2; i++) {
      const result = await iterator.next();
      expect(result.done).toBe(false);
      yielded.push(result.value as DecodedFrame);
    }
    await iterator.return?.();
    // Everything still inside the source got closed on cancel.
    expect(source.debugStats().liveFrames).toBe(0);
    // Already-yielded frames stayed ours (usable, then closed by us).
    expect(yielded.map((d) => classify(centerColor(d.frame)))).toEqual([0, 1]);
    for (const d of yielded) d.frame.close();
    // The source is reusable after cancellation.
    const again = await source.read(5, 6).next();
    expect(again.done).toBe(false);
    const decoded = (again as IteratorYieldResult<DecodedFrame>).value;
    expect(decoded.frameIndex).toBe(5);
    decoded.frame.close();
  });

  it('rejects overlapping reads, bad bounds, and use after dispose', async () => {
    source = await createWebCodecsFrameSource(await fetchClip());
    const iterator = source.read(0, 10);
    expect(() => source!.read(0, 1)).toThrow(InvalidStateError);
    await expect(source.frameAt(0)).rejects.toThrow(InvalidStateError);
    await iterator.return?.();

    expect(() => source!.read(-1, 5)).toThrow(InvalidStateError);
    expect(() => source!.read(3, 3)).toThrow(InvalidStateError);
    expect(() => source!.read(0, 11)).toThrow(InvalidStateError);
    await expect(source.frameAt(10)).rejects.toThrow(InvalidStateError);

    source.dispose();
    source.dispose(); // idempotent
    expect(() => source!.read(0, 1)).toThrow(InvalidStateError);
    await expect(source!.frameAt(0)).rejects.toThrow(InvalidStateError);
  });

  it('rejects a non-MP4 blob with InvalidStateError', async () => {
    const err = await createWebCodecsFrameSource(
      new Blob(['this is definitely not an mp4 container']),
    ).then(
      () => undefined,
      (e: unknown) => e as InvalidStateError,
    );
    expect(err).toBeInstanceOf(InvalidStateError);
    expect(err?.code).toBe('INVALID_STATE');
  });
});
