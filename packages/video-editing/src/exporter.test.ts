import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { unzipSync } from 'fflate';
import { EpochInvalidatedError, decodeRLE } from '@websam/core';
import type { FramePropagationResult, MaskResult, RLEMask } from '@websam/core';
import { AlphaMatteExporter } from './exporter.js';
import { MaskTimeline } from './timeline.js';

const WIDTH = 4;
const HEIGHT = 4;

/** Build a tiny 4x4 RLE mask from raw counts. */
function mask(counts: number[]): RLEMask {
  return { width: WIDTH, height: HEIGHT, counts: Uint32Array.from(counts) };
}

function makeTimeline(frameCount = 4): MaskTimeline {
  return new MaskTimeline({ frameCount, fps: 30, width: WIDTH, height: HEIGHT });
}

// ---------------------------------------------------------------------------
// Fake OffscreenCanvas for the node lane: convertToBlob emits a recognizable
// magic prefix followed by the exact RGBA bytes that were putImageData'd, so
// the zip round-trip can assert on real pixel bytes without a PNG codec.
// (The browser lane, exporter.browser.test.ts, decodes real PNGs.)
// ---------------------------------------------------------------------------

const FAKE_PNG_MAGIC = Uint8Array.of(0xfa, 0x4b, 0x45, 0x50, 0x4e, 0x47);

class FakeOffscreenCanvas {
  lastPut: Uint8ClampedArray | undefined;

  constructor(
    readonly width: number,
    readonly height: number,
  ) {}

  getContext(id: string): unknown {
    if (id !== '2d') return null;
    return {
      createImageData: (width: number, height: number) => ({
        width,
        height,
        data: new Uint8ClampedArray(width * height * 4),
      }),
      putImageData: (imageData: { data: Uint8ClampedArray }) => {
        this.lastPut = new Uint8ClampedArray(imageData.data);
      },
    };
  }

  async convertToBlob(options?: { type?: string }): Promise<Blob> {
    if (this.lastPut === undefined) throw new Error('convertToBlob before putImageData');
    const bytes = new Uint8Array(FAKE_PNG_MAGIC.length + this.lastPut.length);
    bytes.set(FAKE_PNG_MAGIC, 0);
    bytes.set(this.lastPut, FAKE_PNG_MAGIC.length);
    return new Blob([bytes], { type: options?.type ?? '' });
  }
}

/** The fake-PNG bytes the exporter must produce for `rle`: white-on-black opaque RGBA. */
function expectedFakePng(rle: RLEMask): Uint8Array {
  const bits = decodeRLE(rle);
  const bytes = new Uint8Array(FAKE_PNG_MAGIC.length + bits.length * 4);
  bytes.set(FAKE_PNG_MAGIC, 0);
  for (let i = 0; i < bits.length; i++) {
    const value = bits[i] ? 255 : 0;
    const offset = FAKE_PNG_MAGIC.length + i * 4;
    bytes[offset] = value;
    bytes[offset + 1] = value;
    bytes[offset + 2] = value;
    bytes[offset + 3] = 255;
  }
  return bytes;
}

async function zipEntries(blob: Blob): Promise<Record<string, Uint8Array>> {
  return unzipSync(new Uint8Array(await blob.arrayBuffer()));
}

/** Compression method of every local file header in the raw zip bytes. */
function localHeaderMethods(zipBytes: Uint8Array): number[] {
  const view = new DataView(zipBytes.buffer, zipBytes.byteOffset, zipBytes.byteLength);
  const methods: number[] = [];
  for (let i = 0; i + 10 <= zipBytes.length; i++) {
    if (view.getUint32(i, true) === 0x04034b50) methods.push(view.getUint16(i + 8, true));
  }
  return methods;
}

interface TimelineSidecar {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  objects: Record<string, number[]>;
}

function readSidecar(entries: Record<string, Uint8Array>): TimelineSidecar {
  const bytes = entries['timeline.json'];
  expect(bytes).toBeDefined();
  return JSON.parse(new TextDecoder().decode(bytes)) as TimelineSidecar;
}

describe('AlphaMatteExporter matte + png-sequence', () => {
  beforeEach(() => {
    vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('round-trips a single-object timeline: root-level names, PNG bytes, sidecar', async () => {
    const timeline = makeTimeline(4);
    const m0 = mask([3, 5, 8]);
    const m1 = mask([0, 16]);
    const m3 = mask([16]); // all-background matte
    timeline.set('1', 0, m0);
    timeline.set('1', 1, m1);
    timeline.set('1', 3, m3);

    const exporter = new AlphaMatteExporter(timeline);
    const result = await exporter.export({ mode: 'matte', format: 'png-sequence' });

    expect(result.format).toBe('png-sequence');
    expect(result.suggestedFileName).toBe('matte.zip');
    expect(result.framesExported).toBe(3);
    expect(result.blob.type).toBe('application/zip');

    const entries = await zipEntries(result.blob);
    expect(Object.keys(entries).sort()).toEqual([
      'frame-000000.png',
      'frame-000001.png',
      'frame-000003.png',
      'timeline.json',
    ]);
    expect(entries['frame-000000.png']).toEqual(expectedFakePng(m0));
    expect(entries['frame-000001.png']).toEqual(expectedFakePng(m1));
    expect(entries['frame-000003.png']).toEqual(expectedFakePng(m3));

    const sidecar = readSidecar(entries);
    expect(sidecar).toEqual({
      fps: 30,
      frameCount: 4,
      width: WIDTH,
      height: HEIGHT,
      objects: { '1': [0, 1, 3] },
    });
  });

  it("resolves format 'auto' (and the default) to png-sequence", async () => {
    const timeline = makeTimeline(1);
    timeline.set('1', 0, mask([16]));
    const exporter = new AlphaMatteExporter(timeline);

    const auto = await exporter.export({ mode: 'matte', format: 'auto' });
    expect(auto.format).toBe('png-sequence');
    const omitted = await exporter.export({ mode: 'matte' });
    expect(omitted.format).toBe('png-sequence');
  });

  it('puts each object in its own folder when several are tracked', async () => {
    const timeline = makeTimeline(3);
    timeline.set('a', 0, mask([3, 5, 8]));
    timeline.set('a', 1, mask([0, 16]));
    timeline.set('b', 1, mask([16]));

    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte' });
    expect(result.framesExported).toBe(3);

    const entries = await zipEntries(result.blob);
    expect(Object.keys(entries).sort()).toEqual([
      'obj-a/frame-000000.png',
      'obj-a/frame-000001.png',
      'obj-b/frame-000001.png',
      'timeline.json',
    ]);
    expect(readSidecar(entries).objects).toEqual({ a: [0, 1], b: [1] });
  });

  it('writes store-mode entries (fflate is only the container writer)', async () => {
    const timeline = makeTimeline(2);
    timeline.set('1', 0, mask([3, 5, 8]));
    timeline.set('1', 1, mask([0, 16]));

    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte' });
    const zipBytes = new Uint8Array(await result.blob.arrayBuffer());
    const methods = localHeaderMethods(zipBytes);
    expect(methods).toHaveLength(3); // 2 PNGs + timeline.json
    expect(methods).toEqual([0, 0, 0]); // 0 = store
  });

  it('skips holes instead of failing, and reports reality via onProgress + framesExported', async () => {
    const timeline = makeTimeline(5);
    // Sparse: a cancelled propagation left masks only at frames 1 and 2.
    timeline.set('1', 1, mask([3, 5, 8]));
    timeline.set('1', 2, mask([0, 16]));

    const progress: [number, number][] = [];
    const result = await new AlphaMatteExporter(timeline).export({
      mode: 'matte',
      onProgress: (framesDone, frameCount) => progress.push([framesDone, frameCount]),
    });

    expect(result.framesExported).toBe(2);
    expect(progress).toEqual([
      [1, 5],
      [2, 5],
      [3, 5],
      [4, 5],
      [5, 5],
    ]);
    const entries = await zipEntries(result.blob);
    expect(Object.keys(entries).sort()).toEqual([
      'frame-000001.png',
      'frame-000002.png',
      'timeline.json',
    ]);
    expect(readSidecar(entries).objects).toEqual({ '1': [1, 2] });
  });

  it('exports an empty timeline as a zip holding only timeline.json', async () => {
    const timeline = makeTimeline(2);
    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte' });
    expect(result.framesExported).toBe(0);
    const entries = await zipEntries(result.blob);
    expect(Object.keys(entries)).toEqual(['timeline.json']);
    expect(readSidecar(entries).objects).toEqual({});
  });
});

describe('AlphaMatteExporter M2 behavior matrix — still-unimplemented paths', () => {
  const exporter = () => {
    const timeline = makeTimeline(1);
    timeline.set('1', 0, mask([16]));
    return new AlphaMatteExporter(timeline);
  };

  it("rejects 'cutout' mode (any format) with NotImplementedError", async () => {
    for (const format of ['png-sequence', 'webm-vp9-alpha', 'auto', undefined] as const) {
      const err = await exporter()
        .export({ mode: 'cutout', format })
        .then(() => undefined)
        .catch((e: unknown) => e as Error);
      expect(err?.name).toBe('NotImplementedError');
      expect(err?.message).toMatch(/cutout export, lands in M4/);
    }
  });

  it("rejects 'webm-vp9-alpha' format with NotImplementedError", async () => {
    const err = await exporter()
      .export({ mode: 'matte', format: 'webm-vp9-alpha' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(err?.name).toBe('NotImplementedError');
    expect(err?.message).toMatch(/webm-vp9-alpha export, lands in M4/);
  });

  it('rejects png-sequence with InvalidStateError when OffscreenCanvas is missing', async () => {
    expect(typeof OffscreenCanvas).toBe('undefined'); // node lane: no stub here
    const err = await exporter()
      .export({ mode: 'matte' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(err?.name).toBe('InvalidStateError');
    expect(err?.message).toMatch(/OffscreenCanvas/);
  });
});

// ---------------------------------------------------------------------------
// MaskTimeline.collect — the propagate() → timeline bridge.
// ---------------------------------------------------------------------------

function fakeMask(objectId: number, rle: RLEMask): MaskResult {
  return {
    objectId,
    score: 1,
    width: rle.width,
    height: rle.height,
    toRLE: () => rle,
  } as MaskResult;
}

function frame(frameIndex: number, masks: MaskResult[]): FramePropagationResult {
  return { frameIndex, timestampUs: frameIndex * 33_333, masks };
}

async function* stream(
  frames: FramePropagationResult[],
  error?: Error,
): AsyncGenerator<FramePropagationResult> {
  for (const f of frames) yield f;
  if (error !== undefined) throw error;
}

describe('MaskTimeline.collect', () => {
  const init = { frameCount: 4, fps: 30, width: WIDTH, height: HEIGHT };

  it('stores every mask keyed by String(objectId) and returns the timeline', async () => {
    const m10 = mask([3, 5, 8]);
    const m11 = mask([0, 16]);
    const m21 = mask([16]);
    const timeline = await MaskTimeline.collect(
      stream([frame(0, [fakeMask(1, m10)]), frame(1, [fakeMask(1, m11), fakeMask(2, m21)])]),
      init,
    );

    expect(timeline).toBeInstanceOf(MaskTimeline);
    expect(timeline.frameCount).toBe(4);
    expect(timeline.get('1', 0)).toBe(m10);
    expect(timeline.get('1', 1)).toBe(m11);
    expect(timeline.get('2', 1)).toBe(m21);
    expect(timeline.objectIds().sort()).toEqual(['1', '2']);
  });

  it('collects into an existing MaskTimeline passed as init', async () => {
    const existing = makeTimeline(4);
    existing.set('1', 0, mask([16]));
    const m2 = mask([0, 16]);

    const returned = await MaskTimeline.collect(stream([frame(2, [fakeMask(1, m2)])]), existing);
    expect(returned).toBe(existing);
    expect(existing.get('1', 0)).toBeDefined(); // pre-existing masks untouched
    expect(existing.get('1', 2)).toBe(m2);
  });

  it('calls onFrame after the frame is stored, once per frame, in order', async () => {
    const timeline = makeTimeline(4);
    const seen: number[] = [];
    await MaskTimeline.collect(
      stream([frame(0, [fakeMask(1, mask([16]))]), frame(1, [fakeMask(1, mask([0, 16]))])]),
      timeline,
      {
        onFrame: (f) => {
          // stored-before-callback: the demo repaints from the timeline here
          expect(timeline.get('1', f.frameIndex)).toBeDefined();
          seen.push(f.frameIndex);
        },
      },
    );
    expect(seen).toEqual([0, 1]);
  });

  it('stamps options.epoch into set() so stale late writes are rejected', async () => {
    const timeline = makeTimeline(4);
    timeline.set('1', 0, mask([16]));
    const newEpoch = timeline.invalidateAfter('1', 0); // epoch 0 → 1
    expect(newEpoch).toBe(1);

    // A straggler collect still carrying the OLD epoch must not resurrect masks.
    await MaskTimeline.collect(stream([frame(2, [fakeMask(1, mask([0, 16]))])]), timeline, {
      epoch: 0,
    });
    expect(timeline.get('1', 2)).toBeUndefined();
    expect(timeline.epoch('1')).toBe(1);

    // Re-collecting under the new epoch stores.
    const m2 = mask([0, 16]);
    await MaskTimeline.collect(stream([frame(2, [fakeMask(1, m2)])]), timeline, {
      epoch: newEpoch,
    });
    expect(timeline.get('1', 2)).toBe(m2);
  });

  it('propagates EpochInvalidatedError from the stream and keeps already-stored frames', async () => {
    const timeline = makeTimeline(4);
    const err = new EpochInvalidatedError('refined mid-flight');
    const seen: number[] = [];

    await expect(
      MaskTimeline.collect(
        stream([frame(0, [fakeMask(1, mask([16]))]), frame(1, [fakeMask(1, mask([0, 16]))])], err),
        timeline,
        { onFrame: (f) => seen.push(f.frameIndex) },
      ),
    ).rejects.toBe(err);

    // Frames stored before the abort remain; invalidateAfter is the caller's move.
    expect(seen).toEqual([0, 1]);
    expect(timeline.get('1', 0)).toBeDefined();
    expect(timeline.get('1', 1)).toBeDefined();
  });
});
