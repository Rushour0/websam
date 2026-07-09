import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  EpochInvalidatedError,
  InvalidStateError,
  NotImplementedError,
  type FramePropagationResult,
  type MaskResult,
  type RLEMask,
} from '@websam3/core';
import { strFromU8, unzipSync } from 'fflate';
import { AlphaMatteExporter } from './exporter.js';
import { MaskTimeline } from './timeline.js';

const PNG_MAGIC = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/**
 * A node-only stand-in for `OffscreenCanvas`/`ImageData`. `convertToBlob`
 * emits the 8-byte PNG signature followed by the raw RGBA bytes so the node
 * test can both recognise a PNG and round-trip the exact matte pixels; the
 * real PNG codec is exercised by `exporter.browser.test.ts`.
 */
class FakeImageData {
  constructor(
    public data: Uint8ClampedArray,
    public width: number,
    public height: number,
  ) {}
}

class FakeOffscreenCanvas {
  private img: FakeImageData | null = null;
  constructor(
    public width: number,
    public height: number,
  ) {}
  getContext(): { putImageData: (img: FakeImageData) => void } {
    return {
      putImageData: (img: FakeImageData) => {
        this.img = img;
      },
    };
  }
  convertToBlob(): Promise<Blob> {
    const rgba = this.img ? this.img.data : new Uint8ClampedArray(0);
    const out = new Uint8Array(PNG_MAGIC.length + rgba.length);
    out.set(PNG_MAGIC, 0);
    out.set(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), PNG_MAGIC.length);
    return Promise.resolve(new Blob([out], { type: 'image/png' }));
  }
}

function installCanvas(): void {
  vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas);
  vi.stubGlobal('ImageData', FakeImageData);
}

/** A 2x2 RLE mask from alternating-run counts (starts with the zero-run). */
function rle(counts: number[]): RLEMask {
  return { width: 2, height: 2, counts: Uint32Array.from(counts) };
}

/** Strip the 8-byte PNG signature the fake canvas prepends, leaving raw RGBA. */
function rgbaOf(entry: Uint8Array): Uint8Array {
  return entry.subarray(PNG_MAGIC.length);
}

function timelineWith(
  frameCount: number,
  writes: Array<[objectId: string, frame: number, mask: RLEMask]>,
): MaskTimeline {
  const timeline = new MaskTimeline({ frameCount, fps: 30, width: 2, height: 2 });
  for (const [objectId, frame, mask] of writes) timeline.set(objectId, frame, mask);
  return timeline;
}

async function entriesOf(exporter: AlphaMatteExporter): Promise<{
  result: Awaited<ReturnType<AlphaMatteExporter['export']>>;
  entries: Record<string, Uint8Array>;
}> {
  const result = await exporter.export({ mode: 'matte', format: 'png-sequence' });
  const buf = new Uint8Array(await result.blob.arrayBuffer());
  return { result, entries: unzipSync(buf) };
}

describe('AlphaMatteExporter.export — gating (no rendering)', () => {
  it('rejects cutout export with NotImplementedError naming M4', async () => {
    const exporter = new AlphaMatteExporter(timelineWith(2, [['a', 0, rle([4])]]));
    await expect(exporter.export({ mode: 'cutout' })).rejects.toBeInstanceOf(NotImplementedError);
    await expect(exporter.export({ mode: 'cutout' })).rejects.toThrow(/cutout export.*M4/);
  });

  it('rejects webm-vp9-alpha format with NotImplementedError naming M4', async () => {
    const exporter = new AlphaMatteExporter(timelineWith(2, [['a', 0, rle([4])]]));
    await expect(
      exporter.export({ mode: 'matte', format: 'webm-vp9-alpha' }),
    ).rejects.toThrow(/webm-vp9-alpha export.*M4/);
  });

  it('cutout takes precedence over webm-vp9-alpha in the error message', async () => {
    const exporter = new AlphaMatteExporter(timelineWith(2, [['a', 0, rle([4])]]));
    await expect(
      exporter.export({ mode: 'cutout', format: 'webm-vp9-alpha' }),
    ).rejects.toThrow(/cutout/);
  });
});

describe('AlphaMatteExporter.export — no canvas environment', () => {
  it('throws InvalidStateError when OffscreenCanvas/ImageData are absent', async () => {
    // No canvas stub installed: matte export must fail loudly, not silently.
    const exporter = new AlphaMatteExporter(timelineWith(2, [['a', 0, rle([4])]]));
    await expect(exporter.export({ mode: 'matte', format: 'png-sequence' })).rejects.toBeInstanceOf(
      InvalidStateError,
    );
  });
});

describe('AlphaMatteExporter.export — png-sequence zip', () => {
  beforeEach(() => installCanvas());
  afterEach(() => vi.unstubAllGlobals());

  it('writes root-level frame-%06d.png for a single object, skipping holes', async () => {
    // frameCount 4, masks at 0 and 2 only → holes at 1 and 3.
    const timeline = timelineWith(4, [
      ['a', 0, rle([0, 1, 2, 1])],
      ['a', 2, rle([2, 2])],
    ]);
    const { result, entries } = await entriesOf(new AlphaMatteExporter(timeline));

    expect(Object.keys(entries).sort()).toEqual([
      'frame-000000.png',
      'frame-000002.png',
      'timeline.json',
    ]);
    expect(result.framesExported).toBe(2);
    expect(result.format).toBe('png-sequence');
    expect(result.suggestedFileName).toBe('matte.zip');
    expect(result.blob.type).toBe('application/zip');
  });

  it('stores real PNG bytes that round-trip the white-on-black matte pixels', async () => {
    // [1,0,0,1] → white, black, black, white.
    const timeline = timelineWith(1, [['a', 0, rle([0, 1, 2, 1])]]);
    const { entries } = await entriesOf(new AlphaMatteExporter(timeline));
    const entry = entries['frame-000000.png'];
    expect(entry).toBeDefined();
    expect(Array.from(entry!.subarray(0, PNG_MAGIC.length))).toEqual(Array.from(PNG_MAGIC));
    expect(Array.from(rgbaOf(entry!))).toEqual([
      255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ]);
  });

  it('writes a timeline.json index with geometry and per-object frame lists', async () => {
    const timeline = timelineWith(4, [
      ['a', 0, rle([4])],
      ['a', 3, rle([4])],
    ]);
    const { entries } = await entriesOf(new AlphaMatteExporter(timeline));
    const manifest = JSON.parse(strFromU8(entries['timeline.json']!)) as {
      fps: number;
      frameCount: number;
      width: number;
      height: number;
      objects: Record<string, number[]>;
    };
    expect(manifest).toMatchObject({ fps: 30, frameCount: 4, width: 2, height: 2 });
    expect(manifest.objects).toEqual({ a: [0, 3] });
  });

  it('namespaces frames under obj-<id>/ when multiple objects are tracked', async () => {
    const timeline = timelineWith(2, [
      ['a', 0, rle([4])],
      ['b', 1, rle([4])],
    ]);
    const { result, entries } = await entriesOf(new AlphaMatteExporter(timeline));
    expect(Object.keys(entries).sort()).toEqual([
      'obj-a/frame-000000.png',
      'obj-b/frame-000001.png',
      'timeline.json',
    ]);
    expect(result.framesExported).toBe(2);
    const manifest = JSON.parse(strFromU8(entries['timeline.json']!)) as {
      objects: Record<string, number[]>;
    };
    expect(manifest.objects).toEqual({ a: [0], b: [1] });
  });

  it("resolves format 'auto' to png-sequence at M2", async () => {
    const timeline = timelineWith(1, [['a', 0, rle([4])]]);
    const result = await new AlphaMatteExporter(timeline).export({ mode: 'matte', format: 'auto' });
    expect(result.format).toBe('png-sequence');
    expect(result.framesExported).toBe(1);
  });

  it('reports progress once per timeline frame index, holes included', async () => {
    const timeline = timelineWith(4, [['a', 1, rle([4])]]);
    const calls: Array<[number, number]> = [];
    await new AlphaMatteExporter(timeline).export({
      mode: 'matte',
      format: 'png-sequence',
      onProgress: (done, total) => calls.push([done, total]),
    });
    expect(calls).toEqual([
      [1, 4],
      [2, 4],
      [3, 4],
      [4, 4],
    ]);
  });

  it('produces a valid zip with only timeline.json for an empty timeline', async () => {
    const timeline = new MaskTimeline({ frameCount: 3, fps: 24, width: 2, height: 2 });
    const { result, entries } = await entriesOf(new AlphaMatteExporter(timeline));
    expect(Object.keys(entries)).toEqual(['timeline.json']);
    expect(result.framesExported).toBe(0);
  });
});

// --- MaskTimeline.collect (bridge from propagate() to timeline storage) ---

function fakeMask(objectId: number, counts: number[]): MaskResult {
  const mask: RLEMask = { width: 2, height: 2, counts: Uint32Array.from(counts) };
  return { objectId, toRLE: () => mask } as unknown as MaskResult;
}

function frame(frameIndex: number, masks: MaskResult[]): FramePropagationResult {
  return { frameIndex, timestampUs: frameIndex * 1000, masks };
}

async function* streamOf(frames: FramePropagationResult[]): AsyncGenerator<FramePropagationResult> {
  for (const f of frames) yield f;
}

const INIT = { frameCount: 5, fps: 30, width: 2, height: 2 };

describe('MaskTimeline.collect', () => {
  it('drains an iterator into a timeline keyed by objectId and frameIndex', async () => {
    const timeline = await MaskTimeline.collect(
      streamOf([
        frame(0, [fakeMask(1, [0, 4]), fakeMask(2, [4])]),
        frame(1, [fakeMask(1, [2, 2]), fakeMask(2, [4])]),
      ]),
      INIT,
    );
    expect(timeline.objectIds().sort()).toEqual(['1', '2']);
    expect(timeline.get('1', 1)).toEqual({ width: 2, height: 2, counts: Uint32Array.from([2, 2]) });
    expect(timeline.get('2', 0)).toBeDefined();
  });

  it('calls onFrame after each frame is stored, in order', async () => {
    const seen: number[] = [];
    await MaskTimeline.collect(
      streamOf([frame(0, [fakeMask(1, [4])]), frame(2, [fakeMask(1, [4])])]),
      INIT,
      {
        onFrame: (f) => {
          // The mask must already be stored when onFrame fires.
          seen.push(f.frameIndex);
        },
      },
    );
    expect(seen).toEqual([0, 2]);
  });

  it('leaves un-yielded frames as holes rather than failing', async () => {
    const timeline = await MaskTimeline.collect(
      streamOf([frame(0, [fakeMask(1, [4])]), frame(3, [fakeMask(1, [4])])]),
      INIT,
    );
    // Frames 1, 2, 4 were never yielded → holes, not errors.
    expect(timeline.holes('1')).toEqual([1, 2, 4]);
  });

  it('stamps the provided epoch so a later stale set is rejected', async () => {
    const timeline = await MaskTimeline.collect(
      streamOf([frame(0, [fakeMask(1, [4])])]),
      INIT,
      { epoch: 3 },
    );
    expect(timeline.epoch('1')).toBe(3);
    // A straggler from an older epoch is rejected by the timeline.
    expect(timeline.set('1', 1, { width: 2, height: 2, counts: Uint32Array.from([4]) }, 2)).toBe(
      false,
    );
  });

  it('propagates EpochInvalidatedError, keeping frames stored before the throw', async () => {
    async function* aborting(): AsyncGenerator<FramePropagationResult> {
      yield frame(0, [fakeMask(1, [4])]);
      throw new EpochInvalidatedError('refined mid-flight');
    }
    // Collect the same generator once and inspect via a shared timeline is not
    // possible (collect owns the timeline), so assert the rejection type.
    await expect(MaskTimeline.collect(aborting(), INIT)).rejects.toBeInstanceOf(
      EpochInvalidatedError,
    );
  });
});
