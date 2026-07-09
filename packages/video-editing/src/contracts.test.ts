import { describe, expect, it } from 'vitest';
import { AlphaMatteExporter } from './exporter.js';
import { MaskCompositor } from './compositor.js';
import { MaskTimeline } from './timeline.js';

describe('M0 not-implemented surfaces', () => {
  it('AlphaMatteExporter still throws NotImplementedError for the M4 surfaces (cutout)', async () => {
    // M2 makes matte + png-sequence real (see exporter.test.ts); the remaining
    // not-implemented exporter surface is cutout / webm-vp9-alpha, landing M4.
    const timeline = new MaskTimeline({ frameCount: 1, fps: 30, width: 2, height: 2 });
    const exporter = new AlphaMatteExporter(timeline);
    expect(exporter.timeline).toBe(timeline);
    const err = await exporter
      .export({ mode: 'cutout' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(err).toBeInstanceOf(Error);
    expect(err?.name).toBe('NotImplementedError');
    expect(err?.message).toMatch(/cutout export, lands in M4/);
  });

  it('MaskCompositor constructs but composite() throws NotImplementedError', () => {
    const canvas = {} as HTMLCanvasElement; // node test env: only identity matters at M0
    const compositor = new MaskCompositor(canvas);
    expect(compositor.canvas).toBe(canvas);
    expect(() =>
      compositor.composite({} as CanvasImageSource, new Map(), { mode: 'highlight' }),
    ).toThrowError(/MaskCompositor, lands in M4/);
  });
});
