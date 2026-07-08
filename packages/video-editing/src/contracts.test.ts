import { describe, expect, it } from 'vitest';
import { AlphaMatteExporter } from './exporter.js';
import { MaskCompositor } from './compositor.js';
import { MaskTimeline } from './timeline.js';

describe('M0 not-implemented surfaces', () => {
  it('AlphaMatteExporter constructs but export() throws NotImplementedError', async () => {
    const timeline = new MaskTimeline({ frameCount: 1, fps: 30, width: 2, height: 2 });
    const exporter = new AlphaMatteExporter(timeline);
    expect(exporter.timeline).toBe(timeline);
    const err = await exporter
      .export({ mode: 'matte', format: 'png-sequence' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(err).toBeInstanceOf(Error);
    expect(err?.name).toBe('NotImplementedError');
    expect(err?.message).toMatch(/lands in M2 \(PNG-zip\) \/ M4 \(VP9-alpha\)/);
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
