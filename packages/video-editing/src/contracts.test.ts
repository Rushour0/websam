import { describe, expect, it } from 'vitest';
import { AlphaMatteExporter } from './exporter.js';
import { MaskCompositor } from './compositor.js';
import { MaskTimeline } from './timeline.js';

describe('M2 not-implemented surfaces', () => {
  it('AlphaMatteExporter constructs; cutout + VP9-alpha still throw NotImplementedError', async () => {
    const timeline = new MaskTimeline({ frameCount: 1, fps: 30, width: 2, height: 2 });
    const exporter = new AlphaMatteExporter(timeline);
    expect(exporter.timeline).toBe(timeline);
    // matte + png-sequence is real at M2 — see exporter.test.ts. The M4 paths stay gated:
    const cutoutErr = await exporter
      .export({ mode: 'cutout', format: 'png-sequence' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(cutoutErr).toBeInstanceOf(Error);
    expect(cutoutErr?.name).toBe('NotImplementedError');
    expect(cutoutErr?.message).toMatch(/cutout export, lands in M4/);
    const vp9Err = await exporter
      .export({ mode: 'matte', format: 'webm-vp9-alpha' })
      .then(() => undefined)
      .catch((e: unknown) => e as Error);
    expect(vp9Err).toBeInstanceOf(Error);
    expect(vp9Err?.name).toBe('NotImplementedError');
    expect(vp9Err?.message).toMatch(/webm-vp9-alpha export, lands in M4/);
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
