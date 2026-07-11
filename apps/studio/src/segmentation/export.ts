/**
 * `AlphaMatteExporter` integration — studio-contracts.md §4.5.
 *
 * MVP export is `{mode:'matte', format:'png-sequence'}` only — the
 * exporter's `'cutout'` mode and `'webm-vp9-alpha'` format both still throw
 * `NotImplementedError` (friction §0.3). MP4 cutout is a stretch feature
 * that would hand-composite matte+source into an MP4 via mediabunny rather
 * than calling the exporter's cutout mode; not implemented in wave 1.
 * Isolated in this file/try-catch so a failure here never touches
 * `trackState` — `studio-store.ts`'s `exportMp4Cutout` wrapper only ever
 * turns a thrown error here into a `notice` (kind: `'warn'`), never
 * `exportState: {phase:'error'}`.
 */
import { InvalidStateError, NotImplementedError } from '@websam3/core';
import { AlphaMatteExporter } from '@websam3/video-editing';
import type { StudioGet, StudioSet } from '../store/studio-store.js';

/** Trigger a browser download of `blob` named `fileName`, then release the object URL. */
function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Export `clipId`'s tracked masks as a PNG-sequence alpha matte and trigger
 * a download — the MVP export path, exactly `apps/demo/src/VideoTab.tsx`'s
 * `handleExport`.
 */
export async function exportMatte(get: StudioGet, set: StudioSet, clipId: string): Promise<void> {
  const timeline = get().maskTimelines[clipId];
  if (!timeline) {
    throw new InvalidStateError(
      `Clip '${clipId}' has no mask timeline yet — prompt and track an object before exporting.`,
    );
  }

  set({ exportState: { phase: 'running', framesDone: 0, frameCount: timeline.frameCount, kind: 'matte' } });

  const exporter = new AlphaMatteExporter(timeline);
  const result = await exporter.export({
    mode: 'matte',
    format: 'png-sequence',
    onProgress: (framesDone, frameCount) => {
      set({ exportState: { phase: 'running', framesDone, frameCount, kind: 'matte' } });
    },
  });

  downloadBlob(result.blob, result.suggestedFileName);
  set({ exportState: { phase: 'done', fileName: result.suggestedFileName, framesExported: result.framesExported } });
}

/**
 * Stretch: hand-composite `clipId`'s source + matte into an MP4 via
 * mediabunny. Not implemented in wave 1 — always rejects with
 * `NotImplementedError`, which `studio-store.ts` surfaces as a friendly
 * `notice` (never as an `exportState`/`trackState` error) so it can never
 * regress the MVP matte-export gate.
 */
export async function exportMp4Cutout(get: StudioGet, set: StudioSet, clipId: string): Promise<void> {
  void get;
  void set;
  void clipId;
  throw new NotImplementedError(
    "exportMp4Cutout, lands as a post-MVP stretch — use exportMatte (PNG-sequence matte) for now",
  );
}
