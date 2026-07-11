/**
 * Pure-node unit tests of the zustand store's reducer logic — no browser, no
 * segmenter (per studio-contracts.md §6.2). The async segmentation actions
 * (`activateClip`/`addPromptObject`/`startTracking`/...) are covered by the
 * browser gate (`segmentation.browser.test.ts`), not re-tested here; this
 * file only exercises the synchronous timeline/track/tool/selection reducers
 * and the state-machine shapes for `trackState`/`exportState`/`modelStatus`.
 *
 * `importClip` is exercised too, but with `probeClipMeta` unavailable in
 * node (it touches `document.createElement('video')`), so it's stubbed via
 * `vi.mock` — the store's own merge-into-`clips` logic is what's under test,
 * not the DOM probe.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../video/frame-source.js', () => ({
  probeClipMeta: vi.fn(async () => ({
    durationSec: 1,
    fps: 10,
    width: 256,
    height: 256,
    frameCount: 10,
    frameCountGuessed: true,
  })),
}));

// The segmentation seam is exercised end-to-end by the browser gate; stub it
// here so importing studio-store.ts in node never touches @websam3/core.
vi.mock('../segmentation/segmenter-lifecycle.js', () => ({
  loadSegmenter: vi.fn(async () => {
    throw new Error('not used in store unit tests');
  }),
}));
vi.mock('../segmentation/session-manager.js', () => ({
  activateClip: vi.fn(),
  addPromptObject: vi.fn(),
  refineObject: vi.fn(),
  removeObject: vi.fn(),
  disposeClipSession: vi.fn(),
}));
vi.mock('../segmentation/propagate-loop.js', () => ({
  startTracking: vi.fn(),
  cancelTracking: vi.fn(),
}));
vi.mock('../segmentation/export.js', () => ({
  exportMatte: vi.fn(),
  exportMp4Cutout: vi.fn(),
}));

// `URL.createObjectURL`/`revokeObjectURL` don't exist in the node environment.
beforeEach(() => {
  (globalThis as { URL: typeof URL }).URL.createObjectURL ??= vi.fn(() => 'blob:mock');
  (globalThis as { URL: typeof URL }).URL.revokeObjectURL ??= vi.fn();
});

const { useStudioStore } = await import('./studio-store.js');

function resetStore(): void {
  useStudioStore.setState(useStudioStore.getInitialState(), true);
}

function makeFile(name = 'clip.mp4'): File {
  return new File([new Uint8Array([1, 2, 3])], name, { type: 'video/mp4' });
}

describe('studio-store: media + timeline reducers', () => {
  beforeEach(resetStore);

  it('importClip adds a ClipMeta entry keyed by a generated id', async () => {
    await useStudioStore.getState().importClip(makeFile('a.mp4'));
    const clips = Object.values(useStudioStore.getState().clips);
    expect(clips).toHaveLength(1);
    expect(clips[0]).toMatchObject({ fileName: 'a.mp4', frameCount: 10, frameCountGuessed: true });
  });

  it('addTrack/addTimelineClip/moveTimelineClip/trimTimelineClip/removeTimelineClip round-trip cleanly', async () => {
    await useStudioStore.getState().importClip(makeFile());
    const clipId = Object.keys(useStudioStore.getState().clips)[0]!;

    const trackA = useStudioStore.getState().addTrack();
    const trackB = useStudioStore.getState().addTrack();
    expect(useStudioStore.getState().tracks.map((t) => t.id)).toEqual([trackA, trackB]);

    const timelineClipId = useStudioStore.getState().addTimelineClip(clipId, trackA, 5);
    let tc = useStudioStore.getState().timelineClips[timelineClipId];
    expect(tc).toMatchObject({ clipId, trackId: trackA, startFrame: 5, inFrame: 0, outFrame: 9 });
    expect(useStudioStore.getState().tracks.find((t) => t.id === trackA)?.clipIds).toEqual([timelineClipId]);

    useStudioStore.getState().moveTimelineClip(timelineClipId, trackB, 12);
    tc = useStudioStore.getState().timelineClips[timelineClipId]!;
    expect(tc.trackId).toBe(trackB);
    expect(tc.startFrame).toBe(12);
    expect(useStudioStore.getState().tracks.find((t) => t.id === trackA)?.clipIds).toEqual([]);
    expect(useStudioStore.getState().tracks.find((t) => t.id === trackB)?.clipIds).toEqual([timelineClipId]);

    useStudioStore.getState().trimTimelineClip(timelineClipId, 2, 7);
    tc = useStudioStore.getState().timelineClips[timelineClipId]!;
    expect(tc.inFrame).toBe(2);
    expect(tc.outFrame).toBe(7);

    useStudioStore.getState().removeTimelineClip(timelineClipId);
    expect(useStudioStore.getState().timelineClips[timelineClipId]).toBeUndefined();
    expect(useStudioStore.getState().tracks.every((t) => !t.clipIds.includes(timelineClipId))).toBe(true);
  });

  it('removeClip leaves no orphaned timelineClips/tracks/objects references', async () => {
    await useStudioStore.getState().importClip(makeFile());
    const clipId = Object.keys(useStudioStore.getState().clips)[0]!;
    const trackId = useStudioStore.getState().addTrack();
    const timelineClipId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    useStudioStore.getState().selectTimelineClip(timelineClipId);

    useStudioStore.getState().removeClip(clipId);

    const state = useStudioStore.getState();
    expect(state.clips[clipId]).toBeUndefined();
    expect(state.timelineClips[timelineClipId]).toBeUndefined();
    expect(state.tracks.find((t) => t.id === trackId)?.clipIds).toEqual([]);
    expect(state.selection.timelineClipId).toBeNull();
  });

  it('reorderTracks reassigns order to match the given id sequence', () => {
    const a = useStudioStore.getState().addTrack();
    const b = useStudioStore.getState().addTrack();
    const c = useStudioStore.getState().addTrack();
    useStudioStore.getState().reorderTracks([c, a, b]);
    const tracks = useStudioStore.getState().tracks;
    expect(tracks.map((t) => t.id)).toEqual([c, a, b]);
    expect(tracks.map((t) => t.order)).toEqual([0, 1, 2]);
  });
});

describe('studio-store: playback/tool/selection', () => {
  beforeEach(resetStore);

  it('setPlayhead/setIsPlaying/setZoom set the raw value (zoom floored at 0.01)', () => {
    useStudioStore.getState().setPlayhead(42);
    expect(useStudioStore.getState().playhead).toBe(42);

    useStudioStore.getState().setIsPlaying(true);
    expect(useStudioStore.getState().isPlaying).toBe(true);

    useStudioStore.getState().setZoom(-5);
    expect(useStudioStore.getState().zoom).toBe(0.01);
    useStudioStore.getState().setZoom(8);
    expect(useStudioStore.getState().zoom).toBe(8);
  });

  it('setTool switches the active tool', () => {
    useStudioStore.getState().setTool('point-add');
    expect(useStudioStore.getState().tool).toBe('point-add');
    useStudioStore.getState().setTool('box');
    expect(useStudioStore.getState().tool).toBe('box');
  });

  it('setTool is a no-op for prompting tools while trackState is running (friction §0.5 invariant)', () => {
    useStudioStore.setState({ trackState: { phase: 'running', clipId: 'c1', frameIndex: 0, frameCount: 10 } });
    useStudioStore.getState().setTool('point-add');
    expect(useStudioStore.getState().tool).toBe('select');

    useStudioStore.getState().setTool('point-remove');
    expect(useStudioStore.getState().tool).toBe('select');

    useStudioStore.getState().setTool('box');
    expect(useStudioStore.getState().tool).toBe('select');

    // Non-prompting tools remain switchable while tracking runs.
    useStudioStore.getState().setTool('pan');
    expect(useStudioStore.getState().tool).toBe('pan');
    useStudioStore.getState().setTool('select');
    expect(useStudioStore.getState().tool).toBe('select');
  });

  it('selectTimelineClip and selectObject are mutually exclusive', () => {
    useStudioStore.getState().selectObject(3);
    expect(useStudioStore.getState().selection).toEqual({ timelineClipId: null, objectId: 3 });

    useStudioStore.getState().selectTimelineClip('tc-1');
    expect(useStudioStore.getState().selection).toEqual({ timelineClipId: 'tc-1', objectId: null });

    useStudioStore.getState().selectObject(7);
    expect(useStudioStore.getState().selection).toEqual({ timelineClipId: null, objectId: 7 });
  });
});

describe('studio-store: state-machine shapes', () => {
  beforeEach(resetStore);

  it('modelStatus starts idle and can be driven through loading -> ready / error', () => {
    expect(useStudioStore.getState().modelStatus).toEqual({ phase: 'idle' });

    useStudioStore.setState({ modelStatus: { phase: 'loading', progress: { phase: 'download' } } });
    expect(useStudioStore.getState().modelStatus).toMatchObject({ phase: 'loading' });

    useStudioStore.setState({ modelStatus: { phase: 'ready', device: 'wasm', quant: 'fp16', totalBytes: 100 } });
    expect(useStudioStore.getState().modelStatus).toMatchObject({ phase: 'ready', device: 'wasm' });

    useStudioStore.setState({ modelStatus: { phase: 'error', message: 'boom' } });
    expect(useStudioStore.getState().modelStatus).toMatchObject({ phase: 'error', message: 'boom' });
  });

  it('trackState starts idle and can be driven through running -> done / error', () => {
    expect(useStudioStore.getState().trackState).toEqual({ phase: 'idle' });

    useStudioStore.setState({ trackState: { phase: 'running', clipId: 'c1', frameIndex: 0, frameCount: 10 } });
    expect(useStudioStore.getState().trackState).toMatchObject({ phase: 'running', frameIndex: 0 });

    useStudioStore.setState({ trackState: { phase: 'done', clipId: 'c1' } });
    expect(useStudioStore.getState().trackState).toEqual({ phase: 'done', clipId: 'c1' });

    useStudioStore.setState({ trackState: { phase: 'error', message: 'nope' } });
    expect(useStudioStore.getState().trackState).toMatchObject({ phase: 'error', message: 'nope' });
  });

  it('exportState starts idle and can be driven through running -> done / error', () => {
    expect(useStudioStore.getState().exportState).toEqual({ phase: 'idle' });

    useStudioStore.setState({ exportState: { phase: 'running', framesDone: 1, frameCount: 10, kind: 'matte' } });
    expect(useStudioStore.getState().exportState).toMatchObject({ phase: 'running', framesDone: 1 });

    useStudioStore.setState({ exportState: { phase: 'done', fileName: 'out.zip', framesExported: 10 } });
    expect(useStudioStore.getState().exportState).toEqual({ phase: 'done', fileName: 'out.zip', framesExported: 10 });

    useStudioStore.setState({ exportState: { phase: 'error', message: 'disk full' } });
    expect(useStudioStore.getState().exportState).toEqual({ phase: 'error', message: 'disk full' });
  });
});

describe('studio-store: notices', () => {
  beforeEach(resetStore);

  it('setNotice/clearNotice set and clear the notice', () => {
    useStudioStore.getState().setNotice({ title: 't', detail: 'd', kind: 'warn' });
    expect(useStudioStore.getState().notice).toEqual({ title: 't', detail: 'd', kind: 'warn' });
    useStudioStore.getState().clearNotice();
    expect(useStudioStore.getState().notice).toBeNull();
  });
});

describe('studio-store: removeObject', () => {
  beforeEach(resetStore);

  it('removes the object and clears selection if it was selected', () => {
    useStudioStore.setState({
      objects: [
        { objectId: 1, clipId: 'c1', color: '#fff', label: 'object 1', promptFrame: 0 },
        { objectId: 2, clipId: 'c1', color: '#000', label: 'object 2', promptFrame: 0 },
      ],
      selection: { timelineClipId: null, objectId: 1 },
    });

    useStudioStore.getState().removeObject('c1', 1);

    const state = useStudioStore.getState();
    expect(state.objects.map((o) => o.objectId)).toEqual([2]);
    expect(state.selection.objectId).toBeNull();
  });
});
