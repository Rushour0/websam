/**
 * Studio store unit tests for realistic TIMELINE-EDIT flows: import -> place
 * on a track -> trim -> move/reorder -> multi-clip geometry, plus the
 * drag/delete/remix primitives (removeTrack, splitTimelineClip,
 * duplicateTimelineClip). Pure-node, no browser, no segmenter — same mocking
 * pattern as `studio-store.test.ts` (studio-contracts.md §6.2). The async
 * segmentation actions themselves are exercised end-to-end by
 * `../e2e/edit-flows.browser.test.ts`, not here.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../video/frame-source.js', () => ({
  probeClipMeta: vi.fn(async (file: File) => ({
    durationSec: 1,
    fps: 10,
    width: 256,
    height: 256,
    frameCount: 10,
    frameCountGuessed: true,
    // `file` unused but kept to mirror the real probe's signature.
  })),
}));

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

async function importAndTrack(fileName = 'clip.mp4'): Promise<{ clipId: string; trackId: string }> {
  await useStudioStore.getState().importClip(makeFile(fileName));
  const clipId = Object.keys(useStudioStore.getState().clips).find(
    (id) => useStudioStore.getState().clips[id]!.fileName === fileName,
  )!;
  const trackId = useStudioStore.getState().addTrack();
  return { clipId, trackId };
}

describe('edit-flows (unit): trim geometry', () => {
  beforeEach(resetStore);

  it('trimTimelineClip sets the requested in/out range for an in-range trim', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    // Sanity: freshly-placed clip defaults to the full source range (frameCount=10 -> [0,9]).
    expect(useStudioStore.getState().timelineClips[tcId]).toMatchObject({ inFrame: 0, outFrame: 9 });

    useStudioStore.getState().trimTimelineClip(tcId, 2, 6);
    const tc = useStudioStore.getState().timelineClips[tcId]!;
    expect(tc.inFrame).toBe(2);
    expect(tc.outFrame).toBe(6);
  });

  /**
   * studio-contracts.md §6.2 (store-logic unit test plan): "`trimTimelineClip`
   * clamps `inFrame < outFrame` within `[0, frameCount)`." The current
   * implementation (`studio-store.ts`'s `trimTimelineClip` action) assigns
   * `inFrame`/`outFrame` verbatim with NO clamping at all — this test
   * documents the gap against the documented contract. See notes/findings:
   * this is a real, precise, reportable bug, not a test bug.
   */
  it('[contract] trimTimelineClip must clamp inFrame < outFrame within [0, frameCount) — currently does not', async () => {
    const { clipId, trackId } = await importAndTrack(); // clip.frameCount === 10 -> valid range [0, 10)
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);

    // Out-of-range + inverted request: inFrame beyond frameCount, outFrame negative.
    useStudioStore.getState().trimTimelineClip(tcId, 50, -3);
    const tc = useStudioStore.getState().timelineClips[tcId]!;

    expect(tc.inFrame, 'inFrame must be clamped into [0, frameCount)').toBeGreaterThanOrEqual(0);
    expect(tc.inFrame).toBeLessThan(10);
    expect(tc.outFrame, 'outFrame must be clamped into [0, frameCount)').toBeGreaterThanOrEqual(0);
    expect(tc.outFrame).toBeLessThan(10);
    expect(tc.inFrame, 'inFrame must stay strictly less than outFrame (no zero/negative-length trim)').toBeLessThan(tc.outFrame);
  });

  it('[contract] trimTimelineClip must reject/clamp a zero-length trim request (inFrame === outFrame)', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);

    useStudioStore.getState().trimTimelineClip(tcId, 4, 4);
    const tc = useStudioStore.getState().timelineClips[tcId]!;
    expect(tc.inFrame, 'a zero-length trim (inFrame === outFrame) must not be stored verbatim').toBeLessThan(tc.outFrame);
  });
});

describe('edit-flows (unit): move/reorder geometry', () => {
  beforeEach(resetStore);

  it('moveTimelineClip updates startFrame/trackId and keeps track.clipIds consistent across two tracks', async () => {
    const { clipId, trackId: trackA } = await importAndTrack();
    const trackB = useStudioStore.getState().addTrack();

    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackA, 0);
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackA, 20);
    expect(useStudioStore.getState().tracks.find((t) => t.id === trackA)?.clipIds).toEqual([tc1, tc2]);

    useStudioStore.getState().moveTimelineClip(tc1, trackB, 100);
    const state = useStudioStore.getState();
    expect(state.timelineClips[tc1]).toMatchObject({ trackId: trackB, startFrame: 100 });
    expect(state.tracks.find((t) => t.id === trackA)?.clipIds).toEqual([tc2]);
    expect(state.tracks.find((t) => t.id === trackB)?.clipIds).toEqual([tc1]);
    // tc2 untouched.
    expect(state.timelineClips[tc2]).toMatchObject({ trackId: trackA, startFrame: 20 });
  });

  it('moveTimelineClip within the SAME track just updates startFrame, without duplicating the clip id in clipIds', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);

    useStudioStore.getState().moveTimelineClip(tc1, trackId, 50);
    const state = useStudioStore.getState();
    expect(state.timelineClips[tc1]).toMatchObject({ trackId, startFrame: 50 });
    expect(state.tracks.find((t) => t.id === trackId)?.clipIds).toEqual([tc1]);
  });

  it('reorderTracks reassigns order to match the given id sequence (geometry-only, no clip mutation)', async () => {
    const a = useStudioStore.getState().addTrack();
    const b = useStudioStore.getState().addTrack();
    const c = useStudioStore.getState().addTrack();
    const { clipId } = await importAndTrack();
    const tc = useStudioStore.getState().addTimelineClip(clipId, b, 0);

    useStudioStore.getState().reorderTracks([c, a, b]);
    const state = useStudioStore.getState();
    expect(state.tracks.map((t) => t.id)).toEqual([c, a, b]);
    expect(state.tracks.map((t) => t.order)).toEqual([0, 1, 2]);
    // Reordering tracks must not disturb clip placement.
    expect(state.tracks.find((t) => t.id === b)?.clipIds).toEqual([tc]);
    expect(state.timelineClips[tc]).toMatchObject({ trackId: b, startFrame: 0 });
  });

  /**
   * The store contract (studio-contracts.md §2/§6.2) does not document an
   * overlap-prevention rule for `addTimelineClip`/`moveTimelineClip` — this
   * test records the CURRENT (permissive) behavior rather than asserting an
   * unspecified invariant, so a future contract change here is a deliberate
   * decision, not a silent regression.
   */
  it('[observation] moveTimelineClip does not currently prevent two clips overlapping on the same track', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackId, 0); // occupies frames [0, 9] at startFrame 0
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackId, 100);

    useStudioStore.getState().moveTimelineClip(tc2, trackId, 0); // now fully overlaps tc1
    const state = useStudioStore.getState();
    expect(state.timelineClips[tc1]?.startFrame).toBe(0);
    expect(state.timelineClips[tc2]?.startFrame).toBe(0);
    expect(state.tracks.find((t) => t.id === trackId)?.clipIds).toEqual([tc1, tc2]);
  });
});

describe('edit-flows (unit): playhead clamping', () => {
  beforeEach(resetStore);

  /**
   * studio-contracts.md §6.2: "`setPlayhead` clamps to `[0, projectDuration)`."
   * The current implementation (`setPlayhead: (frame) => set({ playhead: frame })`)
   * assigns the raw value with no clamping — this documents the gap.
   */
  it('[contract] setPlayhead must clamp to a non-negative frame index — currently accepts negative values verbatim', () => {
    useStudioStore.getState().setPlayhead(-5);
    expect(useStudioStore.getState().playhead, 'playhead must never go negative').toBeGreaterThanOrEqual(0);
  });
});

describe('edit-flows (unit): removeTimelineClip mid-edit', () => {
  beforeEach(resetStore);

  it('removing one clip from a multi-clip track leaves the remaining clip and track geometry intact', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackId, 20);
    useStudioStore.getState().trimTimelineClip(tc1, 1, 5);

    useStudioStore.getState().removeTimelineClip(tc1);

    const state = useStudioStore.getState();
    expect(state.timelineClips[tc1]).toBeUndefined();
    expect(state.timelineClips[tc2]).toMatchObject({ startFrame: 20 });
    expect(state.tracks.find((t) => t.id === trackId)?.clipIds).toEqual([tc2]);
  });
});

describe('edit-flows (unit): removeTrack', () => {
  beforeEach(resetStore);

  it('removes the track and every timelineClip on it, leaving other tracks/clips intact', async () => {
    const { clipId, trackId: trackA } = await importAndTrack();
    const trackB = useStudioStore.getState().addTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackA, 0);
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackB, 0);

    useStudioStore.getState().removeTrack(trackA);

    const state = useStudioStore.getState();
    expect(state.tracks.map((t) => t.id)).toEqual([trackB]);
    expect(state.timelineClips[tc1]).toBeUndefined();
    expect(state.timelineClips[tc2]).toMatchObject({ trackId: trackB });
    expect(state.tracks.find((t) => t.id === trackB)?.clipIds).toEqual([tc2]);
  });

  it('renormalizes surviving track.order to 0..n-1 preserving relative order', () => {
    const a = useStudioStore.getState().addTrack();
    const b = useStudioStore.getState().addTrack();
    const c = useStudioStore.getState().addTrack();

    useStudioStore.getState().removeTrack(b);

    const state = useStudioStore.getState();
    expect(state.tracks.map((t) => t.id)).toEqual([a, c]);
    expect(state.tracks.map((t) => t.order)).toEqual([0, 1]);
  });

  it('clears selection.timelineClipId iff the selected clip lived on the removed track', async () => {
    const { clipId, trackId: trackA } = await importAndTrack();
    const trackB = useStudioStore.getState().addTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackA, 0);
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackB, 0);

    // Selected clip lives on the removed track -> selection cleared.
    useStudioStore.getState().selectTimelineClip(tc1);
    useStudioStore.getState().removeTrack(trackA);
    expect(useStudioStore.getState().selection.timelineClipId).toBeNull();

    // Selected clip lives on a surviving track -> selection preserved.
    useStudioStore.getState().selectTimelineClip(tc2);
    useStudioStore.getState().removeTrack(trackA); // already gone; no-op re: selection
    expect(useStudioStore.getState().selection.timelineClipId).toBe(tc2);
  });

  it('is a no-op on an unknown track id', async () => {
    const { clipId, trackId } = await importAndTrack();
    useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    const before = useStudioStore.getState();
    const tracksBefore = before.tracks;
    const clipKeysBefore = Object.keys(before.timelineClips);

    useStudioStore.getState().removeTrack('nope');

    const after = useStudioStore.getState();
    expect(after.tracks).toEqual(tracksBefore);
    expect(Object.keys(after.timelineClips)).toEqual(clipKeysBefore);
  });
});

describe('edit-flows (unit): splitTimelineClip', () => {
  beforeEach(resetStore);

  it('splits an untrimmed clip into left/right halves at a project frame', async () => {
    const { clipId, trackId } = await importAndTrack();
    // frameCount=10 -> fresh clip at start 0 occupies project [0,9] with in 0/out 9.
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);

    const newId = useStudioStore.getState().splitTimelineClip(tcId, 4);
    expect(newId).not.toBeNull();
    expect(newId).not.toBe(tcId);

    const state = useStudioStore.getState();
    expect(state.timelineClips[tcId]).toMatchObject({ startFrame: 0, inFrame: 0, outFrame: 3, trackId, clipId });
    expect(state.timelineClips[newId!]).toMatchObject({ startFrame: 4, inFrame: 4, outFrame: 9, trackId, clipId });
    const clipIds = state.tracks.find((t) => t.id === trackId)?.clipIds;
    expect(clipIds).toHaveLength(2);
    expect(clipIds).toContain(tcId);
    expect(clipIds).toContain(newId);
  });

  it('splits a trimmed, moved clip using the raw project playhead frame', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    useStudioStore.getState().trimTimelineClip(tcId, 2, 8); // source [2,8]
    useStudioStore.getState().moveTimelineClip(tcId, trackId, 10); // occupies project [10,16]

    const newId = useStudioStore.getState().splitTimelineClip(tcId, 13);
    expect(newId).not.toBeNull();

    const state = useStudioStore.getState();
    expect(state.timelineClips[tcId]).toMatchObject({ startFrame: 10, inFrame: 2, outFrame: 4 });
    expect(state.timelineClips[newId!]).toMatchObject({ startFrame: 13, inFrame: 5, outFrame: 8 });
  });

  it('rejects splits at either boundary but allows a 1-frame right half', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0); // project [0,9]

    // Boundary: atFrame at the start or one-past-the-last frame -> null, nothing mutated.
    expect(useStudioStore.getState().splitTimelineClip(tcId, 0)).toBeNull();
    expect(useStudioStore.getState().splitTimelineClip(tcId, 10)).toBeNull();
    expect(useStudioStore.getState().timelineClips[tcId]).toMatchObject({ startFrame: 0, inFrame: 0, outFrame: 9 });
    expect(useStudioStore.getState().tracks.find((t) => t.id === trackId)?.clipIds).toEqual([tcId]);

    // Final assertions: splitting at the last frame is legal (1-frame right half).
    const newId = useStudioStore.getState().splitTimelineClip(tcId, 9);
    expect(newId).not.toBeNull();
    const state = useStudioStore.getState();
    expect(state.timelineClips[tcId]).toMatchObject({ startFrame: 0, inFrame: 0, outFrame: 8 });
    expect(state.timelineClips[newId!]).toMatchObject({ startFrame: 9, inFrame: 9, outFrame: 9 });
  });

  it('returns null and mutates nothing for an unknown clip id', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    const before = useStudioStore.getState().timelineClips[tcId];

    expect(useStudioStore.getState().splitTimelineClip('nope', 3)).toBeNull();
    expect(useStudioStore.getState().timelineClips[tcId]).toEqual(before);
  });
});

describe('edit-flows (unit): duplicateTimelineClip', () => {
  beforeEach(resetStore);

  it('places the clone immediately after an untrimmed source, leaving the source unchanged', async () => {
    const { clipId, trackId } = await importAndTrack();
    // frameCount=10 -> source at start 0 occupies project [0,9] with in 0/out 9.
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);

    const dupId = useStudioStore.getState().duplicateTimelineClip(tcId);
    expect(dupId).not.toBeNull();

    const state = useStudioStore.getState();
    expect(state.timelineClips[dupId!]).toMatchObject({ clipId, trackId, inFrame: 0, outFrame: 9, startFrame: 10 });
    expect(state.timelineClips[tcId]).toMatchObject({ startFrame: 0, inFrame: 0, outFrame: 9 });
    expect(state.tracks.find((t) => t.id === trackId)?.clipIds).toHaveLength(2);
  });

  it('preserves the trimmed source range on the clone and places it one duration later', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tcId = useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    useStudioStore.getState().trimTimelineClip(tcId, 2, 6); // source [2,6], duration 5

    const dupId = useStudioStore.getState().duplicateTimelineClip(tcId);
    expect(dupId).not.toBeNull();
    expect(useStudioStore.getState().timelineClips[dupId!]).toMatchObject({ inFrame: 2, outFrame: 6, startFrame: 5 });
  });

  it('shifts the clone right past any same-track overlap', async () => {
    const { clipId, trackId } = await importAndTrack();
    const tc1 = useStudioStore.getState().addTimelineClip(clipId, trackId, 0); // project [0,9]
    const tc2 = useStudioStore.getState().addTimelineClip(clipId, trackId, 10); // project [10,19]

    const dupId = useStudioStore.getState().duplicateTimelineClip(tc1);
    expect(dupId).not.toBeNull();

    const state = useStudioStore.getState();
    const overlaps = (a: { startFrame: number; inFrame: number; outFrame: number }, b: typeof a): boolean => {
      const aStart = a.startFrame;
      const aEnd = a.startFrame + (a.outFrame - a.inFrame);
      const bStart = b.startFrame;
      const bEnd = b.startFrame + (b.outFrame - b.inFrame);
      return aStart <= bEnd && bStart <= aEnd;
    };
    const clip1 = state.timelineClips[tc1]!;
    const clip2 = state.timelineClips[tc2]!;
    const dup = state.timelineClips[dupId!]!;
    expect(overlaps(clip1, clip2)).toBe(false);
    expect(overlaps(clip1, dup)).toBe(false);
    expect(overlaps(clip2, dup)).toBe(false);
    // Pins today's cascade placement: desiredStart 10 collides with tc2, so it shifts past [10,19].
    expect(dup.startFrame).toBeGreaterThanOrEqual(20);
  });

  it('returns null and mutates nothing for an unknown clip id', async () => {
    const { clipId, trackId } = await importAndTrack();
    useStudioStore.getState().addTimelineClip(clipId, trackId, 0);
    const before = useStudioStore.getState();
    const clipKeysBefore = Object.keys(before.timelineClips);
    const tracksBefore = before.tracks;

    expect(useStudioStore.getState().duplicateTimelineClip('nope')).toBeNull();

    const after = useStudioStore.getState();
    expect(Object.keys(after.timelineClips)).toEqual(clipKeysBefore);
    expect(after.tracks).toEqual(tracksBefore);
  });
});
