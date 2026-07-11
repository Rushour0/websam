/**
 * Owns per-clip `VideoSession` lifecycle — studio-contracts.md §4.2.
 *
 * `Segmenter`/`VideoSession` instances are NOT store state (zustand doesn't
 * deep-clone and these hold worker-side/non-serializable handles) — they
 * live in this module's `Map<clipId, ClipEntry>`. `MaskTimeline` instances
 * DO live in the store (`StudioState.maskTimelines`, per contracts.md §2);
 * this file creates them but reads/writes them through the `get`/`set` pair
 * every exported function takes, exactly like `studio-store.ts`'s own
 * actions — that's the "take callbacks/setters instead of importing store
 * internals" seam: only `StudioGet`/`StudioSet` (types) flow from the store
 * into this file, never a value import of the store module itself.
 */
import { InvalidStateError } from '@websam3/core';
import type { Prompt } from '@websam3/core';
import { MaskTimeline } from '@websam3/video-editing';
import type { StudioGet, StudioSet, TrackedObject } from '../store/studio-store.js';
import { getSegmenter } from './segmenter-lifecycle.js';

interface ClipEntry {
  session: import('@websam3/core').VideoSession;
  /**
   * Single "current epoch" for the clip, mirroring the demo's scalar
   * `trackEpochRef` — the most recent `invalidateAfter` result, stamped onto
   * every mask in the next `propagate-loop.ts` drain regardless of which
   * object was refined (matches the demo's actual behavior; flagged as a
   * simplification vs. a literal per-object epoch map).
   */
  epoch: number | undefined;
}

const sessions = new Map<string, ClipEntry>();

/** Registered by `propagate-loop.ts` while a track is running on a clip, so `disposeClipSession` can abort it. */
const trackAborts = new Map<string, () => void>();

/** For `propagate-loop.ts`: register an abort callback for an in-flight track on `clipId`. */
export function registerTrackAbort(clipId: string, abort: () => void): void {
  trackAborts.set(clipId, abort);
}

/** For `propagate-loop.ts`: clear the abort callback once a track finishes/aborts. */
export function unregisterTrackAbort(clipId: string): void {
  trackAborts.delete(clipId);
}

/** Palette cycled by object id — kept distinct at both small size and low alpha (mirrors the demo). */
const OBJECT_COLORS = ['#3d85ff', '#ff6b6b', '#f2c94c', '#9b59ff', '#2ecc71', '#ff9f43'];
function colorForObject(id: number): string {
  return OBJECT_COLORS[(id - 1) % OBJECT_COLORS.length] ?? '#3d85ff';
}

function requireSegmenter() {
  const segmenter = getSegmenter();
  if (!segmenter) {
    throw new InvalidStateError(
      'No segmentation model is loaded yet — load the model before working with a clip.',
    );
  }
  return segmenter;
}

function requireEntry(clipId: string): ClipEntry {
  const entry = sessions.get(clipId);
  if (!entry) {
    throw new InvalidStateError(
      `Clip '${clipId}' has no active session — activate it before prompting/tracking.`,
    );
  }
  return entry;
}

/** The clip's underlying `VideoSession`, or `undefined` if not activated. Read by `propagate-loop.ts`/`export.ts`. */
export function getSession(clipId: string) {
  return sessions.get(clipId)?.session;
}

/** The clip's current track/refine epoch (see {@link ClipEntry.epoch}). Read by `propagate-loop.ts`. */
export function getEpoch(clipId: string): number | undefined {
  return sessions.get(clipId)?.epoch;
}

/**
 * Get-or-create a `VideoSession` for `clipId`, `attachSource` the clip's
 * ORIGINAL Blob (never the preview `<video>` element — friction §0.4), and
 * ensure `store.maskTimelines[clipId]` exists. Reconciles the store's
 * `ClipMeta` sizing fields (`frameCount`/`frameCountGuessed`/`fps`) with the
 * real values `attachSource` reports, since `probeClipMeta`'s pre-model-load
 * guess is only an estimate. Idempotent — re-activating an already-attached
 * clip just flips `activeClipId`.
 */
export async function activateClip(get: StudioGet, set: StudioSet, clipId: string): Promise<void> {
  const clip = get().clips[clipId];
  if (!clip) {
    throw new InvalidStateError(`Clip '${clipId}' does not exist in the store.`);
  }

  let entry = sessions.get(clipId);
  if (!entry) {
    const segmenter = requireSegmenter();
    const session = await segmenter.createVideoSession();
    try {
      const raw = await session.attachSource(clip.blob);
      const frameCountGuessed = raw.frameCount === undefined;
      const frameCount = raw.frameCount ?? Math.max(1, Math.round(clip.durationSec * raw.fps));

      entry = { session, epoch: undefined };
      sessions.set(clipId, entry);

      const timeline = new MaskTimeline({
        frameCount,
        fps: raw.fps,
        width: raw.width,
        height: raw.height,
      });

      set((state) => ({
        clips: {
          ...state.clips,
          [clipId]: { ...clip, frameCount, frameCountGuessed, fps: raw.fps, width: raw.width, height: raw.height },
        },
        maskTimelines: { ...state.maskTimelines, [clipId]: timeline },
      }));
    } catch (err) {
      session.dispose();
      throw err;
    }
  }

  set({ activeClipId: clipId });
}

/**
 * Prompt a new object at `frameIndex` on `clipId`. Rejects if a `propagate()`
 * iterator is currently draining (friction §0.5) — belt-and-braces guard
 * behind the store's own `trackState.phase !== 'running'` UI gating.
 */
export async function addPromptObject(
  get: StudioGet,
  set: StudioSet,
  clipId: string,
  frameIndex: number,
  prompts: Prompt[],
): Promise<void> {
  if (get().trackState.phase === 'running') {
    throw new InvalidStateError('Cannot add an object while tracking is running — cancel first.');
  }
  const entry = requireEntry(clipId);
  const { objectId, mask } = await entry.session.addObject({ frameIndex, prompts });

  const tracked: TrackedObject = {
    objectId,
    clipId,
    color: colorForObject(objectId),
    label: `object ${objectId}`,
    promptFrame: frameIndex,
  };

  // The prompt frame is a CONDITIONING frame: its mask comes from addObject,
  // not from the propagate() loop (which only covers frames after it). Persist
  // it into the clip's MaskTimeline here, or that frame is a permanent hole in
  // the timeline and the matte export silently drops it.
  const timeline = get().maskTimelines[clipId];
  if (timeline) timeline.set(String(objectId), frameIndex, mask.toRLE(), getEpoch(clipId) ?? 0);

  set((state) => ({
    objects: [...state.objects, tracked],
    liveMasksAtFrame: { ...state.liveMasksAtFrame, [objectId]: mask },
    selection: { ...state.selection, objectId },
    // Nudge maskTimelines identity so timeline consumers re-render after the
    // in-place set above.
    maskTimelines: { ...state.maskTimelines },
  }));
}

/**
 * Refine `objectId` at `frameIndex` on `clipId`. Invalidates downstream
 * propagated masks on the clip's `MaskTimeline` (mutated in place, then the
 * store is notified per contracts.md §2's `{...get().maskTimelines}`
 * convention) and stashes the new epoch for the next `startTracking` resume.
 */
export async function refineObject(
  get: StudioGet,
  set: StudioSet,
  clipId: string,
  objectId: number,
  frameIndex: number,
  prompts: Prompt[],
): Promise<void> {
  if (get().trackState.phase === 'running') {
    throw new InvalidStateError('Cannot refine an object while tracking is running — cancel first.');
  }
  const entry = requireEntry(clipId);
  const mask = await entry.session.refineObject(objectId, frameIndex, prompts);

  const timeline = get().maskTimelines[clipId];
  if (timeline) {
    entry.epoch = timeline.invalidateAfter(String(objectId), frameIndex);
  }

  set((state) => ({
    liveMasksAtFrame: { ...state.liveMasksAtFrame, [objectId]: mask },
    maskTimelines: { ...state.maskTimelines },
    objects: state.objects.map((o) =>
      o.clipId === clipId && o.objectId === objectId ? { ...o, promptFrame: frameIndex } : o,
    ),
  }));
}

/**
 * Remove `objectId` from `clipId`'s session and clear its live-overlay
 * mask. `TrackedObject`/`selection` bookkeeping is the store's own
 * `removeObject` action's job (contracts.md — this file only touches the
 * session + the overlay cache it owns).
 */
export function removeObject(get: StudioGet, set: StudioSet, clipId: string, objectId: number): void {
  const entry = sessions.get(clipId);
  entry?.session.removeObject(objectId);

  const liveMasksAtFrame = { ...get().liveMasksAtFrame };
  delete liveMasksAtFrame[objectId];
  set({ liveMasksAtFrame });
}

/**
 * Abort any in-flight track, dispose `clipId`'s `VideoSession`, and drop it
 * from the map. Does not touch store state — `studio-store.ts`'s
 * `removeClip` handles `clips`/`maskTimelines`/`objects` cleanup itself.
 */
export function disposeClipSession(clipId: string): void {
  trackAborts.get(clipId)?.();
  trackAborts.delete(clipId);
  const entry = sessions.get(clipId);
  if (!entry) return;
  entry.session.dispose();
  sessions.delete(clipId);
}

/** Dispose every clip session (app unmount). */
export function disposeAllClips(): void {
  for (const clipId of [...sessions.keys()]) disposeClipSession(clipId);
}
