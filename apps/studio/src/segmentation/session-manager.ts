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
 *
 * Tightly coupled with `propagate-loop.ts` (the epoch + busy-guard span
 * both live across these two files) — see `abortAndClearTrack` below.
 */
import { InvalidStateError } from '@websam3/core';
import type { Prompt } from '@websam3/core';
import { MaskTimeline } from '@websam3/video-editing';
import type { ClipMeta, StudioGet, StudioSet, TrackedObject } from '../store/studio-store.js';
import { getSegmenter } from './segmenter-lifecycle.js';
import { abortAndClearTrack } from './propagate-loop.js';

interface ClipEntry {
  session: import('@websam3/core').VideoSession;
  /**
   * The clip's TRUE monotonic epoch, mirroring core's session-wide
   * `VideoSession` epoch (m2-internal-contracts.md §6.1): starts at `0`,
   * bumped by `refineObject`/`removeObject`/reset — NOT by `addObject`.
   * This is a single per-clip counter, not a per-object one: every
   * `timeline.set`/`invalidateAfter`/`drainInto` call for the clip must use
   * THIS value, never a timeline's own (per-object) `epoch()` return —
   * borrowing the latter desyncs multi-object clips, since each object's
   * timeline-internal epoch advances independently of the others.
   */
  epoch: number;
}

const sessions = new Map<string, ClipEntry>();

/**
 * Module-level per-clip busy guard: held for the duration of
 * `addPromptObject`/`refineObject`/`activateClip`'s in-flight worker RPC
 * awaits, so a fast `startTracking` (or another prompt/refine/activate)
 * can't race in underneath one of them (studio-contracts.md friction §0.5).
 */
const busyClips = new Set<string>();

function assertNotBusy(clipId: string): void {
  if (busyClips.has(clipId)) {
    throw new InvalidStateError(
      `Clip '${clipId}' has a segmentation operation already in flight — try again once it finishes.`,
    );
  }
}

/** For `propagate-loop.ts`: true while `clipId` has an in-flight prompt/refine/activate. */
export function isClipBusy(clipId: string): boolean {
  return busyClips.has(clipId);
}

async function withBusy<T>(clipId: string, fn: () => Promise<T>): Promise<T> {
  assertNotBusy(clipId);
  busyClips.add(clipId);
  try {
    return await fn();
  } finally {
    busyClips.delete(clipId);
  }
}

/**
 * In-flight `activateClip` session-creation promise per clip, so two
 * concurrent `activateClip` calls for the same not-yet-activated clip
 * share one `createVideoSession`/`attachSource` instead of each racing an
 * independent one into `sessions`.
 */
const activations = new Map<string, Promise<void>>();

/**
 * Bumped by `disposeClipSession`. A late-resolving `activateClip` (racing a
 * `removeClip`) reads this before its post-await `sessions.set`/store
 * writes and bails instead of resurrecting a removed clip's session.
 */
const generations = new Map<string, number>();

function currentGeneration(clipId: string): number {
  return generations.get(clipId) ?? 0;
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

/** The clip's current true epoch (see {@link ClipEntry.epoch}). Read by `propagate-loop.ts`. */
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
 *
 * Concurrent-safe: two callers racing `activateClip` on the same
 * not-yet-activated clip share one in-flight creation ({@link activations}),
 * and a `removeClip` that races an in-flight activation wins — the
 * activation notices it's been superseded (via {@link generations}) and
 * never resurrects the removed clip.
 */
export async function activateClip(get: StudioGet, set: StudioSet, clipId: string): Promise<void> {
  const clip = get().clips[clipId];
  if (!clip) {
    throw new InvalidStateError(`Clip '${clipId}' does not exist in the store.`);
  }

  if (!sessions.has(clipId)) {
    let pending = activations.get(clipId);
    if (!pending) {
      pending = createClipSession(get, set, clipId, clip);
      activations.set(clipId, pending);
      pending.finally(() => {
        if (activations.get(clipId) === pending) activations.delete(clipId);
      });
    }
    await pending;
  }

  // A concurrent `removeClip`/`disposeClipSession` may have torn the
  // session down (or kept it from ever being created) while this call was
  // awaiting activation — only flip the active clip if the session is
  // actually present, so a late-resolving activation can't resurrect a
  // removed clip.
  if (sessions.has(clipId)) set({ activeClipId: clipId });
}

async function createClipSession(
  get: StudioGet,
  set: StudioSet,
  clipId: string,
  clip: ClipMeta,
): Promise<void> {
  const generation = currentGeneration(clipId);
  const segmenter = requireSegmenter();
  const session = await segmenter.createVideoSession();
  try {
    const raw = await withBusy(clipId, () => session.attachSource(clip.blob));
    if (currentGeneration(clipId) !== generation) {
      // Superseded by a remove/dispose while attaching — don't resurrect.
      session.dispose();
      return;
    }
    const frameCountGuessed = raw.frameCount === undefined;
    const frameCount = raw.frameCount ?? Math.max(1, Math.round(clip.durationSec * raw.fps));

    sessions.set(clipId, { session, epoch: 0 });

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

/**
 * Prompt a new object at `frameIndex` on `clipId`. Rejects if a `propagate()`
 * iterator is currently draining (friction §0.5) — belt-and-braces guard
 * behind the store's own `trackState.phase !== 'running'` UI gating — and if
 * `clipId` has another prompt/refine/activate in flight.
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
  const { objectId, mask } = await withBusy(clipId, () => entry.session.addObject({ frameIndex, prompts }));

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
  // the timeline and the matte export silently drops it. `addObject` does NOT
  // bump the clip's epoch (m2-internal-contracts.md §6.1) — stamp the
  // CURRENT epoch, unchanged.
  const timeline = get().maskTimelines[clipId];
  if (timeline) timeline.set(String(objectId), frameIndex, mask.toRLE(), entry.epoch);

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
 * Refine `objectId` at `frameIndex` on `clipId`. Bumps the clip's TRUE epoch
 * (see {@link ClipEntry.epoch}), invalidates downstream propagated masks on
 * the clip's `MaskTimeline`, and persists the refined mask at `frameIndex`
 * under the new epoch — mirroring `addPromptObject`'s conditioning-frame
 * write, since a refine's own mask is likewise never re-emitted by
 * `propagate()` and would otherwise be a permanent stale hole.
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
  const mask = await withBusy(clipId, () => entry.session.refineObject(objectId, frameIndex, prompts));

  entry.epoch += 1;
  const timeline = get().maskTimelines[clipId];
  if (timeline) {
    timeline.invalidateAfter(String(objectId), frameIndex);
    timeline.set(String(objectId), frameIndex, mask.toRLE(), entry.epoch);
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
 * session + the overlay cache it owns). Bumps the clip's TRUE epoch (see
 * {@link ClipEntry.epoch}), matching core's `removeObject` epoch bump
 * (m2-internal-contracts.md §6.1).
 */
export function removeObject(get: StudioGet, set: StudioSet, clipId: string, objectId: number): void {
  const entry = sessions.get(clipId);
  entry?.session.removeObject(objectId);
  if (entry) entry.epoch += 1;

  const liveMasksAtFrame = { ...get().liveMasksAtFrame };
  delete liveMasksAtFrame[objectId];
  set({ liveMasksAtFrame });
}

/**
 * Abort any in-flight track, dispose `clipId`'s `VideoSession`, and drop it
 * from the map. Does not touch store state — `studio-store.ts`'s
 * `removeClip` handles `clips`/`maskTimelines`/`objects` cleanup itself.
 *
 * Bumps `clipId`'s generation FIRST so a concurrently-awaiting
 * `activateClip` (see {@link createClipSession}) recognizes it's been
 * superseded before it writes anything back. Aborts via
 * `propagate-loop.ts`'s {@link abortAndClearTrack}, which clears its
 * module-level `activeTrack` lock synchronously — aborting alone would
 * leave that lock held until the aborted iterator unwinds, causing a
 * transient spurious "already running" for the next `startTracking`.
 */
export function disposeClipSession(clipId: string): void {
  generations.set(clipId, currentGeneration(clipId) + 1);
  abortAndClearTrack(clipId);
  const entry = sessions.get(clipId);
  if (!entry) return;
  entry.session.dispose();
  sessions.delete(clipId);
}

/** Dispose every clip session (app unmount). */
export function disposeAllClips(): void {
  for (const clipId of [...sessions.keys()]) disposeClipSession(clipId);
}
