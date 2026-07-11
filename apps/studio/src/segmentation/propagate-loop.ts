/**
 * Drains `VideoSession.propagate()` into a clip's `MaskTimeline` and exposes
 * `startTracking`/`cancelTracking` for `studio-store.ts` — studio-contracts.md §4.3.
 */
import { EpochInvalidatedError, InvalidStateError } from '@websam3/core';
import type { FramePropagationResult } from '@websam3/core';
import type { MaskTimeline } from '@websam3/video-editing';
import type { StudioGet, StudioSet } from '../store/studio-store.js';
import * as sessionManager from './session-manager.js';

/**
 * Drain a propagation iterator into an EXISTING {@link MaskTimeline}.
 *
 * `MaskTimeline.collect` (the public API) is a static factory that always
 * constructs a brand-new timeline from `init` — there is no instance method
 * to resume draining an iterator into a timeline that already holds earlier
 * tracked frames. This is a verbatim reimplementation of
 * `apps/demo/src/VideoTab.tsx`'s `drainInto` helper (friction §0.2, flagged
 * for upstream: an instance `timeline.drain(iterator, {epoch, onFrame})`
 * would remove this duplication), using only public `MaskTimeline` members.
 */
export async function drainInto(
  frames: AsyncIterable<FramePropagationResult>,
  timeline: MaskTimeline,
  epoch: number | undefined,
  onFrame: (frame: FramePropagationResult) => void,
): Promise<void> {
  for await (const frame of frames) {
    for (const mask of frame.masks) {
      timeline.set(String(mask.objectId), frame.frameIndex, mask.toRLE(), epoch);
    }
    onFrame(frame);
  }
}

/**
 * Only one track runs app-wide at a time (mirrors `StudioState.trackState`
 * being a single, not per-clip, field) — same one-active-iterator-per-session
 * rule as the core's contract (friction §0.5), enforced here one level up so
 * `cancelTracking()` (no args, per the store's action signature) always
 * knows what to abort.
 */
let activeTrack: { clipId: string; controller: AbortController } | null = null;

/**
 * Propagate every tracked object on `clipId` from `startFrame` (defaults to
 * the store's current `playhead`) through the end of the clip, draining
 * results into the clip's `MaskTimeline` via {@link drainInto} and updating
 * `playhead`/`liveMasksAtFrame`/`trackState` per yielded frame.
 *
 * Aborts (`AbortError`) and refine-interruptions (`EpochInvalidatedError`)
 * are handled here and resolve normally (idle + a friendly notice for the
 * latter) — anything else is rethrown so `studio-store.ts`'s `startTracking`
 * wrapper maps it onto `trackState: {phase:'error',...}` + a notice.
 */
export async function startTracking(
  get: StudioGet,
  set: StudioSet,
  clipId: string,
  startFrame?: number,
): Promise<void> {
  const session = sessionManager.getSession(clipId);
  const timeline = get().maskTimelines[clipId];
  if (!session || !timeline) {
    throw new InvalidStateError(
      `Clip '${clipId}' has no active session/mask timeline — activate it before tracking.`,
    );
  }
  if (activeTrack) {
    throw new InvalidStateError('A propagate() iterator is already running for another clip.');
  }

  const controller = new AbortController();
  activeTrack = { clipId, controller };
  sessionManager.registerTrackAbort(clipId, () => controller.abort());

  const frameCount = get().clips[clipId]?.frameCount ?? timeline.frameCount;
  const from = startFrame ?? get().playhead;
  // Conditioning (prompt) frames already have their mask (written by
  // addPromptObject). VideoSession.propagate() TRACKS the frames it visits —
  // it does not re-emit a conditioning frame's prompt mask — so tracking a
  // conditioning frame yields an empty mask. Advance the propagate start past
  // any contiguous conditioning frames at `from`, whose masks are already in
  // the timeline. (This is the natural "prompt, then press Track" flow.)
  const promptFrames = new Set(
    get().objects.filter((o) => o.clipId === clipId).map((o) => o.promptFrame),
  );
  let coreStart = from;
  while (promptFrames.has(coreStart)) coreStart += 1;
  set({ trackState: { phase: 'running', clipId, frameIndex: from, frameCount } });

  try {
    const iterator = session.propagate({ startFrame: coreStart, signal: controller.signal });
    await drainInto(iterator, timeline, sessionManager.getEpoch(clipId), (frame) => {
      set((state) => {
        const liveMasksAtFrame = { ...state.liveMasksAtFrame };
        for (const mask of frame.masks) liveMasksAtFrame[mask.objectId] = mask;
        return {
          liveMasksAtFrame,
          playhead: frame.frameIndex,
          trackState: { phase: 'running', clipId, frameIndex: frame.frameIndex, frameCount },
        };
      });
    });
    // The timeline instance was mutated in place by `set()` calls above;
    // bump the outer record once more so any subscriber that only fired on
    // `trackState` transitions also sees the final drained timeline.
    set((state) => ({ maskTimelines: { ...state.maskTimelines } }));
    set({ trackState: controller.signal.aborted ? { phase: 'idle' } : { phase: 'done', clipId } });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      set({ trackState: { phase: 'idle' } });
      return;
    }
    if (err instanceof EpochInvalidatedError) {
      set({
        trackState: { phase: 'idle' },
        notice: {
          title: 'Tracking interrupted by a refine',
          detail:
            'A refine on this object invalidated the in-flight propagation — press Track again to resume from the current frame.',
          kind: 'warn',
        },
      });
      return;
    }
    throw err;
  } finally {
    sessionManager.unregisterTrackAbort(clipId);
    if (activeTrack?.clipId === clipId) activeTrack = null;
  }
}

/** Abort the currently running track, if any (no-op otherwise). */
export function cancelTracking(): void {
  activeTrack?.controller.abort();
}
