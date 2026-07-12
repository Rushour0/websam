/**
 * Pure resolver for audio-track playback (no DOM). AudioPlayback.tsx owns the
 * <audio> elements; this file owns which clips are audible at a playhead and the
 * frame→source-time math, so it is unit-testable in node. KNOWN LIMITATION:
 * store.playhead is driven by PreviewCanvas as the ACTIVE clip's raw source
 * frame (the video placement's startFrame/inFrame are not applied — no timeline
 * compositor yet), while this resolver treats playhead as a PROJECT frame.
 * Audible A/V alignment therefore presumes the active clip's video TimelineClip
 * sits at the identity placement (startFrame 0, inFrame 0 — the addClipAsTracks
 * default); a future compositor removes this precondition.
 */
import type { ClipMeta, TimelineClip, Track } from '../store/studio-store.js';

/** Mirrors frame-source.ts's DEFAULT_FPS_ESTIMATE; used when `fps <= 0`. */
export const DEFAULT_FPS = 30;

/**
 * Project-frame `playhead` translated through THIS TimelineClip's OWN
 * startFrame/inFrame (fully independent of any video track's trim), clamped into
 * `[inFrame, outFrame]`.
 */
export function sourceTimeForPlayhead(
  tc: Pick<TimelineClip, 'startFrame' | 'inFrame' | 'outFrame'>,
  fps: number,
  playhead: number,
): number {
  const f = fps > 0 ? fps : DEFAULT_FPS;
  const offset = Math.min(Math.max(0, playhead - tc.startFrame), tc.outFrame - tc.inFrame);
  return (tc.inFrame + offset) / f;
}

export interface ActiveAudioClipTarget {
  timelineClipId: string;
  clipId: string;
  objectUrl: string;
  sourceTimeSec: number;
  /**
   * Exclusive source-time end of the placement's span, `(tc.outFrame + 1) / f`;
   * AudioPlayback's watchdog pauses elements past it.
   */
  endSourceTimeSec: number;
}

/**
 * For every audio-kind track, resolve the audio TimelineClips whose half-open
 * `[startFrame, startFrame + duration)` span currently covers `playhead`.
 * Deliberately NO dedupe by clipId — two audio tracks over the same source both
 * play.
 */
export function findActiveAudioClips(
  tracks: Track[],
  timelineClips: Record<string, TimelineClip>,
  clips: Record<string, ClipMeta>,
  playhead: number,
): ActiveAudioClipTarget[] {
  const active: ActiveAudioClipTarget[] = [];
  for (const track of tracks) {
    if (track.kind !== 'audio') continue;
    for (const id of track.clipIds) {
      const tc = timelineClips[id];
      if (!tc) continue;
      const clip = clips[tc.clipId];
      if (!clip) continue;
      // Guards above are load-bearing — the store deliberately tolerates
      // dangling references, and a throw in a consumer effect unmounts the app
      // to the ErrorBoundary.
      const duration = tc.outFrame - tc.inFrame + 1;
      if (playhead >= tc.startFrame && playhead < tc.startFrame + duration) {
        active.push({
          timelineClipId: tc.id,
          clipId: tc.clipId,
          objectUrl: clip.objectUrl,
          sourceTimeSec: sourceTimeForPlayhead(tc, clip.fps, playhead),
          endSourceTimeSec: (tc.outFrame + 1) / (clip.fps > 0 ? clip.fps : DEFAULT_FPS),
        });
      }
    }
  }
  return active;
}
