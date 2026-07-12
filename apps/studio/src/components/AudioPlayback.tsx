/**
 * `AudioPlayback.tsx` — headless, store-only timeline audio driver.
 *
 * Mounted ONCE from `App.tsx` (no props, reads/writes only through
 * `useStudioStore`, matching every other component's convention). Renders one
 * hidden `<audio>` per audio-track `TimelineClip` whose span currently covers
 * the playhead, and keeps each element's clock in sync with the project
 * playhead so placed audio tracks actually play during timeline playback.
 *
 * The real work of turning `tracks`/`timelineClips`/`clips`/`playhead` into the
 * set of active audio clips (each carrying its own source-time offset derived
 * from its OWN `inFrame`/`startFrame`, independent of any video track's trim)
 * lives in `../audio/audio-timeline.ts`'s `findActiveAudioClips`; this component
 * is purely the DOM/`<audio>`-element side effect layer around that resolver.
 *
 * PLAYHEAD-SEMANTICS CAVEAT (repeated from the resolver): A/V alignment here
 * presumes the video placement is at `startFrame 0` / `inFrame 0` — the
 * `addClipAsTracks` default. If a video track is trimmed or slid independently
 * of its sibling audio track, the two clocks can diverge; there is no standalone
 * timeline compositor yet, so audio stops when the active clip's video clock
 * ends (PreviewCanvas drives `playhead` from the single active `<video>`, and a
 * frozen playhead is watchdogged below rather than composited).
 *
 * ACCEPTED MVP LIMITS:
 *  - Sub-tolerance A/V drift of up to `DRIFT_TOLERANCE_SEC` (0.15s) is tolerated
 *    between resyncs to avoid seek-stutter — see `DRIFT_TOLERANCE_SEC`.
 *  - `previewVolume` is a single GLOBAL preview level; per-track volume is
 *    future work.
 */
import { useEffect, useMemo, useRef } from 'react';

import { useStudioStore } from '../store/studio-store.js';
import { findActiveAudioClips } from '../audio/audio-timeline.js';

const DRIFT_TOLERANCE_SEC = 0.15; // ~4-5 frames at 30fps; resync only past this to avoid seek-stutter

/**
 * Headless timeline audio driver. See the module doc comment for the full
 * contract (playhead-semantics caveat + accepted MVP limits).
 */
export function AudioPlayback(): React.JSX.Element {
  const tracks = useStudioStore((s) => s.tracks);
  const timelineClips = useStudioStore((s) => s.timelineClips);
  const clips = useStudioStore((s) => s.clips);
  const playhead = useStudioStore((s) => s.playhead);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const previewMuted = useStudioStore((s) => s.previewMuted);
  const previewVolume = useStudioStore((s) => s.previewVolume);

  const activeAudios = useMemo(
    () => findActiveAudioClips(tracks, timelineClips, clips, playhead),
    [tracks, timelineClips, clips, playhead],
  );

  const elementsRef = useRef(new Map<string, HTMLAudioElement>());
  const kickedRef = useRef(new Set<string>()); // per-id 'has had one genuine seek' flags
  const refCallbacksRef = useRef(new Map<string, (el: HTMLAudioElement | null) => void>());
  const endTimesRef = useRef(new Map<string, number>());

  // STABLE per-id ref callbacks (BLOCKER fix: this component re-renders ~30x/s
  // while playing and StrictMode is on — an inline ref closure changes identity
  // every render, so React would call ref(null)/ref(el) each render, pausing and
  // re-kicking audio 30x/s; caching one callback per id keeps identity stable so
  // null fires only on true unmount).
  const getRefCallback = (id: string): ((el: HTMLAudioElement | null) => void) => {
    let cb = refCallbacksRef.current.get(id);
    if (!cb) {
      cb = (el) => {
        if (el) {
          elementsRef.current.set(id, el);
          return;
        }
        // BLOCKER fix: per the HTML spec, a media element that is potentially playing while
        // removed from the document "should play any audio component" and is GC-protected —
        // a detached <audio> KEEPS PLAYING past the trim point unless explicitly paused here.
        const prev = elementsRef.current.get(id);
        prev?.pause();
        elementsRef.current.delete(id);
        kickedRef.current.delete(id);
        refCallbacksRef.current.delete(id);
        endTimesRef.current.delete(id);
      };
      refCallbacksRef.current.set(id, cb);
    }
    return cb;
  };

  // SYNC effect: reconcile each active element's mute/volume, target time, and
  // play/pause state against the store on every relevant change.
  useEffect(() => {
    for (const a of activeAudios) {
      const el = elementsRef.current.get(a.timelineClipId);
      if (!el) continue;
      el.muted = previewMuted; // store-hoisted: the overlay speaker button/slider
      el.volume = previewVolume; // control ALL preview sound, not just the <video>
      endTimesRef.current.set(a.timelineClipId, a.endSourceTimeSec);
      const forceKick = !kickedRef.current.has(a.timelineClipId);
      const needsSeek = Math.abs(el.currentTime - a.sourceTimeSec) > (isPlaying ? DRIFT_TOLERANCE_SEC : 1e-3);
      if (needsSeek || forceKick) {
        // Same Chromium quirk PreviewCanvas's forcedInitialSeekRef guards: a hidden media
        // element does not reliably present its first sample until a REAL (non-no-op) seek.
        // (For audio this is belt-and-braces — play() also starts output — it is NOT the fix
        // for any audible symptom; the load-bearing pieces are play() + currentTime targeting.)
        const base = Math.max(0, a.sourceTimeSec);
        const nudged = base + 0.0001;
        el.currentTime =
          forceKick && !needsSeek
            ? Number.isFinite(el.duration)
              ? Math.min(el.duration, nudged)
              : nudged // NaN-duration-safe: browsers clamp seeks past unknown duration themselves
            : Number.isFinite(el.duration)
              ? Math.min(el.duration, base)
              : base;
        kickedRef.current.add(a.timelineClipId);
      }
      if (isPlaying && el.paused) void el.play().catch(() => undefined); // autoplay-policy safe: always user-gesture-initiated in practice
      else if (!isPlaying && !el.paused) el.pause();
    }
  }, [activeAudios, isPlaying, previewMuted, previewVolume]);

  // WATCHDOG effect (MAJOR fix: Toolbar's play button sets isPlaying with NO
  // active-clip guard and PreviewCanvas's rVFC only ticks while a clip is active
  // and its video plays — with a frozen playhead the sync effect never re-runs,
  // so a play()ed element would run past its span to the end of the source file,
  // defeating trim).
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      for (const [id, el] of elementsRef.current) {
        const end = endTimesRef.current.get(id);
        if (end !== undefined && !el.paused && el.currentTime >= end) el.pause();
      }
    }, 200);
    return () => clearInterval(timer);
  }, [isPlaying]);

  // Unmount cleanup: pause every element so nothing outlives this component.
  useEffect(
    () => () => {
      for (const el of elementsRef.current.values()) el.pause();
    },
    [],
  );

  // Keyed by timelineClipId so a clip leaving the span unmounts its element (the
  // stable ref's null branch pauses it) and re-entering remounts fresh, which is
  // exactly when the kick re-fires.
  return (
    <div aria-hidden style={{ display: 'none' }}>
      {activeAudios.map((a) => (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio key={a.timelineClipId} ref={getRefCallback(a.timelineClipId)} src={a.objectUrl} preload="auto" />
      ))}
    </div>
  );
}
