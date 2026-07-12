/**
 * Small, read-only frame/time helpers for the preview path
 * (studio-contracts.md §4.4). The worker's mp4box+WebCodecs pipeline inside
 * `@websam3/core` stays the source of truth for segmentation-relevant
 * decoding — nothing here touches a `VideoSession`.
 *
 * `probeClipMeta` additionally reports container-level audio presence
 * (`ClipProbe.hasAudio`) via a mediabunny header parse (see `probeHasAudio`),
 * raced against the same `PROBE_TIMEOUT_MS` ceiling that bounds the `<video>`
 * metadata read so import can never hang.
 */

import { ALL_FORMATS, BlobSource, Input } from 'mediabunny';

/** What `probeClipMeta` resolves with — feeds `ClipMeta`'s sizing fields directly. */
export interface ClipProbe {
  durationSec: number;
  /** Estimated fps (no reliable signal from `<video>` metadata alone). */
  fps: number;
  width: number;
  height: number;
  /** `round(durationSec * fps)`, clamped to at least 1. */
  frameCount: number;
  /** Always `true` — corrected once `attachSource()` returns the real count/fps. */
  frameCountGuessed: boolean;
  /**
   * True iff the container demuxes with at least one audio track (mediabunny
   * header parse; false on any parse failure or timeout).
   */
  hasAudio: boolean;
}

/**
 * Default fps assumed until `VideoSession.attachSource` reports the real
 * value — mirrors the demo's `Math.round(duration*fps)` fallback constant.
 * Most consumer video is 24/25/30fps; 30 keeps the frame-count guess in the
 * right ballpark for MediaLibrary thumbnails/duration display pre-model-load.
 */
const DEFAULT_FPS_ESTIMATE = 30;

/**
 * Ceiling on how long `probeClipMeta` waits for the `<video>` element's
 * `loadedmetadata`/`error` events. Documented headless-Chromium flake: for
 * some inputs neither event ever fires, which previously left `importClip`
 * (studio-store.ts) awaiting forever with no error surfaced. 7s comfortably
 * covers real-browser metadata reads (near-instant) while bounding the flake.
 */
const PROBE_TIMEOUT_MS = 7000;

/**
 * Sentinel dimensions used only when the timeout fires — i.e. the browser
 * never told us the real size. `frameCountGuessed` is already `true` in this
 * case (as always), and `activateClip` (session-manager.ts) still corrects
 * everything from `attachSource`'s real values once the model loads.
 */
const FALLBACK_DIM = 0;

function fallbackProbe(): Omit<ClipProbe, 'hasAudio'> {
  return {
    durationSec: 0,
    fps: DEFAULT_FPS_ESTIMATE,
    width: FALLBACK_DIM,
    height: FALLBACK_DIM,
    frameCount: 1,
    frameCountGuessed: true,
  };
}

/**
 * Probe a file's duration/dimensions via a throwaway `<video>` element, used
 * at `importClip` time — before any `Segmenter`/`VideoSession` exists — so
 * `MediaLibrary` can show duration/thumbnails pre-model-load. `fps` is only
 * an estimate; `activateClip` (session-manager.ts) corrects it (and
 * `frameCount`/`frameCountGuessed`) once `attachSource` runs.
 *
 * Races the `<video>` element's events against `PROBE_TIMEOUT_MS`: on real
 * browsers `loadedmetadata` (or a genuine `error`) fires almost immediately
 * and wins the race, so the happy path is unchanged. If neither event fires
 * in time (the flake this guards against), resolves with a best-effort
 * sentinel probe instead of hanging forever — `importClip` always
 * completes, and downstream `attachSource` still corrects the guessed
 * fields once the model loads.
 *
 * Resolves the sizing fields only (`Omit<ClipProbe, 'hasAudio'>`); audio
 * presence is probed separately (`probeHasAudio`) and merged by the public
 * `probeClipMeta` entry point.
 */
function probeVideoElement(file: File | Blob): Promise<Omit<ClipProbe, 'hasAudio'>> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    const url = URL.createObjectURL(file);

    let settled = false;
    const cleanup = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      clearTimeout(timeoutId);
      URL.revokeObjectURL(url);
    };
    const settle = (probe: Omit<ClipProbe, 'hasAudio'>) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(probe);
    };
    const onLoaded = () => {
      const durationSec = Number.isFinite(video.duration) ? video.duration : 0;
      const fps = DEFAULT_FPS_ESTIMATE;
      const frameCount = Math.max(1, Math.round(durationSec * fps));
      settle({
        durationSec,
        fps,
        width: video.videoWidth,
        height: video.videoHeight,
        frameCount,
        frameCountGuessed: true,
      });
    };
    // A genuine decode/format error also resolves with the sentinel probe
    // rather than rejecting — `importClip` has no real fallback path for a
    // rejected probe, and a degraded-but-real clip entry (corrected later by
    // `attachSource`) is strictly better than either a hang or a thrown
    // error mid-import.
    const onError = () => settle(fallbackProbe());
    const timeoutId = setTimeout(() => settle(fallbackProbe()), PROBE_TIMEOUT_MS);

    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('error', onError);
    video.src = url;
    video.load();
  });
}

/**
 * Detect container-level audio presence by demuxing the file header with
 * mediabunny — chosen over `<video>`-element heuristics because the reliable
 * signals aren't available at metadata time in Chromium:
 * `HTMLMediaElement.audioTracks` sits behind the Experimental Web Platform
 * Features flag, and `webkitAudioDecodedByteCount` reads `0` until real decode
 * progress — both give false negatives when we probe. mediabunny@1.50.8 is
 * already a dependency and `BlobSource` does lazy, header-only random-access
 * reads (no decode), so this is a cheap, accurate container introspection.
 *
 * @returns `true` iff the container exposes at least one audio track; `false`
 * on any parse failure (unreadable/unknown container -> treat as video-only).
 */
export async function probeHasAudio(file: File | Blob): Promise<boolean> {
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  try {
    return (await input.getPrimaryAudioTrack()) !== null;
  } catch {
    return false; // unreadable/unknown container -> treat as video-only
  } finally {
    input.dispose();
  }
}

/**
 * Public probe entry point used at `importClip` time: resolves the `<video>`
 * sizing fields together with container-level `hasAudio`.
 *
 * The audio probe is raced against the same `PROBE_TIMEOUT_MS` ceiling that
 * bounds the `<video>` metadata read — an unbounded `probeHasAudio` would
 * reintroduce the exact "import hangs forever" flake `PROBE_TIMEOUT_MS` exists
 * to prevent. `probeHasAudio`'s own `input.dispose()` runs in its `finally`,
 * so a probe that loses the race still frees its reader when it eventually
 * settles. A timed-out audio probe reports `false` (video-only), matching the
 * conservative fallback used everywhere else in this file.
 */
export async function probeClipMeta(file: File | Blob): Promise<ClipProbe> {
  const [base, hasAudio] = await Promise.all([
    probeVideoElement(file),
    Promise.race([
      probeHasAudio(file),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), PROBE_TIMEOUT_MS)),
    ]),
  ]);
  return { ...base, hasAudio };
}

/** Project frame index -> playback time in seconds. */
export function frameIndexToTime(frameIndex: number, fps: number): number {
  return frameIndex / fps;
}

/** Playback time in seconds -> nearest project frame index. */
export function timeToFrameIndex(time: number, fps: number): number {
  return Math.round(time * fps);
}

/**
 * Resolve once `video` lands on `time` (or immediately if already there
 * within a small epsilon) — same pattern as the demo's `seekVideoTo`, used
 * by `PreviewCanvas`'s scrub/track-render loop and `captureFrameBitmap`.
 */
export function seekVideoTo(video: HTMLVideoElement, time: number): Promise<void> {
  const duration = Number.isFinite(video.duration) ? video.duration : time;
  const clamped = Math.max(0, Math.min(time, duration));
  if (Math.abs(video.currentTime - clamped) < 1e-3) return Promise.resolve();
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      resolve();
    };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = clamped;
  });
}

/**
 * Seek + grab a still bitmap of `video` at `atTime` — used for
 * `MediaLibrary` thumbnails. No WebCodecs `VideoDecoder` code lives here for
 * MVP; flagged (studio-contracts.md §4.4) as where a future frame-accurate
 * multi-clip timeline would grow.
 */
export async function captureFrameBitmap(video: HTMLVideoElement, atTime: number): Promise<ImageBitmap> {
  await seekVideoTo(video, atTime);
  return createImageBitmap(video);
}
