/**
 * Small, read-only frame/time helpers for the preview path
 * (studio-contracts.md §4.4). The worker's mp4box+WebCodecs pipeline inside
 * `@websam3/core` stays the source of truth for segmentation-relevant
 * decoding — nothing here touches a `VideoSession`.
 */

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
}

/**
 * Default fps assumed until `VideoSession.attachSource` reports the real
 * value — mirrors the demo's `Math.round(duration*fps)` fallback constant.
 * Most consumer video is 24/25/30fps; 30 keeps the frame-count guess in the
 * right ballpark for MediaLibrary thumbnails/duration display pre-model-load.
 */
const DEFAULT_FPS_ESTIMATE = 30;

/**
 * Probe a file's duration/dimensions via a throwaway `<video>` element, used
 * at `importClip` time — before any `Segmenter`/`VideoSession` exists — so
 * `MediaLibrary` can show duration/thumbnails pre-model-load. `fps` is only
 * an estimate; `activateClip` (session-manager.ts) corrects it (and
 * `frameCount`/`frameCountGuessed`) once `attachSource` runs.
 */
export function probeClipMeta(file: File | Blob): Promise<ClipProbe> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    const url = URL.createObjectURL(file);

    const cleanup = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      URL.revokeObjectURL(url);
    };
    const onLoaded = () => {
      const durationSec = Number.isFinite(video.duration) ? video.duration : 0;
      const fps = DEFAULT_FPS_ESTIMATE;
      const frameCount = Math.max(1, Math.round(durationSec * fps));
      cleanup();
      resolve({
        durationSec,
        fps,
        width: video.videoWidth,
        height: video.videoHeight,
        frameCount,
        frameCountGuessed: true,
      });
    };
    const onError = () => {
      cleanup();
      reject(new Error('probeClipMeta: the browser could not read metadata for this file'));
    };
    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('error', onError);
    video.src = url;
    video.load();
  });
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
