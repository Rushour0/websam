import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createSegmenter } from '@websam3/core';
import type {
  FramePropagationResult,
  LoadProgressEvent,
  MaskResult,
  Prompt,
  Segmenter,
  VideoSession,
} from '@websam3/core';
import { AlphaMatteExporter, MaskTimeline } from '@websam3/video-editing';
// Vite bundles the core worker entry as a dedicated worker chunk and hands us
// its URL — same escape hatch ImageTab uses.
import segmenterWorkerUrl from '@websam3/core/worker?worker&url';

/**
 * Where model manifests + weights are served from. Local dev default is
 * `/models/` (populate `apps/demo/public/models/` via
 * `tools/goldens/fetch-models.mjs`); deployments override with
 * `VITE_WEBSAM_MODELS`.
 */
const MODEL_BASE_URL = (import.meta.env.VITE_WEBSAM_MODELS as string | undefined) ?? '/models/';

/** Optional override for onnxruntime-web's .wasm/.mjs asset base inside the worker. */
const ORT_WASM_PATHS = import.meta.env.VITE_ORT_WASM_PATHS as string | undefined;

/**
 * The tier this tab drives. EdgeTAM is Apache-2.0 (no license gate) and
 * ships tens-of-MB weights — much lighter than the SAM 3 tracker tier the
 * image tab uses.
 */
const MODEL_ID = 'edgetam';
const APPROX_DOWNLOAD = 'tens of MB';

/** Prefetch depth for the propagation pull-credit stream (matches the core default). */
const PREFETCH = 4;

/** Palette cycled by object id — kept distinct at both small size and low alpha. */
const OBJECT_COLORS = ['#3d85ff', '#ff6b6b', '#f2c94c', '#9b59ff', '#2ecc71', '#ff9f43'];

function colorForObject(id: number): string {
  return OBJECT_COLORS[(id - 1) % OBJECT_COLORS.length] ?? '#3d85ff';
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/** One canvas click, in source-frame pixel coordinates. 1 = positive, 0 = negative. */
interface ClickPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

/** A tracked object's client-side bookkeeping (prompts + display color). */
interface TrackedObject {
  id: number;
  color: string;
  /** Frame index the current `points` were prompted at. */
  promptFrame: number;
  points: ClickPoint[];
}

/** Friendly rendering of a (usually typed Websam) error. */
interface ErrorInfo {
  title: string;
  detail: string;
}

type ModelState =
  | { status: 'idle' }
  | { status: 'loading'; event?: LoadProgressEvent; phases: string[] }
  | { status: 'ready'; device: string; quant: string; totalBytes: number }
  | { status: 'error'; error: ErrorInfo };

type SourceState =
  | { status: 'none' }
  | { status: 'attaching'; fileName: string }
  | {
      status: 'ready';
      fileName: string;
      fps: number;
      width: number;
      height: number;
      frameCount: number;
      frameCountGuessed: boolean;
    }
  | { status: 'error'; error: ErrorInfo };

type TrackState =
  | { status: 'idle' }
  | { status: 'running'; frameIndex: number; frameCount: number }
  | { status: 'done' }
  | { status: 'error'; error: ErrorInfo };

type ExportState =
  | { status: 'idle' }
  | { status: 'running'; framesDone: number; frameCount: number }
  | { status: 'done'; framesExported: number; fileName: string }
  | { status: 'error'; error: ErrorInfo };

/**
 * Map a thrown value onto a friendly notice, keyed on the websam error
 * taxonomy — extends {@link describeError} in ImageTab.tsx with the two
 * codes unique to the video path (`EPOCH_INVALIDATED`, and the
 * `HTMLVideoElement` deferral spelled out as `NOT_IMPLEMENTED`).
 */
function describeError(err: unknown): ErrorInfo {
  const message = err instanceof Error ? err.message : String(err);
  const name = err instanceof Error ? err.name : '';
  const code =
    typeof err === 'object' && err !== null && 'code' in err
      ? String((err as { code: unknown }).code)
      : '';

  if (code === 'CROSS_ORIGIN_ISOLATION_REQUIRED' || name === 'CrossOriginIsolationRequiredError') {
    return {
      title: 'Cross-origin isolation required',
      detail:
        'This feature needs crossOriginIsolated === true (COOP/COEP headers — the dev server ' +
        'already sends them). ' +
        message,
    };
  }
  if (code === 'UNSUPPORTED_DEVICE' || name === 'UnsupportedDeviceError') {
    return {
      title: 'No supported compute device',
      detail: `This browser cannot run the '${MODEL_ID}' tier's video graphs. ${message}`,
    };
  }
  if (code === 'WEIGHT_VERIFY_FAILED' || name === 'WeightVerifyError') {
    return {
      title: 'Model weights unavailable or failed verification',
      detail:
        `Could not fetch + verify weights from '${MODEL_BASE_URL}'. Running locally? Populate ` +
        'apps/demo/public/models/ (tools/goldens/fetch-models.mjs) or point VITE_WEBSAM_MODELS at a ' +
        'model host. ' +
        message,
    };
  }
  if (code === 'OUT_OF_MEMORY' || name === 'OutOfMemoryError') {
    return {
      title: 'Out of memory',
      detail: `The device ran out of memory. ${message}`,
    };
  }
  if (code === 'EPOCH_INVALIDATED' || name === 'EpochInvalidatedError') {
    return {
      title: 'Tracking interrupted by a refine',
      detail:
        'A refine on this object invalidated the in-flight propagation — expected when you refine ' +
        `mid-track. Press Track again to resume from the current frame. ${message}`,
    };
  }
  if (code === 'NOT_IMPLEMENTED' || name === 'NotImplementedError') {
    return {
      title: 'Not implemented yet',
      detail: /HTMLVideoElement/.test(message)
        ? `Live-element video sources land in M4 — this demo always attaches a File/Blob. ${message}`
        : message,
    };
  }
  if (code === 'INVALID_STATE' || name === 'InvalidStateError') {
    return { title: 'Invalid state', detail: message };
  }
  return { title: 'Something went wrong', detail: message };
}

function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${n} B`;
}

const PHASE_LABELS: Record<string, string> = {
  manifest: 'fetching manifest',
  download: 'downloading',
  verify: 'verifying',
  'offline-cache': 'loaded from cache',
  compile: 'compiling',
  ready: 'ready',
};

function phaseLabel(event: LoadProgressEvent): string {
  const label = PHASE_LABELS[event.phase] ?? event.phase;
  return event.file !== undefined ? `${label} · ${event.file}` : label;
}

function appendPhase(phases: string[], event: LoadProgressEvent): string[] {
  const label = phaseLabel(event);
  return phases[phases.length - 1] === label ? phases : [...phases, label];
}

function LoadProgress({ event, phases }: { event?: LoadProgressEvent; phases: string[] }) {
  const determinate =
    event?.phase === 'download' &&
    event.loaded !== undefined &&
    event.total !== undefined &&
    event.total > 0;
  const percent = determinate ? Math.min(100, (event.loaded! / event.total!) * 100) : 0;
  return (
    <div className="progress-block">
      <p className="progress-line">
        <span>{event ? phaseLabel(event) : 'starting…'}</span>
        {determinate && (
          <span className="muted mono-inline">
            {formatBytes(event.loaded!)} / {formatBytes(event.total!)}
          </span>
        )}
      </p>
      <div
        className={`progress-track${determinate ? '' : ' progress-indeterminate'}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={determinate ? Math.round(percent) : undefined}
      >
        <div className="progress-fill" style={determinate ? { width: `${percent}%` } : undefined} />
      </div>
      {phases.length > 0 && (
        <ol className="phase-log">
          {phases.map((phase, index) => (
            <li key={`${index}-${phase}`}>{phase}</li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** Resolve once the `<video>` lands on `time` (or immediately if already there). */
function seekVideoTo(video: HTMLVideoElement, time: number): Promise<void> {
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

function toPrompts(points: ClickPoint[]): Prompt[] {
  return points.map((p) => ({ type: 'point', x: p.x, y: p.y, label: p.label }));
}

/**
 * Drain a propagation iterator into an EXISTING {@link MaskTimeline}.
 *
 * `MaskTimeline.collect` (the public API) is a static factory that always
 * constructs a brand-new timeline from `init` — there is no instance method
 * to resume draining an iterator into a timeline that already holds earlier
 * tracked frames. That breaks the refine → invalidateAfter → re-collect
 * flow the timeline's own docs describe ("re-collects into the SAME
 * timeline"). This mirrors `collect`'s body exactly, using only public
 * `MaskTimeline` members (`set`), so the demo can resume into one timeline
 * across a Track → Refine → Track cycle. See the wave-3 handoff notes for
 * the upstream flag.
 */
async function drainInto(
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
 * Scrub-less rotobrush MVP (M2 §8): pick/drop an mp4, load EdgeTAM, click to
 * prompt objects on frame 0, Track to propagate live, pause + refine, then
 * export an alpha-matte PNG sequence. No timeline scrubbing, no backward
 * tracking — those land later.
 */
export function VideoTab() {
  const [modelState, setModelState] = useState<ModelState>({ status: 'idle' });
  const [sourceState, setSourceState] = useState<SourceState>({ status: 'none' });
  const [trackState, setTrackState] = useState<TrackState>({ status: 'idle' });
  const [exportState, setExportState] = useState<ExportState>({ status: 'idle' });
  const [objects, setObjects] = useState<TrackedObject[]>([]);
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [notice, setNotice] = useState<ErrorInfo | null>(null);
  const [dragging, setDragging] = useState(false);

  const segmenterRef = useRef<Segmenter | null>(null);
  const sessionRef = useRef<VideoSession | null>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const pendingFileRef = useRef<File | null>(null);

  const objectsRef = useRef<Map<number, TrackedObject>>(new Map());
  const selectedObjectIdRef = useRef<number | null>(null);
  const masksAtFrameRef = useRef<Map<number, MaskResult>>(new Map());
  const timelineRef = useRef<MaskTimeline | null>(null);
  const timelineInitRef = useRef<{ frameCount: number; fps: number; width: number; height: number } | null>(
    null,
  );
  const trackEpochRef = useRef<number | undefined>(undefined);
  const currentFrameIndexRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const renderingRef = useRef(false);
  const pendingRenderRef = useRef<FramePropagationResult | null>(null);

  useEffect(() => {
    selectedObjectIdRef.current = selectedObjectId;
  }, [selectedObjectId]);
  useEffect(() => {
    currentFrameIndexRef.current = currentFrameIndex;
  }, [currentFrameIndex]);

  // Dispose worker-side resources when the tab unmounts.
  useEffect(
    () => () => {
      abortControllerRef.current?.abort();
      sessionRef.current?.dispose();
      sessionRef.current = null;
      void segmenterRef.current?.dispose();
      segmenterRef.current = null;
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  /** Repaint the display canvas: current video frame, mask overlays, point markers. */
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoElRef.current;
    if (!canvas || !video || canvas.width === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // MaskCompositor stays NotImplementedError until M4 — composite by hand
    // with a plain 2D canvas, one object at a time via globalCompositeOperation.
    ctx.globalCompositeOperation = 'source-over';
    for (const [objectId, mask] of masksAtFrameRef.current) {
      if (mask.width !== canvas.width || mask.height !== canvas.height) continue;
      const overlay = overlayCanvasRef.current ?? document.createElement('canvas');
      overlayCanvasRef.current = overlay;
      overlay.width = mask.width;
      overlay.height = mask.height;
      const octx = overlay.getContext('2d');
      if (!octx) continue;
      const [r, g, b] = hexToRgb(colorForObject(objectId));
      const binary = mask.toBinary();
      const imageData = octx.createImageData(mask.width, mask.height);
      for (let i = 0; i < binary.length; i++) {
        if (binary[i] === 1) {
          const j = i * 4;
          imageData.data[j] = r;
          imageData.data[j + 1] = g;
          imageData.data[j + 2] = b;
          imageData.data[j + 3] = 128;
        }
      }
      octx.putImageData(imageData, 0, 0);
      ctx.drawImage(overlay, 0, 0);
    }

    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? canvas.width / rect.width : 1;
    const radius = 5 * scale;
    for (const obj of objectsRef.current.values()) {
      if (obj.promptFrame !== currentFrameIndexRef.current) continue;
      for (const point of obj.points) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = point.label === 1 ? obj.color : '#c92a2a';
        ctx.fill();
        ctx.lineWidth = Math.max(1.5, radius / 3);
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
    }
  }, []);

  /** Coalesce live-tracking frame updates: seek + repaint, dropping frames if we fall behind. */
  const drainRender = useCallback(async () => {
    if (renderingRef.current) return;
    renderingRef.current = true;
    try {
      const video = videoElRef.current;
      const init = timelineInitRef.current;
      while (pendingRenderRef.current) {
        const frame = pendingRenderRef.current;
        pendingRenderRef.current = null;
        currentFrameIndexRef.current = frame.frameIndex;
        setCurrentFrameIndex(frame.frameIndex);
        for (const mask of frame.masks) masksAtFrameRef.current.set(mask.objectId, mask);
        if (video && init) await seekVideoTo(video, frame.frameIndex / init.fps);
        redraw();
      }
    } finally {
      renderingRef.current = false;
    }
  }, [redraw]);

  const scheduleRenderFrame = useCallback(
    (frame: FramePropagationResult) => {
      pendingRenderRef.current = frame;
      void drainRender();
    },
    [drainRender],
  );

  const loadModel = useCallback(async () => {
    if (segmenterRef.current || modelState.status === 'loading') return;
    setNotice(null);
    setModelState({ status: 'loading', phases: [] });
    try {
      const segmenter = await createSegmenter({
        model: MODEL_ID,
        modelBaseUrl: MODEL_BASE_URL,
        workerUrl: segmenterWorkerUrl,
        wasmPaths: ORT_WASM_PATHS,
        onProgress: (event) => {
          setModelState((prev) =>
            prev.status === 'loading'
              ? { status: 'loading', event, phases: appendPhase(prev.phases, event) }
              : prev,
          );
        },
      });
      segmenterRef.current = segmenter;
      setModelState({
        status: 'ready',
        device: segmenter.device,
        quant: segmenter.model.quant,
        totalBytes: segmenter.model.totalBytes,
      });
      if (pendingFileRef.current) {
        const file = pendingFileRef.current;
        pendingFileRef.current = null;
        void attachFile(file);
      }
    } catch (err) {
      setModelState({ status: 'error', error: describeError(err) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelState.status]);

  const resetInteractionState = useCallback(() => {
    objectsRef.current = new Map();
    setObjects([]);
    setSelectedObjectId(null);
    masksAtFrameRef.current = new Map();
    timelineRef.current = null;
    trackEpochRef.current = undefined;
    setCurrentFrameIndex(0);
    currentFrameIndexRef.current = 0;
    setTrackState({ status: 'idle' });
    setExportState({ status: 'idle' });
  }, []);

  const attachFile = useCallback(
    async (file: File) => {
      const segmenter = segmenterRef.current;
      const video = videoElRef.current;
      if (!segmenter || !video) return;
      setNotice(null);
      resetInteractionState();
      setSourceState({ status: 'attaching', fileName: file.name });

      const url = URL.createObjectURL(file);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = url;

      try {
        await new Promise<void>((resolve, reject) => {
          const onLoaded = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            video.removeEventListener('error', onError);
            resolve();
          };
          const onError = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            video.removeEventListener('error', onError);
            reject(new Error('the browser could not decode this file for preview'));
          };
          video.addEventListener('loadedmetadata', onLoaded);
          video.addEventListener('error', onError);
          video.src = url;
          video.load();
        });

        sessionRef.current?.dispose();
        const session = await segmenter.createVideoSession();
        sessionRef.current = session;
        const info = await session.attachSource(file);

        // `attachSource`'s public contract types `frameCount` as optional
        // (future HTMLVideoElement sources may not know it up front); at M2
        // Blob attach always counts mp4 samples exactly, but MaskTimeline
        // requires a definite frameCount, so fall back to duration*fps.
        const frameCountGuessed = info.frameCount === undefined;
        const frameCount =
          info.frameCount ?? Math.max(1, Math.round((video.duration || 0) * info.fps));

        timelineInitRef.current = {
          frameCount,
          fps: info.fps,
          width: info.width,
          height: info.height,
        };

        await seekVideoTo(video, 0);
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = info.width;
          canvas.height = info.height;
        }
        setSourceState({
          status: 'ready',
          fileName: file.name,
          fps: info.fps,
          width: info.width,
          height: info.height,
          frameCount,
          frameCountGuessed,
        });
        redraw();
      } catch (err) {
        setSourceState({ status: 'error', error: describeError(err) });
      }
    },
    [redraw, resetInteractionState],
  );

  const handleFile = useCallback(
    (file: File) => {
      const looksLikeMp4 = file.type === 'video/mp4' || /\.mp4$/i.test(file.name);
      if (!looksLikeMp4) {
        setNotice({
          title: 'Not an mp4',
          detail: `'${file.name}' is ${file.type || 'unknown'}. This demo attaches mp4 Blobs only.`,
        });
        return;
      }
      if (!segmenterRef.current) {
        pendingFileRef.current = file;
        setSourceState({ status: 'attaching', fileName: file.name });
        setNotice({
          title: 'Loading model first',
          detail: `'${file.name}' will attach automatically once EdgeTAM finishes loading.`,
        });
        return;
      }
      void attachFile(file);
    },
    [attachFile],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const session = sessionRef.current;
      if (!session || sourceState.status !== 'ready') return;
      if (trackState.status === 'running') {
        setNotice({
          title: 'Pause tracking first',
          detail: 'Click Cancel to stop the propagation loop before refining a mask.',
        });
        return;
      }
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const x = Math.min(
        canvas.width - 1,
        Math.max(0, Math.round((event.clientX - rect.left) * (canvas.width / rect.width))),
      );
      const y = Math.min(
        canvas.height - 1,
        Math.max(0, Math.round((event.clientY - rect.top) * (canvas.height / rect.height))),
      );
      const label: 0 | 1 = event.shiftKey ? 0 : 1;
      const point: ClickPoint = { x, y, label };
      const frameIndex = currentFrameIndexRef.current;
      const selectedId = selectedObjectIdRef.current;
      const existing = selectedId !== null ? objectsRef.current.get(selectedId) : undefined;

      void (async () => {
        try {
          if (!existing) {
            const points = [point];
            const { objectId, mask } = await session.addObject({
              frameIndex,
              prompts: toPrompts(points),
            });
            objectsRef.current.set(objectId, {
              id: objectId,
              color: colorForObject(objectId),
              promptFrame: frameIndex,
              points,
            });
            masksAtFrameRef.current.set(objectId, mask);
            setSelectedObjectId(objectId);
            setObjects([...objectsRef.current.values()]);
          } else {
            const points = existing.promptFrame === frameIndex ? [...existing.points, point] : [point];
            const mask = await session.refineObject(existing.id, frameIndex, toPrompts(points));
            objectsRef.current.set(existing.id, { ...existing, promptFrame: frameIndex, points });
            masksAtFrameRef.current.set(existing.id, mask);
            setObjects([...objectsRef.current.values()]);

            // Downstream propagated masks for this object are now stale —
            // drop them and remember the new epoch for the next Track resume.
            if (timelineRef.current) {
              trackEpochRef.current = timelineRef.current.invalidateAfter(String(existing.id), frameIndex);
            }
          }
          redraw();
        } catch (err) {
          setNotice(describeError(err));
        }
      })();
    },
    [sourceState.status, trackState.status, redraw],
  );

  const startTracking = useCallback(async () => {
    const session = sessionRef.current;
    const init = timelineInitRef.current;
    if (!session || !init || objectsRef.current.size === 0) return;
    setNotice(null);
    if (!timelineRef.current) timelineRef.current = new MaskTimeline(init);
    const timeline = timelineRef.current;
    const startFrame = currentFrameIndexRef.current;

    const controller = new AbortController();
    abortControllerRef.current = controller;
    setTrackState({ status: 'running', frameIndex: startFrame, frameCount: init.frameCount });

    try {
      const iterator = session.propagate({ startFrame, signal: controller.signal });
      await drainInto(iterator, timeline, trackEpochRef.current, (frame) => {
        setTrackState({ status: 'running', frameIndex: frame.frameIndex, frameCount: init.frameCount });
        scheduleRenderFrame(frame);
      });
      setTrackState(controller.signal.aborted ? { status: 'idle' } : { status: 'done' });
    } catch (err) {
      const isAbort = err instanceof DOMException && err.name === 'AbortError';
      const isEpochInvalidated =
        err instanceof Error &&
        (err.name === 'EpochInvalidatedError' ||
          (typeof err === 'object' && err !== null && 'code' in err && (err as { code: unknown }).code === 'EPOCH_INVALIDATED'));
      if (isAbort) {
        setTrackState({ status: 'idle' });
      } else if (isEpochInvalidated) {
        setNotice(describeError(err));
        setTrackState({ status: 'idle' });
      } else {
        const info = describeError(err);
        setNotice(info);
        setTrackState({ status: 'error', error: info });
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [scheduleRenderFrame]);

  const cancelTracking = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const startNewObject = useCallback(() => {
    setSelectedObjectId(null);
  }, []);

  const handleExport = useCallback(async () => {
    const timeline = timelineRef.current;
    if (!timeline) return;
    setNotice(null);
    setExportState({ status: 'running', framesDone: 0, frameCount: timeline.frameCount });
    try {
      const exporter = new AlphaMatteExporter(timeline);
      const result = await exporter.export({
        mode: 'matte',
        onProgress: (framesDone, frameCount) => {
          setExportState({ status: 'running', framesDone, frameCount });
        },
      });
      const url = URL.createObjectURL(result.blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = result.suggestedFileName;
      a.click();
      URL.revokeObjectURL(url);
      setExportState({
        status: 'done',
        framesExported: result.framesExported,
        fileName: result.suggestedFileName,
      });
    } catch (err) {
      const info = describeError(err);
      setNotice(info);
      setExportState({ status: 'error', error: info });
    }
  }, []);

  const modelDot =
    modelState.status === 'ready' ? 'done' : modelState.status === 'error' ? 'failed' : 'probing';

  const canPrompt = sourceState.status === 'ready' && trackState.status !== 'running';
  const canTrack = canPrompt && objects.length > 0;
  const canExport = timelineRef.current !== null && trackState.status !== 'running';

  const trackPercent = useMemo(() => {
    if (trackState.status !== 'running' || trackState.frameCount === 0) return 0;
    return Math.min(100, (trackState.frameIndex / trackState.frameCount) * 100);
  }, [trackState]);

  const statusHint = (() => {
    if (modelState.status !== 'ready') return 'load EdgeTAM to start tracking';
    if (sourceState.status !== 'ready') return 'drop an mp4 to begin';
    if (trackState.status === 'running') return 'tracking… click Cancel to pause and refine';
    if (objects.length === 0) return 'click = positive point · shift-click = negative point';
    return 'click adds points to the selected object · Add object starts a new one';
  })();

  return (
    <>
      <section className="card">
        <header className="card-header">
          <h2>model — {MODEL_ID}</h2>
          {modelState.status !== 'idle' && (
            <span className={`dot dot-${modelDot}`} aria-hidden="true" />
          )}
          <span className="muted">
            {modelState.status === 'idle' && 'not loaded'}
            {modelState.status === 'loading' && 'loading…'}
            {modelState.status === 'ready' && 'ready'}
            {modelState.status === 'error' && 'failed'}
          </span>
        </header>

        {modelState.status === 'idle' && (
          <>
            <p className="notice notice-warn">
              First load downloads {APPROX_DOWNLOAD} of EdgeTAM weights and caches them locally
              (OPFS) for later visits. EdgeTAM ships Apache-2.0 — no license acceptance needed.
            </p>
            <button type="button" className="btn btn-primary" onClick={() => void loadModel()}>
              Download &amp; load model ({APPROX_DOWNLOAD})
            </button>
          </>
        )}

        {modelState.status === 'loading' && (
          <LoadProgress event={modelState.event} phases={modelState.phases} />
        )}

        {modelState.status === 'ready' && (
          <dl className="kv">
            <div className="kv-row">
              <dt>device</dt>
              <dd>{modelState.device}</dd>
            </div>
            <div className="kv-row">
              <dt>quant</dt>
              <dd>{modelState.quant}</dd>
            </div>
            <div className="kv-row">
              <dt>weights</dt>
              <dd>{formatBytes(modelState.totalBytes)}</dd>
            </div>
          </dl>
        )}

        {modelState.status === 'error' && (
          <>
            <div className="notice notice-error">
              <strong>{modelState.error.title}.</strong> {modelState.error.detail}
            </div>
            <button type="button" className="btn" onClick={() => setModelState({ status: 'idle' })}>
              Back
            </button>
          </>
        )}
      </section>

      <section className="card">
        <header className="card-header">
          <h2>video</h2>
          <span className="muted">
            {sourceState.status === 'ready'
              ? `${sourceState.width}×${sourceState.height} · ${sourceState.frameCount} frames${sourceState.frameCountGuessed ? ' (estimated)' : ''} · ${sourceState.fps.toFixed(1)} fps`
              : ''}
          </span>
          <button type="button" className="btn btn-small" onClick={() => fileInputRef.current?.click()}>
            {sourceState.status === 'ready' || sourceState.status === 'attaching' ? 'Replace video' : 'Pick video'}
          </button>
        </header>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4"
          className="visually-hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) handleFile(file);
            event.target.value = '';
          }}
        />

        {/* Display-only decode: the worker owns the real frame source. */}
        <video ref={videoElRef} className="hidden-video" muted playsInline preload="auto" />

        <div
          className={`canvas-zone${dragging ? ' canvas-zone-drag' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {sourceState.status === 'ready' || sourceState.status === 'attaching' ? (
            <canvas
              ref={canvasRef}
              className={`image-canvas${canPrompt ? ' image-canvas-ready' : ''}`}
              onClick={handleCanvasClick}
            />
          ) : (
            <button type="button" className="dropzone" onClick={() => fileInputRef.current?.click()}>
              drop an mp4 here, or click to pick one
            </button>
          )}
        </div>

        {sourceState.status === 'attaching' && (
          <p className="muted hint-line">attaching '{sourceState.fileName}'…</p>
        )}
        {sourceState.status === 'error' && (
          <div className="notice notice-error">
            <strong>{sourceState.error.title}.</strong> {sourceState.error.detail}
          </div>
        )}

        <p className="muted hint-line">{statusHint}</p>

        {objects.length > 0 && (
          <div className="object-chips">
            {objects.map((obj) => (
              <button
                key={obj.id}
                type="button"
                className={`object-chip${selectedObjectId === obj.id ? ' object-chip-active' : ''}`}
                style={{ borderColor: obj.color }}
                onClick={() => setSelectedObjectId(obj.id)}
                disabled={trackState.status === 'running'}
              >
                <span className="object-chip-dot" style={{ background: obj.color }} />
                object {obj.id}
              </button>
            ))}
            <button
              type="button"
              className="btn btn-small"
              onClick={startNewObject}
              disabled={trackState.status === 'running'}
            >
              + Add object
            </button>
          </div>
        )}

        <p className="stats-line">
          <span>frame {currentFrameIndex}</span>
          {timelineRef.current && <span>{timelineRef.current.frameCount} frames in timeline</span>}
        </p>

        <div className="track-controls">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void startTracking()}
            disabled={!canTrack}
          >
            Track
          </button>
          <button
            type="button"
            className="btn"
            onClick={cancelTracking}
            disabled={trackState.status !== 'running'}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => void handleExport()}
            disabled={!canExport || exportState.status === 'running'}
          >
            Export matte.zip
          </button>
        </div>

        {trackState.status === 'running' && (
          <div className="progress-block">
            <p className="progress-line">
              <span>
                tracking frame {trackState.frameIndex} / {trackState.frameCount}
              </span>
              <span className="muted mono-inline">{trackPercent.toFixed(0)}%</span>
            </p>
            <div
              className="progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(trackPercent)}
            >
              <div className="progress-fill" style={{ width: `${trackPercent}%` }} />
            </div>
          </div>
        )}

        {exportState.status === 'running' && (
          <div className="progress-block">
            <p className="progress-line">
              <span>
                exporting frame {exportState.framesDone} / {exportState.frameCount}
              </span>
            </p>
            <div
              className="progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round((exportState.framesDone / Math.max(1, exportState.frameCount)) * 100)}
            >
              <div
                className="progress-fill"
                style={{ width: `${(exportState.framesDone / Math.max(1, exportState.frameCount)) * 100}%` }}
              />
            </div>
          </div>
        )}

        {exportState.status === 'done' && (
          <p className="muted hint-line">
            Downloaded {exportState.fileName} — {exportState.framesExported} PNG frame(s) exported.
          </p>
        )}

        {trackState.status === 'error' && (
          <div className="notice notice-error">
            <strong>{trackState.error.title}.</strong> {trackState.error.detail}
          </div>
        )}

        {notice !== null && (
          <div className="notice notice-error">
            <strong>{notice.title}.</strong> {notice.detail}
          </div>
        )}

        <p className="muted hint-line">
          Out of scope at M2: timeline scrubbing, backward tracking, cutout preview, VP9 export,
          HTMLVideoElement sources.
        </p>
      </section>
    </>
  );
}
