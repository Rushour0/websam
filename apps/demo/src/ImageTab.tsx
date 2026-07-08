import { useCallback, useEffect, useRef, useState } from 'react';
import { createSegmenter } from '@websam/core';
import type {
  ImageSession,
  LoadProgressEvent,
  MaskResult,
  Prompt,
  Segmenter,
} from '@websam/core';
// Vite bundles the core worker entry as a dedicated worker chunk and hands us
// its URL — the documented `SegmenterConfig.workerUrl` escape hatch for
// bundlers that break `new URL('./worker.js', import.meta.url)` resolution.
import segmenterWorkerUrl from '@websam/core/worker?worker&url';

/**
 * Where model manifests + weights are served from. Local dev default is
 * `/models/` (populate `apps/demo/public/models/` via
 * `tools/goldens/fetch-models.mjs`); deployments override with
 * `VITE_WEBSAM_MODELS`.
 */
const MODEL_BASE_URL = (import.meta.env.VITE_WEBSAM_MODELS as string | undefined) ?? '/models/';

/** Optional override for onnxruntime-web's .wasm/.mjs asset base inside the worker. */
const ORT_WASM_PATHS = import.meta.env.VITE_ORT_WASM_PATHS as string | undefined;

/** The tier this tab drives; ~300 MB of q4f16 weights on first download. */
const MODEL_ID = 'sam3-tracker';
const APPROX_DOWNLOAD = '~300 MB';

/** One canvas click, in source-image pixel coordinates. 1 = positive, 0 = negative. */
interface ClickPoint {
  x: number;
  y: number;
  label: 0 | 1;
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

type CacheState =
  | { status: 'checking' }
  | { status: 'ready'; files: number; bytes: number }
  | { status: 'unavailable' };

type EncodeState =
  | { status: 'none' }
  | { status: 'running' }
  | { status: 'done'; encodeMs: number }
  | { status: 'error' };

/** Map a thrown value onto a friendly notice, keyed on the websam error taxonomy. */
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
        'This feature needs crossOriginIsolated === true. Serve the page with ' +
        '`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` ' +
        '(the vite dev server and public/_headers already do). ' +
        message,
    };
  }
  if (code === 'UNSUPPORTED_DEVICE' || name === 'UnsupportedDeviceError') {
    return {
      title: 'No supported compute device',
      detail:
        `The '${MODEL_ID}' tier needs WebGPU. Use a recent Chromium-based browser with WebGPU ` +
        'enabled — the capabilities tab shows what this browser reports. ' +
        message,
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
      detail: `The device ran out of memory loading or running the model. ${message}`,
    };
  }
  if (code === 'NOT_IMPLEMENTED' || name === 'NotImplementedError') {
    return { title: 'Not implemented yet', detail: message };
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

/**
 * Read-only probe of the core weight store's persistent backends, so the UI
 * can say "cached — no download" before the user commits to a 300 MB fetch.
 * Mirrors (but never writes) the store layout: OPFS dir `websam-weights` with
 * `<sha256>.bin` + zero-byte `<sha256>.ok` commit markers, and Cache API
 * bucket `websam-weights` for small files.
 */
async function probeWeightCache(): Promise<{ files: number; bytes: number } | undefined> {
  const probeOpfs = async (): Promise<{ files: number; bytes: number }> => {
    const root = await navigator.storage.getDirectory();
    let dir: FileSystemDirectoryHandle;
    try {
      dir = await root.getDirectoryHandle('websam-weights');
    } catch {
      return { files: 0, bytes: 0 }; // OPFS works, nothing stored yet
    }
    const entries = (
      dir as unknown as { entries(): AsyncIterableIterator<[string, FileSystemHandle]> }
    ).entries();
    const committed = new Set<string>();
    const bins = new Map<string, FileSystemFileHandle>();
    for await (const [entryName, handle] of entries) {
      if (handle.kind !== 'file') continue;
      if (entryName.endsWith('.ok')) committed.add(entryName.slice(0, -'.ok'.length));
      else if (entryName.endsWith('.bin')) {
        bins.set(entryName.slice(0, -'.bin'.length), handle as FileSystemFileHandle);
      }
    }
    let files = 0;
    let bytes = 0;
    for (const sha of committed) {
      const bin = bins.get(sha);
      if (!bin) continue; // marker without content — never served by the store
      files += 1;
      bytes += (await bin.getFile()).size;
    }
    return { files, bytes };
  };

  const probeCacheApi = async (): Promise<{ files: number; bytes: number }> => {
    if (!('caches' in globalThis)) throw new Error('Cache API unavailable');
    const cache = await caches.open('websam-weights');
    const keys = await cache.keys();
    let bytes = 0;
    for (const request of keys) {
      const response = await cache.match(request);
      if (response) bytes += (await response.blob()).size;
    }
    return { files: keys.length, bytes };
  };

  const results = await Promise.allSettled([probeOpfs(), probeCacheApi()]);
  const fulfilled = results.filter(
    (r): r is PromiseFulfilledResult<{ files: number; bytes: number }> => r.status === 'fulfilled',
  );
  if (fulfilled.length === 0) return undefined;
  return fulfilled.reduce(
    (acc, r) => ({ files: acc.files + r.value.files, bytes: acc.bytes + r.value.bytes }),
    { files: 0, bytes: 0 },
  );
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

/** Append the event's label to the phase log unless it repeats the last entry. */
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

/**
 * Interactive image segmentation: drop/pick an image, lazily load the
 * sam3-tracker model (explicit consent before the ~300 MB first download,
 * with per-phase progress including 'compile'), encode once, then decode a
 * mask per click — click = positive point, shift-click = negative — with the
 * mask rendered as a semi-transparent overlay and encode/decode latency shown.
 */
export function ImageTab() {
  const [modelState, setModelState] = useState<ModelState>({ status: 'idle' });
  const [cacheState, setCacheState] = useState<CacheState>({ status: 'checking' });
  const [encodeState, setEncodeState] = useState<EncodeState>({ status: 'none' });
  const [imageInfo, setImageInfo] = useState<{ name: string; width: number; height: number } | null>(
    null,
  );
  const [imageVersion, setImageVersion] = useState(0);
  const [points, setPoints] = useState<ClickPoint[]>([]);
  const [decodeInfo, setDecodeInfo] = useState<{ decodeMs: number; score: number } | null>(null);
  const [notice, setNotice] = useState<ErrorInfo | null>(null);
  const [dragging, setDragging] = useState(false);

  const segmenterRef = useRef<Segmenter | null>(null);
  const sessionRef = useRef<ImageSession | null>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageVersionRef = useRef(0);
  const pointsRef = useRef<ClickPoint[]>([]);
  const maskRef = useRef<MaskResult | null>(null);
  const encodeBusyRef = useRef(false);
  const decodeBusyRef = useRef(false);
  const pendingDecodeRef = useRef<ClickPoint[] | null>(null);

  const refreshCache = useCallback(() => {
    void probeWeightCache().then((report) => {
      setCacheState(report ? { status: 'ready', ...report } : { status: 'unavailable' });
    });
  }, []);

  useEffect(() => {
    refreshCache();
  }, [refreshCache]);

  // Dispose worker-side resources when the tab unmounts.
  useEffect(
    () => () => {
      sessionRef.current?.dispose();
      sessionRef.current = null;
      void segmenterRef.current?.dispose();
      segmenterRef.current = null;
    },
    [],
  );

  /** Repaint the display canvas: image, then mask overlay, then point markers. */
  const redraw = useCallback(() => {
    const canvas = displayCanvasRef.current;
    const source = sourceCanvasRef.current;
    if (!canvas || !source || source.width === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0);

    const mask = maskRef.current;
    if (mask && mask.width === canvas.width && mask.height === canvas.height) {
      const overlay = overlayCanvasRef.current ?? document.createElement('canvas');
      overlayCanvasRef.current = overlay;
      overlay.width = mask.width;
      overlay.height = mask.height;
      const octx = overlay.getContext('2d');
      if (octx) {
        const binary = mask.toBinary();
        const imageData = octx.createImageData(mask.width, mask.height);
        // Semi-transparent blue where mask = 1; fully transparent elsewhere.
        for (let i = 0; i < binary.length; i++) {
          if (binary[i] === 1) {
            const j = i * 4;
            imageData.data[j] = 61;
            imageData.data[j + 1] = 133;
            imageData.data[j + 2] = 255;
            imageData.data[j + 3] = 138;
          }
        }
        octx.putImageData(imageData, 0, 0);
        ctx.drawImage(overlay, 0, 0);
      }
    }

    // Point markers, sized against the CSS-scaled display so they stay visible.
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? canvas.width / rect.width : 1;
    const radius = 5 * scale;
    for (const point of pointsRef.current) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = point.label === 1 ? '#2b8a3e' : '#c92a2a';
      ctx.fill();
      ctx.lineWidth = Math.max(1.5, radius / 3);
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
  }, []);

  /** Encode whatever image is currently loaded; re-runs if the image changed mid-encode. */
  const encodeCurrent = useCallback(async () => {
    if (encodeBusyRef.current) return; // the running loop re-checks the image version
    const session = sessionRef.current;
    const source = sourceCanvasRef.current;
    if (!session || !source || source.width === 0) return;
    encodeBusyRef.current = true;
    try {
      let encodedVersion = -1;
      while (encodedVersion !== imageVersionRef.current) {
        encodedVersion = imageVersionRef.current;
        setEncodeState({ status: 'running' });
        const result = await session.encode(source);
        if (imageVersionRef.current === encodedVersion) {
          setEncodeState({ status: 'done', encodeMs: result.encodeMs });
        }
      }
    } catch (err) {
      setEncodeState({ status: 'error' });
      setNotice(describeError(err));
    } finally {
      encodeBusyRef.current = false;
    }
  }, []);

  /** Serialize decodes; a click during an in-flight decode re-decodes the latest points. */
  const runDecode = useCallback(
    async (clickPoints: ClickPoint[]) => {
      const session = sessionRef.current;
      if (!session || !session.isEncoded) return;
      if (decodeBusyRef.current) {
        pendingDecodeRef.current = clickPoints;
        return;
      }
      decodeBusyRef.current = true;
      const version = imageVersionRef.current;
      try {
        let current: ClickPoint[] | null = clickPoints;
        while (current !== null && imageVersionRef.current === version) {
          pendingDecodeRef.current = null;
          if (current.length === 0) {
            maskRef.current = null;
            setDecodeInfo(null);
            redraw();
          } else {
            const prompts: Prompt[] = current.map((p) => ({
              type: 'point',
              x: p.x,
              y: p.y,
              label: p.label,
            }));
            const started = performance.now();
            const masks = await session.decode(prompts);
            const decodeMs = performance.now() - started;
            const best = masks[0];
            if (best && imageVersionRef.current === version) {
              maskRef.current = best;
              setDecodeInfo({ decodeMs, score: best.score });
              redraw();
            }
          }
          current = pendingDecodeRef.current;
        }
      } catch (err) {
        setNotice(describeError(err));
      } finally {
        decodeBusyRef.current = false;
      }
    },
    [redraw],
  );

  const loadModel = useCallback(async () => {
    if (segmenterRef.current || modelState.status === 'loading') return;
    setNotice(null);
    setModelState({ status: 'loading', phases: [] });
    try {
      const segmenter = await createSegmenter({
        model: MODEL_ID,
        acceptLicense: 'sam',
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
      sessionRef.current = await segmenter.createImageSession();
      setModelState({
        status: 'ready',
        device: segmenter.device,
        quant: segmenter.model.quant,
        totalBytes: segmenter.model.totalBytes,
      });
      refreshCache(); // weights are now in the local store
      void encodeCurrent(); // encode the already-dropped image, if any
    } catch (err) {
      setModelState({ status: 'error', error: describeError(err) });
    }
  }, [modelState.status, refreshCache, encodeCurrent]);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        setNotice({ title: 'Not an image', detail: `'${file.name}' is ${file.type || 'unknown'}.` });
        return;
      }
      let bitmap: ImageBitmap;
      try {
        bitmap = await createImageBitmap(file);
      } catch {
        setNotice({ title: 'Could not decode image', detail: `'${file.name}' failed to decode.` });
        return;
      }
      const source = sourceCanvasRef.current ?? document.createElement('canvas');
      sourceCanvasRef.current = source;
      source.width = bitmap.width;
      source.height = bitmap.height;
      source.getContext('2d')?.drawImage(bitmap, 0, 0);
      bitmap.close();

      pointsRef.current = [];
      maskRef.current = null;
      pendingDecodeRef.current = null;
      imageVersionRef.current += 1;
      setPoints([]);
      setDecodeInfo(null);
      setNotice(null);
      setEncodeState({ status: 'none' });
      setImageInfo({ name: file.name, width: source.width, height: source.height });
      setImageVersion(imageVersionRef.current);
    },
    [],
  );

  // After a new image lands (and the <canvas> exists), paint it and encode.
  useEffect(() => {
    if (imageVersion === 0) return;
    const canvas = displayCanvasRef.current;
    const source = sourceCanvasRef.current;
    if (canvas && source) {
      canvas.width = source.width;
      canvas.height = source.height;
    }
    redraw();
    void encodeCurrent();
  }, [imageVersion, redraw, encodeCurrent]);

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (encodeState.status !== 'done') return;
      const canvas = displayCanvasRef.current;
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
      const next = [...pointsRef.current, { x, y, label }];
      pointsRef.current = next;
      setPoints(next);
      redraw(); // show the point immediately, before the decode lands
      void runDecode(next);
    },
    [encodeState.status, redraw, runDecode],
  );

  const clearPoints = useCallback(() => {
    pointsRef.current = [];
    maskRef.current = null;
    pendingDecodeRef.current = null;
    setPoints([]);
    setDecodeInfo(null);
    setNotice(null);
    redraw();
  }, [redraw]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files[0];
      if (file) void handleFile(file);
    },
    [handleFile],
  );

  const cachedEnough = cacheState.status === 'ready' && cacheState.bytes > 200 * 1024 * 1024;
  const modelDot =
    modelState.status === 'ready' ? 'done' : modelState.status === 'error' ? 'failed' : 'probing';

  const statusHint = (() => {
    if (modelState.status !== 'ready') return 'load the model to start segmenting';
    if (!imageInfo) return 'drop an image to segment';
    if (encodeState.status === 'running') return 'encoding image…';
    if (encodeState.status === 'error') return 'encode failed — try another image';
    return 'click = positive point · shift-click = negative point';
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
              First load downloads {APPROX_DOWNLOAD} of SAM&nbsp;3 Tracker weights and caches them
              locally (OPFS) for later visits. The weights ship under the SAM license — loading
              accepts it.
            </p>
            <p className="muted cache-line">
              {cacheState.status === 'checking' && 'checking local weight cache…'}
              {cacheState.status === 'ready' &&
                (cacheState.files > 0
                  ? `${cacheState.files} verified weight file${cacheState.files === 1 ? '' : 's'} cached locally (${formatBytes(cacheState.bytes)})${cachedEnough ? ' — no download expected' : ''}`
                  : 'no cached weights found — the full download will run')}
              {cacheState.status === 'unavailable' &&
                'local weight cache unavailable in this browser — weights re-download each visit'}
            </p>
            <button type="button" className="btn btn-primary" onClick={() => void loadModel()}>
              {cachedEnough ? 'Load model (weights cached)' : `Download & load model (${APPROX_DOWNLOAD})`}
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
            <button
              type="button"
              className="btn"
              onClick={() => setModelState({ status: 'idle' })}
            >
              Back
            </button>
          </>
        )}
      </section>

      <section className="card">
        <header className="card-header">
          <h2>image</h2>
          <span className="muted">{imageInfo ? `${imageInfo.width}×${imageInfo.height}` : ''}</span>
          <button
            type="button"
            className="btn btn-small"
            onClick={() => fileInputRef.current?.click()}
          >
            {imageInfo ? 'Replace image' : 'Pick image'}
          </button>
          <button
            type="button"
            className="btn btn-small"
            onClick={clearPoints}
            disabled={points.length === 0}
          >
            Clear points
          </button>
        </header>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="visually-hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void handleFile(file);
            event.target.value = '';
          }}
        />

        <div
          className={`canvas-zone${dragging ? ' canvas-zone-drag' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {imageInfo ? (
            <canvas
              ref={displayCanvasRef}
              className={`image-canvas${encodeState.status === 'done' ? ' image-canvas-ready' : ''}`}
              onClick={handleCanvasClick}
            />
          ) : (
            <button
              type="button"
              className="dropzone"
              onClick={() => fileInputRef.current?.click()}
            >
              drop an image here, or click to pick one
            </button>
          )}
        </div>

        <p className="muted hint-line">{statusHint}</p>

        {(encodeState.status === 'done' || decodeInfo !== null || points.length > 0) && (
          <p className="stats-line">
            {encodeState.status === 'done' && <span>encode {encodeState.encodeMs.toFixed(0)} ms</span>}
            {decodeInfo !== null && <span>decode {decodeInfo.decodeMs.toFixed(0)} ms</span>}
            {decodeInfo !== null && <span>score {decodeInfo.score.toFixed(3)}</span>}
            {points.length > 0 && (
              <span>
                {points.length} point{points.length === 1 ? '' : 's'}
              </span>
            )}
          </p>
        )}

        {notice !== null && (
          <div className="notice notice-error">
            <strong>{notice.title}.</strong> {notice.detail}
          </div>
        )}
      </section>
    </>
  );
}
