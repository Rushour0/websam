import { createSegmenter } from '@websam3/core';
import type { LoadProgressEvent, MaskResult, Prompt, Segmenter, ImageSession } from '@websam3/core';
// Bundler escape hatch (SegmenterConfig.workerUrl): Vite bundles the core
// inference worker as a dedicated chunk and hands us its URL.
import segmenterWorkerUrl from '@websam3/core/worker?worker&url';

/**
 * Where model manifests + weights are served from. See the README for how
 * to populate this locally — production weight hosting is not live yet.
 */
const MODEL_BASE_URL = (import.meta.env.VITE_WEBSAM_MODELS as string | undefined) ?? '/models/';
const MODEL_ID = 'edgetam'; // Apache-2.0, no license gate, smallest download.

const fileInput = document.getElementById('file-input') as HTMLInputElement;
const statusEl = document.getElementById('status') as HTMLSpanElement;
const progressEl = document.getElementById('progress') as HTMLDivElement;
const progressBar = document.getElementById('progress-bar') as HTMLDivElement;
const progressLabel = document.getElementById('progress-label') as HTMLSpanElement;
const noticeEl = document.getElementById('notice') as HTMLDivElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

let segmenter: Segmenter | null = null;
let session: ImageSession | null = null;
let sourceBitmap: ImageBitmap | null = null;
let mask: MaskResult | null = null;
let loadingModel: Promise<Segmenter> | null = null;

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function showNotice(title: string, detail: string): void {
  noticeEl.hidden = false;
  noticeEl.innerHTML = `<strong>${title}.</strong> ${detail}`;
}

function clearNotice(): void {
  noticeEl.hidden = true;
  noticeEl.textContent = '';
}

function showProgress(event: LoadProgressEvent): void {
  progressEl.hidden = false;
  const determinate = event.phase === 'download' && event.loaded !== undefined && event.total;
  progressBar.style.width = determinate ? `${Math.min(100, (event.loaded! / event.total!) * 100)}%` : '30%';
  progressLabel.textContent = event.file ? `${event.phase} · ${event.file}` : event.phase;
}

function hideProgress(): void {
  progressEl.hidden = true;
}

/** Friendly rendering of the websam error taxonomy (see @websam3/core/errors). */
function describeError(err: unknown): { title: string; detail: string } {
  const message = err instanceof Error ? err.message : String(err);
  const code =
    typeof err === 'object' && err !== null && 'code' in err ? String((err as { code: unknown }).code) : '';

  if (code === 'WEIGHT_VERIFY_FAILED') {
    return {
      title: 'Model weights unavailable',
      detail: `Could not fetch/verify weights from '${MODEL_BASE_URL}'. See the README's "Providing weights" section — hosted weights are pending, so you likely need to populate a local models directory. ${message}`,
    };
  }
  if (code === 'CROSS_ORIGIN_ISOLATION_REQUIRED') {
    return {
      title: 'Cross-origin isolation required',
      detail: `This page needs crossOriginIsolated === true (COOP/COEP headers — already set by vite.config.ts in dev). ${message}`,
    };
  }
  if (code === 'UNSUPPORTED_DEVICE') {
    return { title: 'No supported compute device', detail: message };
  }
  return { title: 'Something went wrong', detail: message };
}

/** Create the segmenter lazily, on first use, not at module load. */
async function ensureSegmenter(): Promise<Segmenter> {
  if (segmenter) return segmenter;
  if (loadingModel) return loadingModel;
  loadingModel = createSegmenter({
    model: MODEL_ID,
    modelBaseUrl: MODEL_BASE_URL,
    workerUrl: segmenterWorkerUrl,
    onProgress: showProgress,
  });
  try {
    segmenter = await loadingModel;
    hideProgress();
    return segmenter;
  } finally {
    loadingModel = null;
  }
}

function redraw(): void {
  if (!sourceBitmap) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sourceBitmap, 0, 0);
  if (!mask || mask.width !== canvas.width || mask.height !== canvas.height) return;

  const binary = mask.toBinary();
  const overlay = ctx.createImageData(mask.width, mask.height);
  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === 1) {
      const j = i * 4;
      overlay.data[j] = 61;
      overlay.data[j + 1] = 133;
      overlay.data[j + 2] = 255;
      overlay.data[j + 3] = 140;
    }
  }
  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = mask.width;
  overlayCanvas.height = mask.height;
  overlayCanvas.getContext('2d')!.putImageData(overlay, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
}

async function handleFile(file: File): Promise<void> {
  clearNotice();
  setStatus('loading model…');
  mask = null;

  try {
    const bitmap = await createImageBitmap(file);
    sourceBitmap = bitmap;
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    redraw();

    const seg = await ensureSegmenter();
    session = await seg.createImageSession();

    setStatus('encoding…');
    await session.encode(bitmap);
    setStatus(`ready (${seg.device}) — click the image`);
  } catch (err) {
    hideProgress();
    const { title, detail } = describeError(err);
    setStatus('error');
    showNotice(title, detail);
  }
}

async function handleClick(event: MouseEvent): Promise<void> {
  if (!session || !session.isEncoded) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round(((event.clientX - rect.left) * canvas.width) / rect.width);
  const y = Math.round(((event.clientY - rect.top) * canvas.height) / rect.height);
  const prompt: Prompt = { type: 'point', x, y, label: event.shiftKey ? 0 : 1 };

  setStatus('decoding…');
  try {
    const masks = await session.decode([prompt]);
    mask = masks[0] ?? null;
    redraw();
    setStatus(mask ? `score ${mask.score.toFixed(3)}` : 'no mask returned');
  } catch (err) {
    const { title, detail } = describeError(err);
    showNotice(title, detail);
    setStatus('error');
  }
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void handleFile(file);
});

canvas.addEventListener('click', (event) => void handleClick(event));

setStatus('idle — pick an image to start');
