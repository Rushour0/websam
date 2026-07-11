/**
 * `PropertiesPanel.tsx` — studio-contracts.md §3.
 *
 * No props; reads/writes only through `useStudioStore`. Shows properties of
 * the current `selection`: trim in/out steppers for a selected timeline
 * clip; the tracked-object list for the active clip (color swatch, label,
 * select, remove via `store.removeObject`); a mask-opacity slider for the
 * selected object; a export-settings sub-panel that mirrors the Toolbar's
 * export actions (`exportMatte` / `exportMp4Cutout`, same store calls, no
 * duplicated logic); and a load-model panel (`store.loadModel`) with
 * progress, the ~120MB/device note, and cached/ready state. Read-only-
 * friendly when nothing is selected — sections simply don't render instead
 * of showing empty/broken controls.
 *
 * CONTRACT NOTE (flagged, not fixed here — this file may only touch
 * PropertiesPanel.tsx): §3 asks for a "mask opacity slider (store state)"
 * and export "range", but `StudioState` (studio-store.ts) has no
 * `maskOpacity`/`setMaskOpacity` field and `exportMatte`/`exportMp4Cutout`
 * take only `clipId` (no range). The opacity slider below is therefore
 * local component state only — it does not yet affect `PreviewCanvas`'s
 * overlay rendering, since that would require a store field this file
 * doesn't own. Export range is not exposed by the seam (`export.ts` always
 * exports the full mask timeline), so no range control is rendered. Both
 * should be added to `StudioState`/`export.ts` in a follow-up.
 */
import { useState } from 'react';
import { Loader2, Trash2 } from 'lucide-react';

import { useStudioStore } from '../store/studio-store.js';
import type { TrackedObject } from '../store/studio-store.js';
import { Button } from './ui/button.js';
import { Slider } from './ui/slider.js';
import { cn } from '../lib/utils.js';

const MODEL_SIZE_NOTE = 'EdgeTAM weights are ~120MB and are cached in the browser after the first load; device (WebGPU vs WASM) is auto-detected.';

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 MB';
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

/** Trim-range steppers for the selected timeline clip. */
function ClipProperties(): React.ReactElement | null {
  const timelineClipId = useStudioStore((s) => s.selection.timelineClipId);
  const timelineClip = useStudioStore((s) => (timelineClipId ? s.timelineClips[timelineClipId] : undefined));
  const clip = useStudioStore((s) => (timelineClip ? s.clips[timelineClip.clipId] : undefined));
  const trimTimelineClip = useStudioStore((s) => s.trimTimelineClip);

  if (!timelineClip || !clip) return null;

  const maxFrame = Math.max(0, clip.frameCount - 1);

  const clampIn = (value: number) => Math.min(Math.max(0, value), timelineClip.outFrame);
  const clampOut = (value: number) => Math.min(Math.max(timelineClip.inFrame, value), maxFrame);

  return (
    <section className="space-y-2 border-b border-border p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Clip</h3>
      <p className="truncate text-sm font-medium" title={clip.fileName}>
        {clip.fileName}
      </p>
      <div className="flex items-center gap-2">
        <label className="flex flex-1 flex-col gap-1 text-xs text-muted-foreground">
          In frame
          <input
            type="number"
            min={0}
            max={timelineClip.outFrame}
            value={timelineClip.inFrame}
            onChange={(e) =>
              trimTimelineClip(timelineClip.id, clampIn(Number(e.target.value)), timelineClip.outFrame)
            }
            className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground"
          />
        </label>
        <label className="flex flex-1 flex-col gap-1 text-xs text-muted-foreground">
          Out frame
          <input
            type="number"
            min={timelineClip.inFrame}
            max={maxFrame}
            value={timelineClip.outFrame}
            onChange={(e) =>
              trimTimelineClip(timelineClip.id, timelineClip.inFrame, clampOut(Number(e.target.value)))
            }
            className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground"
          />
        </label>
      </div>
    </section>
  );
}

/** One row in the tracked-object list for the active clip. */
function ObjectRow({ object, selected }: { object: TrackedObject; selected: boolean }): React.ReactElement {
  const selectObject = useStudioStore((s) => s.selectObject);
  const removeObject = useStudioStore((s) => s.removeObject);

  return (
    <li
      className={cn(
        'flex items-center gap-2 rounded-md px-2 py-1.5 text-sm',
        selected ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50',
      )}
    >
      <button
        type="button"
        onClick={() => selectObject(selected ? null : object.objectId)}
        className="flex flex-1 items-center gap-2 text-left"
      >
        <span
          className="h-3 w-3 shrink-0 rounded-full border border-border"
          style={{ backgroundColor: object.color }}
          aria-hidden
        />
        <span className="truncate">{object.label}</span>
      </button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-7 w-7 shrink-0"
        aria-label={`Remove ${object.label}`}
        onClick={() => removeObject(object.clipId, object.objectId)}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </li>
  );
}

/** Tracked-object list for the active clip, plus a mask-opacity slider for the selected object. */
function ObjectsProperties(): React.ReactElement | null {
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const objects = useStudioStore((s) => s.objects);
  const selectedObjectId = useStudioStore((s) => s.selection.objectId);
  // Local-only per §3 mismatch note above — not wired into PreviewCanvas.
  const [maskOpacity, setMaskOpacity] = useState(0.5);

  if (!activeClipId) return null;

  const clipObjects = objects.filter((o) => o.clipId === activeClipId);
  const selectedObject = clipObjects.find((o) => o.objectId === selectedObjectId) ?? null;

  return (
    <section className="space-y-2 border-b border-border p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Objects</h3>
      {clipObjects.length === 0 ? (
        <p className="text-sm text-muted-foreground">No objects yet — prompt one on the canvas.</p>
      ) : (
        <ul className="space-y-0.5">
          {clipObjects.map((object) => (
            <ObjectRow key={object.objectId} object={object} selected={object.objectId === selectedObjectId} />
          ))}
        </ul>
      )}

      {selectedObject ? (
        <div className="space-y-1.5 pt-1">
          <label className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Mask opacity</span>
            <span>{Math.round(maskOpacity * 100)}%</span>
          </label>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={[maskOpacity]}
            onValueChange={([value]) => setMaskOpacity(value ?? maskOpacity)}
          />
        </div>
      ) : null}
    </section>
  );
}

/** Export-settings sub-panel; mirrors the Toolbar's export actions (no duplicated logic). */
function ExportProperties(): React.ReactElement | null {
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const exportState = useStudioStore((s) => s.exportState);
  const exportMatte = useStudioStore((s) => s.exportMatte);
  const exportMp4Cutout = useStudioStore((s) => s.exportMp4Cutout);

  if (!activeClipId) return null;

  const running = exportState.phase === 'running';

  return (
    <section className="space-y-2 border-b border-border p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Export</h3>
      <div className="flex flex-col gap-2">
        <Button
          type="button"
          size="sm"
          variant="secondary"
          disabled={running}
          onClick={() => void exportMatte(activeClipId)}
        >
          {running && exportState.kind === 'matte' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          Matte (PNG sequence)
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={running}
          onClick={() => void exportMp4Cutout(activeClipId)}
        >
          {running && exportState.kind === 'mp4' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          MP4 cutout (stretch)
        </Button>
      </div>
      {exportState.phase === 'running' ? (
        <p className="text-xs text-muted-foreground">
          Exporting {exportState.kind}… {exportState.framesDone}/{exportState.frameCount} frames
        </p>
      ) : null}
      {exportState.phase === 'done' ? (
        <p className="text-xs text-muted-foreground">
          Done: {exportState.fileName} ({exportState.framesExported} frames)
        </p>
      ) : null}
      {exportState.phase === 'error' ? (
        <p className="text-xs text-destructive">{exportState.message}</p>
      ) : null}
    </section>
  );
}

/** Load-model panel: idle → CTA with size note; loading → progress; ready → device/quant/cached badge. */
function ModelPanel(): React.ReactElement {
  const modelStatus = useStudioStore((s) => s.modelStatus);
  const loadModel = useStudioStore((s) => s.loadModel);

  return (
    <section className="space-y-2 p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model</h3>

      {modelStatus.phase === 'idle' ? (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">{MODEL_SIZE_NOTE}</p>
          <Button type="button" size="sm" onClick={() => void loadModel()}>
            Load model
          </Button>
        </div>
      ) : null}

      {modelStatus.phase === 'loading' ? (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{modelStatus.progress?.phase ?? 'Loading…'}</span>
          </div>
          {modelStatus.progress?.file ? (
            <p className="truncate text-xs text-muted-foreground">{modelStatus.progress.file}</p>
          ) : null}
          {modelStatus.progress?.total ? (
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full bg-primary transition-[width]"
                style={{
                  width: `${Math.min(100, ((modelStatus.progress.loaded ?? 0) / modelStatus.progress.total) * 100)}%`,
                }}
              />
            </div>
          ) : null}
        </div>
      ) : null}

      {modelStatus.phase === 'ready' ? (
        <div className="space-y-1 text-xs text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Ready</span> · cached in browser
          </p>
          <p>
            Device: {modelStatus.device} · Quant: {modelStatus.quant} · {formatBytes(modelStatus.totalBytes)}
          </p>
        </div>
      ) : null}

      {modelStatus.phase === 'error' ? <p className="text-xs text-destructive">{modelStatus.message}</p> : null}
    </section>
  );
}

/**
 * Right-hand properties panel: clip trim, object list + opacity, export
 * settings, and the model-load panel. No props — reads/writes the store
 * directly.
 */
export function PropertiesPanel(): React.ReactElement {
  const hasSelection = useStudioStore(
    (s) => s.selection.timelineClipId !== null || s.selection.objectId !== null || s.activeClipId !== null,
  );

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card text-card-foreground">
      {!hasSelection ? (
        <div className="p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Properties</h3>
          <p className="mt-1 text-sm text-muted-foreground">Select a clip or object to see its properties.</p>
        </div>
      ) : (
        <>
          <ClipProperties />
          <ObjectsProperties />
          <ExportProperties />
        </>
      )}
      <ModelPanel />
    </div>
  );
}
