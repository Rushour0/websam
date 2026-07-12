/**
 * `PropertiesPanel.tsx` — studio-contracts.md §3.
 *
 * No props; reads/writes only through `useStudioStore`. Shows properties of
 * the current `selection`: for a selected timeline clip, trim in/out
 * steppers plus delete/duplicate/split-at-playhead actions
 * (`store.removeTimelineClip` / `duplicateTimelineClip` /
 * `splitTimelineClip`; split is disabled unless the store playhead sits
 * strictly inside the clip's placed span, passed as a raw project frame);
 * the tracked-object list for the active clip (color swatch, label,
 * select, remove via `store.removeObject`); a mask-opacity slider
 * (`store.maskOpacity`/`store.setMaskOpacity`, consumed by `PreviewCanvas`'s
 * overlay rendering); a export-settings sub-panel that mirrors the Toolbar's
 * export actions (`exportMatte` / `exportMp4Cutout`, same store calls, no
 * duplicated logic); and a load-model panel (`store.loadModel`) with
 * progress, the ~120MB/device note, and cached/ready state. Read-only-
 * friendly when nothing is selected — sections simply don't render instead
 * of showing empty/broken controls.
 *
 * CONTRACT NOTE: §3 also asks for an export "range", but
 * `exportMatte`/`exportMp4Cutout` take only `clipId` (no range) — that seam
 * isn't exposed by `export.ts` (always exports the full mask timeline), so
 * no range control is rendered here. Follow-up.
 */
import { Copy, Loader2, Scissors, Trash2 } from 'lucide-react';

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
  const playhead = useStudioStore((s) => s.playhead);
  const removeTimelineClip = useStudioStore((s) => s.removeTimelineClip);
  const duplicateTimelineClip = useStudioStore((s) => s.duplicateTimelineClip);
  const splitTimelineClip = useStudioStore((s) => s.splitTimelineClip);

  if (!timelineClip || !clip) return null;

  const maxFrame = Math.max(0, clip.frameCount - 1);

  const clampIn = (value: number) => Math.min(Math.max(0, value), timelineClip.outFrame);
  const clampOut = (value: number) => Math.min(Math.max(timelineClip.inFrame, value), maxFrame);

  const clipSpan = timelineClip.outFrame - timelineClip.inFrame + 1;
  const canSplit = playhead > timelineClip.startFrame && playhead < timelineClip.startFrame + clipSpan;

  return (
    <section className="space-y-2 border-b border-border p-3">
      <div className="space-y-1">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Clip</h3>
        <p className="truncate text-sm font-medium" title={clip.fileName}>
          {clip.fileName}
        </p>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>In frame</span>
          <span>Out frame</span>
        </div>
        <div className="flex items-center overflow-hidden rounded-md border border-input">
          <input
            type="number"
            aria-label="In frame"
            min={0}
            max={timelineClip.outFrame}
            value={timelineClip.inFrame}
            onChange={(e) =>
              trimTimelineClip(timelineClip.id, clampIn(Number(e.target.value)), timelineClip.outFrame)
            }
            className="h-8 min-w-0 flex-1 bg-background px-2 text-sm tabular-nums text-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />
          <span aria-hidden className="h-4 w-px shrink-0 bg-border" />
          <input
            type="number"
            aria-label="Out frame"
            min={timelineClip.inFrame}
            max={maxFrame}
            value={timelineClip.outFrame}
            onChange={(e) =>
              trimTimelineClip(timelineClip.id, timelineClip.inFrame, clampOut(Number(e.target.value)))
            }
            className="h-8 min-w-0 flex-1 bg-background px-2 text-sm tabular-nums text-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />
        </div>
        <p className="text-[11px] tabular-nums text-muted-foreground">{clipSpan} frames</p>
      </div>
      <div className="flex items-center gap-1 pt-1">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          aria-label="Duplicate clip"
          onClick={() => duplicateTimelineClip(timelineClip.id)}
        >
          <Copy className="h-3.5 w-3.5" />
        </Button>
        <span title={canSplit ? undefined : 'Move the playhead inside this clip to split'}>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            aria-label="Split clip at playhead"
            title={canSplit ? 'Split clip at playhead' : undefined}
            disabled={!canSplit}
            onClick={() => splitTimelineClip(timelineClip.id, playhead)}
          >
            <Scissors className="h-3.5 w-3.5" />
          </Button>
        </span>
        <span aria-hidden className="mx-0.5 h-4 w-px bg-border" />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0 hover:text-destructive"
          aria-label="Delete clip"
          onClick={() => removeTimelineClip(timelineClip.id)}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
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
        selected ? 'bg-accent font-medium text-accent-foreground' : 'hover:bg-accent/50',
      )}
    >
      <button
        type="button"
        onClick={() => selectObject(selected ? null : object.objectId)}
        className="flex flex-1 items-center gap-2 rounded-sm text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
        className="h-7 w-7 shrink-0 hover:text-destructive"
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
  const maskOpacity = useStudioStore((s) => s.maskOpacity);
  const setMaskOpacity = useStudioStore((s) => s.setMaskOpacity);

  if (!activeClipId) return null;

  const clipObjects = objects.filter((o) => o.clipId === activeClipId);
  const selectedObject = clipObjects.find((o) => o.objectId === selectedObjectId) ?? null;

  return (
    <section className="space-y-2 border-b border-border p-3">
      <div className="flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Objects</h3>
        {clipObjects.length > 0 ? (
          <span className="text-[11px] tabular-nums text-muted-foreground">{clipObjects.length}</span>
        ) : null}
      </div>
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
        <div className="space-y-1.5 pt-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Mask opacity</span>
            <span>{Math.round(maskOpacity * 100)}%</span>
          </div>
          <Slider
            aria-label="Mask opacity"
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
          {running && exportState.kind === 'matte' ? <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none" /> : null}
          Matte (PNG sequence)
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={running}
          onClick={() => void exportMp4Cutout(activeClipId)}
        >
          {running && exportState.kind === 'mp4' ? <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none" /> : null}
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
          Done: <span className="text-foreground">{exportState.fileName}</span> ({exportState.framesExported} frames)
        </p>
      ) : null}
      {exportState.phase === 'error' ? (
        <p className="text-xs text-destructive">{exportState.message}</p>
      ) : null}
    </section>
  );
}

/** Load-model panel: idle → CTA with size note; loading → progress; ready → device/quant/cached badge. */
function ModelPanel({ className }: { className?: string }): React.ReactElement {
  const modelStatus = useStudioStore((s) => s.modelStatus);
  const loadModel = useStudioStore((s) => s.loadModel);

  return (
    <section className={cn('space-y-2 p-3', className)}>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model</h3>

      {modelStatus.phase === 'idle' ? (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">{MODEL_SIZE_NOTE}</p>
          <Button type="button" size="sm" className="w-full" onClick={() => void loadModel()}>
            Load model
          </Button>
        </div>
      ) : null}

      {modelStatus.phase === 'loading' ? (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none" />
            <span>{modelStatus.progress?.phase ?? 'Loading…'}</span>
          </div>
          {modelStatus.progress?.file ? (
            <p className="truncate text-xs text-muted-foreground">{modelStatus.progress.file}</p>
          ) : null}
          {modelStatus.progress?.total ? (
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full bg-primary transition-[width] duration-200 motion-reduce:transition-none"
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
  const modelReady = useStudioStore((s) => s.modelStatus.phase === 'ready');

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card text-card-foreground">
      {!modelReady ? <ModelPanel className="border-b border-border" /> : null}
      {!hasSelection ? (
        <div className="space-y-2 border-b border-border p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Properties</h3>
          <p className="text-sm text-muted-foreground">Select a clip or object to see its properties.</p>
        </div>
      ) : (
        <>
          <ClipProperties />
          <ObjectsProperties />
          <ExportProperties />
        </>
      )}
      {modelReady ? (
        <div className="mt-auto border-t border-border">
          <ModelPanel />
        </div>
      ) : null}
    </div>
  );
}
