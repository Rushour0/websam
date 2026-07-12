/**
 * Top-level toolbar (`apps/studio/docs/studio-contracts.md` §3): tool
 * radio-group, play/pause, Track/Cancel (mutually exclusive with prompting
 * per the one-active-iterator rule, friction §0.5), Export actions, and a
 * model-status / resolved-device readout. No props — reads/writes the store
 * directly.
 */
import {
  Cpu,
  Download,
  MessageSquare,
  MousePointer2,
  PanelBottom,
  PanelLeft,
  PanelRight,
  Pause,
  Play,
  Plus,
  Hand,
  Square,
  X,
} from 'lucide-react';

import { Button } from './ui/button.js';
import { ToggleGroup, ToggleGroupItem } from './ui/toggle-group.js';
import { cn } from '../lib/utils.js';
import { useStudioStore } from '../store/studio-store.js';
import type { ModelStatus, PanelVisibility, ToolMode } from '../store/studio-store.js';

/** Human-readable label + tone for the model-status pill. */
function statusLabel(status: ModelStatus): string {
  switch (status.phase) {
    case 'idle':
      return 'Model not loaded';
    case 'loading':
      return status.progress?.phase ? `Loading model — ${status.progress.phase}…` : 'Loading model…';
    case 'ready':
      return `Model ready (${status.quant})`;
    case 'error':
      return `Model error: ${status.message}`;
    default:
      return '';
  }
}

interface PanelToggleProps {
  panelKey: keyof PanelVisibility;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

/** One show/hide toggle for an `App.tsx` panel — reads/writes `store.panels`
 * only; `App.tsx` is the one that actually collapses/expands the panel. */
function PanelToggle({ panelKey, label, icon: Icon }: PanelToggleProps): React.JSX.Element {
  const visible = useStudioStore((s) => s.panels[panelKey]);
  const togglePanel = useStudioStore((s) => s.togglePanel);
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label={visible ? `Hide ${label}` : `Show ${label}`}
      aria-pressed={visible}
      title={visible ? `Hide ${label}` : `Show ${label}`}
      className={cn('h-8 w-8', !visible && 'text-muted-foreground/50')}
      onClick={() => togglePanel(panelKey)}
    >
      <Icon className="h-4 w-4" />
    </Button>
  );
}

/**
 * Studio toolbar: tool selection, playback, tracking, export, panel
 * visibility, and model/device status. See `studio-contracts.md` §3 for the
 * full contract.
 */
export function Toolbar(): React.JSX.Element {
  const tool = useStudioStore((s) => s.tool);
  const setTool = useStudioStore((s) => s.setTool);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const objects = useStudioStore((s) => s.objects);
  const trackState = useStudioStore((s) => s.trackState);
  const startTracking = useStudioStore((s) => s.startTracking);
  const cancelTracking = useStudioStore((s) => s.cancelTracking);
  const exportMatte = useStudioStore((s) => s.exportMatte);
  const exportMp4Cutout = useStudioStore((s) => s.exportMp4Cutout);
  const exportState = useStudioStore((s) => s.exportState);
  const modelStatus = useStudioStore((s) => s.modelStatus);
  const resolvedDevice = useStudioStore((s) => s.resolvedDevice);
  const loadModel = useStudioStore((s) => s.loadModel);

  const isTracking = trackState.phase === 'running';
  const activeClipObjects = activeClipId ? objects.filter((o) => o.clipId === activeClipId) : [];
  const canTrack = Boolean(activeClipId) && activeClipObjects.length > 0 && !isTracking;
  const canExport = Boolean(activeClipId) && exportState.phase !== 'running';

  return (
    <div className="flex items-center gap-3 border-b border-border bg-background px-3 py-2">
      <ToggleGroup
        type="single"
        value={tool}
        onValueChange={(value) => {
          if (!value) return;
          // Friction §0.5: prompting tools are locked out while a propagate()
          // iterator is in flight, mirrored as a store invariant elsewhere,
          // but we also short-circuit here to avoid a flash of dead state.
          if (isTracking && value !== 'select' && value !== 'pan') return;
          setTool(value as ToolMode);
        }}
        variant="outline"
        aria-label="Tool"
      >
        <ToggleGroupItem value="select" aria-label="Select tool">
          <MousePointer2 className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="point-add" aria-label="Add point" disabled={isTracking}>
          <Plus className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="box" aria-label="Box prompt" disabled={isTracking}>
          <Square className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="pan" aria-label="Pan tool">
          <Hand className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>

      <div className="mx-1 h-6 w-px bg-border" />

      <Button
        variant="outline"
        size="icon"
        aria-label={isPlaying ? 'Pause' : 'Play'}
        aria-pressed={isPlaying}
        onClick={() => setIsPlaying(!isPlaying)}
      >
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>

      <div className="mx-1 h-6 w-px bg-border" />

      {isTracking ? (
        <Button variant="destructive" size="sm" onClick={() => cancelTracking()}>
          <X className="h-4 w-4" />
          Cancel
        </Button>
      ) : (
        <Button
          variant="secondary"
          size="sm"
          disabled={!canTrack}
          onClick={() => {
            if (activeClipId) void startTracking(activeClipId);
          }}
        >
          <Play className="h-4 w-4" />
          Track
        </Button>
      )}

      <div className="mx-1 h-6 w-px bg-border" />

      <Button
        variant="outline"
        size="sm"
        disabled={!canExport}
        onClick={() => {
          if (activeClipId) void exportMatte(activeClipId);
        }}
      >
        <Download className="h-4 w-4" />
        Export matte
      </Button>
      <Button
        variant="outline"
        size="sm"
        disabled={!canExport}
        onClick={() => {
          if (activeClipId) void exportMp4Cutout(activeClipId);
        }}
      >
        <Download className="h-4 w-4" />
        Export MP4
      </Button>

      <div className="flex-1" />

      <div className="flex items-center gap-0.5">
        <PanelToggle panelKey="media" label="Media panel" icon={PanelLeft} />
        <PanelToggle panelKey="properties" label="Properties panel" icon={PanelRight} />
        <PanelToggle panelKey="chat" label="Assistant panel" icon={MessageSquare} />
        <PanelToggle panelKey="timeline" label="Timeline" icon={PanelBottom} />
      </div>

      <div className="mx-1 h-6 w-px bg-border" />

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className="flex items-center gap-1.5 rounded-full border border-input px-2.5 py-1"
          role="status"
        >
          <Cpu className="h-3.5 w-3.5" />
          {statusLabel(modelStatus)}
        </span>
        {resolvedDevice ? (
          <span className="rounded-full border border-input px-2.5 py-1">
            {resolvedDevice === 'webgpu' ? 'webgpu' : 'wasm (slow)'}
          </span>
        ) : null}
        {modelStatus.phase === 'idle' ? (
          <Button size="sm" onClick={() => void loadModel()}>
            Load model
          </Button>
        ) : null}
      </div>
    </div>
  );
}
