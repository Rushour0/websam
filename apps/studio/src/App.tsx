/**
 * App/wiring owner — studio-contracts.md §3.
 *
 * Boots the store, mounts `Toolbar` above a `react-resizable-panels` layout
 * (`MediaLibrary | (PreviewCanvas over PropertiesPanel)` with `Timeline`
 * docked below), owns the `<video>`/canvas ref handoff into
 * `src/segmentation/session-manager.ts` (passed as props, not store state,
 * per §3's `PreviewCanvas` contract), and owns the single shared
 * `<DndContext>` for MediaLibrary→Timeline drops and intra-Timeline
 * reorder/move (§3's DnD-ownership note). Also renders a small global error
 * boundary + `notice` toast for the store's `notice` field.
 */
import { Component, useCallback, useRef } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { DndContext } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

import { useStudioStore } from './store/studio-store.js';
import { Toolbar } from './components/Toolbar.js';
import { MediaLibrary } from './components/MediaLibrary.js';
import { PreviewCanvas } from './components/PreviewCanvas.js';
import { PropertiesPanel } from './components/PropertiesPanel.js';
import { Timeline } from './components/Timeline.js';

/**
 * Shape stashed on a `@dnd-kit/core` draggable's `data.current` by
 * `MediaLibrary`/`Timeline` (§3's DnD-ownership note). Matches
 * `MediaLibrary.tsx`'s exported `MediaLibraryDragData` and `Timeline.tsx`'s
 * exported `TimelineClipDragData` verbatim (both files' own JSDoc headers
 * document the same shapes).
 */
type DragData =
  | { kind: 'media-library-clip'; clipId: string }
  | { kind: 'timeline-clip'; timelineClipId: string; trackId: string };

/**
 * Shape stashed on a droppable's `data.current` by `Timeline` (one per track
 * lane) — matches `Timeline.tsx`'s exported `TimelineTrackDropData`
 * verbatim. It carries only `trackId`, not a drop-point frame — `Timeline`'s
 * droppable is the whole lane, not a positioned drop target, so `onDragEnd`
 * derives `startFrame` itself (see `handleDragEnd` below).
 */
interface DropData {
  kind: 'track';
  trackId: string;
}

function isDragData(value: unknown): value is DragData {
  return typeof value === 'object' && value !== null && 'kind' in value;
}

function isDropData(value: unknown): value is DropData {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'track';
}

/** Minimal class-component error boundary — hooks have no boundary equivalent. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console -- last-resort surface for render-phase crashes the store's notice can't catch
    console.error('websam studio: unhandled render error', error, info);
  }

  override render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-background p-8 text-center text-foreground">
          <h1 className="text-lg font-semibold">Something went wrong</h1>
          <p className="max-w-md text-sm text-muted-foreground">{this.state.error.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

function NoticeToast(): JSX.Element | null {
  const notice = useStudioStore((s) => s.notice);
  const clearNotice = useStudioStore((s) => s.clearNotice);

  if (!notice) return null;

  const toneClass =
    notice.kind === 'error'
      ? 'border-destructive/50 bg-destructive text-destructive-foreground'
      : 'border-border bg-secondary text-secondary-foreground';

  return (
    <div
      role="alert"
      className={`pointer-events-auto fixed bottom-4 right-4 z-50 flex max-w-sm flex-col gap-1 rounded-md border p-3 shadow-lg ${toneClass}`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-semibold">{notice.title}</p>
        <button
          type="button"
          onClick={clearNotice}
          className="text-xs opacity-70 hover:opacity-100"
          aria-label="Dismiss notice"
        >
          ✕
        </button>
      </div>
      <p className="text-xs opacity-90">{notice.detail}</p>
    </div>
  );
}

/** Root component — mounted by `main.tsx`. */
export function App(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageContainerRef = useRef<HTMLDivElement>(null);

  const addTimelineClip = useStudioStore((s) => s.addTimelineClip);
  const moveTimelineClip = useStudioStore((s) => s.moveTimelineClip);
  const addTrack = useStudioStore((s) => s.addTrack);
  const tracks = useStudioStore((s) => s.tracks);
  const timelineClips = useStudioStore((s) => s.timelineClips);
  const zoom = useStudioStore((s) => s.zoom);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const active = event.active.data.current;
      const over = event.over?.data.current;
      if (!isDragData(active) || !isDropData(over)) return;

      // MediaLibrary/Timeline drop targets are whole track lanes, not a
      // positioned drop point (`Timeline.tsx`'s `TimelineTrackDropData`
      // carries only `trackId`) — a track has never been created yet on an
      // empty board, so fall back to creating one so the drop still lands.
      const trackId = tracks.some((t) => t.id === over.trackId) ? over.trackId : addTrack();

      if (active.kind === 'media-library-clip') {
        // No drop-point frame is available — append the new clip after the
        // last clip currently on the target track (falls back to frame 0 for
        // an empty track), avoiding an overlap with existing timeline clips.
        const trackClips = Object.values(timelineClips).filter((tc) => tc.trackId === trackId);
        const startFrame = trackClips.reduce(
          (end, tc) => Math.max(end, tc.startFrame + (tc.outFrame - tc.inFrame + 1)),
          0,
        );
        addTimelineClip(active.clipId, trackId, startFrame);
      } else if (active.kind === 'timeline-clip') {
        // Derive the new position from the pointer delta Timeline's own
        // zoom-aware layout uses (`px-per-frame`), applied to the clip's
        // pre-drag `startFrame`.
        const existing = timelineClips[active.timelineClipId];
        const deltaFrames = zoom > 0 ? Math.round(event.delta.x / zoom) : 0;
        const startFrame = Math.max(0, (existing?.startFrame ?? 0) + deltaFrames);
        moveTimelineClip(active.timelineClipId, trackId, startFrame);
      }
    },
    [tracks, timelineClips, zoom, addTrack, addTimelineClip, moveTimelineClip],
  );

  return (
    <ErrorBoundary>
      <div className="flex h-full w-full flex-col bg-background text-foreground">
        <Toolbar />
        <DndContext onDragEnd={handleDragEnd}>
          <div className="min-h-0 flex-1">
            <PanelGroup direction="vertical">
              <Panel defaultSize={70} minSize={30}>
                <PanelGroup direction="horizontal">
                  <Panel defaultSize={20} minSize={12}>
                    <MediaLibrary />
                  </Panel>
                  <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-ring" />
                  <Panel defaultSize={60} minSize={30}>
                    <PreviewCanvas videoRef={videoRef} stageContainerRef={stageContainerRef} />
                  </Panel>
                  <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-ring" />
                  <Panel defaultSize={20} minSize={14}>
                    <PropertiesPanel />
                  </Panel>
                </PanelGroup>
              </Panel>
              <PanelResizeHandle className="h-1 bg-border transition-colors hover:bg-ring" />
              <Panel defaultSize={30} minSize={15}>
                <Timeline />
              </Panel>
            </PanelGroup>
          </div>
        </DndContext>
        <NoticeToast />
      </div>
    </ErrorBoundary>
  );
}
