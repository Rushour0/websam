/**
 * App/wiring owner — studio-contracts.md §3.
 *
 * Boots the store, mounts `Toolbar` above a `react-resizable-panels` layout
 * (`MediaLibrary | PreviewCanvas | PropertiesPanel | ChatPanel` with `Timeline`
 * docked below), owns the `<video>`/canvas ref handoff into
 * `src/segmentation/session-manager.ts` (passed as props, not store state,
 * per §3's `PreviewCanvas` contract), and owns the single shared
 * `<DndContext>` for MediaLibrary→Timeline drops and intra-Timeline
 * reorder/move (§3's DnD-ownership note); media-library drops now call
 * `addClipAsTracks` (auto-splitting into video + audio tracks, landing on
 * either a lane or the timeline board droppable) instead of a single
 * add-track/add-clip pair, and App mounts the store-driven `AudioPlayback`
 * engine. Also renders a small global error boundary + `notice` toast for the
 * store's `notice` field.
 */
import { Component, useCallback, useEffect, useRef } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { DndContext, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';

import { useStudioStore } from './store/studio-store.js';
import type { TimelineClip } from './store/studio-store.js';
import { Toolbar } from './components/Toolbar.js';
import { MediaLibrary } from './components/MediaLibrary.js';
import { PreviewCanvas } from './components/PreviewCanvas.js';
import { PropertiesPanel } from './components/PropertiesPanel.js';
import { ChatPanel } from './components/ChatPanel.js';
import { Timeline } from './components/Timeline.js';
import { AudioPlayback } from './components/AudioPlayback.js';
import { disposeAllClips } from './segmentation/session-manager.js';
import { disposeSegmenter } from './segmentation/segmenter-lifecycle.js';

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
 * Shape stashed on a droppable's `data.current` by `Timeline` — matches
 * `Timeline.tsx`'s exported drop-data shapes verbatim. A `'track'` droppable
 * (one per track lane) carries only `trackId`, not a drop-point frame; the
 * `'timeline-bg'` droppable is the whole timeline board (covers the empty
 * state and the gaps below the lanes). `Timeline`'s droppables are areas, not
 * positioned drop targets, so `onDragEnd` derives `startFrame` itself (see
 * `handleDragEnd` below).
 */
type DropData = { kind: 'track'; trackId: string } | { kind: 'timeline-bg' };

function isDragData(value: unknown): value is DragData {
  return typeof value === 'object' && value !== null && 'kind' in value;
}

function isDropData(value: unknown): value is DropData {
  const kind = (value as { kind?: unknown } | null)?.kind;
  return typeof value === 'object' && value !== null && (kind === 'track' || kind === 'timeline-bg');
}

/**
 * Anti-overlap clamp for `handleDragEnd` (bug #2): pushes `desiredStart`
 * right past any timeline clip already on `trackId` (excluding `excludeId`,
 * the clip being dragged in the intra-timeline move case) that its
 * `[start, start + durationFrames)` span would overlap. Iterates to a fixed
 * point since pushing past one clip can create a new overlap with the next.
 */
function findNonOverlappingStart(
  timelineClips: Record<string, TimelineClip>,
  trackId: string,
  desiredStart: number,
  durationFrames: number,
  excludeId?: string,
): number {
  const others = Object.values(timelineClips)
    .filter((tc) => tc.trackId === trackId && tc.id !== excludeId)
    .map((tc) => ({ start: tc.startFrame, end: tc.startFrame + (tc.outFrame - tc.inFrame + 1) }))
    .sort((a, b) => a.start - b.start);

  let start = Math.max(0, desiredStart);
  let moved = true;
  while (moved) {
    moved = false;
    for (const o of others) {
      const end = start + durationFrames;
      if (start < o.end && end > o.start) {
        start = o.end;
        moved = true;
      }
    }
  }
  return start;
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

  const horizontalGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const verticalGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const panels = useStudioStore((s) => s.panels);

  // Base (all-visible) size weights, same ratios as each Panel's `defaultSize`
  // below — kept here too so a hidden panel's freed share redistributes back
  // proportionally to these weights once every panel is visible again.
  const HORIZONTAL_WEIGHTS = { media: 20, preview: 44, properties: 16, chat: 20 };
  const VERTICAL_WEIGHTS = { main: 70, timeline: 30 };

  /** Percentage layout array for a `PanelGroup`: 0 for each hidden panel,
   * the rest re-normalized so the VISIBLE panels' weights still sum to 100. */
  function computeLayout(weights: Record<string, number>, visible: Record<string, boolean>): number[] {
    const keys = Object.keys(weights);
    const totalVisibleWeight = keys.reduce((sum, k) => sum + (visible[k] ? weights[k]! : 0), 0);
    if (totalVisibleWeight <= 0) return keys.map(() => 0);
    return keys.map((k) => (visible[k] ? (weights[k]! / totalVisibleWeight) * 100 : 0));
  }

  // Store -> panel: whenever `store.panels` changes (Toolbar's toggle buttons
  // write there), assign each `PanelGroup`'s ENTIRE layout array atomically
  // via `setLayout` — deliberately NOT per-panel `.collapse()`/`.expand()`.
  // Confirmed by direct instrumentation: collapsing one panel via its own
  // imperative handle redistributes freed space across ALL siblings
  // (including already-collapsed ones) via the library's own internal
  // layout recompute, which can silently re-grow an already-hidden sibling
  // back to a nonzero size as a side effect of a LATER, unrelated collapse —
  // repeatable in sequence (hide Properties, then hide Chat, silently
  // re-expands Properties). `setLayout([...])` sets every panel's size in
  // one atomic call instead of relying on N independent redistribution
  // passes, which sidesteps that failure mode entirely.
  useEffect(() => {
    horizontalGroupRef.current?.setLayout(
      computeLayout(HORIZONTAL_WEIGHTS, { media: panels.media, preview: true, properties: panels.properties, chat: panels.chat }),
    );
    verticalGroupRef.current?.setLayout(computeLayout(VERTICAL_WEIGHTS, { main: true, timeline: panels.timeline }));
  }, [panels.media, panels.properties, panels.chat, panels.timeline]);

  const addClipAsTracks = useStudioStore((s) => s.addClipAsTracks);
  const moveTimelineClip = useStudioStore((s) => s.moveTimelineClip);
  const addTrack = useStudioStore((s) => s.addTrack);
  const tracks = useStudioStore((s) => s.tracks);
  const timelineClips = useStudioStore((s) => s.timelineClips);
  const zoom = useStudioStore((s) => s.zoom);

  // Bug #3: without an activation constraint, a plain click on a
  // draggable+clickable card (ClipCard/ClipBlock) can be swallowed by a
  // sub-pixel micro-drag before dnd-kit ever fires onClick. Require the
  // pointer to move a few px before a drag starts.
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const active = event.active.data.current;
      const over = event.over?.data.current;
      if (!isDragData(active) || !isDropData(over)) return;

      if (active.kind === 'media-library-clip') {
        // Media drops ALWAYS create fresh track(s): a video track plus, when
        // clips[clipId].hasAudio, an audio track — each with its own TimelineClip.
        // Works from any drop target ('track' lane or the 'timeline-bg' board, which
        // covers the empty state). startFrame 0 needs no overlap pass here: the new
        // tracks start empty and addClipAsTracks runs the store's own
        // findNonOverlappingStart per created track. Known consequence: dropping onto
        // an EXISTING lane no longer appends to it (intended behavior change).
        addClipAsTracks(active.clipId, 0);
      } else if (active.kind === 'timeline-clip') {
        const existing = timelineClips[active.timelineClipId];
        if (!existing) return;
        // Lane drop: use it (with the stale-lane addTrack() fallback, which still
        // matters for intra-timeline moves). Board-background drop: treat as a
        // same-track horizontal move.
        const trackId =
          over.kind === 'track'
            ? (tracks.some((t) => t.id === over.trackId) ? over.trackId : addTrack())
            : existing.trackId;
        const deltaFrames = zoom > 0 ? Math.round(event.delta.x / zoom) : 0;
        const desiredStart = Math.max(0, existing.startFrame + deltaFrames);
        const durationFrames = existing.outFrame - existing.inFrame + 1;
        const startFrame = findNonOverlappingStart(timelineClips, trackId, desiredStart, durationFrames, active.timelineClipId);
        moveTimelineClip(active.timelineClipId, trackId, startFrame);
      }
    },
    [tracks, timelineClips, zoom, addTrack, addClipAsTracks, moveTimelineClip],
  );

  useEffect(() => {
    const onPageHide = (): void => {
      disposeAllClips();
      void disposeSegmenter();
    };
    window.addEventListener('pagehide', onPageHide);
    return () => {
      window.removeEventListener('pagehide', onPageHide);
      disposeAllClips();
      void disposeSegmenter();
    };
  }, []);

  return (
    <ErrorBoundary>
      <div className="flex h-full w-full flex-col bg-background text-foreground">
        <Toolbar />
        <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
          <div className="min-h-0 flex-1">
            <PanelGroup ref={verticalGroupRef} direction="vertical">
              <Panel id="panel-main" defaultSize={70} minSize={30}>
                <PanelGroup ref={horizontalGroupRef} direction="horizontal">
                  <Panel id="panel-media" defaultSize={20} minSize={12} collapsible collapsedSize={0}>
                    <MediaLibrary />
                  </Panel>
                  <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-ring" />
                  <Panel id="panel-preview" defaultSize={44} minSize={30}>
                    <PreviewCanvas videoRef={videoRef} stageContainerRef={stageContainerRef} />
                  </Panel>
                  <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-ring" />
                  <Panel id="panel-properties" defaultSize={16} minSize={14} collapsible collapsedSize={0}>
                    <PropertiesPanel />
                  </Panel>
                  <PanelResizeHandle className="w-1 bg-border transition-colors hover:bg-ring" />
                  <Panel id="panel-chat" defaultSize={20} minSize={14} collapsible collapsedSize={0}>
                    <ChatPanel />
                  </Panel>
                </PanelGroup>
              </Panel>
              <PanelResizeHandle className="h-1 bg-border transition-colors hover:bg-ring" />
              <Panel id="panel-timeline" defaultSize={30} minSize={15} collapsible collapsedSize={0}>
                <Timeline />
              </Panel>
            </PanelGroup>
          </div>
        </DndContext>
        <AudioPlayback />
        <NoticeToast />
      </div>
    </ErrorBoundary>
  );
}
