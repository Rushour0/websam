/**
 * `Timeline.tsx` — horizontal multi-track timeline (studio-contracts.md §3).
 * Takes no props; reads/writes exclusively through `useStudioStore`.
 *
 * Renders `tracks` as lanes with `timelineClips` positioned by
 * `startFrame` and sized by `(outFrame - inFrame + 1) * zoom`. Each clip is a
 * `@dnd-kit/sortable` item (`useSortable`) inside its track's
 * `useDroppable` lane, driven by the App-owned `<DndContext>` — this file
 * only registers draggables/droppables, it does not own `onDragEnd`
 * (contracts.md §3, "DnD ownership"). Hand-rolled pointer-event trim handles
 * on each clip's left/right edge call `store.trimTimelineClip` directly
 * (dnd-kit does drag/reorder, not resize). A draggable playhead scrubber
 * syncs to `store.playhead`, and a ruler renders time ticks derived from
 * `store.zoom`.
 *
 * Edit/remix affordances (all delegate to the store, which owns validation):
 *  - Per-clip hover/selected delete button → `store.removeTimelineClip`.
 *  - Per-track hover delete button → `store.removeTrack`.
 *  - Header buttons: Split-at-playhead → `store.splitTimelineClip`
 *    (raw project-frame playhead), Duplicate → `store.duplicateTimelineClip`.
 *  - Window keydown shortcuts on the selected clip (ignored while typing in a
 *    form field or with a modifier held): Delete/Backspace remove, `s` split,
 *    `d` duplicate. The listener reads store state imperatively via
 *    `getState()` so it never re-binds on selection/playhead changes.
 *  - Auto-fit: when the longest placed clip's duration changes (or the viewport
 *    resizes) `zoom` is fit to the scroll container; the manual zoom slider
 *    remains a working override between those events.
 *
 * Drag payload contract (mirrors `MediaLibrary.tsx`'s `MediaLibraryDragData`
 * for the App-owned `onDragEnd` to discriminate on `event.active.data.current.kind`):
 *  - Track droppable id: `track:${track.id}`, data `{ kind: 'track', trackId }`.
 *  - Timeline-clip sortable id: `timeline-clip:${timelineClip.id}`, data
 *    `{ kind: 'timeline-clip', timelineClipId, trackId }`.
 */
import { useDroppable } from '@dnd-kit/core';
import { SortableContext, horizontalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Copy, Scissors, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

import type { ClipMeta, Track, TimelineClip } from '../store/studio-store.js';
import { useStudioStore } from '../store/studio-store.js';
import { cn } from '../lib/utils.js';
import { Button } from './ui/button.js';

/** Drag payload shape for a Timeline track droppable (see file header). */
export interface TimelineTrackDropData {
  kind: 'track';
  trackId: string;
}

/** Drag payload shape for a Timeline clip sortable item (see file header). */
export interface TimelineClipDragData {
  kind: 'timeline-clip';
  timelineClipId: string;
  trackId: string;
}

const TRACK_HEIGHT = 56;
const RULER_HEIGHT = 28;
const TRACK_GAP = 4;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 40;
/** Assumed fps for ruler time labels when no clip is present to source a real one. */
const FALLBACK_FPS = 30;
const MIN_TRIM_WIDTH_FRAMES = 1;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** "Nice" tick spacing in frames, chosen so ticks land roughly every 60-120px. */
function tickIntervalFrames(zoom: number, fps: number): number {
  const targetPx = 80;
  const rawFrames = targetPx / zoom;
  const rawSeconds = rawFrames / fps;
  const niceSeconds = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600].find((s) => s >= rawSeconds) ?? 3600;
  return Math.max(1, Math.round(niceSeconds * fps));
}

function formatTimecode(frame: number, fps: number): string {
  const totalSec = frame / fps;
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = Math.floor(totalSec % 60);
  const parts = h > 0 ? [h, m, s] : [m, s];
  return parts.map((p, i) => (i === 0 ? String(p) : String(p).padStart(2, '0'))).join(':');
}

interface RulerProps {
  widthPx: number;
  zoom: number;
  fps: number;
  onScrub: (frame: number) => void;
}

function Ruler({ widthPx, zoom, fps, onScrub }: RulerProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const interval = tickIntervalFrames(zoom, fps);
  const totalFrames = Math.ceil(widthPx / zoom);
  const ticks: number[] = [];
  for (let f = 0; f <= totalFrames; f += interval) ticks.push(f);

  const frameFromClientX = useCallback(
    (clientX: number) => {
      const el = containerRef.current;
      if (!el) return 0;
      const rect = el.getBoundingClientRect();
      const x = clientX - rect.left + el.scrollLeft;
      return Math.max(0, Math.round(x / zoom));
    },
    [zoom],
  );

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    onScrub(frameFromClientX(e.clientX));
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.buttons !== 1) return;
    onScrub(frameFromClientX(e.clientX));
  };

  return (
    <div
      ref={containerRef}
      className="relative shrink-0 cursor-pointer select-none border-b border-border bg-muted/40"
      style={{ height: RULER_HEIGHT, width: widthPx }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
    >
      {ticks.map((f) => (
        <div key={f} className="absolute top-0 flex h-full flex-col justify-end" style={{ left: f * zoom }}>
          <div className="h-2 w-px bg-border" />
          <span className="absolute bottom-2 left-1 whitespace-nowrap text-[9px] text-muted-foreground">
            {formatTimecode(f, fps)}
          </span>
        </div>
      ))}
    </div>
  );
}

interface TrimHandleProps {
  side: 'left' | 'right';
  onTrim: (deltaFrames: number) => void;
}

function TrimHandle({ side, onTrim }: TrimHandleProps): React.JSX.Element {
  const startXRef = useRef(0);

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    e.preventDefault();
    startXRef.current = e.clientX;
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.buttons !== 1) return;
    e.stopPropagation();
    const delta = e.clientX - startXRef.current;
    if (delta === 0) return;
    startXRef.current = e.clientX;
    onTrim(delta);
  };

  return (
    <div
      role="slider"
      aria-label={side === 'left' ? 'Trim start' : 'Trim end'}
      aria-orientation="horizontal"
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      className={cn(
        'absolute top-0 z-10 h-full w-2 cursor-ew-resize bg-primary/0 hover:bg-primary/40',
        side === 'left' ? 'left-0' : 'right-0',
      )}
    />
  );
}

interface ClipBlockProps {
  timelineClip: TimelineClip;
  clip: ClipMeta | undefined;
  zoom: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

function ClipBlock({ timelineClip, clip, zoom, isSelected, onSelect }: ClipBlockProps): React.JSX.Element {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: `timeline-clip:${timelineClip.id}`,
    data: { kind: 'timeline-clip', timelineClipId: timelineClip.id, trackId: timelineClip.trackId } satisfies TimelineClipDragData,
  });

  const trimTimelineClip = useStudioStore((s) => s.trimTimelineClip);
  const moveTimelineClip = useStudioStore((s) => s.moveTimelineClip);
  const removeTimelineClip = useStudioStore((s) => s.removeTimelineClip);
  const maxFrame = clip ? Math.max(0, clip.frameCount - 1) : Math.max(timelineClip.outFrame, MIN_TRIM_WIDTH_FRAMES);

  const durationFrames = timelineClip.outFrame - timelineClip.inFrame + 1;
  const widthPx = Math.max(4, durationFrames * zoom);
  const leftPx = timelineClip.startFrame * zoom;

  const handleTrimLeft = (deltaPx: number) => {
    const deltaFrames = Math.round(deltaPx / zoom);
    if (deltaFrames === 0) return;
    const nextIn = clamp(timelineClip.inFrame + deltaFrames, 0, timelineClip.outFrame - MIN_TRIM_WIDTH_FRAMES);
    // A left-trim must advance startFrame by the same amount the in-point
    // actually moved (the clamped delta, not the raw pointer delta) so the
    // block's on-screen left edge (`leftPx = startFrame * zoom`) tracks the
    // new in-point and the out-point/right edge stays fixed in place.
    const appliedDelta = nextIn - timelineClip.inFrame;
    if (appliedDelta === 0) return;
    trimTimelineClip(timelineClip.id, nextIn, timelineClip.outFrame);
    moveTimelineClip(timelineClip.id, timelineClip.trackId, timelineClip.startFrame + appliedDelta);
  };
  const handleTrimRight = (deltaPx: number) => {
    const deltaFrames = Math.round(deltaPx / zoom);
    if (deltaFrames === 0) return;
    const nextOut = clamp(timelineClip.outFrame + deltaFrames, timelineClip.inFrame + MIN_TRIM_WIDTH_FRAMES, maxFrame);
    trimTimelineClip(timelineClip.id, timelineClip.inFrame, nextOut);
  };

  const style = {
    left: leftPx,
    width: widthPx,
    height: TRACK_HEIGHT - 8,
    transform: transform ? CSS.Translate.toString(transform) : undefined,
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(timelineClip.id);
      }}
      className={cn(
        'group/clip absolute top-1 flex cursor-grab items-center overflow-hidden rounded-md border text-[11px] shadow-sm active:cursor-grabbing',
        isSelected ? 'border-primary bg-primary/20' : 'border-input bg-accent/60',
        isDragging && 'opacity-50',
      )}
    >
      <TrimHandle side="left" onTrim={handleTrimLeft} />
      <span className="pointer-events-none truncate px-3 text-foreground">
        {clip?.fileName ?? 'clip'}
      </span>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label={`Delete ${clip?.fileName ?? 'clip'}`}
        className={cn(
          'absolute right-3 top-1/2 z-20 h-5 w-5 shrink-0 -translate-y-1/2',
          isSelected ? 'opacity-100' : 'opacity-0 group-hover/clip:opacity-100',
        )}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          removeTimelineClip(timelineClip.id);
        }}
      >
        <Trash2 className="h-3 w-3" />
      </Button>
      <TrimHandle side="right" onTrim={handleTrimRight} />
    </div>
  );
}

interface TrackRowProps {
  track: Track;
  timelineClips: Record<string, TimelineClip>;
  clips: Record<string, ClipMeta>;
  zoom: number;
  widthPx: number;
  selectedTimelineClipId: string | null;
  onSelectClip: (id: string) => void;
}

function TrackRow({ track, timelineClips, clips, zoom, widthPx, selectedTimelineClipId, onSelectClip }: TrackRowProps): React.JSX.Element {
  const { setNodeRef } = useDroppable({
    id: `track:${track.id}`,
    data: { kind: 'track', trackId: track.id } satisfies TimelineTrackDropData,
  });

  const removeTrack = useStudioStore((s) => s.removeTrack);

  const sortableIds = track.clipIds.map((id) => `timeline-clip:${id}`);

  return (
    <div
      ref={setNodeRef}
      className="group/track relative shrink-0 border-b border-border/60 bg-background"
      style={{ height: TRACK_HEIGHT, width: widthPx }}
    >
      {/*
        First positioned child: paints and hit-tests BELOW later positioned
        siblings (the clip blocks), so a clip at startFrame 0 fully covers it
        and its left trim handle/drag edge keep working — the delete button is
        only clickable on uncovered lane area. No z-index by design.
      */}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label="Delete track"
        className="absolute left-1 top-1/2 h-6 w-6 -translate-y-1/2 rounded-md border border-border bg-background/80 opacity-0 backdrop-blur-sm group-hover/track:opacity-100"
        onClick={() => removeTrack(track.id)}
      >
        <Trash2 className="h-3 w-3" />
      </Button>
      <SortableContext items={sortableIds} strategy={horizontalListSortingStrategy}>
        {track.clipIds.map((id) => {
          const tc = timelineClips[id];
          if (!tc) return null;
          return (
            <ClipBlock
              key={id}
              timelineClip={tc}
              clip={clips[tc.clipId]}
              zoom={zoom}
              isSelected={selectedTimelineClipId === id}
              onSelect={onSelectClip}
            />
          );
        })}
      </SortableContext>
    </div>
  );
}

interface PlayheadProps {
  frame: number;
  zoom: number;
  heightPx: number;
}

function Playhead({ frame, zoom, heightPx }: PlayheadProps): React.JSX.Element {
  return (
    <div
      className="pointer-events-none absolute top-0 z-20 flex flex-col items-center"
      style={{ left: frame * zoom, height: heightPx }}
    >
      <div className="h-2 w-2 -translate-x-1/2 rounded-b-none rounded-t-sm bg-primary" style={{ clipPath: 'polygon(0 0, 100% 0, 50% 100%)' }} />
      <div className="w-px flex-1 bg-primary" />
    </div>
  );
}

/**
 * Horizontal multi-track timeline. See file header for the full contract
 * summary.
 */
export function Timeline(): React.JSX.Element {
  const tracks = useStudioStore((s) => s.tracks);
  const timelineClips = useStudioStore((s) => s.timelineClips);
  const clips = useStudioStore((s) => s.clips);
  const playhead = useStudioStore((s) => s.playhead);
  const zoom = useStudioStore((s) => s.zoom);
  const selection = useStudioStore((s) => s.selection);
  const setPlayhead = useStudioStore((s) => s.setPlayhead);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);
  const setZoom = useStudioStore((s) => s.setZoom);
  const addTrack = useStudioStore((s) => s.addTrack);
  const selectTimelineClip = useStudioStore((s) => s.selectTimelineClip);
  const splitTimelineClip = useStudioStore((s) => s.splitTimelineClip);
  const duplicateTimelineClip = useStudioStore((s) => s.duplicateTimelineClip);

  const [isScrubbing, setIsScrubbing] = useState(false);

  // Window-level keyboard shortcuts for the selected timeline clip. State is
  // read imperatively via `getState()` (never closed over) so the single empty-
  // deps listener always sees the live selection/playhead without re-binding.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const s = useStudioStore.getState();
      const id = s.selection.timelineClipId;
      if (!id) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        s.removeTimelineClip(id);
      } else if (e.key === 's' || e.key === 'S') {
        const tc = s.timelineClips[id];
        if (tc && s.playhead > tc.startFrame && s.playhead < tc.startFrame + (tc.outFrame - tc.inFrame + 1)) {
          e.preventDefault();
          s.splitTimelineClip(id, s.playhead);
        }
      } else if (e.key === 'd' || e.key === 'D') {
        e.preventDefault();
        s.duplicateTimelineClip(id);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const fps = useMemo(() => {
    const clipList = Object.values(clips);
    return clipList[0]?.fps ?? FALLBACK_FPS;
  }, [clips]);

  const maxEndFrame = useMemo(() => {
    let max = 0;
    for (const tc of Object.values(timelineClips)) {
      max = Math.max(max, tc.startFrame + (tc.outFrame - tc.inFrame + 1));
    }
    return max;
  }, [timelineClips]);

  // Total source length of the longest placed clip, used to auto-fit the
  // timeline to the viewport. Falls back to `outFrame + 1` when the source
  // clip metadata is not (yet) available.
  const fullDurationFrames = useMemo(() => {
    let max = 0;
    for (const tc of Object.values(timelineClips)) {
      max = Math.max(max, clips[tc.clipId]?.frameCount ?? tc.outFrame + 1);
    }
    return max;
  }, [timelineClips, clips]);

  // One-second tail past the last clip: keeps a drop zone visible and feeds the
  // same slack into the fit divisor so fit never leaves a permanent scrollbar.
  const tailFrames = Math.round(fps);
  const contentWidthPx =
    fullDurationFrames > 0
      ? Math.max(800, (Math.max(fullDurationFrames, maxEndFrame) + tailFrames) * zoom)
      : Math.max(800, Math.round(fps * 5) * zoom);
  const tracksHeightPx = tracks.length * (TRACK_HEIGHT + TRACK_GAP);

  const selectedTc = selection.timelineClipId ? timelineClips[selection.timelineClipId] : undefined;
  const canSplit =
    selectedTc != null &&
    playhead > selectedTc.startFrame &&
    playhead < selectedTc.startFrame + (selectedTc.outFrame - selectedTc.inFrame + 1);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const prevFullDurationRef = useRef(0);
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el || fullDurationFrames <= 0) {
      prevFullDurationRef.current = fullDurationFrames;
      return;
    }
    const fit = () => {
      const w = el.clientWidth;
      if (w > 0) setZoom(clamp(w / (fullDurationFrames + tailFrames), MIN_ZOOM, MAX_ZOOM));
    };
    if (prevFullDurationRef.current !== fullDurationFrames) {
      prevFullDurationRef.current = fullDurationFrames;
      fit();
    }
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
    // `zoom` is deliberately excluded (and never read inside `fit`) so the
    // manual zoom slider stays a working override between duration/size changes.
  }, [fullDurationFrames, tailFrames, setZoom]);

  const handleScrub = useCallback(
    (frame: number) => {
      if (!isScrubbing) {
        setIsPlaying(false);
        setIsScrubbing(true);
      }
      setPlayhead(Math.max(0, frame));
    },
    [isScrubbing, setIsPlaying, setPlayhead],
  );

  const orderedTracks = useMemo(() => [...tracks].sort((a, b) => a.order - b.order), [tracks]);

  return (
    <div
      className="flex h-full flex-col overflow-hidden bg-background"
      onPointerUp={() => setIsScrubbing(false)}
    >
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-2 py-1">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Timeline</span>
        <button
          type="button"
          onClick={() => addTrack()}
          className="rounded-md border border-input px-2 py-0.5 text-[11px] hover:bg-accent"
        >
          + Track
        </button>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label="Split at playhead"
          title="Split at playhead (S)"
          className="h-6 w-6"
          disabled={!canSplit}
          onClick={() => selectedTc && splitTimelineClip(selectedTc.id, playhead)}
        >
          <Scissors className="h-3.5 w-3.5" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label="Duplicate clip"
          title="Duplicate clip (D)"
          className="h-6 w-6"
          disabled={selection.timelineClipId === null}
          onClick={() => selection.timelineClipId && duplicateTimelineClip(selection.timelineClipId)}
        >
          <Copy className="h-3.5 w-3.5" />
        </Button>
        <div className="ml-auto flex items-center gap-1 text-[11px] text-muted-foreground">
          <span>Zoom</span>
          <input
            type="range"
            min={MIN_ZOOM}
            max={MAX_ZOOM}
            step={0.5}
            value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))}
            className="w-24 accent-primary"
          />
          <span className="w-10 tabular-nums">{zoom.toFixed(1)}x</span>
        </div>
      </div>

      <div ref={scrollContainerRef} className="relative flex-1 overflow-auto">
        <div className="relative" style={{ width: contentWidthPx }}>
          <Ruler widthPx={contentWidthPx} zoom={zoom} fps={fps} onScrub={handleScrub} />

          <div className="relative flex flex-col gap-1 pt-1">
            {orderedTracks.length === 0 ? (
              <p className="p-4 text-center text-xs text-muted-foreground">
                No tracks yet — drop a clip from Media to create one, or click + Track.
              </p>
            ) : (
              orderedTracks.map((track) => (
                <TrackRow
                  key={track.id}
                  track={track}
                  timelineClips={timelineClips}
                  clips={clips}
                  zoom={zoom}
                  widthPx={contentWidthPx}
                  selectedTimelineClipId={selection.timelineClipId}
                  onSelectClip={selectTimelineClip}
                />
              ))
            )}
          </div>

          <Playhead frame={playhead} zoom={zoom} heightPx={RULER_HEIGHT + tracksHeightPx} />
        </div>
      </div>
    </div>
  );
}

export default Timeline;
