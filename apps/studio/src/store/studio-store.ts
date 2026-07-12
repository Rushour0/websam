/**
 * The studio's single zustand store — every component reads/writes app state
 * only through this file (see `apps/studio/docs/studio-contracts.md` §2).
 *
 * Non-serializable runtime objects (`Segmenter`, `VideoSession`) are NOT
 * store state — they live in a module-level `Map` inside
 * `src/segmentation/session-manager.ts`. `MaskTimeline` instances ARE store
 * state (per the contract) but are held as plain object references and
 * mutated in place; callers that mutate one must re-set the `maskTimelines`
 * record with a fresh outer object to notify subscribers, e.g.:
 * `set({ maskTimelines: { ...get().maskTimelines } })`.
 *
 * Uses `subscribeWithSelector` so high-frequency consumers (the preview
 * canvas's `requestVideoFrameCallback` loop driving `playhead`) can subscribe
 * to a narrow slice without whole-tree re-renders:
 * `useStudioStore((s) => s.playhead)`.
 */
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { LoadProgressEvent, MaskResult, Prompt } from '@websam3/core';
import type { MaskTimeline } from '@websam3/video-editing';

// ---------------------------------------------------------------------------
// Segmentation integration seam (owned by other wave files under
// src/segmentation/ and src/video/ — see studio-contracts.md §4). The store
// calls into these; it does not implement the underlying session/exporter
// logic itself. Each delegate receives the store's own `get`/`set` pair so it
// can read/update store state without importing this module (avoids a
// circular value-import; only the `StudioState` type flows the other way).
// ---------------------------------------------------------------------------
import { loadSegmenter } from '../segmentation/segmenter-lifecycle.js';
import {
  activateClip as activateClipSession,
  addPromptObject as addPromptObjectSession,
  refineObject as refineObjectSession,
  removeObject as removeObjectSession,
  disposeClipSession,
} from '../segmentation/session-manager.js';
import { startTracking as startTrackingSession, cancelTracking as cancelTrackingSession } from '../segmentation/propagate-loop.js';
import { exportMatte as exportMatteImpl, exportMp4Cutout as exportMp4CutoutImpl } from '../segmentation/export.js';
import { probeClipMeta } from '../video/frame-source.js';

// ---------------------------------------------------------------------------
// State shape (verbatim from studio-contracts.md §2)
// ---------------------------------------------------------------------------

export interface ClipMeta {
  id: string;
  fileName: string;
  /** ORIGINAL file — `attachSource` needs this exact Blob (never the preview `<video>` element). */
  blob: Blob;
  /** `URL.createObjectURL(blob)`, for `<video src>`. */
  objectUrl: string;
  durationSec: number;
  fps: number;
  width: number;
  height: number;
  frameCount: number;
  frameCountGuessed: boolean;
  /**
   * Whether the source file carries at least one audio track. Populated by
   * `probeClipMeta` at import; survives `session-manager`'s `attachSource`
   * correction because that `set()` spreads `...clip`. Drives
   * `addClipAsTracks`, which only creates a second (audio-kind) Track when true.
   */
  hasAudio: boolean;
}

export interface TimelineClip {
  id: string;
  clipId: string;
  trackId: string;
  /** Position on the track, project frames. */
  startFrame: number;
  /** Trim range, source-clip frame indices. */
  inFrame: number;
  outFrame: number;
}

export type TrackKind = 'video' | 'audio';

export interface Track {
  id: string;
  order: number;
  kind: TrackKind;
  name: string;
  clipIds: string[];
}

export type ToolMode = 'select' | 'point-add' | 'point-remove' | 'box' | 'pan';

export interface TrackedObject {
  objectId: number;
  clipId: string;
  color: string;
  label: string;
  promptFrame: number;
}

export type ModelStatus =
  | { phase: 'idle' }
  | { phase: 'loading'; progress?: { phase: string; loaded?: number; total?: number; file?: string } }
  | { phase: 'ready'; device: 'webgpu' | 'wasm'; quant: string; totalBytes: number }
  | { phase: 'error'; message: string; code?: string };

export type TrackState =
  | { phase: 'idle' }
  | { phase: 'running'; clipId: string; frameIndex: number; frameCount: number }
  | { phase: 'done'; clipId: string }
  | { phase: 'error'; message: string; code?: string };

export type ExportState =
  | { phase: 'idle' }
  | { phase: 'running'; framesDone: number; frameCount: number; kind: 'matte' | 'mp4' }
  | { phase: 'done'; fileName: string; framesExported: number }
  | { phase: 'error'; message: string };

/** One entry per hide/show-able panel in `App.tsx`'s layout (PreviewCanvas is
 * always visible — it's the primary content, not a toggleable panel). */
export interface PanelVisibility {
  media: boolean;
  properties: boolean;
  chat: boolean;
  timeline: boolean;
}

export interface StudioState {
  // media + timeline
  clips: Record<string, ClipMeta>;
  tracks: Track[];
  timelineClips: Record<string, TimelineClip>;
  selection: { timelineClipId: string | null; objectId: number | null };
  playhead: number;
  isPlaying: boolean;
  zoom: number;
  /**
   * Hoisted preview-audio state (verifier fix: the "transient local state"
   * carve-out no longer applies once `AudioPlayback` also needs mute/volume —
   * otherwise the speaker button silently stops working exactly when an audio
   * track supplies the sound). `previewMuted` starts `true` to preserve today's
   * silent-by-default preview; `previewVolume` is `1`.
   */
  previewMuted: boolean;
  previewVolume: number;
  /** Which side/bottom panels are shown — `App.tsx` drives its
   * `react-resizable-panels` collapse/expand from this; `Toolbar.tsx` renders
   * the show/hide toggle buttons (this component only reads/writes the
   * store, per the panel-ownership convention — App owns the panel refs). */
  panels: PanelVisibility;

  // prompting / segmentation UI
  tool: ToolMode;
  activeClipId: string | null;
  objects: TrackedObject[];
  maskTimelines: Record<string, MaskTimeline>;
  liveMasksAtFrame: Record<number, MaskResult>;
  /** Mask-overlay opacity in `PreviewCanvas`, 0..1 (store-owned per studio-contracts.md §3). */
  maskOpacity: number;

  // segmenter lifecycle
  modelStatus: ModelStatus;
  resolvedDevice: 'webgpu' | 'wasm' | null;
  trackState: TrackState;
  exportState: ExportState;
  notice: { title: string; detail: string; kind: 'error' | 'warn' } | null;

  // actions
  importClip: (file: File) => Promise<void>;
  removeClip: (clipId: string) => void;

  addTrack: (kind?: TrackKind, name?: string) => string;
  addClipAsTracks: (clipId: string, startFrame: number) => { videoTrackId: string; audioTrackId: string | null };
  renameTrack: (trackId: string, name: string) => void;
  addTimelineClip: (clipId: string, trackId: string, startFrame: number) => string;
  moveTimelineClip: (timelineClipId: string, trackId: string, startFrame: number) => void;
  trimTimelineClip: (timelineClipId: string, inFrame: number, outFrame: number) => void;
  removeTimelineClip: (timelineClipId: string) => void;
  reorderTracks: (trackIds: string[]) => void;
  removeTrack: (trackId: string) => void;
  splitTimelineClip: (timelineClipId: string, atFrame: number) => string | null;
  duplicateTimelineClip: (timelineClipId: string) => string | null;

  setPlayhead: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setZoom: (pxPerFrame: number) => void;
  setPreviewMuted: (muted: boolean) => void;
  setPreviewVolume: (volume: number) => void;

  togglePanel: (key: keyof PanelVisibility) => void;
  setPanelVisible: (key: keyof PanelVisibility, visible: boolean) => void;

  setTool: (tool: ToolMode) => void;
  selectTimelineClip: (id: string | null) => void;
  selectObject: (objectId: number | null) => void;
  setMaskOpacity: (opacity: number) => void;

  loadModel: () => Promise<void>;

  activateClip: (clipId: string) => Promise<void>;
  addPromptObject: (clipId: string, frameIndex: number, prompts: Prompt[]) => Promise<void>;
  refineObject: (clipId: string, objectId: number, frameIndex: number, prompts: Prompt[]) => Promise<void>;
  removeObject: (clipId: string, objectId: number) => void;
  startTracking: (clipId: string, startFrame?: number) => Promise<void>;
  cancelTracking: () => void;

  exportMatte: (clipId: string) => Promise<void>;
  exportMp4Cutout: (clipId: string) => Promise<void>;

  setNotice: (notice: StudioState['notice']) => void;
  clearNotice: () => void;
}

/**
 * The `get`/`set` pair handed to segmentation-seam delegates
 * (`session-manager.ts`, `propagate-loop.ts`, `export.ts`,
 * `segmenter-lifecycle.ts`) so they can read/mutate store state without
 * importing this module. Matches zustand's own `StateCreator` parameter
 * types, so delegates can be written as plain
 * `(get: StudioGet, set: StudioSet, ...args) => ...` functions.
 */
export type StudioGet = () => StudioState;
export type StudioSet = (
  partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>),
  replace?: false,
) => void;

function genId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Smallest non-negative track position at/after `desiredStart` where a clip of
 * `durationFrames` fits without overlapping any existing same-track clip
 * (independent copy of App.tsx's drop-placement cascade; not imported to avoid a
 * component→store value dependency). `excludeId` skips the clip being moved.
 */
function findNonOverlappingStart(
  timelineClips: Record<string, TimelineClip>,
  trackId: string,
  desiredStart: number,
  durationFrames: number,
  excludeId?: string,
): number {
  const occupied = Object.values(timelineClips)
    .filter((tc) => tc.trackId === trackId && tc.id !== excludeId)
    .map((tc) => ({ start: tc.startFrame, end: tc.startFrame + (tc.outFrame - tc.inFrame + 1) }))
    .sort((a, b) => a.start - b.start);

  let start = Math.max(0, desiredStart);
  let moved = true;
  while (moved) {
    moved = false;
    for (const range of occupied) {
      if (start < range.end && start + durationFrames > range.start) {
        start = range.end;
        moved = true;
      }
    }
  }
  return start;
}

/** Friendly-notice mapping for errors thrown by the segmentation seam. */
function describeThrown(err: unknown): { title: string; detail: string } {
  const detail = err instanceof Error ? err.message : String(err);
  const title = err instanceof Error && err.name !== 'Error' ? err.name : 'Something went wrong';
  return { title, detail };
}

export const useStudioStore = create<StudioState>()(
  subscribeWithSelector((set, get) => ({
    // ---------------------------------------------------------------------
    // initial state
    // ---------------------------------------------------------------------
    clips: {},
    tracks: [],
    timelineClips: {},
    selection: { timelineClipId: null, objectId: null },
    playhead: 0,
    isPlaying: false,
    previewMuted: true,
    previewVolume: 1,
    zoom: 4,
    panels: { media: true, properties: true, chat: true, timeline: true },

    tool: 'select',
    activeClipId: null,
    objects: [],
    maskTimelines: {},
    liveMasksAtFrame: {},
    maskOpacity: 0.5,

    modelStatus: { phase: 'idle' },
    resolvedDevice: null,
    trackState: { phase: 'idle' },
    exportState: { phase: 'idle' },
    notice: null,

    // ---------------------------------------------------------------------
    // media + timeline actions
    // ---------------------------------------------------------------------
    importClip: async (file: File) => {
      const probed = await probeClipMeta(file);
      const id = genId();
      const objectUrl = URL.createObjectURL(file);
      const clip: ClipMeta = {
        id,
        fileName: file.name,
        blob: file,
        objectUrl,
        durationSec: probed.durationSec,
        fps: probed.fps,
        width: probed.width,
        height: probed.height,
        frameCount: probed.frameCount,
        frameCountGuessed: probed.frameCountGuessed,
        hasAudio: probed.hasAudio,
      };
      set((state) => ({ clips: { ...state.clips, [id]: clip } }));
    },

    removeClip: (clipId: string) => {
      const state = get();
      const clip = state.clips[clipId];
      if (!clip) return;

      // Dispose the (possibly warm) segmentation session for this clip — the
      // only place sessions are torn down per contracts.md §4.2.
      disposeClipSession(clipId);
      URL.revokeObjectURL(clip.objectUrl);

      const nextClips = { ...state.clips };
      delete nextClips[clipId];

      const removedTimelineClipIds = new Set(
        Object.values(state.timelineClips)
          .filter((tc) => tc.clipId === clipId)
          .map((tc) => tc.id),
      );
      const nextTimelineClips = { ...state.timelineClips };
      for (const id of removedTimelineClipIds) delete nextTimelineClips[id];

      const nextTracks = state.tracks.map((t) => ({
        ...t,
        clipIds: t.clipIds.filter((id) => !removedTimelineClipIds.has(id)),
      }));

      const nextMaskTimelines = { ...state.maskTimelines };
      delete nextMaskTimelines[clipId];

      const nextObjects = state.objects.filter((o) => o.clipId !== clipId);

      set({
        clips: nextClips,
        timelineClips: nextTimelineClips,
        tracks: nextTracks,
        maskTimelines: nextMaskTimelines,
        objects: nextObjects,
        activeClipId: state.activeClipId === clipId ? null : state.activeClipId,
        selection:
          state.selection.timelineClipId && removedTimelineClipIds.has(state.selection.timelineClipId)
            ? { timelineClipId: null, objectId: null }
            : state.selection,
      });
    },

    addTrack: (kind = 'video', name) => {
      const id = genId();
      set((state) => ({ tracks: [...state.tracks, { id, order: state.tracks.length, kind, name: name ?? `Track ${state.tracks.length + 1}`, clipIds: [] }] }));
      return id;
    },

    // Places a source clip on a brand-new video Track (and, iff the source
    // `hasAudio`, an adjacent brand-new audio Track — `audio.order ===
    // video.order + 1`) as two INDEPENDENT TimelineClips (each its own genId,
    // inFrame 0 / outFrame frameCount-1) referencing the same clipId. Composes
    // the existing addTrack/addTimelineClip; the per-track
    // findNonOverlappingStart is identity on the empty new tracks but preserves
    // the overlap-avoidance contract (and clamps negative startFrame to 0).
    // No-op returning empty ids on unknown clipId.
    addClipAsTracks: (clipId, startFrame) => {
      const clip = get().clips[clipId];
      if (!clip) return { videoTrackId: '', audioTrackId: null }; // unknown clip: mutate nothing
      const durationFrames = Math.max(1, clip.frameCount);
      const videoTrackId = get().addTrack('video', `Video — ${clip.fileName}`);
      get().addTimelineClip(clipId, videoTrackId, findNonOverlappingStart(get().timelineClips, videoTrackId, startFrame, durationFrames));
      let audioTrackId: string | null = null;
      if (clip.hasAudio) {
        audioTrackId = get().addTrack('audio', `Audio — ${clip.fileName}`);
        get().addTimelineClip(clipId, audioTrackId, findNonOverlappingStart(get().timelineClips, audioTrackId, startFrame, durationFrames));
      }
      return { videoTrackId, audioTrackId };
    },

    // Renames a track. Empty/whitespace-only names are a no-op (the inline
    // rename UI can blur with an empty field); the stored name is trimmed.
    renameTrack: (trackId, name) => {
      const trimmed = name.trim();
      if (!trimmed) return; // empty/whitespace rename is a no-op
      set((state) => (state.tracks.some((t) => t.id === trackId)
        ? { tracks: state.tracks.map((t) => (t.id === trackId ? { ...t, name: trimmed } : t)) }
        : {}));
    },

    addTimelineClip: (clipId: string, trackId: string, startFrame: number) => {
      const state = get();
      const clip = state.clips[clipId];
      const id = genId();
      const timelineClip: TimelineClip = {
        id,
        clipId,
        trackId,
        startFrame,
        inFrame: 0,
        outFrame: clip ? Math.max(0, clip.frameCount - 1) : 0,
      };
      set((s) => ({
        timelineClips: { ...s.timelineClips, [id]: timelineClip },
        tracks: s.tracks.map((t) => (t.id === trackId ? { ...t, clipIds: [...t.clipIds, id] } : t)),
      }));
      return id;
    },

    moveTimelineClip: (timelineClipId: string, trackId: string, startFrame: number) => {
      set((state) => {
        const existing = state.timelineClips[timelineClipId];
        if (!existing) return {};
        const fromTrackId = existing.trackId;
        return {
          timelineClips: {
            ...state.timelineClips,
            [timelineClipId]: { ...existing, trackId, startFrame },
          },
          tracks: state.tracks.map((t) => {
            if (t.id === fromTrackId && t.id === trackId) return t;
            if (t.id === fromTrackId) return { ...t, clipIds: t.clipIds.filter((id) => id !== timelineClipId) };
            if (t.id === trackId && !t.clipIds.includes(timelineClipId)) {
              return { ...t, clipIds: [...t.clipIds, timelineClipId] };
            }
            return t;
          }),
        };
      });
    },

    // Clamps `inFrame < outFrame` within `[0, frameCount)` (studio-contracts.md
    // §6.2) — out-of-range or inverted requests are clamped into the valid
    // range, then (if still zero-length after clamping) nudged apart by one
    // frame rather than stored verbatim.
    trimTimelineClip: (timelineClipId: string, inFrame: number, outFrame: number) => {
      set((state) => {
        const existing = state.timelineClips[timelineClipId];
        if (!existing) return {};
        const clip = state.clips[existing.clipId];
        const maxFrame = clip ? Math.max(0, clip.frameCount - 1) : Math.max(0, inFrame, outFrame);

        let finalIn = Math.min(Math.max(0, inFrame), maxFrame);
        let finalOut = Math.min(Math.max(0, outFrame), maxFrame);
        if (finalIn > finalOut) [finalIn, finalOut] = [finalOut, finalIn];
        if (finalIn === finalOut) {
          if (finalOut < maxFrame) finalOut += 1;
          else if (finalIn > 0) finalIn -= 1;
        }

        return {
          timelineClips: {
            ...state.timelineClips,
            [timelineClipId]: { ...existing, inFrame: finalIn, outFrame: finalOut },
          },
        };
      });
    },

    removeTimelineClip: (timelineClipId: string) => {
      set((state) => {
        const nextTimelineClips = { ...state.timelineClips };
        delete nextTimelineClips[timelineClipId];
        return {
          timelineClips: nextTimelineClips,
          tracks: state.tracks.map((t) => ({ ...t, clipIds: t.clipIds.filter((id) => id !== timelineClipId) })),
          selection:
            state.selection.timelineClipId === timelineClipId
              ? { ...state.selection, timelineClipId: null }
              : state.selection,
        };
      });
    },

    reorderTracks: (trackIds: string[]) => {
      set((state) => {
        const byId = new Map(state.tracks.map((t) => [t.id, t]));
        const reordered = trackIds
          .map((id, order) => {
            const track = byId.get(id);
            return track ? { ...track, order } : null;
          })
          .filter((t): t is Track => t !== null);
        return { tracks: reordered };
      });
    },

    // Deletes a track and every timelineClip on it, then renormalizes the
    // surviving tracks' `order` to 0..n-1 preserving relative order (sort by
    // `order` before renumbering so the result is correct regardless of array
    // position). Source clips (`clips`) and their sessions survive — only the
    // timeline placements go. Clears `selection.timelineClipId` iff the selected
    // clip lived on the removed track (mirrors removeTimelineClip). No-op on
    // unknown id.
    removeTrack: (trackId: string) => {
      set((state) => {
        const track = state.tracks.find((t) => t.id === trackId);
        if (!track) return {};

        const removedIds = new Set(track.clipIds);
        const nextTimelineClips = { ...state.timelineClips };
        for (const id of removedIds) delete nextTimelineClips[id];

        return {
          timelineClips: nextTimelineClips,
          tracks: [...state.tracks]
            .filter((t) => t.id !== trackId)
            .sort((a, b) => a.order - b.order)
            .map((t, order) => ({ ...t, order })),
          selection:
            state.selection.timelineClipId && removedIds.has(state.selection.timelineClipId)
              ? { ...state.selection, timelineClipId: null }
              : state.selection,
        };
      });
    },

    // Splits a placed clip at `atFrame` (a PROJECT/timeline frame — callers pass
    // the raw playhead, never `playhead - startFrame`). Valid iff
    // `startFrame < atFrame < startFrame + (outFrame - inFrame + 1)`; returns the
    // new right-half id, or null (mutating nothing) when invalid/unknown. The
    // right half is legal even when it is a single source frame
    // (`inFrame === outFrame`) — intentional, matching addTimelineClip's existing
    // frameCount===1 placement; do NOT tighten to a 2-frame minimum.
    splitTimelineClip: (timelineClipId: string, atFrame: number) => {
      const state = get();
      const existing = state.timelineClips[timelineClipId];
      if (!existing) return null;
      if (atFrame <= existing.startFrame || atFrame >= existing.startFrame + (existing.outFrame - existing.inFrame + 1)) {
        return null;
      }

      const offset = atFrame - existing.startFrame;
      const newId = genId();
      const left: TimelineClip = { ...existing, outFrame: existing.inFrame + offset - 1 };
      const right: TimelineClip = {
        id: newId,
        clipId: existing.clipId,
        trackId: existing.trackId,
        startFrame: atFrame,
        inFrame: existing.inFrame + offset,
        outFrame: existing.outFrame,
      };

      set((s) => ({
        timelineClips: { ...s.timelineClips, [timelineClipId]: left, [newId]: right },
        tracks: s.tracks.map((t) => {
          if (t.id !== existing.trackId) return t;
          const idx = t.clipIds.indexOf(timelineClipId);
          const clipIds = [...t.clipIds];
          clipIds.splice(idx === -1 ? clipIds.length : idx + 1, 0, newId);
          return { ...t, clipIds };
        }),
      }));
      return newId;
    },

    // Clones a placed clip onto the same track at the first non-overlapping
    // position at/after `source.startFrame + duration` (i.e. immediately after
    // the source, shifted right past any collision). Trim range and source clip
    // are copied verbatim. Returns the clone id, or null on unknown id.
    duplicateTimelineClip: (timelineClipId: string) => {
      const state = get();
      const source = state.timelineClips[timelineClipId];
      if (!source) return null;

      const duration = source.outFrame - source.inFrame + 1;
      const startFrame = findNonOverlappingStart(
        state.timelineClips,
        source.trackId,
        source.startFrame + duration,
        duration,
      );
      const newId = genId();
      const clone: TimelineClip = { ...source, id: newId, startFrame };

      set((s) => ({
        timelineClips: { ...s.timelineClips, [newId]: clone },
        tracks: s.tracks.map((t) => {
          if (t.id !== source.trackId) return t;
          const idx = t.clipIds.indexOf(timelineClipId);
          const clipIds = [...t.clipIds];
          clipIds.splice(idx === -1 ? clipIds.length : idx + 1, 0, newId);
          return { ...t, clipIds };
        }),
      }));
      return newId;
    },

    // ---------------------------------------------------------------------
    // playback / view actions
    // ---------------------------------------------------------------------
    // Clamps to a non-negative frame index (studio-contracts.md §6.2:
    // "setPlayhead clamps to [0, projectDuration)"; the upper bound is a
    // per-clip/timeline concern enforced by callers, not tracked here).
    setPlayhead: (frame: number) => set({ playhead: Math.max(0, frame) }),
    setIsPlaying: (playing: boolean) => set({ isPlaying: playing }),
    setZoom: (pxPerFrame: number) => set({ zoom: Math.max(0.01, pxPerFrame) }),
    setPreviewMuted: (muted) => set({ previewMuted: muted }),
    setPreviewVolume: (volume) => set({ previewVolume: Math.min(1, Math.max(0, volume)) }),

    togglePanel: (key) => set((state) => ({ panels: { ...state.panels, [key]: !state.panels[key] } })),
    setPanelVisible: (key, visible) => set((state) => ({ panels: { ...state.panels, [key]: visible } })),

    setTool: (tool: ToolMode) =>
      set((state) => {
        // Prompting tools are disabled while a track is running (one active
        // propagate() iterator per session): ignore the switch, keep the
        // current tool. Non-prompt tools (select/pan) are always allowed.
        const isPromptTool = tool === 'point-add' || tool === 'point-remove' || tool === 'box';
        if (state.trackState.phase === 'running' && isPromptTool) return {};
        return { tool };
      }),
    // Timeline-clip selection and object selection are mutually exclusive —
    // selecting one clears the other so the Properties panel shows a single subject.
    selectTimelineClip: (id: string | null) =>
      set((state) => ({ selection: { ...state.selection, timelineClipId: id, objectId: id === null ? state.selection.objectId : null } })),
    selectObject: (objectId: number | null) =>
      set((state) => ({ selection: { ...state.selection, objectId, timelineClipId: objectId === null ? state.selection.timelineClipId : null } })),
    setMaskOpacity: (opacity: number) => set({ maskOpacity: Math.min(1, Math.max(0, opacity)) }),

    // ---------------------------------------------------------------------
    // segmenter lifecycle
    // ---------------------------------------------------------------------
    loadModel: async () => {
      const status = get().modelStatus;
      if (status.phase === 'loading' || status.phase === 'ready') return;

      set({ modelStatus: { phase: 'loading' } });
      try {
        const segmenter = await loadSegmenter((event: LoadProgressEvent) => {
          set({
            modelStatus: {
              phase: 'loading',
              progress: { phase: event.phase, loaded: event.loaded, total: event.total, file: event.file },
            },
          });
        });
        set({
          modelStatus: {
            phase: 'ready',
            device: segmenter.device,
            quant: segmenter.model.quant,
            totalBytes: segmenter.model.totalBytes,
          },
          resolvedDevice: segmenter.device,
        });
      } catch (err) {
        const { title, detail } = describeThrown(err);
        const code = typeof err === 'object' && err !== null && 'code' in err ? String((err as { code: unknown }).code) : undefined;
        set({ modelStatus: { phase: 'error', message: detail, code }, notice: { title, detail, kind: 'error' } });
      }
    },

    // ---------------------------------------------------------------------
    // segmentation seam (session-manager.ts / propagate-loop.ts)
    // ---------------------------------------------------------------------
    activateClip: async (clipId: string) => {
      try {
        await activateClipSession(get, set, clipId);
      } catch (err) {
        const { title, detail } = describeThrown(err);
        set({ notice: { title, detail, kind: 'error' } });
      }
    },

    addPromptObject: async (clipId: string, frameIndex: number, prompts: Prompt[]) => {
      try {
        await addPromptObjectSession(get, set, clipId, frameIndex, prompts);
      } catch (err) {
        const { title, detail } = describeThrown(err);
        set({ notice: { title, detail, kind: 'error' } });
      }
    },

    refineObject: async (clipId: string, objectId: number, frameIndex: number, prompts: Prompt[]) => {
      try {
        await refineObjectSession(get, set, clipId, objectId, frameIndex, prompts);
      } catch (err) {
        const { title, detail } = describeThrown(err);
        set({ notice: { title, detail, kind: 'error' } });
      }
    },

    removeObject: (clipId: string, objectId: number) => {
      removeObjectSession(get, set, clipId, objectId);
      set((state) => ({
        objects: state.objects.filter((o) => !(o.clipId === clipId && o.objectId === objectId)),
        selection: state.selection.objectId === objectId ? { ...state.selection, objectId: null } : state.selection,
      }));
    },

    startTracking: async (clipId: string, startFrame?: number) => {
      if (get().trackState.phase === 'running') return;
      try {
        await startTrackingSession(get, set, clipId, startFrame);
      } catch (err) {
        const { title, detail } = describeThrown(err);
        const code = typeof err === 'object' && err !== null && 'code' in err ? String((err as { code: unknown }).code) : undefined;
        set({ trackState: { phase: 'error', message: detail, code }, notice: { title, detail, kind: 'error' } });
      }
    },

    cancelTracking: () => {
      cancelTrackingSession();
      set({ trackState: { phase: 'idle' } });
    },

    // ---------------------------------------------------------------------
    // export
    // ---------------------------------------------------------------------
    exportMatte: async (clipId: string) => {
      try {
        await exportMatteImpl(get, set, clipId);
      } catch (err) {
        const { detail } = describeThrown(err);
        set({ exportState: { phase: 'error', message: detail } });
      }
    },

    exportMp4Cutout: async (clipId: string) => {
      try {
        await exportMp4CutoutImpl(get, set, clipId);
      } catch (err) {
        const { title, detail } = describeThrown(err);
        // Stretch path — never touch trackState; surface a friendly notice
        // instead of blocking on exportState (contracts.md §4.5).
        set({ notice: { title, detail, kind: 'warn' } });
      }
    },

    // ---------------------------------------------------------------------
    // notices
    // ---------------------------------------------------------------------
    setNotice: (notice: StudioState['notice']) => set({ notice }),
    clearNotice: () => set({ notice: null }),
  })),
);
