/**
 * `PreviewCanvas.tsx` — the studio's main viewport (studio-contracts.md §3).
 *
 * Renders an offscreen `<video>` bound to `activeClipId`'s `objectUrl`,
 * overlaid with a `react-konva` `<Stage>`/`<Layer>` that:
 *  - draws the current video frame directly (Konva accepts an
 *    `HTMLVideoElement` as an image source — no intermediate canvas);
 *  - draws each tracked object's live mask (`store.liveMasksAtFrame`) as a
 *    semi-transparent colored overlay, only for the frame it actually
 *    applies to (either the frame it was last prompted/refined at, or the
 *    live frame of an in-flight `Track` on this clip) — `liveMasksAtFrame`
 *    itself carries no frame index, so this component is the one place that
 *    reconciles "which frame is this mask valid for";
 *  - draws point/box prompts for the selected object at the current frame.
 *
 * A `requestVideoFrameCallback` loop (rAF fallback) drives `store.playhead`
 * from the video's own clock while playing; conversely, scrubbing the
 * playhead elsewhere (e.g. `Timeline`) seeks this component's `<video>`.
 *
 * Pointer handlers translate stage-local (= displayed-size) coordinates into
 * SOURCE-pixel coordinates and dispatch per `store.tool`:
 *  - `point-add` / `point-remove`: positive/negative point prompts, refining
 *    the selected object on this clip if one exists, else starting a new one
 *    (`shift`-click on `point-add` is accepted as a convenience alias for a
 *    negative point, mirroring `apps/demo/src/VideoTab.tsx`'s UX).
 *  - `box`: drag-to-draw a rectangle, emitted as a single `{type:'box',...}`
 *    prompt on pointer-up.
 *  - `select` / `pan`: no segmentation call.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, RefObject } from 'react';
import Konva from 'konva';
import { Pause, Play, Volume2, VolumeX } from 'lucide-react';
import { Circle, Image as KonvaImage, Layer, Line, Rect, Stage } from 'react-konva';
import type { MaskResult, Prompt } from '@websam3/core';

import { useStudioStore } from '../store/studio-store.js';
import type { ToolMode, TrackedObject } from '../store/studio-store.js';
import { cn } from '../lib/utils.js';
import { timeToFrameIndex } from '../video/frame-source.js';

/** Refs `App.tsx` owns and hands down so the segmentation module and this
 * canvas can share exactly one `<video>` / one layout container without a
 * store round-trip (studio-contracts.md §3). */
export interface PreviewCanvasProps {
  videoRef: RefObject<HTMLVideoElement>;
  stageContainerRef: RefObject<HTMLDivElement>;
}

/** One accumulated click for the object currently being prompted at a frame. */
interface ClickPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/** Recolor a mask's opaque-white `ImageData` into `color` at `opacity` (0..1, from `store.maskOpacity`), then decode to a bitmap. */
async function maskToColoredBitmap(mask: MaskResult, color: string, opacity: number): Promise<ImageBitmap> {
  const imageData = mask.toImageData();
  const [r, g, b] = hexToRgb(color);
  const alpha = Math.round(Math.min(1, Math.max(0, opacity)) * 255);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] !== 0) {
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = alpha;
    }
  }
  return createImageBitmap(imageData);
}

/** Fit `srcW`x`srcH` into `boxW`x`boxH`, preserving aspect ratio (letterbox). */
function fitContain(srcW: number, srcH: number, boxW: number, boxH: number): { width: number; height: number } {
  if (srcW <= 0 || srcH <= 0 || boxW <= 0 || boxH <= 0) return { width: 0, height: 0 };
  const scale = Math.min(boxW / srcW, boxH / srcH);
  return { width: srcW * scale, height: srcH * scale };
}

/** `frame` at `fps` → `m:ss` (caption for the on-canvas playback overlay). */
function formatTime(frame: number, fps: number): string {
  const totalSec = frame / (fps > 0 ? fps : 30);
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

interface OverlaySliderProps {
  ariaLabel: string;
  value: number;
  min: number;
  max: number;
  /** Keyboard/pointer quantum (arrow step; pointer positions snap to it). */
  step: number;
  /** Larger jump for Shift+Arrow (defaults to `step`). */
  bigStep?: number;
  /** Human-readable announcement of `value` for assistive tech (`aria-valuetext`). */
  ariaValueText?: string;
  /** Dims the fill (muted output) without changing the announced value. */
  dimmed?: boolean;
  className?: string;
  onChange: (value: number) => void;
  /** Fired once at the start of any interaction (pointer OR keyboard) before the
   * first `onChange` (e.g. pause-on-scrub, so a seek is not immediately
   * overwritten by the still-running playback loop). */
  onScrubStart?: () => void;
}

/**
 * Minimal-chrome range control for the playback overlay. A bespoke variant of
 * `Timeline.tsx`'s `TrimHandle` precedent (`role="slider"` + full ARIA +
 * keyboard) rather than `ui/slider.tsx`, whose `bg-primary` (near-black) Range
 * is invisible over dark footage and whose Track/Range classNames are locked
 * inside the primitive — see `DESIGN.md` §2/§9. White-alpha on `neutral-950`,
 * hue-free, matching the `#ffffff` Konva prompt/drag-box strokes.
 */
function OverlaySlider({
  ariaLabel,
  value,
  min,
  max,
  step,
  bigStep,
  ariaValueText,
  dimmed,
  className,
  onChange,
  onScrubStart,
}: OverlaySliderProps): React.JSX.Element {
  const trackRef = useRef<HTMLDivElement>(null);
  // True only between this slider's own pointer-down and its release, so a
  // press-drag begun on a sibling control (button) that slides across this
  // track cannot fire onChange mid-click (buttons===1 alone can't tell whose
  // drag it is).
  const isDraggingRef = useRef(false);
  const span = Math.max(1, max - min);
  const pct = Math.min(100, Math.max(0, ((value - min) / span) * 100));

  const valueFromClientX = (clientX: number): number => {
    const el = trackRef.current;
    if (!el) return value;
    const rect = el.getBoundingClientRect();
    const frac = rect.width > 0 ? Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) : 0;
    const raw = min + frac * (max - min);
    return Math.min(max, Math.max(min, Math.round(raw / step) * step));
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    onScrubStart?.();
    isDraggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    onChange(valueFromClientX(e.clientX));
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current || e.buttons !== 1) return;
    onChange(valueFromClientX(e.clientX));
  };
  const endDrag = () => {
    isDraggingRef.current = false;
  };
  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const large = bigStep ?? step;
    let next: number | null = null;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = value - (e.shiftKey ? large : step);
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = value + (e.shiftKey ? large : step);
    else if (e.key === 'Home') next = min;
    else if (e.key === 'End') next = max;
    if (next === null) return;
    e.preventDefault();
    // Match pointer order: signal interaction start (e.g. pause-on-scrub) BEFORE
    // committing the value, so keyboard seek during playback isn't a silent
    // no-op overwritten by the next video frame.
    onScrubStart?.();
    onChange(Math.min(max, Math.max(min, next)));
  };

  return (
    <div
      ref={trackRef}
      role="slider"
      aria-label={ariaLabel}
      aria-orientation="horizontal"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={ariaValueText}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onLostPointerCapture={endDrag}
      onKeyDown={onKeyDown}
      className={cn(
        'relative flex h-6 cursor-pointer items-center rounded-md focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/80 focus-visible:ring-offset-1 focus-visible:ring-offset-neutral-950',
        className,
      )}
    >
      <div className={cn('h-1 w-full overflow-hidden rounded-full bg-white/25', dimmed && 'opacity-40')}>
        <div className="h-full bg-white/90" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/**
 * The studio's video + mask + prompt viewport. See the module doc comment
 * and `studio-contracts.md` §3 for the full contract.
 */
export function PreviewCanvas({ videoRef, stageContainerRef }: PreviewCanvasProps): React.JSX.Element {
  const clip = useStudioStore((s) => (s.activeClipId ? (s.clips[s.activeClipId] ?? null) : null));
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const tool = useStudioStore((s) => s.tool);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);
  const playhead = useStudioStore((s) => s.playhead);
  const setPlayhead = useStudioStore((s) => s.setPlayhead);
  const objects = useStudioStore((s) => s.objects);
  const liveMasksAtFrame = useStudioStore((s) => s.liveMasksAtFrame);
  const maskOpacity = useStudioStore((s) => s.maskOpacity);
  const selection = useStudioStore((s) => s.selection);
  const trackState = useStudioStore((s) => s.trackState);
  const addPromptObject = useStudioStore((s) => s.addPromptObject);
  const refineObject = useStudioStore((s) => s.refineObject);

  const layerRef = useRef<Konva.Layer | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [videoReady, setVideoReady] = useState(false);
  const [, forceRedraw] = useState(0);
  // True once the current clip activation has forced at least one genuine
  // (non-no-op) seek — see the seek effect below for why this is needed.
  const forcedInitialSeekRef = useRef(false);

  // Presentational, DOM-ref-adjacent state driving the offscreen <video>
  // directly — NOT shared app state, so it stays local per the store contract's
  // transient-state carve-out (studio-contracts.md §3). `muted` starts true to
  // preserve today's silent/autoplay-safe behavior (the JSX `muted` attribute
  // moved here so unmuting is possible).
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(true);
  // Level to restore when unmuting via the speaker button (so it returns to the
  // pre-mute volume rather than jumping to full).
  const lastVolumeRef = useRef(1);

  // Prompt tools own click-to-add on the Stage; the overlay bar sits above the
  // Stage and its full-width Seek slider would otherwise intercept a prompt
  // click landing over the bottom strip. Suppress the bar (no pointer events,
  // hidden) whenever a prompt tool is armed; select/pan keep it (they already
  // own click-to-play, so competing hover reveal is coherent there).
  const promptToolActive = tool === 'point-add' || tool === 'point-remove' || tool === 'box';

  // Distinguishes a click (toggle play) from a drag on the select/pan tools —
  // records the pointer-down position so pointer-up can measure travel.
  const clickStartRef = useRef<{ x: number; y: number } | null>(null);

  // Accumulated click points per objectId at its current prompt frame —
  // mirrors the demo's `objectsRef`/`points` pattern (session.refineObject
  // wants the FULL prompt set for a frame each call, not an incremental
  // diff — see @websam3/core's VideoEngine#interact). Transient interaction
  // state only; never the source of truth for a mask (that's the store).
  const pointsByObjectRef = useRef<Map<number, { frameIndex: number; points: ClickPoint[] }>>(new Map());
  const bitmapsRef = useRef<Map<number, { bitmap: ImageBitmap; maskRef: MaskResult; opacity: number }>>(new Map());
  const [bitmapVersion, setBitmapVersion] = useState(0);

  const [dragBox, setDragBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const activeClipObjects = useMemo(
    () => objects.filter((o) => o.clipId === activeClipId),
    [objects, activeClipId],
  );
  const selectedObject: TrackedObject | undefined = useMemo(
    () =>
      selection.objectId !== null
        ? activeClipObjects.find((o) => o.objectId === selection.objectId)
        : undefined,
    [activeClipObjects, selection.objectId],
  );

  // -------------------------------------------------------------------
  // Size the stage to the video's displayed rect within the container.
  // -------------------------------------------------------------------
  useEffect(() => {
    const container = stageContainerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setContainerSize({ width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [stageContainerRef]);

  const displaySize = useMemo(
    () => (clip ? fitContain(clip.width, clip.height, containerSize.width, containerSize.height) : { width: 0, height: 0 }),
    [clip, containerSize],
  );

  // -------------------------------------------------------------------
  // Bind the offscreen <video> to the active clip.
  // -------------------------------------------------------------------
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    setVideoReady(false);
    forcedInitialSeekRef.current = false;
    if (clip) {
      if (video.src !== clip.objectUrl) video.src = clip.objectUrl;
    } else {
      video.removeAttribute('src');
      video.load();
    }
  }, [clip, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onLoaded = () => setVideoReady(true);
    // When playback reaches the clip's end the video simply stops and the rvfc
    // loop stops ticking; without this, store.isPlaying stays true and both the
    // overlay and Toolbar keep showing a Pause icon over a frozen frame.
    const onEnded = () => setIsPlaying(false);
    video.addEventListener('loadeddata', onLoaded);
    video.addEventListener('ended', onEnded);
    return () => {
      video.removeEventListener('loadeddata', onLoaded);
      video.removeEventListener('ended', onEnded);
    };
  }, [videoRef, setIsPlaying]);


  // Sync play/pause with `store.isPlaying`.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) void video.play().catch(() => undefined);
    else video.pause();
  }, [isPlaying, videoRef]);

  // Drive the offscreen <video>'s volume/mute from local state (replaces the
  // former hard `muted` JSX attribute).
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.volume = volume;
    video.muted = muted;
  }, [volume, muted, videoRef]);

  // -------------------------------------------------------------------
  // requestVideoFrameCallback (rAF fallback) loop: video clock -> playhead,
  // and forces a layer redraw every frame (Konva does not detect that an
  // already-bound HTMLVideoElement's pixels changed on its own).
  // -------------------------------------------------------------------
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clip || !isPlaying) return;

    let rvfcHandle: number | null = null;
    let rafHandle: number | null = null;
    const fps = clip.fps > 0 ? clip.fps : 30;

    const onTick = (mediaTime: number) => {
      setPlayhead(timeToFrameIndex(mediaTime, fps));
      layerRef.current?.batchDraw();
    };

    const supportsRVFC = typeof video.requestVideoFrameCallback === 'function';
    if (supportsRVFC) {
      const step = (_now: number, metadata: { mediaTime: number }) => {
        onTick(metadata.mediaTime);
        rvfcHandle = video.requestVideoFrameCallback(step);
      };
      rvfcHandle = video.requestVideoFrameCallback(step);
    } else {
      const step = () => {
        onTick(video.currentTime);
        rafHandle = requestAnimationFrame(step);
      };
      rafHandle = requestAnimationFrame(step);
    }

    return () => {
      if (rvfcHandle !== null && typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(rvfcHandle);
      }
      if (rafHandle !== null) cancelAnimationFrame(rafHandle);
    };
  }, [clip, isPlaying, setPlayhead, videoRef]);

  // While NOT playing (scrub from Timeline, or a track-driven playhead
  // update), seek the shared <video> to match `playhead` and redraw. Waits
  // for `videoReady` so the very first seek of a freshly-activated clip is
  // included (not skipped as a same-render race before load completes).
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clip || isPlaying || !videoReady) return;
    const fps = clip.fps > 0 ? clip.fps : 30;
    const targetTime = playhead / fps;
    const needsSeek = Math.abs(video.currentTime - targetTime) > 1e-3;
    // A hidden (`display: none`) <video> used purely as a canvas image
    // source does not reliably decode/present its first frame in Chromium
    // just because `readyState`/`loadeddata` already report data as
    // available — the compositor only actually produces presentable pixels
    // once a real seek (or play) is issued. So the FIRST time a newly-
    // activated clip becomes ready, force a genuine seek even when
    // `targetTime` numerically already matches `video.currentTime` (a
    // same-value reassignment is a no-op in most engines and would never
    // trigger this). Every subsequent seek behaves as before (skip the
    // no-op case) once that initial kick has happened.
    const forceKick = !forcedInitialSeekRef.current;
    if (needsSeek || forceKick) {
      const base = Math.max(0, Math.min(targetTime, video.duration || targetTime));
      video.currentTime = forceKick && !needsSeek ? Math.min(video.duration || base, base + 0.0001) : base;
      forcedInitialSeekRef.current = true;
    }
    const onSeeked = () => layerRef.current?.batchDraw();
    video.addEventListener('seeked', onSeeked, { once: true });
    // Also redraw immediately in case currentTime was already correct (no
    // 'seeked' event fires) — e.g. re-selecting the same frame.
    layerRef.current?.batchDraw();
    return () => video.removeEventListener('seeked', onSeeked);
  }, [playhead, isPlaying, clip, videoRef, videoReady]);

  // -------------------------------------------------------------------
  // Mask overlay bitmaps: only for the frame each live mask is actually
  // valid at (its object's `promptFrame`, or the in-flight track's current
  // frame on this clip), cached per `${objectId}:${frameIndex}`.
  // -------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const trackingThisClip = trackState.phase === 'running' && trackState.clipId === activeClipId;

    const visible = activeClipObjects
      .map((obj) => {
        const mask = liveMasksAtFrame[obj.objectId];
        if (!mask) return null;
        const validAtFrame = trackingThisClip ? trackState.frameIndex : obj.promptFrame;
        if (validAtFrame !== playhead) return null;
        return { obj, mask };
      })
      .filter((v): v is { obj: TrackedObject; mask: MaskResult } => v !== null);

    const visibleIds = new Set(visible.map((v) => v.obj.objectId));
    for (const [objectId, entry] of bitmapsRef.current) {
      if (!visibleIds.has(objectId)) {
        entry.bitmap.close();
        bitmapsRef.current.delete(objectId);
      }
    }

    void Promise.all(
      visible.map(async ({ obj, mask }) => {
        const cached = bitmapsRef.current.get(obj.objectId);
        if (cached && cached.maskRef === mask && cached.opacity === maskOpacity) return;
        const bitmap = await maskToColoredBitmap(mask, obj.color, maskOpacity);
        if (cancelled) {
          bitmap.close();
          return;
        }
        cached?.bitmap.close();
        bitmapsRef.current.set(obj.objectId, { bitmap, maskRef: mask, opacity: maskOpacity });
      }),
    ).then(() => {
      if (!cancelled) setBitmapVersion((v) => v + 1);
    });

    return () => {
      cancelled = true;
    };
  }, [activeClipObjects, liveMasksAtFrame, playhead, trackState, activeClipId, maskOpacity]);

  useEffect(
    () => () => {
      for (const entry of bitmapsRef.current.values()) entry.bitmap.close();
      bitmapsRef.current.clear();
    },
    [],
  );

  // -------------------------------------------------------------------
  // Pointer handling: stage-local (displayed) coords -> source-pixel coords.
  // -------------------------------------------------------------------
  const stageToSource = useCallback(
    (stageX: number, stageY: number): { x: number; y: number } | null => {
      if (!clip || displaySize.width <= 0 || displaySize.height <= 0) return null;
      const scaleX = clip.width / displaySize.width;
      const scaleY = clip.height / displaySize.height;
      return {
        x: Math.min(clip.width - 1, Math.max(0, Math.round(stageX * scaleX))),
        y: Math.min(clip.height - 1, Math.max(0, Math.round(stageY * scaleY))),
      };
    },
    [clip, displaySize],
  );

  const dispatchPoint = useCallback(
    async (source: { x: number; y: number }, label: 0 | 1) => {
      if (!activeClipId) return;
      const point: ClickPoint = { x: source.x, y: source.y, label };
      const existing = selectedObject;
      if (!existing) {
        if (label === 0) return; // a negative point cannot seed a brand-new object
        pointsByObjectRef.current.clear();
        await addPromptObject(activeClipId, playhead, [{ type: 'point', x: point.x, y: point.y, label: point.label }]);
        return;
      }
      const acc = pointsByObjectRef.current.get(existing.objectId);
      const points = acc && acc.frameIndex === playhead ? [...acc.points, point] : [point];
      pointsByObjectRef.current.set(existing.objectId, { frameIndex: playhead, points });
      const prompts: Prompt[] = points.map((p) => ({ type: 'point', x: p.x, y: p.y, label: p.label }));
      await refineObject(activeClipId, existing.objectId, playhead, prompts);
    },
    [activeClipId, addPromptObject, playhead, refineObject, selectedObject],
  );

  const dispatchBox = useCallback(
    async (a: { x: number; y: number }, b: { x: number; y: number }) => {
      if (!activeClipId) return;
      const prompt: Prompt = {
        type: 'box',
        x1: Math.min(a.x, b.x),
        y1: Math.min(a.y, b.y),
        x2: Math.max(a.x, b.x),
        y2: Math.max(a.y, b.y),
      };
      const existing = selectedObject;
      if (existing) {
        pointsByObjectRef.current.delete(existing.objectId);
        await refineObject(activeClipId, existing.objectId, playhead, [prompt]);
      } else {
        await addPromptObject(activeClipId, playhead, [prompt]);
      }
    },
    [activeClipId, addPromptObject, playhead, refineObject, selectedObject],
  );

  const labelForPointTool = useCallback(
    (tool: ToolMode, shiftKey: boolean): 0 | 1 => {
      if (tool === 'point-remove') return 0;
      return shiftKey ? 0 : 1;
    },
    [],
  );

  const handlePointerDown = useCallback(
    (evt: Konva.KonvaEventObject<PointerEvent>) => {
      if (!activeClipId || trackState.phase === 'running') return;
      // The tools that dispatch no segmentation prompt (`select`/`pan`) instead
      // arm a click-to-play gesture; the actual toggle happens on pointer-up if
      // the pointer barely moved. Prompt tools never reach this branch, so their
      // clicks are never swallowed.
      if (tool === 'select' || tool === 'pan') {
        clickStartRef.current = { x: evt.evt.clientX, y: evt.evt.clientY };
        return;
      }
      if (tool !== 'point-add' && tool !== 'point-remove' && tool !== 'box') return;
      const stage = evt.target.getStage();
      const pos = stage?.getPointerPosition();
      if (!pos) return;
      const source = stageToSource(pos.x, pos.y);
      if (!source) return;

      if (tool === 'box') {
        // Capture the pointer on the stage's DOM element so a release
        // outside the Stage still delivers `pointerup` here (mirrors
        // Timeline.tsx's `TrimHandle` pattern) instead of abandoning the
        // in-progress drag.
        stage?.content?.setPointerCapture(evt.evt.pointerId);
        setDragBox({ x1: source.x, y1: source.y, x2: source.x, y2: source.y });
        return;
      }

      const label = labelForPointTool(tool, evt.evt.shiftKey);
      void dispatchPoint(source, label);
    },
    [activeClipId, dispatchPoint, labelForPointTool, stageToSource, tool, trackState.phase],
  );

  const handlePointerMove = useCallback(
    (evt: Konva.KonvaEventObject<PointerEvent>) => {
      if (!dragBox) return;
      const stage = evt.target.getStage();
      const pos = stage?.getPointerPosition();
      if (!pos) return;
      const source = stageToSource(pos.x, pos.y);
      if (!source) return;
      setDragBox((prev) => (prev ? { ...prev, x2: source.x, y2: source.y } : prev));
    },
    [dragBox, stageToSource],
  );

  const handlePointerUp = useCallback(
    (evt?: Konva.KonvaEventObject<PointerEvent>) => {
      // Click-to-play (select/pan): toggle only if the pointer barely traveled,
      // so a pan-drag is not misread as a click.
      if (clickStartRef.current) {
        const start = clickStartRef.current;
        clickStartRef.current = null;
        const end = evt?.evt;
        const moved = end ? Math.hypot(end.clientX - start.x, end.clientY - start.y) : 0;
        if (moved < 4) setIsPlaying(!isPlaying);
        return;
      }
      if (!dragBox) return;
      const box = dragBox;
      setDragBox(null);
      if (box.x1 === box.x2 && box.y1 === box.y2) return; // no drag distance — ignore
      void dispatchBox({ x: box.x1, y: box.y1 }, { x: box.x2, y: box.y2 });
    },
    [dragBox, dispatchBox, isPlaying, setIsPlaying],
  );

  // -------------------------------------------------------------------
  // Prompt point markers for the selected object at the current frame.
  // -------------------------------------------------------------------
  const currentFramePoints = useMemo(() => {
    if (!selectedObject || !clip || displaySize.width <= 0) return [];
    const acc = pointsByObjectRef.current.get(selectedObject.objectId);
    if (!acc || acc.frameIndex !== playhead) return [];
    const scaleX = displaySize.width / clip.width;
    const scaleY = displaySize.height / clip.height;
    return acc.points.map((p) => ({ ...p, dx: p.x * scaleX, dy: p.y * scaleY }));
  }, [selectedObject, clip, displaySize, playhead, bitmapVersion]);

  const dragBoxDisplay = useMemo(() => {
    if (!dragBox || !clip || displaySize.width <= 0) return null;
    const scaleX = displaySize.width / clip.width;
    const scaleY = displaySize.height / clip.height;
    return {
      x: Math.min(dragBox.x1, dragBox.x2) * scaleX,
      y: Math.min(dragBox.y1, dragBox.y2) * scaleY,
      width: Math.abs(dragBox.x2 - dragBox.x1) * scaleX,
      height: Math.abs(dragBox.y2 - dragBox.y1) * scaleY,
    };
  }, [dragBox, clip, displaySize]);

  // Bump a render every time bitmapVersion changes so the JSX below re-reads
  // `bitmapsRef.current` (imperative cache, not React state, to avoid
  // storing ImageBitmap objects in the store or re-decoding on every tick).
  useEffect(() => {
    forceRedraw((v) => v + 1);
  }, [bitmapVersion]);

  return (
    <div
      ref={stageContainerRef}
      className="group relative flex h-full w-full min-h-0 min-w-0 items-center justify-center overflow-hidden bg-neutral-950"
    >
      {/* Offscreen — the video element is only ever used as a Konva image
          source and as the shared playback clock; it is never displayed
          directly (the Konva layer below is the visible frame). Volume/mute are
          set imperatively (see the volume effect), not via a JSX attribute. */}
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <video ref={videoRef} playsInline style={{ display: 'none' }} />

      {!clip ? (
        <p className="text-sm text-neutral-500">Import and select a clip to start segmenting.</p>
      ) : displaySize.width > 0 && displaySize.height > 0 ? (
        <>
        <Stage
          width={displaySize.width}
          height={displaySize.height}
          style={{ cursor: tool === 'select' || tool === 'pan' ? 'default' : 'crosshair' }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <Layer ref={layerRef}>
            {videoReady && videoRef.current ? (
              <KonvaImage image={videoRef.current} width={displaySize.width} height={displaySize.height} listening={false} />
            ) : null}

            {activeClipObjects.map((obj) => {
              const entry = bitmapsRef.current.get(obj.objectId);
              if (!entry) return null;
              return (
                <KonvaImage
                  key={obj.objectId}
                  image={entry.bitmap}
                  width={displaySize.width}
                  height={displaySize.height}
                  listening={false}
                />
              );
            })}

            {currentFramePoints.map((p, i) => (
              <Circle
                key={i}
                x={p.dx}
                y={p.dy}
                radius={6}
                fill={p.label === 1 ? (selectedObject?.color ?? '#3d85ff') : '#c92a2a'}
                stroke="#ffffff"
                strokeWidth={1.5}
                listening={false}
              />
            ))}

            {dragBoxDisplay ? (
              <Rect
                x={dragBoxDisplay.x}
                y={dragBoxDisplay.y}
                width={dragBoxDisplay.width}
                height={dragBoxDisplay.height}
                stroke="#ffffff"
                dash={[6, 4]}
                strokeWidth={1.5}
                listening={false}
              />
            ) : null}

            {/* Zero-length Line keeps Konva's node-index stable across
                overlay count changes without an extra empty-state branch. */}
            {activeClipObjects.length === 0 && !dragBoxDisplay ? <Line points={[]} listening={false} /> : null}
          </Layer>
        </Stage>

        {/* Playback overlay — a DOM sibling ABOVE the Stage (never inside the
            Konva tree), docked over the bottom letterbox strip. Revealed on
            hover/focus only; opacity-only transition within the §6 200ms budget.
            White-alpha-on-neutral-950 chrome (no light-surface tokens, which
            don't read over video — DESIGN.md §2). `pointer-events-none` while
            hidden (and always while a prompt tool is armed) so the bar never
            eats point/box prompt clicks landing over the bottom strip. */}
        <div
          className={cn(
            'absolute inset-x-0 bottom-0 z-10 flex items-center gap-2 bg-neutral-950/75 px-3 py-2 opacity-0 transition-opacity',
            promptToolActive
              ? 'pointer-events-none'
              : 'pointer-events-none group-hover:pointer-events-auto group-focus-within:pointer-events-auto group-hover:opacity-100 group-focus-within:opacity-100',
          )}
        >
          <button
            type="button"
            aria-label={isPlaying ? 'Pause' : 'Play'}
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex h-7 w-7 items-center justify-center rounded-md text-white/80 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/80 focus-visible:ring-offset-1 focus-visible:ring-offset-neutral-950"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>

          <span className="min-w-[70px] text-[11px] tabular-nums text-white/70">
            {formatTime(playhead, clip.fps)} · {playhead}f
          </span>

          <OverlaySlider
            ariaLabel="Seek"
            className="flex-1"
            value={playhead}
            min={0}
            max={Math.max(0, clip.frameCount - 1)}
            step={1}
            bigStep={10}
            ariaValueText={`${formatTime(playhead, clip.fps)} of ${formatTime(Math.max(0, clip.frameCount - 1), clip.fps)}`}
            onScrubStart={() => setIsPlaying(false)}
            onChange={(v) => setPlayhead(v)}
          />

          <button
            type="button"
            aria-label={muted ? 'Unmute' : 'Mute'}
            onClick={() =>
              setMuted((m) => {
                const next = !m;
                if (!next && volume === 0) {
                  // Unmuting from a zero level: restore the pre-mute volume.
                  setVolume(lastVolumeRef.current > 0 ? lastVolumeRef.current : 1);
                }
                return next;
              })
            }
            className="flex h-7 w-7 items-center justify-center rounded-md text-white/80 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/80 focus-visible:ring-offset-1 focus-visible:ring-offset-neutral-950"
          >
            {muted || volume === 0 ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>

          <OverlaySlider
            ariaLabel="Volume"
            className="w-16"
            value={muted ? 0 : Math.round(volume * 100)}
            min={0}
            max={100}
            step={5}
            dimmed={muted}
            ariaValueText={muted ? 'Muted' : `${Math.round(volume * 100)}%`}
            onChange={(v) => {
              const next = v / 100;
              setVolume(next);
              if (next > 0) lastVolumeRef.current = next;
              setMuted(next === 0);
            }}
          />
        </div>
        </>
      ) : null}
    </div>
  );
}
