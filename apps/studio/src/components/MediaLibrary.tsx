/**
 * `MediaLibrary.tsx` — file-drop/pick zone + imported-clip list
 * (studio-contracts.md §3). Takes no props; reads/writes exclusively through
 * `useStudioStore`.
 *
 * Imports a video file into the store via `store.importClip` (metadata probe
 * only — does NOT `attachSource`, see contracts.md §2/§4.2), lists imported
 * clips with a best-effort first-frame thumbnail (falls back to a filename
 * chip if thumbnail capture fails), and registers each clip as a
 * `@dnd-kit/core` `useDraggable` so it can be dropped onto `Timeline`'s
 * tracks via the App-owned `<DndContext>`. Clicking a clip activates it
 * (`store.activateClip`) and selects it in the properties panel via
 * `store.selectTimelineClip` is NOT used here — clip selection in the
 * library only drives `activeClipId` (the live segmentation session); the
 * `selection.timelineClipId` field is a Timeline-track concept, not a
 * library one, so we do not call `selectTimelineClip` from here.
 */
import { useDraggable } from '@dnd-kit/core';
import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';

import { captureFrameBitmap } from '../video/frame-source.js';
import type { ClipMeta } from '../store/studio-store.js';
import { useStudioStore } from '../store/studio-store.js';
import { cn } from '../lib/utils.js';

/** Drag payload shape the shared `<DndContext>` (owned by `App.tsx`) expects
 * for a MediaLibrary → Timeline drag (contracts.md §3, "DnD ownership"). */
export interface MediaLibraryDragData {
  kind: 'media-library-clip';
  clipId: string;
}

function formatDuration(durationSec: number): string {
  if (!Number.isFinite(durationSec) || durationSec <= 0) return '0:00';
  const total = Math.round(durationSec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface ClipCardProps {
  clip: ClipMeta;
  isActive: boolean;
  onActivate: (clipId: string) => void;
}

function ClipCard({ clip, isActive, onActivate }: ClipCardProps): React.JSX.Element {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: `media-library-clip:${clip.id}`,
    data: { kind: 'media-library-clip', clipId: clip.id } satisfies MediaLibraryDragData,
  });
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
  const [thumbnailFailed, setThumbnailFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    const video = document.createElement('video');
    video.muted = true;
    video.src = clip.objectUrl;
    video.preload = 'auto';

    const run = async () => {
      try {
        await new Promise<void>((resolve, reject) => {
          const onLoaded = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            video.removeEventListener('error', onError);
            resolve();
          };
          const onError = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            video.removeEventListener('error', onError);
            reject(new Error('thumbnail: metadata load failed'));
          };
          video.addEventListener('loadedmetadata', onLoaded);
          video.addEventListener('error', onError);
        });
        if (cancelled) return;
        const bitmap = await captureFrameBitmap(video, 0);
        if (cancelled) return;
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('thumbnail: no 2d context');
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        const blob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        if (cancelled || !blob) return;
        objectUrl = URL.createObjectURL(blob);
        setThumbnailUrl(objectUrl);
      } catch {
        if (!cancelled) setThumbnailFailed(true);
      }
    };
    void run();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clip.id, clip.objectUrl]);

  const style = transform
    ? { transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`, opacity: isDragging ? 0.5 : 1 }
    : undefined;

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
      role="button"
      tabIndex={0}
      onClick={() => onActivate(clip.id)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onActivate(clip.id);
      }}
      className={cn(
        'flex cursor-grab items-center gap-2 rounded-md border p-2 text-left transition-colors active:cursor-grabbing',
        isActive ? 'border-primary bg-accent' : 'border-input hover:bg-accent/50',
      )}
    >
      <div className="flex h-10 w-16 shrink-0 items-center justify-center overflow-hidden rounded bg-muted">
        {thumbnailUrl && !thumbnailFailed ? (
          <img src={thumbnailUrl} alt="" className="h-full w-full object-cover" draggable={false} />
        ) : (
          <span className="px-1 text-center text-[10px] leading-tight text-muted-foreground">
            {clip.fileName}
          </span>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-xs font-medium">{clip.fileName}</div>
        <div className="text-[10px] text-muted-foreground">
          {formatDuration(clip.durationSec)} · {clip.width}×{clip.height}
        </div>
      </div>
    </div>
  );
}

/**
 * File-drop/pick zone + imported-clip list. See file header for the full
 * contract summary.
 */
export function MediaLibrary(): React.JSX.Element {
  const clips = useStudioStore((s) => s.clips);
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const importClip = useStudioStore((s) => s.importClip);
  const activateClip = useStudioStore((s) => s.activateClip);

  const [isDragOver, setIsDragOver] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const importFiles = async (files: FileList | File[]) => {
    const videoFiles = Array.from(files).filter((f) => f.type.startsWith('video/') || /\.(mp4|mov|webm|m4v)$/i.test(f.name));
    if (videoFiles.length === 0) return;
    setIsImporting(true);
    try {
      for (const file of videoFiles) {
        // eslint-disable-next-line no-await-in-loop
        await importClip(file);
      }
    } finally {
      setIsImporting(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer?.files?.length) void importFiles(e.dataTransfer.files);
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void importFiles(e.target.files);
    e.target.value = '';
  };

  const clipList = Object.values(clips);

  return (
    <div className="flex h-full flex-col gap-2 overflow-hidden p-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Media</h2>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
        }}
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed p-4 text-center text-xs transition-colors',
          isDragOver ? 'border-primary bg-accent' : 'border-input text-muted-foreground hover:bg-accent/50',
        )}
      >
        {isImporting ? 'Importing…' : 'Drop video files here or click to browse'}
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,.mp4,.mov,.webm,.m4v"
          multiple
          className="hidden"
          onChange={onFileInputChange}
        />
      </div>

      <div className="flex flex-1 flex-col gap-1 overflow-y-auto">
        {clipList.length === 0 ? (
          <p className="p-2 text-center text-xs text-muted-foreground">No clips imported yet.</p>
        ) : (
          clipList.map((clip) => (
            <ClipCard key={clip.id} clip={clip} isActive={clip.id === activeClipId} onActivate={(id) => void activateClip(id)} />
          ))
        )}
      </div>
    </div>
  );
}

export default MediaLibrary;
