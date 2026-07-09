import type { FramePropagationResult, RLEMask } from '@websam3/core';

/**
 * Construction parameters for a {@link MaskTimeline}.
 */
export interface MaskTimelineInit {
  /** Total number of frames in the video. Valid frame indices are `[0, frameCount)`. */
  frameCount: number;
  /** Frames per second of the source video. */
  fps: number;
  /** Pixel width of every mask on this timeline. */
  width: number;
  /** Pixel height of every mask on this timeline. */
  height: number;
}

/**
 * Options for {@link MaskTimeline.collect}.
 */
export interface CollectOptions {
  /**
   * Called after each frame's masks have been stored. Lets a UI render live
   * from the same single iterator consumer (the propagation iterator contract
   * allows only one consumer).
   */
  onFrame?: (frame: FramePropagationResult) => void;
  /**
   * Epoch stamped into every {@link MaskTimeline.set}. Pairs with
   * {@link MaskTimeline.invalidateAfter} for the refine flow: a re-collect
   * under a newer epoch supersedes stale masks, and a straggler carrying an
   * older epoch is rejected.
   */
  epoch?: number;
}

/**
 * Optional frame range for {@link MaskTimeline.holes}.
 */
export interface FrameRange {
  /** First frame to inspect (inclusive). Defaults to `0`. */
  start?: number;
  /** Frame to stop before (exclusive). Defaults to `frameCount`. */
  end?: number;
}

/**
 * An {@link RLEMask} whose `counts` array has been encoded to a base64
 * string for compact JSON transport.
 */
export type SerializedRLEMask = Omit<RLEMask, 'counts'> & {
  /** Base64-encoded little-endian uint32 run-length counts. */
  counts: string;
};

/**
 * JSON shape produced by {@link MaskTimeline.toJSON} and consumed by
 * {@link MaskTimeline.fromJSON}.
 */
export interface MaskTimelineJSON {
  frameCount: number;
  fps: number;
  width: number;
  height: number;
  /** Per-object tracks keyed by object id. */
  objects: Record<
    string,
    {
      /** The object's current epoch (see {@link MaskTimeline.invalidateAfter}). */
      epoch: number;
      /** Masks keyed by frame index (stringified, as JSON requires). */
      frames: Record<string, SerializedRLEMask>;
    }
  >;
}

interface ObjectTrack {
  epoch: number;
  /** Kept sorted by ascending frame index at all times. */
  frames: Map<number, RLEMask>;
}

/**
 * Sparse per-object mask storage for a video, indexed by frame.
 *
 * Each tracked object owns a sorted `Map<frameIndex, RLEMask>` plus an
 * *epoch* counter that pairs with the `refineObject` contract in
 * `@websam3/core`: when a user refines an object at some frame, downstream
 * propagated masks become stale — {@link invalidateAfter} drops them and
 * bumps the epoch, and any late {@link set} carrying the old epoch is
 * silently rejected instead of resurrecting stale masks.
 */
export class MaskTimeline {
  /** Total number of frames in the video. */
  readonly frameCount: number;
  /** Frames per second of the source video. */
  readonly fps: number;
  /** Pixel width of every mask on this timeline. */
  readonly width: number;
  /** Pixel height of every mask on this timeline. */
  readonly height: number;

  private readonly objects = new Map<string, ObjectTrack>();

  constructor(init: MaskTimelineInit) {
    if (!Number.isInteger(init.frameCount) || init.frameCount <= 0) {
      throw new RangeError(`frameCount must be a positive integer, got ${init.frameCount}`);
    }
    if (!Number.isFinite(init.fps) || init.fps <= 0) {
      throw new RangeError(`fps must be a positive number, got ${init.fps}`);
    }
    if (!Number.isInteger(init.width) || init.width <= 0) {
      throw new RangeError(`width must be a positive integer, got ${init.width}`);
    }
    if (!Number.isInteger(init.height) || init.height <= 0) {
      throw new RangeError(`height must be a positive integer, got ${init.height}`);
    }
    this.frameCount = init.frameCount;
    this.fps = init.fps;
    this.width = init.width;
    this.height = init.height;
  }

  /**
   * Store a mask for `objectId` at `frameIndex`.
   *
   * @param epoch - The epoch the mask was produced under. Defaults to the
   * object's current epoch. A write carrying an epoch *older* than the
   * object's current epoch is rejected (returns `false`) — this is how
   * in-flight propagation results are discarded after
   * {@link invalidateAfter}. A *newer* epoch advances the object's current
   * epoch.
   * @returns `true` if the mask was stored, `false` if it was rejected as stale.
   */
  set(objectId: string, frameIndex: number, rle: RLEMask, epoch?: number): boolean {
    this.assertFrameIndex(frameIndex);
    let track = this.objects.get(objectId);
    if (track === undefined) {
      track = { epoch: 0, frames: new Map() };
      this.objects.set(objectId, track);
    }
    const writeEpoch = epoch ?? track.epoch;
    if (writeEpoch < track.epoch) return false;
    if (writeEpoch > track.epoch) track.epoch = writeEpoch;
    track.frames = setSorted(track.frames, frameIndex, rle);
    return true;
  }

  /** Retrieve the mask for `objectId` at `frameIndex`, if one is stored. */
  get(objectId: string, frameIndex: number): RLEMask | undefined {
    this.assertFrameIndex(frameIndex);
    return this.objects.get(objectId)?.frames.get(frameIndex);
  }

  /**
   * All masks stored at `frameIndex`, keyed by object id. Objects with no
   * mask at that frame are omitted.
   */
  getAll(frameIndex: number): Map<string, RLEMask> {
    this.assertFrameIndex(frameIndex);
    const result = new Map<string, RLEMask>();
    for (const [objectId, track] of this.objects) {
      const rle = track.frames.get(frameIndex);
      if (rle !== undefined) result.set(objectId, rle);
    }
    return result;
  }

  /**
   * Frame indices in `[range.start, range.end)` (defaults: the whole
   * timeline) that have **no** mask for `objectId`, ascending. An unknown
   * object id yields every frame in the range.
   */
  holes(objectId: string, range?: FrameRange): number[] {
    const start = range?.start ?? 0;
    const end = range?.end ?? this.frameCount;
    if (!Number.isInteger(start) || start < 0) {
      throw new RangeError(`range.start must be a non-negative integer, got ${start}`);
    }
    if (!Number.isInteger(end) || end > this.frameCount || end < start) {
      throw new RangeError(`range.end must be an integer in [start, frameCount], got ${end}`);
    }
    const frames = this.objects.get(objectId)?.frames;
    const missing: number[] = [];
    for (let i = start; i < end; i++) {
      if (frames === undefined || !frames.has(i)) missing.push(i);
    }
    return missing;
  }

  /**
   * Drop every mask of `objectId` at frames strictly after `frameIndex`
   * and advance the object's epoch by one.
   *
   * Pairs with the `refineObject` epoch contract in `@websam3/core`: call
   * this when the object is refined at `frameIndex`, then re-propagate
   * under the returned epoch. Any straggler {@link set} still carrying the
   * previous epoch will be rejected.
   *
   * @returns The object's new current epoch.
   */
  invalidateAfter(objectId: string, frameIndex: number): number {
    this.assertFrameIndex(frameIndex);
    let track = this.objects.get(objectId);
    if (track === undefined) {
      track = { epoch: 0, frames: new Map() };
      this.objects.set(objectId, track);
    }
    for (const key of [...track.frames.keys()]) {
      if (key > frameIndex) track.frames.delete(key);
    }
    track.epoch += 1;
    return track.epoch;
  }

  /** The current epoch of `objectId` (`0` for an object never seen). */
  epoch(objectId: string): number {
    return this.objects.get(objectId)?.epoch ?? 0;
  }

  /** Ids of all objects that have ever been written to this timeline. */
  objectIds(): string[] {
    return [...this.objects.keys()];
  }

  /**
   * Serialize the timeline to a plain-JSON structure. Run-length `counts`
   * arrays are base64-encoded (little-endian uint32) to keep payloads
   * compact and `JSON.stringify`-safe.
   */
  toJSON(): MaskTimelineJSON {
    const objects: MaskTimelineJSON['objects'] = {};
    for (const [objectId, track] of this.objects) {
      const frames: Record<string, SerializedRLEMask> = {};
      for (const [frameIndex, rle] of track.frames) {
        const { counts, ...rest } = rle;
        frames[String(frameIndex)] = { ...rest, counts: countsToBase64(counts) };
      }
      objects[objectId] = { epoch: track.epoch, frames };
    }
    return {
      frameCount: this.frameCount,
      fps: this.fps,
      width: this.width,
      height: this.height,
      objects,
    };
  }

  /**
   * Drain a propagation iterator into a fresh timeline, storing every yielded
   * mask by object id and frame index.
   *
   * This is the single-consumer bridge from
   * `VideoSession.propagate()` to timeline storage: the iterator contract
   * permits exactly one consumer, and `collect` is it. Frames with no mask for
   * an object simply leave that frame a hole — sparse tracks are normal (e.g.
   * from a cancelled propagation) and are never treated as an error. Late
   * masks carrying a stale epoch are dropped by {@link set} (which returns
   * `false`), so they are silently skipped rather than resurrecting invalid
   * state.
   *
   * If the iterator throws — notably `EpochInvalidatedError` after a
   * `refineObject` — the rejection propagates to the caller, who typically
   * refines, calls {@link invalidateAfter}, and re-collects into the SAME
   * timeline under the new epoch.
   *
   * @param frames - The async iterable to drain (e.g. `session.propagate()`).
   * @param init - Geometry for the new timeline.
   * @param options - Optional live-render callback and write epoch.
   */
  static async collect(
    frames: AsyncIterable<FramePropagationResult>,
    init: MaskTimelineInit,
    options?: CollectOptions,
  ): Promise<MaskTimeline> {
    const timeline = new MaskTimeline(init);
    for await (const frame of frames) {
      for (const mask of frame.masks) {
        timeline.set(String(mask.objectId), frame.frameIndex, mask.toRLE(), options?.epoch);
      }
      options?.onFrame?.(frame);
    }
    return timeline;
  }

  /** Reconstruct a timeline previously produced by {@link toJSON}. */
  static fromJSON(json: MaskTimelineJSON): MaskTimeline {
    const timeline = new MaskTimeline({
      frameCount: json.frameCount,
      fps: json.fps,
      width: json.width,
      height: json.height,
    });
    for (const [objectId, trackJson] of Object.entries(json.objects)) {
      const frameIndices = Object.keys(trackJson.frames)
        .map(Number)
        .sort((a, b) => a - b);
      for (const frameIndex of frameIndices) {
        const serialized = trackJson.frames[String(frameIndex)];
        if (serialized === undefined) continue;
        const { counts, ...rest } = serialized;
        timeline.set(objectId, frameIndex, { ...rest, counts: base64ToCounts(counts) });
      }
      const track = timeline.objects.get(objectId) ?? { epoch: 0, frames: new Map() };
      track.epoch = trackJson.epoch;
      timeline.objects.set(objectId, track);
    }
    return timeline;
  }

  private assertFrameIndex(frameIndex: number): void {
    if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= this.frameCount) {
      throw new RangeError(
        `frameIndex must be an integer in [0, ${this.frameCount}), got ${frameIndex}`,
      );
    }
  }
}

/**
 * Insert `key` into a frame map while keeping iteration order sorted by
 * ascending frame index. Appends in O(1) when `key` is a replacement or the
 * new maximum (the common case during forward propagation); otherwise
 * rebuilds the map in sorted order.
 */
function setSorted<V>(map: Map<number, V>, key: number, value: V): Map<number, V> {
  if (map.has(key)) {
    map.set(key, value);
    return map;
  }
  let last = -1;
  for (const k of map.keys()) last = k;
  if (key > last) {
    map.set(key, value);
    return map;
  }
  const entries: [number, V][] = [...map.entries(), [key, value]];
  entries.sort((a, b) => a[0] - b[0]);
  return new Map(entries);
}

/** Encode run-length counts as base64 of little-endian uint32 bytes. */
function countsToBase64(counts: ArrayLike<number>): string {
  const bytes = new Uint8Array(counts.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < counts.length; i++) {
    view.setUint32(i * 4, counts[i] ?? 0, true);
  }
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Decode base64 little-endian uint32 bytes back into run-length counts. */
function base64ToCounts(b64: string): Uint32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const counts = new Uint32Array(Math.floor(bytes.length / 4));
  for (let i = 0; i < counts.length; i++) {
    counts[i] = view.getUint32(i * 4, true);
  }
  return counts;
}
