/**
 * The worker-side frame-source abstraction (M2 video path).
 *
 * WHY THIS EXISTS: the video engine ({@link ../worker/video/video-engine.ts})
 * must decode video frames on demand — one at a time for interaction
 * (`addObject`/`refineObject`) and sequentially for propagation — without
 * knowing anything about container demuxing or WebCodecs. Behind
 * {@link FrameSource}, a frame is an opaque {@link VideoFrame} the engine
 * preprocesses and then closes; where it came from (an MP4 + `VideoDecoder`
 * today, an `HTMLVideoElement` at M4, a Node decoder at M5) is invisible.
 *
 * OWNERSHIP CONTRACT: every {@link VideoFrame} handed out by a `FrameSource`
 * (from {@link FrameSource.frameAt} or a {@link DecodedFrame} yielded by
 * {@link FrameSource.read}) transfers ownership to the caller, who MUST call
 * `frame.close()` exactly once. Frames the source decodes but does not hand
 * out — out-of-range decode products, or the tail of a `read()` that is
 * abandoned via `return()`/`break` — are closed by the source itself, so an
 * interrupted read never leaks frames.
 */

/**
 * Container-derived facts about an attached video, computed once at demux
 * time. Mirrors the structured-clone-safe wire shape carried by the worker
 * protocol (`src/worker/protocol.ts`, worker-video wave 2); that module
 * should import this type rather than redeclare it so there is one source of
 * truth for the shape.
 */
export interface VideoSourceInfo {
  /** Exact frame count, from the MP4 sample table (mp4box counts samples). */
  frameCount: number;
  /** `frameCount / durationSeconds`. Variable frame rates are flattened to this average. */
  fps: number;
  /** Coded frame width in pixels. */
  width: number;
  /** Coded frame height in pixels. */
  height: number;
  /** Total presentation duration in microseconds. */
  durationUs: number;
  /** WebCodecs codec string, e.g. `'avc1.64100b'`. */
  codec: string;
}

/**
 * One decoded frame in presentation (display) order. `frame` ownership
 * transfers to the caller (see the module-level ownership contract).
 */
export interface DecodedFrame {
  /** The decoded frame; the caller MUST call `frame.close()` exactly once. */
  frame: VideoFrame;
  /** Zero-based presentation index into the sample table, in `[0, frameCount)`. */
  frameIndex: number;
  /** Presentation timestamp in microseconds (equals `frame.timestamp`). */
  timestampUs: number;
}

/**
 * A half-open presentation-frame span `[startFrame, endFrame)`. Both bounds
 * are optional in {@link FrameSource.read}: `startFrame` defaults to `0`,
 * `endFrame` to {@link VideoSourceInfo.frameCount}.
 */
export interface FrameRange {
  /** Inclusive first presentation frame. */
  startFrame: number;
  /** Exclusive last presentation frame. */
  endFrame: number;
}

/**
 * A demuxed, decodable video source. One instance backs one attached video
 * for the lifetime of a video session.
 */
export interface FrameSource {
  /** Container facts computed at demux time. */
  readonly info: VideoSourceInfo;

  /**
   * Decode a single frame by presentation index (random access; the
   * interaction path). GOP-aware: seeks to the enclosing keyframe and decodes
   * forward internally. Ownership of the returned frame transfers to the
   * caller. Throws `InvalidStateError` if `frameIndex` is out of range or the
   * source is closed.
   */
  frameAt(frameIndex: number): Promise<VideoFrame>;

  /**
   * Sequential forward read over `[startFrame, endFrame)` in presentation
   * order (the propagation path). Lazy and backpressure-aware: frames are
   * decoded a GOP at a time and the generator suspends between GOPs, so a
   * consumer that stops pulling stops the decoder. Each yielded
   * {@link DecodedFrame} transfers frame ownership to the caller; calling
   * `return()` (or `break`ing a `for await`) closes any frames the source has
   * decoded but not yet yielded. Throws `InvalidStateError` if the source is
   * closed.
   */
  read(range?: Partial<FrameRange>): AsyncIterableIterator<DecodedFrame>;

  /** Release demux state. Idempotent; later `frameAt`/`read` throw `InvalidStateError`. */
  close(): Promise<void>;
}
