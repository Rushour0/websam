/**
 * The worker-side frame-source contract — how the video engine pulls decoded
 * frames out of an attached video (docs/m2-internal-contracts.md §4/§5).
 *
 * A {@link FrameSource} lives entirely IN THE WORKER: the main thread hands a
 * `Blob` across the boundary (Blobs structured-clone by reference — no byte
 * copy) and never touches demux or decode. The engine drives the source two
 * ways:
 *
 * - `read(start, end)` — the propagation loop's sequential forward pass. The
 *   iterator is PULL-DRIVEN: decoding only advances while the consumer keeps
 *   calling `next()`, so when the propagation port runs out of credits
 *   (§5.2) the source stops feeding the decoder — decode pauses, buffers
 *   held. That is the documented stall semantics.
 * - `frameAt(i)` — random access for interaction steps (`addObject` /
 *   `refineObject`). GOP-aware: the source seeks to the preceding sync
 *   sample and decodes forward, discarding warm-up frames.
 *
 * Ownership rule (§4.5): every {@link VideoFrame} handed out is owned by the
 * CALLER, who must `frame.close()` it (the engine closes after preprocess).
 * Frames still held inside the source (decoded but not yet yielded) are the
 * source's responsibility and are closed on cancellation/disposal.
 */

/**
 * Metadata for an attached video source.
 *
 * Structurally identical to the `VideoSourceInfo` the worker protocol ships
 * to the main thread (m2-internal-contracts.md §5.1, added to
 * `src/worker/protocol.ts` in wave 2) — that copy is authoritative for the
 * wire; this one must never drift from it.
 */
export interface VideoSourceInfo {
  /** Exact frame count from the mp4 sample table (mp4box counts samples). */
  frameCount: number;
  /** `frameCount / durationSeconds` — VFR clips are flattened to one rate. */
  fps: number;
  width: number;
  height: number;
  durationUs: number;
  /** Codec string of the video track, e.g. `'avc1.640028'`. */
  codec: string;
}

/** One decoded frame yielded by {@link FrameSource.read}. */
export interface DecodedFrame {
  /** Presentation-order frame index, `[0, info.frameCount)`. */
  frameIndex: number;
  /** Presentation timestamp in microseconds (equals `frame.timestamp`). */
  timestampUs: number;
  /** The decoded frame — OWNED BY THE CALLER, who must `close()` it. */
  frame: VideoFrame;
}

/**
 * A demuxed, seekable, worker-owned view of one video track.
 *
 * Implementations: {@link ../video/webcodecs-source.js | WebCodecsFrameSource}
 * (mp4box demux + `VideoDecoder`). At most ONE read iterator may be active at
 * a time (`read`/`frameAt` while another read is in flight throw
 * `InvalidStateError`) — the engine is strictly sequential per session, so
 * this is a programming-error guard, not a scheduling primitive.
 */
export interface FrameSource {
  /** Metadata derived from the container at attach time. */
  readonly info: VideoSourceInfo;
  /**
   * Sequential forward read of presentation frames `[startFrame, endFrame)`.
   *
   * Backpressure-aware: chunks are fed to the decoder only while the
   * consumer pulls, bounded by a small watermark — an un-pulled iterator
   * holds at most a few decoded frames. `return()` (or `throw()`) cancels
   * mid-stream: the decoder is closed and every frame still held by the
   * source is closed; frames already yielded stay the caller's to close.
   *
   * @throws InvalidStateError — bounds outside `[0, frameCount]`, a read
   * already active, or the source is disposed.
   */
  read(startFrame: number, endFrame: number): AsyncIterableIterator<DecodedFrame>;
  /**
   * Decode exactly one frame (GOP-aware random access): seek to the nearest
   * preceding sync sample, decode forward, discard warm-up frames. The
   * returned frame is owned by the caller.
   *
   * @throws InvalidStateError — out-of-range index, a read already active,
   * or the source is disposed.
   */
  frameAt(frameIndex: number): Promise<VideoFrame>;
  /**
   * Release the decoder and every frame the source still holds. Idempotent;
   * any later `read`/`frameAt` throws `InvalidStateError`.
   */
  dispose(): void;
}
