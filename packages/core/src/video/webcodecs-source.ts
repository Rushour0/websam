/**
 * WebCodecs-backed {@link FrameSource}: mp4box demux + `VideoDecoder`, all in
 * the worker (docs/m2-internal-contracts.md §0 frame-source row, §4, §9.2).
 *
 * Split by testability:
 * - PURE sample-table math ({@link SampleTable}) — presentation⇄decode order
 *   mapping, GOP-aware sync-sample lookup, timestamp bookkeeping. Unit-tested
 *   in node.
 * - Container demux ({@link demuxMp4}) — mp4box parsing of the WHOLE blob
 *   (Blobs clone by reference across the worker boundary; the encoded
 *   samples then live in worker memory, ≈ file size — the M2 trade-off,
 *   revisited when streaming attach lands). Runs in node too (mp4box is
 *   environment-agnostic), so it is unit-tested against the committed
 *   fixture.
 * - Decode ({@link WebCodecsFrameSource}) — browser-only (`VideoDecoder`),
 *   covered by `webcodecs-source.browser.test.ts`.
 *
 * Backpressure (§5.2 stall semantics): the read iterator is pull-driven —
 * encoded chunks are fed only while `decodeQueueSize` and the un-consumed
 * frame queue are below small watermarks, so a stalled consumer (propagation
 * port out of credits) stops the decoder instead of ballooning VideoFrames.
 */

import { MP4BoxBuffer, MultiBufferStream, createFile } from 'mp4box';
import type { Movie, Sample, VisualSampleEntry } from 'mp4box';
import { InvalidStateError } from '../errors.js';
import type { DecodedFrame, FrameSource, VideoSourceInfo } from './frame-source.js';

// ---------------------------------------------------------------------------
// Pure sample-table math (unit-tested in node)
// ---------------------------------------------------------------------------

/** Timing of one encoded sample, in track-timescale units, DECODE order. */
export interface SampleTiming {
  /** Decode timestamp — must be non-decreasing across the array. */
  dts: number;
  /** Composition (presentation) timestamp. */
  cts: number;
  /** Sample duration; non-negative. */
  duration: number;
  /** True iff the sample is a sync sample (decodes without references). */
  isSync: boolean;
}

/**
 * Presentation⇄decode bookkeeping for one video track.
 *
 * "Frame index" is PRESENTATION order (what `FrameSource` speaks); mp4box
 * hands samples in DECODE order (dts-ascending). The two differ only when
 * the stream reorders (B-frames) — the math is general either way.
 */
export class SampleTable {
  /** Number of presentation frames (== sample count). */
  readonly frameCount: number;
  /** Total track duration in microseconds (sum of sample durations). */
  readonly durationUs: number;
  /** `frameCount / durationSeconds` — VFR clips flatten to one rate. */
  readonly fps: number;

  /** presentation index → decode index. */
  private readonly presToDecode: Int32Array;
  /** decode index → is_sync bit. */
  private readonly syncFlags: Uint8Array;
  /** presentation index → presentation timestamp (µs, integer). */
  private readonly presTimestampsUs: number[];
  /** exact presentation timestamp (µs) → presentation index. */
  private readonly tsToFrame: Map<number, number>;

  constructor(samples: readonly SampleTiming[], timescale: number) {
    if (!Number.isInteger(timescale) || timescale <= 0) {
      throw new InvalidStateError(`SampleTable: invalid timescale ${timescale}`);
    }
    const n = samples.length;
    if (n === 0) throw new InvalidStateError('SampleTable: empty sample list');

    let totalDuration = 0;
    for (let i = 0; i < n; i++) {
      const s = samples[i] as SampleTiming;
      if (s.duration < 0) {
        throw new InvalidStateError(`SampleTable: negative duration at decode index ${i}`);
      }
      if (i > 0 && s.dts < (samples[i - 1] as SampleTiming).dts) {
        throw new InvalidStateError(
          `SampleTable: samples not in decode order (dts decreases at index ${i})`,
        );
      }
      totalDuration += s.duration;
    }
    if (totalDuration <= 0) {
      throw new InvalidStateError('SampleTable: track has zero total duration');
    }

    // Presentation order = stable sort of decode order by (cts, dts).
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
      const sa = samples[a] as SampleTiming;
      const sb = samples[b] as SampleTiming;
      return sa.cts - sb.cts || sa.dts - sb.dts;
    });

    this.frameCount = n;
    this.durationUs = Math.round((totalDuration * 1e6) / timescale);
    this.fps = (n * 1e6) / this.durationUs;
    this.presToDecode = Int32Array.from(order);
    this.syncFlags = Uint8Array.from(samples, (s) => (s.isSync ? 1 : 0));
    this.presTimestampsUs = order.map((decodeIndex) =>
      Math.round(((samples[decodeIndex] as SampleTiming).cts * 1e6) / timescale),
    );
    this.tsToFrame = new Map(this.presTimestampsUs.map((ts, frameIndex) => [ts, frameIndex]));
  }

  /** Presentation timestamp (µs) of a presentation frame index. */
  timestampUs(frameIndex: number): number {
    this.checkFrameIndex(frameIndex, 'timestampUs');
    return this.presTimestampsUs[frameIndex] as number;
  }

  /** Decode-order position of a presentation frame index. */
  decodeIndexOf(frameIndex: number): number {
    this.checkFrameIndex(frameIndex, 'decodeIndexOf');
    return this.presToDecode[frameIndex] as number;
  }

  /**
   * GOP-aware seek origin: the greatest decode index `d` such that sample
   * `d` is a sync sample and `d <= decodeIndexOf(frameIndex)` — decoding
   * from `d` forward reconstructs the requested frame.
   */
  syncDecodeIndexFor(frameIndex: number): number {
    for (let d = this.decodeIndexOf(frameIndex); d >= 0; d--) {
      if (this.syncFlags[d] === 1) return d;
    }
    throw new InvalidStateError(
      `SampleTable: no sync sample at or before frame ${frameIndex} — unseekable track`,
    );
  }

  /**
   * Upper feed bound for a sequential read of `[startFrame, endFrame)`: the
   * greatest decode index any frame in the range needs (with reordering a
   * later presentation frame can sit EARLIER in decode order).
   */
  lastDecodeIndexFor(startFrame: number, endFrame: number): number {
    this.checkFrameIndex(startFrame, 'lastDecodeIndexFor');
    this.checkFrameIndex(endFrame - 1, 'lastDecodeIndexFor');
    let max = 0;
    for (let i = startFrame; i < endFrame; i++) {
      const d = this.presToDecode[i] as number;
      if (d > max) max = d;
    }
    return max;
  }

  /**
   * Map a decoder-output timestamp back to its presentation frame index.
   * Exact match preferred (chunk timestamps round-trip through the decoder
   * verbatim); otherwise the nearest presentation timestamp wins — a
   * robustness fallback, never the expected path.
   */
  frameIndexForTimestampUs(timestampUs: number): number {
    const exact = this.tsToFrame.get(timestampUs);
    if (exact !== undefined) return exact;
    // Binary search the (ascending) presentation timestamps for the nearest.
    const ts = this.presTimestampsUs;
    let lo = 0;
    let hi = ts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((ts[mid] as number) < timestampUs) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0) {
      const below = ts[lo - 1] as number;
      if (timestampUs - below <= (ts[lo] as number) - timestampUs) return lo - 1;
    }
    return lo;
  }

  private checkFrameIndex(frameIndex: number, method: string): void {
    if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= this.frameCount) {
      throw new InvalidStateError(
        `SampleTable.${method}: frame index ${frameIndex} outside [0, ${this.frameCount})`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// mp4box demux (environment-agnostic; unit-tested in node)
// ---------------------------------------------------------------------------

/** One encoded sample ready to wrap in an `EncodedVideoChunk`. */
interface EncodedSample {
  data: Uint8Array;
  timestampUs: number;
  durationUs: number;
  isSync: boolean;
}

/** Everything {@link WebCodecsFrameSource} needs, extracted from one MP4. */
export interface DemuxResult {
  info: VideoSourceInfo;
  table: SampleTable;
  /** Encoded samples in DECODE order. */
  samples: readonly EncodedSample[];
  config: VideoDecoderConfig;
}

/**
 * Serialize the codec-private description box (`avcC`/`hvcC`) of the track's
 * sample entry, minus its 8-byte box header — the `VideoDecoderConfig
 * .description` WebCodecs expects for AVC/HEVC in length-prefixed (mp4)
 * form. VP9/AV1 need no description; `undefined` is returned for them.
 */
function extractDescription(entries: readonly unknown[]): Uint8Array | undefined {
  for (const entry of entries) {
    const visual = entry as Partial<Pick<VisualSampleEntry, 'avcC' | 'hvcC'>>;
    const box = visual.avcC ?? visual.hvcC;
    if (box) {
      const stream = new MultiBufferStream();
      box.write(stream);
      return new Uint8Array(stream.buffer, 8);
    }
  }
  return undefined;
}

/**
 * Parse a complete MP4 buffer with mp4box and extract the first video
 * track's metadata, sample table, and encoded samples.
 *
 * @throws InvalidStateError — not a parseable MP4, no `moov`, no video
 * track, or a demux/sample inconsistency. The message names the failure;
 * the mp4box cause is chained where one exists.
 */
export function demuxMp4(buffer: ArrayBuffer): DemuxResult {
  const file = createFile();
  let movie: Movie | undefined;
  let demuxError: string | undefined;
  const collected: Sample[] = [];
  file.onError = (module, message) => {
    demuxError = `${module}: ${message}`;
  };
  file.onSamples = (_id, _user, samples) => {
    collected.push(...samples);
  };
  // Extraction options MUST be registered inside onReady — mp4box processes
  // samples during the same appendBuffer that parses moov, and options set
  // after the append are never applied to already-processed data.
  file.onReady = (info) => {
    movie = info;
    const videoTrack = info.videoTracks[0];
    if (videoTrack) {
      file.setExtractionOptions(videoTrack.id, undefined, { nbSamples: videoTrack.nb_samples });
      file.start();
    }
  };
  try {
    file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, 0), /* last */ true);
    file.flush();
  } catch (err) {
    throw new InvalidStateError('demuxMp4: failed to parse MP4 container', { cause: err });
  }
  file.stop();
  if (demuxError !== undefined) {
    throw new InvalidStateError(`demuxMp4: MP4 parse error (${demuxError})`);
  }
  if (!movie) {
    throw new InvalidStateError('demuxMp4: not an MP4 (no moov box found)');
  }
  const track = movie.videoTracks[0];
  if (!track) {
    throw new InvalidStateError('demuxMp4: MP4 contains no video track');
  }
  if (collected.length !== track.nb_samples) {
    throw new InvalidStateError(
      `demuxMp4: extracted ${collected.length} of ${track.nb_samples} samples`,
    );
  }

  const timescale = track.timescale;
  const table = new SampleTable(
    collected.map((s) => ({ dts: s.dts, cts: s.cts, duration: s.duration, isSync: s.is_sync })),
    timescale,
  );
  const samples: EncodedSample[] = collected.map((s, i) => {
    if (!s.data) {
      throw new InvalidStateError(`demuxMp4: sample ${i} carries no data`);
    }
    return {
      data: s.data,
      timestampUs: Math.round((s.cts * 1e6) / timescale),
      durationUs: Math.round((s.duration * 1e6) / timescale),
      isSync: s.is_sync,
    };
  });

  const width = track.video?.width ?? track.track_width;
  const height = track.video?.height ?? track.track_height;
  const info: VideoSourceInfo = {
    frameCount: table.frameCount,
    fps: table.fps,
    width,
    height,
    durationUs: table.durationUs,
    codec: track.codec,
  };
  const description = extractDescription(
    file.getTrackById(track.id)?.mdia?.minf?.stbl?.stsd?.entries ?? [],
  );
  const config: VideoDecoderConfig = {
    codec: track.codec,
    codedWidth: width,
    codedHeight: height,
    ...(description ? { description } : {}),
  };
  return { info, table, samples, config };
}

// ---------------------------------------------------------------------------
// The decoder-driving FrameSource (browser/worker only)
// ---------------------------------------------------------------------------

/** Max encoded chunks queued inside the VideoDecoder before feeding pauses. */
const MAX_DECODE_QUEUE = 4;
/** Max decoded frames held for an un-pulling consumer before feeding pauses. */
const MAX_UNCONSUMED_FRAMES = 2;

/**
 * One in-flight sequential read. Pull-driven: `next()` pumps the decoder;
 * an un-pulled iterator stalls with at most {@link MAX_UNCONSUMED_FRAMES}
 * decoded frames + {@link MAX_DECODE_QUEUE} encoded chunks in flight.
 */
class ReadIterator implements AsyncIterableIterator<DecodedFrame> {
  private readonly decoder: VideoDecoder;
  private readonly queue: DecodedFrame[] = [];
  private waiters: (() => void)[] = [];
  private feedIndex: number;
  private readonly lastFeedIndex: number;
  private remaining: number;
  private flushRequested = false;
  private flushSettled = false;
  private settled = false;
  private failure: Error | undefined;

  constructor(
    private readonly samples: readonly EncodedSample[],
    private readonly table: SampleTable,
    private readonly startFrame: number,
    private readonly endFrame: number,
    config: VideoDecoderConfig,
    private readonly onSettled: () => void,
  ) {
    this.feedIndex = table.syncDecodeIndexFor(startFrame);
    this.lastFeedIndex = table.lastDecodeIndexFor(startFrame, endFrame);
    this.remaining = endFrame - startFrame;
    this.decoder = new VideoDecoder({
      output: (frame) => this.onFrame(frame),
      error: (err) =>
        this.fail(new InvalidStateError(`read(): VideoDecoder error: ${err.message}`, { cause: err })),
    });
    // Re-pump when the decoder drains its input queue — feeding is gated on
    // decodeQueueSize, and without this a paused feed would never resume.
    this.decoder.ondequeue = () => this.pump();
    this.decoder.configure(config);
  }

  /** Frames decoded but not yet yielded (the source's leak witness). */
  get liveFrames(): number {
    return this.queue.length;
  }

  [Symbol.asyncIterator](): AsyncIterableIterator<DecodedFrame> {
    return this;
  }

  async next(): Promise<IteratorResult<DecodedFrame>> {
    for (;;) {
      if (this.failure) {
        const err = this.failure;
        this.cancel();
        throw err;
      }
      if (this.queue.length > 0) {
        const value = this.queue.shift() as DecodedFrame;
        this.remaining--;
        // Settle as soon as the last frame leaves: the source slot frees up
        // without requiring a trailing next(), and leftover frames (none in
        // the normal path) get closed.
        if (this.remaining === 0) this.cancel();
        return { value, done: false };
      }
      if (this.settled || this.remaining === 0) {
        this.cancel();
        return { value: undefined, done: true };
      }
      if (this.flushSettled) {
        this.fail(
          new InvalidStateError(
            `read(): decoder drained after ${this.endFrame - this.startFrame - this.remaining}` +
              ` of ${this.endFrame - this.startFrame} frames`,
          ),
        );
        continue;
      }
      this.pump();
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
  }

  async return(): Promise<IteratorResult<DecodedFrame>> {
    this.cancel();
    return { value: undefined, done: true };
  }

  async throw(err?: unknown): Promise<IteratorResult<DecodedFrame>> {
    this.cancel();
    throw err;
  }

  /**
   * Cancel/settle: close the decoder and every frame still held here.
   * Frames already yielded belong to the consumer. Idempotent.
   */
  cancel(): void {
    if (this.settled) return;
    this.settled = true;
    for (const queued of this.queue) queued.frame.close();
    this.queue.length = 0;
    if (this.decoder.state !== 'closed') this.decoder.close();
    this.onSettled();
    this.notify();
  }

  private onFrame(frame: VideoFrame): void {
    const frameIndex = this.table.frameIndexForTimestampUs(frame.timestamp);
    // Warm-up frames (sync sample → startFrame) and post-range spill are
    // closed immediately — they never count as live.
    if (this.settled || frameIndex < this.startFrame || frameIndex >= this.endFrame) {
      frame.close();
      return;
    }
    this.queue.push({ frameIndex, timestampUs: frame.timestamp, frame });
    this.notify();
  }

  /** Feed the decoder up to the watermarks; flush once everything is fed. */
  private pump(): void {
    if (this.settled || this.failure) return;
    while (
      this.feedIndex <= this.lastFeedIndex &&
      this.decoder.decodeQueueSize < MAX_DECODE_QUEUE &&
      this.queue.length < MAX_UNCONSUMED_FRAMES
    ) {
      const sample = this.samples[this.feedIndex] as EncodedSample;
      this.decoder.decode(
        new EncodedVideoChunk({
          type: sample.isSync ? 'key' : 'delta',
          timestamp: sample.timestampUs,
          duration: sample.durationUs,
          data: sample.data,
        }),
      );
      this.feedIndex++;
    }
    if (this.feedIndex > this.lastFeedIndex && !this.flushRequested) {
      this.flushRequested = true;
      // Reordered tails only surface on flush. Rejection is expected when
      // cancel() closed the decoder mid-flush; real failures already went
      // through the error callback.
      this.decoder.flush().then(
        () => {
          this.flushSettled = true;
          this.notify();
        },
        () => {
          this.flushSettled = true;
          this.notify();
        },
      );
    }
  }

  private fail(err: Error): void {
    if (!this.failure) this.failure = err;
    this.notify();
  }

  private notify(): void {
    const waiters = this.waiters;
    this.waiters = [];
    for (const wake of waiters) wake();
  }
}

/**
 * {@link FrameSource} over mp4box demux + WebCodecs `VideoDecoder`.
 * Construct via {@link createWebCodecsFrameSource}.
 */
export class WebCodecsFrameSource implements FrameSource {
  readonly info: VideoSourceInfo;
  private readonly table: SampleTable;
  private readonly samples: readonly EncodedSample[];
  private readonly config: VideoDecoderConfig;
  private activeRead: ReadIterator | undefined;
  private disposed = false;

  /** @internal Use {@link createWebCodecsFrameSource}. */
  constructor(demuxed: DemuxResult) {
    this.info = demuxed.info;
    this.table = demuxed.table;
    this.samples = demuxed.samples;
    this.config = demuxed.config;
  }

  read(startFrame: number, endFrame: number): AsyncIterableIterator<DecodedFrame> {
    this.ensureIdle('read');
    if (
      !Number.isInteger(startFrame) ||
      !Number.isInteger(endFrame) ||
      startFrame < 0 ||
      startFrame >= endFrame ||
      endFrame > this.info.frameCount
    ) {
      throw new InvalidStateError(
        `read(${startFrame}, ${endFrame}): expected integers with` +
          ` 0 <= start < end <= ${this.info.frameCount}`,
      );
    }
    const iterator = new ReadIterator(
      this.samples,
      this.table,
      startFrame,
      endFrame,
      this.config,
      () => {
        if (this.activeRead === iterator) this.activeRead = undefined;
      },
    );
    this.activeRead = iterator;
    return iterator;
  }

  async frameAt(frameIndex: number): Promise<VideoFrame> {
    this.ensureIdle('frameAt');
    if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= this.info.frameCount) {
      throw new InvalidStateError(
        `frameAt(${frameIndex}): frame index outside [0, ${this.info.frameCount})`,
      );
    }
    const iterator = this.read(frameIndex, frameIndex + 1);
    try {
      const result = await iterator.next();
      if (result.done) {
        throw new InvalidStateError(`frameAt(${frameIndex}): decoder produced no frame`);
      }
      return result.value.frame;
    } finally {
      await iterator.return?.();
    }
  }

  /**
   * Live-resource census (same shape idea as `Backend.debugStats`, §1.1):
   * frames the SOURCE currently holds. Yielded frames are the caller's and
   * never count. Zero whenever no read is active — the browser test's
   * no-dangling-frames witness.
   */
  debugStats(): { liveFrames: number } {
    return { liveFrames: this.activeRead?.liveFrames ?? 0 };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.activeRead?.cancel();
  }

  private ensureIdle(method: string): void {
    if (this.disposed) {
      throw new InvalidStateError(`${method}: FrameSource is disposed`);
    }
    if (this.activeRead) {
      throw new InvalidStateError(
        `${method}: a read() is already active — finish or return() it first` +
          ' (one sequential reader per source)',
      );
    }
  }
}

/**
 * Demux an MP4 blob and stand up a WebCodecs-backed {@link FrameSource}.
 *
 * The blob is read fully into worker memory (encoded bytes only — decoded
 * frames stay bounded by the read watermarks).
 *
 * @throws InvalidStateError — WebCodecs unavailable in this environment,
 * the blob is not a valid MP4 with a video track ({@link demuxMp4}'s
 * errors), or `VideoDecoder` cannot decode the track's codec.
 */
export async function createWebCodecsFrameSource(source: Blob): Promise<WebCodecsFrameSource> {
  if (typeof VideoDecoder !== 'function' || typeof EncodedVideoChunk !== 'function') {
    throw new InvalidStateError(
      'createWebCodecsFrameSource: WebCodecs (VideoDecoder) is not available in this environment',
    );
  }
  const buffer = await source.arrayBuffer();
  const demuxed = demuxMp4(buffer);
  const support = await VideoDecoder.isConfigSupported(demuxed.config).catch(() => undefined);
  if (!support?.supported) {
    throw new InvalidStateError(
      `createWebCodecsFrameSource: VideoDecoder cannot decode codec '${demuxed.info.codec}'`,
    );
  }
  return new WebCodecsFrameSource(demuxed);
}
