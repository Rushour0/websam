/**
 * {@link FrameSource} backed by mp4box (demux) + WebCodecs `VideoDecoder`.
 *
 * Runs in the worker. The whole MP4 is demuxed up front into an in-memory
 * sample table (small: clips are held by reference as a `Blob` and the decoded
 * bytes never cross a thread), from which two decode paths are served:
 *
 *  - {@link WebCodecsFrameSource.frameAt} — random access: seek to the sample's
 *    enclosing keyframe, decode that GOP, return the requested frame.
 *  - {@link WebCodecsFrameSource.read} — sequential forward: walk GOPs from the
 *    one covering `startFrame`, yielding in presentation order with
 *    GOP-granular backpressure (the async generator suspends between GOPs).
 *
 * GOP-awareness comes from the sync-sample table (`is_sync`): a target frame is
 * only decodable after its enclosing keyframe, and closed-GOP display order is
 * recovered by fully decoding each touched GOP and sorting its output by
 * presentation index. Every GOP is decoded from a key frame, so a fresh
 * `VideoDecoder` (frameAt) or a flush between GOPs (read) is always seeded
 * correctly.
 */

import { createFile, DataStream, type ISOFile, MP4BoxBuffer } from 'mp4box';
import { InvalidStateError } from '../errors.js';
import type { DecodedFrame, FrameRange, FrameSource, VideoSourceInfo } from './frame-source.js';

/** One demuxed sample, in decode order, with its bytes copied out of mp4box. */
interface DecodedSample {
  /** Presentation timestamp in microseconds (from `cts`, the decoder chunk timestamp). */
  timestampUs: number;
  /** Whether this sample is a sync sample (keyframe / GOP boundary). */
  isSync: boolean;
  /** Encoded sample bytes (owned copy — mp4box may recycle its buffers). */
  data: Uint8Array;
}

/** Everything demux produces: the sample table plus the decoder config and index maps. */
interface DemuxResult {
  info: VideoSourceInfo;
  config: VideoDecoderConfig;
  /** Samples in decode order (mp4box `sample.number` order). */
  samples: DecodedSample[];
  /** `displayToDecode[displayIndex] = decodeIndex`, sorted by presentation time. */
  displayToDecode: number[];
  /** `timestampUs → displayIndex`, to map a decoded frame back to its presentation index. */
  displayIndexByTs: Map<number, number>;
  /** Decode indices that are sync samples, ascending. */
  syncIndices: number[];
}

/** mp4box sample-entry shape carrying a codec-configuration child box we can serialize. */
interface ConfigBoxHolder {
  write(stream: DataStream): void;
}

function readConfigBox(entry: unknown): ConfigBoxHolder | undefined {
  const e = entry as Record<string, unknown>;
  const box = e['avcC'] ?? e['hvcC'] ?? e['av1C'] ?? e['vpcC'];
  if (box && typeof (box as { write?: unknown }).write === 'function') {
    return box as ConfigBoxHolder;
  }
  return undefined;
}

/**
 * Serialize the codec-configuration box (avcC/hvcC/av1C/vpcC) to the raw bytes
 * WebCodecs wants as `VideoDecoderConfig.description` — the box payload with its
 * 8-byte (size + fourcc) header stripped. `undefined` when the codec carries
 * its parameter sets in-band (e.g. avc3/annexb), in which case no description
 * is needed.
 */
function extractDescription(file: ISOFile, trackId: number): Uint8Array | undefined {
  const trak = file.getTrackById(trackId) as unknown as {
    mdia?: { minf?: { stbl?: { stsd?: { entries?: unknown[] } } } };
  };
  const entries = trak?.mdia?.minf?.stbl?.stsd?.entries ?? [];
  for (const entry of entries) {
    const box = readConfigBox(entry);
    if (box) {
      // Default endianness is BIG_ENDIAN, which is what the ISO box format uses.
      const stream = new DataStream();
      box.write(stream);
      return new Uint8Array(stream.buffer, 8);
    }
  }
  return undefined;
}

/**
 * Demux an MP4 `Blob` into a sample table + decoder config. Rejects with
 * `InvalidStateError` when the blob is not a decodable MP4 with a video track.
 */
async function demux(blob: Blob): Promise<DemuxResult> {
  const file = createFile();
  const arrayBuffer = await blob.arrayBuffer();

  return await new Promise<DemuxResult>((resolve, reject) => {
    let settled = false;
    const fail = (message: string): void => {
      if (settled) return;
      settled = true;
      reject(new InvalidStateError(message));
    };

    file.onError = (module, message) => {
      fail(`not a decodable MP4 (${module}: ${message})`);
    };

    file.onReady = (movie) => {
      const track = movie.videoTracks[0];
      if (!track) {
        fail('MP4 has no video track');
        return;
      }
      const video = track.video;
      if (!video) {
        fail('MP4 video track has no visual sample entry');
        return;
      }

      const collected: DecodedSample[] = [];
      const decodeOrder: number[] = [];

      file.onSamples = (_id, _user, samples) => {
        if (settled) return;
        for (const sample of samples) {
          if (!sample.data) continue;
          const idx = sample.number;
          // Copy: mp4box may release the backing buffer via releaseUsedSamples.
          collected[idx] = {
            timestampUs: Math.round((sample.cts * 1_000_000) / sample.timescale),
            isSync: sample.is_sync,
            data: sample.data.slice(),
          };
          decodeOrder.push(idx);
        }
        file.releaseUsedSamples(track.id, decodeOrder.length);

        if (decodeOrder.length < track.nb_samples) return;

        // All samples in hand — assemble the decode-ordered table.
        const orderedSamples: DecodedSample[] = [];
        for (let i = 0; i < track.nb_samples; i++) {
          const s = collected[i];
          if (!s) {
            fail(`MP4 sample table gap at index ${i}`);
            return;
          }
          orderedSamples.push(s);
        }

        // Presentation order: decode indices sorted by presentation timestamp.
        const displayToDecode = orderedSamples
          .map((_s, decodeIndex) => decodeIndex)
          .sort((a, b) => {
            const ta = orderedSamples[a]!.timestampUs;
            const tb = orderedSamples[b]!.timestampUs;
            return ta === tb ? a - b : ta - tb;
          });

        const displayIndexByTs = new Map<number, number>();
        displayToDecode.forEach((decodeIndex, displayIndex) => {
          displayIndexByTs.set(orderedSamples[decodeIndex]!.timestampUs, displayIndex);
        });

        const syncIndices = orderedSamples
          .map((s, decodeIndex) => (s.isSync ? decodeIndex : -1))
          .filter((v) => v >= 0);
        if (syncIndices.length === 0 || syncIndices[0] !== 0) {
          // Every decodable MP4 starts its first GOP with a sync sample.
          fail('MP4 video track has no leading keyframe');
          return;
        }

        const durationUs = Math.round((track.duration * 1_000_000) / track.timescale);
        const durationSeconds = durationUs / 1_000_000;
        const info: VideoSourceInfo = {
          frameCount: track.nb_samples,
          fps: durationSeconds > 0 ? track.nb_samples / durationSeconds : 0,
          width: video.width,
          height: video.height,
          durationUs,
          codec: track.codec,
        };

        const config: VideoDecoderConfig = {
          codec: track.codec,
          codedWidth: video.width,
          codedHeight: video.height,
        };
        const description = extractDescription(file, track.id);
        if (description) config.description = description;

        settled = true;
        resolve({
          info,
          config,
          samples: orderedSamples,
          displayToDecode,
          displayIndexByTs,
          syncIndices,
        });
      };

      file.setExtractionOptions(track.id, null, { nbSamples: track.nb_samples });
      file.start();
    };

    const mp4Buffer = MP4BoxBuffer.fromArrayBuffer(arrayBuffer, 0);
    try {
      file.appendBuffer(mp4Buffer);
      file.flush();
    } catch (err) {
      fail(`failed to parse MP4: ${(err as Error).message}`);
      return;
    }
    // A valid MP4 fires onReady synchronously during appendBuffer/flush; if it
    // has not, the blob is not a parseable MP4.
    fail('not a decodable MP4 (no moov box found)');
  });
}

/**
 * A short-lived `VideoDecoder` wrapper that decodes one GOP's worth of samples
 * and collects the output frames. `decodeGop` flushes, so on return every
 * output frame for the fed samples has been emitted.
 */
class GopDecoder {
  #decoder: VideoDecoder;
  #pending: VideoFrame[] = [];
  #fatal: Error | null = null;

  constructor(config: VideoDecoderConfig) {
    this.#decoder = new VideoDecoder({
      output: (frame) => this.#pending.push(frame),
      error: (err) => {
        this.#fatal = err;
      },
    });
    try {
      this.#decoder.configure(config);
    } catch (err) {
      throw new InvalidStateError(`unsupported video config: ${(err as Error).message}`);
    }
  }

  /** Decode `samples` (a full GOP, key sample first) and return the output frames. */
  async decodeGop(samples: readonly DecodedSample[]): Promise<VideoFrame[]> {
    this.#pending = [];
    for (const sample of samples) {
      this.#decoder.decode(
        new EncodedVideoChunk({
          type: sample.isSync ? 'key' : 'delta',
          timestamp: sample.timestampUs,
          data: sample.data,
        }),
      );
    }
    await this.#decoder.flush();
    if (this.#fatal) {
      const err = this.#fatal;
      this.#fatal = null;
      throw new InvalidStateError(`video decode failed: ${err.message}`);
    }
    const out = this.#pending;
    this.#pending = [];
    return out;
  }

  /** Close the decoder and any output frames not yet handed out. */
  close(): void {
    for (const frame of this.#pending) {
      try {
        frame.close();
      } catch {
        /* already closed */
      }
    }
    this.#pending = [];
    if (this.#decoder.state !== 'closed') {
      try {
        this.#decoder.close();
      } catch {
        /* already closed */
      }
    }
  }
}

/** {@link FrameSource} over an MP4 `Blob`, using mp4box + WebCodecs. */
export class WebCodecsFrameSource implements FrameSource {
  readonly info: VideoSourceInfo;

  #demux: DemuxResult;
  #closed = false;

  private constructor(demuxed: DemuxResult) {
    this.#demux = demuxed;
    this.info = demuxed.info;
  }

  /** Demux `blob` and construct a source. Rejects `InvalidStateError` on a bad/undecodable MP4. */
  static async create(blob: Blob): Promise<WebCodecsFrameSource> {
    return new WebCodecsFrameSource(await demux(blob));
  }

  /** Largest sync-sample decode index at or before `decodeIndex`. */
  #gopStart(decodeIndex: number): number {
    const { syncIndices } = this.#demux;
    let start = syncIndices[0]!;
    for (const s of syncIndices) {
      if (s <= decodeIndex) start = s;
      else break;
    }
    return start;
  }

  /** Last decode index of the GOP that `decodeIndex` belongs to. */
  #gopEnd(decodeIndex: number): number {
    const { syncIndices, samples } = this.#demux;
    const start = this.#gopStart(decodeIndex);
    const next = syncIndices.find((s) => s > start);
    return next === undefined ? samples.length - 1 : next - 1;
  }

  async frameAt(frameIndex: number): Promise<VideoFrame> {
    this.#assertOpen();
    const { frameCount } = this.info;
    if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= frameCount) {
      throw new InvalidStateError(
        `frameAt(${frameIndex}) out of range [0, ${frameCount})`,
      );
    }

    const targetDecode = this.#demux.displayToDecode[frameIndex]!;
    const gopStart = this.#gopStart(targetDecode);
    const gopEnd = this.#gopEnd(targetDecode);
    const gop = this.#demux.samples.slice(gopStart, gopEnd + 1);

    const decoder = new GopDecoder(this.#demux.config);
    try {
      const frames = await decoder.decodeGop(gop);
      let found: VideoFrame | undefined;
      for (const frame of frames) {
        if (found === undefined && this.#displayIndexOf(frame.timestamp) === frameIndex) {
          found = frame;
        } else {
          frame.close();
        }
      }
      if (!found) {
        throw new InvalidStateError(`frame ${frameIndex} did not decode`);
      }
      return found;
    } finally {
      decoder.close();
    }
  }

  async *read(range?: Partial<FrameRange>): AsyncIterableIterator<DecodedFrame> {
    this.#assertOpen();
    const { frameCount } = this.info;
    const startFrame = Math.max(0, Math.trunc(range?.startFrame ?? 0));
    const endFrame = Math.min(frameCount, Math.trunc(range?.endFrame ?? frameCount));
    if (startFrame >= endFrame) return;

    // Decode-index span that produces the wanted presentation frames.
    const wantedDecode: number[] = [];
    for (let display = startFrame; display < endFrame; display++) {
      wantedDecode.push(this.#demux.displayToDecode[display]!);
    }
    const minDecode = Math.min(...wantedDecode);
    const maxDecode = Math.max(...wantedDecode);
    const stopDecode = this.#gopEnd(maxDecode);

    const decoder = new GopDecoder(this.#demux.config);
    // Frames decoded and in-range but not yet yielded — closed by finally on abort.
    const undelivered = new Set<VideoFrame>();
    try {
      let cursor = this.#gopStart(minDecode);
      while (cursor <= stopDecode) {
        const gopEnd = this.#gopEnd(cursor);
        const gop = this.#demux.samples.slice(cursor, gopEnd + 1);
        const frames = await decoder.decodeGop(gop);

        const inRange: DecodedFrame[] = [];
        for (const frame of frames) {
          const displayIndex = this.#displayIndexOf(frame.timestamp);
          if (displayIndex !== undefined && displayIndex >= startFrame && displayIndex < endFrame) {
            undelivered.add(frame);
            inRange.push({ frame, frameIndex: displayIndex, timestampUs: frame.timestamp });
          } else {
            frame.close();
          }
        }
        inRange.sort((a, b) => a.frameIndex - b.frameIndex);

        for (const item of inRange) {
          undelivered.delete(item.frame);
          yield item;
        }
        cursor = gopEnd + 1;
      }
    } finally {
      for (const frame of undelivered) {
        try {
          frame.close();
        } catch {
          /* already closed */
        }
      }
      undelivered.clear();
      decoder.close();
    }
  }

  async close(): Promise<void> {
    this.#closed = true;
  }

  #displayIndexOf(timestampUs: number): number | undefined {
    return this.#demux.displayIndexByTs.get(timestampUs);
  }

  #assertOpen(): void {
    if (this.#closed) throw new InvalidStateError('frame source is closed');
  }
}

/** Demux an MP4 `Blob` and return a ready {@link FrameSource}. */
export function createWebCodecsFrameSource(blob: Blob): Promise<FrameSource> {
  return WebCodecsFrameSource.create(blob);
}
