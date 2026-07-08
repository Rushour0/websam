/**
 * The concrete {@link ImageSession}: a cheap main-thread view over one
 * encoded slot inside the worker engine.
 *
 * Boundary rules (docs/m1-internal-contracts.md §4.3): the MAIN thread
 * normalizes `ImageSource → ImageBitmap` and TRANSFERS it to the worker
 * (which closes it); the worker computes the coordinate transform and returns
 * it in the encode response; decode payloads come back with their binary
 * masks transferred and are wrapped in {@link MaskResultImpl} together with
 * the stored transform.
 */

import * as Comlink from 'comlink';
import type { CoordinateTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';
import { MaskResultImpl } from '../masks/mask-result.js';
import type { EncodeResult, ImageSession, MaskResult, Prompt } from '../segmenter.js';
import type { RemoteEngine } from './spawn-worker.js';

/** Accepted encode inputs (mirrors the {@link ImageSession.encode} signature). */
type ImageSource = ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas;

/** The abort rejection value: the signal's reason (always set once aborted). */
function abortReason(signal: AbortSignal): unknown {
  // signal.reason defaults to a DOMException named 'AbortError'; the fallback
  // only guards exotic AbortSignal impls that abort without a reason.
  return (signal.reason as unknown) ?? new DOMException('encode was aborted', 'AbortError');
}

/**
 * Normalize an {@link ImageSource} to an {@link ImageBitmap} on the main
 * thread. An `ImageBitmap` input is consumed AS-IS (it will be transferred to
 * the worker and closed there — documented behavior); everything else goes
 * through `createImageBitmap`.
 */
function toImageBitmap(image: ImageSource): ImageBitmap | Promise<ImageBitmap> {
  if (typeof ImageBitmap === 'function' && image instanceof ImageBitmap) {
    return image;
  }
  if (typeof createImageBitmap !== 'function') {
    throw new InvalidStateError(
      'ImageSession.encode: createImageBitmap is not available in this environment ' +
        '(requires a browser window context); pass an ImageBitmap instead',
    );
  }
  return createImageBitmap(image);
}

/**
 * Encode-once / decode-per-click image session, proxying to the worker
 * engine. Sessions are cheap views — each encoded slot holds ≈21 MB of
 * device memory worker-side until {@link dispose} (no M1 hard cap).
 */
export class ImageSessionImpl implements ImageSession {
  readonly #engine: RemoteEngine;
  readonly #sessionId: number;
  /** Set on successful encode; also the `isEncoded` witness. */
  #transform: CoordinateTransform | undefined;
  #disposed = false;

  /**
   * @param engine - The Comlink-wrapped worker engine (or a test double).
   * @param sessionId - Worker-side slot id from `engine.createSession()`.
   */
  constructor(engine: RemoteEngine, sessionId: number) {
    this.#engine = engine;
    this.#sessionId = sessionId;
  }

  /** True once {@link encode} has succeeded (and the session is not disposed). */
  get isEncoded(): boolean {
    return this.#transform !== undefined;
  }

  #assertLive(method: string): void {
    if (this.#disposed) {
      throw new InvalidStateError(`ImageSession.${method} called on a disposed session`);
    }
  }

  /**
   * Run the vision encoder once; embeddings stay worker/device-resident.
   *
   * The input is normalized to an `ImageBitmap` on the main thread and
   * TRANSFERRED to the worker (an `ImageBitmap` input is consumed — do not
   * reuse it). Abort semantics (M1): a signal aborted before dispatch rejects
   * immediately; an abort while the encoder is in flight cannot cancel the
   * ORT run — the result is discarded, the call rejects with the abort
   * reason, and {@link isEncoded} is not updated.
   *
   * @throws InvalidStateError - disposed session, or no `createImageBitmap`
   * in this environment for non-bitmap inputs.
   */
  async encode(image: ImageSource, options?: { signal?: AbortSignal }): Promise<EncodeResult> {
    this.#assertLive('encode');
    const signal = options?.signal;
    if (signal?.aborted) throw abortReason(signal);

    const bitmap = await toImageBitmap(image);
    if (signal?.aborted) {
      bitmap.close();
      throw abortReason(signal);
    }

    let response;
    try {
      response = await this.#engine.encodeImage(
        this.#sessionId,
        Comlink.transfer(bitmap, [bitmap]),
      );
    } catch (err) {
      if (signal?.aborted) throw abortReason(signal);
      throw err;
    }
    if (signal?.aborted) throw abortReason(signal);

    this.#transform = response.transform;
    return { width: response.width, height: response.height, encodeMs: response.encodeMs };
  }

  /**
   * Decode masks for prompts (source-pixel space) against the cached
   * embedding. Each returned mask is an immutable {@link MaskResultImpl}
   * carrying the encode-time coordinate transform.
   *
   * @throws InvalidStateError - disposed session, or {@link encode} has not
   * succeeded yet.
   */
  async decode(
    prompts: Prompt[],
    options?: { multimask?: boolean; objectId?: number },
  ): Promise<MaskResult[]> {
    this.#assertLive('decode');
    const transform = this.#transform;
    if (transform === undefined) {
      throw new InvalidStateError('ImageSession.decode called before encode() succeeded');
    }
    const payloads = await this.#engine.decode(this.#sessionId, {
      prompts,
      multimask: options?.multimask,
      objectId: options?.objectId,
    });
    return payloads.map(
      (payload) =>
        new MaskResultImpl({
          objectId: payload.objectId,
          score: payload.score,
          width: payload.width,
          height: payload.height,
          binaryMask: new Uint8Array(payload.binaryMask),
          transform,
        }),
    );
  }

  /**
   * Release the worker-side encoded slot. Idempotent (repeat calls are
   * no-ops); {@link encode}/{@link decode} after dispose throw
   * {@link InvalidStateError}.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#transform = undefined;
    // Fire-and-forget: the worker-side close needs no answer, and a already-
    // terminated worker must not surface an unhandled rejection here.
    void Promise.resolve(this.#engine.closeSession(this.#sessionId)).catch(() => {});
  }
}
