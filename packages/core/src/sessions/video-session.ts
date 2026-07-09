/**
 * The concrete {@link VideoSession}: the state machine described normatively
 * by docs/m2-internal-contracts.md §6.1 (states/epoch table) and §6.2 (the
 * iterator). See {@link createPropagationIterator} for the pull-credit
 * stream client.
 *
 * KNOWN GAP (flagged for the wave-2 consolidating gate): the frozen §5.1
 * wire shapes (`VideoSourceInfo`, `VideoObjectResult`, `PropagationPushMessage`)
 * carry no `CoordinateTransform`, even though §4.1 says the engine "computes
 * the ONE CoordinateTransform" at `attach()`. `MaskResultImpl` requires a
 * `transform` field (coordinate-contract rule 4), but none of its methods
 * actually USE that field today, so this session synthesizes an
 * identity-ish placeholder from `VideoSourceInfo.width/height` (see
 * {@link #placeholderTransform}) rather than the real model-space transform.
 * Functionally inert now; if a future consumer relies on `MaskResult`
 * carrying the true transform, thread the real one back from
 * `VideoEngine.attach` through `VideoSourceInfo` (or `VideoObjectResult`)
 * instead of reconstructing it here.
 */

import * as Comlink from 'comlink';
import type { CoordinateTransform } from '../coords.js';
import { InvalidStateError, NotImplementedError } from '../errors.js';
import { MaskResultImpl } from '../masks/mask-result.js';
import type { FramePropagationResult, MaskResult, Prompt, VideoSession } from '../segmenter.js';
import type { PropagateRequest, VideoSourceInfo } from '../worker/protocol.js';
import {
  createPropagationIterator,
  type PropagationIteratorHandle,
} from './propagation-iterator.js';
import type { RemoteEngine } from './spawn-worker.js';

/** Initial pull-credit grant (docs/m2-internal-contracts.md §5.2). */
const PREFETCH = 4;

type State = 'idle' | 'propagating' | 'disposed';

/** An iterator whose every `next()`/`throw()` rejects/throws `err`; used for the poisoned-iterator cases in §6.1. */
function poisonedIterator(err: unknown): AsyncIterableIterator<FramePropagationResult> {
  const result: IteratorResult<FramePropagationResult> = { done: true, value: undefined };
  return {
    [Symbol.asyncIterator]() {
      return this;
    },
    async next() {
      throw err;
    },
    async return() {
      return result;
    },
    async throw(e?: unknown) {
      throw e ?? err;
    },
  };
}

/**
 * Build the coordinate-contract placeholder transform for one attached
 * source — see the module-doc "KNOWN GAP" note above.
 */
function placeholderTransform(info: VideoSourceInfo): CoordinateTransform {
  return {
    scaleX: 1,
    scaleY: 1,
    padX: 0,
    padY: 0,
    srcW: info.width,
    srcH: info.height,
    modelSize: Math.max(info.width, info.height),
    mode: 'square-stretch',
  };
}

/**
 * Interactive video object tracking session — proxies to the worker's video
 * engine (docs/m2-internal-contracts.md §4). The MAIN THREAD is
 * epoch-authoritative: `#epoch` is bumped here (never worker-side) and every
 * mutator that races an active `propagate()` iterator cancels it first (§6.1).
 */
export class VideoSessionImpl implements VideoSession {
  readonly #engine: RemoteEngine;
  readonly #sessionId: number;

  #state: State = 'idle';
  #epoch = 0;
  #sourceInfo: VideoSourceInfo | undefined;
  #transform: CoordinateTransform | undefined;
  #activeIterator: PropagationIteratorHandle | undefined;

  /**
   * @param engine - The Comlink-wrapped worker engine (or a test double).
   * @param sessionId - Worker-side video-session slot id from `engine.createVideoSession()`.
   */
  constructor(engine: RemoteEngine, sessionId: number) {
    this.#engine = engine;
    this.#sessionId = sessionId;
  }

  #assertNotDisposed(method: string): void {
    if (this.#state === 'disposed') {
      throw new InvalidStateError(`VideoSession.${method} called on a disposed session`);
    }
  }

  /**
   * If a `propagate()` iterator is currently active, cancel it and wait for
   * it to settle (§6.1: "post `'cancel'` on the active port → await its
   * `'done'` → state='idle'"). No-op when idle. Does NOT touch `#epoch` —
   * callers bump it themselves before invoking this.
   */
  async #cancelActiveIterator(): Promise<void> {
    const active = this.#activeIterator;
    if (!active) return;
    await active.forceCancel();
    // onSettled (wired in #startPropagation) already cleared #activeIterator
    // and flipped #state back to 'idle' as a side effect of forceCancel
    // resolving; this call is a belt-and-suspenders no-op if it already ran.
    this.#activeIterator = undefined;
    if (this.#state === 'propagating') this.#state = 'idle';
  }

  /** Wrap a worker `MaskPayload` in the immutable public {@link MaskResult}. */
  #toMaskResult(payload: {
    objectId: number;
    score: number;
    width: number;
    height: number;
    binaryMask: ArrayBuffer;
  }): MaskResult {
    if (!this.#transform) {
      throw new InvalidStateError('VideoSession: internal — mask decoded before attachSource()');
    }
    return new MaskResultImpl({
      objectId: payload.objectId,
      score: payload.score,
      width: payload.width,
      height: payload.height,
      binaryMask: new Uint8Array(payload.binaryMask),
      transform: this.#transform,
    });
  }

  async attachSource(
    source: Blob | HTMLVideoElement,
  ): Promise<{ frameCount?: number; fps: number; width: number; height: number }> {
    this.#assertNotDisposed('attachSource');
    if (this.#state === 'propagating') {
      throw new InvalidStateError('VideoSession.attachSource called while propagate() is active');
    }
    // Only Blob/File sources are supported at M2; anything else (notably an
    // HTMLVideoElement) lands in M4. Checking for Blob also narrows the union
    // for attachVideoSource below.
    if (!(source instanceof Blob)) {
      throw new NotImplementedError(
        'attachSource(HTMLVideoElement), lands in M4 — pass a Blob/File at M2',
      );
    }
    if (this.#sourceInfo) {
      throw new InvalidStateError('VideoSession.attachSource: one source per session');
    }
    const info = await this.#engine.attachVideoSource(this.#sessionId, source);
    this.#sourceInfo = info;
    this.#transform = placeholderTransform(info);
    return { frameCount: info.frameCount, fps: info.fps, width: info.width, height: info.height };
  }

  async addObject(options: {
    frameIndex: number;
    prompts: Prompt[];
    objectId?: number;
  }): Promise<{ objectId: number; mask: MaskResult }> {
    this.#assertNotDisposed('addObject');
    if (this.#state === 'propagating') {
      throw new InvalidStateError('VideoSession.addObject: finish or cancel propagate() first');
    }
    if (!this.#sourceInfo) {
      throw new InvalidStateError('VideoSession.addObject called before attachSource() succeeded');
    }
    const result = await this.#engine.addVideoObject(this.#sessionId, {
      frameIndex: options.frameIndex,
      prompts: options.prompts,
      objectId: options.objectId,
      epoch: this.#epoch,
    });
    return { objectId: result.objectId, mask: this.#toMaskResult(result.mask) };
  }

  async refineObject(objectId: number, frameIndex: number, prompts: Prompt[]): Promise<MaskResult> {
    this.#assertNotDisposed('refineObject');
    if (!this.#sourceInfo) {
      throw new InvalidStateError(
        'VideoSession.refineObject called before attachSource() succeeded',
      );
    }
    this.#epoch++;
    if (this.#state === 'propagating') {
      await this.#cancelActiveIterator();
    }
    const result = await this.#engine.refineVideoObject(this.#sessionId, {
      objectId,
      frameIndex,
      prompts,
      epoch: this.#epoch,
    });
    return this.#toMaskResult(result.mask);
  }

  /**
   * Synchronous per the public {@link VideoSession.removeObject} contract:
   * the epoch bump (and any active-iterator cancel) happens synchronously
   * where possible, the worker RPC is dispatched fire-and-forget.
   */
  removeObject(objectId: number): void {
    this.#assertNotDisposed('removeObject');
    this.#epoch++;
    const dispatch = () =>
      void Promise.resolve(this.#engine.removeVideoObject(this.#sessionId, objectId)).catch(
        () => {},
      );
    if (this.#state === 'propagating') {
      void this.#cancelActiveIterator().then(dispatch);
    } else {
      dispatch();
    }
  }

  propagate(options?: {
    startFrame?: number;
    endFrame?: number;
    direction?: 'forward' | 'backward';
    signal?: AbortSignal;
  }): AsyncIterableIterator<FramePropagationResult> {
    if (this.#state === 'disposed') {
      return poisonedIterator(
        new InvalidStateError('VideoSession.propagate called on a disposed session'),
      );
    }
    if (this.#state === 'propagating') {
      return poisonedIterator(
        new InvalidStateError(
          'VideoSession.propagate: a previous propagate() iterator is still active — ' +
            'finish or cancel it first',
        ),
      );
    }
    if (!this.#sourceInfo || !this.#transform) {
      return poisonedIterator(
        new InvalidStateError('VideoSession.propagate called before attachSource() succeeded'),
      );
    }
    if (options?.direction === 'backward') {
      return poisonedIterator(
        new NotImplementedError('propagate({direction:"backward"}), lands in M3'),
      );
    }

    const epoch = this.#epoch;
    const startFrame = options?.startFrame ?? 0;
    const endFrame = options?.endFrame ?? this.#sourceInfo.frameCount;

    const channel = new MessageChannel();
    const iterator = createPropagationIterator({
      port: channel.port1,
      capturedEpoch: epoch,
      currentEpoch: () => this.#epoch,
      transform: this.#transform,
      prefetch: PREFETCH,
      signal: options?.signal,
      onSettled: () => {
        if (this.#activeIterator === iterator) {
          this.#activeIterator = undefined;
          if (this.#state === 'propagating') this.#state = 'idle';
        }
      },
    });

    // §6.1: "returns iterator; state→'propagating' on first next()" in
    // spirit — implemented slightly stricter (flipped here, at propagate()
    // call time) so a second propagate() before the consumer ever calls
    // next() is ALSO correctly poisoned, not silently allowed. Flagged for
    // the consolidating gate as a deliberate, conservative deviation.
    this.#state = 'propagating';
    this.#activeIterator = iterator;

    const req: PropagateRequest = { startFrame, endFrame, epoch, prefetch: PREFETCH };
    const port2 = channel.port2;
    this.#engine
      .propagateVideo(this.#sessionId, req, Comlink.transfer(port2, [port2]))
      .catch((err: unknown) => {
        iterator.failWithError(err);
      });

    return iterator;
  }

  /**
   * Synchronous per the public {@link VideoSession.reset} contract; RPC
   * dispatched fire-and-forget. Per §6.1, unlike every other mutator,
   * `reset()` on a disposed session is a no-op (not an error).
   */
  reset(): void {
    if (this.#state === 'disposed') return;
    this.#epoch++;
    const dispatch = () =>
      void Promise.resolve(this.#engine.resetVideoSession(this.#sessionId)).catch(() => {});
    if (this.#state === 'propagating') {
      void this.#cancelActiveIterator().then(dispatch);
    } else {
      dispatch();
    }
  }

  /** Idempotent; further calls after dispose throw {@link InvalidStateError} (§6.1). */
  dispose(): void {
    if (this.#state === 'disposed') return;
    const wasPropagating = this.#state === 'propagating';
    this.#state = 'disposed';
    const dispatch = () =>
      void Promise.resolve(this.#engine.closeVideoSession(this.#sessionId)).catch(() => {});
    if (wasPropagating && this.#activeIterator) {
      void this.#activeIterator.forceCancel().then(dispatch);
    } else {
      dispatch();
    }
  }
}
