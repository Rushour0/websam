/**
 * Client side of the pull-credit propagation port
 * (docs/m2-internal-contracts.md §5.2, §6.2 — NORMATIVE, shared verbatim with
 * the worker-video agent's `src/worker/video/propagation-port.ts`).
 *
 * One `MessagePort` (the main-thread half of a `MessageChannel`; the other
 * half was transferred to the worker via `RemoteEngine.propagateVideo`)
 * carries the whole propagation stream: the main thread posts
 * {@link PropagationPullMessage} credits, the worker posts
 * {@link PropagationPushMessage} frames/done/error. This module wraps that
 * port in the `AsyncIterableIterator` the public {@link VideoSession.propagate}
 * contract promises.
 *
 * Epoch handling (normative, docs/m2-internal-contracts.md §6.1/§6.2): the
 * MAIN THREAD is epoch-authoritative. Every `next()` call first compares the
 * session's live epoch counter against the epoch captured at iterator
 * creation; a mismatch (a `refineObject`/`removeObject`/`reset` bumped it)
 * cancels the worker-side loop, drains to `'done'`, and THROWS
 * {@link EpochInvalidatedError} — never a silent stop, and any frame already
 * received for a stale epoch is discarded, not yielded.
 *
 * `VideoSessionImpl` needs to cancel the loop THE MOMENT it bumps the epoch
 * (refine/remove/reset must not wait for the consumer to happen to call
 * `next()` again — see the §6.1 state table), even while a `next()` call is
 * already in flight and blocked waiting for the worker. {@link
 * PropagationIteratorHandle.forceCancel} is the additive hook for that: it is
 * NOT part of the frozen `AsyncIterableIterator` surface `propagate()`
 * returns to callers, but `createPropagationIterator`'s return type is
 * structurally richer than that surface so `video-session.ts` can hold onto
 * it. Internally, `forceCancel()` and an in-flight `next()` share ONE
 * outstanding "read the next port message" promise (`#pending`) so they
 * never race for it — both wake on the same resolution and each applies its
 * own (idempotent, guarded) follow-up.
 */

import type { CoordinateTransform } from '../coords.js';
import { EpochInvalidatedError } from '../errors.js';
import { MaskResultImpl } from '../masks/mask-result.js';
import type { FramePropagationResult } from '../segmenter.js';
import { envelopeToError } from '../worker/error-envelope.js';
import type { PropagationPullMessage, PropagationPushMessage } from '../worker/protocol.js';

/** Constructor payload for {@link createPropagationIterator}. */
export interface PropagationIteratorInit {
  /** `port1` of the `MessageChannel`; `port2` was transferred to the worker. */
  port: MessagePort;
  /** The session epoch captured when `propagate()` was called. */
  capturedEpoch: number;
  /** Reads the session's LIVE epoch counter (bumped by refine/remove/reset). */
  currentEpoch: () => number;
  /** Stamped on every {@link MaskResultImpl} this iterator yields. */
  transform: CoordinateTransform;
  /** Initial credit grant (4, per §5.2). */
  prefetch: number;
  signal?: AbortSignal;
  /** Called exactly once, when the iterator settles for good. */
  onSettled: (reason: 'done' | 'cancelled' | 'error') => void;
}

/**
 * What {@link createPropagationIterator} actually returns: the public
 * `AsyncIterableIterator` surface plus the internal `forceCancel` hook.
 * Structurally assignable to `AsyncIterableIterator<FramePropagationResult>`
 * — callers of the public {@link VideoSession.propagate} contract never see
 * (or need) the extra member.
 */
export interface PropagationIteratorHandle extends AsyncIterableIterator<FramePropagationResult> {
  /**
   * Cancel this iterator's worker loop RIGHT NOW, independent of whether a
   * consumer `next()` call is currently pending. Posts `'cancel'` (once —
   * idempotent against a concurrently-in-flight `next()`-triggered cancel),
   * awaits the worker's `'done'`/`'error'` reply, and resolves once the
   * iterator has settled. Safe to call multiple times / already-closed.
   */
  forceCancel(): Promise<void>;
  /**
   * Fail this iterator directly (no port round-trip) — for the one case
   * where there is no worker loop to cancel yet: `RemoteEngine.propagateVideo`
   * itself rejected before scheduling the loop. The next (or in-flight)
   * `next()` call throws `err`. No-op once closed.
   */
  failWithError(err: unknown): void;
}

const DONE_RESULT: IteratorResult<FramePropagationResult> = { done: true, value: undefined };

/** One item flowing through the iterator's internal event queue. */
type QueueItem =
  | { kind: 'message'; message: PropagationPushMessage }
  | { kind: 'abort' }
  | { kind: 'fail'; err: unknown };

class PropagationIterator implements PropagationIteratorHandle {
  readonly #port: MessagePort;
  readonly #capturedEpoch: number;
  readonly #currentEpoch: () => number;
  readonly #transform: CoordinateTransform;
  readonly #prefetch: number;
  readonly #signal: AbortSignal | undefined;
  readonly #onSettled: (reason: 'done' | 'cancelled' | 'error') => void;

  #started = false;
  /** True once the iterator has terminated for good (done/error/cancelled-drained). */
  #closed = false;
  /**
   * True iff the epoch was ALREADY stale at the moment this iterator closed
   * via {@link #cancelAndDrain} (covers external {@link forceCancel} closing
   * it out from under a consumer who has not called `next()` again yet).
   * Once set, EVERY subsequent `next()` throws `EpochInvalidatedError`
   * instead of silently returning `{done:true}` — the §6.2 "never a silent
   * stop" rule applies even when nothing was awaiting `next()` at the moment
   * of cancellation.
   */
  #epochStaleAtClose = false;
  #settledCalled = false;
  #cancelPosted = false;

  #queue: QueueItem[] = [];
  /** The ONE outstanding "read the next port/abort event" promise, shared by
   *  every concurrent caller (`next()`'s own wait AND `forceCancel()`'s
   *  drain) so they never register two competing waiters. */
  #pending: Promise<QueueItem> | undefined;
  #resolvePending: ((item: QueueItem) => void) | undefined;

  #draining: Promise<void> | undefined;

  constructor(init: PropagationIteratorInit) {
    this.#port = init.port;
    this.#capturedEpoch = init.capturedEpoch;
    this.#currentEpoch = init.currentEpoch;
    this.#transform = init.transform;
    this.#prefetch = init.prefetch;
    this.#signal = init.signal;
    this.#onSettled = init.onSettled;

    this.#port.onmessage = (ev: MessageEvent<PropagationPushMessage>) => {
      this.#push({ kind: 'message', message: ev.data });
    };
    this.#signal?.addEventListener('abort', () => this.#push({ kind: 'abort' }));
  }

  [Symbol.asyncIterator](): AsyncIterableIterator<FramePropagationResult> {
    return this;
  }

  #push(item: QueueItem): void {
    if (this.#resolvePending) {
      const resolve = this.#resolvePending;
      this.#resolvePending = undefined;
      this.#pending = undefined;
      resolve(item);
    } else {
      this.#queue.push(item);
    }
  }

  /** Returns the shared in-flight "next port event" promise, creating one if none is outstanding. */
  #nextItem(): Promise<QueueItem> {
    if (this.#pending) return this.#pending;
    const head = this.#queue.shift();
    if (head) return Promise.resolve(head);
    this.#pending = new Promise((resolve) => {
      this.#resolvePending = resolve;
    });
    return this.#pending;
  }

  #post(message: PropagationPullMessage): void {
    this.#port.postMessage(message);
  }

  #settleOnce(reason: 'done' | 'cancelled' | 'error'): void {
    if (this.#settledCalled) return;
    this.#settledCalled = true;
    this.#onSettled(reason);
  }

  #epochStale(): boolean {
    return this.#currentEpoch() !== this.#capturedEpoch;
  }

  /**
   * Post `'cancel'` (once) and drain until the worker's guaranteed
   * `'done'`/`'error'` reply; queued/incoming `'frame'` messages in between
   * are discarded (their transferred buffers just become garbage — that IS
   * the backpressure credit release: no further `'pull'` is ever posted for
   * them). Single-flight: concurrent callers (an in-flight `next()`
   * discovering staleness AND an external {@link forceCancel}) share the
   * same drain.
   */
  #cancelAndDrain(): Promise<void> {
    if (this.#closed) return Promise.resolve();
    if (this.#draining) return this.#draining;
    this.#draining = (async () => {
      if (!this.#cancelPosted) {
        this.#cancelPosted = true;
        this.#post({ type: 'cancel' });
      }
      for (;;) {
        const item = await this.#nextItem();
        if (item.kind === 'abort') continue;
        if (item.kind === 'fail') {
          this.#settleOnce('error');
          break;
        }
        const message = item.message;
        if (message.type === 'done') {
          this.#settleOnce(message.cancelled ? 'cancelled' : 'done');
          break;
        }
        if (message.type === 'error') {
          this.#settleOnce('error');
          break;
        }
        // 'frame' while cancelling: drop it.
      }
      if (this.#epochStale()) this.#epochStaleAtClose = true;
      this.#closed = true;
      this.#port.close();
    })();
    return this.#draining;
  }

  async forceCancel(): Promise<void> {
    await this.#cancelAndDrain();
  }

  failWithError(err: unknown): void {
    if (this.#closed) return;
    this.#push({ kind: 'fail', err });
  }

  #toResult(message: Extract<PropagationPushMessage, { type: 'frame' }>): FramePropagationResult {
    return {
      frameIndex: message.frameIndex,
      timestampUs: message.timestampUs,
      masks: message.masks.map(
        (payload) =>
          new MaskResultImpl({
            objectId: payload.objectId,
            score: payload.score,
            width: payload.width,
            height: payload.height,
            binaryMask: new Uint8Array(payload.binaryMask),
            transform: this.#transform,
          }),
      ),
    };
  }

  async next(): Promise<IteratorResult<FramePropagationResult>> {
    if (this.#closed) {
      if (this.#epochStaleAtClose) {
        throw new EpochInvalidatedError(
          'VideoSession.propagate: the session epoch changed (refineObject/removeObject/reset) ' +
            'while this iterator was active — its results are stale',
        );
      }
      return DONE_RESULT;
    }

    for (;;) {
      if (this.#epochStale()) {
        await this.#cancelAndDrain();
        throw new EpochInvalidatedError(
          'VideoSession.propagate: the session epoch changed (refineObject/removeObject/reset) ' +
            'while this iterator was active — its results are stale',
        );
      }
      if (this.#signal?.aborted) {
        await this.#cancelAndDrain();
        throw this.#abortReason();
      }

      if (!this.#started) {
        this.#started = true;
        this.#post({ type: 'pull', credits: this.#prefetch });
      }

      const item = await this.#nextItem();
      if (item.kind === 'abort') continue; // loop back: re-checks signal.aborted above
      if (item.kind === 'fail') {
        this.#closed = true;
        this.#port.close();
        this.#settleOnce('error');
        throw item.err;
      }

      const message = item.message;
      if (message.type === 'frame') {
        if (this.#epochStale()) continue; // discard stale frame; loop re-checks epoch above
        this.#post({ type: 'pull', credits: 1 });
        return { done: false, value: this.#toResult(message) };
      }
      if (message.type === 'done') {
        this.#closed = true;
        this.#port.close();
        if (this.#epochStale()) {
          this.#epochStaleAtClose = true;
          this.#settleOnce('cancelled');
          throw new EpochInvalidatedError(
            'VideoSession.propagate: the session epoch changed while this iterator was ' +
              'finishing — its results are stale',
          );
        }
        this.#settleOnce(message.cancelled ? 'cancelled' : 'done');
        return DONE_RESULT;
      }
      // message.type === 'error'
      this.#closed = true;
      this.#port.close();
      this.#settleOnce('error');
      throw envelopeToError(message.error);
    }
  }

  async return(_value?: unknown): Promise<IteratorResult<FramePropagationResult>> {
    if (!this.#closed) await this.#cancelAndDrain();
    return DONE_RESULT;
  }

  async throw(err?: unknown): Promise<IteratorResult<FramePropagationResult>> {
    if (!this.#closed) await this.#cancelAndDrain();
    throw err;
  }

  #abortReason(): unknown {
    return (this.#signal?.reason as unknown) ?? new DOMException('propagate was aborted', 'AbortError');
  }
}

/**
 * Build the client-side iterator for one `propagate()` call. See the module
 * doc + docs/m2-internal-contracts.md §6.2 for the full behavior contract.
 */
export function createPropagationIterator(init: PropagationIteratorInit): PropagationIteratorHandle {
  return new PropagationIterator(init);
}
