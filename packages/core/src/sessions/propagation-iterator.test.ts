import { describe, expect, it, vi } from 'vitest';
import { computeTransform } from '../coords.js';
import { EpochInvalidatedError, InvalidStateError } from '../errors.js';
import type {
  MaskPayload,
  PropagationPullMessage,
  PropagationPushMessage,
} from '../worker/protocol.js';
import { createPropagationIterator, type PropagationIteratorHandle } from './propagation-iterator.js';

const TRANSFORM = computeTransform(64, 48, 64, 'square-stretch');

function maskPayload(objectId: number, frameIndex: number): MaskPayload {
  const binaryMask = new Uint8Array(64 * 48);
  binaryMask[frameIndex % binaryMask.length] = 1;
  return {
    objectId,
    score: 0.9,
    width: 64,
    height: 48,
    binaryMask: binaryMask.buffer,
  };
}

/**
 * A scripted "worker" driving `port2`: honors pull credits, emits frames in
 * order, and answers `'cancel'` with a `{done, cancelled:true}` immediately
 * (finishing any in-flight "frame" first — modeled as simply not emitting
 * further frames once cancelled).
 */
class ScriptedWorker {
  readonly #port: MessagePort;
  readonly #frames: readonly { frameIndex: number; masks: MaskPayload[] }[];
  #credits = 0;
  #idx = 0;
  #cancelled = false;
  #doneSent = false;
  readonly onCancel: ReturnType<typeof vi.fn>;
  readonly pulls: number[] = [];

  constructor(port: MessagePort, frames: readonly { frameIndex: number; masks: MaskPayload[] }[]) {
    this.#port = port;
    this.#frames = frames;
    this.onCancel = vi.fn();
    port.onmessage = (ev: MessageEvent<PropagationPullMessage>) => this.#handle(ev.data);
  }

  #handle(msg: PropagationPullMessage): void {
    if (msg.type === 'cancel') {
      this.#cancelled = true;
      this.onCancel();
      this.#sendDone(true);
      return;
    }
    this.pulls.push(msg.credits);
    this.#credits += msg.credits;
    this.#pump();
  }

  #pump(): void {
    while (!this.#cancelled && this.#credits > 0 && this.#idx < this.#frames.length) {
      const f = this.#frames[this.#idx++]!;
      this.#credits--;
      const push: PropagationPushMessage = {
        type: 'frame',
        frameIndex: f.frameIndex,
        timestampUs: f.frameIndex * 33_333,
        epoch: 0,
        masks: f.masks,
      };
      this.#port.postMessage(push, f.masks.map((m) => m.binaryMask));
    }
    if (!this.#cancelled && this.#idx >= this.#frames.length) {
      this.#sendDone(false);
    }
  }

  #sendDone(cancelled: boolean): void {
    if (this.#doneSent) return;
    this.#doneSent = true;
    const push: PropagationPushMessage = { type: 'done', framesEmitted: this.#idx, cancelled };
    this.#port.postMessage(push);
  }
}

function makeIterator(
  frames: readonly { frameIndex: number; masks: MaskPayload[] }[],
  overrides: {
    capturedEpoch?: number;
    currentEpoch?: () => number;
    signal?: AbortSignal;
  } = {},
): {
  iterator: PropagationIteratorHandle;
  worker: ScriptedWorker;
  onSettled: ReturnType<typeof vi.fn>;
  /** Every message the iterator itself POSTED on port1 (reliable — no round-trip delivery
   *  wait needed, unlike `worker.pulls` which only records what the worker actually RECEIVED). */
  sent: PropagationPullMessage[];
} {
  const channel = new MessageChannel();
  const worker = new ScriptedWorker(channel.port2, frames);
  const onSettled = vi.fn();
  const sent: PropagationPullMessage[] = [];
  const realPost = channel.port1.postMessage.bind(channel.port1);
  channel.port1.postMessage = ((message: PropagationPullMessage, transfer?: Transferable[]) => {
    sent.push(message);
    return realPost(message, transfer as Transferable[]);
  }) as typeof channel.port1.postMessage;
  const iterator = createPropagationIterator({
    port: channel.port1,
    capturedEpoch: overrides.capturedEpoch ?? 0,
    currentEpoch: overrides.currentEpoch ?? (() => 0),
    transform: TRANSFORM,
    prefetch: 4,
    signal: overrides.signal,
    onSettled,
  });
  return { iterator, worker, onSettled, sent };
}

describe('createPropagationIterator — normal frame flow', () => {
  it('yields frames in order, wraps masks, and settles done on exhaustion', async () => {
    const frames = [
      { frameIndex: 0, masks: [maskPayload(1, 0)] },
      { frameIndex: 1, masks: [maskPayload(1, 1)] },
      { frameIndex: 2, masks: [maskPayload(1, 2)] },
    ];
    const { iterator, onSettled } = makeIterator(frames);

    const seen: number[] = [];
    for await (const frame of iterator) {
      seen.push(frame.frameIndex);
      expect(frame.masks).toHaveLength(1);
      expect(frame.masks[0]!.objectId).toBe(1);
      expect(frame.masks[0]!.toBinary()).toHaveLength(64 * 48);
    }
    expect(seen).toEqual([0, 1, 2]);
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('done');
  });

  it('posts the initial prefetch credit grant then one credit per consumed frame', async () => {
    const frames = [
      { frameIndex: 0, masks: [maskPayload(1, 0)] },
      { frameIndex: 1, masks: [maskPayload(1, 1)] },
    ];
    const { iterator, sent } = makeIterator(frames);
    await iterator.next();
    await iterator.next();
    await iterator.next(); // drains to done
    expect(sent).toEqual([
      { type: 'pull', credits: 4 },
      { type: 'pull', credits: 1 },
      { type: 'pull', credits: 1 },
    ]);
  });

  it('repeated next() after done is idempotent ({done:true})', async () => {
    const { iterator } = makeIterator([]);
    await expect(iterator.next()).resolves.toEqual({ done: true, value: undefined });
    await expect(iterator.next()).resolves.toEqual({ done: true, value: undefined });
  });
});

describe('createPropagationIterator — epoch invalidation', () => {
  it('throws EpochInvalidatedError on next() once the epoch has moved, cancelling the worker', async () => {
    // 6 frames > prefetch(4): the scripted worker only emits 4 up front (it
    // spends all its initial credit) and genuinely has 2 left in-flight when
    // we cancel — unlike a <= prefetch clip, which would already be fully
    // drained (including a NATURAL 'done') before the epoch bump, making
    // this test racy against that unrelated completion.
    let epoch = 0;
    const frames = Array.from({ length: 6 }, (_, frameIndex) => ({
      frameIndex,
      masks: [maskPayload(1, frameIndex)],
    }));
    const { iterator, worker, onSettled } = makeIterator(frames, {
      capturedEpoch: 0,
      currentEpoch: () => epoch,
    });

    const first = await iterator.next();
    expect(first.done).toBe(false);

    epoch = 1; // simulate refineObject bumping the session epoch
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);

    expect(worker.onCancel).toHaveBeenCalledOnce();
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('cancelled');

    // Further next() calls keep throwing (never silently downgrade to
    // {done:true} — that would be the "silent stop" the contract forbids).
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);
  });

  it('discards an already-in-flight frame for a stale epoch instead of yielding it', async () => {
    let epoch = 0;
    const frames = [{ frameIndex: 0, masks: [maskPayload(1, 0)] }];
    const { iterator } = makeIterator(frames, { capturedEpoch: 0, currentEpoch: () => epoch });

    // Bump epoch BEFORE the very first next() — even the first frame must be discarded.
    epoch = 1;
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);
  });
});

describe('createPropagationIterator — forceCancel (video-session external cancel hook)', () => {
  it('cancels an active loop with no consumer next() pending', async () => {
    const frames = [
      { frameIndex: 0, masks: [maskPayload(1, 0)] },
      { frameIndex: 1, masks: [maskPayload(1, 1)] },
    ];
    const { iterator, worker, onSettled } = makeIterator(frames);
    await iterator.forceCancel();
    expect(worker.onCancel).toHaveBeenCalledOnce();
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('cancelled');
  });

  it('unblocks a concurrently pending next() without a race (single shared read)', async () => {
    // No frames ever arrive (the scripted worker naturally completes right
    // after the initial pull for an empty clip) — the point of this test is
    // that `next()`'s own wait and `forceCancel()`'s drain share ONE
    // in-flight port read (never two competing waiters, which would
    // deadlock the loser): both settle, `next()` resolves `{done:true}`, and
    // the worker sees both the pull and the cancel it posted.
    const { iterator, worker, onSettled } = makeIterator([]);
    const pending = iterator.next();
    await iterator.forceCancel();
    await expect(pending).resolves.toEqual({ done: true, value: undefined });
    expect(worker.onCancel).toHaveBeenCalledOnce();
    expect(onSettled).toHaveBeenCalledOnce();
  });

  it('is a safe no-op once already closed', async () => {
    const { iterator } = makeIterator([]);
    await iterator.next(); // drains to done
    await expect(iterator.forceCancel()).resolves.toBeUndefined();
  });
});

describe('createPropagationIterator — return()/break releases backpressure credits', () => {
  it('return() posts cancel, drains to done, and stops pulling further credits', async () => {
    // 6 frames > prefetch(4), same rationale as the epoch test above: keeps
    // the worker genuinely mid-stream (no natural 'done' yet) at break time.
    const frames = Array.from({ length: 6 }, (_, frameIndex) => ({
      frameIndex,
      masks: [maskPayload(1, frameIndex)],
    }));
    const { iterator, worker, onSettled } = makeIterator(frames);

    let count = 0;
    for await (const _frame of iterator) {
      count++;
      if (count === 1) break; // triggers iterator.return()
    }
    expect(count).toBe(1);
    expect(worker.onCancel).toHaveBeenCalledOnce();
    // Only the initial prefetch pull (4) + the credit for the one consumed frame (1) — no
    // further pulls after break/cancel.
    expect(worker.pulls).toEqual([4, 1]);
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('cancelled');
  });

  it('throw() drains to done and rethrows the given error', async () => {
    const { iterator, worker } = makeIterator([{ frameIndex: 0, masks: [maskPayload(1, 0)] }]);
    const boom = new Error('consumer blew up');
    // throw() is always provided by createPropagationIterator (optional on the
    // AsyncIterableIterator surface, guaranteed by this implementation).
    await expect(iterator.throw!(boom)).rejects.toBe(boom);
    expect(worker.onCancel).toHaveBeenCalledOnce();
  });
});

describe('createPropagationIterator — abort signal', () => {
  it('an already-aborted signal rejects next() with the abort reason', async () => {
    const controller = new AbortController();
    controller.abort(new DOMException('nope', 'AbortError'));
    const { iterator, worker } = makeIterator([{ frameIndex: 0, masks: [maskPayload(1, 0)] }], {
      signal: controller.signal,
    });
    await expect(iterator.next()).rejects.toThrow('nope');
    expect(worker.onCancel).toHaveBeenCalledOnce();
  });

  it('aborting mid-wait rejects the pending next() and cancels the worker', async () => {
    const controller = new AbortController();
    const { iterator, worker } = makeIterator([], { signal: controller.signal });
    const pending = iterator.next();
    controller.abort(new DOMException('stop', 'AbortError'));
    await expect(pending).rejects.toThrow('stop');
    expect(worker.onCancel).toHaveBeenCalledOnce();
  });
});

describe('createPropagationIterator — worker error', () => {
  it('rehydrates a WebsamError envelope and throws it from next()', async () => {
    const channel = new MessageChannel();
    channel.port2.onmessage = (ev: MessageEvent<PropagationPullMessage>) => {
      if (ev.data.type === 'pull') {
        const push: PropagationPushMessage = {
          type: 'error',
          error: { name: 'InvalidStateError', message: 'boom', websamCode: 'INVALID_STATE' },
        };
        channel.port2.postMessage(push);
      }
    };
    const onSettled = vi.fn();
    const iterator = createPropagationIterator({
      port: channel.port1,
      capturedEpoch: 0,
      currentEpoch: () => 0,
      transform: TRANSFORM,
      prefetch: 4,
      onSettled,
    });
    await expect(iterator.next()).rejects.toThrow(InvalidStateError);
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('error');
  });
});

describe('createPropagationIterator — failWithError', () => {
  it('fails a pending next() directly, with no port round-trip', async () => {
    const { iterator, onSettled } = makeIterator([]);
    const pending = iterator.next();
    const boom = new Error('propagateVideo RPC rejected');
    iterator.failWithError(boom);
    await expect(pending).rejects.toBe(boom);
    expect(onSettled).toHaveBeenCalledExactlyOnceWith('error');
  });

  it('is a no-op once already closed', async () => {
    const { iterator } = makeIterator([]);
    await iterator.next(); // done
    expect(() => iterator.failWithError(new Error('too late'))).not.toThrow();
  });
});
