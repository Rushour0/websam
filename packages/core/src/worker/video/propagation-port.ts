/**
 * The propagation pull-credit stream — worker side (docs/m2-internal-contracts.md
 * §5.2, NORMATIVE, shared verbatim with the video-session agent's main-side
 * consumer).
 *
 * WHY A RAW MESSAGEPORT: Comlink async-iterator proxying round-trips every
 * `next()` through the RPC layer with no backpressure and no transferable
 * batching. A dedicated port with credits gives both: the worker never
 * decodes/runs more than `prefetch` frames ahead of what the main thread has
 * actually consumed, and every `masks[i].binaryMask` transfers zero-copy.
 *
 * Protocol:
 * - `{type:'pull', credits}` — main thread grants `credits` more frames.
 *   `credits` accumulates; `0` credits held means {@link runPropagationPort}'s
 *   `emit` callback (passed into the engine's `propagate` loop) stays
 *   pending, which stalls the engine BEFORE it decodes the next frame (the
 *   documented stall semantics — decode pressure never runs ahead of
 *   consumption).
 * - `{type:'cancel'}` — finish the in-flight frame WITHOUT emitting it, post
 *   `{type:'done', cancelled:true}`, close the port. The engine's
 *   `isCancelled()` also observes this so it can bail between frames without
 *   waiting on a stalled `emit`.
 * - Loop completion (no error, not cancelled) → `{type:'done', cancelled:false}`.
 * - Any thrown error → `{type:'error', error}` (envelope-serialized via
 *   `error-envelope.ts`, reused directly since this channel is NOT Comlink),
 *   then the port closes. No `'done'` follows an `'error'`.
 */

import { errorToEnvelope } from '../error-envelope.js';
import type { MaskPayload } from '../protocol.js';

/** One propagated frame's masks, in object insertion order. */
export interface PropagationFrame {
  frameIndex: number;
  timestampUs: number;
  epoch: number;
  masks: MaskPayload[];
}

/** Messages the MAIN thread posts on the propagation MessagePort. */
export type PropagationPullMessage = { type: 'pull'; credits: number } | { type: 'cancel' };

/** Messages the WORKER posts on the propagation MessagePort. */
export type PropagationPushMessage =
  | { type: 'frame'; frameIndex: number; timestampUs: number; epoch: number; masks: MaskPayload[] }
  | { type: 'done'; framesEmitted: number; cancelled: boolean }
  | { type: 'error'; error: import('../error-envelope.js').ErrorEnvelope };

/**
 * Drive one propagation loop over `port`. `run` is the engine's
 * `VideoEngine.propagate` bound to a request; it is called with an `emit`
 * that resolves once a credit is available (posting the frame and consuming
 * one credit) and an `isCancelled` poll for between-frame cancellation.
 *
 * Closes `port` exactly once, on `'done'` or `'error'`. Never throws —
 * every failure path is reported over the port so the caller (worker-entry /
 * engine.ts) can fire-and-forget this call.
 */
export function runPropagationPort(
  port: MessagePort,
  run: (
    emit: (frame: PropagationFrame) => Promise<void>,
    isCancelled: () => boolean,
  ) => Promise<void>,
): void {
  let credits = 0;
  let cancelled = false;
  let framesEmitted = 0;
  let closed = false;
  /** Resolves the currently-pending `emit` once credits/cancel state changes. */
  let wake: (() => void) | undefined;

  const close = (): void => {
    if (closed) return;
    closed = true;
    port.onmessage = null;
    port.close();
  };

  port.onmessage = (ev: MessageEvent<PropagationPullMessage>): void => {
    const msg = ev.data;
    if (msg.type === 'pull') {
      credits += msg.credits;
    } else {
      cancelled = true;
    }
    wake?.();
  };

  const isCancelled = (): boolean => cancelled;

  const emit = (frame: PropagationFrame): Promise<void> => {
    if (cancelled) return Promise.resolve(); // finish in-flight work without posting.
    if (credits > 0) {
      credits -= 1;
      const push: PropagationPushMessage = {
        type: 'frame',
        frameIndex: frame.frameIndex,
        timestampUs: frame.timestampUs,
        epoch: frame.epoch,
        masks: frame.masks,
      };
      framesEmitted += 1;
      port.postMessage(
        push,
        frame.masks.map((m) => m.binaryMask),
      );
      return Promise.resolve();
    }
    // Zero credits: stall until a 'pull' (more credits) or 'cancel' arrives.
    return new Promise<void>((resolve) => {
      const check = (): void => {
        if (cancelled || credits > 0) {
          wake = undefined;
          resolve(emit(frame));
        }
      };
      wake = check;
    });
  };

  run(emit, isCancelled)
    .then(() => {
      const push: PropagationPushMessage = { type: 'done', framesEmitted, cancelled };
      port.postMessage(push);
      close();
    })
    .catch((err: unknown) => {
      const push: PropagationPushMessage = { type: 'error', error: errorToEnvelope(err) };
      port.postMessage(push);
      close();
    });
}
