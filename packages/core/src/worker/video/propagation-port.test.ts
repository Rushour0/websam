import { describe, expect, it, vi } from 'vitest';
import { OutOfMemoryError } from '../../errors.js';
import type { MaskPayload } from '../protocol.js';
import { runPropagationPort } from './propagation-port.js';
import type { PropagationFrame, PropagationPushMessage } from './propagation-port.js';

function mask(objectId: number): MaskPayload {
  return { objectId, score: 1, width: 1, height: 1, binaryMask: new ArrayBuffer(1) };
}

function frame(frameIndex: number): PropagationFrame {
  return { frameIndex, timestampUs: frameIndex * 1000, epoch: 0, masks: [mask(1)] };
}

/** Collects every message posted on `port`'s far side. */
function collector(port: MessagePort): { messages: PropagationPushMessage[] } {
  const state = { messages: [] as PropagationPushMessage[] };
  port.onmessage = (ev: MessageEvent<PropagationPushMessage>) => state.messages.push(ev.data);
  return state;
}

describe('runPropagationPort', () => {
  it('emits a frame immediately when credits are already granted (prefetch)', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    runPropagationPort(port2, async (emit) => {
      await emit(frame(0));
    });

    port1.postMessage({ type: 'pull', credits: 4 });
    await vi.waitFor(() => expect(rx.messages.length).toBeGreaterThan(0));

    expect(rx.messages[0]).toMatchObject({ type: 'frame', frameIndex: 0 });
  });

  it('stalls emit at zero credits and resumes on the next pull (backpressure)', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    const order: string[] = [];
    runPropagationPort(port2, async (emit) => {
      await emit(frame(0));
      order.push('frame0 emitted');
      await emit(frame(1));
      order.push('frame1 emitted');
    });

    port1.postMessage({ type: 'pull', credits: 1 });
    await vi.waitFor(() => expect(rx.messages).toHaveLength(1));
    expect(order).toEqual(['frame0 emitted']);

    // No more credits: frame 1 must be stalled, not yet posted.
    await new Promise((r) => setTimeout(r, 20));
    expect(rx.messages).toHaveLength(1);
    expect(order).toEqual(['frame0 emitted']);

    port1.postMessage({ type: 'pull', credits: 1 });
    await vi.waitFor(() => expect(rx.messages.filter((m) => m.type === 'frame')).toHaveLength(2));
    expect(rx.messages[1]).toMatchObject({ type: 'frame', frameIndex: 1 });

    await vi.waitFor(() => expect(rx.messages.some((m) => m.type === 'done')).toBe(true));
    expect(rx.messages.at(-1)).toEqual({ type: 'done', framesEmitted: 2, cancelled: false });
  });

  it('never lets more than `prefetch` frames be posted unconsumed', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    runPropagationPort(port2, async (emit) => {
      for (let i = 0; i < 10; i++) await emit(frame(i));
    });

    port1.postMessage({ type: 'pull', credits: 4 });
    await new Promise((r) => setTimeout(r, 30));
    // Only 4 credits granted: exactly 4 frames posted, loop stalled on the 5th.
    expect(rx.messages.filter((m) => m.type === 'frame')).toHaveLength(4);

    port1.postMessage({ type: 'pull', credits: 6 });
    await vi.waitFor(() => expect(rx.messages.filter((m) => m.type === 'frame')).toHaveLength(10));
    await vi.waitFor(() => expect(rx.messages.at(-1)?.type).toBe('done'));
  });

  it('cancel mid-stream finishes the in-flight frame WITHOUT emitting it, then posts done{cancelled:true}', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    let sawCancelledInLoop = false;
    runPropagationPort(port2, async (emit, isCancelled) => {
      await emit(frame(0));
      // Simulate the stall: no more credits granted, so this call blocks
      // until 'cancel' arrives.
      await emit(frame(1));
      sawCancelledInLoop = isCancelled();
    });

    port1.postMessage({ type: 'pull', credits: 1 });
    await vi.waitFor(() => expect(rx.messages).toHaveLength(1));

    port1.postMessage({ type: 'cancel' });
    await vi.waitFor(() => expect(rx.messages.some((m) => m.type === 'done')).toBe(true));

    // frame(1) was never posted — only the 'done' message follows frame 0.
    expect(rx.messages).toEqual([
      { type: 'frame', frameIndex: 0, timestampUs: 0, epoch: 0, masks: [mask(1)] },
      { type: 'done', framesEmitted: 1, cancelled: true },
    ]);
    expect(sawCancelledInLoop).toBe(true);
  });

  it('isCancelled() observes cancel even between frames (no stall required)', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    let iterations = 0;
    runPropagationPort(port2, async (emit, isCancelled) => {
      port1.postMessage({ type: 'cancel' });
      // Give the port's onmessage a turn.
      await new Promise((r) => setTimeout(r, 5));
      while (!isCancelled() && iterations < 100) {
        iterations++;
      }
    });

    await vi.waitFor(() => expect(rx.messages.some((m) => m.type === 'done')).toBe(true));
    expect(rx.messages.at(-1)).toEqual({ type: 'done', framesEmitted: 0, cancelled: true });
  });

  it('a thrown error posts an envelope and closes the port (no done follows)', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    runPropagationPort(port2, async () => {
      throw new OutOfMemoryError('ring alloc failed');
    });

    await vi.waitFor(() => expect(rx.messages).toHaveLength(1));
    expect(rx.messages[0]).toMatchObject({
      type: 'error',
      error: { name: 'OutOfMemoryError', message: 'ring alloc failed', websamCode: 'OUT_OF_MEMORY' },
    });
  });

  it('transfers every mask.binaryMask (detached on the sender side after postMessage)', async () => {
    const { port1, port2 } = new MessageChannel();
    port1.start();
    const buf = new ArrayBuffer(4);
    const f: PropagationFrame = { frameIndex: 0, timestampUs: 0, epoch: 0, masks: [{ objectId: 1, score: 1, width: 2, height: 2, binaryMask: buf }] };

    const received = new Promise<PropagationPushMessage>((resolve) => {
      port1.onmessage = (ev) => resolve(ev.data);
    });
    runPropagationPort(port2, async (emit) => {
      await emit(f);
    });
    port1.postMessage({ type: 'pull', credits: 1 });

    const msg = await received;
    expect(buf.byteLength).toBe(0); // detached — proves it was TRANSFERRED, not cloned.
    expect(msg).toMatchObject({ type: 'frame', frameIndex: 0 });
  });

  it('closes the port after done (no further onmessage handling)', async () => {
    const { port1, port2 } = new MessageChannel();
    const rx = collector(port1);
    port1.start();

    runPropagationPort(port2, async () => {});
    await vi.waitFor(() => expect(rx.messages.at(-1)?.type).toBe('done'));

    // port2 is closed; posting from port1 after this should not throw, and
    // must not resurrect a second 'done'.
    expect(() => port1.postMessage({ type: 'pull', credits: 1 })).not.toThrow();
    await new Promise((r) => setTimeout(r, 10));
    expect(rx.messages).toHaveLength(1);
  });
});
