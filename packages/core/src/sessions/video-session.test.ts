import { describe, expect, it, vi } from 'vitest';
import { EpochInvalidatedError, InvalidStateError, NotImplementedError } from '../errors.js';
import type { MaskResult } from '../segmenter.js';
import type {
  MaskPayload,
  PropagateRequest,
  PropagationPullMessage,
  PropagationPushMessage,
  VideoObjectResult,
  VideoSourceInfo,
} from '../worker/protocol.js';
import type { RemoteEngine } from './spawn-worker.js';
import { VideoSessionImpl } from './video-session.js';

const SOURCE_INFO: VideoSourceInfo = {
  frameCount: 10,
  fps: 30,
  width: 64,
  height: 48,
  durationUs: 333_333,
  codec: 'avc1.640028',
};

function maskPayload(objectId: number, frameIndex: number): MaskPayload {
  const binaryMask = new Uint8Array(64 * 48);
  binaryMask[frameIndex % binaryMask.length] = 1;
  return { objectId, score: 0.9, width: 64, height: 48, binaryMask: binaryMask.buffer };
}

/** Scripted worker-side responder for a `propagateVideo` port, mirroring the propagation-iterator tests. */
class ScriptedWorker {
  readonly #port: MessagePort;
  readonly #frames: readonly { frameIndex: number; masks: MaskPayload[] }[];
  #credits = 0;
  #idx = 0;
  #cancelled = false;
  #doneSent = false;
  readonly onCancel = vi.fn();

  constructor(port: MessagePort, frames: readonly { frameIndex: number; masks: MaskPayload[] }[]) {
    this.#port = port;
    this.#frames = frames;
    port.onmessage = (ev: MessageEvent<PropagationPullMessage>) => this.#handle(ev.data);
  }

  #handle(msg: PropagationPullMessage): void {
    if (msg.type === 'cancel') {
      this.#cancelled = true;
      this.onCancel();
      this.#sendDone(true);
      return;
    }
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
    if (!this.#cancelled && this.#idx >= this.#frames.length) this.#sendDone(false);
  }

  #sendDone(cancelled: boolean): void {
    if (this.#doneSent) return;
    this.#doneSent = true;
    const push: PropagationPushMessage = { type: 'done', framesEmitted: this.#idx, cancelled };
    this.#port.postMessage(push);
  }
}

/** Builds a fake `RemoteEngine` whose `propagateVideo` wires a {@link ScriptedWorker} onto the transferred port. */
function fakeEngine(overrides: Partial<RemoteEngine> = {}): {
  engine: RemoteEngine;
  workers: ScriptedWorker[];
} {
  const workers: ScriptedWorker[] = [];
  let nextObjectId = 1;
  const engine: RemoteEngine = {
    init: vi.fn(async () => {
      throw new Error('init: not under test');
    }),
    createSession: vi.fn(async () => {
      throw new Error('createSession: not under test');
    }),
    encodeImage: vi.fn(async () => {
      throw new Error('encodeImage: not under test');
    }),
    decode: vi.fn(async () => []),
    closeSession: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
    createVideoSession: vi.fn(async () => 1),
    attachVideoSource: vi.fn(async () => SOURCE_INFO),
    addVideoObject: vi.fn(async (_sessionId, req) => {
      const objectId = req.objectId ?? nextObjectId++;
      const result: VideoObjectResult = {
        objectId,
        epoch: req.epoch,
        mask: maskPayload(objectId, req.frameIndex),
      };
      return result;
    }),
    refineVideoObject: vi.fn(async (_sessionId, req) => {
      const result: VideoObjectResult = {
        objectId: req.objectId,
        epoch: req.epoch,
        mask: maskPayload(req.objectId, req.frameIndex),
      };
      return result;
    }),
    removeVideoObject: vi.fn(async () => {}),
    propagateVideo: vi.fn(async (_sessionId, req: PropagateRequest, port: MessagePort) => {
      const frameCount = req.endFrame - req.startFrame;
      const frames = Array.from({ length: frameCount }, (_, i) => ({
        frameIndex: req.startFrame + i,
        masks: [maskPayload(1, req.startFrame + i)],
      }));
      workers.push(new ScriptedWorker(port, frames));
    }),
    resetVideoSession: vi.fn(async () => {}),
    closeVideoSession: vi.fn(async () => {}),
    ...overrides,
  };
  return { engine, workers };
}

async function attached(overrides: Partial<RemoteEngine> = {}) {
  const { engine, workers } = fakeEngine(overrides);
  const session = new VideoSessionImpl(engine, 1);
  await session.attachSource(new Blob());
  return { session, engine, workers };
}

describe('VideoSessionImpl — attachSource', () => {
  it('resolves with the worker VideoSourceInfo fields', async () => {
    const { engine } = fakeEngine();
    const session = new VideoSessionImpl(engine, 1);
    await expect(session.attachSource(new Blob())).resolves.toEqual({
      frameCount: 10,
      fps: 30,
      width: 64,
      height: 48,
    });
    expect(engine.attachVideoSource).toHaveBeenCalledExactlyOnceWith(1, expect.any(Blob));
  });

  it('rejects a second attachSource with InvalidStateError', async () => {
    const { session } = await attached();
    await expect(session.attachSource(new Blob())).rejects.toThrow(InvalidStateError);
    await expect(session.attachSource(new Blob())).rejects.toThrow(/one source per session/);
  });

  it('HTMLVideoElement input throws NotImplementedError naming M4', async () => {
    const { engine } = fakeEngine();
    const session = new VideoSessionImpl(engine, 1);
    // jsdom is not loaded in the `unit` (node) project — feature-detect and
    // skip gracefully if HTMLVideoElement genuinely does not exist here.
    if (typeof HTMLVideoElement === 'undefined') return;
    const el = Object.create((HTMLVideoElement as unknown as { prototype: object }).prototype) as HTMLVideoElement;
    await expect(session.attachSource(el)).rejects.toThrow(NotImplementedError);
    await expect(session.attachSource(el)).rejects.toThrow(/M4/);
  });

  it('after dispose, throws InvalidStateError', async () => {
    const { engine } = fakeEngine();
    const session = new VideoSessionImpl(engine, 1);
    session.dispose();
    await expect(session.attachSource(new Blob())).rejects.toThrow(InvalidStateError);
  });
});

describe('VideoSessionImpl — addObject', () => {
  it('idle: dispatches addVideoObject and wraps the mask', async () => {
    const { session, engine } = await attached();
    const { objectId, mask } = await session.addObject({ frameIndex: 0, prompts: [] });
    expect(objectId).toBe(1);
    expect(mask.width).toBe(64);
    expect(mask.height).toBe(48);
    expect(engine.addVideoObject).toHaveBeenCalledExactlyOnceWith(1, {
      frameIndex: 0,
      prompts: [],
      objectId: undefined,
      epoch: 0,
    });
  });

  it('does NOT bump the epoch', async () => {
    const { session, engine } = await attached();
    await session.addObject({ frameIndex: 0, prompts: [] });
    await session.addObject({ frameIndex: 1, prompts: [] });
    const calls = vi.mocked(engine.addVideoObject).mock.calls;
    expect(calls[0]?.[1].epoch).toBe(0);
    expect(calls[1]?.[1].epoch).toBe(0);
  });

  it('before attachSource: InvalidStateError', async () => {
    const { engine } = fakeEngine();
    const session = new VideoSessionImpl(engine, 1);
    await expect(session.addObject({ frameIndex: 0, prompts: [] })).rejects.toThrow(
      InvalidStateError,
    );
  });

  it('while propagating: InvalidStateError naming finish-or-cancel', async () => {
    const { session } = await attached();
    session.propagate();
    await expect(session.addObject({ frameIndex: 0, prompts: [] })).rejects.toThrow(
      /finish or cancel propagate/,
    );
  });

  it('after dispose: InvalidStateError', async () => {
    const { session } = await attached();
    session.dispose();
    await expect(session.addObject({ frameIndex: 0, prompts: [] })).rejects.toThrow(
      InvalidStateError,
    );
  });
});

describe('VideoSessionImpl — refineObject', () => {
  it('idle: bumps the epoch and dispatches refineVideoObject with the new epoch', async () => {
    const { session, engine } = await attached();
    const mask = await session.refineObject(1, 3, []);
    expect(mask.objectId).toBe(1);
    expect(engine.refineVideoObject).toHaveBeenCalledExactlyOnceWith(1, {
      objectId: 1,
      frameIndex: 3,
      prompts: [],
      epoch: 1,
    });
  });

  it('while propagating: cancels the active iterator, then dispatches', async () => {
    const { session, engine, workers } = await attached();
    const iterator = session.propagate();
    const first = await iterator.next();
    expect(first.done).toBe(false);

    await session.refineObject(1, 5, []);

    expect(workers[0]?.onCancel).toHaveBeenCalledOnce();
    expect(engine.refineVideoObject).toHaveBeenCalledExactlyOnceWith(
      1,
      expect.objectContaining({ epoch: 1 }),
    );

    // The in-flight iterator's NEXT next() call observes the epoch bump.
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);
  });

  it('after dispose: InvalidStateError', async () => {
    const { session } = await attached();
    session.dispose();
    await expect(session.refineObject(1, 0, [])).rejects.toThrow(InvalidStateError);
  });
});

describe('VideoSessionImpl — removeObject', () => {
  it('idle: synchronous void; bumps epoch and fires removeVideoObject', async () => {
    const { session, engine } = await attached();
    expect(session.removeObject(1)).toBeUndefined();
    await vi.waitFor(() => expect(engine.removeVideoObject).toHaveBeenCalledExactlyOnceWith(1, 1));
    // A subsequent refine/add should now observe epoch 1.
    await session.addObject({ frameIndex: 0, prompts: [] });
    expect(vi.mocked(engine.addVideoObject).mock.calls[0]?.[1].epoch).toBe(1);
  });

  it('while propagating: cancels the active iterator first (fire-and-forget)', async () => {
    const { session, workers } = await attached();
    const iterator = session.propagate();
    await iterator.next();
    session.removeObject(1);
    await vi.waitFor(() => expect(workers[0]?.onCancel).toHaveBeenCalledOnce());
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);
  });

  it('after dispose: throws InvalidStateError', async () => {
    const { session } = await attached();
    session.dispose();
    expect(() => session.removeObject(1)).toThrow(InvalidStateError);
  });
});

describe('VideoSessionImpl — reset', () => {
  it('idle: synchronous void; bumps epoch and fires resetVideoSession', async () => {
    const { session, engine } = await attached();
    session.reset();
    await vi.waitFor(() => expect(engine.resetVideoSession).toHaveBeenCalledExactlyOnceWith(1));
    await session.addObject({ frameIndex: 0, prompts: [] });
    expect(vi.mocked(engine.addVideoObject).mock.calls[0]?.[1].epoch).toBe(1);
  });

  it('while propagating: cancels first, then as idle', async () => {
    const { session, engine, workers } = await attached();
    const iterator = session.propagate();
    await iterator.next();
    session.reset();
    await vi.waitFor(() => expect(engine.resetVideoSession).toHaveBeenCalledOnce());
    expect(workers[0]?.onCancel).toHaveBeenCalledOnce();
    await expect(iterator.next()).rejects.toThrow(EpochInvalidatedError);
  });

  it('after dispose: no-op (does not throw)', async () => {
    const { session } = await attached();
    session.dispose();
    expect(() => session.reset()).not.toThrow();
  });
});

describe('VideoSessionImpl — propagate / the state-machine table', () => {
  it('idle: returns a live iterator yielding frames in [startFrame,endFrame)', async () => {
    const { session } = await attached();
    const seen: number[] = [];
    for await (const frame of session.propagate({ startFrame: 2, endFrame: 5 })) {
      seen.push(frame.frameIndex);
    }
    expect(seen).toEqual([2, 3, 4]);
  });

  it('defaults startFrame=0, endFrame=frameCount', async () => {
    const { session } = await attached();
    const seen: number[] = [];
    for await (const frame of session.propagate()) seen.push(frame.frameIndex);
    expect(seen).toEqual(Array.from({ length: 10 }, (_, i) => i));
  });

  it('a second propagate() while one is active returns a POISONED iterator; the active one is undisturbed', async () => {
    const { session } = await attached();
    const first = session.propagate();
    const firstResult = await first.next();
    expect(firstResult.done).toBe(false);

    const second = session.propagate();
    await expect(second.next()).rejects.toThrow(InvalidStateError);

    // The original iterator keeps working.
    const again = await first.next();
    expect(again.done).toBe(false);
  });

  it('propagate() before attachSource returns a poisoned iterator', async () => {
    const { engine } = fakeEngine();
    const session = new VideoSessionImpl(engine, 1);
    const iterator = session.propagate();
    await expect(iterator.next()).rejects.toThrow(InvalidStateError);
  });

  it("direction:'backward' returns a poisoned iterator with NotImplementedError naming M3", async () => {
    const { session } = await attached();
    const iterator = session.propagate({ direction: 'backward' });
    await expect(iterator.next()).rejects.toThrow(NotImplementedError);
    await expect(iterator.next()).rejects.toThrow(/M3/);
  });

  it('propagate() on a disposed session returns a poisoned iterator', async () => {
    const { session } = await attached();
    session.dispose();
    const iterator = session.propagate();
    await expect(iterator.next()).rejects.toThrow(InvalidStateError);
  });

  it('a rejected propagateVideo start surfaces through the returned iterator', async () => {
    const boom = new Error('worker refused to start the loop');
    const { engine } = fakeEngine({
      propagateVideo: vi.fn(async () => {
        throw boom;
      }),
    });
    const session = new VideoSessionImpl(engine, 1);
    await session.attachSource(new Blob());
    const iterator = session.propagate();
    await expect(iterator.next()).rejects.toBe(boom);
  });

  it('after propagate() settles (done), a fresh propagate() call succeeds', async () => {
    const { session } = await attached();
    for await (const _f of session.propagate({ startFrame: 0, endFrame: 2 })) {
      // drain
    }
    const seen: number[] = [];
    for await (const frame of session.propagate({ startFrame: 5, endFrame: 7 })) {
      seen.push(frame.frameIndex);
    }
    expect(seen).toEqual([5, 6]);
  });
});

describe('VideoSessionImpl — backpressure credit release on break', () => {
  it('break releases credits (no further pulls) and cancels the worker loop', async () => {
    const { session, workers } = await attached();
    let count = 0;
    for await (const _frame of session.propagate({ startFrame: 0, endFrame: 10 })) {
      count++;
      if (count === 1) break;
    }
    expect(count).toBe(1);
    expect(workers[0]?.onCancel).toHaveBeenCalledOnce();
  });
});

describe('VideoSessionImpl — dispose', () => {
  it('idempotent and fire-and-forget closeVideoSession', async () => {
    const { session, engine } = await attached();
    session.dispose();
    session.dispose();
    await vi.waitFor(() => expect(engine.closeVideoSession).toHaveBeenCalledExactlyOnceWith(1));
  });

  it('while propagating: cancels the active iterator before closing', async () => {
    const { session, engine, workers } = await attached();
    const iterator = session.propagate();
    await iterator.next();
    session.dispose();
    await vi.waitFor(() => expect(engine.closeVideoSession).toHaveBeenCalledOnce());
    expect(workers[0]?.onCancel).toHaveBeenCalledOnce();
    // The iterator was cancelled underneath, not left hanging.
    await expect(iterator.next()).resolves.toEqual({ done: true, value: undefined });
  });

  it('use-after-dispose (mask consumers unaffected): a mask already returned stays usable', async () => {
    const { session } = await attached();
    const { mask } = await session.addObject({ frameIndex: 0, prompts: [] });
    session.dispose();
    const stillWorks: MaskResult = mask;
    expect(stillWorks.toBinary()).toHaveLength(64 * 48);
  });
});
