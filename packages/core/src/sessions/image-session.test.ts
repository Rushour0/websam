import { afterEach, describe, expect, it, vi } from 'vitest';
import { computeTransform } from '../coords.js';
import { InvalidStateError } from '../errors.js';
import { MaskResultImpl } from '../masks/mask-result.js';
import type { EncodeResponse, MaskPayload } from '../worker/protocol.js';
import { ImageSessionImpl } from './image-session.js';
import type { RemoteEngine } from './spawn-worker.js';

const SESSION_ID = 3;
const TRANSFORM = computeTransform(4, 2, 8, 'square-stretch');
const ENCODE_RESPONSE: EncodeResponse = {
  width: 4,
  height: 2,
  encodeMs: 1.5,
  transform: TRANSFORM,
};

/** Minimal ImageBitmap stand-in for the node environment. */
class FakeImageBitmap {
  readonly width = 4;
  readonly height = 2;
  closed = false;
  close(): void {
    this.closed = true;
  }
}

function stubBitmapGlobals(): void {
  vi.stubGlobal('ImageBitmap', FakeImageBitmap);
}

function fakeEngine(overrides: Partial<RemoteEngine> = {}): RemoteEngine {
  return {
    init: vi.fn(async () => {
      throw new Error('init: not under test');
    }),
    createSession: vi.fn(async () => SESSION_ID),
    encodeImage: vi.fn(async () => ENCODE_RESPONSE),
    decode: vi.fn(async () => []),
    closeSession: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
    ...overrides,
  };
}

function bitmap(): ImageBitmap {
  return new FakeImageBitmap() as unknown as ImageBitmap;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ImageSessionImpl.encode', () => {
  it('sends an ImageBitmap input as-is and stores the returned transform', async () => {
    stubBitmapGlobals();
    const engine = fakeEngine();
    const session = new ImageSessionImpl(engine, SESSION_ID);
    const input = bitmap();

    expect(session.isEncoded).toBe(false);
    const result = await session.encode(input);

    expect(result).toEqual({ width: 4, height: 2, encodeMs: 1.5 });
    expect(session.isEncoded).toBe(true);
    expect(engine.encodeImage).toHaveBeenCalledOnce();
    const [id, sent] = vi.mocked(engine.encodeImage).mock.calls[0]!;
    expect(id).toBe(SESSION_ID);
    expect(sent).toBe(input);
  });

  it('normalizes non-bitmap inputs through createImageBitmap on the main thread', async () => {
    stubBitmapGlobals();
    const created = bitmap();
    const createImageBitmap = vi.fn(async () => created);
    vi.stubGlobal('createImageBitmap', createImageBitmap);
    const engine = fakeEngine();
    const session = new ImageSessionImpl(engine, SESSION_ID);
    const imageData = { data: new Uint8ClampedArray(32), width: 4, height: 2 } as ImageData;

    await session.encode(imageData);

    expect(createImageBitmap).toHaveBeenCalledExactlyOnceWith(imageData);
    expect(vi.mocked(engine.encodeImage).mock.calls[0]?.[1]).toBe(created);
  });

  it('throws InvalidStateError for non-bitmap inputs when createImageBitmap is missing', async () => {
    // Plain node: neither ImageBitmap nor createImageBitmap exist.
    const session = new ImageSessionImpl(fakeEngine(), SESSION_ID);
    await expect(session.encode({} as ImageData)).rejects.toThrow(InvalidStateError);
  });

  it('rejects with the abort reason pre-dispatch when the signal is already aborted', async () => {
    stubBitmapGlobals();
    const engine = fakeEngine();
    const session = new ImageSessionImpl(engine, SESSION_ID);
    const controller = new AbortController();
    controller.abort();

    await expect(session.encode(bitmap(), { signal: controller.signal })).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(engine.encodeImage).not.toHaveBeenCalled();
    expect(session.isEncoded).toBe(false);
  });

  it('discards an in-flight result on abort: rejects AbortError, isEncoded stays false', async () => {
    stubBitmapGlobals();
    let resolveEncode!: (r: EncodeResponse) => void;
    const engine = fakeEngine({
      encodeImage: vi.fn(
        () => new Promise<EncodeResponse>((resolve) => (resolveEncode = resolve)),
      ),
    });
    const session = new ImageSessionImpl(engine, SESSION_ID);
    const controller = new AbortController();

    const pending = session.encode(bitmap(), { signal: controller.signal });
    await vi.waitFor(() => expect(engine.encodeImage).toHaveBeenCalledOnce());
    controller.abort();
    resolveEncode(ENCODE_RESPONSE);

    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(session.isEncoded).toBe(false);
  });

  it('propagates worker errors when not aborted', async () => {
    stubBitmapGlobals();
    const boom = new InvalidStateError('no encoded slot');
    const engine = fakeEngine({
      encodeImage: vi.fn(async () => {
        throw boom;
      }),
    });
    const session = new ImageSessionImpl(engine, SESSION_ID);
    await expect(session.encode(bitmap())).rejects.toBe(boom);
  });
});

describe('ImageSessionImpl.decode', () => {
  it('throws InvalidStateError before a successful encode', async () => {
    const session = new ImageSessionImpl(fakeEngine(), SESSION_ID);
    await expect(session.decode([{ type: 'point', x: 1, y: 1, label: 1 }])).rejects.toThrow(
      InvalidStateError,
    );
  });

  it('forwards prompts/options and wraps payloads in MaskResultImpl with the transform', async () => {
    stubBitmapGlobals();
    const payload: MaskPayload = {
      objectId: 5,
      score: 0.87,
      width: 4,
      height: 2,
      binaryMask: new Uint8Array([1, 0, 0, 1, 1, 1, 0, 0]).buffer,
    };
    const engine = fakeEngine({ decode: vi.fn(async () => [payload]) });
    const session = new ImageSessionImpl(engine, SESSION_ID);
    await session.encode(bitmap());

    const prompts = [{ type: 'point', x: 2, y: 1, label: 1 } as const];
    const masks = await session.decode(prompts, { multimask: true, objectId: 5 });

    expect(engine.decode).toHaveBeenCalledExactlyOnceWith(SESSION_ID, {
      prompts,
      multimask: true,
      objectId: 5,
    });
    expect(masks).toHaveLength(1);
    const mask = masks[0]!;
    expect(mask).toBeInstanceOf(MaskResultImpl);
    expect(mask.objectId).toBe(5);
    expect(mask.score).toBe(0.87);
    expect(mask.width).toBe(4);
    expect(mask.height).toBe(2);
    expect(mask.toBinary()).toEqual(new Uint8Array([1, 0, 0, 1, 1, 1, 0, 0]));
    expect((mask as MaskResultImpl).transform).toEqual(TRANSFORM);
  });
});

describe('ImageSessionImpl.dispose', () => {
  it('closes the worker session slot exactly once (idempotent)', async () => {
    const engine = fakeEngine();
    const session = new ImageSessionImpl(engine, SESSION_ID);
    session.dispose();
    session.dispose();
    expect(engine.closeSession).toHaveBeenCalledExactlyOnceWith(SESSION_ID);
  });

  it('encode/decode after dispose throw InvalidStateError', async () => {
    stubBitmapGlobals();
    const session = new ImageSessionImpl(fakeEngine(), SESSION_ID);
    session.dispose();
    await expect(session.encode(bitmap())).rejects.toThrow(InvalidStateError);
    await expect(session.decode([])).rejects.toThrow(InvalidStateError);
    expect(session.isEncoded).toBe(false);
  });

  it('swallows closeSession rejections (worker may already be terminated)', async () => {
    const engine = fakeEngine({
      closeSession: vi.fn(async () => {
        throw new Error('worker terminated');
      }),
    });
    const session = new ImageSessionImpl(engine, SESSION_ID);
    expect(() => session.dispose()).not.toThrow();
    // Let the fire-and-forget rejection settle; an unhandled rejection here
    // would fail the test run.
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
});
