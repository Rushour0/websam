import { afterEach, describe, expect, it, vi } from 'vitest';

// loadOrt dynamic-imports onnxruntime-web; in node unit tests the module is
// mocked so no real wasm runtime ever loads.
vi.mock('onnxruntime-web', () => ({ env: { wasm: {} } }));

type LoadOrt = typeof import('./ort-env.js').loadOrt;

/**
 * Fresh module state per test: loadOrt memoizes at module scope, and the
 * mocked ort env must be wiped since the mock instance can survive
 * vi.resetModules().
 */
async function freshLoadOrt(): Promise<LoadOrt> {
  vi.resetModules();
  const ortMock = (await import('onnxruntime-web')) as unknown as {
    env: { wasm: Record<string, unknown> };
  };
  ortMock.env.wasm = {};
  return (await import('./ort-env.js')).loadOrt;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadOrt env flag application', () => {
  it('always disables the ort proxy worker (we already run in a worker)', async () => {
    const loadOrt = await freshLoadOrt();
    const ort = await loadOrt();
    expect(ort.env.wasm.proxy).toBe(false);
  });

  it('forces numThreads = 1 when the page is not cross-origin isolated', async () => {
    vi.stubGlobal('crossOriginIsolated', false);
    const loadOrt = await freshLoadOrt();
    const ort = await loadOrt();
    expect(ort.env.wasm.numThreads).toBe(1);
  });

  it('leaves numThreads to ort default when cross-origin isolated and unspecified', async () => {
    vi.stubGlobal('crossOriginIsolated', true);
    const loadOrt = await freshLoadOrt();
    const ort = await loadOrt();
    expect(ort.env.wasm.numThreads).toBeUndefined();
  });

  it('honors an explicit numThreads even without isolation', async () => {
    vi.stubGlobal('crossOriginIsolated', false);
    const loadOrt = await freshLoadOrt();
    const ort = await loadOrt({ numThreads: 4 });
    expect(ort.env.wasm.numThreads).toBe(4);
  });

  it('applies wasmPaths iff provided', async () => {
    const loadOrt = await freshLoadOrt();
    const ort = await loadOrt({ wasmPaths: '/ort-assets/' });
    expect(ort.env.wasm.wasmPaths).toBe('/ort-assets/');

    const loadOrtDefault = await freshLoadOrt();
    const ortDefault = await loadOrtDefault();
    expect(ortDefault.env.wasm.wasmPaths).toBeUndefined();
  });
});

describe('loadOrt memoization', () => {
  it('imports ort exactly once: repeat calls return the same promise and module', async () => {
    const loadOrt = await freshLoadOrt();
    const first = loadOrt({ wasmPaths: '/a/', numThreads: 2 });
    const second = loadOrt({ wasmPaths: '/a/', numThreads: 2 });
    expect(second).toBe(first);
    expect(await second).toBe(await first);
  });

  it('treats no-options calls as equal options', async () => {
    const loadOrt = await freshLoadOrt();
    const first = loadOrt();
    expect(loadOrt(undefined)).toBe(first);
    expect(loadOrt({})).toBe(first);
  });

  it('throws InvalidStateError on a second call with different options', async () => {
    const loadOrt = await freshLoadOrt();
    await loadOrt({ wasmPaths: '/a/' });
    // Class identity does not survive vi.resetModules(); match name + code
    // (which is exactly what the error taxonomy recommends anyway).
    let caught: unknown;
    try {
      loadOrt({ wasmPaths: '/b/' });
    } catch (err) {
      caught = err;
    }
    expect(caught).toMatchObject({ name: 'InvalidStateError', code: 'INVALID_STATE' });
    expect(() => loadOrt({ wasmPaths: '/a/', numThreads: 2 })).toThrow(/different options/);
    expect(() => loadOrt()).toThrow(/different options/);
  });
});
