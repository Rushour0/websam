import { StrictMode } from 'react';
import { render, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { Segmenter } from '@websam/core';
import {
  useSegmenter,
  type SegmenterLoader,
  type UseSegmenterResult,
} from './use-segmenter';

/** Minimal stand-in for a loaded segmenter; identity is all the tests need. */
function fakeSegmenter(): Segmenter {
  return { __fake: 'segmenter' } as unknown as Segmenter;
}

function Probe({
  loader,
  onResult,
}: {
  loader: SegmenterLoader;
  onResult: (result: UseSegmenterResult) => void;
}) {
  onResult(useSegmenter(undefined, { loader }));
  return null;
}

function renderProbe(loader: SegmenterLoader) {
  const results: UseSegmenterResult[] = [];
  const utils = render(
    <StrictMode>
      <Probe loader={loader} onResult={(result) => results.push(result)} />
    </StrictMode>,
  );
  const latest = () => results.at(-1);
  return { ...utils, latest };
}

describe('useSegmenter', () => {
  it('invokes the loader exactly once under StrictMode double-mount', async () => {
    const segmenter = fakeSegmenter();
    const loader = vi.fn<SegmenterLoader>(async () => segmenter);

    const { latest } = renderProbe(loader);

    await waitFor(() => expect(latest()?.status).toBe('ready'));
    expect(loader).toHaveBeenCalledTimes(1);
    expect(latest()?.segmenter).toBe(segmenter);
    expect(latest()?.error).toBeUndefined();
  });

  it('surfaces a NotImplementedError rejection as status "error"', async () => {
    // Mirrors core's M0 behavior: createSegmenter rejects with NotImplementedError.
    const failure = Object.assign(new Error('createSegmenter, lands in M1'), {
      name: 'NotImplementedError',
    });
    const loader = vi.fn<SegmenterLoader>(async () => {
      throw failure;
    });

    const { latest } = renderProbe(loader);

    await waitFor(() => expect(latest()?.status).toBe('error'));
    expect(latest()?.error).toBe(failure);
    expect(latest()?.error?.name).toBe('NotImplementedError');
    expect(latest()?.segmenter).toBeNull();
    // StrictMode-safe on the failure path too: still a single invocation.
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('forwards loader progress to the hook result', async () => {
    const segmenter = fakeSegmenter();
    const loader = vi.fn<SegmenterLoader>(async (_config, context) => {
      context.onProgress(0.5);
      return segmenter;
    });

    const { latest } = renderProbe(loader);

    await waitFor(() => expect(latest()?.status).toBe('ready'));
    expect(latest()?.progress).toBe(0.5);
  });

  it('aborts the load after the last subscribed component unmounts', async () => {
    let signal: AbortSignal | undefined;
    const loader = vi.fn<SegmenterLoader>((_config, context) => {
      signal = context.signal;
      return new Promise<Segmenter>(() => {
        // Never settles: simulates a long model download.
      });
    });

    const { latest, unmount } = renderProbe(loader);

    await waitFor(() => expect(latest()?.status).toBe('loading'));
    // StrictMode's mount -> cleanup -> remount must NOT have aborted the load.
    expect(signal?.aborted).toBe(false);

    unmount();
    await waitFor(() => expect(signal?.aborted).toBe(true));
    expect(loader).toHaveBeenCalledTimes(1);
  });
});
