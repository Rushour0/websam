import { NotImplementedError } from '@websam3/core';
import type { SegmenterConfig } from '@websam3/core';

/**
 * Options for {@link useImageSession}.
 *
 * Placeholder contract — the full option surface (source image, prompt
 * handling, mask post-processing) lands in M1.
 */
export interface UseImageSessionOptions {
  /** Segmenter configuration forwarded to the underlying session. */
  config?: SegmenterConfig;
}

/**
 * Options for {@link useVideoSession}.
 *
 * Placeholder contract — the full option surface (video source, tracked
 * object management, frame scheduling) lands in M2.
 */
export interface UseVideoSessionOptions {
  /** Segmenter configuration forwarded to the underlying session. */
  config?: SegmenterConfig;
}

/**
 * Hook for prompt-based segmentation of a single image.
 *
 * @remarks Not implemented at M0 — lands in M1. Calling it throws
 * `NotImplementedError`.
 */
export function useImageSession(options?: UseImageSessionOptions): never {
  void options;
  throw new NotImplementedError('useImageSession, lands in M1');
}

/**
 * Hook for object tracking and mask propagation across video frames.
 *
 * @remarks Not implemented at M0 — lands in M2. Calling it throws
 * `NotImplementedError`.
 */
export function useVideoSession(options?: UseVideoSessionOptions): never {
  void options;
  throw new NotImplementedError('useVideoSession, lands in M2');
}
