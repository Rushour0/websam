/**
 * @websam3/video-editing — mask timeline storage, alpha-matte export, and
 * mask compositing on top of `@websam3/core`.
 *
 * @packageDocumentation
 */
export {
  MaskTimeline,
  type MaskTimelineInit,
  type MaskTimelineJSON,
  type FrameRange,
  type CollectOptions,
  type SerializedRLEMask,
} from './timeline.js';
export {
  AlphaMatteExporter,
  type ExportMode,
  type ExportFormat,
  type ExportOptions,
  type ExportResult,
} from './exporter.js';
export {
  MaskCompositor,
  type CompositeMode,
  type CompositeOptions,
} from './compositor.js';
