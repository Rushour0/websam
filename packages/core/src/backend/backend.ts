/**
 * The Backend abstraction — the highest-leverage architectural contract in
 * websam.
 *
 * WHY THIS EXISTS: the video engine (memory bank, prompt encoder/decoder
 * scheduling, frame pipeline) talks ONLY to this interface — never to
 * `ort.env.webgpu`, `GPUBuffer`, or any onnxruntime-web specifics directly.
 * That indirection is what makes a Node backend possible at M5: `GPUBuffer`
 * has no Node analog, so any code path that touched WebGPU objects directly
 * would have to be rewritten. Behind {@link Backend}, "device memory" is an
 * opaque {@link DeviceTensor} that may be a GPUBuffer (WebGPU), a WASM heap
 * region (browser CPU), or a Node-side buffer — the engine cannot tell and
 * must not care.
 *
 * The contract is deliberately tiny: sessions that run graphs, tensors that
 * live somewhere, one region-copy primitive for the memory-bank ring, and an
 * explicit readback for the rare device→CPU crossings.
 */

/**
 * Where a tensor's bytes physically live.
 *
 * - `'device'` — accelerator-resident (e.g. a GPUBuffer). Feeding a
 *   device tensor back into {@link BackendSession.run} must not copy
 *   through the CPU.
 * - `'cpu'` — host-resident, readable without an async round trip.
 */
export type TensorLocation = 'device' | 'cpu';

/** Element types a backend must understand (mirrors ONNX tensor types websam uses). */
export type DType = 'float32' | 'float16' | 'int64' | 'uint8' | 'int32' | 'bool';

/**
 * An opaque handle to tensor storage owned by a {@link Backend}.
 *
 * The engine never sees the underlying GPUBuffer / ArrayBuffer; it can only
 * pass the handle back into backend methods. Ownership: whoever received the
 * tensor (from {@link Backend.allocTensor} or {@link BackendSession.run})
 * must call {@link DeviceTensor.dispose} exactly once when done.
 */
export interface DeviceTensor {
  /** Logical shape, e.g. `[1, 256, 64, 64]`. */
  readonly shape: readonly number[];
  /** Element type of the tensor. */
  readonly dtype: DType;
  /** Where the bytes live; determines whether readback is needed. */
  readonly location: TensorLocation;
  /** Release the underlying storage. Idempotent calls after the first throw `InvalidStateError`. */
  dispose(): void;
}

/**
 * Declares, per graph output name, where the backend should materialize that
 * output. Keeping hot outputs (image embeddings, memory features) at
 * `'device'` is what lets the video loop run without per-frame readbacks;
 * only mask logits destined for compositing come back to `'cpu'`.
 */
export interface IOBindingPlan {
  /** Output tensor name → desired location for that output. */
  outputLocations: Record<string, TensorLocation>;
}

/**
 * A graph (ONNX model component) to compile into a session. Exactly one of
 * `bytes` or `url` is provided; backends that stream compilation may prefer
 * `url`, others fetch it themselves.
 */
export type GraphAsset = { readonly name: string } & (
  | { readonly bytes: Uint8Array; readonly url?: undefined }
  | { readonly url: string; readonly bytes?: undefined }
);

/**
 * A compiled, runnable graph. One websam model is several sessions (image
 * encoder, prompt encoder, mask decoder, memory attention…) sharing one
 * {@link Backend}, so device tensors flow between sessions copy-free.
 */
export interface BackendSession {
  /**
   * Run the graph.
   *
   * @param feeds - Input name → tensor. Device tensors are bound in place.
   * @param fetches - Optional subset of output names to compute; defaults to
   * all graph outputs.
   * @returns Output name → tensor, each located per the session's
   * {@link IOBindingPlan} (default `'cpu'` for unlisted outputs).
   */
  run(
    feeds: Record<string, DeviceTensor>,
    fetches?: readonly string[],
  ): Promise<Record<string, DeviceTensor>>;

  /** Release the compiled session and all backend-internal buffers it owns. */
  dispose(): Promise<void>;
}

/**
 * A concrete execution environment (WebGPU, browser WASM, or — at M5 —
 * Node). Everything above this interface is environment-agnostic.
 */
export interface Backend {
  /** Which environment this backend drives. */
  readonly kind: 'webgpu' | 'wasm' | 'node';

  /**
   * Probe and acquire the underlying device. Must be called (and awaited)
   * before any other method. Throws `UnsupportedDeviceError` if the
   * environment cannot run this backend at all.
   */
  init(): Promise<void>;

  /**
   * Compile a graph into a runnable session honoring the given I/O binding
   * plan (which outputs stay on-device vs. come back to the CPU).
   */
  createSession(graph: GraphAsset, plan?: IOBindingPlan): Promise<BackendSession>;

  /**
   * Allocate an uninitialized tensor. Throws `OutOfMemoryError` when the
   * device or host cannot satisfy the allocation.
   */
  allocTensor(shape: readonly number[], dtype: DType, location: TensorLocation): DeviceTensor;

  /**
   * Create a tensor initialized from host data (`'cpu'` location; sessions
   * bind it to the device at run time). `data` must be the matching typed
   * array for `dtype` — `int64` takes a `BigInt64Array`, `float16` raw half
   * bits in a `Uint16Array`, `bool` 0/1 bytes in a `Uint8Array` — with
   * exactly shape-product elements. This is how the image path feeds pixels
   * and prompt points, and how the video loop will feed each frame.
   */
  uploadTensor(data: ArrayBufferView, shape: readonly number[], dtype: DType): DeviceTensor;

  /**
   * THE memory-bank ring primitive: copy `src` into slot `slotIndex` of the
   * ring buffer tensor `dst` (whose leading dimension is the slot axis),
   * entirely on-device — no CPU round trip. The video memory bank is a fixed
   * ring of past-frame features; each tracked frame overwrites the oldest
   * slot with exactly this call.
   *
   * Slot-shape rule (M2, relaxed): `src` must have exactly
   * `dst.shape.slice(1)`'s ELEMENT COUNT and the same dtype — the copy is a
   * contiguous byte copy, reshape-free. Literal shape equality is NOT
   * required: the video engine copies a whole per-object KV ring
   * (`[S, T, D]`) into one batch slot of a `[B, S*T, D]` graph input, and
   * requiring shape equality would force pointless reshapes on an opaque
   * handle.
   *
   * @param src - Tensor with `dst.shape.slice(1)`'s element count and the same dtype as `dst`.
   * @param dst - Ring tensor; `dst.shape[0]` is the slot count.
   * @param slotIndex - Slot to overwrite, in `[0, dst.shape[0])`.
   */
  copyRegion(src: DeviceTensor, dst: DeviceTensor, slotIndex: number): void;

  /**
   * Explicit device→CPU crossing. The only sanctioned way to observe tensor
   * bytes; typed-array view matches the tensor's {@link DType}
   * (`float16` reads back as a `Uint16Array` of raw half bits).
   */
  readback(tensor: DeviceTensor): Promise<ArrayBufferView>;

  /** Live-resource census for leak gates. Optional; browser backends implement it in M2. */
  debugStats?(): { liveTensors: number; liveBytes: number };

  /** Release the device and every resource this backend still owns. */
  dispose(): Promise<void>;
}
