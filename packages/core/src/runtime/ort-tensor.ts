/**
 * The one {@link DeviceTensor} implementation for both browser backends — a
 * thin wrapper over `ort.Tensor` — plus the shared CPU tensor
 * construction/readback helpers the backend method bodies delegate to.
 *
 * Location mapping: an ort tensor is `'device'`-located iff ort reports
 * `location === 'gpu-buffer'`; everything else ort can report for our
 * tensors (`'cpu'`, `'cpu-pinned'`) is `'cpu'` under the Backend contract.
 */
import type { DeviceTensor, DType, TensorLocation } from '../backend/backend.js';
import type { OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError, OutOfMemoryError } from '../errors.js';

type OrtTensor = import('onnxruntime-web').Tensor;

/**
 * Matching JS typed-array constructor per {@link DType}. `float16` carries
 * raw half bits in a `Uint16Array` (per the Backend readback contract);
 * `bool` is 0/1 bytes in a `Uint8Array` — both mirror ort's own data map.
 */
const TYPED_ARRAY_CTORS = {
  float32: Float32Array,
  float16: Uint16Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  int32: Int32Array,
  bool: Uint8Array,
} as const satisfies Record<DType, new (length: number) => ArrayBufferView>;

function isSupportedDType(type: string): type is DType {
  return Object.hasOwn(TYPED_ARRAY_CTORS, type);
}

/**
 * Wraps an `ort.Tensor` as a {@link DeviceTensor}. Both `WebGpuBackend` and
 * `WasmBackend` produce only this implementation, so backend code can
 * unwrap feeds via {@link OrtDeviceTensor.ortTensor} with an `instanceof`
 * guard.
 */
export class OrtDeviceTensor implements DeviceTensor {
  /** Internal-only escape hatch: the wrapped onnxruntime-web tensor. */
  readonly ortTensor: OrtTensor;

  readonly #onDispose: ((tensor: OrtDeviceTensor) => void) | undefined;
  #disposed = false;

  private constructor(ortTensor: OrtTensor, onDispose?: (tensor: OrtDeviceTensor) => void) {
    this.ortTensor = ortTensor;
    this.#onDispose = onDispose;
  }

  /**
   * Wrap an ort tensor. Location is `'device'` iff
   * `ortTensor.location === 'gpu-buffer'`.
   *
   * @param ortTensor - The tensor to wrap.
   * @param onDispose - Internal bookkeeping hook (backends untrack the
   * wrapper when it is disposed); invoked exactly once, after the ort
   * tensor is released.
   * @throws InvalidStateError when the ort element type is outside websam's
   * {@link DType} union.
   */
  static wrap(
    ortTensor: OrtTensor,
    onDispose?: (tensor: OrtDeviceTensor) => void,
  ): OrtDeviceTensor {
    if (!isSupportedDType(ortTensor.type)) {
      throw new InvalidStateError(
        `OrtDeviceTensor.wrap: unsupported ort tensor type '${ortTensor.type}'`,
      );
    }
    return new OrtDeviceTensor(ortTensor, onDispose);
  }

  /** Logical shape (ort `dims`). */
  get shape(): readonly number[] {
    return this.ortTensor.dims;
  }

  /** Element type; validated against {@link DType} at {@link wrap} time. */
  get dtype(): DType {
    return this.ortTensor.type as DType;
  }

  /** `'device'` iff the ort tensor lives in a GPU buffer. */
  get location(): TensorLocation {
    return this.ortTensor.location === 'gpu-buffer' ? 'device' : 'cpu';
  }

  /** Whether {@link dispose} has run (internal; guards readback + backend sweeps). */
  get disposed(): boolean {
    return this.#disposed;
  }

  /** Release the ort tensor. Second call throws {@link InvalidStateError} per the contract. */
  dispose(): void {
    if (this.#disposed) {
      throw new InvalidStateError('OrtDeviceTensor.dispose called on an already-disposed tensor');
    }
    this.#disposed = true;
    this.ortTensor.dispose();
    this.#onDispose?.(this);
  }
}

/** Structural view of the ort Tensor constructor for the (type, data, dims) CPU form. */
type CpuTensorCtor = new (
  type: DType,
  data: ArrayBufferView,
  dims: readonly number[],
) => OrtTensor;

/**
 * Build a `'cpu'`-located tensor from host data (the shared body of
 * `Backend.uploadTensor`). ort validates that `data`'s typed-array kind
 * matches `dtype` (`int64` takes a `BigInt64Array`) and that its length
 * equals the shape product.
 */
export function createCpuTensor(
  ort: OrtModule,
  data: ArrayBufferView,
  shape: readonly number[],
  dtype: DType,
  onDispose?: (tensor: OrtDeviceTensor) => void,
): OrtDeviceTensor {
  const Tensor = ort.Tensor as unknown as CpuTensorCtor;
  return OrtDeviceTensor.wrap(new Tensor(dtype, data, shape), onDispose);
}

/**
 * Zero-copy reshape VIEW of `tensor`: a new {@link DeviceTensor} over the same
 * underlying storage with different `dims` (element count must match). Used to
 * present the memory-bank ring `[M,T,C]` as the `[1,M,T,C]` batch-first shape
 * the memory-attention graph declares, without copying.
 *
 * The view shares the source's data/GPU buffer; disposing the view does NOT
 * free that storage (a CPU tensor dispose is a no-op; `fromGpuBuffer` does not
 * own the buffer). The source tensor remains the owner.
 */
export function reshapeOrtView(
  ort: OrtModule,
  tensor: DeviceTensor,
  shape: readonly number[],
): OrtDeviceTensor {
  const src = (tensor as OrtDeviceTensor).ortTensor;
  const want = shape.reduce((a, b) => a * b, 1);
  const have = src.dims.reduce((a, b) => a * b, 1);
  if (want !== have) {
    throw new InvalidStateError(
      `reshape: element count ${want} ([${shape.join(',')}]) != source ${have} ([${src.dims.join(',')}])`,
    );
  }
  const dtype = src.type as DType;
  if (src.location === 'gpu-buffer') {
    const buffer = (src as unknown as { gpuBuffer: unknown }).gpuBuffer;
    const fromGpuBuffer = (
      ort.Tensor as unknown as {
        fromGpuBuffer(b: unknown, opts: { dataType: DType; dims: readonly number[] }): OrtTensor;
      }
    ).fromGpuBuffer;
    return OrtDeviceTensor.wrap(fromGpuBuffer(buffer, { dataType: dtype, dims: shape }));
  }
  const Tensor = ort.Tensor as unknown as CpuTensorCtor;
  return OrtDeviceTensor.wrap(new Tensor(dtype, src.data as ArrayBufferView, shape));
}

/**
 * Build a zeroed `'cpu'`-located tensor (the shared body of
 * `Backend.allocTensor` for `location: 'cpu'`).
 *
 * @throws RangeError on a non-integer or negative dimension.
 * @throws OutOfMemoryError when the host cannot satisfy the allocation.
 */
export function allocCpuTensor(
  ort: OrtModule,
  shape: readonly number[],
  dtype: DType,
  onDispose?: (tensor: OrtDeviceTensor) => void,
): OrtDeviceTensor {
  let length = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new RangeError(
        `allocCpuTensor: invalid dimension ${dim} in shape [${shape.join(', ')}]`,
      );
    }
    length *= dim;
  }
  const Ctor = TYPED_ARRAY_CTORS[dtype];
  let data: ArrayBufferView;
  try {
    data = new Ctor(length);
  } catch (err) {
    if (err instanceof RangeError) {
      throw new OutOfMemoryError(
        `allocCpuTensor: cannot allocate ${length} ${dtype} elements for shape [${shape.join(', ')}]`,
        { cause: err },
      );
    }
    throw err;
  }
  return createCpuTensor(ort, data, shape, dtype, onDispose);
}

/**
 * The shared body of `Backend.readback`: cpu tensors return a typed-array
 * view over their existing data (no copy); device tensors cross the
 * device→CPU boundary via `ortTensor.getData()` (`float16` reads back as a
 * `Uint16Array` of raw half bits per the contract).
 *
 * @throws InvalidStateError when the tensor is foreign (not an
 * {@link OrtDeviceTensor}) or already disposed.
 */
export async function readbackTensor(tensor: DeviceTensor): Promise<ArrayBufferView> {
  if (!(tensor instanceof OrtDeviceTensor)) {
    throw new InvalidStateError(
      'readback: tensor was not created by this backend (expected an OrtDeviceTensor)',
    );
  }
  if (tensor.disposed) {
    throw new InvalidStateError('readback: tensor is disposed');
  }
  if (tensor.location === 'device') {
    return (await tensor.ortTensor.getData()) as ArrayBufferView;
  }
  return tensor.ortTensor.data as ArrayBufferView;
}
