import { describe, expect, it, vi } from 'vitest';
import type { DeviceTensor } from '../backend/backend.js';
import type { OrtModule } from '../backend/webgpu-backend.js';
import { InvalidStateError, OutOfMemoryError } from '../errors.js';
import { allocCpuTensor, createCpuTensor, OrtDeviceTensor, readbackTensor } from './ort-tensor.js';

type OrtTensor = import('onnxruntime-web').Tensor;

/** Minimal structural stand-in for ort.Tensor (node unit tests never load real ort). */
class FakeOrtTensor {
  location = 'cpu';
  disposeCalls = 0;

  constructor(
    readonly type: string,
    readonly data: unknown,
    readonly dims: readonly number[] = [],
  ) {}

  dispose(): void {
    this.disposeCalls += 1;
    this.location = 'none';
  }

  async getData(): Promise<unknown> {
    return this.data;
  }
}

const asOrt = (t: FakeOrtTensor): OrtTensor => t as unknown as OrtTensor;
const fakeOrt = { Tensor: FakeOrtTensor } as unknown as OrtModule;

describe('OrtDeviceTensor.wrap', () => {
  it('exposes shape/dtype and maps ort cpu location to cpu', () => {
    const tensor = OrtDeviceTensor.wrap(asOrt(new FakeOrtTensor('float32', new Float32Array(4), [2, 2])));
    expect(tensor.shape).toEqual([2, 2]);
    expect(tensor.dtype).toBe('float32');
    expect(tensor.location).toBe('cpu');
  });

  it("maps location 'gpu-buffer' — and only that — to 'device'", () => {
    const onGpu = new FakeOrtTensor('float32', undefined, [1]);
    onGpu.location = 'gpu-buffer';
    expect(OrtDeviceTensor.wrap(asOrt(onGpu)).location).toBe('device');

    const pinned = new FakeOrtTensor('float32', new Float32Array(1), [1]);
    pinned.location = 'cpu-pinned';
    expect(OrtDeviceTensor.wrap(asOrt(pinned)).location).toBe('cpu');
  });

  it('rejects ort element types outside the DType union', () => {
    expect(() => OrtDeviceTensor.wrap(asOrt(new FakeOrtTensor('string', ['x'], [1])))).toThrow(
      InvalidStateError,
    );
    expect(() => OrtDeviceTensor.wrap(asOrt(new FakeOrtTensor('uint16', new Uint16Array(1), [1])))).toThrow(
      InvalidStateError,
    );
  });
});

describe('OrtDeviceTensor.dispose', () => {
  it('disposes the underlying ort tensor exactly once and notifies onDispose', () => {
    const inner = new FakeOrtTensor('float32', new Float32Array(1), [1]);
    const onDispose = vi.fn();
    const tensor = OrtDeviceTensor.wrap(asOrt(inner), onDispose);
    tensor.dispose();
    expect(inner.disposeCalls).toBe(1);
    expect(tensor.disposed).toBe(true);
    expect(onDispose).toHaveBeenCalledExactlyOnceWith(tensor);
  });

  it('throws InvalidStateError on a second dispose (underlying not re-disposed)', () => {
    const inner = new FakeOrtTensor('float32', new Float32Array(1), [1]);
    const tensor = OrtDeviceTensor.wrap(asOrt(inner));
    tensor.dispose();
    expect(() => tensor.dispose()).toThrow(InvalidStateError);
    expect(inner.disposeCalls).toBe(1);
  });
});

describe('createCpuTensor', () => {
  it('constructs an ort tensor from (dtype, data, shape)', () => {
    const data = Float32Array.from([1, 2, 3, 4, 5, 6]);
    const tensor = createCpuTensor(fakeOrt, data, [2, 3], 'float32');
    expect(tensor.shape).toEqual([2, 3]);
    expect(tensor.dtype).toBe('float32');
    expect(tensor.location).toBe('cpu');
    expect((tensor.ortTensor as unknown as FakeOrtTensor).data).toBe(data);
  });

  it('carries int64 host data as BigInt64Array', () => {
    const data = BigInt64Array.from([1n, 0n, 1n]);
    const tensor = createCpuTensor(fakeOrt, data, [1, 1, 3], 'int64');
    expect(tensor.dtype).toBe('int64');
    expect((tensor.ortTensor as unknown as FakeOrtTensor).data).toBe(data);
  });
});

describe('allocCpuTensor', () => {
  it.each([
    ['float32', Float32Array],
    ['float16', Uint16Array],
    ['int64', BigInt64Array],
    ['uint8', Uint8Array],
    ['int32', Int32Array],
    ['bool', Uint8Array],
  ] as const)('allocates a zeroed %s tensor with the matching typed array', (dtype, Ctor) => {
    const tensor = allocCpuTensor(fakeOrt, [2, 3], dtype);
    const data = (tensor.ortTensor as unknown as FakeOrtTensor).data;
    expect(data).toBeInstanceOf(Ctor);
    expect((data as { length: number }).length).toBe(6);
    for (const value of data as Iterable<number | bigint>) {
      expect(value === 0 || value === 0n).toBe(true);
    }
  });

  it('supports zero-sized dims (empty [1, 0, 4] boxes tensor)', () => {
    const tensor = allocCpuTensor(fakeOrt, [1, 0, 4], 'float32');
    expect((tensor.ortTensor as unknown as FakeOrtTensor).data).toHaveLength(0);
  });

  it('rejects non-integer and negative dims with RangeError', () => {
    expect(() => allocCpuTensor(fakeOrt, [1, 1.5], 'float32')).toThrow(RangeError);
    expect(() => allocCpuTensor(fakeOrt, [-1, 2], 'float32')).toThrow(RangeError);
  });

  it('maps an unsatisfiable allocation to OutOfMemoryError', () => {
    expect(() => allocCpuTensor(fakeOrt, [2 ** 40, 2 ** 40], 'float32')).toThrow(OutOfMemoryError);
  });
});

describe('readbackTensor', () => {
  it('returns the existing data view (no copy) for cpu tensors', async () => {
    const data = Float32Array.from([1, 2, 3]);
    const tensor = createCpuTensor(fakeOrt, data, [3], 'float32');
    await expect(readbackTensor(tensor)).resolves.toBe(data);
  });

  it('reads device tensors back via getData()', async () => {
    const bits = new Uint16Array([0x3c00]); // f16 raw half bits per the contract
    const inner = new FakeOrtTensor('float16', bits, [1]);
    inner.location = 'gpu-buffer';
    const getData = vi.spyOn(inner, 'getData');
    const tensor = OrtDeviceTensor.wrap(asOrt(inner));
    await expect(readbackTensor(tensor)).resolves.toBe(bits);
    expect(getData).toHaveBeenCalledOnce();
  });

  it('rejects disposed tensors with InvalidStateError', async () => {
    const tensor = createCpuTensor(fakeOrt, new Float32Array(1), [1], 'float32');
    tensor.dispose();
    await expect(readbackTensor(tensor)).rejects.toBeInstanceOf(InvalidStateError);
  });

  it('rejects tensors that are not OrtDeviceTensor instances', async () => {
    const foreign = {
      shape: [1],
      dtype: 'float32',
      location: 'cpu',
      dispose() {},
    } as DeviceTensor;
    await expect(readbackTensor(foreign)).rejects.toBeInstanceOf(InvalidStateError);
  });
});
