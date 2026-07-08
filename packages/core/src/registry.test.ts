import { describe, expect, it } from 'vitest';
import { InvalidStateError } from './errors.js';
import { getModel, listModels, registerModel, type ModelSpec } from './registry.js';

describe('built-in model tiers', () => {
  it('pre-registers edgetam as permissive and universal', () => {
    const spec = getModel('edgetam');
    expect(spec).toBeDefined();
    expect(spec).toMatchObject({
      arch: 'edgetam',
      inputSize: 1024,
      supportsVideo: true,
      license: 'apache-2.0',
      devices: { webgpu: true, wasm: true },
    });
    expect(spec?.requiresLicenseAcceptance).toBeFalsy();
    expect(spec?.manifestUrl).toMatch(/^https:\/\//);
  });

  it('pre-registers sam3 tiers as sam-licensed with required acceptance', () => {
    for (const id of ['sam3-tracker', 'sam3-560'] as const) {
      const spec = getModel(id);
      expect(spec, id).toBeDefined();
      expect(spec?.arch).toBe('sam3-tracker');
      expect(spec?.license).toBe('sam-license');
      expect(spec?.requiresLicenseAcceptance).toBe(true);
      expect(spec?.supportsVideo).toBe(true);
    }
    expect(getModel('sam3-560')?.inputSize).toBe(560);
  });

  it('ids are capability tiers: no quant suffixes', () => {
    for (const spec of listModels()) {
      expect(spec.id).not.toMatch(/fp16|int8|q4/i);
    }
  });
});

describe('registerModel / getModel / listModels', () => {
  const custom: ModelSpec = {
    id: 'test-tier',
    displayName: 'Test Tier',
    arch: 'edgetam',
    inputSize: 512,
    supportsVideo: false,
    license: 'apache-2.0',
    manifestUrl: 'https://example.com/manifest.json',
    devices: { webgpu: true, wasm: true },
  };

  it('registers and retrieves a custom tier, visible in listModels', () => {
    registerModel(custom);
    expect(getModel('test-tier')).toEqual(custom);
    expect(listModels().map((s) => s.id)).toContain('test-tier');
  });

  it('rejects duplicate ids with InvalidStateError', () => {
    expect(() => registerModel({ ...custom, id: 'edgetam' })).toThrow(InvalidStateError);
    expect(() => registerModel({ ...custom, id: 'edgetam' })).toThrow(/already registered/);
  });

  it('returns undefined for unknown ids', () => {
    expect(getModel('no-such-model')).toBeUndefined();
  });

  it('returns defensive copies — mutating a result does not corrupt the registry', () => {
    const a = getModel('edgetam');
    if (!a) throw new Error('edgetam must be registered');
    a.inputSize = 1;
    expect(getModel('edgetam')?.inputSize).toBe(1024);
  });
});
