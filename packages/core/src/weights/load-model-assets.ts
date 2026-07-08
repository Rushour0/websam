/**
 * loadModelAssets: manifest fetch + parse, quant selection, and verified
 * (cached, resumable) download of every graph a model needs.
 *
 * Runs entirely in the worker in M1 (OPFS/Cache/fetch all work there, and
 * bytes never cross a thread). Progress emission follows the LoadPhase table
 * in docs/m1-internal-contracts.md §1.4.1: this module emits `manifest`,
 * `download`, `verify`, and `offline-cache`; `compile`/`ready` belong to the
 * worker engine and sessions layers.
 *
 * Errors: manifest invalid / digest mismatch → {@link WeightVerifyError};
 * requested quant absent from the manifest → {@link InvalidStateError}
 * naming the available quants; network failure → the fetch `TypeError`
 * propagates untouched (callers see the real cause).
 */

import { InvalidStateError, WeightVerifyError } from '../errors.js';
import type { LoadProgressEvent } from '../index.js'; // type-only import: no cycle at runtime
import type { ModelSpec } from '../registry.js';
import {
  parseModelManifest,
  type GraphManifestEntry,
  type GraphRole,
  type ModelManifest,
  type Quant,
  type WeightFileRef,
} from './manifest.js';
import { createWeightStore, MemoryWeightStore, type WeightStore } from './weight-store.js';

/** Configuration for {@link loadModelAssets}. */
export interface LoadAssetsConfig {
  /** Rebase for weight file paths AND the manifest itself when set. */
  modelBaseUrl?: string;
  /** false → MemoryWeightStore (verify but never persist). */
  cache: boolean;
  /** Ordered quant preference from resolveDevice (§2.3). */
  quantPreference: readonly Quant[];
  /** Graph roles to load. M1: `['visionEncoder', 'promptDecoder']`. */
  roles: readonly GraphRole[];
  /** Test seam. */
  fetchImpl?: typeof fetch;
  /** Test seam. */
  store?: WeightStore;
}

/** Everything a backend needs to compile the model's graphs. */
export interface LoadedModelAssets {
  manifest: ModelManifest;
  /** First quantPreference entry available for ALL requested roles. */
  quant: Quant;
  /** ORT-ready bytes per role. Uint8Array (not Blob/ArrayBuffer): the sole consumer is
   *  InferenceSession.create(Uint8Array); Blob would force an extra async hop per graph. */
  graphs: Map<GraphRole, Uint8Array>;
  totalBytes: number;
}

/** Extra resume attempts after a mid-stream network failure or short body. */
const DOWNLOAD_RESUME_ATTEMPTS = 2;

function ensureTrailingSlash(url: string): string {
  return url.endsWith('/') ? url : `${url}/`;
}

/**
 * Resolve `path` against `base`, tolerating relative bases (e.g.
 * `modelBaseUrl: '/models/'`) where `new URL` would throw.
 */
function resolveUrl(path: string, base: string): string {
  try {
    return new URL(path, base).toString();
  } catch {
    return ensureTrailingSlash(base) + path;
  }
}

/** Manifest URL: the spec's, or its filename rebased onto `modelBaseUrl`. */
function resolveManifestUrl(spec: ModelSpec, modelBaseUrl: string | undefined): string {
  if (modelBaseUrl === undefined) return spec.manifestUrl;
  const withoutQuery = spec.manifestUrl.split('?')[0]?.split('#')[0] ?? '';
  const name = withoutQuery.split('/').pop();
  return resolveUrl(name !== undefined && name.length > 0 ? name : 'manifest.json', ensureTrailingSlash(modelBaseUrl));
}

/**
 * A ReadableStream over the file at `url` that transparently resumes with
 * `Range: bytes=<received>-` after mid-stream failures (up to
 * {@link DOWNLOAD_RESUME_ATTEMPTS} times). Servers that ignore Range (a 200
 * on a ranged retry) are handled by discarding the already-received prefix,
 * so downstream hashing always sees one contiguous byte sequence. Emits
 * `download` progress per chunk and one `verify` event at end of stream.
 */
function downloadStream(
  fetchImpl: typeof fetch,
  url: string,
  ref: WeightFileRef,
  onProgress: ((e: LoadProgressEvent) => void) | undefined,
): ReadableStream<Uint8Array> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let received = 0;
  let skip = 0; // bytes to drop when a resume came back as a full 200 body
  let attemptsLeft = DOWNLOAD_RESUME_ATTEMPTS;

  async function openReader(): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    const resuming = received > 0;
    const res = await fetchImpl(
      url,
      resuming ? { headers: { Range: `bytes=${received}-` } } : undefined,
    );
    if (resuming && res.status === 200) {
      skip = received; // server ignored Range: drop the prefix we already have
    } else if (!res.ok) {
      throw new WeightVerifyError(
        `Weight file '${ref.path}' fetch failed with HTTP ${res.status} (${url})`,
      );
    }
    if (!res.body) {
      throw new WeightVerifyError(`Weight file '${ref.path}' response has no body (${url})`);
    }
    reader = res.body.getReader();
    return reader;
  }

  return new ReadableStream<Uint8Array>({
    async pull(controller): Promise<void> {
      for (;;) {
        const activeReader = reader ?? (await openReader());
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          result = await activeReader.read();
        } catch (err) {
          reader = undefined;
          if (attemptsLeft > 0) {
            attemptsLeft--;
            continue; // reopen with a Range request
          }
          throw err;
        }
        if (result.done) {
          if (received < ref.bytes && attemptsLeft > 0) {
            // Short body (connection dropped without an error): resume.
            reader = undefined;
            attemptsLeft--;
            continue;
          }
          onProgress?.({ phase: 'verify', file: ref.path });
          controller.close();
          return;
        }
        let chunk = result.value;
        if (skip > 0) {
          const drop = Math.min(skip, chunk.byteLength);
          skip -= drop;
          chunk = chunk.subarray(drop);
          if (chunk.byteLength === 0) continue;
        }
        received += chunk.byteLength;
        onProgress?.({ phase: 'download', file: ref.path, loaded: received, total: ref.bytes });
        controller.enqueue(chunk);
        return;
      }
    },
    cancel(): void {
      void reader?.cancel().catch(() => undefined);
    },
  });
}

/**
 * Fetch + validate the model manifest, pick the quant, and produce verified
 * ORT-ready bytes for every requested graph role — served from the
 * {@link WeightStore} when previously cached, streamed (with Range resume)
 * from the network otherwise.
 */
export async function loadModelAssets(
  spec: ModelSpec,
  config: LoadAssetsConfig,
  onProgress?: (e: LoadProgressEvent) => void,
): Promise<LoadedModelAssets> {
  const fetchImpl = config.fetchImpl ?? fetch;
  const manifestUrl = resolveManifestUrl(spec, config.modelBaseUrl);

  onProgress?.({ phase: 'manifest' });
  const manifestRes = await fetchImpl(manifestUrl);
  if (!manifestRes.ok) {
    throw new WeightVerifyError(
      `Model manifest fetch failed with HTTP ${manifestRes.status} (${manifestUrl})`,
    );
  }
  const manifest = parseModelManifest(await manifestRes.json(), manifestUrl);
  if (manifest.tier !== spec.id) {
    throw new WeightVerifyError(
      `Model manifest at ${manifestUrl} serves tier '${manifest.tier}', expected '${spec.id}'`,
    );
  }

  const entries: [GraphRole, GraphManifestEntry][] = config.roles.map((role) => {
    const entry = manifest.graphs[role];
    if (!entry) {
      throw new InvalidStateError(
        `Model manifest for '${spec.id}' has no '${role}' graph (has: ${Object.keys(manifest.graphs).join(', ')})`,
      );
    }
    return [role, entry];
  });

  // First preference available for ALL requested roles.
  const quant = config.quantPreference.find((q) =>
    entries.every(([, entry]) => entry.files[q] !== undefined),
  );
  if (quant === undefined) {
    const availableForAll = (['fp32', 'fp16', 'int8', 'q4f16'] as const).filter((q) =>
      entries.every(([, entry]) => entry.files[q] !== undefined),
    );
    throw new InvalidStateError(
      `No quant in preference [${config.quantPreference.join(', ')}] is available for all of ` +
        `[${config.roles.join(', ')}] in the '${spec.id}' manifest; available: ` +
        `[${availableForAll.join(', ') || 'none'}]`,
    );
  }

  const store = config.store ?? (config.cache ? createWeightStore() : new MemoryWeightStore());
  const fileBase = config.modelBaseUrl !== undefined ? ensureTrailingSlash(config.modelBaseUrl) : manifestUrl;

  const graphs = new Map<GraphRole, Uint8Array>();
  let totalBytes = 0;
  for (const [role, entry] of entries) {
    const ref = entry.files[quant];
    if (!ref) continue; // unreachable: quant selection guarantees presence
    totalBytes += ref.bytes;

    const cached = await store.get(ref);
    let blob: Blob;
    if (cached) {
      onProgress?.({ phase: 'offline-cache', file: ref.path, loaded: ref.bytes, total: ref.bytes });
      blob = cached;
    } else {
      const fileUrl = resolveUrl(ref.path, fileBase);
      blob = await store.put(ref, downloadStream(fetchImpl, fileUrl, ref, onProgress));
    }
    graphs.set(role, new Uint8Array(await blob.arrayBuffer()));
  }

  return { manifest, quant, graphs, totalBytes };
}
