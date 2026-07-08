import { useEffect, useState } from 'react';
import { WasmBackend, WebGpuBackend, listModels } from '@websam/core';

/**
 * A backend probe or model-registry entry rendered generically. The demo
 * deliberately couples only to the *names* `WebGpuBackend.probe`,
 * `WasmBackend.probe` and `listModels`, not to their exact result shapes,
 * so it keeps working as the core contracts evolve through milestones.
 */
type Report = Record<string, unknown>;

/** Lifecycle of one backend probe run on mount. */
type ProbeState =
  | { status: 'probing' }
  | { status: 'done'; report: Report }
  | { status: 'failed'; message: string };

function toReport(value: unknown): Report {
  if (typeof value === 'object' && value !== null) {
    return { ...(value as Report) };
  }
  return { result: value };
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'string') return value === '' ? '—' : value;
  if (typeof value === 'number' || typeof value === 'bigint') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function findString(report: Report, keyPattern: RegExp): string | undefined {
  for (const [key, value] of Object.entries(report)) {
    if (keyPattern.test(key) && typeof value === 'string') return value;
  }
  return undefined;
}

function findBoolean(report: Report, keyPattern: RegExp): boolean | undefined {
  for (const [key, value] of Object.entries(report)) {
    if (keyPattern.test(key) && typeof value === 'boolean') return value;
  }
  return undefined;
}

/**
 * Pick the device the page should recommend: an explicit `recommended*`
 * field from either probe wins, then WebGPU availability, then a plain
 * `navigator.gpu` presence check as the last resort.
 */
function recommendDevice(webgpu: ProbeState, wasm: ProbeState): string {
  const reports: Report[] = [];
  if (webgpu.status === 'done') reports.push(webgpu.report);
  if (wasm.status === 'done') reports.push(wasm.report);
  for (const report of reports) {
    const explicit = findString(report, /recommend/i);
    if (explicit !== undefined) return explicit;
  }
  if (webgpu.status === 'done') {
    const available = findBoolean(webgpu.report, /available|supported/i);
    if (available !== undefined) return available ? 'webgpu' : 'wasm';
  }
  if (webgpu.status === 'failed') return 'wasm';
  return typeof navigator !== 'undefined' && 'gpu' in navigator ? 'webgpu' : 'wasm';
}

function licenseBadgeClass(license: string): string {
  if (/apache|mit|bsd/i.test(license)) return 'badge badge-permissive';
  if (/sam/i.test(license)) return 'badge badge-restricted';
  return 'badge';
}

function ReportRows({ report }: { report: Report }) {
  const entries = Object.entries(report);
  if (entries.length === 0) {
    return <p className="muted">Probe returned an empty report.</p>;
  }
  return (
    <dl className="kv">
      {entries.map(([key, value]) => (
        <div className="kv-row" key={key}>
          <dt>{key}</dt>
          <dd>{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function ProbeCard({ title, state }: { title: string; state: ProbeState }) {
  return (
    <section className="card">
      <header className="card-header">
        <h2>{title}</h2>
        <span className={`dot dot-${state.status}`} aria-hidden="true" />
        <span className="muted">
          {state.status === 'probing' && 'probing…'}
          {state.status === 'done' && 'ok'}
          {state.status === 'failed' && 'failed'}
        </span>
      </header>
      {state.status === 'done' && <ReportRows report={state.report} />}
      {state.status === 'failed' && <p className="error">{state.message}</p>}
    </section>
  );
}

function ModelCard({ model }: { model: Report }) {
  const id = findString(model, /^(id|modelId|name)$/) ?? 'unknown model';
  const license = findString(model, /license/i) ?? 'unknown license';
  const rest = Object.entries(model).filter(
    ([key, value]) => !/^(id|modelId|name)$/.test(key) && !/license/i.test(key) && value !== undefined,
  );
  return (
    <section className="card">
      <header className="card-header">
        <h2>{id}</h2>
        <span className={licenseBadgeClass(license)}>{license}</span>
      </header>
      {rest.length > 0 && (
        <dl className="kv">
          {rest.map(([key, value]) => (
            <div className="kv-row" key={key}>
              <dt>{key}</dt>
              <dd>{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

/**
 * Capability-probe status page for the future rotobrush demo. On mount it
 * runs `WebGpuBackend.probe()` and `WasmBackend.probe()` from `@websam/core`
 * and renders both capability reports, the browser environment facts that
 * gate multithreaded WASM (crossOriginIsolated), the recommended device, and
 * the registered model tiers from `listModels()` with license badges.
 */
export function App() {
  const [webgpu, setWebgpu] = useState<ProbeState>({ status: 'probing' });
  const [wasm, setWasm] = useState<ProbeState>({ status: 'probing' });
  const [models, setModels] = useState<Report[] | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const run = async (probe: () => unknown, set: (state: ProbeState) => void) => {
      try {
        const result: unknown = await Promise.resolve(probe());
        if (!cancelled) set({ status: 'done', report: toReport(result) });
      } catch (error) {
        if (!cancelled) {
          set({ status: 'failed', message: error instanceof Error ? error.message : String(error) });
        }
      }
    };

    void run(() => WebGpuBackend.probe(), setWebgpu);
    void run(() => WasmBackend.probe(), setWasm);

    void (async () => {
      try {
        const listed: unknown = await Promise.resolve(listModels());
        if (!cancelled) setModels(Array.isArray(listed) ? listed.map(toReport) : [toReport(listed)]);
      } catch (error) {
        if (!cancelled) setModelsError(error instanceof Error ? error.message : String(error));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;

  return (
    <main className="page">
      <header className="page-header">
        <h1>
          websam rotobrush <span className="tag">pre-alpha</span>
        </h1>
        <p className="muted">
          SAM-family interactive video segmentation in the browser. Nothing to brush yet — this page
          reports what your browser can run.
        </p>
      </header>

      <section className="card">
        <header className="card-header">
          <h2>environment</h2>
        </header>
        <dl className="kv">
          <div className="kv-row">
            <dt>crossOriginIsolated</dt>
            <dd>{formatValue(isolated)}</dd>
          </div>
          <div className="kv-row">
            <dt>navigator.gpu</dt>
            <dd>{formatValue(typeof navigator !== 'undefined' && 'gpu' in navigator)}</dd>
          </div>
          <div className="kv-row">
            <dt>hardwareConcurrency</dt>
            <dd>{formatValue(typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : undefined)}</dd>
          </div>
          <div className="kv-row">
            <dt>recommended device</dt>
            <dd>{recommendDevice(webgpu, wasm)}</dd>
          </div>
        </dl>
      </section>

      <ProbeCard title="webgpu backend" state={webgpu} />
      <ProbeCard title="wasm backend" state={wasm} />

      <section className="models">
        <h2 className="section-title">registered model tiers</h2>
        {models === null && modelsError === null && <p className="muted">loading model registry…</p>}
        {modelsError !== null && <p className="error">{modelsError}</p>}
        {models !== null && models.length === 0 && <p className="muted">No models registered.</p>}
        {models !== null &&
          models.map((model, index) => (
            <ModelCard model={model} key={findString(model, /^(id|modelId|name)$/) ?? index} />
          ))}
      </section>

      <footer className="page-footer muted">
        <p>
          Model weights are downloaded at runtime and carry their own licenses; all code is MIT.{' '}
          <a href="https://github.com/Rushour0/websam">source</a>
        </p>
      </footer>
    </main>
  );
}
