/**
 * `ChatPanel.tsx` — conversational video-editing panel backed by the fabri
 * gateway (studio-contracts.md §3 style: no props; reads/writes exclusively
 * through `useStudioStore`).
 *
 * The user types an edit instruction; on send we upload the currently ACTIVE
 * clip's original `Blob` plus the task text to the fabri gateway's `POST /runs`
 * (multipart), then tail `GET /runs/{sessionId}/events` (SSE) as a live chat
 * transcript. On the run's terminal event we fetch the run result, and for its
 * `outputPath` we fetch the artifact back and either `importClip` it (video) or
 * offer a download, so a produced clip shows up in `MediaLibrary` like any
 * other import.
 *
 * GATEWAY CONTRACT: this panel talks to a small self-hosted FastAPI app
 * (`integrations/fabri/service/`), NOT to fabri's own `fabri serve` HTTP API.
 * Its base URL comes from `VITE_FABRI_GATEWAY_URL` (default
 * `http://localhost:8787`). The gateway's CORS/Origin allowlist
 * (`FABRI_GATEWAY_CORS_ORIGIN`, default allows `http://localhost:5173` and
 * `http://127.0.0.1:5173`) MUST include the Studio origin — both `fetch` and
 * `EventSource` here are CORS-mode cross-origin requests, so Studio's own
 * COOP/COEP response headers do NOT govern them; the gateway's CORS response
 * headers do. If the gateway origin is missing from the allowlist the browser
 * blocks the request before any of this code sees a response.
 */
import { useEffect, useRef, useState } from 'react';
import { Bot, Loader2, Send } from 'lucide-react';

import { useStudioStore } from '../store/studio-store.js';
import { cn } from '../lib/utils.js';
import { Button } from './ui/button.js';

/** Gateway base URL (see file header). Trailing slashes stripped so
 * `${GATEWAY_URL}/runs` never double-slashes. */
const GATEWAY_URL = ((import.meta.env?.VITE_FABRI_GATEWAY_URL as string | undefined) ?? 'http://localhost:8787').replace(/\/+$/, '');
/** Artifact filenames we treat as re-importable video (vs. a plain download). */
const VIDEO_EXT_RE = /\.(mp4|mov|webm|m4v)$/i;
/** Stable message id. `crypto.randomUUID` is undefined in non-secure contexts
 * (e.g. `http://<lan-ip>:5173`), so the fallback is MANDATORY — same pattern as
 * `studio-store.ts`'s `genId`. */
const genId = (): string =>
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
/** Trace event types that end a run. `'error'` is deliberately EXCLUDED: fabri's
 * mid-run `'error'` events are always followed by a real terminal, and a planner
 * `'error'` can even precede a successful `'final'` — closing on `'error'` would
 * drop successful runs' artifacts. */
const TERMINAL_TYPES = new Set(['final', 'failed', 'incomplete']);

type ChatRole = 'user' | 'agent' | 'status' | 'error';
interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
}
type RunPhase = 'idle' | 'running';

/**
 * Conversational video-editing panel. See file header for the full gateway
 * contract.
 */
export function ChatPanel(): React.JSX.Element {
  const activeClipId = useStudioStore((s) => s.activeClipId);
  const clips = useStudioStore((s) => s.clips);
  const importClip = useStudioStore((s) => s.importClip);
  const setNotice = useStudioStore((s) => s.setNotice);

  const activeClip = activeClipId ? clips[activeClipId] : undefined;

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [phase, setPhase] = useState<RunPhase>('idle');

  const eventSourceRef = useRef<EventSource | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const append = (role: ChatRole, text: string) =>
    setMessages((m) => [...m, { id: genId(), role, text }]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages]);

  useEffect(
    () => () => {
      eventSourceRef.current?.close();
    },
    [],
  );

  /** Fetch the run result + artifact and land it in the Media Library (or offer
   * a download). Always drops back to `idle`. */
  const finishRun = async (
    sessionId: string,
    terminal: { type: string; text?: string; reason?: string },
  ): Promise<void> => {
    try {
      if (terminal.type !== 'final') {
        append('error', `Run ended: ${terminal.type}${terminal.reason ? ` — ${terminal.reason}` : ''}`);
        return;
      }
      // Dedupe the final narration text INSIDE the functional update — reading
      // `messages` in this async closure would be stale.
      if (terminal.text) {
        const finalText = terminal.text;
        setMessages((m) =>
          m[m.length - 1]?.text === finalText ? m : [...m, { id: genId(), role: 'agent', text: finalText }],
        );
      }

      const resultRes = await fetch(`${GATEWAY_URL}/runs/${sessionId}/result`);
      const result = (await resultRes.json()) as {
        structured_output?: { outputPath?: string } | null;
      };
      const outputPath = result.structured_output?.outputPath;
      if (!outputPath) {
        append('status', 'Run finished with no output artifact.');
        return;
      }

      const artifactRes = await fetch(
        `${GATEWAY_URL}/runs/${sessionId}/artifact?path=${encodeURIComponent(outputPath)}`,
      );
      if (!artifactRes.ok) throw new Error(`gateway responded ${artifactRes.status}`);
      const blob = await artifactRes.blob();
      const name = outputPath.split('/').pop() ?? 'fabri-output';

      if (VIDEO_EXT_RE.test(name)) {
        await importClip(new File([blob], name, { type: blob.type || 'video/mp4' }));
        append('status', `Added ${name} to Media Library.`);
      } else {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        a.click();
        URL.revokeObjectURL(url);
        append('status', `Downloaded ${name}.`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      append('error', msg);
      setNotice({ title: 'Fabri run failed', detail: msg, kind: 'error' });
    } finally {
      setPhase('idle');
    }
  };

  const handleSend = async (): Promise<void> => {
    const task = input.trim();
    if (!task || !activeClip || phase === 'running') return;

    append('user', task);
    setInput('');
    setPhase('running');

    try {
      const form = new FormData();
      form.append('video', new File([activeClip.blob], activeClip.fileName, { type: activeClip.blob.type || 'video/mp4' }));
      form.append('task', task);

      const res = await fetch(`${GATEWAY_URL}/runs`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`gateway responded ${res.status}`);
      const { sessionId } = (await res.json()) as { sessionId: string };

      const es = new EventSource(`${GATEWAY_URL}/runs/${sessionId}/events`);
      eventSourceRef.current = es;

      // Trace events arrive UNNAMED → handled by `onmessage`.
      es.onmessage = (e) => {
        let evt: { type: string; text?: string; name?: string; reason?: string };
        try {
          evt = JSON.parse(e.data);
        } catch {
          return; // skip unparseable lines
        }
        switch (evt.type) {
          case 'narration':
            if (evt.text) append('agent', evt.text);
            break;
          case 'tool_call':
            append('status', `ran ${evt.name}`);
            break;
          case 'error':
            // Do NOT close the stream — a real terminal always follows.
            append('error', evt.text ?? 'agent error');
            break;
          default:
            // thought / step_started / step_finished / usage / others → ignore
            break;
        }
        if (TERMINAL_TYPES.has(evt.type)) {
          es.close();
          eventSourceRef.current = null;
          void finishRun(sessionId, evt);
        }
      };

      // Gateway's named `end` sentinel. If the ref still points at this source,
      // no terminal type was seen (covers the AgentProtocolError path where
      // fabri emits no terminal event) → treat as an unexpected end.
      es.addEventListener('end', () => {
        if (eventSourceRef.current === es) {
          es.close();
          eventSourceRef.current = null;
          setPhase('idle');
          append('error', 'Run ended unexpectedly.');
        }
      });

      // Network drop. The ref-guard means this won't fire after our own clean
      // close (we null the ref first).
      es.onerror = () => {
        if (eventSourceRef.current === es) {
          es.close();
          eventSourceRef.current = null;
          setPhase('idle');
          append('error', 'Lost connection to the fabri gateway.');
        }
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setPhase('idle');
      append('error', msg);
      setNotice({ title: 'Fabri run failed', detail: msg, kind: 'error' });
    }
  };

  return (
    <div className="flex h-full flex-col gap-2 overflow-hidden p-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Assistant</h2>
        <span className="truncate text-[10px] text-muted-foreground">{activeClip?.fileName}</span>
      </div>

      <div ref={logRef} role="log" aria-live="polite" className="flex flex-1 flex-col gap-1.5 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="p-2 text-center text-xs text-muted-foreground">
            Ask the assistant to edit the active clip — e.g. “cut out the car”.
          </p>
        ) : (
          messages.map((m) => {
            if (m.role === 'status') {
              return (
                <div key={m.id} className="self-start px-1 text-[10px] italic text-muted-foreground">
                  {m.text}
                </div>
              );
            }
            if (m.role === 'error') {
              return (
                <div
                  key={m.id}
                  className="self-start max-w-[85%] rounded-md border border-destructive/50 bg-destructive/10 px-2 py-1.5 text-xs text-destructive"
                >
                  {m.text}
                </div>
              );
            }
            if (m.role === 'user') {
              return (
                <div
                  key={m.id}
                  className="self-end max-w-[85%] rounded-md bg-primary px-2 py-1.5 text-xs text-primary-foreground"
                >
                  {m.text}
                </div>
              );
            }
            return (
              <div
                key={m.id}
                className={cn('self-start max-w-[85%] rounded-md border border-input bg-accent/50 px-2 py-1.5 text-xs')}
              >
                <Bot className="mr-1 inline h-3 w-3" />
                {m.text}
              </div>
            );
          })
        )}
        {phase === 'running' ? (
          <div className="flex items-center gap-1.5 self-start px-1 text-[10px] italic text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" /> working…
          </div>
        ) : null}
      </div>

      {!activeClip ? (
        <p className="rounded-md border border-dashed border-input p-2 text-center text-[10px] text-muted-foreground">
          Import a clip and click it in Media to enable the assistant.
        </p>
      ) : null}

      <form
        className="flex items-end gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          void handleSend();
        }}
      >
        <textarea
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void handleSend();
            }
          }}
          placeholder="Describe an edit…"
          disabled={!activeClip || phase === 'running'}
          className="min-h-0 flex-1 resize-none rounded-md border border-input bg-background px-2 py-1.5 text-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Message the video assistant"
        />
        <Button
          type="submit"
          size="icon"
          aria-label="Send"
          disabled={!activeClip || phase === 'running' || input.trim().length === 0}
        >
          {phase === 'running' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </form>
    </div>
  );
}

export default ChatPanel;
