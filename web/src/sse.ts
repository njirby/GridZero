import { useStore } from "./store";
import { withToken, authHeaders } from "./api";
import type { HarnessEvent } from "./types";

// EventSource wrapper for /event with auto-reconnect (native Last-Event-ID)
// + a silence timer that force-reconnects if the stream stalls.
const SILENCE_MS = 45000;

export function startEventStream(url = "/event") {
  const st = useStore.getState();
  st.setConn("connecting");
  let es: EventSource | null = null;
  let silenceTimer: number | undefined;

  const arm = () => {
    if (silenceTimer) clearTimeout(silenceTimer);
    // Well above the backend's ~15 s ping so quiet periods don't force a reconnect.
    silenceTimer = window.setTimeout(() => {
      useStore.getState().setConn("reconnecting");
      es?.close();
      es = null;
      start();
    }, SILENCE_MS);
  };

  const start = () => {
    // Manual reconnects send no Last-Event-ID, so resume explicitly from the last applied seq.
    const last = useStore.getState().lastSeq;
    const full = last >= 0 ? `${url}${url.includes("?") ? "&" : "?"}after_seq=${last}` : url;
    es = new EventSource(withToken(full));
    es.onopen = () => { useStore.getState().setConn("connected"); arm(); };
    es.onerror = () => {
      // EventSource auto-reconnects; mark reconnecting and reset silence timer.
      useStore.getState().setConn("reconnecting");
      arm();
    };
    es.onmessage = (m) => {
      arm();
      try {
        const ev = JSON.parse(m.data) as HarnessEvent;
        useStore.getState().ingest(ev);
      } catch {
        /* ignore malformed frame */
      }
    };
  };

  start();
  return () => {
    if (silenceTimer) clearTimeout(silenceTimer);
    es?.close();
  };
}

export async function loadMeta(): Promise<void> {
  try {
    const r = await fetch("/api/grid/meta", { headers: authHeaders() });
    useStore.getState().setMeta(await r.json());
  } catch {
    /* meta is non-fatal */
  }
}
