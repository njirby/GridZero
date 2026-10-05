import { useStore } from "./store";
import type { HarnessEvent } from "./types";

// EventSource wrapper for /event with auto-reconnect (native Last-Event-ID)
// + a silence timer that force-reconnects if the stream stalls.
export function startEventStream(url = "/event") {
  const st = useStore.getState();
  st.setConn("connecting");
  let es: EventSource | null = null;
  let silenceTimer: number | undefined;

  const arm = () => {
    if (silenceTimer) clearTimeout(silenceTimer);
    silenceTimer = window.setTimeout(() => {
      useStore.getState().setConn("reconnecting");
      es?.close();
      es = null;
      start();
    }, 10000);
  };

  const start = () => {
    es = new EventSource(url);
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
    const r = await fetch("/api/grid/meta");
    useStore.getState().setMeta(await r.json());
  } catch {
    /* meta is non-fatal */
  }
}
