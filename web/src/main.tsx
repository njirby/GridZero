import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { startEventStream, loadMeta } from "./sse";
import { useStore } from "./store";
import { resync } from "./api";
import "./styles.css";

const root = createRoot(document.getElementById("root")!);
root.render(<React.StrictMode><ErrorBoundary><App /></ErrorBoundary></React.StrictMode>);

loadMeta();
startEventStream();
resync(); // adopt mode/running/sim from /state on startup

// Resync from /state whenever a seq gap is detected (the flag is cleared by resync()).
useStore.subscribe((s, prev) => {
  if (s.needResync && !prev.needResync) void resync();
});
