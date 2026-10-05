import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { startEventStream, loadMeta } from "./sse";
import { useStore } from "./store";
import { resync } from "./api";
import "./styles.css";

const root = createRoot(document.getElementById("root")!);
root.render(<React.StrictMode><App /></React.StrictMode>);

loadMeta();
startEventStream();

// react to detected gaps -> resync from /state
const unsub = useStore.subscribe((s, prev) => {
  if (s.needResync && !prev.needResync) resync();
});
unsub();
