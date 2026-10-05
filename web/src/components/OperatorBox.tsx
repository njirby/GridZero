import { useState } from "react";
import { controls } from "../api";

export function OperatorBox() {
  const [text, setText] = useState("");
  const send = () => {
    if (!text.trim()) return;
    controls.instruction(text.trim());
    setText("");
  };
  return (
    <div className="operatorbox chat-composer" data-testid="operatorbox">
      <label htmlFor="agent-instruction">MESSAGE THE AGENT</label>
      <textarea
        id="agent-instruction"
        value={text}
        placeholder="Give the agent a direction…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send(); }}
        rows={2}
      />
      <div className="operatorbox-actions">
        <button onClick={send}>Send instruction</button>
        <button onClick={() => controls.singleStep()}>Step without instruction</button>
        <span className="hint">Delivered on the agent’s next step · Ctrl/⌘ + Enter to send</span>
      </div>
    </div>
  );
}
