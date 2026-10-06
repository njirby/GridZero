// gridtrace: opencode plugin that captures authoritative token IDs + logprobs per LLM turn.
//
// Injects a custom fetch for the policy provider that:
//   1. adds `logprobs` + `return_token_ids` to /chat/completions requests (vLLM extensions),
//   2. tees the SSE response into one JSONL line per request under GRID_TRACE_DIR.
//
// Storage: delta-encoded prompts. Consecutive turns' prompts are exact token-prefixes of
// each other (verified), so after the first turn we store only the NEW tokens
// (previous completion + tool-result rendering). On a prefix break (compaction, session
// restart) we fall back to a full prompt and restart the chain.
//
// Record: {ts, session, model, mode: "full"|"delta", prompt_token_ids,
//          completion_token_ids, logprobs, top_logprobs?, finish_reason, usage, n_tools}
//
// Env: GRID_TRACE_DIR (required), GRID_TRACE_PROVIDER (default vllm4b),
//      GRID_TRACE_TOP_LOGPROBS (default 0; 1 = also record top-1 alternatives)
import fs from "node:fs"

const DIR = process.env.GRID_TRACE_DIR
if (!DIR) {
  console.error("[gridtrace] GRID_TRACE_DIR not set; plugin inactive")
} else {
  fs.mkdirSync(DIR, { recursive: true })
}
const PROVIDER = process.env.GRID_TRACE_PROVIDER || "vllm4b"
const TOP = (process.env.GRID_TRACE_TOP_LOGPROBS || "0") === "1"
const mark = (name, val) => { if (DIR) fs.writeFileSync(DIR + "/" + name, String(val)) }

// session -> previous turn's full prompt (chain anchor)
const chains = new Map()

function parseSSE(text, sessionId, requestModel, nTools) {
  let pti = null
  const completionTokenIds = []
  const logprobs = []
  const topLogprobs = []
  let finishReason = null
  let usage = null
  for (const line of text.split("\n")) {
    if (!line.startsWith("data:")) continue
    const payload = line.slice(5).trim()
    if (payload === "[DONE]") continue
    let ev
    try { ev = JSON.parse(payload) } catch { continue }
    if (Array.isArray(ev.prompt_token_ids)) pti = ev.prompt_token_ids
    if (ev.usage) usage = ev.usage
    for (const ch of ev.choices ?? []) {
      if (Array.isArray(ch.token_ids)) completionTokenIds.push(...ch.token_ids)
      if (ch.finish_reason) finishReason = ch.finish_reason
      const lc = ch.logprobs && ch.logprobs.content
      if (Array.isArray(lc)) {
        for (const t of lc) {
          logprobs.push(t.logprob)
          if (TOP && Array.isArray(t.top_logprobs)) topLogprobs.push(t.top_logprobs)
        }
      }
    }
  }
  if (!Array.isArray(pti)) pti = []
  let mode = "full"
  let stored = pti
  const prev = chains.get(sessionId)
  if (prev && pti.length >= prev.length && pti.slice(0, prev.length).every((t, i) => t === prev[i])) {
    stored = pti.slice(prev.length)
    mode = "delta"
  }
  chains.set(sessionId, pti)
  const rec = { ts: Date.now(), session: sessionId, model: requestModel, mode,
                prompt_token_ids: stored, completion_token_ids: completionTokenIds,
                logprobs, finish_reason: finishReason, usage, n_tools: nTools || 0 }
  if (TOP) rec.top_logprobs = topLogprobs
  return rec
}

export default async (input) => {
  return {
    config: (cfg) => {
      const pkeys = Object.keys((cfg && cfg.provider) || {})
      mark("CONFIG_HOOK", pkeys.join(","))
      const p = cfg && cfg.provider && cfg.provider[PROVIDER]
      if (!p) { mark("NO_PROVIDER", pkeys.join(",")); return }
      p.options = p.options || {}
      const prev = p.options.fetch
      p.options.fetch = async (url, init) => {
        const u = typeof url === "string" ? url : (url && url.url) || String(url)
        if (!u.includes("/chat/completions")) return (prev || fetch)(url, init)
        let body = null
        if (init && init.body && typeof init.body === "string") { try { body = JSON.parse(init.body) } catch {} }
        if (body) {
          body.logprobs = true
          body.top_logprobs = TOP ? 1 : 0
          body.return_token_ids = true
          init = Object.assign({}, init, { body: JSON.stringify(body) })
        }
        const res = await (prev || fetch)(url, init)
        const ct = (res && res.headers && res.headers.get) ? (res.headers.get("content-type") || "") : ""
        if (!(res && res.ok && ct.includes("event-stream") && res.body)) return res
        const text = await res.text()
        try {
          let sid = "unknown"
          const h = init && init.headers
          if (h) {
            const v = (typeof h.get === "function") ? h.get("x-session-id") : (h["x-session-id"] || h["X-Session-Id"])
            if (v) sid = String(v)
          }
          const rec = parseSSE(text, sid, body ? body.model : "unknown", body && Array.isArray(body.tools) ? body.tools.length : 0)
          fs.appendFileSync(DIR + "/" + sid + ".jsonl", JSON.stringify(rec) + "\n")
        } catch (e) {
          fs.appendFileSync(DIR + "/TEE_ERR.log", String((e && e.stack) || e) + "\n")
        }
        return new Response(text, { status: res.status, headers: { "content-type": ct } })
      }
      mark("FETCH_INJECTED", String(!!prev))
    },
  }
}
