import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Proxy target for `npm run dev`: whichever backend (real or mock) is on :8731.
// The mock (contracts/mock/mock_sim_server.py) defaults to :8731, and the real
// backend (backend/app/main.py) also binds :8731 — so one target covers both.
const TARGET = (typeof process !== "undefined" && process.env.PROXY_TARGET) || "http://127.0.0.1:8731";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Loopback by default: the dev proxy forwards reset/attack/control. Set DEV_HOST=0.0.0.0 to expose it.
    host: (typeof process !== "undefined" && process.env.DEV_HOST) || "127.0.0.1",
    allowedHosts: typeof process !== "undefined" && !!process.env.DEV_HOST ? true : undefined,
    proxy: {
      "/event": { target: TARGET, changeOrigin: true },
      "/state": { target: TARGET, changeOrigin: true },
      "/control": { target: TARGET, changeOrigin: true },
      "/sim": { target: TARGET, changeOrigin: true },
      "/api": { target: TARGET, changeOrigin: true },
      "/render": { target: TARGET, changeOrigin: true },
      "/model": { target: TARGET, changeOrigin: true },
      "/models": { target: TARGET, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    css: false,
  },
});
