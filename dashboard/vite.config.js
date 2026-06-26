import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API + metrics to the FastAPI backend (default :8000), so the
// dashboard can call /v1/* and /metrics without CORS config.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
