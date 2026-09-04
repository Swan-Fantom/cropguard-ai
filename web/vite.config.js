import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The frontend talks to the Node API at /api. In dev we proxy /api to the Node
// server (default :4000) so the browser only ever sees one origin (no CORS fuss).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:4000",
        changeOrigin: true,
      },
    },
  },
});
