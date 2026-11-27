import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Backend for the dev proxy. Override to point a second dev server at another backend copy.
const BACKEND = process.env.MEDSIM_BACKEND_URL || 'http://127.0.0.1:8787';

const PROXY = {
  '/ollama': {
    target: 'http://localhost:11434',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/ollama/, ''),
  },
  // Claude Managed Agents proxy, keeps the API key server-side.
  // The `/agent/sessions/:id/stream` endpoint is SSE; Vite's proxy
  // handles that correctly as long as we disable buffering on the
  // server side (see backend/server.py). Trailing slash is required
  // so the prefix doesn't catch frontend routes like /agentic-rounds.
  '^/agent/': {
    target: BACKEND,
    changeOrigin: true,
  },
  // Real-time voice: Gemini Live & Observer WebSockets + REST control endpoints.
  '^/voice/': {
    target: BACKEND,
    changeOrigin: true,
    ws: true,
  },
  // Teacher Portal lessons + typed Q&A (backend/teacher.py).
  '^/teach/': {
    target: BACKEND,
    changeOrigin: true,
  },
  // Patient-facing links: Murf voice prescription MP3 and follow-up .ics invite.
  '^/rx/': {
    target: BACKEND,
    changeOrigin: true,
  },
  // Asha follow-up call API (the SPA route /call/<token> stays on the frontend).
  '^/call-api/': {
    target: BACKEND,
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  resolve: {
    preserveSymlinks: true,
  },
  server: {
    port: 5173,
    open: false, // never auto-open a browser tab; reuse the one already open
    proxy: PROXY,
  },
  preview: {
    port: 5173,
    proxy: PROXY,
  },
});
