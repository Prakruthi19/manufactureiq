import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api → FastAPI on :8000, so the browser never
// makes a cross-origin request (no CORS headaches during development).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
