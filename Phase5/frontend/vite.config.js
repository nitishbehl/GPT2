import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    sourcemap: false,    // disable source maps in production build
  },
  server: {
    sourcemap: false,    // disable eval-based source maps during dev server
  },
})

