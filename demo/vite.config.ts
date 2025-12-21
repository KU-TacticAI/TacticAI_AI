import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  server: {
    // Allow all hosts (development only)
    allowedHosts: true,
    host: true,      // = 0.0.0.0 바인딩
    port: 5173,
    proxy: {
      '/ai': {
        target: '<your gateway url>',
        changeOrigin: true,
        secure: false,
      }
    }
  },
});
