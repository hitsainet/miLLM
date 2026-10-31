import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { readFileSync } from 'fs'

// https://vite.dev/config/
const APP_VERSION: string = JSON.parse(readFileSync(path.resolve(import.meta.dirname, 'package.json'), 'utf8')).version

export default defineConfig({
  plugins: [react()],
  // The sidebar's version comes from package.json (src/config/version.ts); it was a hand-kept
  // literal that nothing tied to the package (ENH-004).
  define: { __APP_VERSION__: JSON.stringify(APP_VERSION) },
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
      '@components': path.resolve(import.meta.dirname, './src/components'),
      '@pages': path.resolve(import.meta.dirname, './src/pages'),
      '@hooks': path.resolve(import.meta.dirname, './src/hooks'),
      '@stores': path.resolve(import.meta.dirname, './src/stores'),
      '@services': path.resolve(import.meta.dirname, './src/services'),
      '@types': path.resolve(import.meta.dirname, './src/types'),
      '@utils': path.resolve(import.meta.dirname, './src/utils'),
    },
  },
  server: {
    port: 3000,
    allowedHosts: ['millm.hitsai.local', 'localhost'],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
