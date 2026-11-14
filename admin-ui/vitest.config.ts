import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path'
import { readFileSync } from 'fs';

const APP_VERSION: string = JSON.parse(readFileSync(path.resolve(import.meta.dirname, 'package.json'), 'utf8')).version;

export default defineConfig({
  plugins: [react()],
  // The sidebar's version comes from package.json (src/config/version.ts); it was a hand-kept
  // literal that nothing tied to the package (ENH-004).
  define: { __APP_VERSION__: JSON.stringify(APP_VERSION) },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: ['**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'tests/**',
        '**/*.d.ts',
        '**/*.config.*',
        '**/types/**',
      ],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
      '@components': path.resolve(import.meta.dirname, './src/components'),
      '@pages': path.resolve(import.meta.dirname, './src/pages'),
      '@stores': path.resolve(import.meta.dirname, './src/stores'),
      '@hooks': path.resolve(import.meta.dirname, './src/hooks'),
      '@services': path.resolve(import.meta.dirname, './src/services'),
      '@types': path.resolve(import.meta.dirname, './src/types'),
      '@utils': path.resolve(import.meta.dirname, './src/utils'),
    },
  },
});
