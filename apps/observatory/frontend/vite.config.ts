import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.tsx'],
    setupFiles: ['./src/test/setup.ts'],
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: ['terminal.local'],
    proxy: {
      '/api': 'http://127.0.0.1:7861',
      '/health': 'http://127.0.0.1:7861',
      '/version': 'http://127.0.0.1:7861'
    }
  }
});
