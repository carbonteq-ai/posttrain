import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { loadEnv } from 'vite';
import { defineConfig } from 'vitest/config';

export default defineConfig(({ mode }) => {
  const apiTarget = loadEnv(mode, '.', 'OBSERVATORY_').OBSERVATORY_API_URL
    ?? 'http://127.0.0.1:7861';
  return {
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
        '/api': apiTarget,
        '/health': apiTarget,
        '/version': apiTarget,
      },
    }
  };
});
